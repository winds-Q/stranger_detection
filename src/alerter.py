import glob
import logging
import os
import queue
import re
import smtplib
import threading
import time
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Alerter:
    def __init__(self, config, snapshot_callback=None):
        alert_cfg = config.alert
        self._enabled = alert_cfg.get("enabled", True)
        self._cooldown_seconds = alert_cfg.get("cooldown_seconds", 180)
        self._smtp_server = alert_cfg.get("smtp_server", "smtp.gmail.com")
        self._smtp_port = alert_cfg.get("smtp_port", 587)
        self._sender_email = alert_cfg.get("sender_email", "")
        # Secrets must never be loaded from YAML or returned through the Web API.
        self._sender_password = os.environ.get("STRANGER_DETECTION_SMTP_PASSWORD", "")
        self._receiver_emails = self._normalize_receivers(
            alert_cfg.get("receiver_emails", [])
        )
        if not self._receiver_emails:
            self._receiver_emails = self._normalize_receivers(
                alert_cfg.get("receiver_email", "")
            )

        snapshot_cfg = config.snapshot
        snapshot_path = snapshot_cfg.get("save_to", "./snapshots/")
        self._snapshot_dir = config.resolve_path(snapshot_path)
        self._max_snapshots = snapshot_cfg.get("max_snapshots", 100)

        self._last_alert_times = {}
        self._snapshot_callback = snapshot_callback

        if self._enabled:
            logger.info("报警模块已初始化 (cooldown=%ds)", self._cooldown_seconds)
        else:
            logger.info("报警模块已禁用")

    def send_alert(
        self,
        frame: np.ndarray,
        stranger_id: str = "unknown",
        bypass_cooldown: bool = False,
        save_snapshot: bool = True,
    ) -> bool:
        if not self._enabled:
            logger.debug("报警已禁用，跳过发送")
            return False

        now = time.time()
        last_alert_time = self._last_alert_times.get(stranger_id, 0.0)
        if not bypass_cooldown and now - last_alert_time < self._cooldown_seconds:
            remaining = self._cooldown_seconds - (now - last_alert_time)
            logger.debug(
                "%s 冷却中，%.0f 秒后可再次报警", stranger_id, remaining
            )
            return False

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            logger.error("截图编码失败")
            return False
        image_data = buffer.tobytes()

        # 本地留证不应依赖邮件配置或网络状态。
        if save_snapshot:
            try:
                snapshot_path = self._save_snapshot(image_data)
                if self._snapshot_callback:
                    self._snapshot_callback(stranger_id, snapshot_path)
            except OSError as exc:
                logger.error("保存陌生人截图失败: %s", exc)
        if not bypass_cooldown:
            self._last_alert_times[stranger_id] = now

        if not self._sender_email or not self._receiver_emails:
            logger.warning("邮件发送方或接收方未配置，截图已保存但跳过邮件")
            return False
        if not self._sender_password:
            logger.warning("SMTP 授权码未配置，截图已保存但跳过邮件")
            return False

        msg = MIMEMultipart()
        msg["Subject"] = f"[陌生人警报] 检测到陌生人 ({stranger_id})"
        msg["From"] = self._sender_email
        msg["To"] = ", ".join(self._receiver_emails)

        html = self._build_html(stranger_id)
        msg.attach(MIMEText(html, "html", "utf-8"))

        img_part = MIMEImage(image_data, name="stranger.jpg")
        msg.attach(img_part)

        server = None
        try:
            if self._smtp_port == 465:
                server = smtplib.SMTP_SSL(
                    self._smtp_server, self._smtp_port, timeout=15
                )
            else:
                server = smtplib.SMTP(
                    self._smtp_server, self._smtp_port, timeout=15
                )
                server.ehlo()
                server.starttls()
                server.ehlo()
            server.login(self._sender_email, self._sender_password)
            server.sendmail(
                self._sender_email,
                self._receiver_emails,
                msg.as_string(),
            )
        except (smtplib.SMTPException, OSError) as e:
            logger.error("邮件发送失败: %s", e)
            return False
        finally:
            if server:
                try:
                    server.quit()
                except (smtplib.SMTPException, OSError):
                    logger.debug("关闭 SMTP 连接失败", exc_info=True)

        logger.info("报警邮件已发送至 %d 个收件地址", len(self._receiver_emails))
        return True

    @property
    def is_configured(self) -> bool:
        return bool(
            self._enabled
            and self._sender_email
            and self._receiver_emails
            and self._sender_password
        )

    def send_test_email(self) -> bool:
        """Verify SMTP connectivity without saving a snapshot or using cooldown."""
        return self.send_alert(
            np.full((32, 32, 3), 245, dtype=np.uint8),
            stranger_id="smtp-test",
            bypass_cooldown=True,
            save_snapshot=False,
        )

    @staticmethod
    def _normalize_receivers(value) -> list[str]:
        if isinstance(value, str):
            candidates = re.split(r"[,;\n]", value)
        elif isinstance(value, (list, tuple, set)):
            candidates = value
        else:
            candidates = []

        receivers = []
        seen = set()
        for candidate in candidates:
            address = str(candidate).strip()
            key = address.lower()
            if (
                address
                and re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", address)
                and key not in seen
            ):
                receivers.append(address)
                seen.add(key)
        return receivers

    def _save_snapshot(self, image_data: bytes) -> str:
        """保存陌生人截图到本地，超出 max_snapshots 时自动清理旧截图。"""
        os.makedirs(self._snapshot_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"stranger_{timestamp}.jpg"
        filepath = os.path.join(self._snapshot_dir, filename)

        with open(filepath, "wb") as f:
            f.write(image_data)
        logger.info("陌生人截图已保存: %s", filepath)

        # 清理超出数量的旧截图
        snapshots = sorted(
            glob.glob(os.path.join(self._snapshot_dir, "stranger_*.jpg")),
            key=os.path.getmtime,
        )
        if len(snapshots) > self._max_snapshots:
            for old_path in snapshots[: len(snapshots) - self._max_snapshots]:
                os.remove(old_path)
                logger.debug("已清理旧截图: %s", old_path)
        return filepath

    @staticmethod
    def _build_html(stranger_id: str = "unknown") -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            "<html><body>"
            f"<h2>陌生人警报</h2>"
            f"<p>临时身份: {stranger_id}</p>"
            f"<p>检测时间: {now}</p>"
            f"<p>系统检测到陌生人在笔记本摄像头范围内，请及时查看。</p>"
            "</body></html>"
        )


class AsyncAlertDispatcher:
    """在后台队列中发送告警，避免 SMTP 阻塞检测线程。"""

    def __init__(
        self,
        alerter: Alerter,
        cooldown_seconds: float = 180,
        queue_size: int = 20,
        retry_count: int = 2,
        retry_backoff_seconds: float = 2,
        clock=time.monotonic,
        result_callback=None,
    ):
        self._alerter = alerter
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._retry_count = max(0, int(retry_count))
        self._retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._clock = clock
        self._result_callback = result_callback
        self._last_submitted = {}
        self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker,
            name="alert-dispatcher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame: np.ndarray, stranger_id: str) -> bool:
        now = self._clock()
        last_time = self._last_submitted.get(stranger_id)
        if (
            last_time is not None
            and now - last_time < self._cooldown_seconds
        ):
            return False
        try:
            self._queue.put_nowait((frame.copy(), stranger_id))
        except queue.Full:
            logger.error("报警队列已满，丢弃告警: %s", stranger_id)
            return False
        self._last_submitted[stranger_id] = now
        return True

    def _worker(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                frame, stranger_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._deliver_with_retry(frame, stranger_id)
            finally:
                self._queue.task_done()

    def _deliver_with_retry(self, frame: np.ndarray, stranger_id: str) -> None:
        attempts = 1 if not self._alerter.is_configured else self._retry_count + 1
        for attempt in range(attempts):
            success = self._alerter.send_alert(
                frame,
                stranger_id,
                bypass_cooldown=attempt > 0,
                save_snapshot=attempt == 0,
            )
            if success:
                self._notify_result(stranger_id, True)
                return
            if attempt + 1 < attempts:
                delay = self._retry_backoff_seconds * (2 ** attempt)
                logger.warning(
                    "告警发送失败，%.1f 秒后重试 (%d/%d): %s",
                    delay, attempt + 2, attempts, stranger_id,
                )
                if self._stop_event.wait(delay):
                    self._notify_result(stranger_id, False)
                    return
        self._notify_result(stranger_id, False)

    def _notify_result(self, stranger_id: str, success: bool) -> None:
        if not self._result_callback:
            return
        try:
            self._result_callback(stranger_id, success)
        except Exception:
            logger.exception("处理报警发送结果失败")

    def close(self, timeout: float = 10) -> None:
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("报警发送线程未在超时时间内退出")

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()
