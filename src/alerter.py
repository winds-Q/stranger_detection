import glob
import logging
import os
import smtplib
import time
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Alerter:
    def __init__(self, config):
        alert_cfg = config.alert
        self._enabled = alert_cfg.get("enabled", True)
        self._cooldown_seconds = alert_cfg.get("cooldown_seconds", 180)
        self._smtp_server = alert_cfg.get("smtp_server", "smtp.gmail.com")
        self._smtp_port = alert_cfg.get("smtp_port", 587)
        self._sender_email = alert_cfg.get("sender_email", "")
        self._sender_password = alert_cfg.get("sender_password", "")
        env_password = os.environ.get("STRANGER_DETECTION_SMTP_PASSWORD")
        if env_password:
            self._sender_password = env_password
        self._receiver_email = alert_cfg.get("receiver_email", "")

        snapshot_cfg = config.snapshot
        self._snapshot_dir = os.path.abspath(snapshot_cfg.get("save_to", "./snapshots/"))
        self._max_snapshots = snapshot_cfg.get("max_snapshots", 100)

        self._last_alert_times = {}

        if self._enabled:
            logger.info("报警模块已初始化 (cooldown=%ds)", self._cooldown_seconds)
        else:
            logger.info("报警模块已禁用")

    def send_alert(self, frame: np.ndarray, stranger_id: str = "unknown") -> bool:
        if not self._enabled:
            logger.debug("报警已禁用，跳过发送")
            return False

        now = time.time()
        last_alert_time = self._last_alert_times.get(stranger_id, 0.0)
        if now - last_alert_time < self._cooldown_seconds:
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
        try:
            self._save_snapshot(image_data)
        except OSError as exc:
            logger.error("保存陌生人截图失败: %s", exc)
        self._last_alert_times[stranger_id] = now

        if not self._sender_email or not self._receiver_email:
            logger.warning("邮件发送方或接收方未配置，截图已保存但跳过邮件")
            return False
        if not self._sender_password:
            logger.warning("SMTP 授权码未配置，截图已保存但跳过邮件")
            return False

        msg = MIMEMultipart()
        msg["Subject"] = f"[陌生人警报] 检测到陌生人 ({stranger_id})"
        msg["From"] = self._sender_email
        msg["To"] = self._receiver_email

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
            server.sendmail(self._sender_email, self._receiver_email, msg.as_string())
        except (smtplib.SMTPException, OSError) as e:
            logger.error("邮件发送失败: %s", e)
            return False
        finally:
            if server:
                try:
                    server.quit()
                except (smtplib.SMTPException, OSError):
                    logger.debug("关闭 SMTP 连接失败", exc_info=True)

        logger.info("报警邮件已发送至 %s", self._receiver_email)
        return True

    def _save_snapshot(self, image_data: bytes) -> None:
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
