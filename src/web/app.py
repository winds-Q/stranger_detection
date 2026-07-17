import json
import logging
import os
import queue
import re
import sys
import threading
import time

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import Camera
from detector import FaceDetector
from recognizer import (
    FaceRecognizer,
    StrangerTracker,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from alerter import Alerter, AsyncAlertDispatcher
from config_loader import Config
from logger import setup_logger
from processing import FrameProcessingController, StrangerConfirmation
from events import StrangerEventManager
from visual import annotate_frame
from health import RuntimeHealth
from storage import AlertEventRepository
from cleanup import RetentionCleaner, RetentionWorker
from face_validation import FaceImageValidator

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

_detection_thread: threading.Thread | None = None
_stop_event = threading.Event()
_startup_event = threading.Event()
_log_queue: queue.Queue = queue.Queue(maxsize=200)
_known_faces_dir = os.path.join(PROJECT_ROOT, "known_faces")
_reload_event = threading.Event()  # 通知检测线程重新加载熟人库
_detection_error: str | None = None
_allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
_camera_operation_lock = threading.Lock()
_camera_scan_max_devices = 6
_health = RuntimeHealth()
_active_recognizer = None
_active_alerter = None
_active_alert_dispatcher = None
_event_repository = AlertEventRepository(os.path.join(PROJECT_ROOT, "data", "alerts.db"))


_handler = None


def _init_sse_handler():
    global _handler
    if _handler is not None:
        return
    _handler = _SSELogHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(_handler)


class _SSELogHandler(logging.Handler):
    def emit(self, record):
        try:
            entry = self.format(record)
            _log_queue.put_nowait(entry)
        except queue.Full:
            pass


def _reload_recognizer(tolerance: float) -> FaceRecognizer:
    """重新加载熟人库。"""
    return FaceRecognizer(
        known_faces_dir=_known_faces_dir,
        tolerance=tolerance,
    )


def _detection_loop(config):
    global _detection_error, _active_recognizer, _active_alerter
    global _active_alert_dispatcher
    _log = logging.getLogger(__name__)
    camera = None
    alert_dispatcher = None

    try:
        camera = Camera(
            device_id=config.camera["device_id"],
            width=config.camera.get("width", 640),
            height=config.camera.get("height", 480),
            read_failure_threshold=config.camera.get(
                "read_failure_threshold", 10
            ),
            reconnect_interval_seconds=config.camera.get(
                "reconnect_interval_seconds", 5
            ),
        )
        detector = FaceDetector(
            model=config.recognition.get("model", "hog"),
            upsample=config.recognition.get("upsample", 1),
        )
        recognizer = FaceRecognizer(
            known_faces_dir=_known_faces_dir,
            tolerance=config.recognition.get("tolerance", 0.5),
        )
        stranger_tracker = StrangerTracker(
            tolerance=config.recognition.get("stranger_tolerance", 0.5),
            retention_seconds=config.recognition.get(
                "stranger_retention_seconds", 3600
            ),
            max_samples=config.recognition.get("stranger_max_samples", 5),
        )
        alerter = Alerter(
            config,
            snapshot_callback=lambda event_id, path: (
                _event_repository.update_notification(
                    event_id, "pending", snapshot_path=path
                )
            ),
        )
        alert_dispatcher = AsyncAlertDispatcher(
            alerter,
            cooldown_seconds=config.alert.get("cooldown_seconds", 180),
            queue_size=config.alert.get("queue_size", 20),
            retry_count=config.alert.get("retry_count", 2),
            retry_backoff_seconds=config.alert.get(
                "retry_backoff_seconds", 2
            ),
            result_callback=lambda event_id, success: (
                _event_repository.update_notification(
                    event_id, "sent" if success else "failed"
                )
            ),
        )
        processor = FrameProcessingController(**config.processing)
        confirmation = StrangerConfirmation(**config.detection_confirmation)
        event_manager = StrangerEventManager(**config.stranger_tracking)
        _detection_error = None
        _active_recognizer = recognizer
        _active_alerter = alerter
        _active_alert_dispatcher = alert_dispatcher
        _health.reset()
        _health.set_camera_state(camera.state)
        _startup_event.set()
        _log.info("检测已启动")

        while not _stop_event.is_set():
            # 检查是否需要重新加载熟人库
            if _reload_event.is_set():
                _reload_event.clear()
                recognizer = _reload_recognizer(
                    tolerance=config.recognition.get("tolerance", 0.5)
                )
                _log.info("熟人库已重新加载 (%d 人)", recognizer.known_count)

            frame = camera.get_frame()
            _health.set_camera_state(camera.state)
            if frame is None:
                time.sleep(0.1)
                continue
            _health.record_frame()

            if not processor.should_process():
                time.sleep(0.01)
                continue

            detection_frame = processor.prepare_frame(frame)
            _health.record_detection()
            face_locations = detector.detect(detection_frame)
            if not face_locations:
                for departed_id in event_manager.mark_departures():
                    confirmation.reset(departed_id)
                    _event_repository.mark_departed(departed_id)
                time.sleep(0.05)
                continue

            face_encodings = detector.encode(detection_frame, face_locations)
            original_locations = processor.restore_locations(face_locations)
            annotations = []
            alert_event_ids = []
            for encoding, location in zip(face_encodings, original_locations):
                known_name = recognizer.recognize(encoding)
                if known_name is None:
                    stranger_id = stranger_tracker.identify(encoding)
                    annotations.append((location, stranger_id, True))
                    confirmed = confirmation.observe(stranger_id)
                    event_id = event_manager.observe(stranger_id, confirmed)
                    if event_id:
                        _event_repository.record_observation(event_id, stranger_id)
                        alert_event_ids.append(event_id)
                else:
                    annotations.append((location, known_name, False))

            if alert_event_ids:
                alert_frame = annotate_frame(frame, annotations)
                for event_id in alert_event_ids:
                    if alert_dispatcher.submit(alert_frame, event_id):
                        _health.record_alert()

            confirmation.cleanup()
            for departed_id in event_manager.mark_departures():
                confirmation.reset(departed_id)
                _event_repository.mark_departed(departed_id)

            time.sleep(0.1)
    except Exception as exc:
        _detection_error = str(exc)
        _log.exception("检测线程异常退出: %s", exc)
    finally:
        _startup_event.set()
        if camera is not None:
            camera.release()
        _health.set_camera_state("disconnected")
        if alert_dispatcher is not None:
            alert_dispatcher.close()
        _active_recognizer = None
        _active_alerter = None
        _active_alert_dispatcher = None
        _log.info("检测已停止")


# ── Routes ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    running = _detection_thread is not None and _detection_thread.is_alive()
    health = _health.snapshot(
        known_faces=_active_recognizer.known_count if _active_recognizer else 0,
        smtp_configured=_active_alerter.is_configured if _active_alerter else False,
        alert_queue_pending=(
            _active_alert_dispatcher.pending_count
            if _active_alert_dispatcher else 0
        ),
    )
    return jsonify({"running": running, "error": _detection_error, **health})


@app.route("/api/cameras")
def api_cameras():
    cfg = _load_yaml_config()
    camera_cfg = cfg.get("camera", {})
    return jsonify({
        "selected_device_id": camera_cfg.get("device_id", 0),
        "scan_max_devices": _camera_scan_max_devices,
    })


@app.route("/api/events")
def api_events():
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        return jsonify({"ok": False, "message": "limit 必须是整数"}), 400
    status = request.args.get("status") or None
    if status not in (None, "pending", "sent", "failed"):
        return jsonify({"ok": False, "message": "通知状态不合法"}), 400
    events = _event_repository.list_events(limit=limit, status=status)
    for event in events:
        event["handled"] = bool(event["handled"])
        event["has_snapshot"] = bool(event.pop("snapshot_path", None))
        event.pop("notification_error", None)
    return jsonify(events)


@app.route("/api/events/<int:event_id>/handled", methods=["POST"])
def api_event_handled(event_id):
    data = request.get_json(silent=True) or {}
    handled = bool(data.get("handled", True))
    if not _event_repository.set_handled(event_id, handled):
        return jsonify({"ok": False, "message": "告警事件不存在"}), 404
    return jsonify({"ok": True, "handled": handled})


@app.route("/api/events/<int:event_id>", methods=["DELETE"])
def api_delete_event(event_id):
    deleted = _event_repository.delete_event(event_id)
    if deleted is None:
        return jsonify({"ok": False, "message": "告警事件不存在"}), 404
    snapshot_path = deleted["snapshot_path"]
    if snapshot_path and request.args.get("delete_snapshot") == "1":
        _safe_delete_snapshot(snapshot_path)
    return jsonify({"ok": True})


@app.route("/api/events/<int:event_id>/snapshot")
def api_event_snapshot(event_id):
    event = _event_repository.get_event(event_id)
    if not event or not event.get("snapshot_path"):
        return jsonify({"ok": False, "message": "截图不存在"}), 404
    path = os.path.abspath(event["snapshot_path"])
    snapshot_root = os.path.abspath(os.path.join(PROJECT_ROOT, "snapshots"))
    try:
        if os.path.commonpath([path, snapshot_root]) != snapshot_root:
            raise ValueError
    except ValueError:
        return jsonify({"ok": False, "message": "截图路径不合法"}), 400
    if not os.path.isfile(path):
        return jsonify({"ok": False, "message": "截图文件不存在"}), 404
    return send_file(path, mimetype="image/jpeg", conditional=True)


def _safe_delete_snapshot(snapshot_path):
    path = os.path.abspath(snapshot_path)
    snapshot_root = os.path.abspath(os.path.join(PROJECT_ROOT, "snapshots"))
    try:
        if os.path.commonpath([path, snapshot_root]) == snapshot_root and os.path.isfile(path):
            os.remove(path)
    except (ValueError, OSError):
        logging.getLogger(__name__).warning("删除告警截图失败")


@app.route("/api/cameras/scan", methods=["POST"])
def api_scan_cameras():
    if _detection_thread and _detection_thread.is_alive():
        return jsonify({
            "ok": False,
            "message": "请先停止检测，再扫描摄像头",
        }), 409

    data = request.get_json(silent=True) or {}
    try:
        max_devices = int(data.get("max_devices", _camera_scan_max_devices))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "扫描数量必须是整数"}), 400
    if not 1 <= max_devices <= 10:
        return jsonify({"ok": False, "message": "扫描数量必须在 1-10 之间"}), 400

    if not _camera_operation_lock.acquire(blocking=False):
        return jsonify({"ok": False, "message": "摄像头操作正在进行中"}), 409

    devices = []
    try:
        for device_id in range(max_devices):
            cap = cv2.VideoCapture(device_id)
            try:
                if not cap.isOpened():
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                height, width = frame.shape[:2]
                devices.append({
                    "device_id": device_id,
                    "width": int(width),
                    "height": int(height),
                    "label": f"摄像头 {device_id} ({width}×{height})",
                })
            finally:
                cap.release()
    finally:
        _camera_operation_lock.release()

    return jsonify({"ok": True, "devices": devices})


@app.route("/api/cameras/select", methods=["POST"])
def api_select_camera():
    if _detection_thread and _detection_thread.is_alive():
        return jsonify({
            "ok": False,
            "message": "请先停止检测，再切换摄像头",
        }), 409

    data = request.get_json(silent=True) or {}
    try:
        device_id = int(data.get("device_id"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "摄像头编号必须是整数"}), 400
    if not 0 <= device_id <= 9:
        return jsonify({"ok": False, "message": "摄像头编号必须在 0-9 之间"}), 400

    cfg = _load_yaml_config()
    camera_cfg = cfg.setdefault("camera", {})
    camera_cfg["device_id"] = device_id
    _save_yaml_config(cfg)
    return jsonify({
        "ok": True,
        "message": f"已选择摄像头 {device_id}，下次启动检测时生效",
        "device_id": device_id,
    })


@app.route("/api/detect/start", methods=["POST"])
def api_start():
    global _detection_thread, _detection_error

    if _detection_thread and _detection_thread.is_alive():
        return jsonify({"ok": False, "message": "检测已在运行中"}), 409
    if _camera_operation_lock.locked():
        return jsonify({
            "ok": False,
            "message": "摄像头扫描正在进行，请稍后再启动检测",
        }), 409

    _stop_event.clear()
    _startup_event.clear()
    _detection_error = None

    config_path = os.environ.get(
        "STRANGER_DETECTION_CONFIG",
        os.path.join(PROJECT_ROOT, "config.yaml"),
    )
    config = Config(config_path)

    _detection_thread = threading.Thread(
        target=_detection_loop, args=(config,), daemon=True
    )
    _detection_thread.start()

    if not _startup_event.wait(timeout=5):
        _stop_event.set()
        return jsonify({"ok": False, "message": "检测启动超时，请检查摄像头"}), 503
    if _detection_error or not _detection_thread.is_alive():
        return jsonify({
            "ok": False,
            "message": f"检测启动失败: {_detection_error or '线程已退出'}",
        }), 503
    return jsonify({"ok": True, "message": "检测已启动"})


@app.route("/api/detect/stop", methods=["POST"])
def api_stop():
    global _detection_thread

    if not _detection_thread or not _detection_thread.is_alive():
        return jsonify({"ok": False, "message": "检测未在运行"}), 409

    _stop_event.set()
    _detection_thread.join(timeout=10)
    if _detection_thread.is_alive():
        return jsonify({"ok": False, "message": "停止超时，检测线程仍在运行"}), 503
    return jsonify({"ok": True, "message": "检测已停止"})


@app.route("/api/faces")
def api_faces():
    entries = []
    if os.path.isdir(_known_faces_dir):
        for fname in os.listdir(_known_faces_dir):
            path = os.path.join(_known_faces_dir, fname)
            extension = os.path.splitext(fname)[1].lower()
            if (
                os.path.isfile(path)
                and extension in SUPPORTED_IMAGE_EXTENSIONS
            ):
                entries.append({
                    "name": fname,
                    "size": os.path.getsize(path),
                })
    return jsonify(entries)


@app.route("/api/faces/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "message": "未选择文件"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename or "")
    if not filename:
        return jsonify({"ok": False, "message": "文件名为空"}), 400

    extension = os.path.splitext(filename)[1].lower()
    if extension not in _allowed_extensions:
        return jsonify({"ok": False, "message": "仅支持 jpg、jpeg、png、bmp 图片"}), 400

    image_data = file.read()
    image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"ok": False, "message": "文件不是有效图片"}), 400

    validation = FaceImageValidator(Config(_CONFIG_PATH), _known_faces_dir).validate(
        image, filename
    )
    if not validation.ok:
        return jsonify({"ok": False, "message": validation.message}), 400

    os.makedirs(_known_faces_dir, exist_ok=True)
    save_path = os.path.join(_known_faces_dir, filename)
    with open(save_path, "wb") as output:
        output.write(image_data)
    _reload_event.set()  # 通知检测线程重新加载熟人库
    return jsonify({"ok": True, "message": f"{filename} 已上传"})


@app.route("/api/faces/<filename>", methods=["DELETE"])
def api_delete_face(filename):
    safe_name = secure_filename(filename)
    if not safe_name or safe_name != filename:
        return jsonify({"ok": False, "message": "文件名不合法"}), 400
    path = os.path.join(_known_faces_dir, safe_name)
    if not os.path.isfile(path):
        return jsonify({"ok": False, "message": "文件不存在"}), 404
    os.remove(path)
    _reload_event.set()  # 通知检测线程重新加载熟人库
    return jsonify({"ok": True, "message": f"{filename} 已删除"})


@app.route("/api/logs/stream")
def api_log_stream():
    def generate():
        yield "retry: 3000\n\n"
        while True:
            try:
                msg = _log_queue.get(timeout=1)
                yield f"data: {json.dumps({'msg': msg})}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


_CONFIG_PATH = os.environ.get(
    "STRANGER_DETECTION_CONFIG",
    os.path.join(PROJECT_ROOT, "config.yaml"),
)


def _load_yaml_config():
    if not os.path.isfile(_CONFIG_PATH):
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml_config(data):
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


@app.route("/api/config")
def api_get_config():
    cfg = _load_yaml_config()
    alert_cfg = cfg.get("alert", {})
    password = alert_cfg.get("sender_password", "")
    env_password = os.environ.get("STRANGER_DETECTION_SMTP_PASSWORD")
    if env_password:
        password = env_password
    masked = "****" if password else ""
    receiver_value = alert_cfg.get(
        "receiver_emails",
        alert_cfg.get("receiver_email", ""),
    )
    if isinstance(receiver_value, list):
        receiver_value = ", ".join(str(item) for item in receiver_value)

    return jsonify({
        "enabled": alert_cfg.get("enabled", True),
        "cooldown_seconds": alert_cfg.get("cooldown_seconds", 180),
        "smtp_server": alert_cfg.get("smtp_server", ""),
        "smtp_port": alert_cfg.get("smtp_port", 587),
        "sender_email": alert_cfg.get("sender_email", ""),
        "sender_password": masked,
        "receiver_email": receiver_value,
        "has_env_password": bool(env_password),
    })


@app.route("/api/config", methods=["POST"])
def api_update_config():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"ok": False, "message": "请求数据为空"}), 400

    try:
        smtp_port = int(data.get("smtp_port", 587))
        cooldown = int(data.get("cooldown_seconds", 180))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "端口和冷却时间必须是整数"}), 400
    if not 1 <= smtp_port <= 65535:
        return jsonify({"ok": False, "message": "SMTP 端口必须在 1-65535 之间"}), 400
    if cooldown < 0:
        return jsonify({"ok": False, "message": "冷却时间不能小于 0"}), 400

    cfg = _load_yaml_config()

    alert_cfg = cfg.setdefault("alert", {})
    if "enabled" in data:
        alert_cfg["enabled"] = bool(data["enabled"])
    if "cooldown_seconds" in data:
        alert_cfg["cooldown_seconds"] = cooldown
    if "smtp_server" in data:
        alert_cfg["smtp_server"] = str(data["smtp_server"])
    if "smtp_port" in data:
        alert_cfg["smtp_port"] = smtp_port
    if "sender_email" in data:
        alert_cfg["sender_email"] = str(data["sender_email"])
    if "sender_password" in data and data["sender_password"] and data["sender_password"] != "****":
        alert_cfg["sender_password"] = str(data["sender_password"])
    if "receiver_email" in data:
        receiver_text = str(data["receiver_email"])
        receivers = Alerter._normalize_receivers(receiver_text)
        raw_receivers = [
            item.strip()
            for item in re.split(r"[,;\n]", receiver_text)
            if item.strip()
        ]
        if any(not Alerter._normalize_receivers(item) for item in raw_receivers):
            return jsonify({
                "ok": False,
                "message": "一个或多个收件邮箱格式不正确",
            }), 400
        alert_cfg["receiver_emails"] = receivers
        alert_cfg.pop("receiver_email", None)

    _save_yaml_config(cfg)
    return jsonify({"ok": True, "message": "配置已保存"})


def main():
    setup_logger(log_dir=os.path.join(PROJECT_ROOT, "logs"))
    logging.getLogger().setLevel(logging.INFO)
    _init_sse_handler()
    cleanup_config = Config(_CONFIG_PATH)
    cleaner = RetentionCleaner(
        _event_repository,
        snapshot_dir=os.path.join(PROJECT_ROOT, "snapshots"),
        log_dir=os.path.join(PROJECT_ROOT, "logs"),
        snapshots_days=cleanup_config.retention.get("snapshots_days", 7),
        logs_days=cleanup_config.retention.get("logs_days", 14),
        events_days=cleanup_config.retention.get("events_days", 30),
    )
    retention_worker = RetentionWorker(
        cleaner,
        interval_hours=cleanup_config.retention.get("cleanup_interval_hours", 12),
    )
    retention_worker.start()
    host = os.environ.get("STRANGER_DETECTION_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("STRANGER_DETECTION_WEB_PORT", "5050"))
    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    finally:
        retention_worker.close()


if __name__ == "__main__":
    main()
