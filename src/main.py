import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera import Camera
from detector import FaceDetector
from recognizer import FaceRecognizer, StrangerTracker
from alerter import Alerter, AsyncAlertDispatcher
from config_loader import Config
from logger import setup_logger
from processing import FrameProcessingController, StrangerConfirmation
from events import StrangerEventManager
from visual import annotate_frame
from storage import AlertEventRepository
from cleanup import RetentionCleaner, RetentionWorker

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    setup_logger(log_dir=os.path.join(PROJECT_ROOT, "logs"))

    config_path = os.environ.get(
        "STRANGER_DETECTION_CONFIG",
        os.path.join(PROJECT_ROOT, "config.yaml"),
    )
    config = Config(config_path)
    database_path = config.database.get("path", "./data/alerts.db")
    if not os.path.isabs(database_path):
        database_path = os.path.join(PROJECT_ROOT, database_path)
    event_repository = AlertEventRepository(database_path)
    retention_worker = RetentionWorker(
        RetentionCleaner(
            event_repository,
            snapshot_dir=os.path.join(PROJECT_ROOT, "snapshots"),
            log_dir=os.path.join(PROJECT_ROOT, "logs"),
            snapshots_days=config.retention.get("snapshots_days", 7),
            logs_days=config.retention.get("logs_days", 14),
            events_days=config.retention.get("events_days", 30),
        ),
        interval_hours=config.retention.get("cleanup_interval_hours", 12),
    )
    retention_worker.start()

    camera = Camera(
        device_id=config.camera["device_id"],
        width=config.camera.get("width", 640),
        height=config.camera.get("height", 480),
        read_failure_threshold=config.camera.get("read_failure_threshold", 10),
        reconnect_interval_seconds=config.camera.get(
            "reconnect_interval_seconds", 5
        ),
    )
    detector = FaceDetector(
        model=config.recognition.get("model", "hog"),
        upsample=config.recognition.get("upsample", 1),
    )
    recognizer = FaceRecognizer(
        known_faces_dir=os.path.join(PROJECT_ROOT, "known_faces"),
        tolerance=config.recognition.get("tolerance", 0.5),
    )
    stranger_tracker = StrangerTracker(
        tolerance=config.recognition.get("stranger_tolerance", 0.5),
        retention_seconds=config.recognition.get(
            "stranger_retention_seconds", 3600
        ),
        max_samples=config.recognition.get("stranger_max_samples", 5),
    )
    alerter = Alerter(config)
    alert_dispatcher = AsyncAlertDispatcher(
        alerter,
        cooldown_seconds=config.alert.get("cooldown_seconds", 180),
        queue_size=config.alert.get("queue_size", 20),
        retry_count=config.alert.get("retry_count", 2),
        retry_backoff_seconds=config.alert.get("retry_backoff_seconds", 2),
        result_callback=lambda event_id, success: (
            event_repository.update_notification(
                event_id, "sent" if success else "failed"
            )
        ),
    )
    processor = FrameProcessingController(**config.processing)
    confirmation = StrangerConfirmation(**config.detection_confirmation)
    event_manager = StrangerEventManager(**config.stranger_tracking)

    logger.info("陌生人检测系统已启动 (config=%s)", config_path)

    known_last_logged = {}
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            if not processor.should_process():
                time.sleep(0.01)
                continue

            detection_frame = processor.prepare_frame(frame)
            face_locations = detector.detect(detection_frame)
            if not face_locations:
                for departed_id in event_manager.mark_departures():
                    confirmation.reset(departed_id)
                    event_repository.mark_departed(departed_id)
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
                        event_repository.record_observation(event_id, stranger_id)
                        alert_event_ids.append(event_id)
                else:
                    annotations.append((location, known_name, False))
                    now = time.monotonic()
                    if now - known_last_logged.get(known_name, 0) >= 30:
                        logger.info("检测到熟人：%s", known_name)
                        known_last_logged[known_name] = now

            if alert_event_ids:
                alert_frame = annotate_frame(frame, annotations)
                for event_id in alert_event_ids:
                    alert_dispatcher.submit(alert_frame, event_id)

            confirmation.cleanup()
            for departed_id in event_manager.mark_departures():
                confirmation.reset(departed_id)
                event_repository.mark_departed(departed_id)

            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("收到退出信号，正在关闭...")
    finally:
        camera.release()
        alert_dispatcher.close()
        retention_worker.close()
        logger.info("系统已退出")


if __name__ == "__main__":
    main()
