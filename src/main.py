import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera import Camera
from detector import FaceDetector
from recognizer import FaceRecognizer, StrangerTracker
from alerter import Alerter
from config_loader import Config
from logger import setup_logger
from processing import FrameProcessingController, StrangerConfirmation
from events import StrangerEventManager

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    setup_logger(log_dir=os.path.join(PROJECT_ROOT, "logs"))

    config_path = os.environ.get(
        "STRANGER_DETECTION_CONFIG",
        os.path.join(PROJECT_ROOT, "config.yaml"),
    )
    config = Config(config_path)

    camera = Camera(
        device_id=config.camera["device_id"],
        width=config.camera.get("width", 640),
        height=config.camera.get("height", 480),
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
    processor = FrameProcessingController(**config.processing)
    confirmation = StrangerConfirmation(**config.detection_confirmation)
    event_manager = StrangerEventManager(**config.stranger_tracking)

    logger.info("陌生人检测系统已启动 (config=%s)", config_path)

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
                time.sleep(0.05)
                continue

            face_encodings = detector.encode(detection_frame, face_locations)

            for encoding in face_encodings:
                if recognizer.is_stranger(encoding):
                    stranger_id = stranger_tracker.identify(encoding)
                    confirmed = confirmation.observe(stranger_id)
                    event_id = event_manager.observe(stranger_id, confirmed)
                    if event_id:
                        alerter.send_alert(frame, event_id)

            confirmation.cleanup()
            for departed_id in event_manager.mark_departures():
                confirmation.reset(departed_id)

            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("收到退出信号，正在关闭...")
    finally:
        camera.release()
        logger.info("系统已退出")


if __name__ == "__main__":
    main()
