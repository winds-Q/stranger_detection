import io
import os
import sqlite3
import sys
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TEST_TEMP_ROOT = os.path.join(PROJECT_ROOT, "tests", ".tmp")
os.makedirs(TEST_TEMP_ROOT, exist_ok=True)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from alerter import Alerter, AsyncAlertDispatcher
from camera import Camera
from config_loader import Config
from recognizer import FaceRecognizer, StrangerTracker
from processing import FrameProcessingController, StrangerConfirmation
from events import StrangerEventManager
from visual import annotate_frame
from health import RuntimeHealth
from storage import AlertEventRepository
from web import app as web_app


class AlerterTests(unittest.TestCase):
    def setUp(self):
        for name in os.listdir(TEST_TEMP_ROOT):
            if name.startswith("stranger_") and name.endswith(".jpg"):
                os.remove(os.path.join(TEST_TEMP_ROOT, name))

    def _config(self, directory, port=587, password="secret"):
        config = Config(os.path.join(directory, "missing.yaml"))
        config._data["alert"].update({
            "enabled": True,
            "cooldown_seconds": 60,
            "smtp_server": "smtp.example.com",
            "smtp_port": port,
            "sender_email": "sender@example.com",
            "sender_password": password,
            "receiver_email": "receiver@example.com",
        })
        config._data["snapshot"].update({
            "save_to": directory,
            "max_snapshots": 10,
        })
        return config

    def test_snapshot_is_saved_when_password_is_missing(self):
        alerter = Alerter(self._config(TEST_TEMP_ROOT, password=""))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        self.assertFalse(alerter.send_alert(frame))
        snapshots = [
            name for name in os.listdir(TEST_TEMP_ROOT)
            if name.startswith("stranger_") and name.endswith(".jpg")
        ]
        self.assertEqual(1, len(snapshots))

    def test_cooldown_is_independent_for_each_stranger(self):
        alerter = Alerter(self._config(TEST_TEMP_ROOT, password=""))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        self.assertFalse(alerter.send_alert(frame, "stranger-1"))
        first_count = len([
            name for name in os.listdir(TEST_TEMP_ROOT)
            if name.startswith("stranger_") and name.endswith(".jpg")
        ])
        self.assertFalse(alerter.send_alert(frame, "stranger-1"))
        self.assertFalse(alerter.send_alert(frame, "stranger-2"))
        final_count = len([
            name for name in os.listdir(TEST_TEMP_ROOT)
            if name.startswith("stranger_") and name.endswith(".jpg")
        ])

        self.assertEqual(1, first_count)
        self.assertEqual(2, final_count)

    @patch("alerter.smtplib.SMTP_SSL")
    def test_port_465_uses_smtp_ssl(self, smtp_ssl):
        smtp_ssl.return_value = MagicMock()
        alerter = Alerter(self._config(TEST_TEMP_ROOT, port=465))

        self.assertTrue(alerter.send_alert(np.zeros((8, 8, 3), dtype=np.uint8)))
        smtp_ssl.assert_called_once_with("smtp.example.com", 465, timeout=15)

    @patch("alerter.smtplib.SMTP")
    def test_sends_one_alert_to_multiple_receivers(self, smtp):
        smtp.return_value = MagicMock()
        config = self._config(TEST_TEMP_ROOT)
        config._data["alert"]["receiver_emails"] = [
            "first@example.com",
            "second@example.com",
            "FIRST@example.com",
        ]
        alerter = Alerter(config)

        self.assertTrue(alerter.send_alert(np.zeros((8, 8, 3), dtype=np.uint8)))

        sendmail_args = smtp.return_value.sendmail.call_args.args
        self.assertEqual(
            ["first@example.com", "second@example.com"],
            sendmail_args[1],
        )

    def test_parses_comma_semicolon_and_newline_receiver_text(self):
        self.assertEqual(
            ["a@example.com", "b@example.com", "c@example.com"],
            Alerter._normalize_receivers(
                "a@example.com, b@example.com;c@example.com\nA@example.com"
            ),
        )
        self.assertEqual([], Alerter._normalize_receivers("not-an-email"))


class AsyncAlertDispatcherTests(unittest.TestCase):
    def test_dispatches_in_background_and_retries_failure(self):
        alerter = MagicMock()
        alerter.is_configured = True
        alerter.send_alert.side_effect = [False, True]
        results = []
        dispatcher = AsyncAlertDispatcher(
            alerter,
            cooldown_seconds=0,
            retry_count=1,
            retry_backoff_seconds=0,
            result_callback=lambda event_id, success: results.append((event_id, success)),
        )

        self.assertTrue(dispatcher.submit(np.zeros((4, 4, 3)), "event-1"))
        dispatcher._queue.join()
        dispatcher.close()

        self.assertEqual(2, alerter.send_alert.call_count)
        first_call = alerter.send_alert.call_args_list[0]
        retry_call = alerter.send_alert.call_args_list[1]
        self.assertFalse(first_call.kwargs["bypass_cooldown"])
        self.assertTrue(retry_call.kwargs["bypass_cooldown"])
        self.assertFalse(retry_call.kwargs["save_snapshot"])
        self.assertEqual([("event-1", True)], results)

    def test_rejects_duplicate_event_during_cooldown(self):
        now = [10.0]
        alerter = MagicMock()
        alerter.is_configured = False
        dispatcher = AsyncAlertDispatcher(
            alerter,
            cooldown_seconds=180,
            clock=lambda: now[0],
        )

        self.assertTrue(dispatcher.submit(np.zeros((2, 2, 3)), "event-1"))
        self.assertFalse(dispatcher.submit(np.zeros((2, 2, 3)), "event-1"))
        dispatcher._queue.join()
        dispatcher.close()


class CameraTests(unittest.TestCase):
    def test_reconnects_after_consecutive_read_failures(self):
        frame = np.ones((4, 4, 3), dtype=np.uint8)
        first_capture = MagicMock()
        first_capture.isOpened.return_value = True
        first_capture.read.side_effect = [(False, None), (False, None)]
        first_capture.get.side_effect = [640, 480]
        second_capture = MagicMock()
        second_capture.isOpened.return_value = True
        second_capture.read.return_value = (True, frame)
        second_capture.get.side_effect = [640, 480]
        factory = MagicMock(side_effect=[first_capture, second_capture])
        now = [0.0]
        camera = Camera(
            read_failure_threshold=2,
            reconnect_interval_seconds=5,
            capture_factory=factory,
            clock=lambda: now[0],
        )

        self.assertIsNone(camera.get_frame())
        self.assertIsNone(camera.get_frame())
        self.assertFalse(camera.is_opened())
        now[0] = 4.9
        self.assertIsNone(camera.get_frame())
        self.assertEqual(1, factory.call_count)
        now[0] = 5.0
        self.assertTrue(np.array_equal(frame, camera.get_frame()))
        self.assertEqual(2, factory.call_count)
        camera.release()


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.client = web_app.app.test_client()

    def tearDown(self):
        web_app._stop_event.set()
        thread = web_app._detection_thread
        if thread and thread.is_alive():
            thread.join(timeout=2)
        web_app._detection_thread = None
        web_app._detection_error = None

    def test_web_paths_point_to_project_root(self):
        self.assertEqual(PROJECT_ROOT, web_app.PROJECT_ROOT)
        self.assertEqual(
            os.path.join(PROJECT_ROOT, "known_faces"),
            web_app._known_faces_dir,
        )

    def test_status_contains_health_fields_without_secrets(self):
        response = self.client.get("/api/status")
        data = response.get_json()
        for field in (
            "camera_state", "detection_fps", "known_faces", "alerts_today",
            "smtp_configured", "alert_queue_pending", "last_frame_time",
        ):
            self.assertIn(field, data)
        self.assertNotIn("sender_password", data)

    def test_faces_api_hides_gitkeep_and_non_image_files(self):
        with (
            patch.object(web_app, "_known_faces_dir", TEST_TEMP_ROOT),
            patch("web.app.os.listdir", return_value=[".gitkeep", "notes.txt"]),
            patch("web.app.os.path.isfile", return_value=True),
        ):
                response = self.client.get("/api/faces")
        self.assertEqual([], response.get_json())

    def test_invalid_upload_is_rejected(self):
        with patch.object(web_app, "_known_faces_dir", TEST_TEMP_ROOT):
            response = self.client.post(
                "/api/faces/upload",
                data={"file": (io.BytesIO(b"not an image"), "person.jpg")},
                content_type="multipart/form-data",
            )
        self.assertEqual(400, response.status_code)
        self.assertIn("有效图片", response.get_json()["message"])

    def test_valid_upload_uses_safe_filename(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)

        upload_path = os.path.join(TEST_TEMP_ROOT, "person.jpg")
        if os.path.exists(upload_path):
            os.remove(upload_path)
        with patch.object(web_app, "_known_faces_dir", TEST_TEMP_ROOT):
            response = self.client.post(
                "/api/faces/upload",
                data={
                    "file": (
                        io.BytesIO(encoded.tobytes()),
                        "../../person.jpg",
                    )
                },
                content_type="multipart/form-data",
            )
            self.assertTrue(os.path.isfile(upload_path))
        os.remove(upload_path)

        self.assertEqual(200, response.status_code)

    def test_start_reports_camera_failure(self):
        with patch.object(web_app, "Camera", side_effect=RuntimeError("camera unavailable")):
            response = self.client.post("/api/detect/start")

        self.assertEqual(503, response.status_code)
        self.assertIn("camera unavailable", response.get_json()["message"])

    def test_config_validation(self):
        response = self.client.post(
            "/api/config",
            json={"smtp_port": 70000, "cooldown_seconds": -1},
        )
        self.assertEqual(400, response.status_code)

    def test_config_api_accepts_multiple_receiver_addresses(self):
        with patch.object(web_app, "_load_yaml_config", return_value={}), patch.object(
            web_app, "_save_yaml_config"
        ) as save_config:
            response = self.client.post(
                "/api/config",
                json={"receiver_email": "a@example.com; b@example.com"},
            )

        self.assertEqual(200, response.status_code)
        saved = save_config.call_args.args[0]
        self.assertEqual(
            ["a@example.com", "b@example.com"],
            saved["alert"]["receiver_emails"],
        )

    def test_config_api_rejects_invalid_receiver_address(self):
        with patch.object(web_app, "_load_yaml_config", return_value={}):
            response = self.client.post(
                "/api/config",
                json={"receiver_email": "valid@example.com, invalid"},
            )
        self.assertEqual(400, response.status_code)

    def test_log_stream_declares_reconnect_delay(self):
        response = self.client.get("/api/logs/stream", buffered=False)
        try:
            self.assertEqual(b"retry: 3000\n\n", next(response.response))
        finally:
            response.close()

    def test_camera_scan_returns_available_devices_and_releases_all(self):
        unavailable = MagicMock()
        unavailable.isOpened.return_value = False
        available = MagicMock()
        available.isOpened.return_value = True
        available.read.return_value = (
            True,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        captures = [available, unavailable]
        with patch("web.app.cv2.VideoCapture", side_effect=captures):
            response = self.client.post(
                "/api/cameras/scan",
                json={"max_devices": 2},
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(0, response.get_json()["devices"][0]["device_id"])
        self.assertEqual(640, response.get_json()["devices"][0]["width"])
        available.release.assert_called_once()
        unavailable.release.assert_called_once()

    def test_camera_selection_updates_config(self):
        with patch.object(
            web_app,
            "_load_yaml_config",
            return_value={"camera": {"device_id": 0}},
        ), patch.object(web_app, "_save_yaml_config") as save_config:
            response = self.client.post(
                "/api/cameras/select",
                json={"device_id": 2},
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, save_config.call_args.args[0]["camera"]["device_id"])

    def test_camera_selection_rejects_invalid_device_id(self):
        response = self.client.post(
            "/api/cameras/select",
            json={"device_id": 20},
        )
        self.assertEqual(400, response.status_code)

    def test_detection_start_is_blocked_during_camera_scan(self):
        web_app._camera_operation_lock.acquire()
        try:
            response = self.client.post("/api/detect/start")
        finally:
            web_app._camera_operation_lock.release()
        self.assertEqual(409, response.status_code)


class StrangerTrackerTests(unittest.TestCase):
    def test_same_face_gets_same_id_and_different_face_gets_new_id(self):
        tracker = StrangerTracker(tolerance=0.5)
        face_a = np.zeros(128)
        face_a_changed = np.full(128, 0.01)
        face_b = np.ones(128)

        first_id = tracker.identify(face_a)
        self.assertEqual(first_id, tracker.identify(face_a_changed))
        self.assertNotEqual(first_id, tracker.identify(face_b))

    def test_retains_multiple_recent_samples_without_unbounded_growth(self):
        tracker = StrangerTracker(tolerance=0.5, max_samples=3)
        stranger_id = tracker.identify(np.zeros(128))

        for value in (0.01, 0.02, 0.03, 0.04):
            self.assertEqual(
                stranger_id,
                tracker.identify(np.full(128, value)),
            )

        self.assertEqual(3, tracker.sample_count(stranger_id))

    def test_matches_any_sample_in_identity_history(self):
        tracker = StrangerTracker(tolerance=0.5, max_samples=5)
        stranger_id = tracker.identify(np.zeros(128))
        tracker.identify(np.full(128, 0.04))

        self.assertEqual(stranger_id, tracker.identify(np.zeros(128)))


class FaceRecognizerLoadingTests(unittest.TestCase):
    def test_ignores_gitkeep_without_loading_it_as_an_image(self):
        with (
            patch("recognizer.os.path.isdir", return_value=True),
            patch("recognizer.os.listdir", return_value=[".gitkeep"]),
            patch("recognizer.os.path.isfile", return_value=True),
            patch("recognizer.face_recognition.load_image_file") as loader,
        ):
                recognizer = FaceRecognizer(TEST_TEMP_ROOT)
        loader.assert_not_called()
        self.assertEqual(0, recognizer.known_count)


class FrameProcessingControllerTests(unittest.TestCase):
    def test_skips_frames_and_limits_detection_fps(self):
        now = [0.0]
        controller = FrameProcessingController(
            detect_every_n_frames=2,
            max_detection_fps=2,
            clock=lambda: now[0],
        )

        self.assertTrue(controller.should_process())
        self.assertFalse(controller.should_process())
        now[0] = 0.2
        self.assertFalse(controller.should_process())
        self.assertFalse(controller.should_process())
        now[0] = 0.6
        self.assertTrue(controller.should_process())

    def test_resizes_frame_and_restores_locations(self):
        controller = FrameProcessingController(detection_scale=0.5)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        resized = controller.prepare_frame(frame)

        self.assertEqual((50, 100, 3), resized.shape)
        self.assertEqual([(20, 40, 60, 10)], controller.restore_locations([
            (10, 20, 30, 5)
        ]))


class StrangerConfirmationTests(unittest.TestCase):
    def test_requires_enough_hits_and_minimum_duration(self):
        now = [0.0]
        confirmation = StrangerConfirmation(
            window_seconds=3,
            required_hits=3,
            minimum_duration_seconds=1,
            clock=lambda: now[0],
        )

        self.assertFalse(confirmation.observe("stranger-1"))
        now[0] = 0.5
        self.assertFalse(confirmation.observe("stranger-1"))
        now[0] = 1.0
        self.assertTrue(confirmation.observe("stranger-1"))

    def test_confirmation_is_independent_and_old_hits_expire(self):
        now = [0.0]
        confirmation = StrangerConfirmation(
            window_seconds=1,
            required_hits=2,
            minimum_duration_seconds=0,
            clock=lambda: now[0],
        )

        self.assertFalse(confirmation.observe("stranger-1"))
        self.assertFalse(confirmation.observe("stranger-2"))
        now[0] = 2.0
        self.assertFalse(confirmation.observe("stranger-1"))


class StrangerEventManagerTests(unittest.TestCase):
    def test_tracks_stay_departure_and_reentry(self):
        now = [0.0]
        manager = StrangerEventManager(
            leave_timeout_seconds=30,
            clock=lambda: now[0],
            session_id="test",
        )

        self.assertIsNone(manager.observe("stranger-1", confirmed=False))
        self.assertEqual(
            "test-stranger-1-event-1",
            manager.observe("stranger-1", confirmed=True),
        )
        now[0] = 20
        self.assertEqual(
            "test-stranger-1-event-1",
            manager.observe("stranger-1", confirmed=True),
        )
        now[0] = 51
        self.assertEqual(
            "test-stranger-1-event-2",
            manager.observe("stranger-1", confirmed=True),
        )

    def test_marks_departed_strangers(self):
        now = [0.0]
        manager = StrangerEventManager(leave_timeout_seconds=5, clock=lambda: now[0])
        manager.observe("stranger-1", confirmed=True)
        now[0] = 6

        self.assertEqual(["stranger-1"], manager.mark_departures())
        self.assertFalse(manager.get_event("stranger-1").active)


class VisualAnnotationTests(unittest.TestCase):
    def test_draws_red_and_green_face_annotations_on_copy(self):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        annotated = annotate_frame(frame, [
            ((20, 60, 70, 10), "stranger-1", True),
            ((20, 140, 70, 90), "alice", False),
        ])

        self.assertFalse(np.any(frame))
        self.assertGreater(int(annotated[:, :, 2].max()), 0)
        self.assertGreater(int(annotated[:, :, 1].max()), 0)


class RuntimeHealthTests(unittest.TestCase):
    def test_records_activity_fps_and_daily_alerts(self):
        now = [1000.0]
        health = RuntimeHealth(clock=lambda: now[0])
        health.set_camera_state("connected")
        health.record_frame()
        health.record_detection()
        now[0] = 1000.5
        health.record_detection()
        health.record_alert()
        snapshot = health.snapshot(known_faces=3, smtp_configured=True, alert_queue_pending=2)

        self.assertEqual("connected", snapshot["camera_state"])
        self.assertEqual(2.0, snapshot["detection_fps"])
        self.assertEqual(3, snapshot["known_faces"])
        self.assertEqual(1, snapshot["alerts_today"])
        self.assertEqual(2, snapshot["alert_queue_pending"])


class AlertEventRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.database_path = os.path.join(TEST_TEMP_ROOT, "test_alerts.db")
        for suffix in ("", "-wal", "-shm"):
            path = self.database_path + suffix
            if os.path.exists(path):
                os.remove(path)

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            path = self.database_path + suffix
            if os.path.exists(path):
                os.remove(path)

    def test_records_updates_and_closes_event(self):
        repository = AlertEventRepository(self.database_path)
        repository.record_observation("event-1", "stranger-1")
        repository.record_observation("event-1", "stranger-1")
        repository.update_notification("event-1", "sent")
        repository.mark_departed("stranger-1")

        self.assertEqual(1, repository.count())
        connection = sqlite3.connect(self.database_path)
        try:
            row = connection.execute(
                "SELECT notification_status, left_at FROM alert_events"
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual("sent", row[0])
        self.assertIsNotNone(row[1])


if __name__ == "__main__":
    unittest.main()
