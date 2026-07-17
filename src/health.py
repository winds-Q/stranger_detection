import threading
import time
from collections import deque
from datetime import datetime


class RuntimeHealth:
    def __init__(self, clock=time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self._camera_state = "disconnected"
            self._last_frame_time = None
            self._last_detection_time = None
            self._last_alert_time = None
            self._alerts_today = 0
            self._alerts_date = datetime.fromtimestamp(self._clock()).date()
            self._detection_times = deque(maxlen=30)

    def set_camera_state(self, state: str) -> None:
        with self._lock:
            self._camera_state = state

    def record_frame(self) -> None:
        with self._lock:
            self._last_frame_time = self._clock()

    def record_detection(self) -> None:
        now = self._clock()
        with self._lock:
            self._last_detection_time = now
            self._detection_times.append(now)

    def record_alert(self) -> None:
        now = self._clock()
        today = datetime.fromtimestamp(now).date()
        with self._lock:
            if today != self._alerts_date:
                self._alerts_today = 0
                self._alerts_date = today
            self._alerts_today += 1
            self._last_alert_time = now

    def snapshot(self, known_faces=0, smtp_configured=False, alert_queue_pending=0):
        with self._lock:
            detection_fps = 0.0
            if len(self._detection_times) >= 2:
                duration = self._detection_times[-1] - self._detection_times[0]
                if duration > 0:
                    detection_fps = (len(self._detection_times) - 1) / duration
            return {
                "camera_state": self._camera_state,
                "camera_reconnecting": self._camera_state == "reconnecting",
                "known_faces": int(known_faces),
                "last_frame_time": self._format_time(self._last_frame_time),
                "last_detection_time": self._format_time(self._last_detection_time),
                "last_alert_time": self._format_time(self._last_alert_time),
                "alerts_today": self._alerts_today,
                "smtp_configured": bool(smtp_configured),
                "alert_queue_pending": int(alert_queue_pending),
                "detection_fps": round(detection_fps, 1),
            }

    @staticmethod
    def _format_time(timestamp):
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")
