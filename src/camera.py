import logging
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        read_failure_threshold: int = 10,
        reconnect_interval_seconds: float = 5,
        capture_factory=cv2.VideoCapture,
        clock=time.monotonic,
    ):
        self._device_id = device_id
        self._width = width
        self._height = height
        self._read_failure_threshold = max(1, int(read_failure_threshold))
        self._reconnect_interval_seconds = max(
            0.1, float(reconnect_interval_seconds)
        )
        self._capture_factory = capture_factory
        self._clock = clock
        self._cap = None
        self._consecutive_failures = 0
        self._next_reconnect_time = 0.0
        self._closed = False
        self._open(raise_on_failure=True)

    def _open(self, raise_on_failure: bool) -> bool:
        cap = self._capture_factory(self._device_id)
        if not cap.isOpened():
            cap.release()
            self._cap = None
            if raise_on_failure:
                raise RuntimeError(f"无法打开摄像头 (device_id={self._device_id})")
            logger.warning("摄像头重连失败，稍后重试 (device_id=%d)", self._device_id)
            self._next_reconnect_time = (
                self._clock() + self._reconnect_interval_seconds
            )
            return False

        self._cap = cap
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width != self._width or actual_height != self._height:
            logger.warning(
                "摄像头分辨率设置为 %dx%d，实际分辨率为 %dx%d",
                self._width, self._height, actual_width, actual_height
            )

        self._consecutive_failures = 0
        logger.info(
            "摄像头已打开 (device_id=%d, %dx%d)",
            self._device_id, actual_width, actual_height,
        )
        return True

    def get_frame(self) -> Optional[np.ndarray]:
        if self._closed:
            return None
        if self._cap is None:
            if self._clock() < self._next_reconnect_time:
                return None
            if not self._open(raise_on_failure=False):
                return None

        ret, frame = self._cap.read()
        if not ret:
            self._consecutive_failures += 1
            logger.warning(
                "读取摄像头帧失败 (%d/%d)",
                self._consecutive_failures,
                self._read_failure_threshold,
            )
            if self._consecutive_failures >= self._read_failure_threshold:
                logger.error("摄像头连续读取失败，准备自动重连")
                self._release_capture()
                self._next_reconnect_time = (
                    self._clock() + self._reconnect_interval_seconds
                )
            return None
        self._consecutive_failures = 0
        return frame

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def state(self) -> str:
        if self._closed:
            return "disconnected"
        if self.is_opened():
            return "connected"
        return "reconnecting"

    def _release_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def release(self) -> None:
        was_opened = self.is_opened()
        self._closed = True
        self._release_capture()
        if was_opened:
            logger.info("摄像头已释放")

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
        return False

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.release()
