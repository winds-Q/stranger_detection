import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        self._device_id = device_id
        self._cap = cv2.VideoCapture(device_id)

        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"无法打开摄像头 (device_id={device_id})")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width != width or actual_height != height:
            logger.warning(
                "摄像头分辨率设置为 %dx%d，实际分辨率为 %dx%d",
                width, height, actual_width, actual_height
            )

        logger.info("摄像头已打开 (device_id=%d, %dx%d)", device_id, actual_width, actual_height)

    def get_frame(self) -> Optional[np.ndarray]:
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("读取摄像头帧失败")
            return None
        return frame

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()
            logger.info("摄像头已释放")

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
        return False

    def __del__(self) -> None:
        self.release()
