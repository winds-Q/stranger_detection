import time
from collections import deque
from typing import List, Tuple

import cv2
import numpy as np

FaceLocation = Tuple[int, int, int, int]


class FrameProcessingController:
    """控制检测频率并负责检测画面的缩放与坐标还原。"""

    def __init__(
        self,
        detect_every_n_frames: int = 3,
        max_detection_fps: float = 4.0,
        detection_scale: float = 0.5,
        clock=time.monotonic,
    ):
        self._detect_every_n_frames = max(1, int(detect_every_n_frames))
        self._max_detection_fps = max(0.0, float(max_detection_fps))
        self._detection_scale = min(1.0, max(0.1, float(detection_scale)))
        self._clock = clock
        self._frame_count = 0
        self._last_detection_time = None

    def should_process(self) -> bool:
        self._frame_count += 1
        if (self._frame_count - 1) % self._detect_every_n_frames != 0:
            return False

        now = self._clock()
        if self._max_detection_fps > 0 and self._last_detection_time is not None:
            minimum_interval = 1.0 / self._max_detection_fps
            if now - self._last_detection_time < minimum_interval:
                return False

        self._last_detection_time = now
        return True

    def prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if self._detection_scale == 1.0:
            return frame
        return cv2.resize(
            frame,
            None,
            fx=self._detection_scale,
            fy=self._detection_scale,
            interpolation=cv2.INTER_AREA,
        )

    def restore_locations(
        self, locations: List[FaceLocation]
    ) -> List[FaceLocation]:
        if self._detection_scale == 1.0:
            return list(locations)
        factor = 1.0 / self._detection_scale
        return [
            tuple(int(round(value * factor)) for value in location)
            for location in locations
        ]


class StrangerConfirmation:
    """要求陌生人在时间窗口内连续多次出现后才确认。"""

    def __init__(
        self,
        window_seconds: float = 3,
        required_hits: int = 4,
        minimum_duration_seconds: float = 1,
        clock=time.monotonic,
    ):
        self._window_seconds = max(0.1, float(window_seconds))
        self._required_hits = max(1, int(required_hits))
        self._minimum_duration_seconds = max(
            0.0, float(minimum_duration_seconds)
        )
        self._clock = clock
        self._hits = {}

    def observe(self, stranger_id: str) -> bool:
        now = self._clock()
        hits = self._hits.setdefault(stranger_id, deque())
        hits.append(now)
        self._prune(hits, now)
        return (
            len(hits) >= self._required_hits
            and now - hits[0] >= self._minimum_duration_seconds
        )

    def cleanup(self) -> None:
        now = self._clock()
        for stranger_id in list(self._hits):
            hits = self._hits[stranger_id]
            self._prune(hits, now)
            if not hits:
                del self._hits[stranger_id]

    def reset(self, stranger_id: str) -> None:
        self._hits.pop(stranger_id, None)

    def _prune(self, hits: deque, now: float) -> None:
        cutoff = now - self._window_seconds
        while hits and hits[0] < cutoff:
            hits.popleft()
