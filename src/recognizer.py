import logging
import os
import time
from typing import List

import face_recognition
import numpy as np

logger = logging.getLogger(__name__)


class FaceRecognizer:
    def __init__(self, known_faces_dir: str, tolerance: float = 0.5):
        self._tolerance = tolerance
        self._known_encodings: List[np.ndarray] = []
        self._known_names: List[str] = []

        if not os.path.isdir(known_faces_dir):
            logger.warning("熟人照片目录不存在: %s", known_faces_dir)
            return

        for filename in os.listdir(known_faces_dir):
            filepath = os.path.join(known_faces_dir, filename)
            if not os.path.isfile(filepath):
                continue

            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if not encodings:
                logger.warning("未在 %s 中检测到人脸，已跳过", filename)
                continue

            name = os.path.splitext(filename)[0]
            if len(encodings) > 1:
                logger.warning(
                    "在 %s 中检测到 %d 张人脸，取第一张",
                    filename, len(encodings)
                )

            self._known_encodings.append(encodings[0])
            self._known_names.append(name)
            logger.info("已注册熟人: %s", name)

        logger.info(
            "熟人库加载完成: %d 人 (tolerance=%.2f)",
            len(self._known_names), tolerance
        )

    def is_stranger(self, face_encoding: np.ndarray) -> bool:
        if not self._known_encodings:
            return True

        distances = face_recognition.face_distance(
            self._known_encodings, face_encoding
        )

        min_distance = np.min(distances)
        if min_distance <= self._tolerance:
            matched_index = np.argmin(distances)
            logger.debug(
                "匹配熟人 %s (距离=%.4f)", self._known_names[matched_index], min_distance
            )
            return False

        logger.info("检测到陌生人 (最短距离=%.4f > tolerance=%.2f)", min_distance, self._tolerance)
        return True

    @property
    def known_count(self) -> int:
        return len(self._known_names)


class StrangerTracker:
    """用人脸编码为陌生人分配临时 ID，供报警冷却区分不同人员。"""

    def __init__(
        self,
        tolerance: float = 0.5,
        retention_seconds: int = 3600,
        max_entries: int = 200,
    ):
        self._tolerance = tolerance
        self._retention_seconds = retention_seconds
        self._max_entries = max_entries
        self._entries = {}
        self._next_id = 1

    def identify(self, face_encoding: np.ndarray) -> str:
        now = time.time()
        self._remove_expired(now)

        if self._entries:
            stranger_ids = list(self._entries)
            encodings = [
                self._entries[stranger_id]["encoding"]
                for stranger_id in stranger_ids
            ]
            distances = face_recognition.face_distance(encodings, face_encoding)
            matched_index = int(np.argmin(distances))
            if distances[matched_index] <= self._tolerance:
                stranger_id = stranger_ids[matched_index]
                entry = self._entries[stranger_id]
                # 轻微融合最新编码，降低角度和光线变化造成的身份漂移。
                entry["encoding"] = (
                    entry["encoding"] * 0.8 + face_encoding * 0.2
                )
                entry["last_seen"] = now
                return stranger_id

        stranger_id = f"stranger-{self._next_id}"
        self._next_id += 1
        self._entries[stranger_id] = {
            "encoding": face_encoding.copy(),
            "last_seen": now,
        }
        self._trim_oldest()
        logger.info("发现新的陌生人临时身份: %s", stranger_id)
        return stranger_id

    def _remove_expired(self, now: float) -> None:
        expired = [
            stranger_id
            for stranger_id, entry in self._entries.items()
            if now - entry["last_seen"] > self._retention_seconds
        ]
        for stranger_id in expired:
            del self._entries[stranger_id]

    def _trim_oldest(self) -> None:
        while len(self._entries) > self._max_entries:
            oldest_id = min(
                self._entries,
                key=lambda stranger_id: self._entries[stranger_id]["last_seen"],
            )
            del self._entries[oldest_id]
