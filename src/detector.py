import logging
from typing import List, Tuple

import cv2
import face_recognition
import numpy as np

logger = logging.getLogger(__name__)

FaceLocation = Tuple[int, int, int, int]
FaceEncoding = np.ndarray


class FaceDetector:
    def __init__(self, model: str = "hog", upsample: int = 1):
        self._model = model
        self._upsample = upsample
        logger.info("人脸检测器已初始化 (model=%s, upsample=%d)", model, upsample)

    def detect(self, frame: np.ndarray) -> List[FaceLocation]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(
            rgb,
            number_of_times_to_upsample=self._upsample,
            model=self._model,
        )
        return locations

    def encode(
        self, frame: np.ndarray, face_locations: List[FaceLocation]
    ) -> List[FaceEncoding]:
        if not face_locations:
            return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(
            rgb,
            known_face_locations=face_locations,
        )
        return encodings
