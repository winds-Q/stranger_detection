import os
from dataclasses import dataclass

import cv2
import face_recognition
import numpy as np

from recognizer import SUPPORTED_IMAGE_EXTENSIONS


@dataclass(frozen=True)
class FaceValidationResult:
    ok: bool
    message: str


class FaceImageValidator:
    """Validate a known-face image before it enters the recognition library."""

    def __init__(self, config, known_faces_dir: str):
        rules = config.get("face_upload_validation", {})
        self._known_faces_dir = known_faces_dir
        self._min_face_size = int(rules.get("min_face_size", 80))
        self._min_blur_score = float(rules.get("min_blur_score", 60))
        self._min_brightness = float(rules.get("min_brightness", 35))
        self._max_brightness = float(rules.get("max_brightness", 225))
        self._duplicate_image_difference = float(
            rules.get("duplicate_image_difference", 2.0)
        )

    def validate(self, image: np.ndarray, target_filename: str = "") -> FaceValidationResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        if brightness < self._min_brightness:
            return FaceValidationResult(False, "照片过暗，请在光线充足处重新拍摄")
        if brightness > self._max_brightness:
            return FaceValidationResult(False, "照片过亮，请避免强光或过度曝光")
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < self._min_blur_score:
            return FaceValidationResult(False, "照片较模糊，请上传清晰、对焦准确的正脸照片")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        if not locations:
            return FaceValidationResult(False, "照片中未检测到人脸")
        if len(locations) != 1:
            return FaceValidationResult(False, "熟人照片必须只包含一张人脸")
        top, right, bottom, left = locations[0]
        if min(right - left, bottom - top) < self._min_face_size:
            return FaceValidationResult(False, "人脸区域过小，请上传距离更近的照片")

        encodings = face_recognition.face_encodings(rgb, locations)
        if not encodings:
            return FaceValidationResult(False, "无法提取人脸特征，请更换正脸照片")
        duplicate = self._find_duplicate(image, target_filename)
        if duplicate:
            return FaceValidationResult(False, f"该图片与已有样本 {duplicate} 重复")
        return FaceValidationResult(True, "照片质量检查通过")

    def _find_duplicate(self, image: np.ndarray, target_filename: str) -> str | None:
        """Detect the same image, not merely another photo of the same person."""
        if not os.path.isdir(self._known_faces_dir):
            return None
        candidate = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        candidate = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for filename in os.listdir(self._known_faces_dir):
            if filename == target_filename:
                continue
            path = os.path.join(self._known_faces_dir, filename)
            if not os.path.isfile(path) or os.path.splitext(filename)[1].lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            try:
                known_image = cv2.imread(path, cv2.IMREAD_COLOR)
                if known_image is None:
                    continue
                known_image = cv2.resize(known_image, (64, 64), interpolation=cv2.INTER_AREA)
                known_image = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            except (OSError, ValueError, cv2.error):
                continue
            difference = float(np.mean(np.abs(candidate - known_image)))
            if difference <= self._duplicate_image_difference:
                return filename
        return None
