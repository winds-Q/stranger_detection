from typing import Iterable, Tuple

import cv2
import numpy as np

FaceLocation = Tuple[int, int, int, int]
Annotation = Tuple[FaceLocation, str, bool]


def annotate_frame(
    frame: np.ndarray,
    annotations: Iterable[Annotation],
) -> np.ndarray:
    """在画面副本中标记熟人（绿）和陌生人（红）。"""
    output = frame.copy()
    height, width = output.shape[:2]

    for (top, right, bottom, left), label, is_stranger in annotations:
        top = max(0, min(height - 1, top))
        bottom = max(0, min(height - 1, bottom))
        left = max(0, min(width - 1, left))
        right = max(0, min(width - 1, right))
        color = (0, 0, 255) if is_stranger else (0, 180, 0)

        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        text = label or ("stranger" if is_stranger else "known")
        text_size, _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(0, top - text_size[1] - 8)
        cv2.rectangle(
            output,
            (left, label_top),
            (min(width - 1, left + text_size[0] + 8), top),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            output,
            text,
            (left + 4, max(text_size[1], top - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output
