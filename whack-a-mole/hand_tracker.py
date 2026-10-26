"""Modul pelacakan centroid tangan berbasis segmentasi warna kulit HSV dan moments."""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

try:
    from .config import (
        CENTROID_SMOOTHING_FACTOR,
        LOWER_SKIN_1,
        LOWER_SKIN_2,
        MIN_HAND_CONTOUR_AREA,
        MORPH_KERNEL_SIZE,
        UPPER_SKIN_1,
        UPPER_SKIN_2,
    )
except ImportError:
    from config import (
        CENTROID_SMOOTHING_FACTOR,
        LOWER_SKIN_1,
        LOWER_SKIN_2,
        MIN_HAND_CONTOUR_AREA,
        MORPH_KERNEL_SIZE,
        UPPER_SKIN_1,
        UPPER_SKIN_2,
    )


@dataclass
class HandTrackingResult:
    """Struktur data hasil deteksi tangan pada satu frame."""

    centroid: Optional[Tuple[int, int]]
    contour: Optional[np.ndarray]
    area: float
    mask: np.ndarray
    is_detected: bool


class HandTracker:
    """Pelacak centroid tangan pemain menggunakan OpenCV murni.

    Mendukung isolasi Playing Section: seluruh area Non-Playing Section (tempat wajah/kepala)
    dinolkan dari masker sehingga wajah 100% mustahil terdeteksi sebagai kursor.
    """

    def __init__(
        self,
        smoothing: float = CENTROID_SMOOTHING_FACTOR,
        min_area: int = MIN_HAND_CONTOUR_AREA,
    ):
        self.smoothing = smoothing
        self.min_area = min_area
        self.prev_centroid: Optional[Tuple[int, int]] = None
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)

    def reset(self) -> None:
        """Mereset riwayat posisi centroid."""
        self.prev_centroid = None

    def process_frame(
        self,
        frame: np.ndarray,
        play_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> HandTrackingResult:
        """Memproses frame citra dan menghasilkan koordinat centroid tangan di dalam Playing Section."""
        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, LOWER_SKIN_1, UPPER_SKIN_1)
        mask2 = cv2.inRange(hsv, LOWER_SKIN_2, UPPER_SKIN_2)
        raw_mask = cv2.bitwise_or(mask1, mask2)

        # Jika play_rect aktif, nolkan seluruh area di luar Playing Section (Safe Zone)
        if play_rect is not None:
            px1, py1, px2, py2 = play_rect
            isolated_mask = np.zeros_like(raw_mask)
            isolated_mask[py1:py2, px1:px2] = raw_mask[py1:py2, px1:px2]
        else:
            isolated_mask = raw_mask

        # Pembersihan noise mask
        cleaned_mask = cv2.morphologyEx(isolated_mask, cv2.MORPH_OPEN, self._kernel_open, iterations=2)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, self._kernel_close, iterations=2)
        cleaned_mask = cv2.dilate(cleaned_mask, self._kernel_open, iterations=1)

        # Deteksi kontur eksternal di dalam area aktif
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = None
        max_area = 0.0
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area > max_area:
                max_area = area
                largest_contour = cnt

        if largest_contour is None or max_area < self.min_area:
            self.prev_centroid = None
            return HandTrackingResult(
                centroid=None,
                contour=None,
                area=max_area,
                mask=cleaned_mask,
                is_detected=False,
            )

        moments = cv2.moments(largest_contour)
        if moments["m00"] == 0:
            self.prev_centroid = None
            return HandTrackingResult(
                centroid=None,
                contour=largest_contour,
                area=max_area,
                mask=cleaned_mask,
                is_detected=False,
            )

        raw_cx = int(moments["m10"] / moments["m00"])
        raw_cy = int(moments["m01"] / moments["m00"])

        # Stabilisasi posisi kursor
        if self.prev_centroid is not None:
            smooth_cx = int(self.smoothing * raw_cx + (1.0 - self.smoothing) * self.prev_centroid[0])
            smooth_cy = int(self.smoothing * raw_cy + (1.0 - self.smoothing) * self.prev_centroid[1])
            final_centroid = (smooth_cx, smooth_cy)
        else:
            final_centroid = (raw_cx, raw_cy)

        self.prev_centroid = final_centroid

        return HandTrackingResult(
            centroid=final_centroid,
            contour=largest_contour,
            area=max_area,
            mask=cleaned_mask,
            is_detected=True,
        )
