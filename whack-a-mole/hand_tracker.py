"""Modul pelacakan centroid tangan berbasis segmentasi multi-ruang warna (YCrCb + HSV).

Mendukung isolasi Playing Section (zeroing non-playing safe zone) dan Kalibrasi
Warna Kulit Interaktif untuk akurasi optimal tanpa gangguan wajah pemain.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

try:
    from .config import (
        CENTROID_SMOOTHING_FACTOR,
        LOWER_SKIN_1,
        LOWER_SKIN_2,
        LOWER_SKIN_YCRCB,
        MAX_HAND_CONTOUR_AREA,
        MIN_HAND_CONTOUR_AREA,
        MORPH_KERNEL_SIZE,
        UPPER_SKIN_1,
        UPPER_SKIN_2,
        UPPER_SKIN_YCRCB,
    )
except ImportError:
    from config import (
        CENTROID_SMOOTHING_FACTOR,
        LOWER_SKIN_1,
        LOWER_SKIN_2,
        LOWER_SKIN_YCRCB,
        MAX_HAND_CONTOUR_AREA,
        MIN_HAND_CONTOUR_AREA,
        MORPH_KERNEL_SIZE,
        UPPER_SKIN_1,
        UPPER_SKIN_2,
        UPPER_SKIN_YCRCB,
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

    Mendukung isolasi Playing Section: seluruh area Non-Playing Section (tempat kepala/tubuh)
    dinolkan dari masker biner sehingga wajah 100% mustahil membajak kursor.
    """

    def __init__(
        self,
        smoothing: float = CENTROID_SMOOTHING_FACTOR,
        min_area: int = MIN_HAND_CONTOUR_AREA,
        max_area: int = MAX_HAND_CONTOUR_AREA,
    ):
        self.smoothing = smoothing
        self.min_area = min_area
        self.max_area = max_area
        self.prev_centroid: Optional[Tuple[int, int]] = None

        # Rentang segmentasi warna kulit default
        self.ycrcb_lower = LOWER_SKIN_YCRCB.copy()
        self.ycrcb_upper = UPPER_SKIN_YCRCB.copy()
        self.hsv_lower1 = LOWER_SKIN_1.copy()
        self.hsv_upper1 = UPPER_SKIN_1.copy()
        self.hsv_lower2 = LOWER_SKIN_2.copy()
        self.hsv_upper2 = UPPER_SKIN_2.copy()

        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)

        self.is_calibrated: bool = False
        self.calib_info: Optional[Dict] = None

    def reset(self) -> None:
        """Mereset riwayat posisi centroid."""
        self.prev_centroid = None

    def calibrate_skin_sample(self, roi_bgr: np.ndarray) -> Dict:
        """Mengkalibrasi rentang warna kulit spesifik dari area telapak tangan."""
        if roi_bgr.size == 0:
            return {}

        # 1. Sampel statistik YCrCb
        ycrcb_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        mean_y, std_y = cv2.meanStdDev(ycrcb_roi)
        cr_mean, cr_std = mean_y[1][0], std_y[1][0]
        cb_mean, cb_std = mean_y[2][0], std_y[2][0]

        cr_tol = max(20, int(cr_std * 2.8))
        cb_tol = max(20, int(cb_std * 2.8))
        self.ycrcb_lower = np.array([0, max(0, int(cr_mean - cr_tol)), max(0, int(cb_mean - cb_tol))], dtype=np.uint8)
        self.ycrcb_upper = np.array([255, min(255, int(cr_mean + cr_tol)), min(255, int(cb_mean + cb_tol))], dtype=np.uint8)

        # 2. Sampel statistik HSV
        hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mean_hsv, std_hsv = cv2.meanStdDev(hsv_roi)
        h_mean, h_std = mean_hsv[0][0], std_hsv[0][0]
        s_mean, s_std = mean_hsv[1][0], std_hsv[1][0]
        h_tol = max(16, int(h_std * 2.6))
        s_tol = max(35, int(s_std * 2.6))

        self.hsv_lower1 = np.array([max(0, int(h_mean - h_tol)), max(25, int(s_mean - s_tol)), 40], dtype=np.uint8)
        self.hsv_upper1 = np.array([min(179, int(h_mean + h_tol)), min(255, int(s_mean + s_tol)), 255], dtype=np.uint8)

        self.is_calibrated = True
        self.calib_info = {
            "cr_range": [int(self.ycrcb_lower[1]), int(self.ycrcb_upper[1])],
            "cb_range": [int(self.ycrcb_lower[2]), int(self.ycrcb_upper[2])],
            "h_range": [int(self.hsv_lower1[0]), int(self.hsv_upper1[0])],
            "s_range": [int(self.hsv_lower1[1]), int(self.hsv_upper1[1])],
        }
        return self.calib_info

    def load_saved_calibration(self, calib_data: Optional[Dict]) -> None:
        """Memuat rentang kalibrasi tersimpan."""
        if not calib_data:
            return
        try:
            cr = calib_data.get("cr_range", [133, 175])
            cb = calib_data.get("cb_range", [77, 127])
            h = calib_data.get("h_range", [0, 25])
            s = calib_data.get("s_range", [25, 210])

            self.ycrcb_lower = np.array([0, cr[0], cb[0]], dtype=np.uint8)
            self.ycrcb_upper = np.array([255, cr[1], cb[1]], dtype=np.uint8)
            self.hsv_lower1 = np.array([h[0], s[0], 40], dtype=np.uint8)
            self.hsv_upper1 = np.array([h[1], s[1], 255], dtype=np.uint8)
            self.is_calibrated = True
            self.calib_info = calib_data
        except Exception:
            pass

    def get_skin_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Menghasilkan masker warna kulit gabungan YCrCb dan HSV."""
        blurred = cv2.GaussianBlur(frame_bgr, (7, 7), 0)

        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_hsv1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        mask_hsv2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

        # Gabungkan YCrCb dan HSV
        combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        return combined

    def process_frame(
        self,
        frame: np.ndarray,
        play_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> HandTrackingResult:
        """Memproses frame citra dan menghasilkan koordinat centroid tangan di dalam Playing Section."""
        raw_mask = self.get_skin_mask(frame)

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

        candidates = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area or area > self.max_area:
                continue

            moments = cv2.moments(cnt)
            if moments["m00"] == 0:
                continue

            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            # Skor kontur: prioritas luas dan kontinuitas temporal kursor
            score = min(area, 30000.0) * 0.1
            if self.prev_centroid is not None:
                dist = math.hypot(cx - self.prev_centroid[0], cy - self.prev_centroid[1])
                if dist < 250:
                    score += (250 - dist) * 20.0

            candidates.append((score, cnt, (cx, cy), area))

        if not candidates:
            self.prev_centroid = None
            return HandTrackingResult(
                centroid=None,
                contour=None,
                area=0.0,
                mask=cleaned_mask,
                is_detected=False,
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_contour, (raw_cx, raw_cy), max_area = candidates[0]

        # Exponential moving average untuk stabilisasi posisi kursor
        if self.prev_centroid is not None:
            smooth_cx = int(self.smoothing * raw_cx + (1.0 - self.smoothing) * self.prev_centroid[0])
            smooth_cy = int(self.smoothing * raw_cy + (1.0 - self.smoothing) * self.prev_centroid[1])
            final_centroid = (smooth_cx, smooth_cy)
        else:
            final_centroid = (raw_cx, raw_cy)

        self.prev_centroid = final_centroid

        return HandTrackingResult(
            centroid=final_centroid,
            contour=best_contour,
            area=max_area,
            mask=cleaned_mask,
            is_detected=True,
        )
