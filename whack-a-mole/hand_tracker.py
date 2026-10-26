"""Modul pelacakan centroid tangan berbasis segmentasi multi-ruang warna (YCrCb + HSV).

Dilengkapi dengan heuristik Multi-Candidate Scoring Anti-Wajah dan kontinuitas temporal
agar kursor tangan tidak terkunci/terjebak pada kepala atau wajah pemain di tengah layar.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
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

    Menggunakan kombinasi segmentasi warna kulit YCrCb + HSV, filter morfologi,
    dan pembobotan kontur cerdas untuk membedakan tangan dari wajah pemain.
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

        # Rentang segmentasi warna kulit (default adaptif)
        self.ycrcb_lower = LOWER_SKIN_YCRCB.copy()
        self.ycrcb_upper = UPPER_SKIN_YCRCB.copy()
        self.hsv_lower1 = LOWER_SKIN_1.copy()
        self.hsv_upper1 = UPPER_SKIN_1.copy()
        self.hsv_lower2 = LOWER_SKIN_2.copy()
        self.hsv_upper2 = UPPER_SKIN_2.copy()

        self._morph_kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._morph_kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)

        self.is_calibrated: bool = False
        self.calibration_mode: bool = False

    def reset(self) -> None:
        """Mereset riwayat posisi centroid."""
        self.prev_centroid = None

    def calibrate_skin_sample(self, roi_bgr: np.ndarray) -> None:
        """Mengkalibrasi rentang warna kulit spesifik dari area telapak tangan."""
        if roi_bgr.size == 0:
            return

        ycrcb_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        mean_y, std_y = cv2.meanStdDev(ycrcb_roi)
        cr_mean, cr_std = mean_y[1][0], std_y[1][0]
        cb_mean, cb_std = mean_y[2][0], std_y[2][0]

        cr_tol = max(20, int(cr_std * 2.8))
        cb_tol = max(20, int(cb_std * 2.8))
        self.ycrcb_lower = np.array([0, max(0, int(cr_mean - cr_tol)), max(0, int(cb_mean - cb_tol))], dtype=np.uint8)
        self.ycrcb_upper = np.array([255, min(255, int(cr_mean + cr_tol)), min(255, int(cb_mean + cb_tol))], dtype=np.uint8)

        hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mean_hsv, std_hsv = cv2.meanStdDev(hsv_roi)
        h_mean, h_std = mean_hsv[0][0], std_hsv[0][0]
        s_mean, s_std = mean_hsv[1][0], std_hsv[1][0]
        h_tol = max(16, int(h_std * 2.6))
        s_tol = max(35, int(s_std * 2.6))

        self.hsv_lower1 = np.array([max(0, int(h_mean - h_tol)), max(25, int(s_mean - s_tol)), 40], dtype=np.uint8)
        self.hsv_upper1 = np.array([min(179, int(h_mean + h_tol)), min(255, int(s_mean + s_tol)), 255], dtype=np.uint8)

        self.is_calibrated = True
        self.calibration_mode = False

    def get_skin_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Menghasilkan masker biner warna kulit melalui perpaduan ruang warna YCrCb dan HSV."""
        blurred = cv2.GaussianBlur(frame_bgr, (7, 7), 0)

        # 1. Segmentasi pada ruang warna YCrCb (invarian intensitas cahaya)
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)

        # 2. Segmentasi pada ruang warna HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_hsv1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        mask_hsv2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

        # 3. Penggabungan logika AND untuk memotong noise latar belakang
        combined_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        # 4. Operasi Morfologi Citra (Opening membuang noise, Closing menutup lubang telapak)
        opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self._morph_kernel_open, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._morph_kernel_close, iterations=2)
        cleaned_mask = cv2.dilate(closed, self._morph_kernel_open, iterations=1)

        return cleaned_mask

    def _score_candidate_contour(
        self,
        contour: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, int, int]:
        """Menilai kelayakan kontur sebagai tangan pemain (Anti-Wajah & Kontinuitas Temporal)."""
        area = float(cv2.contourArea(contour))
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return -9999.0, 0, 0

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        score = 0.0

        # Kontribusi luas (dibatasi agar wajah besar tidak mendominasi otomatis)
        score += min(area, 25000.0) * 0.04

        # Analisis Geometri Bentuk (Solidity & Cacat Kecembungan / Defects)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(1.0, hull_area)

        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defect_count = 0
        if hull_indices is not None and len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    # Hitung celah lembah nyata antar jari/ibu jari
                    for d in range(defects.shape[0]):
                        depth = defects[d, 0, 3] / 256.0
                        if depth > 12.0:
                            defect_count += 1
            except Exception:
                pass

        # Filter Heuristik Anti-Wajah:
        # Wajah biasanya berada di sepertiga atas layar tengah (y < 0.38 H dan |x - 0.5W| < 0.20W)
        # dengan bentuk oval kompak (solidity tinggi > 0.82 dan tanpa celah jari)
        is_face_position = (cy < frame_height * 0.38) and (abs(cx - frame_width * 0.5) < frame_width * 0.20)
        if is_face_position and solidity > 0.82 and defect_count == 0:
            score -= 4500.0  # Penalti besar untuk zona kepala/wajah
        else:
            score += 2000.0  # Bonus untuk area bermain aktif

        # Bonus posisi vertikal: tangan umumnya masuk dari arah bawah/samping layar
        score += (cy / float(frame_height)) * 1400.0

        # Bonus bentuk tangan (keberadaan jari / celah antar jari)
        if defect_count > 0:
            score += min(defect_count, 6) * 350.0

        # Kontinuitas Temporal (Hand Tracking Lock):
        # Berikan bonus sangat tinggi pada kontur yang dekat dengan posisi tangan frame sebelumnya
        if self.prev_centroid is not None:
            dist_prev = math.hypot(cx - self.prev_centroid[0], cy - self.prev_centroid[1])
            if dist_prev < 200.0:
                score += (200.0 - dist_prev) * 20.0  # Mengunci pergerakan tangan saat di-drag

        return score, cx, cy

    def process_frame(self, frame: np.ndarray) -> HandTrackingResult:
        """Memproses frame citra BGR dan mengekstrak centroid tangan paling optimal."""
        h, w = frame.shape[:2]
        cleaned_mask = self.get_skin_mask(frame)

        # Cari kontur eksternal
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area or area > self.max_area:
                continue

            score, cx, cy = self._score_candidate_contour(cnt, w, h)
            if score > 0.0:
                candidates.append((score, cnt, area, cx, cy))

        # Jika tidak ada kandidat lolos
        if not candidates:
            # Toleransi jeda singkat sebelum melepas kursor
            self.prev_centroid = None
            return HandTrackingResult(
                centroid=None,
                contour=None,
                area=0.0,
                mask=cleaned_mask,
                is_detected=False,
            )

        # Pilih kandidat dengan skor kecocokan tangan tertinggi
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_contour, best_area, raw_cx, raw_cy = candidates[0]

        # Stabilisasi posisi kursor menggunakan Exponential Moving Average (EMA)
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
            area=best_area,
            mask=cleaned_mask,
            is_detected=True,
        )
