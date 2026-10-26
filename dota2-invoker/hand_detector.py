"""
Pure OpenCV Hand and Naruto Seal Detector (Tanpa MediaPipe)
Praktikum Pengolahan Citra dan Visi Komputer (PCV)

Metodologi:
1. Konversi Model Warna BGR -> YCrCb & HSV untuk segmentasi warna kulit manusia.
2. Operasi Morfologi Citra (Opening & Closing) untuk eliminasi derau dan penutupan rongga kontur.
3. Ekstraksi Kontur Eksternal Terbesar (cv2.findContours).
4. Analisis Geometri Bentuk:
   - Aspect Ratio (h / w)
   - Solidity (Area Kontur / Area Convex Hull)
   - Convexity Defects (Kedalaman lekukan antar jari)
   - Distribusi Lebar Vertikal (Pointed Apex vs Flat/Wide Seal)
5. State Machine & Temporal Debouncing (Hold Time Meter) untuk akurasi pemicuan stabil.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, Dict, Any


class HandSealDetector:
    def __init__(self):
        # Default skin color bounds (YCrCb + HSV)
        self.ycrcb_lower = np.array([0, 133, 77], dtype=np.uint8)
        self.ycrcb_upper = np.array([255, 173, 127], dtype=np.uint8)

        self.hsv_lower = np.array([0, 25, 40], dtype=np.uint8)
        self.hsv_upper = np.array([30, 220, 255], dtype=np.uint8)

        # Morphological kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Temporal Debounce / Hold state
        self.candidate_seal: Optional[str] = None
        self.candidate_frames: int = 0
        self.required_hold_frames: int = 6  # ~0.2 detik pada 30 FPS
        self.last_trigger_time: float = 0.0
        self.cooldown_sec: float = 0.55

        # Custom calibration values
        self.is_calibrated: bool = False
        self.calibration_active: bool = False

    def calibrate_skin_sample(self, roi_bgr: np.ndarray):
        """Kalibrasi ambang batas warna kulit secara interaktif dari ROI sampel."""
        ycrcb_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # YCrCb mean & std
        mean_y, std_y = cv2.meanStdDev(ycrcb_roi)
        # Ambil Cr dan Cb dengan toleransi std
        cr_mean, cr_std = mean_y[1][0], std_y[1][0]
        cb_mean, cb_std = mean_y[2][0], std_y[2][0]

        cr_tol = max(20, int(cr_std * 2.8))
        cb_tol = max(20, int(cb_std * 2.8))

        self.ycrcb_lower = np.array([0, max(0, int(cr_mean - cr_tol)), max(0, int(cb_mean - cb_tol))], dtype=np.uint8)
        self.ycrcb_upper = np.array([255, min(255, int(cr_mean + cr_tol)), min(255, int(cb_mean + cb_tol))], dtype=np.uint8)

        # HSV mean & std
        mean_hsv, std_hsv = cv2.meanStdDev(hsv_roi)
        h_mean, h_std = mean_hsv[0][0], std_hsv[0][0]
        s_mean, s_std = mean_hsv[1][0], std_hsv[1][0]
        h_tol = max(15, int(h_std * 2.5))
        s_tol = max(35, int(s_std * 2.5))
        self.hsv_lower = np.array([max(0, int(h_mean - h_tol)), max(20, int(s_mean - s_tol)), 30], dtype=np.uint8)
        self.hsv_upper = np.array([min(180, int(h_mean + h_tol)), min(255, int(s_mean + s_tol)), 255], dtype=np.uint8)

        self.is_calibrated = True
        print(f"[PCV Calibration] YCrCb: Cr({self.ycrcb_lower[1]}-{self.ycrcb_upper[1]}), Cb({self.ycrcb_lower[2]}-{self.ycrcb_upper[2]}), HSV: H({self.hsv_lower[0]}-{self.hsv_upper[0]})")

    def get_skin_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Segmentasi warna kulit multi-model warna (YCrCb & HSV) + Morfologi Citra."""
        # Konversi Ruang Warna
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Kombinasi Masker (Intersection / Logika AND terbobot)
        combined_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        # Operasi Morfologi Citra:
        # 1. Opening: Erosi diikuti Dilasi untuk menghapus derau garam/titik latar belakang
        opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel_open, iterations=2)
        # 2. Closing: Dilasi diikuti Erosi untuk menutup rongga kecil di dalam telapak tangan
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)

        # Gaussian smoothing ringan untuk menghaluskan gerigi tepi kontur
        cleaned = cv2.GaussianBlur(closed, (5, 5), 0)
        _, final_mask = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
        return final_mask

    def classify_seal(self, contour: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[str, float, Dict[str, Any]]:
        """
        Klasifikasi Segel Tangan Naruto menggunakan Parameter Geometri Klasik:
        - Aspect Ratio (AR)
        - Solidity (S)
        - Convexity Defects (Jumlah & Kedalaman)
        - Distribusi Lebar Puncak Kontur
        """
        area = cv2.contourArea(contour)
        if area < 6500:
            return "NONE", 0.0, {}

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(h) / max(1, w)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / max(1.0, hull_area)

        # Momen Spasial untuk menghitung titik pusat massa (Centroid)
        m = cv2.moments(contour)
        cx = int(m["m10"] / max(1, m["m00"])) if m["m00"] > 0 else x + w // 2
        cy = int(m["m01"] / max(1, m["m00"])) if m["m00"] > 0 else y + h // 2

        # Analisis Convexity Defects
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defect_count = 0
        deep_defects = []
        if hull_indices is not None and len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    defects = defects.reshape(-1, 4)
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i]
                        depth = d / 256.0  # Konversi ke piksel
                        if depth > 18.0:
                            defect_count += 1
                            deep_defects.append(contour[f][0])
            except cv2.error:
                defect_count = 0

        # Analisis Lebar Puncak Kontur (Top 15% dari tinggi tangan)
        top_y_limit = y + int(h * 0.15)
        top_pts = contour[contour[:, :, 1] <= top_y_limit]
        if len(top_pts) > 0:
            top_pts_2d = top_pts.reshape(-1, 2)
            top_min_x = np.min(top_pts_2d[:, 0])
            top_max_x = np.max(top_pts_2d[:, 0])
            top_width = top_max_x - top_min_x
            top_ratio = top_width / max(1, w)
        else:
            top_ratio = 1.0

        features = {
            "area": area,
            "bbox": (x, y, w, h),
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "defect_count": defect_count,
            "top_ratio": top_ratio,
            "centroid": (cx, cy)
        }

        # --- Aturan Klasifikasi Segel Naruto ---
        detected_seal = "UNKNOWN"
        confidence = 0.5

        # 1. TIGER SEAL (Segel Harimau - Tora) -> EXORT (E)
        # Dua jari telunjuk tegak ke atas rapat, jari lain mengunci.
        # Ciri: Sangat vertikal (AR tinggi), puncak lancip (top_ratio kecil), solidity sedang-tinggi.
        if aspect_ratio >= 1.35 and top_ratio <= 0.42 and 0.65 <= solidity <= 0.88:
            detected_seal = "TIGER"
            confidence = min(0.95, 0.6 + (aspect_ratio - 1.35) * 0.2 + (0.45 - top_ratio))

        # 2. SNAKE SEAL (Segel Ular - Mi / Doa) -> QUAS (Q)
        # Kedua telapak tangan menempel rapat tegak ke atas seperti berdoa.
        # Ciri: Aspek vertikal lonjong, solidity sangat tinggi (minim rongga), sedikit lekukan.
        elif aspect_ratio >= 1.20 and solidity >= 0.84 and defect_count <= 2:
            detected_seal = "SNAKE"
            confidence = min(0.95, 0.7 + (solidity - 0.84) * 2.0)

        # 3. CLONE / CROSS SEAL (Segel Silang / Burung) -> WEX (W)
        # Dua jari telunjuk bersilangan atau bentuk salib 'X' / '+'.
        # Ciri: Proporsional horizontal-vertikal (AR ~ 0.8 - 1.3), solidity sedang, ada tonjolan horizontal.
        elif 0.75 <= aspect_ratio <= 1.30 and 0.54 <= solidity <= 0.78 and defect_count in [2, 3, 4]:
            detected_seal = "CROSS"
            confidence = 0.85

        # 4. RAM / FIST SEAL (Segel Kambing / Kepal Erat) -> INVOKE (R)
        # Kedua tangan mengepal erat terkunci di depan dada.
        # Ciri: Bentuk padat membulat, AR ~ 0.8 - 1.25, solidity tinggi, 0 atau 1 lekukan.
        elif 0.80 <= aspect_ratio <= 1.25 and solidity >= 0.83 and defect_count <= 1:
            detected_seal = "RAM"
            confidence = min(0.95, 0.75 + (solidity - 0.83) * 1.5)

        # 5. OPEN PALM (Telapak Tangan Terbuka) -> QUICK CAST D
        # 1 tangan terbuka menghadap kamera dengan jari mengembang.
        # Ciri: Banyak lekukan jari (defects >= 3), solidity rendah (< 0.72).
        elif defect_count >= 3 and solidity < 0.72 and aspect_ratio <= 1.4:
            detected_seal = "OPEN_PALM"
            confidence = 0.88

        # 6. DUAL PALM / BOAR (Dua Tangan Terbuka Menyamping / Segel Babi) -> QUICK CAST F
        # Dua telapak tangan terbuka lebar ke depan.
        # Ciri: Lebar melebihi tinggi (AR <= 0.75 atau w > h * 1.3).
        elif aspect_ratio <= 0.75 and area > 18000:
            detected_seal = "DUAL_PALM"
            confidence = 0.85

        return detected_seal, confidence, features

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], float, Dict[str, Any], np.ndarray]:
        """
        Memproses satu frame:
        - Segmentasi & pembersihan masker.
        - Deteksi kontur utama.
        - Klasifikasi segel dengan meteran penahanan (Debounce Hold Meter).
        Returns:
            triggered_action: str atau None (hanya memicu saat meter penuh).
            progress: float (0.0 s.d. 1.0 pengisian meteran).
            info: metadata visual.
            mask: citra biner masker tangan.
        """
        now = time.time()
        mask = self.get_skin_mask(frame_bgr)

        # Cari Kontur
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        triggered_action = None
        progress = 0.0
        info = {
            "seal": "NONE",
            "contour": None,
            "hull": None,
            "features": {},
            "cooldown": max(0.0, self.cooldown_sec - (now - self.last_trigger_time))
        }

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            seal_name, confidence, feats = self.classify_seal(largest_contour, frame_bgr.shape[:2])
            hull = cv2.convexHull(largest_contour)

            info["seal"] = seal_name
            info["contour"] = largest_contour
            info["hull"] = hull
            info["features"] = feats
            info["confidence"] = confidence

            # Logika Pemicuan & Debouncing
            if seal_name not in ["NONE", "UNKNOWN"]:
                if (now - self.last_trigger_time) >= self.cooldown_sec:
                    if self.candidate_seal == seal_name:
                        self.candidate_frames += 1
                    else:
                        self.candidate_seal = seal_name
                        self.candidate_frames = 1

                    progress = min(1.0, float(self.candidate_frames) / self.required_hold_frames)

                    # Jika pose ditahan stabil hingga mencapai ambang batas frame
                    if self.candidate_frames >= self.required_hold_frames:
                        triggered_action = seal_name
                        self.last_trigger_time = now
                        self.candidate_frames = 0
                        self.candidate_seal = None
                        progress = 0.0
                else:
                    progress = 0.0
            else:
                self.candidate_seal = None
                self.candidate_frames = max(0, self.candidate_frames - 1)
                progress = 0.0
        else:
            self.candidate_seal = None
            self.candidate_frames = 0
            progress = 0.0

        return triggered_action, progress, info, mask
