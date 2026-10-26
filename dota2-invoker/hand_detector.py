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
from collections import deque, Counter
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

        # Temporal Debounce & Majority Voting Buffer
        self.rolling_history: deque = deque(maxlen=5)
        self.candidate_seal: Optional[str] = None
        self.candidate_frames: int = 0
        self.required_hold_frames: int = 4  # ~130ms (sangat responsif & kebal 1-frame jitter)
        self.last_trigger_time: float = 0.0
        self.cooldown_sec: float = 0.35

        # Custom calibration values
        self.is_calibrated: bool = False
        self.calibration_active: bool = False
        self.calib_info: Dict[str, Any] = {}

    def reset_default_calibration(self):
        """Mengembalikan ambang batas warna kulit ke standar universal."""
        self.ycrcb_lower = np.array([0, 133, 77], dtype=np.uint8)
        self.ycrcb_upper = np.array([255, 173, 127], dtype=np.uint8)
        self.hsv_lower = np.array([0, 25, 40], dtype=np.uint8)
        self.hsv_upper = np.array([30, 220, 255], dtype=np.uint8)
        self.is_calibrated = False
        self.calib_info = {}
        print("[PCV Calibration] Ambang batas warna kulit dikembalikan ke default universal.")

    def calibrate_skin_sample(self, roi_bgr: np.ndarray) -> Dict[str, Any]:
        """Kalibrasi ambang batas warna kulit secara interaktif dari ROI sampel."""
        ycrcb_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # YCrCb mean & std
        mean_y, std_y = cv2.meanStdDev(ycrcb_roi)
        cr_mean, cr_std = mean_y[1][0], std_y[1][0]
        cb_mean, cb_std = mean_y[2][0], std_y[2][0]

        cr_tol = max(22, int(cr_std * 3.0))
        cb_tol = max(22, int(cb_std * 3.0))

        self.ycrcb_lower = np.array([0, max(0, int(cr_mean - cr_tol)), max(0, int(cb_mean - cb_tol))], dtype=np.uint8)
        self.ycrcb_upper = np.array([255, min(255, int(cr_mean + cr_tol)), min(255, int(cb_mean + cb_tol))], dtype=np.uint8)

        # HSV mean & std
        mean_hsv, std_hsv = cv2.meanStdDev(hsv_roi)
        h_mean, h_std = mean_hsv[0][0], std_hsv[0][0]
        s_mean, s_std = mean_hsv[1][0], std_hsv[1][0]
        h_tol = max(18, int(h_std * 2.8))
        s_tol = max(40, int(s_std * 2.8))
        self.hsv_lower = np.array([max(0, int(h_mean - h_tol)), max(20, int(s_mean - s_tol)), 30], dtype=np.uint8)
        self.hsv_upper = np.array([min(180, int(h_mean + h_tol)), min(255, int(s_mean + s_tol)), 255], dtype=np.uint8)

        self.is_calibrated = True
        self.calib_info = {
            "cr_range": (int(self.ycrcb_lower[1]), int(self.ycrcb_upper[1])),
            "cb_range": (int(self.ycrcb_lower[2]), int(self.ycrcb_upper[2])),
            "h_range": (int(self.hsv_lower[0]), int(self.hsv_upper[0])),
            "mean_bgr": [int(x) for x in cv2.mean(roi_bgr)[:3]]
        }
        print(f"[PCV Calibration] Berhasil: Cr{self.calib_info['cr_range']}, Cb{self.calib_info['cb_range']}, H{self.calib_info['h_range']}")
        return self.calib_info

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

    def classify_seal(self, contour: np.ndarray, mask: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[str, float, Dict[str, Any]]:
        """
        Klasifikasi Gestur Tangan (1 s.d. 5 Jari dan Kepal) menggunakan:
        1. Distance Transform untuk menentukan pusat telapak tangan (Palm Center & Radius)
        2. Convex Hull Fingertip Distance Metric (Titik puncak di luar lingkaran telapak)
        3. Convexity Defects dengan Filter Sudut Lancip (Aturan Cosinus)
        """
        area = cv2.contourArea(contour)
        if area < 6000:
            return "NONE", 0.0, {}

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(h) / max(1, w)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / max(1.0, hull_area)

        # 1. Distance Transform untuk menemukan Pusat Telapak Tangan (Palm Center)
        # Gambar kontur tangan saja sebagai masker biner solid tanpa noise/lubang di dalam
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1, offset=(-x, -y))
        dist_transform = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        palm_center = (x + max_loc[0], y + max_loc[1])
        palm_radius = float(max_val)

        # 2. Deteksi Ujung Jari Nyata (Fingertips) Berdasarkan Polar Angle & Jarak Rasio
        candidates = []
        if palm_radius > 15:
            for pt in hull[:, 0]:
                dx = float(pt[0] - palm_center[0])
                dy = float(pt[1] - palm_center[1])
                dist = np.hypot(dx, dy)
                ratio = dist / palm_radius
                is_wrist = (dy > (palm_radius * 0.20)) and (abs(dx) < (palm_radius * 1.35))
                # Ambang batas rasio 1.90: mencakup ujung jari yang sedikit miring/foreshortened ke arah kamera
                if not is_wrist and ratio > 1.90:
                    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
                    candidates.append((pt, dist, angle_deg))

        # Urutkan kandidat dari jarak terjauh ke terdekat (titik puncak jari diproses duluan)
        candidates.sort(key=lambda item: item[1], reverse=True)

        # Polar-Aware NMS:
        # Titik dianggap duplikat dari jari yang sama jika sudut polar sangat dekat (< 15 derajat)
        # ATAU jarak Euclidean sangat rapat (< 32 px).
        # Ini mencegah dua jari berdekatan (seperti telunjuk & tengah) saling menghapus,
        # sekaligus menghapus titik ganda pada puncak jari yang sama.
        accepted_tips = []
        for pt, dist, angle in candidates:
            duplicate = False
            for p_acc, _, a_acc in accepted_tips:
                angle_diff = abs(angle - a_acc)
                if angle_diff > 180.0:
                    angle_diff = 360.0 - angle_diff
                euc = np.hypot(pt[0] - p_acc[0], pt[1] - p_acc[1])
                if angle_diff < 15.0 or euc < 32.0:
                    duplicate = True
                    break
            if not duplicate:
                accepted_tips.append((pt, dist, angle))

        finger_tips = [item[0] for item in accepted_tips]
        tip_count = len(finger_tips)

        # 3. Analisis Celah Antar Jari (Convexity Defects) dengan Filter Sudut Cosinus
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defect_count = 0
        deep_defects = []
        if hull_indices is not None and len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    defects = defects.reshape(-1, 4)
                    for i in range(defects.shape[0]):
                        s_idx, e_idx, f_idx, d = defects[i]
                        start = contour[s_idx][0]
                        end = contour[e_idx][0]
                        far = contour[f_idx][0]
                        depth = d / 256.0

                        a = float(np.linalg.norm(end - start))
                        b = float(np.linalg.norm(far - start))
                        c = float(np.linalg.norm(end - far))

                        if b * c > 0:
                            cos_val = np.clip((b**2 + c**2 - a**2) / (2.0 * b * c), -1.0, 1.0)
                            angle = float(np.degrees(np.arccos(cos_val)))

                            # Celah antar jari valid jika sudut lancip (<= 88 derajat) dan cukup dalam
                            if angle <= 88.0 and depth > 18.0 and far[1] < (y + int(h * 0.82)):
                                defect_count += 1
                                deep_defects.append(far)
            except cv2.error:
                defect_count = 0

        # Rasio Puncak 15% Teratas
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
            "palm_center": palm_center,
            "palm_radius": palm_radius,
            "finger_tips": finger_tips,
            "tip_count": tip_count,
            "defect_count": defect_count,
            "deep_defects": deep_defects,
            "top_ratio": top_ratio,
            "centroid": palm_center  # Gunakan palm center sebagai titik pusat stabil
        }

        # --- Klasifikasi Presisi & Responsif (Direct Tip Count Mapping) ---
        # Filter Anti-Wajah: Wajah bulat pejal tanpa celah lekukan jari
        if tip_count >= 2 and defect_count == 0 and solidity >= 0.85:
            return "NONE", 0.0, {}

        confidence = 0.95
        if tip_count >= 5 or (tip_count == 4 and aspect_ratio <= 1.34 and defect_count >= 3):
            detected_seal = "5_FINGERS"   # 5 Jari (Telapak Terbuka) -> Quick Cast Slot 1 (D)
        elif tip_count == 4:
            detected_seal = "4_FINGERS"   # 4 Jari -> Invoke (R)
        elif tip_count == 3:
            detected_seal = "3_FINGERS"   # 3 Jari -> Exort (E)
        elif tip_count == 2:
            detected_seal = "2_FINGERS"   # 2 Jari -> Wex (W)
        elif tip_count == 1:
            detected_seal = "1_FINGER"    # 1 Jari -> Quas (Q)
        elif tip_count == 0 and area >= 5000:
            detected_seal = "FIST"        # 0 Jari (Kepal) -> Quick Cast Slot 2 (F)
        else:
            detected_seal = "NONE"

        return detected_seal, confidence, features

    def process_frame(self, frame_bgr: np.ndarray, roi_box: Optional[Tuple[int, int, int, int]] = None) -> Tuple[Optional[str], float, Dict[str, Any], np.ndarray]:
        """
        Memproses satu frame:
        - Segmentasi & pembersihan masker.
        - Isolasi area interaksi (Hand Zone) jika roi_box diberikan.
        - Deteksi kontur utama & klasifikasi gestur.
        - Logika penahanan stabil (Debounce Hold Meter).
        """
        now = time.time()
        mask = self.get_skin_mask(frame_bgr)

        # Jika Hand Zone aktif, eliminasi 100% gangguan gorden/rambut/wajah di luar kotak
        if roi_box is not None:
            rx1, ry1, rx2, ry2 = roi_box
            zone_mask = np.zeros_like(mask)
            zone_mask[ry1:ry2, rx1:rx2] = 255
            mask = cv2.bitwise_and(mask, zone_mask)

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
            candidates = []
            for cnt in contours:
                cnt_area = cv2.contourArea(cnt)
                if cnt_area < 5000 or cnt_area > 75000:
                    continue

                seal_name, confidence, feats = self.classify_seal(cnt, mask, frame_bgr.shape[:2])
                if seal_name != "NONE":
                    tip_cnt = feats.get("tip_count", 0)
                    defect_cnt = feats.get("defect_count", 0)
                    # Skor: Prioritas utama kontur tangan yang memiliki jari aktif (> 0)
                    score = (tip_cnt * 60000) + (defect_cnt * 5000) + cnt_area
                    candidates.append((score, cnt, seal_name, confidence, feats))

            if candidates:
                # Ambil kandidat dengan skor tertinggi (tangan asli selalu menang atas wajah)
                candidates.sort(key=lambda x: x[0], reverse=True)
                _, best_cnt, seal_name, confidence, feats = candidates[0]

                hull = cv2.convexHull(best_cnt)
                info["seal"] = seal_name
                info["contour"] = best_cnt
                info["hull"] = hull
                info["features"] = feats
                info["confidence"] = confidence
            else:
                seal_name = "NONE"

            # --- Temporal Filtering (Majority Voting atas 5 Frame Terakhir) ---
            if seal_name not in ["NONE", "UNKNOWN"]:
                self.rolling_history.append(seal_name)
            else:
                self.rolling_history.append("NONE")

            valid_history = [s for s in self.rolling_history if s not in ["NONE", "UNKNOWN"]]
            if valid_history:
                pose_counts = Counter(valid_history)
                smoothed_seal, _ = pose_counts.most_common(1)[0]
            else:
                smoothed_seal = "NONE"

            # Tampilkan gestur yang telah distabilkan pada antarmuka
            if smoothed_seal != "NONE":
                info["seal"] = smoothed_seal

            # --- Logika Pemicuan dengan Akumulator Leaky Bucket ---
            # Mengizinkan micro-jitter 1 frame tanpa membatalkan seluruh progres hold meter
            if smoothed_seal not in ["NONE", "UNKNOWN"]:
                if (now - self.last_trigger_time) >= self.cooldown_sec:
                    if self.candidate_seal == smoothed_seal:
                        self.candidate_frames += 1
                    else:
                        # Pengurangan gradual (decay), bukan wipeout instan
                        self.candidate_frames = max(0, self.candidate_frames - 1)
                        if self.candidate_frames == 0:
                            self.candidate_seal = smoothed_seal
                            self.candidate_frames = 1

                    progress = min(1.0, float(self.candidate_frames) / self.required_hold_frames)

                    # Jika pose ditahan stabil hingga mencapai ambang batas frame
                    if self.candidate_frames >= self.required_hold_frames:
                        triggered_action = self.candidate_seal
                        self.last_trigger_time = now
                        self.candidate_frames = 0
                        self.candidate_seal = None
                        self.rolling_history.clear()
                        progress = 0.0
                else:
                    progress = 0.0
            else:
                self.candidate_frames = max(0, self.candidate_frames - 1)
                if self.candidate_frames == 0:
                    self.candidate_seal = None
                progress = 0.0
        else:
            self.rolling_history.append("NONE")
            self.candidate_frames = max(0, self.candidate_frames - 1)
            if self.candidate_frames == 0:
                self.candidate_seal = None
            progress = 0.0

        return triggered_action, progress, info, mask
