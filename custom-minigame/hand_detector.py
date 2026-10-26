"""
Pure OpenCV Hand and Naruto Seal Detector for Custom Mini-Game
Praktikum Pengolahan Citra dan Visi Komputer (PCV)
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, Dict, Any


class HandSealDetector:
    def __init__(self):
        # Rentang Warna Kulit YCrCb + HSV
        self.ycrcb_lower = np.array([0, 133, 77], dtype=np.uint8)
        self.ycrcb_upper = np.array([255, 173, 127], dtype=np.uint8)

        self.hsv_lower = np.array([0, 25, 40], dtype=np.uint8)
        self.hsv_upper = np.array([30, 220, 255], dtype=np.uint8)

        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        self.candidate_seal: Optional[str] = None
        self.candidate_frames: int = 0
        self.required_hold_frames: int = 5  # Cepat & responsif untuk pertempuran game (~0.16 detik)
        self.last_trigger_time: float = 0.0
        self.cooldown_sec: float = 0.40

    def calibrate_skin_sample(self, roi_bgr: np.ndarray):
        """Kalibrasi interaktif warna kulit di ruangan pencahayaan saat demo."""
        ycrcb_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        mean_y, std_y = cv2.meanStdDev(ycrcb_roi)
        cr_mean, cr_std = mean_y[1][0], std_y[1][0]
        cb_mean, cb_std = mean_y[2][0], std_y[2][0]

        cr_tol = max(20, int(cr_std * 2.6))
        cb_tol = max(20, int(cb_std * 2.6))

        self.ycrcb_lower = np.array([0, max(0, int(cr_mean - cr_tol)), max(0, int(cb_mean - cb_tol))], dtype=np.uint8)
        self.ycrcb_upper = np.array([255, min(255, int(cr_mean + cr_tol)), min(255, int(cb_mean + cb_tol))], dtype=np.uint8)

    def get_skin_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        combined_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel_open, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)

        cleaned = cv2.GaussianBlur(closed, (5, 5), 0)
        _, final_mask = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
        return final_mask

    def classify_seal(self, contour: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        area = cv2.contourArea(contour)
        if area < 6000:
            return "NONE", 0.0, {}

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(h) / max(1, w)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / max(1.0, hull_area)

        # Momen Spasial Centroid
        m = cv2.moments(contour)
        cx = int(m["m10"] / max(1, m["m00"])) if m["m00"] > 0 else x + w // 2
        cy = int(m["m01"] / max(1, m["m00"])) if m["m00"] > 0 else y + h // 2

        # Analisis Cacat Kecembungan (Convexity Defects)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defect_count = 0
        if hull_indices is not None and len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    defects = defects.reshape(-1, 4)
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i]
                        if (d / 256.0) > 18.0:
                            defect_count += 1
            except cv2.error:
                defect_count = 0

        # Rasio Puncak 15% Teratas
        top_y_limit = y + int(h * 0.15)
        top_pts = contour[contour[:, :, 1] <= top_y_limit]
        if len(top_pts) > 0:
            top_pts_2d = top_pts.reshape(-1, 2)
            top_min_x = np.min(top_pts_2d[:, 0])
            top_max_x = np.max(top_pts_2d[:, 0])
            top_ratio = (top_max_x - top_min_x) / max(1, w)
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

        detected_seal = "UNKNOWN"
        confidence = 0.5

        # 1. TIGER SEAL -> Katon Fireball
        if aspect_ratio >= 1.35 and top_ratio <= 0.42 and 0.65 <= solidity <= 0.88:
            detected_seal = "TIGER"
            confidence = 0.92

        # 2. SNAKE SEAL -> Hyoton Ice Barrier
        elif aspect_ratio >= 1.20 and solidity >= 0.84 and defect_count <= 2:
            detected_seal = "SNAKE"
            confidence = 0.94

        # 3. CROSS SEAL -> Raiton Lightning Strike
        elif 0.75 <= aspect_ratio <= 1.30 and 0.54 <= solidity <= 0.78 and defect_count in [2, 3, 4]:
            detected_seal = "CROSS"
            confidence = 0.88

        # 4. RAM SEAL -> Fuuton Rasenshuriken
        elif 0.80 <= aspect_ratio <= 1.25 and solidity >= 0.83 and defect_count <= 1:
            detected_seal = "RAM"
            confidence = 0.90

        # 5. OPEN PALM -> Chakra Burst
        elif defect_count >= 3 and solidity < 0.72 and aspect_ratio <= 1.4:
            detected_seal = "OPEN_PALM"
            confidence = 0.85

        return detected_seal, confidence, features

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], float, Dict[str, Any], np.ndarray]:
        now = time.time()
        mask = self.get_skin_mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        triggered_action = None
        progress = 0.0
        info = {
            "seal": "NONE",
            "contour": None,
            "hull": None,
            "features": {}
        }

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            seal_name, confidence, feats = self.classify_seal(largest_contour)
            hull = cv2.convexHull(largest_contour)

            info["seal"] = seal_name
            info["contour"] = largest_contour
            info["hull"] = hull
            info["features"] = feats
            info["confidence"] = confidence

            if seal_name not in ["NONE", "UNKNOWN"]:
                if (now - self.last_trigger_time) >= self.cooldown_sec:
                    if self.candidate_seal == seal_name:
                        self.candidate_frames += 1
                    else:
                        self.candidate_seal = seal_name
                        self.candidate_frames = 1

                    progress = min(1.0, float(self.candidate_frames) / self.required_hold_frames)

                    if self.candidate_frames >= self.required_hold_frames:
                        triggered_action = seal_name
                        self.last_trigger_time = now
                        self.candidate_frames = 0
                        self.candidate_seal = None
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
