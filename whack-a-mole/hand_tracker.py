"""Modul pelacakan tangan & klasifikasi Hand Pose (HIGH_FIVE vs FIST) berbasis OpenCV murni.

Menggunakan kombinasi analisis fitur bentuk (Convexity Defects, Solidity, Aspect Ratio),
Auto Face Blackout, Preset Mode switching, dan Cursor Position Freeze/Damping.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    from .config import (
        CENTROID_SMOOTHING_FACTOR,
        DEFECT_MAX_ANGLE_DEG,
        DEFECT_MIN_DEPTH_RATIO,
        FIST_ASPECT_RATIO_MAX,
        FIST_ASPECT_RATIO_MIN,
        FIST_FREEZE_DURATION,
        FIST_MAX_DEFECTS,
        FIST_MIN_SOLIDITY,
        HIGH_FIVE_MAX_SOLIDITY,
        HIGH_FIVE_MIN_DEFECTS,
        LOWER_GLOVE_HSV,
        LOWER_OBJECT_HSV,
        LOWER_SKIN_1,
        LOWER_SKIN_2,
        LOWER_SKIN_YCRCB,
        MAX_HAND_CONTOUR_AREA,
        MIN_HAND_CONTOUR_AREA,
        MORPH_KERNEL_SIZE,
        POSE_FIST,
        POSE_HIGH_FIVE,
        POSE_NONE,
        PRESET_GLOVE,
        PRESET_NAMES,
        PRESET_OBJECT,
        PRESET_SKIN,
        UPPER_GLOVE_HSV,
        UPPER_OBJECT_HSV,
        UPPER_SKIN_1,
        UPPER_SKIN_2,
        UPPER_SKIN_YCRCB,
    )
except ImportError:
    from config import (
        CENTROID_SMOOTHING_FACTOR,
        DEFECT_MAX_ANGLE_DEG,
        DEFECT_MIN_DEPTH_RATIO,
        FIST_ASPECT_RATIO_MAX,
        FIST_ASPECT_RATIO_MIN,
        FIST_FREEZE_DURATION,
        FIST_MAX_DEFECTS,
        FIST_MIN_SOLIDITY,
        HIGH_FIVE_MAX_SOLIDITY,
        HIGH_FIVE_MIN_DEFECTS,
        LOWER_GLOVE_HSV,
        LOWER_OBJECT_HSV,
        LOWER_SKIN_1,
        LOWER_SKIN_2,
        LOWER_SKIN_YCRCB,
        MAX_HAND_CONTOUR_AREA,
        MIN_HAND_CONTOUR_AREA,
        MORPH_KERNEL_SIZE,
        POSE_FIST,
        POSE_HIGH_FIVE,
        POSE_NONE,
        PRESET_GLOVE,
        PRESET_NAMES,
        PRESET_OBJECT,
        PRESET_SKIN,
        UPPER_GLOVE_HSV,
        UPPER_OBJECT_HSV,
        UPPER_SKIN_1,
        UPPER_SKIN_2,
        UPPER_SKIN_YCRCB,
    )


@dataclass
class HandTrackingResult:
    """Struktur data hasil deteksi dan klasifikasi gestur tangan pada satu frame."""

    centroid: Optional[Tuple[int, int]]  # Posisi stabil (termasuk freeze damping saat FIST)
    raw_centroid: Optional[Tuple[int, int]]
    stabilized_centroid: Optional[Tuple[int, int]]
    pose: str  # POSE_HIGH_FIVE, POSE_FIST, atau POSE_NONE
    raw_pose: str
    is_detected: bool
    is_fist: bool
    is_high_five: bool
    is_fist_click: bool  # Pemicu klik satu kali (edge trigger) saat berubah ke FIST
    contour: Optional[np.ndarray]
    hull: Optional[np.ndarray]
    defects_pts: List[Tuple[int, int]] = field(default_factory=list)
    defect_count: int = 0
    solidity: float = 0.0
    aspect_ratio: float = 0.0
    area: float = 0.0
    mask: Optional[np.ndarray] = None
    preset_mode: int = PRESET_SKIN
    preset_name: str = PRESET_NAMES[PRESET_SKIN]


class HandTracker:
    """Pelacak tangan dan pengklasifikasi Hand Pose murni OpenCV tanpa deep learning."""

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

        # Preset mode segmentasi aktif (Skin / Glove / Object)
        self.preset_mode: int = PRESET_SKIN

        # Rentang segmentasi warna kulit default
        self.ycrcb_lower = LOWER_SKIN_YCRCB.copy()
        self.ycrcb_upper = UPPER_SKIN_YCRCB.copy()
        self.hsv_lower1 = LOWER_SKIN_1.copy()
        self.hsv_upper1 = UPPER_SKIN_1.copy()
        self.hsv_lower2 = LOWER_SKIN_2.copy()
        self.hsv_upper2 = UPPER_SKIN_2.copy()

        # Rentang untuk Black Glove Mode & High Contrast Object
        self.glove_lower = LOWER_GLOVE_HSV.copy()
        self.glove_upper = UPPER_GLOVE_HSV.copy()
        self.object_lower = LOWER_OBJECT_HSV.copy()
        self.object_upper = UPPER_OBJECT_HSV.copy()

        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)

        # Inisialisasi Haar Cascade frontal face jika modul classifier tersedia di OpenCV
        self.face_cascade = None
        if hasattr(cv2, "CascadeClassifier"):
            try:
                face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                cascade = cv2.CascadeClassifier(face_path)
                if not cascade.empty():
                    self.face_cascade = cascade
            except Exception:
                self.face_cascade = None

        self.is_calibrated: bool = False
        self.calib_info: Optional[Dict] = None

        # Temporal smoothing untuk stabilitas pose
        self.pose_history: deque = deque(maxlen=5)
        self.prev_smoothed_pose: str = POSE_NONE

        # Mekanisme Cursor Freeze/Damping saat transisi ke FIST
        self.last_high_five_pos: Optional[Tuple[int, int]] = None
        self.last_high_five_time: float = 0.0
        self.fist_transition_time: float = 0.0

    def reset(self) -> None:
        """Mereset riwayat posisi centroid dan pose buffer."""
        self.prev_centroid = None
        self.pose_history.clear()
        self.prev_smoothed_pose = POSE_NONE
        self.last_high_five_pos = None

    def cycle_preset_mode(self) -> str:
        """Berganti ke mode preset segmentasi berikutnya (Skin -> Glove -> Object)."""
        self.preset_mode = (self.preset_mode + 1) % len(PRESET_NAMES)
        return PRESET_NAMES[self.preset_mode]

    def calibrate_skin_sample(self, roi_bgr: np.ndarray) -> Dict:
        """Mengkalibrasi rentang warna dari area telapak tangan atau objek sampling."""
        if roi_bgr.size == 0:
            return {}

        if self.preset_mode == PRESET_SKIN:
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

        elif self.preset_mode == PRESET_OBJECT:
            # Kalibrasi objek kontras tinggi
            hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            mean_hsv, std_hsv = cv2.meanStdDev(hsv_roi)
            h_mean, h_std = mean_hsv[0][0], std_hsv[0][0]
            s_mean, s_std = mean_hsv[1][0], std_hsv[1][0]
            h_tol = max(18, int(h_std * 2.8))
            s_tol = max(40, int(s_std * 2.8))

            self.object_lower = np.array([max(0, int(h_mean - h_tol)), max(40, int(s_mean - s_tol)), 60], dtype=np.uint8)
            self.object_upper = np.array([min(179, int(h_mean + h_tol)), 255, 255], dtype=np.uint8)

        self.is_calibrated = True
        self.calib_info = {
            "mode": self.preset_mode,
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

    def get_segmentation_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Menghasilkan masker biner sesuai preset aktif (Skin, Black Glove, atau High-Contrast Object)."""
        blurred = cv2.GaussianBlur(frame_bgr, (7, 7), 0)

        if self.preset_mode == PRESET_SKIN:
            ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
            mask_ycrcb = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)

            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask_hsv1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
            mask_hsv2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
            mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

            combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
            return combined

        elif self.preset_mode == PRESET_GLOVE:
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask_glove = cv2.inRange(hsv, self.glove_lower, self.glove_upper)
            return mask_glove

        else:
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask_obj = cv2.inRange(hsv, self.object_lower, self.object_upper)
            return mask_obj

    def blackout_faces(self, mask: np.ndarray, frame_bgr: np.ndarray) -> None:
        """Menghapus area wajah dan leher pemain pada masker biner untuk mencegah face hijacking."""
        h, w = mask.shape[:2]

        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60)
            )
            for (fx, fy, fw, fh) in faces:
                neck_margin = int(fh * 0.45)
                y1 = max(0, fy - 10)
                y2 = min(h, fy + fh + neck_margin)
                x1 = max(0, fx - 15)
                x2 = min(w, fx + fw + 15)
                mask[y1:y2, x1:x2] = 0

    def extract_pose_features(
        self, contour: np.ndarray
    ) -> Tuple[str, int, float, float, List[Tuple[int, int]], Optional[np.ndarray]]:
        """Menganalisis Convexity Defects, Solidity, dan Aspect Ratio untuk mengklasifikasi pose tangan."""
        area = float(cv2.contourArea(contour))
        if area <= 0:
            return POSE_NONE, 0, 0.0, 0.0, [], None

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / max(1.0, float(h))

        # 1. Convex Hull
        hull_points = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull_points))
        solidity = area / max(1.0, hull_area)

        # 2. Convexity Defects
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        valid_defects_count = 0
        defect_pts: List[Tuple[int, int]] = []

        if hull_indices is not None and len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    min_dim = min(w, h)
                    depth_thresh = DEFECT_MIN_DEPTH_RATIO * min_dim
                    defects = defects.reshape(-1, 4)

                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i]
                        depth = d / 256.0

                        if depth < depth_thresh:
                            continue

                        start = contour[s][0]
                        end = contour[e][0]
                        far = contour[f][0]

                        # Vektor celah antar jari
                        a = start.astype(float) - far.astype(float)
                        b = end.astype(float) - far.astype(float)
                        dot_val = float(np.dot(a, b))
                        norm_prod = float(np.linalg.norm(a) * np.linalg.norm(b))

                        if norm_prod > 1e-4:
                            cos_val = np.clip(dot_val / norm_prod, -1.0, 1.0)
                            angle = float(np.degrees(np.arccos(cos_val)))
                        else:
                            angle = 180.0

                        # Defek valid: celah antar jari membentuk sudut tajam (< 90 derajat)
                        if angle < DEFECT_MAX_ANGLE_DEG:
                            valid_defects_count += 1
                            defect_pts.append((int(far[0]), int(far[1])))
            except cv2.error:
                valid_defects_count = 0

        # 3. Klasifikasi Hand Pose
        raw_pose = POSE_NONE
        if valid_defects_count >= HIGH_FIVE_MIN_DEFECTS or solidity < HIGH_FIVE_MAX_SOLIDITY:
            raw_pose = POSE_HIGH_FIVE
        elif (
            valid_defects_count <= FIST_MAX_DEFECTS
            and solidity > FIST_MIN_SOLIDITY
            and (FIST_ASPECT_RATIO_MIN <= aspect_ratio <= FIST_ASPECT_RATIO_MAX)
        ):
            raw_pose = POSE_FIST

        return raw_pose, valid_defects_count, solidity, aspect_ratio, defect_pts, hull_points

    def process_frame(
        self,
        frame: np.ndarray,
        play_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> HandTrackingResult:
        """Memproses frame citra, mengisolasi area bermain, mendeteksi kontur, dan mengekstrak pose tangan."""
        current_time = time.time()
        raw_mask = self.get_segmentation_mask(frame)

        # 1. Eliminasi Wajah Otomatis (Auto Face Blackout)
        self.blackout_faces(raw_mask, frame)

        # 2. Isolasi Playing Section (zeroing Non-Playing Safe Zone)
        if play_rect is not None:
            px1, py1, px2, py2 = play_rect
            isolated_mask = np.zeros_like(raw_mask)
            isolated_mask[py1:py2, px1:px2] = raw_mask[py1:py2, px1:px2]
        else:
            isolated_mask = raw_mask

        # 3. Pembersihan morfologi
        cleaned_mask = cv2.morphologyEx(isolated_mask, cv2.MORPH_OPEN, self._kernel_open, iterations=2)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, self._kernel_close, iterations=2)
        cleaned_mask = cv2.dilate(cleaned_mask, self._kernel_open, iterations=1)

        # 4. Deteksi kontur eksternal
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

            score = min(area, 30000.0) * 0.1
            if self.prev_centroid is not None:
                dist = math.hypot(cx - self.prev_centroid[0], cy - self.prev_centroid[1])
                if dist < 260:
                    score += (260 - dist) * 20.0

            candidates.append((score, cnt, (cx, cy), area))

        if not candidates:
            self.prev_centroid = None
            self.pose_history.append(POSE_NONE)
            return HandTrackingResult(
                centroid=None,
                raw_centroid=None,
                stabilized_centroid=None,
                pose=POSE_NONE,
                raw_pose=POSE_NONE,
                is_detected=False,
                is_fist=False,
                is_high_five=False,
                is_fist_click=False,
                contour=None,
                hull=None,
                defects_pts=[],
                defect_count=0,
                solidity=0.0,
                aspect_ratio=0.0,
                area=0.0,
                mask=cleaned_mask,
                preset_mode=self.preset_mode,
                preset_name=PRESET_NAMES[self.preset_mode],
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_contour, (raw_cx, raw_cy), max_area = candidates[0]

        # Stabilisasi pergerakan centroid dengan filter EMA
        if self.prev_centroid is not None:
            smooth_cx = int(self.smoothing * raw_cx + (1.0 - self.smoothing) * self.prev_centroid[0])
            smooth_cy = int(self.smoothing * raw_cy + (1.0 - self.smoothing) * self.prev_centroid[1])
            smoothed_centroid = (smooth_cx, smooth_cy)
        else:
            smoothed_centroid = (raw_cx, raw_cy)

        self.prev_centroid = smoothed_centroid

        # 5. Ekstraksi fitur bentuk dan klasifikasi pose
        raw_pose, defect_count, solidity, aspect_ratio, defect_pts, hull_points = (
            self.extract_pose_features(best_contour)
        )

        # 6. Temporal smoothing (Majority voting buffer 5 frame)
        self.pose_history.append(raw_pose)
        smoothed_pose = Counter(self.pose_history).most_common(1)[0][0]

        # 7. Deteksi transisi klik (Edge Trigger: HIGH_FIVE -> FIST)
        is_fist_click = False
        if smoothed_pose == POSE_FIST and self.prev_smoothed_pose == POSE_HIGH_FIVE:
            is_fist_click = True
            self.fist_transition_time = current_time

        # 8. Cursor Position Freeze / Damping
        if smoothed_pose == POSE_HIGH_FIVE:
            self.last_high_five_pos = smoothed_centroid
            self.last_high_five_time = current_time
            stabilized_centroid = smoothed_centroid
        elif smoothed_pose == POSE_FIST:
            # Jika baru saja transisi ke FIST dalam grace period, kunci kursor ke posisi Open Palm terakhir
            if (
                self.last_high_five_pos is not None
                and (current_time - self.last_high_five_time) < FIST_FREEZE_DURATION
            ):
                stabilized_centroid = self.last_high_five_pos
            else:
                stabilized_centroid = smoothed_centroid
        else:
            stabilized_centroid = smoothed_centroid

        self.prev_smoothed_pose = smoothed_pose

        return HandTrackingResult(
            centroid=stabilized_centroid,
            raw_centroid=(raw_cx, raw_cy),
            stabilized_centroid=stabilized_centroid,
            pose=smoothed_pose,
            raw_pose=raw_pose,
            is_detected=True,
            is_fist=(smoothed_pose == POSE_FIST),
            is_high_five=(smoothed_pose == POSE_HIGH_FIVE),
            is_fist_click=is_fist_click,
            contour=best_contour,
            hull=hull_points,
            defects_pts=defect_pts,
            defect_count=defect_count,
            solidity=solidity,
            aspect_ratio=aspect_ratio,
            area=max_area,
            mask=cleaned_mask,
            preset_mode=self.preset_mode,
            preset_name=PRESET_NAMES[self.preset_mode],
        )
