"""Modul pelacakan tangan & klasifikasi Hand Pose murni OpenCV tanpa Learning Model / XML Classifier.

Menerapkan 4 pilar pengolahan citra klasik:
1. Background Subtraction Berbasis Referensi (Tombol 'B') via cv2.absdiff()
2. Eliminasi Wajah via Topologi Hierarki Kontur (cv2.RETR_CCOMP) & Lubang Internal
3. Mekanisme "Open-Palm Lock-On" & Nearest-Neighbor Spatial Tracking
4. Ekstraksi Fitur Geometri Bentuk (Convexity Defects, Solidity, Aspect Ratio) & Cursor Freeze
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
        BG_DIFF_THRESHOLD,
        CENTROID_SMOOTHING_FACTOR,
        DEFECT_MAX_ANGLE_DEG,
        DEFECT_MIN_DEPTH_RATIO,
        FACE_MIN_HOLE_AREA,
        FACE_MIN_INTERNAL_HOLES,
        FIST_ASPECT_RATIO_MAX,
        FIST_ASPECT_RATIO_MIN,
        FIST_FREEZE_DURATION,
        FIST_MAX_DEFECTS,
        FIST_MIN_SOLIDITY,
        HIGH_FIVE_MAX_SOLIDITY,
        HIGH_FIVE_MIN_DEFECTS,
        LOCKON_LOST_TIMEOUT_FRAMES,
        LOCKON_MAX_DISTANCE,
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
        BG_DIFF_THRESHOLD,
        CENTROID_SMOOTHING_FACTOR,
        DEFECT_MAX_ANGLE_DEG,
        DEFECT_MIN_DEPTH_RATIO,
        FACE_MIN_HOLE_AREA,
        FACE_MIN_INTERNAL_HOLES,
        FIST_ASPECT_RATIO_MAX,
        FIST_ASPECT_RATIO_MIN,
        FIST_FREEZE_DURATION,
        FIST_MAX_DEFECTS,
        FIST_MIN_SOLIDITY,
        HIGH_FIVE_MAX_SOLIDITY,
        HIGH_FIVE_MIN_DEFECTS,
        LOCKON_LOST_TIMEOUT_FRAMES,
        LOCKON_MAX_DISTANCE,
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
    is_locked: bool  # Status Open-Palm Lock-On aktif
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
    internal_holes_count: int = 0
    bg_captured: bool = False
    mask: Optional[np.ndarray] = None
    preset_mode: int = PRESET_SKIN
    preset_name: str = PRESET_NAMES[PRESET_SKIN]


class HandTracker:
    """Pelacak tangan klasik murni OpenCV tanpa Learning Model dan tanpa Safe Zone statis."""

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

        # 1. Background Subtraction Berbasis Referensi (Tombol 'B')
        self.bg_frame: Optional[np.ndarray] = None
        self.bg_gray: Optional[np.ndarray] = None
        self.bg_color_mask: Optional[np.ndarray] = None

        # 2. State Open-Palm Lock-On & Nearest-Neighbor Tracking
        self.is_hand_locked: bool = False
        self.last_hand_centroid: Optional[Tuple[int, int]] = None
        self.lock_lost_frames: int = 0

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

        self.is_calibrated: bool = False
        self.calib_info: Optional[Dict] = None

        # Temporal smoothing untuk stabilitas pose
        self.pose_history: deque = deque(maxlen=5)
        self.prev_smoothed_pose: str = POSE_NONE

        # Mekanisme Cursor Freeze/Damping saat transisi ke FIST
        self.last_high_five_pos: Optional[Tuple[int, int]] = None
        self.last_high_five_time: float = 0.0

    def capture_background(self, frame_bgr: np.ndarray) -> bool:
        """Merekam frame referensi latar belakang saat tombol 'B' ditekan pemain."""
        if frame_bgr is None or frame_bgr.size == 0:
            return False
        self.bg_frame = frame_bgr.copy()
        gray = cv2.cvtColor(self.bg_frame, cv2.COLOR_BGR2GRAY)
        self.bg_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        self.bg_color_mask = self.get_color_mask(self.bg_frame)
        self.reset_lock()
        return True

    def clear_background(self) -> None:
        """Menghapus referensi latar belakang (Bypass mode)."""
        self.bg_frame = None
        self.bg_gray = None
        self.bg_color_mask = None
        self.reset_lock()

    def reset_lock(self) -> None:
        """Mereset status lock-on tangan agar mencari gestur HIGH_FIVE baru."""
        self.is_hand_locked = False
        self.last_hand_centroid = None
        self.lock_lost_frames = 0
        self.prev_centroid = None
        self.pose_history.clear()
        self.prev_smoothed_pose = POSE_NONE

    def cycle_preset_mode(self) -> str:
        """Berganti ke mode preset segmentasi berikutnya (Skin -> Glove -> Object)."""
        self.preset_mode = (self.preset_mode + 1) % len(PRESET_NAMES)
        if self.bg_frame is not None:
            self.bg_color_mask = self.get_color_mask(self.bg_frame)
        self.reset_lock()
        return PRESET_NAMES[self.preset_mode]

    def calibrate_skin_sample(self, roi_bgr: np.ndarray) -> Dict:
        """Mengkalibrasi rentang warna dari area sampling telapak tangan."""
        if roi_bgr.size == 0:
            return {}

        if self.preset_mode == PRESET_SKIN:
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

        elif self.preset_mode == PRESET_OBJECT:
            hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            mean_hsv, std_hsv = cv2.meanStdDev(hsv_roi)
            h_mean, h_std = mean_hsv[0][0], std_hsv[0][0]
            s_mean, s_std = mean_hsv[1][0], std_hsv[1][0]
            h_tol = max(18, int(h_std * 2.8))
            s_tol = max(40, int(s_std * 2.8))

            self.object_lower = np.array([max(0, int(h_mean - h_tol)), max(40, int(s_mean - s_tol)), 60], dtype=np.uint8)
            self.object_upper = np.array([min(179, int(h_mean + h_tol)), 255, 255], dtype=np.uint8)

        self.is_calibrated = True
        if self.bg_frame is not None:
            self.bg_color_mask = self.get_color_mask(self.bg_frame)
        self.reset_lock()
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

    def get_color_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Menghasilkan masker biner berdasarkan warna (Skin / Black Glove / Object)."""
        blurred = cv2.GaussianBlur(frame_bgr, (7, 7), 0)

        if self.preset_mode == PRESET_SKIN:
            ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
            mask_ycrcb = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)

            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask_hsv1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
            mask_hsv2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
            mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

            return cv2.bitwise_and(mask_ycrcb, mask_hsv)

        elif self.preset_mode == PRESET_GLOVE:
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            return cv2.inRange(hsv, self.glove_lower, self.glove_upper)

        else:
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            return cv2.inRange(hsv, self.object_lower, self.object_upper)

    def extract_pose_features(
        self, contour: np.ndarray
    ) -> Tuple[str, int, float, float, List[Tuple[int, int]], Optional[np.ndarray]]:
        """Menganalisis Convexity Defects, Solidity, dan Aspect Ratio pada kontur."""
        if contour is None or len(contour) < 3:
            return POSE_NONE, 0, 0.0, 0.0, [], None

        contour = np.ascontiguousarray(contour).reshape(-1, 1, 2)
        area = float(cv2.contourArea(contour))
        if area <= 0:
            return POSE_NONE, 0, 0.0, 0.0, [], None

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / max(1.0, float(h))

        hull_points = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull_points))
        solidity = area / max(1.0, hull_area)

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

                        a = start.astype(float) - far.astype(float)
                        b = end.astype(float) - far.astype(float)
                        dot_val = float(np.dot(a, b))
                        norm_prod = float(np.linalg.norm(a) * np.linalg.norm(b))

                        if norm_prod > 1e-4:
                            cos_val = np.clip(dot_val / norm_prod, -1.0, 1.0)
                            angle = float(np.degrees(np.arccos(cos_val)))
                        else:
                            angle = 180.0

                        if angle < DEFECT_MAX_ANGLE_DEG:
                            valid_defects_count += 1
                            defect_pts.append((int(far[0]), int(far[1])))
            except cv2.error:
                valid_defects_count = 0

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
        """Memproses frame menggunakan 4 pilar PCD klasik tanpa model / XML classifier dan tanpa Safe Zone statis."""
        current_time = time.time()

        # 1. Background Subtraction Berbasis Referensi (Tombol 'B') via cv2.absdiff()
        color_mask = self.get_color_mask(frame)

        if self.bg_frame is not None:
            # Perbedaan multi-channel (BGR + Chroma YCrCb) agar kebal terhadap metamerisme grayscale
            blurred_curr = cv2.GaussianBlur(frame, (7, 7), 0)
            blurred_bg = cv2.GaussianBlur(self.bg_frame, (7, 7), 0)

            diff_bgr = cv2.absdiff(blurred_curr, blurred_bg)
            diff_max_bgr = np.max(diff_bgr, axis=2)

            curr_ycrcb = cv2.cvtColor(blurred_curr, cv2.COLOR_BGR2YCrCb)
            bg_ycrcb = cv2.cvtColor(blurred_bg, cv2.COLOR_BGR2YCrCb)
            diff_ycrcb = cv2.absdiff(curr_ycrcb, bg_ycrcb)
            diff_chroma = np.maximum(diff_ycrcb[:, :, 1], diff_ycrcb[:, :, 2])

            diff_total = np.maximum(diff_max_bgr, diff_chroma)

            # Artefak statis = area yang SUDAH terdeteksi saat rekam BG dan TIDAK mengalami perubahan (diff < BG_DIFF_THRESHOLD)
            if self.bg_color_mask is not None:
                static_bg_artifacts = cv2.bitwise_and(
                    self.bg_color_mask,
                    cv2.inRange(diff_total, 0, BG_DIFF_THRESHOLD),
                )
                static_bg_artifacts = cv2.dilate(static_bg_artifacts, self._kernel_open, iterations=2)
                raw_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(static_bg_artifacts))
            else:
                _, fg_mask = cv2.threshold(diff_total, BG_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
                fg_mask = cv2.dilate(fg_mask, self._kernel_close, iterations=2)
                raw_mask = cv2.bitwise_and(color_mask, fg_mask)
        else:
            raw_mask = color_mask

        # Mask awal sebelum closing besar (untuk analisis topologi lubang internal RETR_CCOMP)
        pre_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, self._kernel_open, iterations=1)

        # 2. Eliminasi Wajah via Topologi Hierarki Kontur (cv2.RETR_CCOMP)
        ccomp_contours, hierarchy = cv2.findContours(pre_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        candidate_data = []  # List Tuple: (contour, (cx, cy), area, internal_holes, defects, solidity, AR, defect_pts, hull, raw_pose)

        if hierarchy is not None and len(ccomp_contours) > 0:
            for i, h_info in enumerate(hierarchy[0]):
                next_idx, prev_idx, first_child, parent = h_info

                # Hanya proses kontur eksternal (parent == -1)
                if parent != -1:
                    continue

                cnt = ccomp_contours[i]
                area = float(cv2.contourArea(cnt))
                if area < self.min_area or area > self.max_area:
                    continue

                moments = cv2.moments(cnt)
                if moments["m00"] == 0:
                    continue
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                # Hitung jumlah lubang internal (child contours dengan area >= FACE_MIN_HOLE_AREA)
                internal_holes = 0
                child_idx = first_child
                while child_idx != -1:
                    child_area = float(cv2.contourArea(ccomp_contours[child_idx]))
                    if child_area >= FACE_MIN_HOLE_AREA:
                        internal_holes += 1
                    child_idx = hierarchy[0][child_idx][0]  # Sibling berikutnya dari child

                # Ekstraksi fitur bentuk
                raw_pose, defect_count, solidity, aspect_ratio, defect_pts, hull_points = (
                    self.extract_pose_features(cnt)
                )

                # Eliminasi Wajah: jika memiliki >= 3 lubang (mata, alis, mulut) dan defek sela jari < 2 -> REJECT
                if internal_holes >= FACE_MIN_INTERNAL_HOLES and defect_count < 2:
                    continue

                candidate_data.append(
                    (
                        cnt,
                        (cx, cy),
                        area,
                        internal_holes,
                        defect_count,
                        solidity,
                        aspect_ratio,
                        defect_pts,
                        hull_points,
                        raw_pose,
                    )
                )

        # Mask bersih final untuk visualisasi PiP
        cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, self._kernel_open, iterations=2)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, self._kernel_close, iterations=2)

        # 3. Mekanisme "Open-Palm Lock-On" & Nearest-Neighbor Spatial Tracking
        chosen_candidate = None

        if not self.is_hand_locked:
            # Belum terkunci: HANYA BOLEH mengunci kontur yang memenuhi syarat HIGH_FIVE ketat
            for cand in candidate_data:
                cnt, (cx, cy), area, holes, defects, solidity, ar, defect_pts, hull, raw_pose = cand

                # Kunci Open-Palm Lock-On: Defek celah jari >= 3 dan Solidity < 0.82
                if defects >= HIGH_FIVE_MIN_DEFECTS and solidity <= 0.82:
                    self.is_hand_locked = True
                    self.last_hand_centroid = (cx, cy)
                    self.lock_lost_frames = 0
                    self.pose_history.clear()
                    self.pose_history.append(raw_pose)
                    chosen_candidate = cand
                    break

            if chosen_candidate is None:
                # Belum ada Open Palm terdeteksi: jangan pilih apa pun (wajah/kepala terhindar)
                self.pose_history.append(POSE_NONE)
                return HandTrackingResult(
                    centroid=None,
                    raw_centroid=None,
                    stabilized_centroid=None,
                    pose=POSE_NONE,
                    raw_pose=POSE_NONE,
                    is_detected=False,
                    is_locked=False,
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
                    internal_holes_count=0,
                    bg_captured=(self.bg_frame is not None),
                    mask=cleaned_mask,
                    preset_mode=self.preset_mode,
                    preset_name=PRESET_NAMES[self.preset_mode],
                )

        else:
            # Sudah terkunci: Cari kontur kandidat terdekat dari posisi terakhir (Nearest-Neighbor)
            if self.last_hand_centroid is not None:
                lx, ly = self.last_hand_centroid
                in_range_cands = []

                for cand in candidate_data:
                    cx, cy = cand[1]
                    dist = math.hypot(cx - lx, cy - ly)
                    if dist <= LOCKON_MAX_DISTANCE:
                        in_range_cands.append((dist, cand))

                if in_range_cands:
                    in_range_cands.sort(key=lambda x: x[0])
                    chosen_candidate = in_range_cands[0][1]
                    self.last_hand_centroid = chosen_candidate[1]
                    self.lock_lost_frames = 0
                else:
                    self.lock_lost_frames += 1
                    if self.lock_lost_frames > LOCKON_LOST_TIMEOUT_FRAMES:
                        # Tangan keluar dari layar / hilang melebihi batas toleransi -> Lepas kunci
                        self.is_hand_locked = False
                        self.last_hand_centroid = None
                        self.lock_lost_frames = 0

            if chosen_candidate is None:
                self.pose_history.append(POSE_NONE)
                return HandTrackingResult(
                    centroid=None,
                    raw_centroid=None,
                    stabilized_centroid=None,
                    pose=POSE_NONE,
                    raw_pose=POSE_NONE,
                    is_detected=False,
                    is_locked=self.is_hand_locked,
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
                    internal_holes_count=0,
                    bg_captured=(self.bg_frame is not None),
                    mask=cleaned_mask,
                    preset_mode=self.preset_mode,
                    preset_name=PRESET_NAMES[self.preset_mode],
                )

        (
            best_contour,
            (raw_cx, raw_cy),
            max_area,
            holes_count,
            defect_count,
            solidity,
            aspect_ratio,
            defect_pts,
            hull_points,
            raw_pose,
        ) = chosen_candidate

        # Stabilisasi posisi kursor dengan EMA
        if self.prev_centroid is not None:
            smooth_cx = int(self.smoothing * raw_cx + (1.0 - self.smoothing) * self.prev_centroid[0])
            smooth_cy = int(self.smoothing * raw_cy + (1.0 - self.smoothing) * self.prev_centroid[1])
            smoothed_centroid = (smooth_cx, smooth_cy)
        else:
            smoothed_centroid = (raw_cx, raw_cy)

        self.prev_centroid = smoothed_centroid

        # 4. Temporal Smoothing Buffer & Edge Trigger
        self.pose_history.append(raw_pose)
        smoothed_pose = Counter(self.pose_history).most_common(1)[0][0]

        is_fist_click = False
        if smoothed_pose == POSE_FIST and self.prev_smoothed_pose == POSE_HIGH_FIVE:
            is_fist_click = True

        # Cursor Freeze Damping saat mengepal
        if smoothed_pose == POSE_HIGH_FIVE:
            self.last_high_five_pos = smoothed_centroid
            self.last_high_five_time = current_time
            stabilized_centroid = smoothed_centroid
        elif smoothed_pose == POSE_FIST:
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
            is_locked=True,
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
            internal_holes_count=holes_count,
            bg_captured=(self.bg_frame is not None),
            mask=cleaned_mask,
            preset_mode=self.preset_mode,
            preset_name=PRESET_NAMES[self.preset_mode],
        )
