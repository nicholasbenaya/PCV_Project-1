"""Konfigurasi parameter visual, segmentasi citra, gameplay, level, dan audio (Full Screen HD 1280x720)."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

# Pengaturan kanvas dan kamera (HD Widescreen 1280x720)
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
WINDOW_TITLE = "Tugas Game - Whack-A-Mole Virtual (Dual-Section Safe Area & Classical PCD)"

# Preset Dual-Section (Playing Section vs Non-Playing Safe Zone)
ZONE_PRESETS: List[Dict[str, Any]] = [
    {
        "name": "TANGAN KANAN (Kiri: Wajah Diblokir)",
        "play_rect": (420, 0, 1280, 720),
        "safe_rect": (0, 0, 420, 720),
        "split_line": ((420, 0), (420, 720)),
    },
    {
        "name": "TANGAN KIRI (Kanan: Wajah Diblokir)",
        "play_rect": (0, 0, 860, 720),
        "safe_rect": (860, 0, 1280, 720),
        "split_line": ((860, 0), (860, 720)),
    },
    {
        "name": "TANGAN BAWAH (Atas: Wajah Diblokir)",
        "play_rect": (0, 240, 1280, 720),
        "safe_rect": (0, 0, 1280, 240),
        "split_line": ((0, 240), (1280, 240)),
    },
    {
        "name": "LAYAR PENUH (Tanpa Safe Zone)",
        "play_rect": (0, 0, 1280, 720),
        "safe_rect": (0, 0, 0, 0),
        "split_line": ((0, 0), (0, 0)),
    },
]

# Rentang segmentasi warna kulit default pada HSV (H: 0..179 di OpenCV)
LOWER_SKIN_1 = np.array([0, 30, 50], dtype=np.uint8)
UPPER_SKIN_1 = np.array([25, 210, 255], dtype=np.uint8)
LOWER_SKIN_2 = np.array([170, 30, 50], dtype=np.uint8)
UPPER_SKIN_2 = np.array([180, 210, 255], dtype=np.uint8)

# Rentang segmentasi warna kulit default pada YCrCb (invarian pencahayaan)
LOWER_SKIN_YCRCB = np.array([0, 133, 77], dtype=np.uint8)
UPPER_SKIN_YCRCB = np.array([255, 175, 127], dtype=np.uint8)

# Preset Segmentasi Alternatif untuk Demo
PRESET_SKIN = 0
PRESET_GLOVE = 1
PRESET_OBJECT = 2
PRESET_NAMES = ["Skin (YCrCb+HSV)", "Black Glove", "High-Contrast Object"]

# Rentang untuk Black Glove Mode (Threshold Value gelap pada latar terang)
LOWER_GLOVE_HSV = np.array([0, 0, 0], dtype=np.uint8)
UPPER_GLOVE_HSV = np.array([180, 255, 75], dtype=np.uint8)

# Rentang untuk High-Contrast Object Backup (default: warna biru / merah terang)
LOWER_OBJECT_HSV = np.array([95, 120, 80], dtype=np.uint8)
UPPER_OBJECT_HSV = np.array([130, 255, 255], dtype=np.uint8)

# Parameter Background Subtraction Berbasis Referensi (Tombol 'B')
BG_DIFF_THRESHOLD = 18  # Ambang batas selisih piksel multi-channel (cv2.absdiff)

# Parameter Eliminasi Wajah via Topologi Hierarki Kontur (cv2.RETR_CCOMP)
FACE_MIN_INTERNAL_HOLES = 3  # Ambang batas lubang internal mata/mulut
FACE_MIN_HOLE_AREA = 25.0  # Luas minimal child contour untuk dihitung sebagai lubang

# Parameter "Open-Palm Lock-On" & Nearest-Neighbor Tracking
LOCKON_MAX_DISTANCE = 180.0  # Radius maksimal pelacakan nearest-neighbor (piksel)
LOCKON_LOST_TIMEOUT_FRAMES = 24  # Toleransi frame hilang sebelum melepaskan lock-on (~0.8 detik)

# Parameter pembersihan kontur dan pelacakan
MIN_HAND_CONTOUR_AREA = 2500  # Luas piksel minimal kontur tangan (menolak noise/serpihan kecil)
MIN_HAND_WIDTH = 55  # Lebar minimal bounding box tangan (piksel)
MIN_HAND_HEIGHT = 55  # Tinggi minimal bounding box tangan (piksel)
MAX_HAND_CONTOUR_AREA = 140000  # Batas atas untuk mengeliminasi background masif
CENTROID_SMOOTHING_FACTOR = 0.65  # Bobot EMA untuk stabilisasi kursor
MORPH_KERNEL_SIZE = (5, 5)
BUTTON_HOVER_TRIGGER_TIME = 0.85  # Waktu tahan kursor tangan alternatif (detik)

# Definisi Hand Pose
POSE_NONE = "NONE"
POSE_HIGH_FIVE = "HIGH_FIVE"
POSE_FIST = "FIST"

# Threshold Klasifikasi 2 Hand Pose Sederhana (HIGH_FIVE move vs FIST whack)
# Aturan sesuai kebutuhan: minimal deteksi 1 titik merah (defect >= 1) -> HIGH_FIVE, sisanya -> FIST (whack)
HIGH_FIVE_MIN_DEFECTS = 1  # Minimal 1 titik merah (celah sela jari terbuka)
HIGH_FIVE_MAX_SOLIDITY = 0.80
DEFECT_MAX_ANGLE_DEG = 85.0  # Sudut tajam celah sela jari (< 85 derajat)
DEFECT_MIN_DEPTH_RATIO = 0.08  # Kedalaman minimal celah sela jari

FIST_MAX_DEFECTS = 0  # 0 titik merah (jari mengepal rapat tanpa celah)
FIST_MIN_SOLIDITY = 0.80
FIST_ASPECT_RATIO_MIN = 0.55
FIST_ASPECT_RATIO_MAX = 1.45
FIST_FREEZE_DURATION = 0.35  # Durasi pembekuan posisi kursor saat transisi ke FIST (detik)

# Pilihan preset ukuran grid (baris, kolom)
GRID_PRESETS: List[Tuple[int, int]] = [
    (2, 2),  # 4 Lubang - Pemula
    (2, 3),  # 6 Lubang - Klasik Default
    (3, 3),  # 9 Lubang - Menantang
    (3, 4),  # 12 Lubang - Ekstrem Luas
]
DEFAULT_GRID_INDEX = 1  # 2x3 sebagai default

# Parameter gameplay umum
HIT_DEBOUNCE_COOLDOWN = 0.38  # Interval minimal antar pukulan valid (detik)
HIT_EFFECT_DURATION = 0.45  # Durasi animasi visual pukulan (detik)

# Parameter Mode Survival
SURVIVAL_INITIAL_LIVES = 3
SURVIVAL_MAX_LIVES = 5
SURVIVAL_TIER_INTERVAL = 18.0  # Durasi transisi kenaikan intensitas (detik)


@dataclass
class LevelConfig:
    """Konfigurasi spesifik per level progression (Level 1 - 10)."""

    level_num: int
    title: str
    target_hits: int
    duration_seconds: float
    mole_min_lifespan: float
    mole_max_lifespan: float
    simultaneous_moles: int
    bgm_intensity: int  # 1 (Santai) sampai 4 (Ekstrem)


# Definisi 10 level bertingkat
LEVEL_CONFIGS: Dict[int, LevelConfig] = {
    1: LevelConfig(1, "Pemanasan", 8, 30.0, 1.70, 2.00, 1, 1),
    2: LevelConfig(2, "Refleks Cepat", 10, 30.0, 1.50, 1.80, 1, 1),
    3: LevelConfig(3, "Lubang Lincah", 12, 30.0, 1.35, 1.60, 1, 2),
    4: LevelConfig(4, "Fokus Ganda", 14, 28.0, 1.20, 1.45, 1, 2),
    5: LevelConfig(5, "Irama Cepat", 16, 28.0, 1.10, 1.30, 1, 2),
    6: LevelConfig(6, "Ujian Akurasi", 18, 26.0, 1.00, 1.20, 1, 3),
    7: LevelConfig(7, "Serbuan Mole", 20, 25.0, 0.90, 1.10, 1, 3),
    8: LevelConfig(8, "Badai Refleks", 22, 25.0, 0.80, 1.00, 2, 3),
    9: LevelConfig(9, "Zona Ekstrem", 25, 25.0, 0.70, 0.90, 2, 4),
    10: LevelConfig(10, "Grandmaster", 30, 25.0, 0.55, 0.75, 2, 4),
}

# Warna BGR untuk rendering UI
COLOR_PRIMARY_CYAN = (0, 220, 255)
COLOR_TEXT_WHITE = (245, 245, 245)
COLOR_TEXT_DARK = (20, 20, 20)
COLOR_TIMER_NORMAL = (0, 235, 0)
COLOR_TIMER_CRITICAL = (0, 0, 245)
COLOR_HIT_FLASH = (0, 255, 120)
COLOR_RETICLE = (255, 220, 0)
COLOR_HIGH_FIVE = (0, 255, 120)  # Hijau untuk OPEN PALM / MOVE
COLOR_FIST = (0, 50, 255)  # Merah untuk FIST / WHACK
COLOR_HOLE_OUTER = (80, 80, 80)
COLOR_HOLE_INNER = (35, 35, 35)
COLOR_MOLE_BODY = (40, 120, 225)
COLOR_BUTTON_BG = (42, 42, 45)
COLOR_BUTTON_HOVER = (65, 65, 75)
COLOR_BUTTON_BORDER = (0, 200, 255)
COLOR_HEART_RED = (0, 0, 235)
COLOR_LOCKED_GRAY = (90, 90, 90)
