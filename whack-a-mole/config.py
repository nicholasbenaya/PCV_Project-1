"""Konfigurasi parameter visual, segmentasi citra, gameplay, level, dan audio (HD 1280x720)."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

# Pengaturan kanvas dan kamera diperbesar ke HD widescreen
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
WINDOW_TITLE = "Tugas Game - Whack-A-Mole Virtual (HD 1280x720)"

# Rentang segmentasi warna kulit manusia default pada HSV (H: 0..179 di OpenCV)
LOWER_SKIN_1 = np.array([0, 30, 50], dtype=np.uint8)
UPPER_SKIN_1 = np.array([25, 210, 255], dtype=np.uint8)
LOWER_SKIN_2 = np.array([170, 30, 50], dtype=np.uint8)
UPPER_SKIN_2 = np.array([180, 210, 255], dtype=np.uint8)

# Rentang segmentasi warna kulit default pada YCrCb (invarian pencahayaan)
LOWER_SKIN_YCRCB = np.array([0, 133, 77], dtype=np.uint8)
UPPER_SKIN_YCRCB = np.array([255, 175, 127], dtype=np.uint8)

# Parameter pembersihan kontur dan pelacakan
MIN_HAND_CONTOUR_AREA = 1800  # Luas piksel minimal kontur tangan
MAX_HAND_CONTOUR_AREA = 120000  # Batas atas untuk mengeliminasi background masif
CENTROID_SMOOTHING_FACTOR = 0.65  # Bobot EMA untuk stabilisasi kursor
MORPH_KERNEL_SIZE = (5, 5)
BUTTON_HOVER_TRIGGER_TIME = 0.85  # Waktu tahan kursor tangan untuk klik virtual (detik)

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
]

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
COLOR_HOLE_OUTER = (80, 80, 80)
COLOR_HOLE_INNER = (35, 35, 35)
COLOR_MOLE_BODY = (40, 120, 225)
COLOR_BUTTON_BG = (42, 42, 45)
COLOR_BUTTON_HOVER = (65, 65, 75)
COLOR_BUTTON_BORDER = (0, 200, 255)
COLOR_HEART_RED = (0, 0, 235)
COLOR_LOCKED_GRAY = (90, 90, 90)
