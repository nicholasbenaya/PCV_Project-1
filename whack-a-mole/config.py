"""Konfigurasi parameter visual, segmentasi citra, dan gameplay Whack-A-Mole."""

from typing import Any, Dict, List
import numpy as np

# Pengaturan kanvas dan kamera diperbesar ke HD widescreen
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
WINDOW_TITLE = "Tugas Game - Whack-A-Mole Virtual"

# Rentang segmentasi warna kulit manusia pada HSV (H: 0..179 di OpenCV)
LOWER_SKIN_1 = np.array([0, 30, 50], dtype=np.uint8)
UPPER_SKIN_1 = np.array([25, 210, 255], dtype=np.uint8)
LOWER_SKIN_2 = np.array([170, 30, 50], dtype=np.uint8)
UPPER_SKIN_2 = np.array([180, 210, 255], dtype=np.uint8)

# Parameter pembersihan kontur dan pelacakan
MIN_HAND_CONTOUR_AREA = 1800  # Luas piksel minimal kontur tangan
CENTROID_SMOOTHING_FACTOR = 0.65  # Bobot EMA (0.0 = kaku, 1.0 = raw tanpa filter)
MORPH_KERNEL_SIZE = (5, 5)

# Parameter gameplay
DEFAULT_GAME_DURATION = 30.0  # Durasi sesi permainan (detik)
HIT_DEBOUNCE_COOLDOWN = 0.38  # Interval minimal antar pukulan valid (detik)
MOLE_MIN_LIFESPAN = 1.40  # Batas bawah durasi kemunculan mole (detik)
MOLE_MAX_LIFESPAN = 1.90  # Batas atas durasi kemunculan mole (detik)
HIT_EFFECT_DURATION = 0.45  # Durasi animasi visual pukulan (detik)

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
