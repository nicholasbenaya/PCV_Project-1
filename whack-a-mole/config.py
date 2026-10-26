"""Konfigurasi parameter visual, segmentasi citra, dan gameplay Whack-A-Mole."""

import numpy as np

# Pengaturan kamera webcam
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
WINDOW_TITLE = "Tugas Game - Whack-A-Mole Virtual"

# Rentang segmentasi warna kulit manusia pada HSV (H: 0..179 di OpenCV)
# Rentang pertama: oranye-kuning dan merah muda
LOWER_SKIN_1 = np.array([0, 35, 60], dtype=np.uint8)
UPPER_SKIN_1 = np.array([22, 190, 255], dtype=np.uint8)
# Rentang kedua: menangani pembungkus rona merah pekat (wrap-around 170-180)
LOWER_SKIN_2 = np.array([170, 35, 60], dtype=np.uint8)
UPPER_SKIN_2 = np.array([180, 190, 255], dtype=np.uint8)

# Parameter pembersihan kontur dan pelacakan
MIN_HAND_CONTOUR_AREA = 2200  # Luas piksel minimal untuk menyaring noise latar
CENTROID_SMOOTHING_FACTOR = 0.65  # Bobot EMA (0.0 = kaku, 1.0 = raw tanpa filter)
MORPH_KERNEL_SIZE = (5, 5)

# Parameter gameplay
DEFAULT_GAME_DURATION = 30.0  # Durasi sesi permainan (detik)
HIT_DEBOUNCE_COOLDOWN = 0.40  # Interval minimal antar pukulan valid (detik)
MOLE_MIN_LIFESPAN = 1.50  # Batas bawah durasi kemunculan mole (detik)
MOLE_MAX_LIFESPAN = 2.00  # Batas atas durasi kemunculan mole (detik)
HIT_EFFECT_DURATION = 0.45  # Durasi animasi visual pukulan (detik)

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
