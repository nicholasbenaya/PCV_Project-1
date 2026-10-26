"""
Visual Effects (VFX) & Procedural Magic Circle Overlay via Pure OpenCV
Memberikan efek visual anime/magis pada feed kamera saat tangan merapal segel:
- Lingkaran sihir berputar (Runic Circle)
- Partikel elemen mengambang (Quas Es, Wex Petir, Exort Api)
- Indikator HUD Dota 2 (Slot Orb, Slot Spell D & F, Cooldown Meter)
"""

import cv2
import numpy as np
import math
import time
from typing import List, Tuple, Dict, Any


class Particle:
    def __init__(self, x: float, y: float, vx: float, vy: float, color: Tuple[int, int, int], radius: float, lifetime: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.radius = radius
        self.lifetime = lifetime
        self.max_life = lifetime

    def update(self, dt: float) -> bool:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.lifetime -= dt
        return self.lifetime > 0

    def draw(self, img: np.ndarray):
        alpha = max(0.0, self.lifetime / self.max_life)
        r = max(1, int(self.radius * alpha))
        cv2.circle(img, (int(self.x), int(self.y)), r, self.color, -1)


class VFXOverlay:
    def __init__(self, minimalist_mode: bool = True):
        self.angle: float = 0.0
        self.particles: List[Particle] = []
        self.last_time: float = time.time()
        self.spell_banner: str = ""
        self.spell_banner_timer: float = 0.0
        self.minimalist_mode: bool = minimalist_mode

    def trigger_spell_burst(self, cx: int, cy: int, spell_name: str, color: Tuple[int, int, int]):
        """Memicu banner pengumuman mantra yang bersih."""
        self.spell_banner = spell_name
        self.spell_banner_timer = 0.8  # Durasi banner ringkas (0.8 detik)

        if not self.minimalist_mode:
            for _ in range(35):
                spd = np.random.uniform(40, 260)
                rad = np.random.uniform(0, 2 * math.pi)
                vx = spd * math.cos(rad)
                vy = spd * math.sin(rad)
                p_color = (
                    int(np.clip(color[0] + np.random.randint(-30, 30), 0, 255)),
                    int(np.clip(color[1] + np.random.randint(-30, 30), 0, 255)),
                    int(np.clip(color[2] + np.random.randint(-30, 30), 0, 255)),
                )
                self.particles.append(Particle(cx, cy, vx, vy, p_color, np.random.uniform(4, 9), np.random.uniform(0.4, 0.9)))

    def draw_magic_circle(self, canvas: np.ndarray, cx: int, cy: int, radius: int, color: Tuple[int, int, int]):
        """Menggambar lingkaran sihir geometris berlapis yang berputar secara prosedural."""
        # Lingkaran luar berigi
        cv2.circle(canvas, (cx, cy), radius, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), radius - 12, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), int(radius * 0.55), color, 1, cv2.LINE_AA)

        # Runic tick marks
        num_ticks = 12
        for i in range(num_ticks):
            theta = self.angle + (2 * math.pi / num_ticks) * i
            x1 = int(cx + (radius - 12) * math.cos(theta))
            y1 = int(cy + (radius - 12) * math.sin(theta))
            x2 = int(cx + radius * math.cos(theta))
            y2 = int(cy + radius * math.sin(theta))
            cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # Bintang segi-enam (Hexagram) berputar di dalam
        pts1 = []
        pts2 = []
        for i in range(3):
            th1 = -self.angle + (2 * math.pi / 3) * i
            th2 = -self.angle + (2 * math.pi / 3) * i + math.pi / 3
            pts1.append([int(cx + (radius * 0.55) * math.cos(th1)), int(cy + (radius * 0.55) * math.sin(th1))])
            pts2.append([int(cx + (radius * 0.55) * math.cos(th2)), int(cy + (radius * 0.55) * math.sin(th2))])

        cv2.polylines(canvas, [np.array(pts1, dtype=np.int32)], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [np.array(pts2, dtype=np.int32)], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    def draw_elemental_sparks(self, canvas: np.ndarray, cx: int, cy: int, seal_name: str):
        """Memancarkan partikel sesuai elemen gestur yang sedang dipegang."""
        if seal_name in ["3_FINGERS", "TIGER"]:  # Exort - Api (Merah / Oranye)
            color = (30, 110, 255)
            vx = np.random.uniform(-40, 40)
            vy = np.random.uniform(-140, -40)  # Membara ke atas
            self.particles.append(Particle(cx + np.random.randint(-25, 25), cy + np.random.randint(-15, 15), vx, vy, color, 5, 0.45))
        elif seal_name in ["1_FINGER", "SNAKE"]:  # Quas - Es (Cyan / Biru Es)
            color = (255, 230, 80)
            vx = np.random.uniform(-60, 60)
            vy = np.random.uniform(-60, 60)
            self.particles.append(Particle(cx + np.random.randint(-20, 20), cy + np.random.randint(-20, 20), vx, vy, color, 4, 0.4))
        elif seal_name in ["2_FINGERS", "CROSS"]:  # Wex - Petir (Ungu / Pink)
            color = (255, 60, 200)
            vx = np.random.uniform(-120, 120)
            vy = np.random.uniform(-120, 120)
            self.particles.append(Particle(cx + np.random.randint(-30, 30), cy + np.random.randint(-30, 30), vx, vy, color, 3, 0.25))
        elif seal_name in ["4_FINGERS", "RAM"]:  # Invoke - Emerald / Gold
            color = (60, 240, 170)
            vx = np.random.uniform(-80, 80)
            vy = np.random.uniform(-80, 80)
            self.particles.append(Particle(cx + np.random.randint(-25, 25), cy + np.random.randint(-25, 25), vx, vy, color, 4, 0.35))
        elif seal_name in ["5_FINGERS", "OPEN_PALM"]:  # Quick Cast D - Gold
            color = (80, 255, 255)
            vx = np.random.uniform(-100, 100)
            vy = np.random.uniform(-100, 100)
            self.particles.append(Particle(cx + np.random.randint(-30, 30), cy + np.random.randint(-30, 30), vx, vy, color, 5, 0.3))
        elif seal_name in ["FIST", "DUAL_PALM"]:  # Quick Cast F - Violet Burst
            color = (255, 120, 200)
            vx = np.random.uniform(-100, 100)
            vy = np.random.uniform(-100, 100)
            self.particles.append(Particle(cx + np.random.randint(-30, 30), cy + np.random.randint(-30, 30), vx, vy, color, 5, 0.3))

    def update_and_render(self, frame: np.ndarray, info: Dict[str, Any], progress: float):
        """Update dan render seluruh overlay partikel, lingkaran sihir, dan HUD."""
        now = time.time()
        dt = max(0.001, min(0.1, now - self.last_time))
        self.last_time = now

        self.angle += 1.2 * dt  # Kecepatan rotasi lingkaran sihir

        overlay = frame.copy()
        seal = info.get("seal", "NONE")
        centroid = info.get("features", {}).get("centroid", None)

        # 1. Gambar Lingkaran Sihir & Partikel di tangan pemain
        if centroid is not None and seal not in ["NONE", "UNKNOWN"]:
            cx, cy = centroid
            color_map = {
                "1_FINGER": (255, 220, 70),   # Quas (Ice Cyan)
                "2_FINGERS": (255, 80, 220),  # Wex (Lightning Magenta)
                "3_FINGERS": (40, 130, 255),  # Exort (Fire Orange/Red)
                "4_FINGERS": (60, 240, 180),  # Invoke (Gold/Emerald)
                "5_FINGERS": (80, 255, 255),  # Cast D (Yellow Beam)
                "FIST": (255, 120, 200),      # Cast F (Violet Burst)
                "TIGER": (40, 130, 255),
                "SNAKE": (255, 220, 70),
                "CROSS": (255, 80, 220),
                "RAM": (60, 240, 180),
                "OPEN_PALM": (80, 255, 255),
                "DUAL_PALM": (255, 200, 255)
            }
            seal_color = color_map.get(seal, (200, 200, 200))
            radius = 45

            if not self.minimalist_mode:
                self.draw_magic_circle(overlay, cx, cy, int(80 + 35 * progress), seal_color)
                self.draw_elemental_sparks(overlay, cx, cy, seal)
            else:
                # Indikator target sederhana & elegan
                cv2.circle(overlay, (cx, cy), radius, seal_color, 2, cv2.LINE_AA)
                cv2.circle(overlay, (cx, cy), 4, seal_color, -1, cv2.LINE_AA)

            # Circular Progress Meter saat menahan pose
            if progress > 0.0:
                axes = (radius + 12, radius + 12)
                cv2.ellipse(overlay, (cx, cy), axes, 0, -90, -90 + int(360 * progress), (0, 255, 230), 4, cv2.LINE_AA)

        # Update Partikel (hanya jika bukan mode minimalis)
        if not self.minimalist_mode:
            self.particles = [p for p in self.particles if p.update(dt)]
            for p in self.particles:
                p.draw(overlay)
            cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        else:
            # Pada mode minimalis, langsung timpa tanpa blend berat untuk frame rate 60 FPS maksimal
            cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)

        # 2. Render Banner Pengumuman Jurus (Jika baru saja terpicu)
        if self.spell_banner_timer > 0.0:
            self.spell_banner_timer -= dt
            h, w = frame.shape[:2]
            text = f"★ {self.spell_banner} ★"
            font = cv2.FONT_HERSHEY_DUPLEX
            t_size = cv2.getTextSize(text, font, 1.0, 2)[0]
            tx = (w - t_size[0]) // 2
            ty = int(h * 0.28)

            # Background Box bersih semi-transparan
            cv2.rectangle(frame, (tx - 20, ty - 32), (tx + t_size[0] + 20, ty + 14), (20, 24, 32), -1)
            cv2.rectangle(frame, (tx - 20, ty - 32), (tx + t_size[0] + 20, ty + 14), (0, 230, 255), 2)
            cv2.putText(frame, text, (tx, ty), font, 1.0, (0, 240, 255), 2, cv2.LINE_AA)
