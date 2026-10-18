"""
Entities for Shinobi Jutsu Battle (Custom OpenCV Game)
Mendefinisikan Player, Boss Monster, Proyektil Sihir, dan Sistem Partikel.
"""

import cv2
import numpy as np
import math
import time
from typing import List, Tuple, Optional


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


class DamageText:
    def __init__(self, x: int, y: int, text: str, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.lifetime = 1.0
        self.max_life = 1.0

    def update(self, dt: float) -> bool:
        self.y -= int(35 * dt)
        self.lifetime -= dt
        return self.lifetime > 0

    def draw(self, img: np.ndarray):
        cv2.putText(img, self.text, (int(self.x), int(self.y)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, self.color, 2, cv2.LINE_AA)


class Projectile:
    def __init__(self, x: float, y: float, vx: float, vy: float, ptype: str, is_player: bool):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ptype = ptype
        self.is_player = is_player
        self.radius = 16
        self.damage = 30
        self.color = (0, 0, 255)
        self.angle = 0.0

        if ptype == "FIREBALL":
            self.color = (30, 110, 255)  # Orange/Red
            self.damage = 45
            self.radius = 22
        elif ptype == "LIGHTNING":
            self.color = (255, 80, 220)  # Magenta/Lightning
            self.damage = 35
            self.radius = 14
        elif ptype == "ICE_SHARD":
            self.color = (255, 230, 80)  # Cyan
            self.damage = 25
            self.radius = 16
        elif ptype == "RASENSHURIKEN":
            self.color = (60, 240, 170)  # Emerald / Wind
            self.damage = 70
            self.radius = 32
        elif ptype == "CHAKRA_ORB":
            self.color = (80, 255, 255)  # Yellow
            self.damage = 20
            self.radius = 12
        elif ptype == "BOSS_DARK_FIRE":
            self.color = (80, 20, 200)   # Dark Purple
            self.damage = 20
            self.radius = 18
        elif ptype == "BOSS_SHADOW_BLAST":
            self.color = (40, 40, 240)   # Crimson Red
            self.damage = 30
            self.radius = 24

    def update(self, dt: float) -> bool:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += 8.0 * dt
        # Hapus proyektil jika keluar arena (asumsi arena lebar 800)
        return (0 <= self.x <= 850) and (0 <= self.y <= 720)

    def draw(self, img: np.ndarray):
        cx, cy = int(self.x), int(self.y)
        if self.ptype == "RASENSHURIKEN":
            cv2.circle(img, (cx, cy), self.radius, self.color, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), self.radius + 6, (255, 255, 255), 2, cv2.LINE_AA)
            # Bilah shuriken berputar
            for i in range(4):
                th = self.angle + (math.pi / 2) * i
                bx = int(cx + (self.radius + 15) * math.cos(th))
                by = int(cy + (self.radius + 15) * math.sin(th))
                cv2.line(img, (cx, cy), (bx, by), (255, 255, 255), 3, cv2.LINE_AA)
        elif self.ptype == "LIGHTNING":
            cv2.circle(img, (cx, cy), self.radius, self.color, -1, cv2.LINE_AA)
            # Ekor kilat zigzag
            cv2.line(img, (cx - 25, cy + np.random.randint(-8, 8)), (cx, cy), (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(img, (cx, cy), self.radius, self.color, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), int(self.radius * 0.5), (255, 255, 255), -1, cv2.LINE_AA)


class Player:
    def __init__(self, x: int = 80, y: int = 360):
        self.x = x
        self.y = y
        self.hp = 100
        self.max_hp = 100
        self.chakra = 100
        self.max_chakra = 100
        self.shield_timer = 0.0
        self.last_cast_name = ""

    def update(self, dt: float):
        if self.shield_timer > 0.0:
            self.shield_timer -= dt
        # Regenerasi Chakra perlahan
        self.chakra = min(self.max_chakra, self.chakra + 6.0 * dt)

    def draw(self, img: np.ndarray):
        # Karakter Shinobi (Silhouette dengan aura)
        cx, cy = self.x, self.y

        # Aura Pelindung Es (Jika Shield aktif)
        if self.shield_timer > 0.0:
            cv2.circle(img, (cx, cy), 58, (255, 220, 80), 3, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 65, (255, 255, 200), 1, cv2.LINE_AA)

        # Tubuh Shinobi (Stylized anime ninja)
        cv2.circle(img, (cx, cy - 22), 16, (40, 40, 50), -1)       # Kepala
        cv2.circle(img, (cx, cy - 22), 16, (0, 215, 255), 2)        # Ikat Kepala Konoha
        cv2.rectangle(img, (cx - 18, cy - 5), (cx + 18, cy + 40), (25, 30, 45), -1)  # Jubah
        cv2.line(img, (cx - 18, cy + 40), (cx - 12, cy + 65), (25, 30, 45), 5)      # Kaki Kiri
        cv2.line(img, (cx + 18, cy + 40), (cx + 12, cy + 65), (25, 30, 45), 5)      # Kaki Kanan


class Boss:
    def __init__(self, x: int = 680, y: int = 360):
        self.x = x
        self.y = y
        self.hp = 500
        self.max_hp = 500
        self.attack_timer = 2.0
        self.anim_phase = 0.0
        self.state = "IDLE"

    def update(self, dt: float) -> Optional[Projectile]:
        self.anim_phase += 2.5 * dt
        self.attack_timer -= dt

        # Gerakan mengambang naik-turun halus (Sine Wave)
        self.y = int(360 + 50 * math.sin(self.anim_phase))

        if self.attack_timer <= 0.0 and self.hp > 0:
            self.attack_timer = np.random.uniform(1.8, 3.2)
            # Tembakkan proyektil kegelapan ke arah pemain
            ptype = "BOSS_DARK_FIRE" if np.random.rand() > 0.4 else "BOSS_SHADOW_BLAST"
            spd = 240.0
            vx = -spd
            vy = np.random.uniform(-40, 40)
            return Projectile(self.x - 45, self.y, vx, vy, ptype, is_player=False)
        return None

    def draw(self, img: np.ndarray):
        if self.hp <= 0:
            return

        cx, cy = self.x, self.y
        # Tubuh monster iblis (Demon Aura)
        cv2.circle(img, (cx, cy), 65, (30, 15, 60), -1)
        cv2.circle(img, (cx, cy), 68, (80, 20, 180), 3, cv2.LINE_AA)

        # Tanduk Iblis Kiri & Kanan
        pts_horn_l = np.array([[cx - 30, cy - 40], [cx - 65, cy - 90], [cx - 15, cy - 50]], np.int32)
        pts_horn_r = np.array([[cx + 30, cy - 40], [cx + 65, cy - 90], [cx + 15, cy - 50]], np.int32)
        cv2.fillPoly(img, [pts_horn_l], (20, 20, 80))
        cv2.fillPoly(img, [pts_horn_r], (20, 20, 80))

        # Mata Merah Menyala
        cv2.circle(img, (cx - 24, cy - 12), 10, (30, 30, 240), -1)
        cv2.circle(img, (cx + 24, cy - 12), 10, (30, 30, 240), -1)
        cv2.circle(img, (cx - 24, cy - 12), 4, (255, 255, 255), -1)
        cv2.circle(img, (cx + 24, cy - 12), 4, (255, 255, 255), -1)

        # Taring Menyeringai
        cv2.ellipse(img, (cx, cy + 22), (28, 14), 0, 0, 180, (200, 200, 220), 4)
