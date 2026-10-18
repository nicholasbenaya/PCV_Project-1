"""
Shinobi Jutsu Battle - Pure OpenCV 2D Game Engine & Renderer
Mata Kuliah: Praktikum Pengolahan Citra dan Visi Komputer (PCV)
"""

import cv2
import numpy as np
import time
import math
from typing import List, Tuple
from entities import Player, Boss, Projectile, Particle, DamageText


class GameEngine:
    def __init__(self, arena_w: int = 800, arena_h: int = 720):
        self.w = arena_w
        self.h = arena_h

        self.player = Player(x=90, y=360)
        self.boss = Boss(x=680, y=360)

        self.projectiles: List[Projectile] = []
        self.particles: List[Particle] = []
        self.damage_texts: List[DamageText] = []

        self.score = 0
        self.combo = 0
        self.state = "PLAYING"  # "PLAYING", "VICTORY", "GAME_OVER"
        self.last_update_time = time.time()
        self.screen_shake = 0

    def reset(self):
        self.player = Player(x=90, y=360)
        self.boss = Boss(x=680, y=360)
        self.projectiles.clear()
        self.particles.clear()
        self.damage_texts.clear()
        self.score = 0
        self.combo = 0
        self.state = "PLAYING"
        self.last_update_time = time.time()

    def cast_player_jutsu(self, seal_name: str):
        """Memicu peluncuran jurus berdasarkan segel tangan yang dikenali."""
        if self.state != "PLAYING":
            return

        px, py = self.player.x + 35, self.player.y
        if seal_name == "TIGER":
            # Katon: Fireball
            self.player.last_cast_name = "KATON: GOUKAKYUU!"
            self.projectiles.append(Projectile(px, py, 420.0, 0.0, "FIREBALL", is_player=True))
            self.combo += 1
        elif seal_name == "SNAKE":
            # Hyoton: Ice Barrier & Heal
            self.player.last_cast_name = "HYOTON: ICE BARRIER!"
            self.player.shield_timer = 5.0
            self.player.hp = min(self.player.max_hp, self.player.hp + 20)
            self.damage_texts.append(DamageText(self.player.x, self.player.y - 45, "+20 HP RECOVER!", (255, 240, 80)))
        elif seal_name == "CROSS":
            # Raiton: Lightning Strike
            self.player.last_cast_name = "RAITON: CHIDORI RAY!"
            self.projectiles.append(Projectile(px, py, 650.0, 0.0, "LIGHTNING", is_player=True))
            self.combo += 1
        elif seal_name == "RAM":
            # Fuuton: Rasenshuriken
            self.player.last_cast_name = "FUUTON: RASENSHURIKEN!"
            self.projectiles.append(Projectile(px, py, 320.0, 0.0, "RASENSHURIKEN", is_player=True))
            self.combo += 1
        elif seal_name == "OPEN_PALM":
            # Chakra Burst
            self.player.last_cast_name = "CHAKRA GATLING!"
            for dy in [-25, 0, 25]:
                self.projectiles.append(Projectile(px, py + dy, 520.0, dy * 1.5, "CHAKRA_ORB", is_player=True))
            self.combo += 1

    def update(self):
        now = time.time()
        dt = max(0.001, min(0.05, now - self.last_update_time))
        self.last_update_time = now

        if self.state != "PLAYING":
            return

        self.player.update(dt)
        boss_proj = self.boss.update(dt)
        if boss_proj:
            self.projectiles.append(boss_proj)

        # Update Proyektil
        surviving_proj = []
        for p in self.projectiles:
            if p.update(dt):
                hit = False
                # Cek tabrakan proyektil Pemain -> Boss
                if p.is_player and self.boss.hp > 0:
                    dist = math.hypot(p.x - self.boss.x, p.y - self.boss.y)
                    if dist < (p.radius + 55):
                        hit = True
                        dmg = p.damage + (self.combo * 2)
                        self.boss.hp -= dmg
                        self.score += dmg * 10
                        self.damage_texts.append(DamageText(self.boss.x - 30, self.boss.y - 40, f"-{dmg}!", (0, 180, 255)))
                        self.screen_shake = 5

                        # Partikel benturan
                        for _ in range(16):
                            spd = np.random.uniform(50, 180)
                            ang = np.random.uniform(0, 2 * math.pi)
                            self.particles.append(Particle(p.x, p.y, spd * math.cos(ang), spd * math.sin(ang), p.color, 4, 0.35))

                # Cek tabrakan proyektil Boss -> Pemain
                elif not p.is_player and self.player.hp > 0:
                    dist = math.hypot(p.x - self.player.x, p.y - self.player.y)
                    if dist < (p.radius + 35):
                        hit = True
                        if self.player.shield_timer > 0.0:
                            # Ditangkis oleh Ice Shield
                            self.damage_texts.append(DamageText(self.player.x, self.player.y - 40, "BLOCKED!", (255, 230, 80)))
                        else:
                            self.player.hp -= p.damage
                            self.combo = 0
                            self.damage_texts.append(DamageText(self.player.x, self.player.y - 40, f"-{p.damage}!", (50, 50, 255)))
                            self.screen_shake = 6

                        for _ in range(12):
                            spd = np.random.uniform(40, 140)
                            ang = np.random.uniform(0, 2 * math.pi)
                            self.particles.append(Particle(p.x, p.y, spd * math.cos(ang), spd * math.sin(ang), p.color, 3, 0.3))

                if not hit:
                    surviving_proj.append(p)
        self.projectiles = surviving_proj

        # Update Partikel & Angka Damage
        self.particles = [pt for pt in self.particles if pt.update(dt)]
        self.damage_texts = [dtx for dtx in self.damage_texts if dtx.update(dt)]

        # Cek Kondisi Menang / Kalah
        if self.boss.hp <= 0:
            self.state = "VICTORY"
        elif self.player.hp <= 0:
            self.state = "GAME_OVER"

    def render_arena(self) -> np.ndarray:
        arena = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # Latar belakang arena sci-fi/anime gradasi gelap
        for y in range(0, self.h, 40):
            cv2.line(arena, (0, y), (self.w, y), (18, 20, 28), 1)
        for x in range(0, self.w, 40):
            cv2.line(arena, (x, 0), (x, self.h), (18, 20, 28), 1)

        # Gambar Karakter
        self.player.draw(arena)
        self.boss.draw(arena)

        # Gambar Proyektil & Partikel
        for p in self.projectiles:
            p.draw(arena)
        for pt in self.particles:
            pt.draw(arena)
        for dtx in self.damage_texts:
            dtx.draw(arena)

        # --- HUD ATAS: Status Bar Boss & Player ---
        # 1. Bar Boss (Merah/Ungu)
        boss_bw = 360
        boss_bx = self.w - boss_bw - 30
        cv2.putText(arena, "BOSS: SUSANOO SHADOW DEMON", (boss_bx, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (200, 140, 255), 1, cv2.LINE_AA)
        cv2.rectangle(arena, (boss_bx, 42), (boss_bx + boss_bw, 58), (35, 20, 50), -1)
        boss_fill = max(0, int(boss_bw * (self.boss.hp / self.boss.max_hp)))
        if boss_fill > 0:
            cv2.rectangle(arena, (boss_bx, 42), (boss_bx + boss_fill, 58), (50, 40, 240), -1)
        cv2.rectangle(arena, (boss_bx, 42), (boss_bx + boss_bw, 58), (180, 180, 220), 1)

        # 2. Bar Player HP (Hijau) & Chakra (Biru)
        pl_bw = 250
        pl_bx = 30
        cv2.putText(arena, f"SHINOBI HP: {self.player.hp}/100", (pl_bx, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (100, 255, 140), 1, cv2.LINE_AA)
        cv2.rectangle(arena, (pl_bx, 42), (pl_bx + pl_bw, 54), (20, 40, 25), -1)
        pl_fill = max(0, int(pl_bw * (self.player.hp / self.player.max_hp)))
        if pl_fill > 0:
            cv2.rectangle(arena, (pl_bx, 42), (pl_bx + pl_fill, 54), (60, 220, 100), -1)
        cv2.rectangle(arena, (pl_bx, 42), (pl_bx + pl_bw, 54), (160, 220, 180), 1)

        # Chakra Bar
        cv2.rectangle(arena, (pl_bx, 60), (pl_bx + pl_bw, 68), (15, 25, 45), -1)
        ch_fill = int(pl_bw * (self.player.chakra / self.player.max_chakra))
        cv2.rectangle(arena, (pl_bx, 60), (pl_bx + ch_fill, 68), (255, 200, 60), -1)

        # Teks Skor & Combo
        cv2.putText(arena, f"SCORE: {self.score}", (320, 34), cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        if self.combo > 1:
            cv2.putText(arena, f"{self.combo}x COMBO!", (320, 62), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)

        # Banner Jurus yang Baru Dikeluarkan
        if self.player.last_cast_name:
            cv2.putText(arena, self.player.last_cast_name, (pl_bx, self.h - 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA)

        # Layar Game Over / Victory
        if self.state == "VICTORY":
            cv2.rectangle(arena, (120, 240), (self.w - 120, 480), (15, 30, 20), -1)
            cv2.rectangle(arena, (120, 240), (self.w - 120, 480), (0, 255, 180), 3)
            cv2.putText(arena, "★ SHINOBI VICTORY! ★", (210, 330), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 180), 2, cv2.LINE_AA)
            cv2.putText(arena, "Tekan 'R' pada keyboard untuk Main Lagi", (220, 400), cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 240, 230), 1, cv2.LINE_AA)
        elif self.state == "GAME_OVER":
            cv2.rectangle(arena, (120, 240), (self.w - 120, 480), (30, 15, 20), -1)
            cv2.rectangle(arena, (120, 240), (self.w - 120, 480), (50, 40, 255), 3)
            cv2.putText(arena, "☠ DEFEATED! ☠", (270, 330), cv2.FONT_HERSHEY_DUPLEX, 1.2, (50, 50, 255), 2, cv2.LINE_AA)
            cv2.putText(arena, "Tekan 'R' pada keyboard untuk Mengulang", (220, 400), cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 240), 1, cv2.LINE_AA)

        return arena
