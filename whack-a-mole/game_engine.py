"""Mesin logika game (State Machine, Deteksi Tabrakan, Siklus Hidup Permainan)."""

from enum import Enum
import random
from typing import List, Optional, Tuple
import numpy as np

try:
    from .config import (
        DEFAULT_FRAME_HEIGHT,
        DEFAULT_FRAME_WIDTH,
        DEFAULT_GAME_DURATION,
        HIT_DEBOUNCE_COOLDOWN,
        HIT_EFFECT_DURATION,
        MOLE_MAX_LIFESPAN,
        MOLE_MIN_LIFESPAN,
        ZONE_PRESETS,
    )
    from .entities import HitEffect, Hole, Mole, ScoreManager
    from .hand_tracker import HandTrackingResult
    from .ui_renderer import UIRenderer
except ImportError:
    from config import (
        DEFAULT_FRAME_HEIGHT,
        DEFAULT_FRAME_WIDTH,
        DEFAULT_GAME_DURATION,
        HIT_DEBOUNCE_COOLDOWN,
        HIT_EFFECT_DURATION,
        MOLE_MAX_LIFESPAN,
        MOLE_MIN_LIFESPAN,
        ZONE_PRESETS,
    )
    from entities import HitEffect, Hole, Mole, ScoreManager
    from hand_tracker import HandTrackingResult
    from ui_renderer import UIRenderer


class GameState(Enum):
    """Status tahapan permainan."""

    START = 0
    PLAYING = 1
    GAMEOVER = 2


class GameEngine:
    """Mesin utama pengontrol alur permainan Whack-A-Mole dengan sistem Dual-Section."""

    def __init__(
        self,
        frame_width: int = DEFAULT_FRAME_WIDTH,
        frame_height: int = DEFAULT_FRAME_HEIGHT,
        game_duration: float = DEFAULT_GAME_DURATION,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.game_duration = game_duration

        self.state: GameState = GameState.START
        self.score_manager = ScoreManager()
        self.renderer = UIRenderer()

        # Konfigurasi section aktif
        self.zone_idx = 0
        self.active_zone = ZONE_PRESETS[self.zone_idx]

        self.holes: List[Hole] = self._generate_holes(self.active_zone["play_rect"])
        self.mole: Optional[Mole] = None
        self.hit_effect: Optional[HitEffect] = None

        self.game_start_time: float = 0.0
        self.last_hit_time: float = 0.0

    def _generate_holes(self, play_rect: Tuple[int, int, int, int]) -> List[Hole]:
        """Menghasilkan 6 lubang target pada grid 2 baris x 3 kolom di dalam Playing Section."""
        px1, py1, px2, py2 = play_rect
        pw = px2 - px1
        ph = py2 - py1

        xs = [
            px1 + int(pw * 0.20),
            px1 + int(pw * 0.50),
            px1 + int(pw * 0.80),
        ]
        ys = [
            py1 + int(ph * 0.40),
            py1 + int(ph * 0.74),
        ]
        radius = int(min(pw, ph) * 0.12)

        holes: List[Hole] = []
        idx = 0
        for y in ys:
            for x in xs:
                holes.append(Hole(index=idx, x=x, y=y, radius=radius))
                idx += 1
        return holes

    def switch_zone(self) -> None:
        """Mengganti sisi tata letak Playing Section (Kanan / Kiri / Bawah)."""
        self.zone_idx = (self.zone_idx + 1) % len(ZONE_PRESETS)
        self.active_zone = ZONE_PRESETS[self.zone_idx]
        self.holes = self._generate_holes(self.active_zone["play_rect"])

    def update_frame_dimensions(self, width: int, height: int) -> None:
        """Memperbarui grid bila dimensi kanvas berubah."""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self.holes = self._generate_holes(self.active_zone["play_rect"])

    def start_game(self, current_time: float) -> None:
        """Memulai sesi permainan baru."""
        self.state = GameState.PLAYING
        self.score_manager.reset_session()
        self.game_start_time = current_time
        self.last_hit_time = 0.0
        self.hit_effect = None

        initial_hole = random.randint(0, len(self.holes) - 1)
        initial_lifespan = random.uniform(MOLE_MIN_LIFESPAN, MOLE_MAX_LIFESPAN)
        self.mole = Mole(
            initial_hole_idx=initial_hole,
            spawn_time=current_time,
            lifespan=initial_lifespan,
        )

    def restart_game(self, current_time: float) -> None:
        """Mereset dan langsung memulai kembali sesi permainan."""
        self.start_game(current_time)

    def update(self, hand_result: HandTrackingResult, current_time: float) -> None:
        """Memperbarui status game berdasarkan waktu dan posisi tangan pemain."""
        if self.state != GameState.PLAYING:
            return

        elapsed = current_time - self.game_start_time
        time_left = self.game_duration - elapsed

        # Periksa batas waktu permainan
        if time_left <= 0:
            self.state = GameState.GAMEOVER
            return

        if self.mole is None:
            return

        # Pindahkan mole jika durasi kemunculannya habis tanpa dipukul
        if self.mole.is_expired(current_time):
            self.mole.respawn(
                current_time=current_time,
                total_holes=len(self.holes),
                min_lifespan=MOLE_MIN_LIFESPAN,
                max_lifespan=MOLE_MAX_LIFESPAN,
            )

        # Logika deteksi tabrakan (hit detection)
        if hand_result.is_detected and hand_result.centroid is not None:
            active_hole = self.holes[self.mole.current_hole_idx]
            is_inside_hole = active_hole.contains(hand_result.centroid)
            is_cooldown_ready = (current_time - self.last_hit_time) >= HIT_DEBOUNCE_COOLDOWN

            if is_inside_hole and is_cooldown_ready:
                self.score_manager.record_hit()
                self.last_hit_time = current_time
                self.hit_effect = HitEffect(
                    position=active_hole.center,
                    spawn_time=current_time,
                    duration=HIT_EFFECT_DURATION,
                )

                # Langsung munculkan mole baru di lubang lain
                self.mole.respawn(
                    current_time=current_time,
                    total_holes=len(self.holes),
                    min_lifespan=MOLE_MIN_LIFESPAN,
                    max_lifespan=MOLE_MAX_LIFESPAN,
                )

    def render(
        self,
        frame: np.ndarray,
        hand_result: HandTrackingResult,
        current_time: float,
        show_mask_pip: bool = False,
    ) -> None:
        """Merender seluruh tampilan antarmuka game pada frame aktif."""
        # 1. Gambar pembagian visual section (Safe Zone & Playing Section)
        self.renderer.draw_sections_overlay(frame, self.active_zone)

        # 2. Render elemen game
        if self.state == GameState.START:
            self.renderer.draw_holes_and_mole(frame, self.holes, None, None, current_time)
            self.renderer.draw_hud(
                frame,
                self.score_manager,
                self.game_duration,
                hand_result.is_detected,
                self.active_zone["play_rect"],
            )
            self.renderer.draw_hand_cursor(frame, hand_result.centroid)
            self.renderer.draw_overlay_panel(
                frame,
                "WHACK-A-MOLE VIRTUAL (HD DUAL-SECTION)",
                [
                    "Sisi Kiri adalah Safe Zone: Kepala & tubuh Anda terblokir 100%.",
                    "Arahkan telapak tangan Anda ke Playing Section di sisi kanan.",
                    f"Satu sesi berlangsung selama {int(self.game_duration)} detik.",
                    "[H] Ganti Sisi Tangan (Kanan/Kiri/Bawah) | [M] Toggle Mask | [Q] Keluar",
                ],
                "Tekan [SPASI] untuk Mulai Bermain",
            )

        elif self.state == GameState.PLAYING:
            elapsed = current_time - self.game_start_time
            time_left = max(0.0, self.game_duration - elapsed)

            self.renderer.draw_holes_and_mole(
                frame, self.holes, self.mole, self.hit_effect, current_time
            )
            self.renderer.draw_hud(
                frame,
                self.score_manager,
                time_left,
                hand_result.is_detected,
                self.active_zone["play_rect"],
            )
            self.renderer.draw_hand_cursor(frame, hand_result.centroid)

        elif self.state == GameState.GAMEOVER:
            self.renderer.draw_holes_and_mole(frame, self.holes, None, None, current_time)
            self.renderer.draw_hud(
                frame,
                self.score_manager,
                0.0,
                hand_result.is_detected,
                self.active_zone["play_rect"],
            )
            self.renderer.draw_hand_cursor(frame, hand_result.centroid)
            self.renderer.draw_overlay_panel(
                frame,
                "GAME OVER",
                [
                    f"Skor Akhir Anda: {self.score_manager.score}",
                    f"Rekor Tertinggi: {self.score_manager.high_score}",
                    "Tangan Anda terdeteksi dengan sangat akurat tanpa gangguan wajah!",
                    "[Q] Keluar dari Permainan",
                ],
                "Tekan [R] untuk Main Lagi",
            )

        # Inset Picture-in-Picture untuk monitoring mask biner
        if show_mask_pip and hand_result.mask is not None:
            self.renderer.draw_mask_pip(frame, hand_result.mask)
