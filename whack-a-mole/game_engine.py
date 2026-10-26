"""Mesin logika game (State Machine, Hand Pose Interactivity, Survival Mode, Level Mode, Audio)."""

from enum import Enum
import random
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from .audio_system import AudioSystem
    from .config import (
        DEFAULT_FRAME_HEIGHT,
        DEFAULT_FRAME_WIDTH,
        GRID_PRESETS,
        HIT_DEBOUNCE_COOLDOWN,
        HIT_EFFECT_DURATION,
        LEVEL_CONFIGS,
        ZONE_PRESETS,
        LevelConfig,
    )
    from .entities import (
        HitEffect,
        Hole,
        InteractiveButton,
        Mole,
        ScoreManager,
        SurvivalTracker,
    )
    from .hand_tracker import HandTracker, HandTrackingResult
    from .save_manager import SaveManager
    from .ui_renderer import UIRenderer
except ImportError:
    from audio_system import AudioSystem
    from config import (
        DEFAULT_FRAME_HEIGHT,
        DEFAULT_FRAME_WIDTH,
        GRID_PRESETS,
        HIT_DEBOUNCE_COOLDOWN,
        HIT_EFFECT_DURATION,
        LEVEL_CONFIGS,
        ZONE_PRESETS,
        LevelConfig,
    )
    from entities import (
        HitEffect,
        Hole,
        InteractiveButton,
        Mole,
        ScoreManager,
        SurvivalTracker,
    )
    from hand_tracker import HandTracker, HandTrackingResult
    from save_manager import SaveManager
    from ui_renderer import UIRenderer


class GameState(Enum):
    """Status tahapan dan layar permainan."""

    MENU = 0
    LEVEL_SELECT = 1
    GRID_SELECT = 2
    PLAYING_SURVIVAL = 3
    PLAYING_LEVEL = 4
    LEVEL_CLEARED = 5
    GAMEOVER = 6
    CALIBRATION = 7


class GameEngine:
    """Mesin pengontrol alur permainan, logika tabrakan gestur FIST, dan integrasi audio."""

    def __init__(
        self,
        frame_width: int = DEFAULT_FRAME_WIDTH,
        frame_height: int = DEFAULT_FRAME_HEIGHT,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.save_manager = SaveManager()
        self.audio = AudioSystem()
        self.renderer = UIRenderer()
        self.score_manager = ScoreManager()

        self.state: GameState = GameState.MENU

        # Preset zona partisi (Playing Section vs Non-Playing Safe Zone)
        self.zone_index: int = 0
        self.active_zone: Dict = ZONE_PRESETS[self.zone_index]

        # Konfigurasi grid lubang
        self.selected_grid_idx = self.save_manager.selected_grid_index
        self.holes: List[Hole] = []
        self._rebuild_holes()

        # Daftar mole aktif (mendukung multi-mole)
        self.moles: List[Mole] = []
        self.hit_effects: List[HitEffect] = []
        self.last_hit_time: float = 0.0

        # Mode Survival
        self.survival_tracker: Optional[SurvivalTracker] = None

        # Mode Level
        self.current_level: int = 1
        self.level_start_time: float = 0.0

        # Koleksi tombol interaktif per layar
        self.menu_buttons: List[InteractiveButton] = []
        self.level_buttons: List[InteractiveButton] = []
        self.grid_buttons: List[InteractiveButton] = []
        self._rebuild_all_buttons()

    def switch_zone(self) -> str:
        """Mengganti orientasi Playing Section (Kanan, Kiri, atau Bawah)."""
        self.zone_index = (self.zone_index + 1) % len(ZONE_PRESETS)
        self.active_zone = ZONE_PRESETS[self.zone_index]
        self._rebuild_holes()
        self._rebuild_all_buttons()
        return self.active_zone["name"]

    def get_calibration_box(self) -> Tuple[int, int, int, int]:
        """Menghitung kotak area target kalibrasi di tengah Playing Section."""
        px1, py1, px2, py2 = self.active_zone["play_rect"]
        box_size = 220
        cx = (px1 + px2) // 2
        cy = (py1 + py2) // 2
        return (cx - box_size // 2, cy - box_size // 2, cx + box_size // 2, cy + box_size // 2)

    def _rebuild_holes(self) -> None:
        """Menghitung ulang posisi lubang di dalam Playing Section."""
        rows, cols = GRID_PRESETS[self.selected_grid_idx]
        px1, py1, px2, py2 = self.active_zone["play_rect"]
        pw = px2 - px1
        ph = py2 - py1

        top_margin = py1 + int(ph * 0.22)
        bottom_margin = py1 + int(ph * 0.88)
        left_margin = px1 + int(pw * 0.12)
        right_margin = px1 + int(pw * 0.88)

        step_x = (right_margin - left_margin) // (cols - 1) if cols > 1 else 0
        step_y = (bottom_margin - top_margin) // (rows - 1) if rows > 1 else 0

        base_radius = int(min(pw, ph) * (0.12 if max(rows, cols) <= 3 else 0.085))

        self.holes.clear()
        idx = 0
        for r in range(rows):
            y = top_margin + r * step_y if rows > 1 else (top_margin + bottom_margin) // 2
            for c in range(cols):
                x = left_margin + c * step_x if cols > 1 else (left_margin + right_margin) // 2
                self.holes.append(Hole(index=idx, x=x, y=y, radius=base_radius))
                idx += 1

    def _rebuild_all_buttons(self) -> None:
        """Membangun layout tombol interaktif di dalam Playing Section."""
        px1, py1, px2, py2 = self.active_zone["play_rect"]
        pw = px2 - px1

        # 1. Tombol Menu Utama (6 tombol)
        self.menu_buttons.clear()
        btn_w = min(int(pw * 0.78), 580)
        btn_h = 46
        bx1 = px1 + (pw - btn_w) // 2
        start_y = py1 + 205
        spacing = 58

        menu_items = [
            ("MODE SURVIVAL (Endless)", "action_survival", "[1]"),
            ("MODE LEVEL (1 - 10)", "action_level_select", "[2]"),
            ("PENGATURAN GRID", "action_grid_select", "[3]"),
            ("KALIBRASI WARNA KULIT", "action_calibration", "[4] / [C]"),
            ("PENGATURAN AUDIO (ON/OFF)", "action_toggle_audio", "[5] / [S]"),
            ("KELUAR PERMAINAN", "action_exit", "[6] / [Q]"),
        ]

        for i, (lbl, act, hotkey) in enumerate(menu_items):
            by = start_y + i * spacing
            self.menu_buttons.append(
                InteractiveButton(
                    rect=(bx1, by, bx1 + btn_w, by + btn_h),
                    label=lbl,
                    action_id=act,
                    hotkey_desc=hotkey,
                )
            )

        # 2. Tombol Level Select (2 baris x 5 kolom)
        self.level_buttons.clear()
        lvl_w = min(int(pw * 0.16), 120)
        lvl_h = 62
        margin_x = px1 + int(pw * 0.06)
        step_lx = (pw - 2 * int(pw * 0.06) - 5 * lvl_w) // 4 + lvl_w
        start_ly = py1 + 180
        step_ly = 90

        for lvl in range(1, 11):
            row = 0 if lvl <= 5 else 1
            col = (lvl - 1) % 5
            x1 = margin_x + col * step_lx
            y1 = start_ly + row * step_ly
            locked = lvl > self.save_manager.unlocked_level
            hotkey = f"[{lvl}]" if lvl < 10 else "[0]"
            self.level_buttons.append(
                InteractiveButton(
                    rect=(x1, y1, x1 + lvl_w, y1 + lvl_h),
                    label=f"LEVEL {lvl}",
                    action_id=f"level_{lvl}",
                    hotkey_desc=hotkey,
                    locked=locked,
                )
            )

        # Tombol kembali di Level Select
        back_w = 260
        back_h = 46
        back_x1 = px1 + (pw - back_w) // 2
        back_y1 = py2 - 80
        self.level_buttons.append(
            InteractiveButton(
                rect=(back_x1, back_y1, back_x1 + back_w, back_y1 + back_h),
                label="KEMBALI KE MENU",
                action_id="action_menu",
                hotkey_desc="[ESC]",
            )
        )

        # 3. Tombol Grid Select (4 Kartu)
        self.grid_buttons.clear()
        gw = min(int(pw * 0.78), 580)
        gh = 56
        gx1 = px1 + (pw - gw) // 2
        start_gy = py1 + 170
        step_gy = 72

        grid_labels = [
            ("GRID 2 x 2  (4 Lubang - Pemula)", "grid_0", "[1]"),
            ("GRID 2 x 3  (6 Lubang - Standar)", "grid_1", "[2]"),
            ("GRID 3 x 3  (9 Lubang - Menantang)", "grid_2", "[3]"),
            ("GRID 3 x 4  (12 Lubang - Luas)", "grid_3", "[4]"),
        ]

        for i, (lbl, act, hotkey) in enumerate(grid_labels):
            gy = start_gy + i * step_gy
            self.grid_buttons.append(
                InteractiveButton(
                    rect=(gx1, gy, gx1 + gw, gy + gh),
                    label=lbl,
                    action_id=act,
                    hotkey_desc=hotkey,
                )
            )

        # Tombol kembali di Grid Select
        self.grid_buttons.append(
            InteractiveButton(
                rect=(back_x1, back_y1, back_x1 + back_w, back_y1 + back_h),
                label="KEMBALI KE MENU",
                action_id="action_menu",
                hotkey_desc="[ESC]",
            )
        )

    def update_frame_dimensions(self, width: int, height: int) -> None:
        """Memperbarui dimensi grid dan tombol bila ukuran jendela berubah."""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self._rebuild_holes()
            self._rebuild_all_buttons()

    def start_survival_mode(self, current_time: float) -> None:
        """Memulai sesi Mode Survival."""
        self.state = GameState.PLAYING_SURVIVAL
        self.score_manager.reset_session()
        self.survival_tracker = SurvivalTracker(start_time=current_time)
        self.hit_effects.clear()
        self.last_hit_time = 0.0

        self.moles = [
            Mole(
                mole_id=1,
                initial_hole_idx=random.randint(0, len(self.holes) - 1),
                spawn_time=current_time,
                lifespan=1.85,
            )
        ]

        self.audio.play_sfx("click")
        self.audio.set_bgm_intensity(1)

    def start_level_mode(self, level_num: int, current_time: float) -> None:
        """Memulai sesi Mode Level Progression."""
        self.current_level = level_num
        self.state = GameState.PLAYING_LEVEL
        self.score_manager.reset_session()
        self.level_start_time = current_time
        self.hit_effects.clear()
        self.last_hit_time = 0.0

        cfg = LEVEL_CONFIGS[level_num]

        self.moles = []
        forbidden = []
        for i in range(cfg.simultaneous_moles):
            available = [idx for idx in range(len(self.holes)) if idx not in forbidden]
            chosen_hole = random.choice(available if available else list(range(len(self.holes))))
            forbidden.append(chosen_hole)
            self.moles.append(
                Mole(
                    mole_id=i + 1,
                    initial_hole_idx=chosen_hole,
                    spawn_time=current_time,
                    lifespan=random.uniform(cfg.mole_min_lifespan, cfg.mole_max_lifespan),
                )
            )

        self.audio.play_sfx("click")
        self.audio.set_bgm_intensity(cfg.bgm_intensity)

    def handle_button_action(self, action_id: str, current_time: float) -> Optional[str]:
        """Mengeksekusi perintah dari tombol UI."""
        self.audio.play_sfx("click")

        if action_id == "action_survival":
            self.start_survival_mode(current_time)
        elif action_id == "action_level_select":
            self.state = GameState.LEVEL_SELECT
        elif action_id == "action_grid_select":
            self.state = GameState.GRID_SELECT
        elif action_id == "action_calibration":
            self.state = GameState.CALIBRATION
        elif action_id == "action_toggle_audio":
            self.audio.toggle_sound()
        elif action_id == "action_exit":
            return "exit"
        elif action_id == "action_menu":
            self.audio.stop_bgm()
            self.state = GameState.MENU
        elif action_id.startswith("level_"):
            lvl = int(action_id.split("_")[1])
            if lvl <= self.save_manager.unlocked_level:
                self.start_level_mode(lvl, current_time)
        elif action_id.startswith("grid_"):
            idx = int(action_id.split("_")[1])
            self.selected_grid_idx = idx
            self.save_manager.set_grid_index(idx)
            self._rebuild_holes()
        return None

    def update(self, hand_result: HandTrackingResult, current_time: float) -> Optional[str]:
        """Memperbarui alur logika per frame dengan validasi Hand Pose (HIGH_FIVE move vs FIST whack)."""
        centroid = hand_result.centroid

        # 1. Update status tombol di layar menu
        if self.state in (GameState.MENU, GameState.LEVEL_SELECT, GameState.GRID_SELECT):
            buttons = (
                self.menu_buttons
                if self.state == GameState.MENU
                else (
                    self.level_buttons
                    if self.state == GameState.LEVEL_SELECT
                    else self.grid_buttons
                )
            )

            is_fist_ready = hand_result.is_fist and (current_time - self.last_hit_time) >= HIT_DEBOUNCE_COOLDOWN

            for btn in buttons:
                clicked = btn.update_interaction(
                    point=centroid,
                    is_fist_click=hand_result.is_fist_click,
                    is_fist=is_fist_ready,
                )
                if clicked:
                    self.last_hit_time = current_time
                    res = self.handle_button_action(btn.action_id, current_time)
                    if res == "exit":
                        return "exit"
                    break
            return None

        # 2. Kalibrasi tidak memproses gameplay
        if self.state == GameState.CALIBRATION:
            return None

        # 3. Update Mode Survival
        if self.state == GameState.PLAYING_SURVIVAL and self.survival_tracker is not None:
            current_tier = self.survival_tracker.get_intensity_tier(current_time)
            self.audio.set_bgm_intensity(current_tier)

            target_moles = self.survival_tracker.get_simultaneous_moles_count(
                current_time, len(self.holes)
            )
            speed_min, speed_max = self.survival_tracker.get_mole_speed(current_time)

            while len(self.moles) < target_moles:
                forbidden = [m.current_hole_idx for m in self.moles]
                avail = [i for i in range(len(self.holes)) if i not in forbidden]
                chosen = random.choice(avail if avail else list(range(len(self.holes))))
                self.moles.append(
                    Mole(
                        mole_id=len(self.moles) + 1,
                        initial_hole_idx=chosen,
                        spawn_time=current_time,
                        lifespan=random.uniform(speed_min, speed_max),
                    )
                )

            forbidden_holes = [m.current_hole_idx for m in self.moles]
            for mole in self.moles:
                if mole.is_expired(current_time):
                    self.audio.play_sfx("miss")
                    self.score_manager.record_miss()
                    is_alive = self.survival_tracker.lose_life()

                    if not is_alive:
                        self.audio.stop_bgm()
                        self.audio.play_sfx("game_over")
                        self.save_manager.record_survival_score(self.score_manager.score)
                        self.state = GameState.GAMEOVER
                        return None

                    mole.respawn(
                        current_time,
                        forbidden_holes,
                        len(self.holes),
                        speed_min,
                        speed_max,
                    )

            # AKSI PUKUL (WHACK!): HANYA dieksekusi jika tangan berpose FIST (Kepal)!
            # Jika tangan berpose HIGH_FIVE (Open Palm), kursor hanya membidik / move.
            if hand_result.is_detected and hand_result.is_fist and centroid is not None:
                can_hit = (current_time - self.last_hit_time) >= HIT_DEBOUNCE_COOLDOWN
                if can_hit:
                    for mole in self.moles:
                        hole = self.holes[mole.current_hole_idx]
                        if hole.contains(centroid):
                            self.audio.play_sfx("whack")
                            self.score_manager.record_hit()
                            self.last_hit_time = current_time

                            self.hit_effects.append(
                                HitEffect(
                                    position=hole.center,
                                    spawn_time=current_time,
                                    duration=HIT_EFFECT_DURATION,
                                    combo=self.score_manager.combo,
                                )
                            )

                            forbidden = [m.current_hole_idx for m in self.moles]
                            mole.respawn(
                                current_time,
                                forbidden,
                                len(self.holes),
                                speed_min,
                                speed_max,
                            )
                            break

        # 4. Update Mode Level Progression
        elif self.state == GameState.PLAYING_LEVEL:
            cfg = LEVEL_CONFIGS[self.current_level]
            elapsed = current_time - self.level_start_time
            time_left = cfg.duration_seconds - elapsed

            if self.score_manager.hits_count >= cfg.target_hits:
                self.audio.stop_bgm()
                self.audio.play_sfx("level_clear")
                self.save_manager.record_level_score(self.current_level, self.score_manager.score)
                self.save_manager.unlock_next_level(self.current_level)
                self._rebuild_all_buttons()
                self.state = GameState.LEVEL_CLEARED
                return None

            if time_left <= 0:
                self.audio.stop_bgm()
                self.audio.play_sfx("game_over")
                self.state = GameState.GAMEOVER
                return None

            forbidden = [m.current_hole_idx for m in self.moles]
            for mole in self.moles:
                if mole.is_expired(current_time):
                    self.audio.play_sfx("miss")
                    self.score_manager.record_miss()
                    mole.respawn(
                        current_time,
                        forbidden,
                        len(self.holes),
                        cfg.mole_min_lifespan,
                        cfg.mole_max_lifespan,
                    )

            # AKSI PUKUL (WHACK!): Wajib berpose FIST
            if hand_result.is_detected and hand_result.is_fist and centroid is not None:
                can_hit = (current_time - self.last_hit_time) >= HIT_DEBOUNCE_COOLDOWN
                if can_hit:
                    for mole in self.moles:
                        hole = self.holes[mole.current_hole_idx]
                        if hole.contains(centroid):
                            self.audio.play_sfx("whack")
                            self.score_manager.record_hit()
                            self.last_hit_time = current_time

                            self.hit_effects.append(
                                HitEffect(
                                    position=hole.center,
                                    spawn_time=current_time,
                                    duration=HIT_EFFECT_DURATION,
                                    combo=self.score_manager.combo,
                                )
                            )

                            forbidden = [m.current_hole_idx for m in self.moles]
                            mole.respawn(
                                current_time,
                                forbidden,
                                len(self.holes),
                                cfg.mole_min_lifespan,
                                cfg.mole_max_lifespan,
                            )
                            break

        self.hit_effects = [e for e in self.hit_effects if e.is_alive(current_time)]
        return None

    def render(
        self,
        frame: np.ndarray,
        hand_result: HandTrackingResult,
        hand_tracker: HandTracker,
        current_time: float,
        show_pip_diagnostic: bool = False,
    ) -> None:
        """Merender seluruh lapisan antarmuka grafis ke frame kamera."""
        # 1. Gambar partisi Dual-Section (Playing Section vs Non-Playing Safe Zone)
        self.renderer.draw_sections_overlay(frame, self.active_zone)

        play_rect = self.active_zone["play_rect"]

        # 2. Render elemen spesifik tiap GameState
        if self.state == GameState.MENU:
            rows, cols = GRID_PRESETS[self.selected_grid_idx]
            grid_desc = f"{rows}x{cols} ({rows * cols} Lubang)"
            self.renderer.draw_main_menu(
                frame,
                self.menu_buttons,
                grid_desc=grid_desc,
                audio_enabled=self.audio.enabled,
                is_calibrated=hand_tracker.is_calibrated,
                survival_high_score=self.save_manager.survival_high_score,
                unlocked_level=self.save_manager.unlocked_level,
                preset_name=hand_result.preset_name,
                current_pose=hand_result.pose,
                play_rect=play_rect,
                current_time=current_time,
            )

        elif self.state == GameState.CALIBRATION:
            box_rect = self.get_calibration_box()
            self.renderer.draw_calibration_view(
                frame,
                box_rect=box_rect,
                calib_info=hand_tracker.calib_info,
                is_calibrated=hand_tracker.is_calibrated,
            )

        elif self.state == GameState.LEVEL_SELECT:
            self.renderer.draw_level_select(
                frame,
                self.level_buttons,
                unlocked_level=self.save_manager.unlocked_level,
                high_scores=self.save_manager.level_high_scores,
                play_rect=play_rect,
                current_time=current_time,
            )

        elif self.state == GameState.GRID_SELECT:
            self.renderer.draw_grid_select(
                frame,
                self.grid_buttons,
                selected_index=self.selected_grid_idx,
                play_rect=play_rect,
                current_time=current_time,
            )

        elif self.state == GameState.PLAYING_SURVIVAL and self.survival_tracker is not None:
            self.renderer.draw_holes_and_moles(
                frame, self.holes, self.moles, self.hit_effects, current_time
            )
            self.renderer.draw_survival_hud(
                frame,
                self.score_manager,
                self.survival_tracker,
                self.save_manager.survival_high_score,
                hand_result.is_detected,
                current_pose=hand_result.pose,
                play_rect=play_rect,
                current_time=current_time,
            )

        elif self.state == GameState.PLAYING_LEVEL:
            cfg = LEVEL_CONFIGS[self.current_level]
            elapsed = current_time - self.level_start_time
            time_left = max(0.0, cfg.duration_seconds - elapsed)
            best = self.save_manager.level_high_scores.get(str(self.current_level), 0)

            self.renderer.draw_holes_and_moles(
                frame, self.holes, self.moles, self.hit_effects, current_time
            )
            self.renderer.draw_level_hud(
                frame,
                cfg,
                self.score_manager,
                time_left,
                best,
                hand_result.is_detected,
                current_pose=hand_result.pose,
                play_rect=play_rect,
            )

        elif self.state == GameState.LEVEL_CLEARED:
            self.renderer.draw_holes_and_moles(frame, self.holes, [], [], current_time)
            cfg = LEVEL_CONFIGS[self.current_level]
            next_unlocked = self.current_level < 10

            lines = [
                f"Level {self.current_level}: {cfg.title} Berhasil Dituntaskan!",
                f"Skor Diperoleh: {self.score_manager.score}",
                f"Kombo Tertinggi: x{self.score_manager.max_combo}",
                f"Status: Level {self.current_level + 1} Berhasil Dibuka!" if next_unlocked else "SELAMAT! ANDA TELAH MENUNTASKAN SELURUH LEVEL!",
            ]

            action_prompt = "Tekan [SPASI] untuk Level Selanjutnya" if next_unlocked else "Tekan [SPASI] ke Menu Utama"
            self.renderer.draw_modal_dialog(
                frame,
                "LEVEL TUNTAS! (CLEARED)",
                (0, 255, 120),
                lines,
                action_prompt,
                secondary_prompt="[R] Main Ulang Level Ini  |  [M] / [ESC] Menu Utama",
            )

        elif self.state == GameState.GAMEOVER:
            self.renderer.draw_holes_and_moles(frame, self.holes, [], [], current_time)

            if self.survival_tracker is not None:
                elapsed = self.survival_tracker.get_elapsed(current_time)
                lines = [
                    f"Skor Akhir Survival: {self.score_manager.score}",
                    f"Waktu Bertahan: {elapsed:.1f} detik",
                    f"Kombo Tertinggi: x{self.score_manager.max_combo}",
                    f"Rekor Tertinggi: {self.save_manager.survival_high_score}",
                ]
            else:
                cfg = LEVEL_CONFIGS[self.current_level]
                lines = [
                    f"Gagal Menuntaskan Level {self.current_level}: {cfg.title}",
                    f"Pukulan Mole: {self.score_manager.hits_count} / {cfg.target_hits}",
                    f"Waktu Bermain Habis!",
                ]

            self.renderer.draw_modal_dialog(
                frame,
                "GAME OVER",
                (0, 0, 240),
                lines,
                "Tekan [R] untuk Coba Lagi",
                secondary_prompt="[M] / [ESC] Kembali ke Menu Utama",
            )

        # 3. Gambar kursor tangan cerdas (membedakan HIGH_FIVE dan FIST)
        self.renderer.draw_hand_cursor(frame, hand_result)

        # 4. Jendela PiP Diagnostik Fitur Bentuk PCV (Convex Hull, Defek, Solidity, Pose)
        if show_pip_diagnostic and hand_result.mask is not None:
            self.renderer.draw_diagnostic_pip(frame, hand_result)
