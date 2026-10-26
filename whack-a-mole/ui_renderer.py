"""Modul visualisasi OpenCV lengkap (HD 1280x720) dengan Dual-Section dan Kalibrasi."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    from .config import (
        COLOR_BUTTON_BG,
        COLOR_BUTTON_BORDER,
        COLOR_BUTTON_HOVER,
        COLOR_HEART_RED,
        COLOR_HIT_FLASH,
        COLOR_HOLE_INNER,
        COLOR_HOLE_OUTER,
        COLOR_LOCKED_GRAY,
        COLOR_MOLE_BODY,
        COLOR_PRIMARY_CYAN,
        COLOR_RETICLE,
        COLOR_TEXT_DARK,
        COLOR_TEXT_WHITE,
        COLOR_TIMER_CRITICAL,
        COLOR_TIMER_NORMAL,
        GRID_PRESETS,
        LEVEL_CONFIGS,
        LevelConfig,
    )
    from .entities import HitEffect, Hole, InteractiveButton, Mole, ScoreManager, SurvivalTracker
except ImportError:
    from config import (
        COLOR_BUTTON_BG,
        COLOR_BUTTON_BORDER,
        COLOR_BUTTON_HOVER,
        COLOR_HEART_RED,
        COLOR_HIT_FLASH,
        COLOR_HOLE_INNER,
        COLOR_HOLE_OUTER,
        COLOR_LOCKED_GRAY,
        COLOR_MOLE_BODY,
        COLOR_PRIMARY_CYAN,
        COLOR_RETICLE,
        COLOR_TEXT_DARK,
        COLOR_TEXT_WHITE,
        COLOR_TIMER_CRITICAL,
        COLOR_TIMER_NORMAL,
        GRID_PRESETS,
        LEVEL_CONFIGS,
        LevelConfig,
    )
    from entities import HitEffect, Hole, InteractiveButton, Mole, ScoreManager, SurvivalTracker


class UIRenderer:
    """Renderer antarmuka grafis untuk seluruh state permainan Whack-A-Mole pada resolusi HD."""

    @staticmethod
    def draw_text_with_outline(
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        scale: float = 0.7,
        color: Tuple[int, int, int] = COLOR_TEXT_WHITE,
        outline_color: Tuple[int, int, int] = COLOR_TEXT_DARK,
        thickness: int = 2,
    ) -> None:
        """Menggambar teks dengan outline gelap agar kontras di latar belakang apa pun."""
        x, y = pos
        cv2.putText(
            img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            outline_color,
            thickness + 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def draw_sections_overlay(self, frame: np.ndarray, zone_cfg: Dict) -> None:
        """Merender batas visual Playing Section dan Non-Playing Section (Safe Zone)."""
        sx1, sy1, sx2, sy2 = zone_cfg["safe_rect"]
        px1, py1, px2, py2 = zone_cfg["play_rect"]

        # Redupkan Non-Playing Safe Zone (tempat wajah/tubuh berada)
        safe_region = frame[sy1:sy2, sx1:sx2]
        dimmed_safe = (safe_region * 0.42).astype(np.uint8)
        frame[sy1:sy2, sx1:sx2] = dimmed_safe

        # Garis pembatas neon
        p_start, p_end = zone_cfg["split_line"]
        cv2.line(frame, p_start, p_end, COLOR_PRIMARY_CYAN, 3, cv2.LINE_AA)

        # Label pada Non-Playing Section
        self.draw_text_with_outline(
            frame,
            "NON-PLAYING SECTION",
            (sx1 + 25, 42),
            scale=0.65,
            color=(180, 190, 210),
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            "[SAFE ZONE] Wajah & Tubuh Terblokir 100%",
            (sx1 + 25, 70),
            scale=0.48,
            color=(0, 255, 120),
            thickness=1,
        )
        self.draw_text_with_outline(
            frame,
            "Kepala Anda berada di sini dengan aman tanpa mengganggu kursor",
            (sx1 + 25, 96),
            scale=0.40,
            color=(160, 160, 160),
            thickness=1,
        )

        # Label pada Playing Section
        self.draw_text_with_outline(
            frame,
            "PLAYING SECTION [ZONA TANGAN AKTIF]",
            (px1 + 25, 42),
            scale=0.72,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            f"Tata Letak: {zone_cfg['name']}  |  [H] Ganti Sisi Tangan",
            (px1 + 25, 70),
            scale=0.48,
            color=(255, 220, 100),
            thickness=1,
        )

    def draw_button(
        self,
        frame: np.ndarray,
        button: InteractiveButton,
        current_time: float,
    ) -> None:
        """Merender tombol interaktif dengan efek hover dan meteran pengisian virtual touch."""
        x1, y1, x2, y2 = button.rect
        w, h = x2 - x1, y2 - y1

        if button.locked:
            bg_color = (25, 25, 25)
            border_color = COLOR_LOCKED_GRAY
            text_color = COLOR_LOCKED_GRAY
        elif button.is_hovered:
            bg_color = COLOR_BUTTON_HOVER
            border_color = (0, 255, 255)
            text_color = (0, 255, 255)
        else:
            bg_color = COLOR_BUTTON_BG
            border_color = COLOR_BUTTON_BORDER
            text_color = COLOR_TEXT_WHITE

        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

        label_text = button.label if not button.locked else f"{button.label} [TERKUNCI]"
        t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0]
        tx = x1 + (w - t_size[0]) // 2
        ty = y1 + (h + t_size[1]) // 2 - 2

        self.draw_text_with_outline(
            frame,
            label_text,
            (tx, ty),
            scale=0.58,
            color=text_color,
            thickness=2,
        )

        if button.hotkey_desc:
            self.draw_text_with_outline(
                frame,
                button.hotkey_desc,
                (x1 + 10, y1 + 18),
                scale=0.42,
                color=(170, 170, 170),
                thickness=1,
            )

        ratio = button.get_hover_ratio(current_time)
        if ratio > 0.0:
            fill_w = int(w * ratio)
            cv2.rectangle(frame, (x1, y2 - 4), (x1 + fill_w, y2), (0, 255, 120), -1)

    def draw_main_menu(
        self,
        frame: np.ndarray,
        buttons: List[InteractiveButton],
        grid_desc: str,
        audio_enabled: bool,
        is_calibrated: bool,
        survival_high_score: int,
        unlocked_level: int,
        play_rect: Tuple[int, int, int, int],
        current_time: float,
    ) -> None:
        """Merender tampilan Menu Utama lengkap dengan opsi mode dan navigasi."""
        px1, py1, px2, py2 = play_rect
        pw = px2 - px1

        # Judul Menu di dalam Playing Section
        title_x = px1 + 35
        self.draw_text_with_outline(
            frame,
            "WHACK-A-MOLE VIRTUAL",
            (title_x, 118),
            scale=1.05,
            color=COLOR_PRIMARY_CYAN,
            thickness=3,
        )

        # Status badge ringkas
        audio_str = "ON" if audio_enabled else "OFF"
        calib_str = "TERKALIBRASI" if is_calibrated else "DEFAULT"
        calib_color = (0, 255, 120) if is_calibrated else (255, 220, 100)

        badges_line = f"Grid: {grid_desc}  |  Audio: {audio_str}  |  Kalibrasi: {calib_str}"
        self.draw_text_with_outline(
            frame,
            badges_line,
            (title_x, 146),
            scale=0.46,
            color=calib_color,
            thickness=1,
        )

        stats_line = f"Rekor Survival: {survival_high_score}  |  Level Terbuka: {unlocked_level}/10"
        self.draw_text_with_outline(
            frame,
            stats_line,
            (title_x, 170),
            scale=0.46,
            color=(180, 220, 255),
            thickness=1,
        )

        # Render seluruh tombol menu
        for btn in buttons:
            self.draw_button(frame, btn, current_time)

        # Petunjuk interaksi bawah
        hint = "Arahkan tangan ke tombol untuk memilih, atau tekan angka [1-6] | [C] Kalibrasi | [H] Ganti Sisi"
        self.draw_text_with_outline(
            frame,
            hint,
            (px1 + 25, py2 - 25),
            scale=0.44,
            color=(160, 240, 160),
            thickness=1,
        )

    def draw_calibration_view(
        self,
        frame: np.ndarray,
        box_rect: Tuple[int, int, int, int],
        calib_info: Optional[Dict],
        is_calibrated: bool,
    ) -> None:
        """Merender panduan kalibrasi warna kulit interaktif."""
        x1, y1, x2, y2 = box_rect
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Kotak target kalibrasi warna hijau
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 3)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 120), -1)

        # Garis bidik sudut (corner brackets)
        arm = 18
        cv2.line(frame, (x1, y1), (x1 + arm, y1), (0, 255, 120), 4)
        cv2.line(frame, (x1, y1), (x1, y1 + arm), (0, 255, 120), 4)
        cv2.line(frame, (x2, y1), (x2 - arm, y1), (0, 255, 120), 4)
        cv2.line(frame, (x2, y1), (x2, y1 + arm), (0, 255, 120), 4)
        cv2.line(frame, (x1, y2), (x1 + arm, y2), (0, 255, 120), 4)
        cv2.line(frame, (x1, y2), (x1, y2 - arm), (0, 255, 120), 4)
        cv2.line(frame, (x2, y2), (x2 - arm, y2), (0, 255, 120), 4)
        cv2.line(frame, (x2, y2), (x2, y2 - arm), (0, 255, 120), 4)

        # Teks petunjuk langkah kalibrasi
        self.draw_text_with_outline(
            frame,
            "MODE KALIBRASI WARNA KULIT TANGAN",
            (x1 - 60, y1 - 35),
            scale=0.72,
            color=(0, 255, 120),
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            "Arahkan telapak tangan Anda tepat di dalam kotak hijau ini",
            (x1 - 95, y2 + 35),
            scale=0.55,
            color=(240, 240, 240),
            thickness=1,
        )
        self.draw_text_with_outline(
            frame,
            "Tekan [SPASI] atau [ENTER] untuk Mengunci Warna Kulit",
            (x1 - 85, y2 + 65),
            scale=0.55,
            color=(0, 220, 255),
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            "Tekan [C] atau [ESC] untuk Selesai / Kembali ke Menu",
            (x1 - 65, y2 + 95),
            scale=0.48,
            color=(180, 180, 180),
            thickness=1,
        )

        # Tampilkan informasi status kalibrasi
        status_txt = "STATUS: SUDAH TERKALIBRASI" if is_calibrated else "STATUS: MENGGUNAKAN PROFIL DEFAULT"
        col = (0, 255, 120) if is_calibrated else (255, 200, 50)
        self.draw_text_with_outline(
            frame,
            status_txt,
            (x1 - 30, y1 - 10),
            scale=0.48,
            color=col,
            thickness=1,
        )

    def draw_level_select(
        self,
        frame: np.ndarray,
        buttons: List[InteractiveButton],
        unlocked_level: int,
        high_scores: dict,
        play_rect: Tuple[int, int, int, int],
        current_time: float,
    ) -> None:
        """Merender layar pemilihan level 1 - 10 di dalam Playing Section."""
        px1, py1, px2, py2 = play_rect

        self.draw_text_with_outline(
            frame,
            "PILIH LEVEL (1 - 10)",
            (px1 + 35, 115),
            scale=0.92,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            "Selesaikan level sebelumnya untuk membuka kunci level selanjutnya",
            (px1 + 35, 142),
            scale=0.46,
            color=(180, 180, 180),
            thickness=1,
        )

        for btn in buttons:
            self.draw_button(frame, btn, current_time)

            if btn.action_id.startswith("level_"):
                lvl = int(btn.action_id.split("_")[1])
                cfg: Optional[LevelConfig] = LEVEL_CONFIGS.get(lvl)
                if cfg:
                    x1, y1, x2, y2 = btn.rect
                    best = high_scores.get(str(lvl), 0)
                    info = f"Tgt: {cfg.target_hits} | Rekor: {best}"
                    self.draw_text_with_outline(
                        frame,
                        info,
                        (x1 + 6, y2 - 8),
                        scale=0.36,
                        color=(200, 200, 200) if lvl <= unlocked_level else COLOR_LOCKED_GRAY,
                        thickness=1,
                    )

        hint = "Pilih level dengan hover tangan atau tekan angka [1-9] / [0] | [ESC] Menu"
        self.draw_text_with_outline(
            frame,
            hint,
            (px1 + 25, py2 - 20),
            scale=0.44,
            color=(160, 240, 160),
            thickness=1,
        )

    def draw_grid_select(
        self,
        frame: np.ndarray,
        buttons: List[InteractiveButton],
        selected_index: int,
        play_rect: Tuple[int, int, int, int],
        current_time: float,
    ) -> None:
        """Merender layar pengaturan ukuran grid arena di dalam Playing Section."""
        px1, py1, px2, py2 = play_rect

        self.draw_text_with_outline(
            frame,
            "PENGATURAN UKURAN GRID",
            (px1 + 35, 115),
            scale=0.92,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            "Pilih tata letak lubang sesuai selera dan kenyamanan bermain Anda",
            (px1 + 35, 142),
            scale=0.46,
            color=(180, 180, 180),
            thickness=1,
        )

        for btn in buttons:
            self.draw_button(frame, btn, current_time)

            if btn.action_id.startswith("grid_"):
                idx = int(btn.action_id.split("_")[1])
                rows, cols = GRID_PRESETS[idx]
                x1, y1, x2, y2 = btn.rect

                if idx == selected_index:
                    self.draw_text_with_outline(
                        frame,
                        "[SEDANG AKTIF]",
                        (x2 - 130, y1 + 22),
                        scale=0.44,
                        color=(0, 255, 120),
                        thickness=1,
                    )

                # Gambar diagram miniatur lubang
                preview_cx = x1 + 55
                preview_cy = (y1 + y2) // 2
                spacing_x = 18
                spacing_y = 15
                start_x = preview_cx - (cols - 1) * spacing_x // 2
                start_y = preview_cy - (rows - 1) * spacing_y // 2
                for r in range(rows):
                    for c in range(cols):
                        dot_x = start_x + c * spacing_x
                        dot_y = start_y + r * spacing_y
                        cv2.circle(frame, (dot_x, dot_y), 4, (0, 220, 255), -1, cv2.LINE_AA)

        hint = "Hover tangan untuk memilih atau tekan [1-4] | [ESC] Kembali ke Menu"
        self.draw_text_with_outline(
            frame,
            hint,
            (px1 + 25, py2 - 20),
            scale=0.44,
            color=(160, 240, 160),
            thickness=1,
        )

    def draw_survival_hud(
        self,
        frame: np.ndarray,
        score_manager: ScoreManager,
        survival_tracker: SurvivalTracker,
        high_score: int,
        hand_detected: bool,
        play_rect: Tuple[int, int, int, int],
        current_time: float,
    ) -> None:
        """Merender HUD atas khusus Mode Survival di dalam Playing Section."""
        px1, py1, px2, py2 = play_rect
        pw = px2 - px1

        # Nyawa / HP (Hearts)
        heart_x = px1 + 25
        self.draw_text_with_outline(
            frame,
            "NYAWA:",
            (heart_x, 118),
            scale=0.55,
            color=(200, 200, 200),
            thickness=1,
        )
        for i in range(survival_tracker.max_lives):
            hx = heart_x + 70 + i * 24
            hy = 112
            color = COLOR_HEART_RED if i < survival_tracker.lives else (50, 50, 50)
            cv2.circle(frame, (hx, hy), 8, color, -1, cv2.LINE_AA)

        # Skor & Combo
        combo_str = f" (x{score_manager.combo} COMBO)" if score_manager.combo >= 2 else ""
        self.draw_text_with_outline(
            frame,
            f"SKOR: {score_manager.score}{combo_str}",
            (heart_x, 146),
            scale=0.72,
            color=(0, 230, 255),
            thickness=2,
        )

        # Tingkat Intensitas di Tengah Playing Section
        tier = survival_tracker.get_intensity_tier(current_time)
        stars = "★" * tier + "☆" * (4 - tier)
        tier_colors = [
            (0, 230, 100),
            (0, 220, 255),
            (0, 140, 255),
            (0, 0, 255),
        ]
        t_color = tier_colors[tier - 1]
        tier_text = f"INTENSITAS {tier} {stars}"
        t_size = cv2.getTextSize(tier_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        self.draw_text_with_outline(
            frame,
            tier_text,
            (px1 + (pw - t_size[0]) // 2, 118),
            scale=0.75,
            color=t_color,
            thickness=2,
        )

        elapsed = survival_tracker.get_elapsed(current_time)
        time_text = f"SURVIVE: {elapsed:.1f}s"
        tt_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0]
        self.draw_text_with_outline(
            frame,
            time_text,
            (px1 + (pw - tt_size[0]) // 2, 146),
            scale=0.58,
            color=(220, 220, 220),
            thickness=1,
        )

        # Status Tangan
        hand_color = (0, 240, 100) if hand_detected else (0, 80, 240)
        hand_text = "TANGAN: AKTIF" if hand_detected else "TANGAN: HILANG"
        cv2.circle(frame, (px2 - 190, 114), 7, hand_color, -1, cv2.LINE_AA)
        self.draw_text_with_outline(
            frame,
            hand_text,
            (px2 - 175, 118),
            scale=0.50,
            color=hand_color,
            thickness=1,
        )
        self.draw_text_with_outline(
            frame,
            f"REKOR: {high_score}",
            (px2 - 175, 144),
            scale=0.50,
            color=(180, 180, 180),
            thickness=1,
        )

    def draw_level_hud(
        self,
        frame: np.ndarray,
        level_cfg: LevelConfig,
        score_manager: ScoreManager,
        time_left: float,
        best_score: int,
        hand_detected: bool,
        play_rect: Tuple[int, int, int, int],
    ) -> None:
        """Merender HUD atas khusus Mode Level Progression."""
        px1, py1, px2, py2 = play_rect
        pw = px2 - px1

        self.draw_text_with_outline(
            frame,
            f"LEVEL {level_cfg.level_num}: {level_cfg.title}",
            (px1 + 25, 118),
            scale=0.75,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )

        hits = score_manager.hits_count
        target = level_cfg.target_hits
        self.draw_text_with_outline(
            frame,
            f"Target Mole: {hits} / {target}",
            (px1 + 25, 146),
            scale=0.55,
            color=(220, 220, 220),
            thickness=1,
        )

        # Progress bar
        bar_x = px1 + 190
        bar_w = 120
        ratio = max(0.0, min(1.0, hits / target))
        cv2.rectangle(frame, (bar_x, 134), (bar_x + bar_w, 146), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, 134), (bar_x + int(bar_w * ratio), 146), (0, 255, 120), -1)
        cv2.rectangle(frame, (bar_x, 134), (bar_x + bar_w, 146), (120, 120, 120), 1)

        # Timer Countdown di Tengah
        time_color = COLOR_TIMER_NORMAL if time_left > 5.0 else COLOR_TIMER_CRITICAL
        time_text = f"WAKTU: {max(0.0, time_left):.1f}s"
        t_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.88, 2)[0]
        self.draw_text_with_outline(
            frame,
            time_text,
            (px1 + (pw - t_size[0]) // 2, 132),
            scale=0.88,
            color=time_color,
            thickness=2,
        )

        hand_color = (0, 240, 100) if hand_detected else (0, 80, 240)
        hand_text = "TANGAN: AKTIF" if hand_detected else "TANGAN: HILANG"
        cv2.circle(frame, (px2 - 190, 114), 7, hand_color, -1, cv2.LINE_AA)
        self.draw_text_with_outline(
            frame,
            hand_text,
            (px2 - 175, 118),
            scale=0.50,
            color=hand_color,
            thickness=1,
        )
        self.draw_text_with_outline(
            frame,
            f"REKOR: {best_score}",
            (px2 - 175, 144),
            scale=0.50,
            color=(180, 180, 180),
            thickness=1,
        )

    def draw_holes_and_moles(
        self,
        frame: np.ndarray,
        holes: List[Hole],
        moles: List[Mole],
        hit_effects: List[HitEffect],
        current_time: float,
    ) -> None:
        """Merender struktur lubang serta mole aktif (mendukung multi-mole)."""
        active_indices = {m.current_hole_idx: m for m in moles}

        for hole in holes:
            hx, hy, r = hole.x, hole.y, hole.radius

            cv2.circle(frame, (hx, hy), r, COLOR_HOLE_INNER, -1, cv2.LINE_AA)
            cv2.circle(frame, (hx, hy), r, COLOR_HOLE_OUTER, 3, cv2.LINE_AA)
            cv2.ellipse(
                frame,
                (hx, hy + int(r * 0.45)),
                (int(r * 0.70), int(r * 0.35)),
                0,
                0,
                360,
                (20, 20, 20),
                -1,
                cv2.LINE_AA,
            )

            if hole.index in active_indices:
                mole = active_indices[hole.index]
                ratio = mole.get_time_ratio(current_time)
                mole_r = int(r * 0.75)

                cv2.circle(frame, (hx, hy), mole_r, COLOR_MOLE_BODY, -1, cv2.LINE_AA)
                cv2.circle(frame, (hx, hy), mole_r, COLOR_PRIMARY_CYAN, 3, cv2.LINE_AA)

                eye_x = int(mole_r * 0.35)
                eye_y = hy - int(mole_r * 0.20)
                cv2.circle(frame, (hx - eye_x, eye_y), 6, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx + eye_x, eye_y), 6, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx - eye_x + 1, eye_y - 1), 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx + eye_x + 1, eye_y - 1), 2, (255, 255, 255), -1, cv2.LINE_AA)

                cv2.ellipse(
                    frame,
                    (hx, hy + int(mole_r * 0.15)),
                    (int(mole_r * 0.40), int(mole_r * 0.25)),
                    0,
                    0,
                    360,
                    (170, 215, 255),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(frame, (hx, hy + int(mole_r * 0.08)), 5, (30, 30, 30), -1, cv2.LINE_AA)

                angle_end = int(360 * ratio)
                if angle_end > 0:
                    cv2.ellipse(
                        frame,
                        (hx, hy),
                        (r + 6, r + 6),
                        -90,
                        0,
                        angle_end,
                        COLOR_PRIMARY_CYAN,
                        3,
                        cv2.LINE_AA,
                    )

        for eff in hit_effects:
            if eff.is_alive(current_time):
                p = eff.get_progress(current_time)
                pulse_r = int(holes[0].radius * (0.8 + p * 1.5))
                cv2.circle(frame, eff.position, pulse_r, COLOR_HIT_FLASH, 3, cv2.LINE_AA)

                combo_txt = f" +{10 * max(1, eff.combo)}"
                if eff.combo >= 2:
                    combo_txt += f" (x{eff.combo} COMBO)"

                float_y = eff.position[1] - int(p * 50)
                self.draw_text_with_outline(
                    frame,
                    f"HIT!{combo_txt}",
                    (eff.position[0] - 60, float_y),
                    scale=0.68,
                    color=COLOR_HIT_FLASH,
                    thickness=2,
                )

    def draw_hand_cursor(
        self,
        frame: np.ndarray,
        centroid: Optional[Tuple[int, int]],
        hover_progress: float = 0.0,
    ) -> None:
        """Merender reticle kursor bidik di posisi centroid tangan pemain."""
        if centroid is None:
            return

        cx, cy = centroid
        cv2.circle(frame, (cx, cy), 22, COLOR_RETICLE, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1, cv2.LINE_AA)

        arm = 14
        cv2.line(frame, (cx - 22 - arm, cy), (cx - 22, cy), COLOR_RETICLE, 2, cv2.LINE_AA)
        cv2.line(frame, (cx + 22, cy), (cx + 22 + arm, cy), COLOR_RETICLE, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 22 - arm), (cx, cy - 22), COLOR_RETICLE, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + 22), (cx, cy + 22 + arm), COLOR_RETICLE, 2, cv2.LINE_AA)

        if hover_progress > 0.0:
            angle = int(360 * hover_progress)
            cv2.ellipse(
                frame,
                (cx, cy),
                (28, 28),
                -90,
                0,
                angle,
                (0, 255, 120),
                3,
                cv2.LINE_AA,
            )

    def draw_modal_dialog(
        self,
        frame: np.ndarray,
        title: str,
        title_color: Tuple[int, int, int],
        lines: List[str],
        action_prompt: str,
        secondary_prompt: str = "",
    ) -> None:
        """Merender kotak dialog modal serbaguna untuk Level Cleared dan Game Over."""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        box_w, box_h = int(w * 0.65), int(h * 0.60)
        bx1 = (w - box_w) // 2
        by1 = (h - box_h) // 2
        bx2, by2 = bx1 + box_w, by1 + box_h

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30, 30, 30), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), title_color, 2)

        t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
        tx = (w - t_size[0]) // 2
        self.draw_text_with_outline(
            frame,
            title,
            (tx, by1 + 60),
            scale=1.1,
            color=title_color,
            thickness=3,
        )
        cv2.line(frame, (bx1 + 30, by1 + 80), (bx2 - 30, by1 + 80), (70, 70, 70), 1)

        curr_y = by1 + 130
        for line in lines:
            l_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0]
            lx = (w - l_size[0]) // 2
            self.draw_text_with_outline(
                frame,
                line,
                (lx, curr_y),
                scale=0.65,
                color=(220, 220, 220),
                thickness=1,
            )
            curr_y += 40

        p_size = cv2.getTextSize(action_prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)[0]
        px = (w - p_size[0]) // 2
        self.draw_text_with_outline(
            frame,
            action_prompt,
            (px, by2 - 45),
            scale=0.82,
            color=(0, 255, 120),
            thickness=2,
        )

        if secondary_prompt:
            s_size = cv2.getTextSize(secondary_prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
            sx = (w - s_size[0]) // 2
            self.draw_text_with_outline(
                frame,
                secondary_prompt,
                (sx, by2 - 18),
                scale=0.52,
                color=(180, 180, 180),
                thickness=1,
            )

    def draw_mask_pip(self, frame: np.ndarray, mask: np.ndarray) -> None:
        """Menampilkan jendela kecil Picture-in-Picture untuk memonitor mask segmentasi."""
        h, w = frame.shape[:2]
        pip_w, pip_h = 200, 150
        pip_x, pip_y = w - pip_w - 20, h - pip_h - 20

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        small_mask = cv2.resize(mask_bgr, (pip_w, pip_h))

        cv2.rectangle(
            frame,
            (pip_x - 2, pip_y - 2),
            (pip_x + pip_w + 2, pip_y + pip_h + 2),
            COLOR_PRIMARY_CYAN,
            2,
        )
        frame[pip_y : pip_y + pip_h, pip_x : pip_x + pip_w] = small_mask

        self.draw_text_with_outline(
            frame,
            "Mask Kulit (Playing Only)",
            (pip_x + 5, pip_y + 20),
            scale=0.45,
            color=(0, 255, 255),
            thickness=1,
        )
