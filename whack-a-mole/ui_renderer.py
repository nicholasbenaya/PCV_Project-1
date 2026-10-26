"""Modul visualisasi OpenCV lengkap (Menu Utama, Level Select, Grid Select, HUD, & Efek)."""

from typing import List, Optional, Tuple
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
    """Renderer antarmuka grafis untuk seluruh state permainan Whack-A-Mole."""

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

    def draw_button(
        self,
        frame: np.ndarray,
        button: InteractiveButton,
        current_time: float,
    ) -> None:
        """Merender tombol interaktif dengan efek hover dan meteran pengisian virtual touch."""
        x1, y1, x2, y2 = button.rect
        w, h = x2 - x1, y2 - y1

        # Warna latar tombol
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

        # Kotak isi tombol
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

        # Label tombol
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

        # Badge petunjuk tombol keyboard (hotkey) jika ada
        if button.hotkey_desc:
            self.draw_text_with_outline(
                frame,
                button.hotkey_desc,
                (x1 + 8, y1 + 18),
                scale=0.42,
                color=(170, 170, 170),
                thickness=1,
            )

        # Garis pengisian progres jika sedang di-hover oleh tangan
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
        survival_high_score: int,
        unlocked_level: int,
        current_time: float,
    ) -> None:
        """Merender tampilan Menu Utama lengkap dengan opsi mode dan navigasi."""
        h, w = frame.shape[:2]

        # Latar redup semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Judul Game
        self.draw_text_with_outline(
            frame,
            "WHACK-A-MOLE VIRTUAL",
            (w // 2 - 225, 60),
            scale=1.15,
            color=COLOR_PRIMARY_CYAN,
            thickness=3,
        )
        self.draw_text_with_outline(
            frame,
            "Game Interaktif Kontrol Tangan Berbasis OpenCV Murni",
            (w // 2 - 190, 88),
            scale=0.48,
            color=(190, 190, 190),
            thickness=1,
        )

        # Bilah status singkat (Grid Aktif & Audio)
        audio_str = "ON" if audio_enabled else "OFF (Bisu)"
        status_line = f"Grid: {grid_desc}  |  Audio: {audio_str}  |  Rekor Survival: {survival_high_score}  |  Level Terbuka: {unlocked_level}/10"
        s_size = cv2.getTextSize(status_line, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
        self.draw_text_with_outline(
            frame,
            status_line,
            ((w - s_size[0]) // 2, 118),
            scale=0.44,
            color=(255, 220, 100),
            thickness=1,
        )

        # Render semua tombol menu
        for btn in buttons:
            self.draw_button(frame, btn, current_time)

        # Petunjuk interaksi bawah
        hint = "Hover tangan di tombol atau tekan [1-5] | [C] Kalibrasi Kulit | [P] Mask PiP"
        h_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
        self.draw_text_with_outline(
            frame,
            hint,
            ((w - h_size[0]) // 2, h - 22),
            scale=0.44,
            color=(160, 240, 160),
            thickness=1,
        )

    def draw_level_select(
        self,
        frame: np.ndarray,
        buttons: List[InteractiveButton],
        unlocked_level: int,
        high_scores: dict,
        current_time: float,
    ) -> None:
        """Merender layar pemilihan level 1 - 10."""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        self.draw_text_with_outline(
            frame,
            "PILIH LEVEL (1 - 10)",
            (w // 2 - 150, 48),
            scale=0.95,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            "Selesaikan level sebelumnya untuk membuka kunci level selanjutnya",
            (w // 2 - 230, 72),
            scale=0.46,
            color=(180, 180, 180),
            thickness=1,
        )

        # Gambar kartu level
        for btn in buttons:
            self.draw_button(frame, btn, current_time)

            # Tambahan informasi skor dan target di bawah tombol
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

        hint = "Pilih level dengan hover tangan atau tekan angka [1-9] / [0 untuk Lvl 10] | [ESC] Menu"
        h_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        self.draw_text_with_outline(
            frame,
            hint,
            ((w - h_size[0]) // 2, h - 18),
            scale=0.42,
            color=(160, 240, 160),
            thickness=1,
        )

    def draw_grid_select(
        self,
        frame: np.ndarray,
        buttons: List[InteractiveButton],
        selected_index: int,
        current_time: float,
    ) -> None:
        """Merender layar pengaturan ukuran grid arena."""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        self.draw_text_with_outline(
            frame,
            "PENGATURAN UKURAN GRID",
            (w // 2 - 180, 52),
            scale=0.95,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            "Pilih tata letak lubang sesuai selera dan ruang bermain Anda",
            (w // 2 - 200, 78),
            scale=0.46,
            color=(180, 180, 180),
            thickness=1,
        )

        for btn in buttons:
            self.draw_button(frame, btn, current_time)

            # Gambar visual miniatur tata letak lubang di dalam tombol
            if btn.action_id.startswith("grid_"):
                idx = int(btn.action_id.split("_")[1])
                rows, cols = GRID_PRESETS[idx]
                x1, y1, x2, y2 = btn.rect

                # Status aktif
                if idx == selected_index:
                    self.draw_text_with_outline(
                        frame,
                        "[SEDANG DIPILIH]",
                        (x2 - 120, y1 + 22),
                        scale=0.40,
                        color=(0, 255, 120),
                        thickness=1,
                    )

                # Gambar preview mini dot
                preview_cx = x1 + 50
                preview_cy = (y1 + y2) // 2
                spacing_x = 16
                spacing_y = 14
                start_x = preview_cx - (cols - 1) * spacing_x // 2
                start_y = preview_cy - (rows - 1) * spacing_y // 2
                for r in range(rows):
                    for c in range(cols):
                        dot_x = start_x + c * spacing_x
                        dot_y = start_y + r * spacing_y
                        cv2.circle(frame, (dot_x, dot_y), 4, (0, 220, 255), -1, cv2.LINE_AA)

        hint = "Hover tangan untuk memilih atau tekan [1-4] | [ESC] Kembali ke Menu"
        h_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
        self.draw_text_with_outline(
            frame,
            hint,
            ((w - h_size[0]) // 2, h - 22),
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
        current_time: float,
    ) -> None:
        """Merender HUD atas khusus Mode Survival (Nyawa, Intensitas, Combo, Waktu)."""
        h, w = frame.shape[:2]
        hud_h = 75

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
        cv2.line(overlay, (0, hud_h), (w, hud_h), (60, 60, 60), 2)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

        # Nyawa / HP (Hearts)
        heart_x = 25
        self.draw_text_with_outline(
            frame,
            "NYAWA:",
            (heart_x, 34),
            scale=0.55,
            color=(200, 200, 200),
            thickness=1,
        )
        for i in range(survival_tracker.max_lives):
            hx = heart_x + 65 + i * 22
            hy = 28
            color = COLOR_HEART_RED if i < survival_tracker.lives else (50, 50, 50)
            cv2.circle(frame, (hx, hy), 7, color, -1, cv2.LINE_AA)

        # Skor & Combo
        combo_str = f" (x{score_manager.combo} COMBO)" if score_manager.combo >= 2 else ""
        self.draw_text_with_outline(
            frame,
            f"SKOR: {score_manager.score}{combo_str}",
            (heart_x, 62),
            scale=0.68,
            color=(0, 230, 255),
            thickness=2,
        )

        # Tingkat Intensitas di Tengah
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
        t_size = cv2.getTextSize(tier_text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
        self.draw_text_with_outline(
            frame,
            tier_text,
            ((w - t_size[0]) // 2, 34),
            scale=0.72,
            color=t_color,
            thickness=2,
        )

        elapsed = survival_tracker.get_elapsed(current_time)
        time_text = f"SURVIVE: {elapsed:.1f}s"
        tt_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        self.draw_text_with_outline(
            frame,
            time_text,
            ((w - tt_size[0]) // 2, 60),
            scale=0.55,
            color=(220, 220, 220),
            thickness=1,
        )

        # Status Tangan
        hand_color = (0, 240, 100) if hand_detected else (0, 80, 240)
        hand_text = "TANGAN: AKTIF" if hand_detected else "TANGAN: HILANG"
        cv2.circle(frame, (w - 180, 32), 6, hand_color, -1, cv2.LINE_AA)
        self.draw_text_with_outline(
            frame,
            hand_text,
            (w - 165, 36),
            scale=0.48,
            color=hand_color,
            thickness=1,
        )
        self.draw_text_with_outline(
            frame,
            f"REKOR: {high_score}",
            (w - 165, 60),
            scale=0.48,
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
    ) -> None:
        """Merender HUD atas khusus Mode Level Progression."""
        h, w = frame.shape[:2]
        hud_h = 75

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
        cv2.line(overlay, (0, hud_h), (w, hud_h), (60, 60, 60), 2)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

        # Judul Level
        self.draw_text_with_outline(
            frame,
            f"LEVEL {level_cfg.level_num}: {level_cfg.title}",
            (25, 34),
            scale=0.72,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )

        # Target Progress Bar
        hits = score_manager.hits_count
        target = level_cfg.target_hits
        progress_str = f"Target Mole: {hits} / {target}"
        self.draw_text_with_outline(
            frame,
            progress_str,
            (25, 60),
            scale=0.52,
            color=(220, 220, 220),
            thickness=1,
        )
        # Bar mini
        bar_x = 180
        bar_w = 120
        ratio = max(0.0, min(1.0, hits / target))
        cv2.rectangle(frame, (bar_x, 48), (bar_x + bar_w, 58), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, 48), (bar_x + int(bar_w * ratio), 58), (0, 255, 120), -1)
        cv2.rectangle(frame, (bar_x, 48), (bar_x + bar_w, 58), (120, 120, 120), 1)

        # Timer Countdown di Tengah
        time_color = COLOR_TIMER_NORMAL if time_left > 5.0 else COLOR_TIMER_CRITICAL
        time_text = f"WAKTU: {max(0.0, time_left):.1f}s"
        t_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)[0]
        self.draw_text_with_outline(
            frame,
            time_text,
            ((w - t_size[0]) // 2, 48),
            scale=0.82,
            color=time_color,
            thickness=2,
        )

        # Status Tangan & Rekor Level
        hand_color = (0, 240, 100) if hand_detected else (0, 80, 240)
        hand_text = "TANGAN: AKTIF" if hand_detected else "TANGAN: HILANG"
        cv2.circle(frame, (w - 180, 32), 6, hand_color, -1, cv2.LINE_AA)
        self.draw_text_with_outline(
            frame,
            hand_text,
            (w - 165, 36),
            scale=0.48,
            color=hand_color,
            thickness=1,
        )
        self.draw_text_with_outline(
            frame,
            f"REKOR: {best_score}",
            (w - 165, 60),
            scale=0.48,
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

            # Lubang dasar
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

            # Gambar mole bila lubang ini memiliki mole aktif
            if hole.index in active_indices:
                mole = active_indices[hole.index]
                ratio = mole.get_time_ratio(current_time)
                mole_r = int(r * 0.75)

                cv2.circle(frame, (hx, hy), mole_r, COLOR_MOLE_BODY, -1, cv2.LINE_AA)
                cv2.circle(frame, (hx, hy), mole_r, COLOR_PRIMARY_CYAN, 3, cv2.LINE_AA)

                eye_x = int(mole_r * 0.35)
                eye_y = hy - int(mole_r * 0.20)
                cv2.circle(frame, (hx - eye_x, eye_y), 5, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx + eye_x, eye_y), 5, (20, 20, 20), -1, cv2.LINE_AA)
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
                cv2.circle(frame, (hx, hy + int(mole_r * 0.08)), 4, (30, 30, 30), -1, cv2.LINE_AA)

                # Indikator busur timer kemunculan mole
                angle_end = int(360 * ratio)
                if angle_end > 0:
                    cv2.ellipse(
                        frame,
                        (hx, hy),
                        (r + 5, r + 5),
                        -90,
                        0,
                        angle_end,
                        COLOR_PRIMARY_CYAN,
                        3,
                        cv2.LINE_AA,
                    )

        # Animasi efek pukulan aktif
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
                    scale=0.62,
                    color=COLOR_HIT_FLASH,
                    thickness=2,
                )

    def draw_hand_cursor(
        self,
        frame: np.ndarray,
        centroid: Optional[Tuple[int, int]],
        hover_progress: float = 0.0,
    ) -> None:
        """Merender kursor reticle di posisi centroid tangan pemain."""
        if centroid is None:
            return

        cx, cy = centroid
        cv2.circle(frame, (cx, cy), 18, COLOR_RETICLE, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)

        arm = 12
        cv2.line(frame, (cx - 18 - arm, cy), (cx - 18, cy), COLOR_RETICLE, 2, cv2.LINE_AA)
        cv2.line(frame, (cx + 18, cy), (cx + 18 + arm, cy), COLOR_RETICLE, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 18 - arm), (cx, cy - 18), COLOR_RETICLE, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + 18), (cx, cy + 18 + arm), COLOR_RETICLE, 2, cv2.LINE_AA)

        # Lingkaran indikator progres virtual click saat hovering di tombol
        if hover_progress > 0.0:
            angle = int(360 * hover_progress)
            cv2.ellipse(
                frame,
                (cx, cy),
                (24, 24),
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

        box_w, box_h = int(w * 0.74), int(h * 0.65)
        bx1 = (w - box_w) // 2
        by1 = (h - box_h) // 2
        bx2, by2 = bx1 + box_w, by1 + box_h

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30, 30, 30), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), title_color, 2)

        t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 3)[0]
        tx = (w - t_size[0]) // 2
        self.draw_text_with_outline(
            frame,
            title,
            (tx, by1 + 55),
            scale=1.05,
            color=title_color,
            thickness=3,
        )
        cv2.line(frame, (bx1 + 30, by1 + 75), (bx2 - 30, by1 + 75), (70, 70, 70), 1)

        curr_y = by1 + 120
        for line in lines:
            l_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0]
            lx = (w - l_size[0]) // 2
            self.draw_text_with_outline(
                frame,
                line,
                (lx, curr_y),
                scale=0.62,
                color=(220, 220, 220),
                thickness=1,
            )
            curr_y += 36

        # Tombol aksi
        p_size = cv2.getTextSize(action_prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        px = (w - p_size[0]) // 2
        self.draw_text_with_outline(
            frame,
            action_prompt,
            (px, by2 - 45),
            scale=0.75,
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
        """Menampilkan inset kecil Picture-in-Picture untuk monitoring masker kulit."""
        h, w = frame.shape[:2]
        pip_w, pip_h = 160, 120
        pip_x, pip_y = w - pip_w - 15, h - pip_h - 15

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
            "Mask Kulit",
            (pip_x + 5, pip_y + 18),
            scale=0.45,
            color=(0, 255, 255),
            thickness=1,
        )

    def draw_calibration_box(self, frame: np.ndarray, box_rect: Tuple[int, int, int, int]) -> None:
        """Merender kotak panduan kalibrasi warna kulit di tengah frame."""
        x1, y1, x2, y2 = box_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 6, (0, 255, 120), -1)

        self.draw_text_with_outline(
            frame,
            "MODE KALIBRASI WARNA KULIT",
            (x1 - 35, y1 - 25),
            scale=0.62,
            color=(0, 255, 120),
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            "Arahkan telapak tangan ke dalam kotak hijau",
            (x1 - 65, y2 + 25),
            scale=0.50,
            color=(240, 240, 240),
            thickness=1,
        )
        self.draw_text_with_outline(
            frame,
            "Tekan [ENTER] atau [SPASI] untuk Kunci Kalibrasi | [C] Batal",
            (x1 - 90, y2 + 48),
            scale=0.44,
            color=(0, 220, 255),
            thickness=1,
        )
