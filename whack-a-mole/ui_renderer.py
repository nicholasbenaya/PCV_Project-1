"""Modul visualisasi OpenCV lengkap (Full Screen HD 1280x720) dengan Open-Palm Lock-On & PiP Diagnostik."""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    from .config import (
        COLOR_BUTTON_BG,
        COLOR_BUTTON_BORDER,
        COLOR_BUTTON_HOVER,
        COLOR_FIST,
        COLOR_HEART_RED,
        COLOR_HIGH_FIVE,
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
        POSE_FIST,
        POSE_HIGH_FIVE,
        PRESET_NAMES,
        LevelConfig,
    )
    from .entities import HitEffect, Hole, InteractiveButton, Mole, ScoreManager, SurvivalTracker
except ImportError:
    from config import (
        COLOR_BUTTON_BG,
        COLOR_BUTTON_BORDER,
        COLOR_BUTTON_HOVER,
        COLOR_FIST,
        COLOR_HEART_RED,
        COLOR_HIGH_FIVE,
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
        POSE_FIST,
        POSE_HIGH_FIVE,
        PRESET_NAMES,
        LevelConfig,
    )
    from entities import HitEffect, Hole, InteractiveButton, Mole, ScoreManager, SurvivalTracker


class UIRenderer:
    """Renderer antarmuka grafis untuk seluruh state permainan Whack-A-Mole pada layar penuh HD."""

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
        """Merender komponen tombol UI interaktif dengan efek hover/fist highlight."""
        x1, y1, x2, y2 = button.rect
        hovered = button.is_hovered
        locked = button.locked

        bg_color = (
            COLOR_LOCKED_GRAY
            if locked
            else (COLOR_BUTTON_HOVER if hovered else COLOR_BUTTON_BG)
        )
        border_color = (
            (60, 60, 60)
            if locked
            else ((0, 255, 120) if hovered else COLOR_BUTTON_BORDER)
        )
        border_thickness = 3 if hovered else 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)

        lbl = button.label + (" [TERKUNCI]" if locked else "")
        lbl_color = (
            (140, 140, 140)
            if locked
            else ((0, 255, 120) if hovered else COLOR_TEXT_WHITE)
        )

        font_scale = 0.58 if len(lbl) < 30 else 0.50
        text_size = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        tx = x1 + 24
        ty = y1 + (y2 - y1 + text_size[1]) // 2

        self.draw_text_with_outline(
            frame,
            lbl,
            (tx, ty),
            scale=font_scale,
            color=lbl_color,
            thickness=2,
        )

        if button.hotkey_desc:
            h_size = cv2.getTextSize(
                button.hotkey_desc, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            )[0]
            hx = x2 - h_size[0] - 20
            self.draw_text_with_outline(
                frame,
                button.hotkey_desc,
                (hx, ty),
                scale=0.48,
                color=(180, 220, 255),
                thickness=1,
            )

    def draw_main_menu(
        self,
        frame: np.ndarray,
        buttons: List[InteractiveButton],
        grid_desc: str,
        audio_enabled: bool,
        is_calibrated: bool,
        survival_high_score: int,
        unlocked_level: int,
        preset_name: str,
        current_pose: str,
        is_locked: bool,
        bg_captured: bool,
        current_time: float,
    ) -> None:
        """Merender tampilan Menu Utama lengkap pada seluruh layar."""
        w = frame.shape[1]

        # Judul di Tengah Layar
        title = "WHACK-A-MOLE VIRTUAL (PURE OPENCV)"
        t_w = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0][0]
        self.draw_text_with_outline(
            frame,
            title,
            ((w - t_w) // 2, 65),
            scale=1.1,
            color=COLOR_PRIMARY_CYAN,
            thickness=3,
        )

        subtitle = "Open-Palm Lock-On  |  Nearest-Neighbor Tracking  |  Full Screen Arena"
        s_w = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0][0]
        self.draw_text_with_outline(
            frame,
            subtitle,
            ((w - s_w) // 2, 94),
            scale=0.50,
            color=(0, 255, 120),
            thickness=1,
        )

        # Baris Indikator Sistem
        audio_str = "ON" if audio_enabled else "OFF"
        bg_str = "AKTIF ([B] / [N] Reset)" if bg_captured else "BYPASS / OFF ([B] Rekam)"
        bg_col = (0, 255, 120) if bg_captured else (255, 180, 80)
        lock_str = "TERKUNCI" if is_locked else "MENUNGGU TELAPAK (OPEN)"
        lock_col = (0, 255, 120) if is_locked else (0, 220, 255)

        info_line_1 = f"Grid: {grid_desc}  |  Audio: {audio_str}  |  BG Sub: {bg_str}  |  Mode: {preset_name}"
        i1_w = cv2.getTextSize(info_line_1, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0]
        self.draw_text_with_outline(
            frame,
            info_line_1,
            ((w - i1_w) // 2, 122),
            scale=0.44,
            color=bg_col,
            thickness=1,
        )

        info_line_2 = f"Hand Lock: {lock_str}  |  Gestur: [{current_pose}]  |  Survival: {survival_high_score}  |  Level: {unlocked_level}/10"
        i2_w = cv2.getTextSize(info_line_2, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0]
        self.draw_text_with_outline(
            frame,
            info_line_2,
            ((w - i2_w) // 2, 146),
            scale=0.44,
            color=lock_col,
            thickness=1,
        )

        # Render seluruh tombol menu
        for btn in buttons:
            self.draw_button(frame, btn, current_time)

        # Petunjuk interaksi bawah
        hint = "[B] Rekam BG  |  [N] Reset/Bypass BG  |  [OPEN PALM] Kunci  |  [FIST] Pukul/Klik  |  [V] PiP PCV  |  [M] Mode"
        h_w = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        self.draw_text_with_outline(
            frame,
            hint,
            ((w - h_w) // 2, frame.shape[0] - 22),
            scale=0.42,
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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 3)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 120), -1)

        arm = 18
        cv2.line(frame, (x1, y1), (x1 + arm, y1), (0, 255, 120), 4)
        cv2.line(frame, (x1, y1), (x1, y1 + arm), (0, 255, 120), 4)
        cv2.line(frame, (x2, y1), (x2 - arm, y1), (0, 255, 120), 4)
        cv2.line(frame, (x2, y1), (x2, y1 + arm), (0, 255, 120), 4)
        cv2.line(frame, (x1, y2), (x1 + arm, y2), (0, 255, 120), 4)
        cv2.line(frame, (x1, y2), (x1, y2 - arm), (0, 255, 120), 4)
        cv2.line(frame, (x2, y2), (x2 - arm, y2), (0, 255, 120), 4)
        cv2.line(frame, (x2, y2), (x2, y2 - arm), (0, 255, 120), 4)

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
        current_time: float,
    ) -> None:
        """Merender layar pemilihan level 1 - 10 pada layar penuh."""
        w = frame.shape[1]

        title = "PILIH LEVEL (1 - 10)"
        t_w = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)[0][0]
        self.draw_text_with_outline(
            frame,
            title,
            ((w - t_w) // 2, 85),
            scale=0.95,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )

        sub = f"Level Terbuka: {unlocked_level}/10 | Buka level berikutnya dengan menyelesaikan level sebelumnya"
        sub_w = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0]
        self.draw_text_with_outline(
            frame,
            sub,
            ((w - sub_w) // 2, 118),
            scale=0.46,
            color=(180, 220, 255),
            thickness=1,
        )

        for btn in buttons:
            self.draw_button(frame, btn, current_time)

        hint = "Arahkan tangan [OPEN PALM] ke kartu level lalu kepalkan tangan [FIST] untuk Memilih | [ESC] Menu"
        h_w = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0]
        self.draw_text_with_outline(
            frame,
            hint,
            ((w - h_w) // 2, frame.shape[0] - 25),
            scale=0.44,
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
        """Merender layar konfigurasi ukuran grid pada layar penuh."""
        w = frame.shape[1]

        title = "PENGATURAN UKURAN GRID LUBANG"
        t_w = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.92, 2)[0][0]
        self.draw_text_with_outline(
            frame,
            title,
            ((w - t_w) // 2, 85),
            scale=0.92,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )

        sub = "Pilih tata letak arena bermain yang diinginkan (Tersimpan otomatis)"
        sub_w = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0]
        self.draw_text_with_outline(
            frame,
            sub,
            ((w - sub_w) // 2, 118),
            scale=0.46,
            color=(180, 220, 255),
            thickness=1,
        )

        for i, btn in enumerate(buttons[:-1]):
            self.draw_button(frame, btn, current_time)
            if i == selected_index:
                x1, y1, x2, y2 = btn.rect
                cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 255, 120), 2)
                self.draw_text_with_outline(
                    frame,
                    "[AKTIF]",
                    (x2 - 100, y1 + 34),
                    scale=0.48,
                    color=(0, 255, 120),
                    thickness=2,
                )

        self.draw_button(frame, buttons[-1], current_time)

        hint = "Arahkan [OPEN PALM] lalu kepalkan tangan [FIST] untuk Memilih Grid | [ESC] Menu"
        h_w = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0]
        self.draw_text_with_outline(
            frame,
            hint,
            ((w - h_w) // 2, frame.shape[0] - 25),
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
        current_pose: str,
        is_locked: bool,
        current_time: float,
    ) -> None:
        """Merender HUD atas khusus Mode Survival pada layar penuh."""
        w = frame.shape[1]

        # Skor di Kiri
        self.draw_text_with_outline(
            frame,
            f"SKOR: {score_manager.score}",
            (35, 45),
            scale=0.82,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )

        combo = score_manager.combo
        if combo >= 3:
            multiplier = 4 if combo >= 10 else (3 if combo >= 6 else 2)
            self.draw_text_with_outline(
                frame,
                f"KOMBO x{multiplier}! ({combo}x Hit)",
                (35, 75),
                scale=0.58,
                color=(0, 255, 120),
                thickness=2,
            )

        # Timer & Tier di Tengah
        elapsed = survival_tracker.get_elapsed(current_time)
        tier = survival_tracker.get_intensity_tier(current_time)
        tier_names = {1: "SANTAI", 2: "INTENS", 3: "CEPAT", 4: "EKSTREM"}
        timer_text = f"SURVIVAL: {elapsed:.1f}s | TIER {tier}: {tier_names.get(tier, 'MAX')}"
        t_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)[0]
        self.draw_text_with_outline(
            frame,
            timer_text,
            ((w - t_size[0]) // 2, 45),
            scale=0.68,
            color=(255, 220, 0),
            thickness=2,
        )

        # Simbol Nyawa
        heart_x = (w - t_size[0]) // 2
        for i in range(survival_tracker.max_lives):
            color = COLOR_HEART_RED if i < survival_tracker.lives else (80, 80, 80)
            cv2.circle(frame, (heart_x + i * 26 + 10, 72), 8, color, -1, cv2.LINE_AA)

        # Status Gestur & Kunci di Kanan
        lock_col = (0, 255, 120) if is_locked else (0, 200, 255)
        lock_lbl = "LOCKED" if is_locked else "CARING OPEN"
        self.draw_text_with_outline(
            frame,
            f"[{lock_lbl}] [{current_pose}]",
            (w - 240, 45),
            scale=0.55,
            color=lock_col,
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            f"REKOR: {high_score}",
            (w - 240, 75),
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
        current_pose: str,
        is_locked: bool,
    ) -> None:
        """Merender HUD atas khusus Mode Level Progression pada layar penuh."""
        w = frame.shape[1]

        self.draw_text_with_outline(
            frame,
            f"LEVEL {level_cfg.level_num}: {level_cfg.title}",
            (35, 45),
            scale=0.80,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )

        hits = score_manager.hits_count
        target = level_cfg.target_hits
        self.draw_text_with_outline(
            frame,
            f"Target Mole: {hits} / {target}",
            (35, 75),
            scale=0.55,
            color=(220, 220, 220),
            thickness=1,
        )

        # Progress bar
        bar_x = 220
        bar_w = 120
        ratio = max(0.0, min(1.0, hits / target))
        cv2.rectangle(frame, (bar_x, 63), (bar_x + bar_w, 75), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, 63), (bar_x + int(bar_w * ratio), 75), (0, 255, 120), -1)
        cv2.rectangle(frame, (bar_x, 63), (bar_x + bar_w, 75), (120, 120, 120), 1)

        # Timer di Tengah
        time_color = COLOR_TIMER_NORMAL if time_left > 5.0 else COLOR_TIMER_CRITICAL
        time_text = f"WAKTU: {max(0.0, time_left):.1f}s"
        t_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.88, 2)[0]
        self.draw_text_with_outline(
            frame,
            time_text,
            ((w - t_size[0]) // 2, 55),
            scale=0.88,
            color=time_color,
            thickness=2,
        )

        lock_col = (0, 255, 120) if is_locked else (0, 200, 255)
        lock_lbl = "LOCKED" if is_locked else "SEARCHING"
        self.draw_text_with_outline(
            frame,
            f"[{lock_lbl}] [{current_pose}]",
            (w - 240, 45),
            scale=0.55,
            color=lock_col,
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            f"REKOR: {best_score}",
            (w - 240, 75),
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
        """Merender struktur lubang serta mole aktif pada seluruh area layar."""
        active_indices = {m.current_hole_idx: m for m in moles}

        for hole in holes:
            hx, hy = hole.x, hole.y
            r = hole.radius

            cv2.ellipse(frame, (hx, hy + 6), (r, int(r * 0.72)), 0, 0, 360, (25, 25, 25), -1, cv2.LINE_AA)
            cv2.ellipse(frame, (hx, hy), (r, int(r * 0.70)), 0, 0, 360, COLOR_HOLE_OUTER, 3, cv2.LINE_AA)
            cv2.ellipse(frame, (hx, hy), (r - 4, int((r - 4) * 0.68)), 0, 0, 360, COLOR_HOLE_INNER, -1, cv2.LINE_AA)

            if hole.index in active_indices:
                mole = active_indices[hole.index]
                time_ratio = mole.get_time_ratio(current_time)

                up_offset = int(r * 0.40 * time_ratio)
                body_cy = hy - up_offset
                body_rx = int(r * 0.62)
                body_ry = int(r * 0.68)

                cv2.ellipse(frame, (hx, body_cy), (body_rx, body_ry), 0, 0, 360, COLOR_MOLE_BODY, -1, cv2.LINE_AA)
                cv2.ellipse(frame, (hx, body_cy), (body_rx, body_ry), 0, 0, 360, (20, 60, 140), 2, cv2.LINE_AA)

                eye_y = body_cy - 12
                eye_spacing = 15
                cv2.circle(frame, (hx - eye_spacing, eye_y), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx + eye_spacing, eye_y), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx - eye_spacing + 1, eye_y), 2, (10, 10, 10), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx + eye_spacing - 1, eye_y), 2, (10, 10, 10), -1, cv2.LINE_AA)

                nose_y = body_cy + 2
                cv2.ellipse(frame, (hx, nose_y), (8, 5), 0, 0, 360, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.line(frame, (hx - 12, nose_y), (hx - 32, nose_y - 2), (220, 220, 220), 1, cv2.LINE_AA)
                cv2.line(frame, (hx + 12, nose_y), (hx + 32, nose_y - 2), (220, 220, 220), 1, cv2.LINE_AA)

                timer_color = COLOR_TIMER_NORMAL if time_ratio > 0.3 else COLOR_TIMER_CRITICAL
                angle_span = int(360 * time_ratio)
                cv2.ellipse(
                    frame,
                    (hx, hy),
                    (r + 6, int((r + 6) * 0.70)),
                    0,
                    -90,
                    -90 + angle_span,
                    timer_color,
                    3,
                    cv2.LINE_AA,
                )

        for effect in hit_effects:
            prog = effect.get_progress(current_time)
            ex, ey = effect.position
            expand_r = int(25 + 55 * prog)
            alpha_color = (
                int(COLOR_HIT_FLASH[0] * (1.0 - prog)),
                int(COLOR_HIT_FLASH[1] * (1.0 - prog)),
                int(COLOR_HIT_FLASH[2] * (1.0 - prog)),
            )

            cv2.circle(frame, (ex, ey), expand_r, alpha_color, 3, cv2.LINE_AA)
            self.draw_text_with_outline(
                frame,
                f"WHACK! +{100 * effect.combo}",
                (ex - 60, ey - expand_r - 8),
                scale=0.65,
                color=(0, 255, 120),
                thickness=2,
            )

    def draw_hand_cursor(
        self,
        frame: np.ndarray,
        hand_result: Any,
    ) -> None:
        """Merender kursor visual yang membedakan status lock-on, HIGH_FIVE (Move), dan FIST (Whack)."""
        w, h = frame.shape[1], frame.shape[0]

        if hand_result is None or not hand_result.is_detected or hand_result.centroid is None:
            # Belum terkunci: Tampilkan banner instruksi lock-on
            banner = "TUNJUKKAN TELAPAK TANGAN (HIGH-FIVE) UNTUK MENGUNCI KURSOR (LOCK-ON)"
            b_w = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0][0]
            self.draw_text_with_outline(
                frame,
                banner,
                ((w - b_w) // 2, h - 55),
                scale=0.52,
                color=(0, 255, 255),
                thickness=2,
            )
            return

        cx, cy = hand_result.centroid

        if hand_result.is_fist:
            cv2.circle(frame, (cx, cy), 28, COLOR_FIST, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 38, (0, 160, 255), 3, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1, cv2.LINE_AA)

            arm = 16
            cv2.line(frame, (cx - 38 - arm, cy), (cx - 38, cy), COLOR_FIST, 3, cv2.LINE_AA)
            cv2.line(frame, (cx + 38, cy), (cx + 38 + arm, cy), COLOR_FIST, 3, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - 38 - arm), (cx, cy - 38), COLOR_FIST, 3, cv2.LINE_AA)
            cv2.line(frame, (cx, cy + 38), (cx, cy + 38 + arm), COLOR_FIST, 3, cv2.LINE_AA)

            self.draw_text_with_outline(
                frame,
                "WHACK! (FIST)",
                (cx - 65, cy - 48),
                scale=0.62,
                color=COLOR_FIST,
                thickness=2,
            )

        elif hand_result.is_high_five:
            cv2.circle(frame, (cx, cy), 26, COLOR_HIGH_FIVE, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 6, COLOR_HIGH_FIVE, -1, cv2.LINE_AA)

            arm = 14
            cv2.line(frame, (cx - 26 - arm, cy), (cx - 26, cy), COLOR_HIGH_FIVE, 2, cv2.LINE_AA)
            cv2.line(frame, (cx + 26, cy), (cx + 26 + arm, cy), COLOR_HIGH_FIVE, 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - 26 - arm), (cx, cy - 26), COLOR_HIGH_FIVE, 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy + 26), (cx, cy + 26 + arm), COLOR_HIGH_FIVE, 2, cv2.LINE_AA)

            self.draw_text_with_outline(
                frame,
                "MOVE (OPEN)",
                (cx - 55, cy - 35),
                scale=0.55,
                color=COLOR_HIGH_FIVE,
                thickness=2,
            )

        else:
            cv2.circle(frame, (cx, cy), 22, COLOR_RETICLE, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)

            arm = 12
            cv2.line(frame, (cx - 22 - arm, cy), (cx - 22, cy), COLOR_RETICLE, 2, cv2.LINE_AA)
            cv2.line(frame, (cx + 22, cy), (cx + 22 + arm, cy), COLOR_RETICLE, 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - 22 - arm), (cx, cy - 22), COLOR_RETICLE, 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy + 22), (cx, cy + 22 + arm), COLOR_RETICLE, 2, cv2.LINE_AA)

            self.draw_text_with_outline(
                frame,
                "LOCKED",
                (cx - 30, cy - 30),
                scale=0.48,
                color=COLOR_RETICLE,
                thickness=1,
            )

    def draw_diagnostic_pip(
        self,
        frame: np.ndarray,
        hand_result: Any,
    ) -> None:
        """Merender jendela Picture-in-Picture (PiP) diagnostik untuk demo fitur bentuk PCV ke penguji."""
        if hand_result is None or hand_result.mask is None:
            return

        h, w = frame.shape[:2]
        pip_w, pip_h = 280, 210
        pip_x, pip_y = w - pip_w - 20, h - pip_h - 20

        mask_gray = hand_result.mask
        mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
        pip_canvas = cv2.resize(mask_bgr, (pip_w, pip_h))

        scale_x = pip_w / float(w)
        scale_y = pip_h / float(h)

        if hand_result.contour is not None:
            scaled_cnt = (hand_result.contour.astype(float) * [scale_x, scale_y]).astype(np.int32)
            cv2.drawContours(pip_canvas, [scaled_cnt], -1, (255, 200, 0), 1)

        if hand_result.hull is not None:
            scaled_hull = (hand_result.hull.astype(float) * [scale_x, scale_y]).astype(np.int32)
            cv2.drawContours(pip_canvas, [scaled_hull], -1, (0, 255, 120), 2)

        for pt in hand_result.defects_pts:
            px = int(pt[0] * scale_x)
            py = int(pt[1] * scale_y)
            cv2.circle(pip_canvas, (px, py), 4, (0, 0, 255), -1)
            cv2.circle(pip_canvas, (px, py), 6, (0, 255, 255), 1)

        border_col = (
            COLOR_HIGH_FIVE
            if hand_result.is_high_five
            else (COLOR_FIST if hand_result.is_fist else COLOR_PRIMARY_CYAN)
        )
        cv2.rectangle(frame, (pip_x - 3, pip_y - 3), (pip_x + pip_w + 3, pip_y + pip_h + 3), border_col, 2)
        frame[pip_y : pip_y + pip_h, pip_x : pip_x + pip_w] = pip_canvas

        # Teks diagnostik PCV di dalam kotak PiP
        bg_status = "BG: AKTIF" if hand_result.bg_captured else "BG: NON-AKTIF [B]"
        self.draw_text_with_outline(
            frame,
            f"DIAGNOSTIK PCV [V] | {bg_status}",
            (pip_x + 6, pip_y + 18),
            scale=0.36,
            color=(0, 255, 255),
            thickness=1,
        )

        lock_text = "LOCK: TERKUNCI" if hand_result.is_locked else "LOCK: CARI TELAPAK"
        self.draw_text_with_outline(
            frame,
            f"{lock_text} | POSE: {hand_result.pose}",
            (pip_x + 6, pip_y + 36),
            scale=0.40,
            color=border_col,
            thickness=2,
        )

        defects_txt = f"Defek: {hand_result.defect_count} | Holes: {hand_result.internal_holes_count} | Sol: {hand_result.solidity:.2f}"
        self.draw_text_with_outline(
            frame,
            defects_txt,
            (pip_x + 6, pip_y + 54),
            scale=0.34,
            color=(230, 230, 230),
            thickness=1,
        )

        ar_txt = f"AR: {hand_result.aspect_ratio:.2f} | Mode: {hand_result.preset_name}"
        self.draw_text_with_outline(
            frame,
            ar_txt,
            (pip_x + 6, pip_y + 70),
            scale=0.34,
            color=(200, 200, 200),
            thickness=1,
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
