"""Modul rendering grafis antarmuka (UI) dan elemen visual game berbasis OpenCV (HD 1280x720)."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    from .config import (
        COLOR_HIT_FLASH,
        COLOR_HOLE_INNER,
        COLOR_HOLE_OUTER,
        COLOR_MOLE_BODY,
        COLOR_PRIMARY_CYAN,
        COLOR_RETICLE,
        COLOR_TEXT_DARK,
        COLOR_TEXT_WHITE,
        COLOR_TIMER_CRITICAL,
        COLOR_TIMER_NORMAL,
    )
    from .entities import HitEffect, Hole, Mole, ScoreManager
except ImportError:
    from config import (
        COLOR_HIT_FLASH,
        COLOR_HOLE_INNER,
        COLOR_HOLE_OUTER,
        COLOR_MOLE_BODY,
        COLOR_PRIMARY_CYAN,
        COLOR_RETICLE,
        COLOR_TEXT_DARK,
        COLOR_TEXT_WHITE,
        COLOR_TIMER_CRITICAL,
        COLOR_TIMER_NORMAL,
    )
    from entities import HitEffect, Hole, Mole, ScoreManager


class UIRenderer:
    """Pengelola rendering seluruh antarmuka grafis dan elemen visual game pada resolusi HD."""

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
        """Menggambar teks bergaris tepi gelap agar kontras di latar webcam apa pun."""
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
            (sx1 + 25, 45),
            scale=0.68,
            color=(180, 190, 210),
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            "[SAFE ZONE] Wajah & Tubuh Terblokir 100%",
            (sx1 + 25, 75),
            scale=0.48,
            color=(0, 255, 120),
            thickness=1,
        )
        self.draw_text_with_outline(
            frame,
            "Kepala Anda berada di sini dengan aman tanpa mengganggu kursor",
            (sx1 + 25, 102),
            scale=0.40,
            color=(160, 160, 160),
            thickness=1,
        )

        # Label pada Playing Section
        self.draw_text_with_outline(
            frame,
            "PLAYING SECTION [ZONA TANGAN AKTIF]",
            (px1 + 25, 45),
            scale=0.72,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            f"Tata Letak: {zone_cfg['name']}  |  [H] Ganti Sisi",
            (px1 + 25, 75),
            scale=0.48,
            color=(255, 220, 100),
            thickness=1,
        )

    def draw_hud(
        self,
        frame: np.ndarray,
        score_manager: ScoreManager,
        time_left: float,
        hand_detected: bool,
        play_rect: Tuple[int, int, int, int],
    ) -> None:
        """Merender bilah HUD atas berisi skor, timer, dan status deteksi tangan."""
        px1, py1, px2, py2 = play_rect
        pw = px2 - px1

        # Informasi skor pemain
        self.draw_text_with_outline(
            frame,
            f"SKOR: {score_manager.score}",
            (px1 + 25, 120),
            scale=0.88,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            f"TERTINGGI: {score_manager.high_score}",
            (px1 + 25, 146),
            scale=0.52,
            color=(180, 180, 180),
            thickness=1,
        )

        # Indikator sisa waktu di tengah playing section
        time_color = COLOR_TIMER_NORMAL if time_left > 5.0 else COLOR_TIMER_CRITICAL
        time_text = f"WAKTU: {max(0.0, time_left):.1f}s"
        t_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)[0]
        timer_x = px1 + (pw - t_size[0]) // 2
        self.draw_text_with_outline(
            frame,
            time_text,
            (timer_x, 130),
            scale=0.95,
            color=time_color,
            thickness=2,
        )

        # Indikator status sensor tangan
        hand_color = (0, 240, 100) if hand_detected else (0, 80, 240)
        hand_text = "TANGAN: AKTIF" if hand_detected else "TANGAN: MENUNGGU"
        cv2.circle(frame, (px2 - 200, 124), 7, hand_color, -1, cv2.LINE_AA)
        self.draw_text_with_outline(
            frame,
            hand_text,
            (px2 - 185, 130),
            scale=0.52,
            color=hand_color,
            thickness=1,
        )

    def draw_holes_and_mole(
        self,
        frame: np.ndarray,
        holes: List[Hole],
        mole: Optional[Mole],
        hit_effect: Optional[HitEffect],
        current_time: float,
    ) -> None:
        """Merender struktur lubang 2x3, mole target, serta efek animasi pukulan."""
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

            if mole is not None and hole.index == mole.current_hole_idx:
                ratio = mole.get_time_ratio(current_time)
                mole_r = int(r * 0.75)

                cv2.circle(frame, (hx, hy), mole_r, COLOR_MOLE_BODY, -1, cv2.LINE_AA)
                cv2.circle(frame, (hx, hy), mole_r, COLOR_PRIMARY_CYAN, 3, cv2.LINE_AA)

                eye_offset_x = int(mole_r * 0.35)
                eye_y = hy - int(mole_r * 0.20)
                cv2.circle(frame, (hx - eye_offset_x, eye_y), 6, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx + eye_offset_x, eye_y), 6, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx - eye_offset_x + 1, eye_y - 1), 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx + eye_offset_x + 1, eye_y - 1), 2, (255, 255, 255), -1, cv2.LINE_AA)

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
                        (r + 7, r + 7),
                        -90,
                        0,
                        angle_end,
                        COLOR_PRIMARY_CYAN,
                        3,
                        cv2.LINE_AA,
                    )

        if hit_effect is not None and hit_effect.is_alive(current_time):
            progress = hit_effect.get_progress(current_time)
            pulse_radius = int(holes[0].radius * (0.8 + progress * 1.5))
            cv2.circle(frame, hit_effect.position, pulse_radius, COLOR_HIT_FLASH, 4, cv2.LINE_AA)

            float_y = hit_effect.position[1] - int(progress * 55)
            self.draw_text_with_outline(
                frame,
                "+1 HIT!",
                (hit_effect.position[0] - 50, float_y),
                scale=1.0,
                color=COLOR_HIT_FLASH,
                thickness=2,
            )

    def draw_hand_cursor(self, frame: np.ndarray, centroid: Optional[Tuple[int, int]]) -> None:
        """Merender reticle kursor bidik di titik centroid tangan pemain."""
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

    def draw_overlay_panel(
        self,
        frame: np.ndarray,
        title: str,
        subtitle_lines: List[str],
        button_prompt: str,
    ) -> None:
        """Merender panel modal dialog di tengah layar untuk Start dan Game Over."""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        box_w, box_h = int(w * 0.65), int(h * 0.60)
        bx1 = (w - box_w) // 2
        by1 = (h - box_h) // 2
        bx2, by2 = bx1 + box_w, by1 + box_h

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30, 30, 30), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_PRIMARY_CYAN, 2)

        t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
        tx = (w - t_size[0]) // 2
        self.draw_text_with_outline(
            frame,
            title,
            (tx, by1 + 60),
            scale=1.1,
            color=COLOR_PRIMARY_CYAN,
            thickness=3,
        )
        cv2.line(frame, (bx1 + 30, by1 + 80), (bx2 - 30, by1 + 80), (70, 70, 70), 1)

        curr_y = by1 + 130
        for line in subtitle_lines:
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
            curr_y += 42

        p_size = cv2.getTextSize(button_prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
        px = (w - p_size[0]) // 2
        self.draw_text_with_outline(
            frame,
            button_prompt,
            (px, by2 - 40),
            scale=0.85,
            color=COLOR_HIT_FLASH,
            thickness=2,
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
