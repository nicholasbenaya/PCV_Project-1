"""Modul rendering grafis antarmuka (UI) dan elemen visual game berbasis OpenCV."""

from typing import List, Optional, Tuple
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
    """Pengelola rendering seluruh antarmuka grafis dan elemen visual game."""

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

    def draw_hud(
        self,
        frame: np.ndarray,
        score_manager: ScoreManager,
        time_left: float,
        hand_detected: bool,
    ) -> None:
        """Merender bilah HUD atas berisi skor, timer, dan status deteksi tangan."""
        h, w = frame.shape[:2]
        hud_h = 75

        # Latar HUD semi-transparan menggunakan alpha blending
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (25, 25, 25), -1)
        cv2.line(overlay, (0, hud_h), (w, hud_h), (60, 60, 60), 2)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Informasi skor pemain
        self.draw_text_with_outline(
            frame,
            f"SKOR: {score_manager.score}",
            (25, 36),
            scale=0.85,
            color=COLOR_PRIMARY_CYAN,
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            f"TERTINGGI: {score_manager.high_score}",
            (25, 62),
            scale=0.55,
            color=(180, 180, 180),
            thickness=1,
        )

        # Indikator sisa waktu
        time_color = COLOR_TIMER_NORMAL if time_left > 5.0 else COLOR_TIMER_CRITICAL
        time_text = f"WAKTU: {max(0.0, time_left):.1f}s"
        text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        timer_x = (w - text_size[0]) // 2
        self.draw_text_with_outline(
            frame,
            time_text,
            (timer_x, 48),
            scale=0.9,
            color=time_color,
            thickness=2,
        )

        # Indikator status sensor tangan
        hand_color = (0, 240, 100) if hand_detected else (0, 80, 240)
        hand_text = "TANGAN: AKTIF" if hand_detected else "TANGAN: TIDAK TERDETEKSI"
        cv2.circle(frame, (w - 230, 42), 7, hand_color, -1, cv2.LINE_AA)
        self.draw_text_with_outline(
            frame,
            hand_text,
            (w - 215, 48),
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

            # Lubang dasar dengan efek kedalaman
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

            # Gambar mole bila berada di lubang ini
            if mole is not None and hole.index == mole.current_hole_idx:
                ratio = mole.get_time_ratio(current_time)
                mole_r = int(r * 0.75)

                # Badan mole
                cv2.circle(frame, (hx, hy), mole_r, COLOR_MOLE_BODY, -1, cv2.LINE_AA)
                cv2.circle(frame, (hx, hy), mole_r, COLOR_PRIMARY_CYAN, 3, cv2.LINE_AA)

                # Wajah mole (mata)
                eye_offset_x = int(mole_r * 0.35)
                eye_y = hy - int(mole_r * 0.20)
                cv2.circle(frame, (hx - eye_offset_x, eye_y), 5, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx + eye_offset_x, eye_y), 5, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx - eye_offset_x + 1, eye_y - 1), 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx + eye_offset_x + 1, eye_y - 1), 2, (255, 255, 255), -1, cv2.LINE_AA)

                # Moncong dan hidung
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

                # Indikator busur lingkaran waktu tersisa
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

        # Animasi feedback visual pukulan
        if hit_effect is not None and hit_effect.is_alive(current_time):
            progress = hit_effect.get_progress(current_time)
            pulse_radius = int(holes[0].radius * (0.8 + progress * 1.5))
            cv2.circle(frame, hit_effect.position, pulse_radius, COLOR_HIT_FLASH, 4, cv2.LINE_AA)

            float_y = hit_effect.position[1] - int(progress * 55)
            self.draw_text_with_outline(
                frame,
                "+1 HIT!",
                (hit_effect.position[0] - 50, float_y),
                scale=0.9,
                color=COLOR_HIT_FLASH,
                thickness=2,
            )

    def draw_hand_cursor(self, frame: np.ndarray, centroid: Optional[Tuple[int, int]]) -> None:
        """Merender reticle kursor bidik di titik centroid tangan pemain."""
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

        box_w, box_h = int(w * 0.72), int(h * 0.65)
        bx1 = (w - box_w) // 2
        by1 = (h - box_h) // 2
        bx2, by2 = bx1 + box_w, by1 + box_h

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30, 30, 30), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_PRIMARY_CYAN, 2)

        # Header judul panel
        t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        tx = (w - t_size[0]) // 2
        self.draw_text_with_outline(
            frame,
            title,
            (tx, by1 + 55),
            scale=1.0,
            color=COLOR_PRIMARY_CYAN,
            thickness=3,
        )
        cv2.line(frame, (bx1 + 30, by1 + 75), (bx2 - 30, by1 + 75), (70, 70, 70), 1)

        # Isi baris teks informasi
        curr_y = by1 + 120
        for line in subtitle_lines:
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
            curr_y += 38

        # Tombol aksi di bagian bawah panel
        p_size = cv2.getTextSize(button_prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        px = (w - p_size[0]) // 2
        self.draw_text_with_outline(
            frame,
            button_prompt,
            (px, by2 - 35),
            scale=0.75,
            color=COLOR_HIT_FLASH,
            thickness=2,
        )

    def draw_mask_pip(self, frame: np.ndarray, mask: np.ndarray) -> None:
        """Menampilkan jendela kecil Picture-in-Picture untuk memonitor mask segmentasi."""
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
