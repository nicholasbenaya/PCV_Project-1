"""Game Whack-A-Mole Virtual berbasis OpenCV murni (Skrip Tunggal / Single Script).

Versi awal tanpa Main Menu untuk pengujian langsung.
Menggunakan segmentasi warna kulit HSV dan moments kontur (cv2.moments).
"""

import math
import random
import time
import cv2
import numpy as np

# Batas segmentasi warna kulit manusia pada HSV (Hue di OpenCV: 0..179).
LOWER_SKIN_1 = np.array([0, 35, 60], dtype=np.uint8)
UPPER_SKIN_1 = np.array([22, 190, 255], dtype=np.uint8)
LOWER_SKIN_2 = np.array([170, 35, 60], dtype=np.uint8)
UPPER_SKIN_2 = np.array([180, 190, 255], dtype=np.uint8)

# State game
STATE_START = 0
STATE_PLAYING = 1
STATE_GAMEOVER = 2

# Parameter gameplay
GAME_DURATION = 30.0  # Durasi permainan per sesi (detik)
HIT_COOLDOWN = 0.40  # Waktu debounce pukulan (detik)
MOLE_MIN_TIME = 1.50  # Batas bawah durasi kemunculan mole (detik)
MOLE_MAX_TIME = 2.00  # Batas atas durasi kemunculan mole (detik)
MIN_HAND_AREA = 2200  # Area kontur minimum untuk diakui sebagai tangan pemain


def extract_hand_centroid(frame, prev_centroid=None, smoothing=0.65):
    """Mengekstrak titik tengah (centroid) tangan menggunakan segmentasi HSV dan moments."""
    # Reduksi noise sensor webcam menggunakan Gaussian Blur
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Segmentasi rentang warna kulit
    mask1 = cv2.inRange(hsv, LOWER_SKIN_1, UPPER_SKIN_1)
    mask2 = cv2.inRange(hsv, LOWER_SKIN_2, UPPER_SKIN_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Pembersihan noise mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Deteksi kontur eksternal
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_contour = cnt

    if largest_contour is None or max_area < MIN_HAND_AREA:
        return None, mask, None

    # Perhitungan titik berat (centroid) kontur tangan menggunakan moments
    moments = cv2.moments(largest_contour)
    if moments["m00"] == 0:
        return None, mask, largest_contour

    raw_cx = int(moments["m10"] / moments["m00"])
    raw_cy = int(moments["m01"] / moments["m00"])

    # Exponential moving average untuk stabilisasi posisi kursor
    if prev_centroid is not None:
        cx = int(smoothing * raw_cx + (1.0 - smoothing) * prev_centroid[0])
        cy = int(smoothing * raw_cy + (1.0 - smoothing) * prev_centroid[1])
    else:
        cx, cy = raw_cx, raw_cy

    return (cx, cy), mask, largest_contour


def compute_grid_holes(frame_width, frame_height):
    """Menghitung koordinat tengah 6 lubang (grid 2 baris x 3 kolom)."""
    xs = [int(frame_width * 0.22), int(frame_width * 0.50), int(frame_width * 0.78)]
    ys = [int(frame_height * 0.42), int(frame_height * 0.76)]

    holes = []
    for y in ys:
        for x in xs:
            holes.append((x, y))

    radius = int(min(frame_width, frame_height) * 0.11)
    return holes, radius


def draw_text_with_outline(img, text, pos, scale=0.7, color=(255, 255, 255), thickness=2):
    """Menggambar teks dengan outline gelap agar kontras di atas feed webcam."""
    x, y = pos
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (20, 20, 20),
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


def draw_hud(frame, score, time_left, high_score, hand_detected):
    """Merender bilah HUD atas berisi skor, sisa waktu, dan status pelacakan tangan."""
    h, w = frame.shape[:2]
    hud_h = 75

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (25, 25, 25), -1)
    cv2.line(overlay, (0, hud_h), (w, hud_h), (60, 60, 60), 2)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    draw_text_with_outline(
        frame,
        f"SKOR: {score}",
        (25, 36),
        scale=0.85,
        color=(0, 230, 255),
        thickness=2,
    )
    draw_text_with_outline(
        frame,
        f"TERTINGGI: {high_score}",
        (25, 62),
        scale=0.55,
        color=(180, 180, 180),
        thickness=1,
    )

    time_color = (0, 235, 0) if time_left > 5 else (0, 0, 245)
    time_text = f"WAKTU: {max(0.0, time_left):.1f}s"
    text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    timer_x = (w - text_size[0]) // 2
    draw_text_with_outline(
        frame,
        time_text,
        (timer_x, 48),
        scale=0.9,
        color=time_color,
        thickness=2,
    )

    hand_color = (0, 240, 100) if hand_detected else (0, 80, 240)
    hand_text = "TANGAN: AKTIF" if hand_detected else "TANGAN: TIDAK TERDETEKSI"
    cv2.circle(frame, (w - 230, 42), 7, hand_color, -1, cv2.LINE_AA)
    draw_text_with_outline(
        frame,
        hand_text,
        (w - 215, 48),
        scale=0.52,
        color=hand_color,
        thickness=1,
    )


def draw_holes_and_mole(frame, holes, radius, active_idx, mole_spawn_time, mole_lifespan, hit_effect):
    """Merender lubang galian, mole aktif, serta animasi pukulan."""
    current_time = time.time()

    for idx, (hx, hy) in enumerate(holes):
        cv2.circle(frame, (hx, hy), radius, (35, 35, 35), -1, cv2.LINE_AA)
        cv2.circle(frame, (hx, hy), radius, (80, 80, 80), 3, cv2.LINE_AA)
        cv2.ellipse(
            frame,
            (hx, hy + int(radius * 0.45)),
            (int(radius * 0.7), int(radius * 0.35)),
            0,
            0,
            360,
            (20, 20, 20),
            -1,
            cv2.LINE_AA,
        )

        if idx == active_idx:
            elapsed = current_time - mole_spawn_time
            ratio = max(0.0, min(1.0, 1.0 - (elapsed / mole_lifespan)))

            mole_r = int(radius * 0.75)
            cv2.circle(frame, (hx, hy), mole_r, (40, 120, 225), -1, cv2.LINE_AA)
            cv2.circle(frame, (hx, hy), mole_r, (0, 200, 255), 3, cv2.LINE_AA)

            eye_offset_x = int(mole_r * 0.35)
            eye_y = hy - int(mole_r * 0.20)
            cv2.circle(frame, (hx - eye_offset_x, eye_y), 5, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(frame, (hx + eye_offset_x, eye_y), 5, (20, 20, 20), -1, cv2.LINE_AA)
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
            cv2.circle(frame, (hx, hy + int(mole_r * 0.08)), 4, (30, 30, 30), -1, cv2.LINE_AA)

            angle_end = int(360 * ratio)
            if angle_end > 0:
                cv2.ellipse(
                    frame,
                    (hx, hy),
                    (radius + 6, radius + 6),
                    -90,
                    0,
                    angle_end,
                    (0, 240, 255),
                    3,
                    cv2.LINE_AA,
                )

    if hit_effect is not None:
        eff_pos, eff_time = hit_effect
        age = current_time - eff_time
        if age < 0.45:
            pulse_r = int(radius * (0.8 + age * 1.5))
            alpha_color = (0, 255, 120)
            cv2.circle(frame, eff_pos, pulse_r, alpha_color, 4, cv2.LINE_AA)

            float_y = eff_pos[1] - int(age * 55)
            draw_text_with_outline(
                frame,
                "+1 HIT!",
                (eff_pos[0] - 50, float_y),
                scale=0.9,
                color=(0, 255, 100),
                thickness=2,
            )


def draw_hand_cursor(frame, centroid):
    """Merender reticle penanda posisi tangan pemain."""
    if centroid is None:
        return

    cx, cy = centroid
    cv2.circle(frame, (cx, cy), 18, (255, 220, 0), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)

    arm = 12
    cv2.line(frame, (cx - 18 - arm, cy), (cx - 18, cy), (255, 220, 0), 2, cv2.LINE_AA)
    cv2.line(frame, (cx + 18, cy), (cx + 18 + arm, cy), (255, 220, 0), 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 18 - arm), (cx, cy - 18), (255, 220, 0), 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + 18), (cx, cy + 18 + arm), (255, 220, 0), 2, cv2.LINE_AA)


def draw_overlay_panel(frame, title, subtitle_lines, button_prompt):
    """Merender modal panel dialog untuk layar awal dan game over."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    box_w, box_h = int(w * 0.72), int(h * 0.65)
    bx1 = (w - box_w) // 2
    by1 = (h - box_h) // 2
    bx2, by2 = bx1 + box_w, by1 + box_h

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30, 30, 30), -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 255), 2)

    t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
    tx = (w - t_size[0]) // 2
    draw_text_with_outline(
        frame,
        title,
        (tx, by1 + 55),
        scale=1.0,
        color=(0, 220, 255),
        thickness=3,
    )
    cv2.line(frame, (bx1 + 30, by1 + 75), (bx2 - 30, by1 + 75), (70, 70, 70), 1)

    curr_y = by1 + 120
    for line in subtitle_lines:
        l_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0]
        lx = (w - l_size[0]) // 2
        draw_text_with_outline(
            frame,
            line,
            (lx, curr_y),
            scale=0.62,
            color=(220, 220, 220),
            thickness=1,
        )
        curr_y += 38

    p_size = cv2.getTextSize(button_prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
    px = (w - p_size[0]) // 2
    draw_text_with_outline(
        frame,
        button_prompt,
        (px, by2 - 35),
        scale=0.75,
        color=(0, 255, 120),
        thickness=2,
    )


def choose_next_hole(current_idx):
    """Memilih indeks lubang baru secara acak yang berbeda dari lubang saat ini."""
    choices = [i for i in range(6) if i != current_idx]
    return random.choice(choices)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Whack-A-Mole Virtual (OpenCV - Standalone v1)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    state = STATE_START
    score = 0
    high_score = 0
    game_start_time = 0.0

    active_hole_idx = 0
    mole_spawn_time = 0.0
    mole_lifespan = random.uniform(MOLE_MIN_TIME, MOLE_MAX_TIME)

    last_hit_time = 0.0
    hit_effect = None

    smoothed_centroid = None
    show_mask_pip = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame kamera.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        holes, radius = compute_grid_holes(w, h)

        centroid, skin_mask, hand_contour = extract_hand_centroid(frame, smoothed_centroid)
        smoothed_centroid = centroid
        current_time = time.time()

        if state == STATE_START:
            draw_holes_and_mole(frame, holes, radius, -1, 0, 1, None)
            draw_hud(frame, 0, GAME_DURATION, high_score, centroid is not None)
            draw_hand_cursor(frame, centroid)
            draw_overlay_panel(
                frame,
                "WHACK-A-MOLE VIRTUAL (V1)",
                [
                    "Arahkan telapak tangan ke mole yang muncul di layar.",
                    "Pukul tepat di atas mole untuk mengumpulkan skor.",
                    "Satu sesi berlangsung selama 30 detik.",
                    "[M] Toggle Mask Segmentasi | [Q] Keluar",
                ],
                "Tekan [SPASI] untuk Mulai",
            )

        elif state == STATE_PLAYING:
            elapsed = current_time - game_start_time
            time_left = GAME_DURATION - elapsed

            if time_left <= 0:
                time_left = 0
                state = STATE_GAMEOVER
                if score > high_score:
                    high_score = score
            else:
                if (current_time - mole_spawn_time) >= mole_lifespan:
                    active_hole_idx = choose_next_hole(active_hole_idx)
                    mole_spawn_time = current_time
                    mole_lifespan = random.uniform(MOLE_MIN_TIME, MOLE_MAX_TIME)

                if centroid is not None:
                    target_x, target_y = holes[active_hole_idx]
                    distance = math.hypot(centroid[0] - target_x, centroid[1] - target_y)

                    if distance <= radius and (current_time - last_hit_time) >= HIT_COOLDOWN:
                        score += 1
                        last_hit_time = current_time
                        hit_effect = ((target_x, target_y), current_time)

                        active_hole_idx = choose_next_hole(active_hole_idx)
                        mole_spawn_time = current_time
                        mole_lifespan = random.uniform(MOLE_MIN_TIME, MOLE_MAX_TIME)

            draw_holes_and_mole(
                frame,
                holes,
                radius,
                active_hole_idx,
                mole_spawn_time,
                mole_lifespan,
                hit_effect,
            )
            draw_hud(frame, score, time_left, high_score, centroid is not None)
            draw_hand_cursor(frame, centroid)

        elif state == STATE_GAMEOVER:
            draw_holes_and_mole(frame, holes, radius, -1, 0, 1, None)
            draw_hud(frame, score, 0.0, high_score, centroid is not None)
            draw_overlay_panel(
                frame,
                "GAME OVER",
                [
                    f"Skor Akhir Anda: {score}",
                    f"Rekor Tertinggi: {high_score}",
                    "Gerakan tangan Anda terdeteksi dengan baik!",
                    "[Q] Keluar dari Permainan",
                ],
                "Tekan [R] untuk Main Lagi",
            )

        if show_mask_pip:
            pip_w, pip_h = 160, 120
            pip_x, pip_y = w - pip_w - 15, h - pip_h - 15
            mask_bgr = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
            small_mask = cv2.resize(mask_bgr, (pip_w, pip_h))
            cv2.rectangle(
                frame,
                (pip_x - 2, pip_y - 2),
                (pip_x + pip_w + 2, pip_y + pip_h + 2),
                (0, 200, 255),
                2,
            )
            frame[pip_y : pip_y + pip_h, pip_x : pip_x + pip_w] = small_mask
            draw_text_with_outline(
                frame,
                "Mask Kulit",
                (pip_x + 5, pip_y + 18),
                scale=0.45,
                color=(0, 255, 255),
                thickness=1,
            )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key == ord(" "):
            if state == STATE_START:
                state = STATE_PLAYING
                score = 0
                game_start_time = time.time()
                active_hole_idx = random.randint(0, 5)
                mole_spawn_time = time.time()
                mole_lifespan = random.uniform(MOLE_MIN_TIME, MOLE_MAX_TIME)
        elif key in (ord("r"), ord("R")):
            if state == STATE_GAMEOVER:
                state = STATE_PLAYING
                score = 0
                game_start_time = time.time()
                active_hole_idx = random.randint(0, 5)
                mole_spawn_time = time.time()
                mole_lifespan = random.uniform(MOLE_MIN_TIME, MOLE_MAX_TIME)
        elif key in (ord("m"), ord("M")):
            show_mask_pip = not show_mask_pip

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
