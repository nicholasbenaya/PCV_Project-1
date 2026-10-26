"""Game Whack-A-Mole Virtual berbasis OpenCV murni (Resolusi HD 1280x720).

Dilengkapi sistem Dual-Section:
1. Playing Section: Area interaktif tempat mole dan lubang berada; deteksi tangan aktif di sini.
2. Non-Playing Section: Area tempat kepala dan tubuh pemain berada; deteksi dinolkan 100%
   sehingga wajah/kepala mustahil membajak kursor permainan.
"""

import math
import random
import time
import cv2
import numpy as np

# Resolusi kanvas diperbesar ke resolusi HD widescreen
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

# Batas segmentasi warna kulit manusia pada HSV (Hue di OpenCV: 0..179)
LOWER_SKIN_1 = np.array([0, 30, 50], dtype=np.uint8)
UPPER_SKIN_1 = np.array([25, 210, 255], dtype=np.uint8)
LOWER_SKIN_2 = np.array([170, 30, 50], dtype=np.uint8)
UPPER_SKIN_2 = np.array([180, 210, 255], dtype=np.uint8)

# State permainan
STATE_START = 0
STATE_PLAYING = 1
STATE_GAMEOVER = 2

# Parameter gameplay
GAME_DURATION = 30.0  # Durasi permainan per sesi (detik)
HIT_COOLDOWN = 0.38  # Waktu debounce pukulan (detik)
MOLE_MIN_TIME = 1.40  # Batas bawah durasi kemunculan mole (detik)
MOLE_MAX_TIME = 1.90  # Batas atas durasi kemunculan mole (detik)
MIN_HAND_AREA = 1800  # Luas piksel minimal tangan di playing section

# Preset tata letak section
ZONE_PRESETS = [
    {
        "name": "TANGAN KANAN (Kiri: Wajah Diblokir)",
        "play_rect": (420, 0, 1280, 720),
        "safe_rect": (0, 0, 420, 720),
        "split_line": ((420, 0), (420, 720)),
    },
    {
        "name": "TANGAN KIRI (Kanan: Wajah Diblokir)",
        "play_rect": (0, 0, 860, 720),
        "safe_rect": (860, 0, 1280, 720),
        "split_line": ((860, 0), (860, 720)),
    },
    {
        "name": "TANGAN BAWAH (Atas: Wajah Diblokir)",
        "play_rect": (0, 240, 1280, 720),
        "safe_rect": (0, 0, 1280, 240),
        "split_line": ((0, 240), (1280, 240)),
    },
]


def extract_hand_in_playing_section(frame, play_rect, prev_centroid=None, smoothing=0.65):
    """Mengekstrak centroid tangan HANYA di dalam Playing Section.

    Seluruh area di luar Playing Section (Non-Playing Section tempat kepala/tubuh berada)
    dihapus dan dinolkan secara mutlak dari masker biner sehingga wajah tidak terdeteksi.
    """
    px1, py1, px2, py2 = play_rect

    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_SKIN_1, UPPER_SKIN_1)
    mask2 = cv2.inRange(hsv, LOWER_SKIN_2, UPPER_SKIN_2)
    raw_mask = cv2.bitwise_or(mask1, mask2)

    # Nolkan seluruh area di luar Playing Section (Safe Zone Exclusion)
    isolated_mask = np.zeros_like(raw_mask)
    isolated_mask[py1:py2, px1:px2] = raw_mask[py1:py2, px1:px2]

    # Pembersihan morfologi
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(isolated_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    cleaned_mask = cv2.dilate(cleaned_mask, kernel_open, iterations=1)

    # Ekstraksi kontur eksternal di dalam playing section
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < MIN_HAND_AREA or area > 120000:
            continue

        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # Skor kontur: prioritas tangan dan kontinuitas temporal
        score = min(area, 30000.0) * 0.1
        if prev_centroid is not None:
            dist = math.hypot(cx - prev_centroid[0], cy - prev_centroid[1])
            if dist < 250:
                score += (250 - dist) * 20.0

        candidates.append((score, cnt, (cx, cy), area))

    if not candidates:
        return None, cleaned_mask, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_contour, (raw_cx, raw_cy), max_area = candidates[0]

    # Exponential moving average untuk stabilisasi posisi kursor
    if prev_centroid is not None:
        cx = int(smoothing * raw_cx + (1.0 - smoothing) * prev_centroid[0])
        cy = int(smoothing * raw_cy + (1.0 - smoothing) * prev_centroid[1])
    else:
        cx, cy = raw_cx, raw_cy

    return (cx, cy), cleaned_mask, best_contour


def compute_grid_holes_in_section(play_rect):
    """Menghitung koordinat 6 lubang (grid 2 baris x 3 kolom) di dalam Playing Section."""
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

    holes = []
    for y in ys:
        for x in xs:
            holes.append((x, y))

    radius = int(min(pw, ph) * 0.12)
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


def draw_sections_overlay(frame, zone_cfg):
    """Merender pembagian 2 section visual (Non-Playing Safe Zone & Playing Section)."""
    sx1, sy1, sx2, sy2 = zone_cfg["safe_rect"]
    px1, py1, px2, py2 = zone_cfg["play_rect"]

    # Redupkan Non-Playing Section tempat kepala/wajah berada
    safe_region = frame[sy1:sy2, sx1:sx2]
    dimmed_safe = (safe_region * 0.45).astype(np.uint8)
    frame[sy1:sy2, sx1:sx2] = dimmed_safe

    # Garis pembatas section
    p_start, p_end = zone_cfg["split_line"]
    cv2.line(frame, p_start, p_end, (0, 220, 255), 3, cv2.LINE_AA)

    # Label pada Non-Playing Section
    safe_cx = sx1 + (sx2 - sx1) // 2
    draw_text_with_outline(
        frame,
        "NON-PLAYING SECTION",
        (sx1 + 25, 45),
        scale=0.68,
        color=(180, 190, 210),
        thickness=2,
    )
    draw_text_with_outline(
        frame,
        "[SAFE ZONE] Wajah & Tubuh Terblokir 100%",
        (sx1 + 25, 75),
        scale=0.48,
        color=(0, 255, 120),
        thickness=1,
    )
    draw_text_with_outline(
        frame,
        "Kepala Anda berada di sini dengan aman tanpa mengganggu kursor",
        (sx1 + 25, 102),
        scale=0.40,
        color=(160, 160, 160),
        thickness=1,
    )

    # Label pada Playing Section
    draw_text_with_outline(
        frame,
        "PLAYING SECTION [ZONA TANGAN AKTIF]",
        (px1 + 25, 45),
        scale=0.72,
        color=(0, 220, 255),
        thickness=2,
    )
    draw_text_with_outline(
        frame,
        f"Tata Letak: {zone_cfg['name']}  |  [H] Ganti Sisi / Zona",
        (px1 + 25, 75),
        scale=0.48,
        color=(255, 220, 100),
        thickness=1,
    )


def draw_hud(frame, score, time_left, high_score, hand_detected, play_rect):
    """Merender bilah HUD skor, timer, dan status di dalam Playing Section."""
    px1, py1, px2, py2 = play_rect
    pw = px2 - px1

    # Teks Skor
    draw_text_with_outline(
        frame,
        f"SKOR: {score}",
        (px1 + 25, 120),
        scale=0.88,
        color=(0, 230, 255),
        thickness=2,
    )
    draw_text_with_outline(
        frame,
        f"TERTINGGI: {high_score}",
        (px1 + 25, 146),
        scale=0.52,
        color=(180, 180, 180),
        thickness=1,
    )

    # Countdown timer di tengah playing section
    time_color = (0, 235, 0) if time_left > 5.0 else (0, 0, 245)
    time_text = f"WAKTU: {max(0.0, time_left):.1f}s"
    t_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)[0]
    timer_x = px1 + (pw - t_size[0]) // 2
    draw_text_with_outline(
        frame,
        time_text,
        (timer_x, 130),
        scale=0.95,
        color=time_color,
        thickness=2,
    )

    # Status Deteksi Tangan
    hand_color = (0, 240, 100) if hand_detected else (0, 80, 240)
    hand_text = "TANGAN: AKTIF" if hand_detected else "TANGAN: MENUNGGU"
    cv2.circle(frame, (px2 - 200, 124), 7, hand_color, -1, cv2.LINE_AA)
    draw_text_with_outline(
        frame,
        hand_text,
        (px2 - 185, 130),
        scale=0.52,
        color=hand_color,
        thickness=1,
    )


def draw_holes_and_mole(frame, holes, radius, active_idx, mole_spawn_time, mole_lifespan, hit_effect):
    """Merender lubang galian, mole aktif, serta animasi pukulan."""
    current_time = time.time()

    for idx, (hx, hy) in enumerate(holes):
        # Lubang dasar
        cv2.circle(frame, (hx, hy), radius, (35, 35, 35), -1, cv2.LINE_AA)
        cv2.circle(frame, (hx, hy), radius, (80, 80, 80), 3, cv2.LINE_AA)
        cv2.ellipse(
            frame,
            (hx, hy + int(radius * 0.45)),
            (int(radius * 0.70), int(radius * 0.35)),
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
                    (radius + 7, radius + 7),
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
                scale=1.0,
                color=(0, 255, 100),
                thickness=2,
            )


def draw_hand_cursor(frame, centroid):
    """Merender reticle penanda posisi tangan pemain di dalam Playing Section."""
    if centroid is None:
        return

    cx, cy = centroid
    cv2.circle(frame, (cx, cy), 22, (255, 220, 0), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1, cv2.LINE_AA)

    arm = 14
    cv2.line(frame, (cx - 22 - arm, cy), (cx - 22, cy), (255, 220, 0), 2, cv2.LINE_AA)
    cv2.line(frame, (cx + 22, cy), (cx + 22 + arm, cy), (255, 220, 0), 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 22 - arm), (cx, cy - 22), (255, 220, 0), 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + 22), (cx, cy + 22 + arm), (255, 220, 0), 2, cv2.LINE_AA)


def draw_overlay_panel(frame, title, subtitle_lines, button_prompt):
    """Merender modal panel dialog untuk layar awal dan game over."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    box_w, box_h = int(w * 0.65), int(h * 0.60)
    bx1 = (w - box_w) // 2
    by1 = (h - box_h) // 2
    bx2, by2 = bx1 + box_w, by1 + box_h

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30, 30, 30), -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 255), 2)

    t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
    tx = (w - t_size[0]) // 2
    draw_text_with_outline(
        frame,
        title,
        (tx, by1 + 60),
        scale=1.1,
        color=(0, 220, 255),
        thickness=3,
    )
    cv2.line(frame, (bx1 + 30, by1 + 80), (bx2 - 30, by1 + 80), (70, 70, 70), 1)

    curr_y = by1 + 130
    for line in subtitle_lines:
        l_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0]
        lx = (w - l_size[0]) // 2
        draw_text_with_outline(
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
    draw_text_with_outline(
        frame,
        button_prompt,
        (px, by2 - 40),
        scale=0.85,
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

    # Inisialisasi kamera webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Whack-A-Mole Virtual (HD 1280x720 - Dual Section)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    zone_idx = 0
    active_zone = ZONE_PRESETS[zone_idx]

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
        ret, raw_frame = cap.read()
        if not ret:
            print("Gagal membaca frame kamera.")
            break

        # Flip horizontal (cermin natural)
        flipped = cv2.flip(raw_frame, 1)

        # Pastikan ukuran kanvas selalu HD 1280x720
        frame = cv2.resize(flipped, (CANVAS_WIDTH, CANVAS_HEIGHT))

        # Tata letak lubang di dalam Playing Section
        holes, radius = compute_grid_holes_in_section(active_zone["play_rect"])

        # Deteksi tangan HANYA di dalam Playing Section (Non-playing zone dinolkan)
        centroid, skin_mask, hand_contour = extract_hand_in_playing_section(
            frame,
            active_zone["play_rect"],
            smoothed_centroid,
        )
        smoothed_centroid = centroid
        current_time = time.time()

        # Render background dan pembagian section visual
        draw_sections_overlay(frame, active_zone)

        if state == STATE_START:
            draw_holes_and_mole(frame, holes, radius, -1, 0, 1, None)
            draw_hud(frame, 0, GAME_DURATION, high_score, centroid is not None, active_zone["play_rect"])
            draw_hand_cursor(frame, centroid)
            draw_overlay_panel(
                frame,
                "WHACK-A-MOLE VIRTUAL (HD DUAL-SECTION)",
                [
                    "Sisi Kiri adalah Safe Zone: Kepala & tubuh Anda terblokir 100%.",
                    "Arahkan telapak tangan Anda ke Playing Section di sisi kanan.",
                    "Pukul tepat di atas mole untuk mengumpulkan skor.",
                    "[H] Ganti Sisi Tangan (Kanan/Kiri/Bawah) | [M] Toggle Mask | [Q] Keluar",
                ],
                "Tekan [SPASI] untuk Mulai Bermain",
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
            draw_hud(
                frame,
                score,
                time_left,
                high_score,
                centroid is not None,
                active_zone["play_rect"],
            )
            draw_hand_cursor(frame, centroid)

        elif state == STATE_GAMEOVER:
            draw_holes_and_mole(frame, holes, radius, -1, 0, 1, None)
            draw_hud(
                frame,
                score,
                0.0,
                high_score,
                centroid is not None,
                active_zone["play_rect"],
            )
            draw_overlay_panel(
                frame,
                "GAME OVER",
                [
                    f"Skor Akhir Anda: {score}",
                    f"Rekor Tertinggi: {high_score}",
                    "Tangan Anda terdeteksi dengan sangat akurat tanpa gangguan wajah!",
                    "[Q] Keluar dari Permainan",
                ],
                "Tekan [R] untuk Main Lagi",
            )

        # Inset Picture-in-Picture untuk monitoring masker biner di pojok
        if show_mask_pip:
            pip_w, pip_h = 200, 150
            pip_x, pip_y = CANVAS_WIDTH - pip_w - 20, CANVAS_HEIGHT - pip_h - 20
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
                "Mask Kulit (Playing Only)",
                (pip_x + 5, pip_y + 20),
                scale=0.45,
                color=(0, 255, 255),
                thickness=1,
            )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("h"), ord("H"), 9):  # 'H' atau TAB untuk ganti sisi Playing Section
            zone_idx = (zone_idx + 1) % len(ZONE_PRESETS)
            active_zone = ZONE_PRESETS[zone_idx]
            smoothed_centroid = None
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
