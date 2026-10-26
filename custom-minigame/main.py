"""
Main Runner: Shinobi Jutsu Battle (Custom Game via Pure OpenCV)
Praktikum Pengolahan Citra dan Visi Komputer (PCV) - Project 1
Window: "Tugas Game - Shinobi Jutsu Battle"

Kombinasi Dual-Pane:
- Sisi Kiri (480x720): Kamera Detektor Segel Tangan OpenCV & Efek Magis
- Sisi Kanan (800x720): Arena Pertarungan 2D Pertempuran Shinobi vs Demon Boss

Kontrol Keyboard:
- 'C': Mode Kalibrasi Warna Kulit Adaptif
- 'R': Reset / Restart Pertandingan
- 'Q' atau 'ESC': Keluar dari Game
"""

import cv2
import numpy as np
import time
import math
from hand_detector import HandSealDetector
from game_engine import GameEngine


def draw_cam_pane(frame: np.ndarray, info: dict, progress: float, mask: np.ndarray, angle: float) -> np.ndarray:
    """Merender panel kamera berukuran 480x720 dengan visualisasi lingkaran sihir dan HUD segel."""
    cam_h, cam_w = 720, 480
    resized_cam = cv2.resize(frame, (cam_w, cam_h))

    seal = info.get("seal", "NONE")
    centroid = info.get("features", {}).get("centroid", None)

    # 1. Gambar Lingkaran Sihir & Kontur
    if centroid is not None and seal not in ["NONE", "UNKNOWN"]:
        # Skalakan koordinat centroid ke dimensi 480x720
        orig_h, orig_w = frame.shape[:2]
        cx = int(centroid[0] * (cam_w / orig_w))
        cy = int(centroid[1] * (cam_h / orig_h))

        color_map = {
            "TIGER": (30, 110, 255),      # Orange/Red Fire
            "SNAKE": (255, 230, 80),      # Cyan Ice
            "CROSS": (255, 80, 220),      # Magenta Lightning
            "RAM": (60, 240, 170),        # Wind Green
            "OPEN_PALM": (80, 255, 255),  # Yellow
        }
        color = color_map.get(seal, (200, 200, 200))
        rad = int(70 + 25 * progress)

        # Runic circles
        cv2.circle(resized_cam, (cx, cy), rad, color, 2, cv2.LINE_AA)
        cv2.circle(resized_cam, (cx, cy), rad - 10, color, 1, cv2.LINE_AA)
        for i in range(8):
            th = angle + (2 * math.pi / 8) * i
            x1 = int(cx + (rad - 10) * math.cos(th))
            y1 = int(cy + (rad - 10) * math.sin(th))
            x2 = int(cx + rad * math.cos(th))
            y2 = int(cy + rad * math.sin(th))
            cv2.line(resized_cam, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        if progress > 0.0:
            cv2.ellipse(resized_cam, (cx, cy), (rad + 15, rad + 15), 0, -90, -90 + int(360 * progress), (0, 255, 255), 3, cv2.LINE_AA)

    # 2. Gambar Kontur Asli (Jika ada)
    if info.get("contour") is not None:
        orig_h, orig_w = frame.shape[:2]
        cnt_scaled = info["contour"].copy()
        cnt_scaled[:, 0, 0] = (cnt_scaled[:, 0, 0] * (cam_w / orig_w)).astype(np.int32)
        cnt_scaled[:, 0, 1] = (cnt_scaled[:, 0, 1] * (cam_h / orig_h)).astype(np.int32)
        cv2.drawContours(resized_cam, [cnt_scaled], -1, (0, 230, 160), 2, cv2.LINE_AA)

    # 3. PiP Skin Mask (Pojok Kanan Atas)
    pip_w, pip_h = 130, 90
    pip_x = cam_w - pip_w - 12
    pip_y = 12
    mask_pip = cv2.cvtColor(cv2.resize(mask, (pip_w, pip_h)), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(resized_cam, (pip_x - 2, pip_y - 2), (pip_x + pip_w + 2, pip_y + pip_h + 2), (0, 215, 255), 1)
    resized_cam[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = mask_pip
    cv2.putText(resized_cam, "Skin Mask", (pip_x + 5, pip_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

    # 4. Panel Info Segel & Panduan Jurus di Bawah
    panel_y = cam_h - 180
    overlay = resized_cam[panel_y:cam_h, 0:cam_w].copy()
    cv2.rectangle(overlay, (0, 0), (cam_w, 180), (12, 14, 22), -1)
    cv2.addWeighted(overlay, 0.88, resized_cam[panel_y:cam_h, 0:cam_w], 0.12, 0, resized_cam[panel_y:cam_h, 0:cam_w])

    cv2.putText(resized_cam, f"SEAL: {seal}", (15, panel_y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 230, 255), 2, cv2.LINE_AA)

    # Meteran penahanan (Hold Progress)
    bar_w = 260
    cv2.rectangle(resized_cam, (15, panel_y + 36), (15 + bar_w, panel_y + 48), (40, 45, 60), -1)
    fill_w = int(bar_w * progress)
    if fill_w > 0:
        cv2.rectangle(resized_cam, (15, panel_y + 36), (15 + fill_w, panel_y + 48), (0, 255, 200), -1)
    cv2.rectangle(resized_cam, (15, panel_y + 36), (15 + bar_w, panel_y + 48), (180, 180, 180), 1)

    # Panduan Jurus
    guide_y = panel_y + 68
    guides = [
        ("TIGER (Harimau) -> Katon Fireball (45 DMG)", (50, 120, 255)),
        ("SNAKE (Ular/Doa) -> Hyoton Shield (+20 HP)", (255, 230, 80)),
        ("CROSS (Silang)   -> Raiton Lightning (35 DMG)", (255, 100, 220)),
        ("RAM (Kepal)      -> Rasenshuriken (70 DMG)", (60, 240, 180)),
        ("PALM (Terbuka)   -> Chakra Gatling (20 DMG)", (80, 255, 255)),
    ]
    for i, (txt, col) in enumerate(guides):
        cv2.putText(resized_cam, txt, (15, guide_y + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    return resized_cam


def main():
    window_name = "Tugas Game - Shinobi Jutsu Battle"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat diakses! Pastikan webcam terhubung.")
        return

    detector = HandSealDetector()
    engine = GameEngine(arena_w=800, arena_h=720)

    calibration_mode = False
    vfx_angle = 0.0
    last_time = time.time()
    consecutive_drop_count = 0

    print("===================================================================")
    print("      SHINOBI JUTSU BATTLE: WIZARD DUEL (PURE OPENCV)")
    print("===================================================================")
    print(" Rapalkan segel tangan di depan webcam untuk menyerang Demon Boss!")
    print(" Shortcut: [C] Kalibrasi Warna Kulit | [R] Restart Game | [Q/ESC] Keluar")
    print("===================================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_drop_count += 1
            if consecutive_drop_count > 45:
                print("[ERROR] Kamera terputus setelah 45 frame gagal.")
                break
            time.sleep(0.01)
            continue
        consecutive_drop_count = 0

        frame = cv2.flip(frame, 1)
        now = time.time()
        dt = max(0.001, min(0.1, now - last_time))
        last_time = now
        vfx_angle += 1.5 * dt

        # Mode Kalibrasi Warna Kulit
        if calibration_mode:
            h, w = frame.shape[:2]
            box_size = 140
            bx1, by1 = (w - box_size) // 2, (h - box_size) // 2
            bx2, by2 = bx1 + box_size, by1 + box_size
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            cv2.putText(frame, "Posisikan telapak tangan di kotak kuning", (bx1 - 70, by1 - 18),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 255, 255), 2)
            cv2.putText(frame, "Tekan 'C' untuk MENGUNCI KALIBRASI", (bx1 - 50, by2 + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 255, 255), 2)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('c'), ord('C')]:
                roi = frame[by1:by2, bx1:bx2]
                detector.calibrate_skin_sample(roi)
                calibration_mode = False
            elif key in [ord('q'), ord('Q'), 27]:
                break

            cv2.imshow(window_name, cv2.resize(frame, (1280, 720)))
            continue

        # Proses Deteksi Segel Tangan
        triggered_action, progress, info, mask = detector.process_frame(frame)

        # Jika segel tertahan dan terpicu -> Lempar jurus ke arena game!
        if triggered_action:
            engine.cast_player_jutsu(triggered_action)

        # Update Logika Game Arena
        engine.update()

        # Render Kedua Sisi
        cam_pane = draw_cam_pane(frame, info, progress, mask, vfx_angle)
        arena_pane = engine.render_arena()

        # Gabungkan secara horizontal menjadi frame 1280x720
        combined = np.hstack([cam_pane, arena_pane])

        # Efek Screen Shake jika terjadi benturan serangan
        if engine.screen_shake > 0:
            engine.screen_shake -= 1
            dx = np.random.randint(-6, 6)
            dy = np.random.randint(-6, 6)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            combined = cv2.warpAffine(combined, M, (1280, 720))

        # Tampilkan ke Jendela Tunggal "Tugas Game - Shinobi Jutsu Battle"
        cv2.imshow(window_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break
        elif key in [ord('c'), ord('C')]:
            calibration_mode = True
        elif key in [ord('r'), ord('R')]:
            engine.reset()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Game Shinobi Jutsu Battle selesai ditutup.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL CRASH] Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nTekan Enter untuk keluar...")
