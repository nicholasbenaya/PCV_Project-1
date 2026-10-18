"""
Main Runner: Dota 2 Invoker Naruto Hand Seals Detector
Praktikum Pengolahan Citra dan Visi Komputer (PCV) - Project 1
Window: "Tugas Game - Dota 2 Invoker"

Navigasi Kontrol Keyboard:
- 'C': Mode Kalibrasi Warna Kulit (Letakkan tangan di kotak tengah, lalu tekan 'C')
- 'I': Toggle DirectInput Hardware Key Injection (LIVE / SIMULATION)
- 'M': Toggle Tampilan Masker Biner Hasil Morfologi (PiP)
- 'R': Reset Orb Invoker ke Default (EEE)
- 'Q' atau 'ESC': Keluar dari Aplikasi
"""

import cv2
import numpy as np
import time
from hand_detector import HandSealDetector
from vfx_overlay import VFXOverlay
from invoker_controller import InvokerController


def main():
    window_name = "Tugas Game - Dota 2 Invoker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat diakses! Pastikan webcam terhubung.")
        return

    # Inisialisasi Modul
    detector = HandSealDetector()
    vfx = VFXOverlay()
    invoker = InvokerController(direct_input_enabled=True)

    # Status Kontrol
    show_mask_pip = True
    calibration_mode = False
    fps_time = time.time()
    fps = 0.0

    print("===================================================================")
    print("      DOTA 2 INVOKER NARUTO HAND SEALS (PURE OPENCV)")
    print("===================================================================")
    print(" - TIGER SEAL     -> EXORT (E)    [Elemen Api]")
    print(" - SNAKE SEAL     -> QUAS (Q)     [Elemen Es]")
    print(" - CROSS/BIRD     -> WEX (W)      [Elemen Petir]")
    print(" - RAM/FIST       -> INVOKE (R)   [Rapal Mantra]")
    print(" - OPEN PALM      -> QUICK CAST D [Mantra Slot 1]")
    print(" - DUAL PALM      -> QUICK CAST F [Mantra Slot 2]")
    print("-------------------------------------------------------------------")
    print(" Shortcut: [C] Kalibrasi | [I] Toggle DirectInput | [M] Mask PiP | [Q] Keluar")
    print("===================================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Cermin horizontal agar gerakan tangan intuitif
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Hitung FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, now - fps_time))
        fps_time = now

        # Mode Kalibrasi Warna Kulit
        if calibration_mode:
            box_size = 140
            bx1 = (w - box_size) // 2
            by1 = (h - box_size) // 2
            bx2 = bx1 + box_size
            by2 = by1 + box_size

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            cv2.putText(frame, "Letakkan telapak tangan di dalam kotak kuning", (bx1 - 80, by1 - 18),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Tekan 'C' untuk MENGUNCI KALIBRASI WARNA", (bx1 - 70, by2 + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('c'), ord('C')]:
                roi = frame[by1:by2, bx1:bx2]
                detector.calibrate_skin_sample(roi)
                calibration_mode = False
            elif key in [ord('q'), ord('Q'), 27]:
                break

            cv2.imshow(window_name, frame)
            continue

        # Proses Deteksi Tangan & Segel (Murni OpenCV)
        triggered_action, progress, info, mask = detector.process_frame(frame)

        # Tangani Aksi yang Terpicu
        if triggered_action:
            cx, cy = info["features"].get("centroid", (w // 2, h // 2))

            if triggered_action == "TIGER":
                invoker.add_orb("E")
                vfx.trigger_spell_burst(cx, cy, "EXORT (E)", (40, 100, 255))
            elif triggered_action == "SNAKE":
                invoker.add_orb("Q")
                vfx.trigger_spell_burst(cx, cy, "QUAS (Q)", (255, 210, 80))
            elif triggered_action == "CROSS":
                invoker.add_orb("W")
                vfx.trigger_spell_burst(cx, cy, "WEX (W)", (255, 70, 220))
            elif triggered_action == "RAM":
                spell_name, col = invoker.invoke()
                vfx.trigger_spell_burst(cx, cy, f"INVOKE: {spell_name}", col)
            elif triggered_action == "OPEN_PALM":
                spell_name, col = invoker.cast_spell_d()
                vfx.trigger_spell_burst(cx, cy, f"CAST D: {spell_name}!", col)
            elif triggered_action == "DUAL_PALM":
                spell_name, col = invoker.cast_spell_f()
                vfx.trigger_spell_burst(cx, cy, f"CAST F: {spell_name}!", col)

        # Render Visual Effects & Procedural Magic Circle Overlay
        vfx.update_and_render(frame, info, progress)

        # Gambar Kontur dan Convex Hull Tangan
        if info["contour"] is not None:
            cv2.drawContours(frame, [info["contour"]], -1, (0, 240, 160), 2, cv2.LINE_AA)
            cv2.drawContours(frame, [info["hull"]], -1, (255, 255, 255), 1, cv2.LINE_AA)

        # Render Antarmuka HUD Dota 2 di bagian bawah
        current_seal = info.get("seal", "NONE")
        invoker.render_dota_hud(frame, current_seal, progress)

        # Header Info & Shortcut Bar di bagian atas
        top_bar_h = 42
        cv2.rectangle(frame, (0, 0), (w, top_bar_h), (18, 20, 28), -1)
        cv2.line(frame, (0, top_bar_h), (w, top_bar_h), (50, 60, 80), 1)

        input_status = "LIVE DIRECTX" if invoker.input_sender.enabled else "SIMULASI"
        status_color = (80, 255, 120) if invoker.input_sender.enabled else (120, 180, 255)

        cv2.putText(frame, f"PCV PROJECT 1: DOTA 2 INVOKER | FPS: {fps:.1f}", (15, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (230, 230, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, f"INPUT: {input_status} [I]", (w - 230, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, status_color, 1, cv2.LINE_AA)

        # Tampilan Picture-in-Picture (PiP) Masker Biner Hasil Morfologi
        if show_mask_pip:
            pip_w, pip_h = 160, 110
            pip_x = w - pip_w - 15
            pip_y = top_bar_h + 15
            mask_rgb = cv2.cvtColor(cv2.resize(mask, (pip_w, pip_h)), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(frame, (pip_x - 2, pip_y - 2), (pip_x + pip_w + 2, pip_y + pip_h + 2), (0, 215, 255), 1)
            frame[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = mask_rgb
            cv2.putText(frame, "Skin Mask (PiP)", (pip_x + 5, pip_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        # Tampilkan Frame ke Window
        cv2.imshow(window_name, frame)

        # Event Key Handling
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break
        elif key in [ord('c'), ord('C')]:
            calibration_mode = True
        elif key in [ord('i'), ord('I')]:
            invoker.input_sender.enabled = not invoker.input_sender.enabled
            print(f"[PCV Input Mode] DirectInput: {'LIVE' if invoker.input_sender.enabled else 'SIMULATED'}")
        elif key in [ord('m'), ord('M')]:
            show_mask_pip = not show_mask_pip
        elif key in [ord('r'), ord('R')]:
            invoker.active_orbs = ["E", "E", "E"]
            invoker.last_action_msg = "Orb di-reset ke [E, E, E]"

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Aplikasi Dota 2 Invoker selesai ditutup.")


if __name__ == "__main__":
    main()
