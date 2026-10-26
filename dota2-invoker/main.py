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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat diakses! Pastikan webcam terhubung.")
        return

    # Inisialisasi Modul
    detector = HandSealDetector()
    vfx = VFXOverlay(minimalist_mode=True)
    invoker = InvokerController(direct_input_enabled=True)

    # Status Kontrol
    show_mask_pip = True
    calibration_mode = False
    toast_msg = ""
    toast_timer = 0.0
    fps_time = time.time()
    fps = 0.0

    print("===================================================================")
    print("        DOTA 2 INVOKER HAND POSE CONTROLLER (PURE OPENCV)")
    print("===================================================================")
    print(" - 1 JARI         -> QUAS (Q)     [Elemen Es]")
    print(" - 2 JARI         -> WEX (W)      [Elemen Petir]")
    print(" - 3 JARI         -> EXORT (E)    [Elemen Api]")
    print(" - 4 JARI         -> INVOKE (R)   [Rapal Mantra]")
    print(" - 5 JARI         -> QUICK CAST D [Mantra Slot 1]")
    print(" - KEPAL (FIST)   -> QUICK CAST F [Mantra Slot 2]")
    print("-------------------------------------------------------------------")
    print(" Shortcut: [C] Kalibrasi | [I] Toggle DirectInput | [M] Mask PiP | [Q] Keluar")
    print("===================================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Cermin horizontal agar gerakan tangan intuitif
        frame = cv2.flip(frame, 1)
        # Standarisasi resolusi ke 1280x720 untuk rendering HUD & UI proporsional
        frame = cv2.resize(frame, (1280, 720))
        h, w = frame.shape[:2]

        # Hitung FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, now - fps_time))
        fps_time = now

        # Mode Kalibrasi Warna Kulit (Besar & Jelas dengan Visual Swatch)
        if calibration_mode:
            box_w, box_h = 300, 300
            bx1 = (w - box_w) // 2
            by1 = (h - box_h) // 2
            bx2 = bx1 + box_w
            by2 = by1 + box_h

            roi = frame[by1:by2, bx1:bx2]
            mean_b = int(np.mean(roi[:, :, 0]))
            mean_g = int(np.mean(roi[:, :, 1]))
            mean_r = int(np.mean(roi[:, :, 2]))
            ycrcb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
            mean_cr = int(np.mean(ycrcb_roi[:, :, 1]))
            mean_cb = int(np.mean(ycrcb_roi[:, :, 2]))

            # Kotak target kalibrasi utama (Hijau Cerah & Siku Sudut)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 200), 3)
            # Siku sudut kotak
            c_len = 35
            cv2.line(frame, (bx1, by1), (bx1 + c_len, by1), (255, 255, 255), 4)
            cv2.line(frame, (bx1, by1), (bx1, by1 + c_len), (255, 255, 255), 4)
            cv2.line(frame, (bx2, by1), (bx2 - c_len, by1), (255, 255, 255), 4)
            cv2.line(frame, (bx2, by1), (bx2, by1 + c_len), (255, 255, 255), 4)
            cv2.line(frame, (bx1, by2), (bx1 + c_len, by2), (255, 255, 255), 4)
            cv2.line(frame, (bx1, by2), (bx1, by2 - c_len), (255, 255, 255), 4)
            cv2.line(frame, (bx2, by2), (bx2 - c_len, by2), (255, 255, 255), 4)
            cv2.line(frame, (bx2, by2), (bx2, by2 - c_len), (255, 255, 255), 4)

            # Panel Informasi Live Swatch di Samping Kotak
            p_x = bx2 + 25
            cv2.rectangle(frame, (p_x, by1), (p_x + 260, by1 + 220), (20, 24, 32), -1)
            cv2.rectangle(frame, (p_x, by1), (p_x + 260, by1 + 220), (0, 255, 200), 1)

            cv2.putText(frame, "SAMPEL KULIT (LIVE):", (p_x + 15, by1 + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.50, (200, 230, 255), 1, cv2.LINE_AA)

            # Kotak warna rata-rata yang sedang dibaca
            cv2.rectangle(frame, (p_x + 15, by1 + 45), (p_x + 90, by1 + 105), (mean_b, mean_g, mean_r), -1)
            cv2.rectangle(frame, (p_x + 15, by1 + 45), (p_x + 90, by1 + 105), (255, 255, 255), 2)

            cv2.putText(frame, f"Cr: {mean_cr}", (p_x + 105, by1 + 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 230), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Cb: {mean_cb}", (p_x + 105, by1 + 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 230), 1, cv2.LINE_AA)

            cv2.putText(frame, f"RGB: ({mean_r}, {mean_g}, {mean_b})", (p_x + 15, by1 + 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 200, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, "Penuhi kotak dgn telapak,", (p_x + 15, by1 + 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, "lalu tekan tombol [C].", (p_x + 15, by1 + 188),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 200), 1, cv2.LINE_AA)

            # Petunjuk di Atas & Bawah Kotak
            cv2.putText(frame, "[MODE KALIBRASI WARNA KULIT]", (bx1 - 40, by1 - 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, "Posisikan seluruh telapak tangan di dalam kotak target hijau", (bx1 - 90, by2 + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "[C] Kunci Kalibrasi  |  [D] Reset Default  |  [ESC] Batal", (bx1 - 80, by2 + 65),
                        cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 230, 255), 1, cv2.LINE_AA)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('c'), ord('C')]:
                calib = detector.calibrate_skin_sample(roi)
                toast_msg = f"KALIBRASI DISIMPAN! Cr{calib['cr_range']}, Cb{calib['cb_range']}"
                toast_timer = 2.5
                calibration_mode = False
            elif key in [ord('d'), ord('D')]:
                detector.reset_default_calibration()
                toast_msg = "KALIBRASI DI-RESET KE DEFAULT UNIVERSAL"
                toast_timer = 2.5
                calibration_mode = False
            elif key in [ord('q'), ord('Q'), 27]:
                calibration_mode = False

            cv2.imshow(window_name, frame)
            continue

        # Proses Deteksi Tangan & Segel (Murni OpenCV)
        triggered_action, progress, info, mask = detector.process_frame(frame)

        # Tangani Aksi yang Terpicu
        if triggered_action:
            cx, cy = info["features"].get("centroid", (w // 2, h // 2))

            if triggered_action in ["1_FINGER", "SNAKE"]:
                invoker.add_orb("Q")
                vfx.trigger_spell_burst(cx, cy, "QUAS (Q)", (255, 210, 80))
            elif triggered_action in ["2_FINGERS", "CROSS"]:
                invoker.add_orb("W")
                vfx.trigger_spell_burst(cx, cy, "WEX (W)", (255, 70, 220))
            elif triggered_action in ["3_FINGERS", "TIGER"]:
                invoker.add_orb("E")
                vfx.trigger_spell_burst(cx, cy, "EXORT (E)", (40, 100, 255))
            elif triggered_action in ["4_FINGERS", "RAM"]:
                spell_name, col = invoker.invoke()
                vfx.trigger_spell_burst(cx, cy, f"INVOKE (R): {spell_name}", col)
            elif triggered_action in ["5_FINGERS", "OPEN_PALM"]:
                spell_name, col = invoker.cast_spell_d()
                vfx.trigger_spell_burst(cx, cy, f"CAST D: {spell_name}!", col)
            elif triggered_action in ["FIST", "DUAL_PALM"]:
                spell_name, col = invoker.cast_spell_f()
                vfx.trigger_spell_burst(cx, cy, f"CAST F: {spell_name}!", col)

        # Render Visual Effects & Procedural Magic Circle Overlay
        vfx.update_and_render(frame, info, progress)

        # Gambar Kontur, Convex Hull, dan Titik Lekukan Jari (Defects)
        if info["contour"] is not None:
            cv2.drawContours(frame, [info["contour"]], -1, (0, 240, 160), 2, cv2.LINE_AA)
            cv2.drawContours(frame, [info["hull"]], -1, (255, 255, 255), 1, cv2.LINE_AA)

            # Gambar titik-titik lembah antar jari untuk visualisasi
            deep_defects = info["features"].get("deep_defects", [])
            for pt in deep_defects:
                cv2.circle(frame, tuple(pt), 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(pt), 8, (255, 255, 255), 1, cv2.LINE_AA)

        # Render Antarmuka HUD Dota 2 di bagian bawah
        current_seal = info.get("seal", "NONE")
        invoker.render_dota_hud(frame, current_seal, progress)

        # Header Info & Shortcut Bar di bagian atas (Bersih & Rapi)
        top_bar_h = 42
        cv2.rectangle(frame, (0, 0), (w, top_bar_h), (18, 20, 28), -1)
        cv2.line(frame, (0, top_bar_h), (w, top_bar_h), (50, 60, 80), 1)

        input_status = "LIVE DIRECTX" if invoker.input_sender.enabled else "SIMULASI"
        status_color = (80, 255, 120) if invoker.input_sender.enabled else (120, 180, 255)

        cv2.putText(frame, f"PCV INVOKER | FPS: {fps:.1f}", (15, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (230, 230, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, "[C] Kalibrasi  [D] Reset Default  [I] Input Mode  [M] Mask  [R] Reset Orbs", (240, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 190, 210), 1, cv2.LINE_AA)
        cv2.putText(frame, f"INPUT: {input_status} [I]", (w - 230, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, status_color, 1, cv2.LINE_AA)

        # Banner Notifikasi Konfirmasi Kalibrasi / Status
        if toast_timer > 0:
            toast_timer -= 0.03
            t_box_w = 640
            t_bx = (w - t_box_w) // 2
            cv2.rectangle(frame, (t_bx, top_bar_h + 12), (t_bx + t_box_w, top_bar_h + 46), (20, 30, 24), -1)
            cv2.rectangle(frame, (t_bx, top_bar_h + 12), (t_bx + t_box_w, top_bar_h + 46), (0, 255, 180), 2)
            cv2.putText(frame, f"[OK] {toast_msg}", (t_bx + 15, top_bar_h + 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 180), 1, cv2.LINE_AA)

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
        elif key in [ord('d'), ord('D')]:
            detector.reset_default_calibration()
            toast_msg = "KALIBRASI DI-RESET KE DEFAULT UNIVERSAL"
            toast_timer = 2.5
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
