"""
Main Runner: Dota 2 Invoker Hand Pose Controller (Pure OpenCV)
Praktikum Pengolahan Citra dan Visi Komputer (PCV) - Project 1
Window: "Tugas Game - Dota 2 Invoker"

Navigasi Kontrol Keyboard:
- 'C': Mode Kalibrasi Warna Kulit (Letakkan tangan di kotak tengah, lalu tekan 'C')
- 'Z': Toggle Hand Zone ROI (Area aman tangan, memblokir gorden dan rambut)
- 'I': Toggle DirectInput Hardware Key Injection (LIVE / SIMULATION)
- 'M': Toggle Tampilan Masker Biner Hasil Morfologi (PiP)
- 'O': Reset Orb Invoker ke Default (EEE) [Huruf O, bukan R]
- 'K': Reset Kalibrasi ke Default Universal [Huruf K, bukan D]
- 'ESC' atau 'X': Keluar dari Aplikasi [Bukan Q, agar tidak tabrakan dengan Quas]
"""

import cv2
import numpy as np
import time
import os
import sys
import logging
import traceback
from hand_detector import HandSealDetector
from vfx_overlay import VFXOverlay
from invoker_controller import InvokerController

# Setup Crash Logging
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash.log")
logging.basicConfig(
    filename=log_file,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Dota2Invoker")


def main():
    logger.info("=== APLIKASI DOTA 2 INVOKER DIMULAI ===")
    window_name = "Tugas Game - Dota 2 Invoker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Inisialisasi Webcam: Prioritaskan DirectShow (CAP_DSHOW) di Windows untuk kestabilan tinggi & 0 lag
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.warning("DirectShow tidak tersedia, mencoba default backend...")
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        err_msg = "[ERROR] Kamera tidak dapat diakses! Pastikan webcam terhubung dan tidak dipakai aplikasi lain."
        print(err_msg)
        logger.error(err_msg)
        return

    logger.info("Webcam berhasil diinisialisasi pada resolusi 1280x720.")

    # Inisialisasi Modul
    detector = HandSealDetector()
    vfx = VFXOverlay(minimalist_mode=True)
    invoker = InvokerController(direct_input_enabled=True)

    # Status Kontrol & Preset Hand Zone (Mengabaikan Wajah 100%)
    show_mask_pip = True
    calibration_mode = False
    use_hand_zone = True

    zone_presets = [
        ("TANGAN KANAN (WAJAH DIKECUALIKAN)", (390, 50, 1240, 670)),
        ("TANGAN KIRI (WAJAH DIKECUALIKAN)", (40, 50, 800, 670)),
        ("TENGAH / BEBAS", (260, 50, 1020, 670))
    ]
    zone_idx = 0  # Default Tangan Kanan: Sisi kiri frame (tempat kepala/badan) diabaikan 100%
    hand_zone = zone_presets[zone_idx][1]

    # Status Live Diagnostik & Telemetri
    show_debug = True
    telemetry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_telemetry.log")
    with open(telemetry_path, "w", encoding="utf-8") as tf:
        tf.write(f"=== SESI TESTING LIVE TELEMETRI DIMULAI PADA {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    last_telemetry_time = 0.0
    last_logged_seal = "NONE"

    toast_msg = "HAND ZONE: TANGAN KANAN AKTIF [H: Ganti Sisi | TAB: Debug]"
    toast_timer = 3.5
    fps_time = time.time()
    fps = 0.0
    consecutive_drop_count = 0

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
    print(" Shortcut: [C] Kalibrasi | [H] Ganti Sisi | [Z] Hand Zone | [ESC] Keluar")
    print("===================================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_drop_count += 1
            if consecutive_drop_count > 45:
                err_msg = "[ERROR] Kamera terputus (45 frame berturut-turut gagal dibaca)."
                print(err_msg)
                logger.error(err_msg)
                break
            time.sleep(0.01)
            continue
        consecutive_drop_count = 0

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

        # Gambar Area Interaksi Tangan (Hand Zone) jika aktif
        if use_hand_zone:
            zx1, zy1, zx2, zy2 = hand_zone
            mode_name, _ = zone_presets[zone_idx]

            # Redupkan area luar (tempat kepala/badan berada) agar terlihat aman dan terabaikan
            if zone_idx == 0:  # Tangan Kanan -> Redupkan area kiri layar tempat wajah berada
                frame[:, :zx1] = (frame[:, :zx1] * 0.40).astype(np.uint8)
                cv2.putText(frame, "[SAFE] WAJAH & TUBUH DIABAIKAN", (25, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 150, 170), 1, cv2.LINE_AA)
            elif zone_idx == 1:  # Tangan Kiri -> Redupkan area kanan layar
                frame[:, zx2:] = (frame[:, zx2:] * 0.40).astype(np.uint8)
                cv2.putText(frame, "[SAFE] WAJAH & TUBUH DIABAIKAN", (zx2 + 15, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 150, 170), 1, cv2.LINE_AA)

            # Gambar border Hand Zone
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 230, 255), 2)
            cv2.putText(frame, f"HAND ZONE [{mode_name}] [H: Ganti Sisi | Z: Toggle]",
                        (zx1 + 10, zy1 + 25), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)

        # Proses Deteksi Tangan (Isolasi Hand Zone aktif)
        active_roi = hand_zone if use_hand_zone else None
        triggered_action, progress, info, mask = detector.process_frame(frame, roi_box=active_roi)

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

        # Render Visual Effects Minimalis
        vfx.update_and_render(frame, info, progress)

        # Gambar Kontur, Convex Hull, Lingkaran Telapak, dan Ujung Jari
        if info["contour"] is not None:
            cv2.drawContours(frame, [info["contour"]], -1, (0, 240, 160), 2, cv2.LINE_AA)
            cv2.drawContours(frame, [info["hull"]], -1, (255, 255, 255), 1, cv2.LINE_AA)

            feats = info.get("features", {})
            palm_center = feats.get("palm_center", None)
            palm_radius = feats.get("palm_radius", 0.0)

            # Gambar lingkaran telapak tangan (Palm Circle)
            if palm_center is not None and palm_radius > 15:
                cv2.circle(frame, palm_center, int(palm_radius), (0, 255, 200), 2, cv2.LINE_AA)
                cv2.circle(frame, palm_center, 4, (0, 255, 200), -1)

            # Gambar titik ujung jari nyata (Fingertips) dengan titik merah cerah
            finger_tips = feats.get("finger_tips", [])
            for tip in finger_tips:
                cv2.circle(frame, tuple(tip), 8, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(tip), 10, (255, 255, 255), 2, cv2.LINE_AA)

            # Indikator teks jumlah jari di atas telapak tangan
            tip_count = feats.get("tip_count", 0)
            if palm_center is not None:
                cv2.putText(frame, f"{tip_count} JARI", (palm_center[0] - 35, palm_center[1] - int(palm_radius) - 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)

        # Render Antarmuka HUD Dota 2 di bagian bawah
        current_seal = info.get("seal", "NONE")
        invoker.render_dota_hud(frame, current_seal, progress)

        # Panel Live Diagnostik Real-Time (TAB untuk Toggle On/Off)
        if show_debug:
            dbg_w, dbg_h = 330, 115
            dbg_x, dbg_y = 15, 60
            dbg_bg = frame[dbg_y:dbg_y+dbg_h, dbg_x:dbg_x+dbg_w].copy()
            cv2.rectangle(dbg_bg, (0, 0), (dbg_w, dbg_h), (12, 16, 24), -1)
            cv2.addWeighted(dbg_bg, 0.82, frame[dbg_y:dbg_y+dbg_h, dbg_x:dbg_x+dbg_w], 0.18, 0, frame[dbg_y:dbg_y+dbg_h, dbg_x:dbg_x+dbg_w])
            cv2.rectangle(frame, (dbg_x, dbg_y), (dbg_x+dbg_w, dbg_y+dbg_h), (0, 220, 255), 1)

            feats = info.get("features", {})
            t_cnt = feats.get("tip_count", 0)
            d_cnt = feats.get("defect_count", 0)
            c_area = int(feats.get("area", 0))

            cv2.putText(frame, "LIVE DIAGNOSTIK [TAB: Toggle]", (dbg_x + 10, dbg_y + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Pose Terbaca : {current_seal}", (dbg_x + 10, dbg_y + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Ujung Jari   : {t_cnt} JARI (Celah: {d_cnt})", (dbg_x + 10, dbg_y + 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 240, 180), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Hold Meter   : {int(progress * 100)}% ({detector.candidate_frames}/{detector.required_hold_frames})", (dbg_x + 10, dbg_y + 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Luas Kontur  : {c_area} px", (dbg_x + 10, dbg_y + 106),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 170, 190), 1, cv2.LINE_AA)

        # Rekam Log Telemetri Real-Time (untuk pemantauan langsung)
        now_sec = time.time()
        if (current_seal != last_logged_seal) or (current_seal != "NONE" and now_sec - last_telemetry_time >= 0.40):
            last_logged_seal = current_seal
            last_telemetry_time = now_sec
            feats = info.get("features", {})
            log_line = (f"{time.strftime('%H:%M:%S')} | Pose: {current_seal:10s} | "
                        f"Tips: {feats.get('tip_count', 0)} | Defects: {feats.get('defect_count', 0)} | "
                        f"Progress: {int(progress * 100):3d}% | Area: {int(feats.get('area', 0))} px\n")
            with open(telemetry_path, "a", encoding="utf-8") as tf:
                tf.write(log_line)

        # Header Info & Shortcut Bar di bagian atas (Bersih & Rapi)
        top_bar_h = 42
        cv2.rectangle(frame, (0, 0), (w, top_bar_h), (18, 20, 28), -1)
        cv2.line(frame, (0, top_bar_h), (w, top_bar_h), (50, 60, 80), 1)

        input_status = "LIVE DIRECTX" if invoker.input_sender.enabled else "SIMULASI"
        status_color = (80, 255, 120) if invoker.input_sender.enabled else (120, 180, 255)

        cv2.putText(frame, f"PCV INVOKER | FPS: {fps:.1f}", (15, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (230, 230, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, "[C] Kalibrasi  [H] Sisi Tangan  [TAB] Live Debug  [Z] Hand Zone  [I] Input  [ESC] Keluar", (185, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 190, 210), 1, cv2.LINE_AA)
        cv2.putText(frame, f"INPUT: {input_status} [I]", (w - 225, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, status_color, 1, cv2.LINE_AA)

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
        # PERINGATAN: JANGAN gunakan Q, W, E, R, D, F untuk kontrol antarmuka di sini!
        # Karena DirectInput menginjeksi tombol tersebut ke sistem Windows. Jika game window sedang fokus,
        # cv2.waitKey akan membaca tombol mantra tersebut dan dapat memicu penutupan/reset yang tidak disengaja.
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('x'), ord('X')]:  # Keluar HANYA dengan tombol ESC atau X
            logger.info("Pengguna menekan ESC/X untuk keluar.")
            break
        elif key in [9, ord('\t')]:  # TAB untuk Toggle Panel Live Diagnostik
            show_debug = not show_debug
            toast_msg = f"LIVE DIAGNOSTIK: {'AKTIF' if show_debug else 'NONAKTIF'}"
            toast_timer = 2.0
            logger.info(f"Live debug HUD: {show_debug}")
        elif key in [ord('c'), ord('C')]:
            calibration_mode = True
        elif key in [ord('h'), ord('H')]:  # H untuk Ganti Sisi Tangan (Kanan / Kiri / Bebas)
            zone_idx = (zone_idx + 1) % len(zone_presets)
            hand_zone = zone_presets[zone_idx][1]
            mode_name = zone_presets[zone_idx][0]
            toast_msg = f"ZONA: {mode_name}"
            toast_timer = 2.5
            logger.info(f"Hand zone diganti ke: {mode_name}")
        elif key in [ord('z'), ord('Z')]:
            use_hand_zone = not use_hand_zone
            status_str = "AKTIF (Gorden & Rambut Terblokir)" if use_hand_zone else "NONAKTIF (Full Frame)"
            toast_msg = f"HAND ZONE: {status_str}"
            toast_timer = 2.5
            logger.info(f"Hand zone diubah: {status_str}")
        elif key in [ord('k'), ord('K')]:  # K untuk Kalibrasi default (bukan D)
            detector.reset_default_calibration()
            toast_msg = "KALIBRASI DI-RESET KE DEFAULT UNIVERSAL"
            toast_timer = 2.5
            logger.info("Kalibrasi di-reset ke default universal.")
        elif key in [ord('i'), ord('I')]:
            invoker.input_sender.enabled = not invoker.input_sender.enabled
            status_log = 'LIVE DIRECTX' if invoker.input_sender.enabled else 'SIMULATED'
            print(f"[PCV Input Mode] DirectInput: {status_log}")
            logger.info(f"Input mode diubah ke: {status_log}")
        elif key in [ord('m'), ord('M')]:
            show_mask_pip = not show_mask_pip
        elif key in [ord('o'), ord('O')]:  # O untuk Reset Orbs (bukan R)
            invoker.active_orbs = ["E", "E", "E"]
            invoker.last_action_msg = "Orb di-reset ke [E, E, E]"
            toast_msg = "ORBS DI-RESET KE [E, E, E]"
            toast_timer = 2.0
            logger.info("Orb di-reset ke default [E, E, E].")

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Aplikasi ditutup dengan bersih.")
    print("[INFO] Aplikasi Dota 2 Invoker selesai ditutup.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("FATAL UNHANDLED CRASH:")
        print(f"\n[FATAL CRASH] Terjadi error tak terduga pada program: {e}")
        print(f"[LOG FILE] Riwayat error lengkap telah dicatat di: {log_file}")
        traceback.print_exc()
        input("\nTekan Enter untuk keluar...")
