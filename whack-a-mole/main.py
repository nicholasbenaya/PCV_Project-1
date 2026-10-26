"""Entry point aplikasi game Whack-A-Mole Virtual berbasis OpenCV murni."""

import argparse
import os
import sys
import time
import cv2

# Memastikan direktori modul masuk ke sys.path untuk eksekusi langsung
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from .config import (
        DEFAULT_FRAME_HEIGHT,
        DEFAULT_FRAME_WIDTH,
        WINDOW_TITLE,
    )
    from .game_engine import GameEngine, GameState
    from .hand_tracker import HandTracker
except ImportError:
    from config import (
        DEFAULT_FRAME_HEIGHT,
        DEFAULT_FRAME_WIDTH,
        WINDOW_TITLE,
    )
    from game_engine import GameEngine, GameState
    from hand_tracker import HandTracker


def parse_arguments() -> argparse.Namespace:
    """Menguraikan argumen baris perintah saat menjalankan program."""
    parser = argparse.ArgumentParser(
        description="Game Whack-A-Mole Virtual murni OpenCV (Lock-On, BG Subtraction, & PiP Diagnostik)."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Indeks perangkat kamera webcam (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help=f"Lebar frame kamera (default: {DEFAULT_FRAME_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_FRAME_HEIGHT,
        help=f"Tinggi frame kamera (default: {DEFAULT_FRAME_HEIGHT}).",
    )
    return parser.parse_args()


def run_game() -> None:
    """Menginisialisasi perangkat input dan menjalankan loop utama permainan."""
    args = parse_arguments()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(
            f"Error: Kamera dengan indeks {args.camera} tidak dapat diakses.",
            file=sys.stderr,
        )
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    hand_tracker = HandTracker()
    game_engine = GameEngine(
        frame_width=args.width,
        frame_height=args.height,
    )

    # Muat konfigurasi kalibrasi warna kulit jika ada di penyimpanan lokal
    hand_tracker.load_saved_calibration(game_engine.save_manager.calib_data)

    # Tampilkan jendela PiP diagnostik fitur bentuk PCV secara default
    show_pip_diagnostic = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Peringatan: Gagal membaca frame dari webcam.", file=sys.stderr)
                break

            # Flip horizontal agar cermin natural bagi koordinasi tangan pemain
            frame = cv2.flip(frame, 1)

            # Resize ke resolusi target jika rasio webcam berbeda
            h, w = frame.shape[:2]
            if (w, h) != (args.width, args.height):
                frame = cv2.resize(frame, (args.width, args.height))
                h, w = args.height, args.width

            game_engine.update_frame_dimensions(w, h)
            current_time = time.time()

            # 4 Pilar PCD Klasik: BG Subtraction, RETR_CCOMP Face Rejection, Open-Palm Lock-On, dan Nearest-Neighbor
            hand_result = hand_tracker.process_frame(frame)

            # Update alur game, interaksi tombol (FIST click), dan logika tabrakan (FIST whack)
            action_status = game_engine.update(hand_result, current_time)
            if action_status == "exit":
                break

            # Render seluruh elemen visual antarmuka pada layar penuh
            game_engine.render(
                frame,
                hand_result,
                hand_tracker,
                current_time,
                show_pip_diagnostic=show_pip_diagnostic,
            )

            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF

            # Penanganan tombol keluar / kembali secara global
            if key == 27:  # ESC key
                if game_engine.state == GameState.CALIBRATION:
                    game_engine.state = GameState.MENU
                elif game_engine.state in (GameState.LEVEL_SELECT, GameState.GRID_SELECT):
                    game_engine.state = GameState.MENU
                    game_engine.audio.stop_bgm()
                elif game_engine.state in (GameState.PLAYING_SURVIVAL, GameState.PLAYING_LEVEL):
                    game_engine.state = GameState.MENU
                    game_engine.audio.stop_bgm()
                elif game_engine.state in (GameState.LEVEL_CLEARED, GameState.GAMEOVER):
                    game_engine.state = GameState.MENU
                    game_engine.audio.stop_bgm()
                elif game_engine.state == GameState.MENU:
                    break

            # Rekam Frame Referensi Latar Belakang (Background Subtraction) dengan tombol 'B'
            elif key in (ord("b"), ord("B")):
                hand_tracker.capture_background(frame)
                game_engine.audio.play_sfx("bonk")

            # Reset / Hapus Frame Referensi Latar Belakang (Bypass BG Subtraction) dengan tombol 'N'
            elif key in (ord("n"), ord("N")):
                hand_tracker.clear_background()
                game_engine.audio.play_sfx("bonk")

            # Toggle Jendela PiP Diagnostik Fitur Bentuk PCV dengan tombol 'V' atau 'P'
            elif key in (ord("v"), ord("V"), ord("p"), ord("P")):
                show_pip_diagnostic = not show_pip_diagnostic

            # Ganti Preset Mode Segmentasi [1] Skin, [2] Black Glove, [3] High-Contrast Object dengan tombol 'M'
            elif key in (ord("m"), ord("M")):
                new_preset_name = hand_tracker.cycle_preset_mode()
                game_engine.audio.play_sfx("click")

            # Toggle Master Audio dengan tombol 'S'
            elif key in (ord("s"), ord("S")):
                game_engine.audio.toggle_sound()

            # Masuk atau keluar mode kalibrasi dengan tombol 'C'
            elif key in (ord("c"), ord("C")):
                if game_engine.state == GameState.CALIBRATION:
                    bx1, by1, bx2, by2 = game_engine.get_calibration_box()
                    roi = frame[by1:by2, bx1:bx2]
                    calib_dict = hand_tracker.calibrate_skin_sample(roi)
                    game_engine.save_manager.set_calibration_data(calib_dict)
                    game_engine.audio.play_sfx("bonk")
                    game_engine.state = GameState.MENU
                else:
                    game_engine.state = GameState.CALIBRATION
                    game_engine.audio.play_sfx("click")

            # Kontrol saat di layar Kalibrasi
            elif game_engine.state == GameState.CALIBRATION:
                if key in (13, 32):  # ENTER atau SPASI untuk mengunci kalibrasi
                    bx1, by1, bx2, by2 = game_engine.get_calibration_box()
                    roi = frame[by1:by2, bx1:bx2]
                    calib_dict = hand_tracker.calibrate_skin_sample(roi)
                    game_engine.save_manager.set_calibration_data(calib_dict)
                    game_engine.audio.play_sfx("bonk")
                    game_engine.state = GameState.MENU

            # Kontrol saat di Menu Utama
            elif game_engine.state == GameState.MENU:
                if key in (ord("q"), ord("Q")):
                    break
                elif key == ord("1"):
                    game_engine.start_survival_mode(current_time)
                elif key == ord("2"):
                    game_engine.state = GameState.LEVEL_SELECT
                    game_engine.audio.play_sfx("click")
                elif key == ord("3"):
                    game_engine.state = GameState.GRID_SELECT
                    game_engine.audio.play_sfx("click")
                elif key == ord("4"):
                    game_engine.state = GameState.CALIBRATION
                    game_engine.audio.play_sfx("click")
                elif key == ord("5"):
                    game_engine.audio.toggle_sound()
                elif key == ord("6"):
                    break

            # Kontrol saat di Pemilihan Level
            elif game_engine.state == GameState.LEVEL_SELECT:
                if ord("1") <= key <= ord("9"):
                    lvl = key - ord("0")
                    if lvl <= game_engine.save_manager.unlocked_level:
                        game_engine.start_level_mode(lvl, current_time)
                elif key == ord("0"):
                    if 10 <= game_engine.save_manager.unlocked_level:
                        game_engine.start_level_mode(10, current_time)

            # Kontrol saat di Pemilihan Grid
            elif game_engine.state == GameState.GRID_SELECT:
                if ord("1") <= key <= ord("4"):
                    grid_idx = key - ord("1")
                    game_engine.handle_button_action(f"grid_{grid_idx}", current_time)

            # Kontrol saat Level Berhasil Dituntaskan
            elif game_engine.state == GameState.LEVEL_CLEARED:
                if key == ord(" "):
                    if game_engine.current_level < 10:
                        game_engine.start_level_mode(game_engine.current_level + 1, current_time)
                    else:
                        game_engine.state = GameState.MENU
                elif key in (ord("r"), ord("R")):
                    game_engine.start_level_mode(game_engine.current_level, current_time)

            # Kontrol saat Game Over
            elif game_engine.state == GameState.GAMEOVER:
                if key in (ord("r"), ord("R")):
                    if game_engine.survival_tracker is not None:
                        game_engine.start_survival_mode(current_time)
                    else:
                        game_engine.start_level_mode(game_engine.current_level, current_time)

    finally:
        game_engine.audio.stop_bgm()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_game()
