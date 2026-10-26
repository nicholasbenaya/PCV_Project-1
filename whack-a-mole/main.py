"""Entry point aplikasi game Whack-A-Mole Virtual berbasis OpenCV murni (HD 1280x720)."""

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
        DEFAULT_GAME_DURATION,
        WINDOW_TITLE,
    )
    from .game_engine import GameEngine, GameState
    from .hand_tracker import HandTracker
except ImportError:
    from config import (
        DEFAULT_FRAME_HEIGHT,
        DEFAULT_FRAME_WIDTH,
        DEFAULT_GAME_DURATION,
        WINDOW_TITLE,
    )
    from game_engine import GameEngine, GameState
    from hand_tracker import HandTracker


def parse_arguments() -> argparse.Namespace:
    """Menguraikan argumen baris perintah saat menjalankan program."""
    parser = argparse.ArgumentParser(
        description="Game Whack-A-Mole Virtual berbasis OpenCV murni dengan sistem Dual-Section."
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
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_GAME_DURATION,
        help=f"Durasi per sesi permainan dalam detik (default: {DEFAULT_GAME_DURATION}).",
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
        game_duration=args.duration,
    )

    show_mask_pip = False

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                print("Peringatan: Gagal membaca frame dari webcam.", file=sys.stderr)
                break

            # Flip horizontal agar cermin natural bagi koordinasi tangan pemain
            flipped = cv2.flip(raw_frame, 1)

            # Resize otomatis ke resolusi HD kanvas 1280x720
            frame = cv2.resize(flipped, (args.width, args.height))

            current_time = time.time()

            # Deteksi tangan HANYA di dalam Playing Section (Wajah di Non-playing section dinolkan 100%)
            hand_result = hand_tracker.process_frame(
                frame,
                play_rect=game_engine.active_zone["play_rect"],
            )

            # Update logika game dan deteksi tabrakan
            game_engine.update(hand_result, current_time)

            # Render seluruh elemen visual antarmuka
            game_engine.render(
                frame,
                hand_result,
                current_time,
                show_mask_pip=show_mask_pip,
            )

            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key in (ord("h"), ord("H"), 9):  # 'H' atau TAB untuk menukar sisi Playing Section
                game_engine.switch_zone()
                hand_tracker.reset()
            elif key == ord(" "):
                if game_engine.state == GameState.START:
                    game_engine.start_game(current_time)
            elif key in (ord("r"), ord("R")):
                if game_engine.state == GameState.GAMEOVER:
                    game_engine.restart_game(current_time)
            elif key in (ord("m"), ord("M")):
                show_mask_pip = not show_mask_pip

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_game()
