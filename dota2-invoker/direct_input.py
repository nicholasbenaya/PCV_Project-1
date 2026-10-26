"""
DirectX Hardware Input Sender for Dota 2 (Windows API via ctypes)
Mengirim hardware scan codes standar DirectInput agar 100% terbaca oleh Source 2 Engine (Dota 2).
"""

import ctypes
import time

# Windows API Constants
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

# DirectX Hardware Scancodes
SCANCODES = {
    'Q': 0x10,
    'W': 0x11,
    'E': 0x12,
    'R': 0x13,
    'D': 0x20,
    'F': 0x21,
    'B': 0x30,
    'SPACE': 0x39,
    '1': 0x02,
    '2': 0x03,
    '3': 0x04,
    '4': 0x05,
}

# C Structs for SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort)
    ]

class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class Input_I(ctypes.Union):
    _fields_ = [
        ("ki", KeyBdInput),
        ("mi", MouseInput),
        ("hi", HardwareInput)
    ]

class Input(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ii", Input_I)
    ]


class DirectInputSender:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def press_key(self, key_name: str, hold_time: float = 0.02):
        """Kirim penekanan tombol hardware scancode ke Windows."""
        key_name = key_name.upper()
        if not self.enabled:
            print(f"[DirectInput Simulation] Key pressed: {key_name}")
            return

        if key_name not in SCANCODES:
            print(f"[DirectInput Warning] Scancode untuk '{key_name}' belum terdaftar.")
            return

        scancode = SCANCODES[key_name]

        try:
            # Key Down
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.ki = KeyBdInput(0, scancode, KEYEVENTF_SCANCODE, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

            time.sleep(hold_time)

            # Key Up
            ii_.ki = KeyBdInput(0, scancode, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        except Exception as e:
            print(f"[DirectInput Error] Gagal mengeksekusi SendInput untuk key '{key_name}': {e}")


if __name__ == "__main__":
    sender = DirectInputSender(enabled=False)
    print("Testing DirectInput Sender (Simulasi)...")
    sender.press_key('Q')
    sender.press_key('W')
    sender.press_key('E')
    sender.press_key('R')
    print("Test Selesai.")
