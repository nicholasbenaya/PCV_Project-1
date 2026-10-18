"""
Invoker Game Logic & State Controller
Mengelola 3 Orb aktif (Quas, Wex, Exort), Resep 10 Mantra Invoker,
Slot Mantra D dan F (Quick Cast), dan integrasi dengan DirectInput sender.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
from direct_input import DirectInputSender


# Resep 10 Mantra Invoker berdasarkan kombinasi 3 orb terurut
INVOKER_SPELLS = {
    "QQQ": {"name": "Cold Snap", "color": (255, 200, 80), "desc": "Freezing stun"},
    "QQW": {"name": "Ghost Walk", "color": (240, 180, 160), "desc": "Invisibility & slow"},
    "QQE": {"name": "Ice Wall", "color": (255, 160, 120), "desc": "Glacial barrier"},
    "WWW": {"name": "EMP", "color": (255, 80, 220), "desc": "Mana burn explosion"},
    "QWW": {"name": "Tornado", "color": (220, 120, 255), "desc": "Cyclone lift & purge"},
    "EWW": {"name": "Alacrity", "color": (160, 140, 255), "desc": "Attack speed & damage buff"},
    "EEE": {"name": "Sun Strike", "color": (40, 100, 255), "desc": "Global pure damage strike"},
    "EEQ": {"name": "Forge Spirit", "color": (40, 160, 255), "desc": "Armor-melting summons"},
    "EEW": {"name": "Chaos Meteor", "color": (30, 80, 240), "desc": "Flaming meteor impact"},
    "EQW": {"name": "Deafening Blast", "color": (180, 180, 255), "desc": "Sonic wave knockback & disarm"}
}


class InvokerController:
    def __init__(self, direct_input_enabled: bool = True):
        self.input_sender = DirectInputSender(enabled=direct_input_enabled)
        self.active_orbs: List[str] = ["E", "E", "E"]  # Default 3 Exort
        self.slot_d: str = "Sun Strike"
        self.slot_f: str = "Chaos Meteor"
        self.last_action_msg: str = "Siap merapal mantra..."
        self.last_action_time: float = time.time()

    def add_orb(self, orb: str):
        """Menambahkan orb (Q, W, E) dan menggeser orb tertua."""
        orb = orb.upper()
        if orb in ["Q", "W", "E"]:
            self.active_orbs.pop(0)
            self.active_orbs.append(orb)
            self.input_sender.press_key(orb)
            orb_names = {"Q": "Quas (Es)", "W": "Wex (Petir)", "E": "Exort (Api)"}
            self.last_action_msg = f"Orb Ditambahkan: {orb_names[orb]}"
            self.last_action_time = time.time()

    def invoke(self) -> Tuple[str, Tuple[int, int, int]]:
        """Mengeksekusi Invoke (R) untuk mengombinasikan 3 orb menjadi mantra baru."""
        self.input_sender.press_key('R')

        # Urutkan orb secara alfabetis untuk mencocokkan resep
        recipe_key = "".join(sorted(self.active_orbs))
        spell_info = INVOKER_SPELLS.get(recipe_key, {"name": "Deafening Blast", "color": (200, 200, 255)})
        new_spell = spell_info["name"]

        # Pergeseran slot mantra
        if new_spell == self.slot_d:
            self.last_action_msg = f"Invoke: {new_spell} (Sudah di Slot D)"
        elif new_spell == self.slot_f:
            # Tukar slot D dan F
            self.slot_f = self.slot_d
            self.slot_d = new_spell
            self.last_action_msg = f"Invoke: {new_spell} (Dipindahkan ke Slot D)"
        else:
            self.slot_f = self.slot_d
            self.slot_d = new_spell
            self.last_action_msg = f"Invoke Mantra Baru: {new_spell}!"

        self.last_action_time = time.time()
        return self.slot_d, spell_info["color"]

    def cast_spell_d(self) -> Tuple[str, Tuple[int, int, int]]:
        """Quick Cast Mantra Slot 1 (D)."""
        self.input_sender.press_key('D')
        color = self.get_spell_color(self.slot_d)
        self.last_action_msg = f"★ QUICK CAST (D): {self.slot_d}! ★"
        self.last_action_time = time.time()
        return self.slot_d, color

    def cast_spell_f(self) -> Tuple[str, Tuple[int, int, int]]:
        """Quick Cast Mantra Slot 2 (F)."""
        self.input_sender.press_key('F')
        color = self.get_spell_color(self.slot_f)
        self.last_action_msg = f"★ QUICK CAST (F): {self.slot_f}! ★"
        self.last_action_time = time.time()
        return self.slot_f, color

    def get_spell_color(self, spell_name: str) -> Tuple[int, int, int]:
        for k, v in INVOKER_SPELLS.items():
            if v["name"] == spell_name:
                return v["color"]
        return (200, 200, 200)

    def render_dota_hud(self, frame: np.ndarray, current_seal: str, progress: float):
        """Merender antarmuka bergaya HUD Dota 2 di bagian bawah frame OpenCV."""
        h, w = frame.shape[:2]
        hud_h = 105
        hud_y = h - hud_h

        # Latar belakang HUD semi-transparan
        hud_bg = frame[hud_y:h, 0:w].copy()
        cv2.rectangle(hud_bg, (0, 0), (w, hud_h), (12, 14, 20), -1)
        cv2.line(hud_bg, (0, 0), (w, 0), (0, 180, 220), 2)
        cv2.addWeighted(hud_bg, 0.88, frame[hud_y:h, 0:w], 0.12, 0, frame[hud_y:h, 0:w])

        # --- 1. Tiga Lingkaran Orb Aktif ---
        orb_colors = {
            "Q": ((255, 210, 80), "Quas (Q)"),
            "W": ((255, 70, 210), "Wex (W)"),
            "E": ((40, 120, 255), "Exort (E)")
        }
        start_x = 35
        for i, orb in enumerate(self.active_orbs):
            cx = start_x + i * 55
            cy = hud_y + 48
            color, label = orb_colors.get(orb, ((180, 180, 180), orb))
            cv2.circle(frame, (cx, cy), 22, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 24, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, orb, (cx - 8, cy + 8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (15, 15, 25), 2, cv2.LINE_AA)

        cv2.putText(frame, "ACTIVE ORBS", (start_x, hud_y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 210, 230), 1, cv2.LINE_AA)

        # Tombol Invoke (R)
        inv_x = start_x + 3 * 55 + 20
        inv_y = hud_y + 48
        cv2.rectangle(frame, (inv_x - 22, inv_y - 22), (inv_x + 22, inv_y + 22), (40, 190, 140), -1)
        cv2.rectangle(frame, (inv_x - 22, inv_y - 22), (inv_x + 22, inv_y + 22), (255, 255, 255), 2)
        cv2.putText(frame, "R", (inv_x - 8, inv_y + 8), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "INVOKE", (inv_x - 22, hud_y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 210, 230), 1, cv2.LINE_AA)

        # --- 2. Slot Mantra D dan F (Quick Cast) ---
        slot_d_x = inv_x + 75
        slot_f_x = slot_d_x + 165

        # Slot D Box
        color_d = self.get_spell_color(self.slot_d)
        cv2.rectangle(frame, (slot_d_x, hud_y + 18), (slot_d_x + 145, hud_y + 78), (25, 28, 40), -1)
        cv2.rectangle(frame, (slot_d_x, hud_y + 18), (slot_d_x + 145, hud_y + 78), color_d, 2)
        cv2.circle(frame, (slot_d_x + 18, hud_y + 48), 12, color_d, -1)
        cv2.putText(frame, "D", (slot_d_x + 12, hud_y + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(frame, self.slot_d, (slot_d_x + 36, hud_y + 52), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "SPELL 1 (D)", (slot_d_x, hud_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 210, 230), 1, cv2.LINE_AA)

        # Slot F Box
        color_f = self.get_spell_color(self.slot_f)
        cv2.rectangle(frame, (slot_f_x, hud_y + 18), (slot_f_x + 145, hud_y + 78), (25, 28, 40), -1)
        cv2.rectangle(frame, (slot_f_x, hud_y + 18), (slot_f_x + 145, hud_y + 78), color_f, 2)
        cv2.circle(frame, (slot_f_x + 18, hud_y + 48), 12, color_f, -1)
        cv2.putText(frame, "F", (slot_f_x + 12, hud_y + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(frame, self.slot_f, (slot_f_x + 36, hud_y + 52), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "SPELL 2 (F)", (slot_f_x, hud_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 210, 230), 1, cv2.LINE_AA)

        # --- 3. Status Pose Tangan Saat Ini & Charge Meter ---
        status_x = slot_f_x + 170
        cv2.putText(frame, f"POSE: {current_seal}", (status_x, hud_y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.68, (0, 235, 255), 2, cv2.LINE_AA)

        # Progress bar pengisian hold
        bar_w = 180
        bar_h = 14
        bar_y = hud_y + 50
        cv2.rectangle(frame, (status_x, bar_y), (status_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            cv2.rectangle(frame, (status_x, bar_y), (status_x + fill_w, bar_y + bar_h), (0, 255, 230), -1)
        cv2.rectangle(frame, (status_x, bar_y), (status_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

        cv2.putText(frame, self.last_action_msg, (status_x, hud_y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 180), 1, cv2.LINE_AA)
