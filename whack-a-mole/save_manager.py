"""Manajer penyimpanan data persistensi level progression, rekor skor, dan kalibrasi."""

import json
import os
from typing import Dict, Optional

try:
    from .config import DEFAULT_GRID_INDEX
except ImportError:
    from config import DEFAULT_GRID_INDEX


class SaveManager:
    """Pengelola pembacaan dan penyimpanan file JSON progres permainan."""

    def __init__(self, filename: str = "save_data.json"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.join(base_dir, filename)

        self.unlocked_level: int = 1
        self.level_high_scores: Dict[str, int] = {}
        self.level_cleared: Dict[str, bool] = {}
        self.survival_high_score: int = 0
        self.selected_grid_index: int = DEFAULT_GRID_INDEX
        self.calib_data: Optional[Dict] = None

        self.load()

    def load(self) -> None:
        """Memuat data tersimpan dari disk jika ada."""
        if not os.path.exists(self.filepath):
            self.save()
            return

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.unlocked_level = int(data.get("unlocked_level", 1))
            self.level_high_scores = data.get("level_high_scores", {})
            self.level_cleared = data.get("level_cleared", {})
            self.survival_high_score = int(data.get("survival_high_score", 0))
            self.selected_grid_index = int(data.get("selected_grid_index", DEFAULT_GRID_INDEX))
            self.calib_data = data.get("calib_data", None)
        except Exception:
            self.unlocked_level = 1
            self.level_high_scores = {}
            self.level_cleared = {}
            self.survival_high_score = 0
            self.selected_grid_index = DEFAULT_GRID_INDEX
            self.calib_data = None

    def save(self) -> None:
        """Menyimpan data progres ke file JSON."""
        data = {
            "unlocked_level": self.unlocked_level,
            "level_high_scores": self.level_high_scores,
            "level_cleared": self.level_cleared,
            "survival_high_score": self.survival_high_score,
            "selected_grid_index": self.selected_grid_index,
            "calib_data": self.calib_data,
        }
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def unlock_next_level(self, current_level: int) -> None:
        """Membuka level berikutnya setelah pemain menuntaskan level saat ini."""
        self.level_cleared[str(current_level)] = True
        next_level = current_level + 1
        if next_level > self.unlocked_level and next_level <= 10:
            self.unlocked_level = next_level
        self.save()

    def record_level_score(self, level_num: int, score: int) -> None:
        """Memperbarui skor tertinggi pada level tertentu."""
        key = str(level_num)
        prev = self.level_high_scores.get(key, 0)
        if score > prev:
            self.level_high_scores[key] = score
            self.save()

    def record_survival_score(self, score: int) -> bool:
        """Memperbarui rekor skor tertinggi mode survival."""
        if score > self.survival_high_score:
            self.survival_high_score = score
            self.save()
            return True
        return False

    def set_grid_index(self, index: int) -> None:
        """Menyimpan preferensi grid terpilih."""
        self.selected_grid_index = index
        self.save()

    def set_calibration_data(self, calib: Dict) -> None:
        """Menyimpan profil kalibrasi warna kulit pemain."""
        self.calib_data = calib
        self.save()
