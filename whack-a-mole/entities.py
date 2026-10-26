"""Entitas objek dalam game Whack-A-Mole (Hole, Mole, HitEffect, Score, Survival, UI Button)."""

import math
import random
from typing import List, Optional, Tuple

try:
    from .config import (
        BUTTON_HOVER_TRIGGER_TIME,
        SURVIVAL_INITIAL_LIVES,
        SURVIVAL_MAX_LIVES,
        SURVIVAL_TIER_INTERVAL,
    )
except ImportError:
    from config import (
        BUTTON_HOVER_TRIGGER_TIME,
        SURVIVAL_INITIAL_LIVES,
        SURVIVAL_MAX_LIVES,
        SURVIVAL_TIER_INTERVAL,
    )


class Hole:
    """Representasi lubang galian pada arena permainan."""

    def __init__(self, index: int, x: int, y: int, radius: int):
        self.index = index
        self.x = x
        self.y = y
        self.radius = radius

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def contains(self, point: Optional[Tuple[int, int]]) -> bool:
        """Memeriksa apakah koordinat titik berada di dalam radius lubang."""
        if point is None:
            return False
        dist = math.hypot(point[0] - self.x, point[1] - self.y)
        return dist <= self.radius


class Mole:
    """Objek target mole yang muncul di lubang tertentu."""

    def __init__(
        self,
        mole_id: int,
        initial_hole_idx: int,
        spawn_time: float,
        lifespan: float,
    ):
        self.mole_id = mole_id
        self.current_hole_idx = initial_hole_idx
        self.spawn_time = spawn_time
        self.lifespan = lifespan

    def is_expired(self, current_time: float) -> bool:
        """Memeriksa apakah durasi kemunculan mole telah terlampaui."""
        return (current_time - self.spawn_time) >= self.lifespan

    def respawn(
        self,
        current_time: float,
        forbidden_holes: List[int],
        total_holes: int,
        min_lifespan: float,
        max_lifespan: float,
    ) -> int:
        """Memindahkan mole ke lubang acak selain lubang yang dilarang."""
        available = [i for i in range(total_holes) if i not in forbidden_holes]
        if not available:
            available = [i for i in range(total_holes) if i != self.current_hole_idx]
        if not available:
            available = list(range(total_holes))

        self.current_hole_idx = random.choice(available)
        self.spawn_time = current_time
        self.lifespan = random.uniform(min_lifespan, max_lifespan)
        return self.current_hole_idx

    def get_time_ratio(self, current_time: float) -> float:
        """Menghitung sisa durasi kemunculan dalam rentang normalisasi 0.0 hingga 1.0."""
        elapsed = current_time - self.spawn_time
        remaining = max(0.0, self.lifespan - elapsed)
        return min(1.0, remaining / max(0.001, self.lifespan))


class HitEffect:
    """Efek animasi visual saat mole berhasil dipukul."""

    def __init__(
        self,
        position: Tuple[int, int],
        spawn_time: float,
        duration: float = 0.45,
        combo: int = 1,
    ):
        self.position = position
        self.spawn_time = spawn_time
        self.duration = duration
        self.combo = combo

    def is_alive(self, current_time: float) -> bool:
        return (current_time - self.spawn_time) < self.duration

    def get_progress(self, current_time: float) -> float:
        age = current_time - self.spawn_time
        return max(0.0, min(1.0, age / self.duration))


class ScoreManager:
    """Pencatat skor, combo pukulan beruntun, dan hitungan target."""

    def __init__(self):
        self.score: int = 0
        self.hits_count: int = 0
        self.combo: int = 0
        self.max_combo: int = 0

    def record_hit(self) -> int:
        """Menghitung perolehan skor pukulan dengan multiplier combo bertingkat."""
        self.hits_count += 1
        self.combo += 1
        if self.combo > self.max_combo:
            self.max_combo = self.combo

        # Multiplier: combo 1-2 = 1x, 3-5 = 2x, 6-9 = 3x, >=10 = 4x
        multiplier = 1
        if self.combo >= 10:
            multiplier = 4
        elif self.combo >= 6:
            multiplier = 3
        elif self.combo >= 3:
            multiplier = 2

        points = 10 * multiplier
        self.score += points
        return points

    def record_miss(self) -> None:
        """Mereset rentetan combo bila mole kabur tanpa dipukul."""
        self.combo = 0

    def reset_session(self) -> None:
        self.score = 0
        self.hits_count = 0
        self.combo = 0
        self.max_combo = 0


class SurvivalTracker:
    """Pengelola siklus hidup, HP nyawa, dan intensitas mode Survival."""

    def __init__(self, start_time: float):
        self.lives: int = SURVIVAL_INITIAL_LIVES
        self.max_lives: int = SURVIVAL_MAX_LIVES
        self.start_time: float = start_time

    def get_elapsed(self, current_time: float) -> float:
        return max(0.0, current_time - self.start_time)

    def get_intensity_tier(self, current_time: float) -> int:
        """Menghitung tingkat intensitas 1 sampai 4 berdasarkan durasi bertahan."""
        elapsed = self.get_elapsed(current_time)
        tier = int(elapsed // SURVIVAL_TIER_INTERVAL) + 1
        return max(1, min(4, tier))

    def get_mole_speed(self, current_time: float) -> Tuple[float, float]:
        """Mengembalikan batas waktu kemunculan mole (semakin lama semakin cepat)."""
        tier = self.get_intensity_tier(current_time)
        if tier == 1:
            return (1.60, 1.95)
        elif tier == 2:
            return (1.25, 1.55)
        elif tier == 3:
            return (0.95, 1.20)
        else:
            return (0.60, 0.85)

    def get_simultaneous_moles_count(self, current_time: float, total_holes: int) -> int:
        """Mengizinkan 2 mole muncul bersamaan jika grid cukup besar dan intensitas tinggi."""
        tier = self.get_intensity_tier(current_time)
        if total_holes >= 6 and tier >= 3:
            return 2
        return 1

    def lose_life(self) -> bool:
        """Mengurangi nyawa saat mole kabur. Mengembalikan False jika pemain kalah."""
        self.lives = max(0, self.lives - 1)
        return self.lives > 0

    def gain_life(self) -> None:
        """Menambah nyawa bonus (misal setelah bonus combo)."""
        self.lives = min(self.max_lives, self.lives + 1)


class InteractiveButton:
    """Komponen tombol UI interaktif yang dapat diklik lewat keyboard atau hover tangan."""

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        label: str,
        action_id: str,
        hotkey_desc: str = "",
        enabled: bool = True,
        locked: bool = False,
    ):
        self.rect = rect  # (x1, y1, x2, y2)
        self.label = label
        self.action_id = action_id
        self.hotkey_desc = hotkey_desc
        self.enabled = enabled
        self.locked = locked

        self.hover_start_time: Optional[float] = None
        self.is_hovered: bool = False

    def contains(self, point: Optional[Tuple[int, int]]) -> bool:
        if point is None:
            return False
        x, y = point
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def update_hover(self, point: Optional[Tuple[int, int]], current_time: float) -> bool:
        """Memperbarui status hover dan mendeteksi klik virtual saat waktu tahan terpenuhi."""
        if not self.enabled or self.locked:
            self.hover_start_time = None
            self.is_hovered = False
            return False

        hovering = self.contains(point)
        self.is_hovered = hovering

        if hovering:
            if self.hover_start_time is None:
                self.hover_start_time = current_time
            elif (current_time - self.hover_start_time) >= BUTTON_HOVER_TRIGGER_TIME:
                # Waktu tahan terpenuhi -> Picu aksi klik
                self.hover_start_time = None
                return True
        else:
            self.hover_start_time = None

        return False

    def get_hover_ratio(self, current_time: float) -> float:
        """Mengembalikan rasio pengisian lingkaran tahan hover (0.0 s.d 1.0)."""
        if self.hover_start_time is None or not self.is_hovered:
            return 0.0
        elapsed = current_time - self.hover_start_time
        return max(0.0, min(1.0, elapsed / BUTTON_HOVER_TRIGGER_TIME))
