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
        time_bonus: float = 0.0,
    ):
        self.position = position
        self.spawn_time = spawn_time
        self.duration = duration
        self.combo = combo
        self.time_bonus = time_bonus

    def is_alive(self, current_time: float) -> bool:
        return (current_time - self.spawn_time) < self.duration

    def get_progress(self, current_time: float) -> float:
        elapsed = current_time - self.spawn_time
        return min(1.0, max(0.0, elapsed / self.duration))


class ScoreManager:
    """Pengelola perolehan skor, hit counter, kombo, dan akurasi sesi bermain."""

    def __init__(self):
        self.score: int = 0
        self.hits_count: int = 0
        self.misses_count: int = 0
        self.combo: int = 0
        self.max_combo: int = 0

    def reset_session(self) -> None:
        self.score = 0
        self.hits_count = 0
        self.misses_count = 0
        self.combo = 0
        self.max_combo = 0

    def record_hit(self) -> int:
        """Mencatat pukulan sukses dan menghitung kombo pelipat ganda."""
        self.hits_count += 1
        self.combo += 1
        if self.combo > self.max_combo:
            self.max_combo = self.combo

        # Multiplier kombo bertingkat
        multiplier = 1
        if self.combo >= 10:
            multiplier = 4
        elif self.combo >= 6:
            multiplier = 3
        elif self.combo >= 3:
            multiplier = 2

        added_points = 100 * multiplier
        self.score += added_points
        return added_points

    def record_miss(self) -> None:
        """Mereset kombo saat mole kabur sebelum dipukul."""
        self.misses_count += 1
        self.combo = 0

    @property
    def accuracy(self) -> float:
        total = self.hits_count + self.misses_count
        if total == 0:
            return 100.0
        return (self.hits_count / total) * 100.0


class SurvivalTracker:
    """Pelacak tingkat kesulitan dinamis untuk Mode Survival."""

    def __init__(self, start_time: float):
        self.start_time = start_time
        self.lives = SURVIVAL_INITIAL_LIVES
        self.max_lives = SURVIVAL_MAX_LIVES

    def get_elapsed(self, current_time: float) -> float:
        return max(0.0, current_time - self.start_time)

    def get_intensity_tier(self, current_time: float) -> int:
        """Mengembalikan tingkat intensitas (1 s.d 4) berdasarkan durasi bertahan."""
        elapsed = self.get_elapsed(current_time)
        tier = int(elapsed // SURVIVAL_TIER_INTERVAL) + 1
        return min(4, max(1, tier))

    def get_mole_speed(self, current_time: float) -> Tuple[float, float]:
        """Mengembalikan batas waktu kemunculan mole (lebih lama dan adaptif proporsional terhadap waktu bertahan)."""
        elapsed = self.get_elapsed(current_time)
        # Menurun secara proporsional dan bertahap terhadap durasi bertahan (rentang 0 s.d 150 detik)
        decay = min(1.0, max(0.0, elapsed / 150.0))
        # Dari 2.70 - 3.20 detik di awal -> turun bertahap hingga 1.30 - 1.70 detik saat bertahan sangat lama
        min_lifespan = 2.70 - (1.40 * decay)
        max_lifespan = 3.20 - (1.50 * decay)
        return (min_lifespan, max_lifespan)

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
    """Komponen tombol UI interaktif yang dapat dipilih lewat HIGH_FIVE (hover) dan FIST (click)."""

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

    def update_interaction(
        self,
        point: Optional[Tuple[int, int]],
        is_fist_click: bool = False,
        is_fist: bool = False,
    ) -> bool:
        """Memperbarui status hover (HIGH_FIVE) dan memicu aksi saat gestur kepalan (FIST) terdeteksi."""
        if not self.enabled or self.locked:
            self.is_hovered = False
            return False

        hovering = self.contains(point)
        self.is_hovered = hovering

        # Eksekusi aksi jika tombol di-hover dan terjadi gestur kepalan (FIST)
        if hovering and (is_fist_click or is_fist):
            return True

        return False

    def update_hover(self, point: Optional[Tuple[int, int]], current_time: float) -> bool:
        """Fallback kompatibilitas: hover biasa."""
        if not self.enabled or self.locked:
            self.hover_start_time = None
            self.is_hovered = False
            return False

        hovering = self.contains(point)
        self.is_hovered = hovering
        return False

    def get_hover_ratio(self, current_time: float) -> float:
        """Mengembalikan rasio pengisian lingkaran visual hover."""
        return 0.0
