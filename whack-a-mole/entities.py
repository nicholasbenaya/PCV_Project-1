"""Entitas objek dalam game Whack-A-Mole (Hole, Mole, HitEffect, ScoreManager)."""

import math
import random
from typing import Optional, Tuple


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
        initial_hole_idx: int = 0,
        spawn_time: float = 0.0,
        lifespan: float = 1.75,
    ):
        self.current_hole_idx = initial_hole_idx
        self.spawn_time = spawn_time
        self.lifespan = lifespan

    def is_expired(self, current_time: float) -> bool:
        """Memeriksa apakah durasi kemunculan mole telah terlampaui."""
        return (current_time - self.spawn_time) >= self.lifespan

    def respawn(
        self,
        current_time: float,
        total_holes: int = 6,
        min_lifespan: float = 1.5,
        max_lifespan: float = 2.0,
    ) -> int:
        """Memindahkan mole ke lubang acak baru selain lubang saat ini."""
        choices = [i for i in range(total_holes) if i != self.current_hole_idx]
        self.current_hole_idx = random.choice(choices)
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
    ):
        self.position = position
        self.spawn_time = spawn_time
        self.duration = duration

    def is_alive(self, current_time: float) -> bool:
        """Memeriksa apakah animasi efek masih berlangsung."""
        return (current_time - self.spawn_time) < self.duration

    def get_progress(self, current_time: float) -> float:
        """Mengembalikan nilai progres animasi dari 0.0 hingga 1.0."""
        age = current_time - self.spawn_time
        return max(0.0, min(1.0, age / self.duration))


class ScoreManager:
    """Pencatat skor sesi saat ini dan rekor skor tertinggi."""

    def __init__(self):
        self.score: int = 0
        self.high_score: int = 0
        self.hits_count: int = 0

    def record_hit(self) -> None:
        """Menambah skor pukulan berhasil dan memperbarui rekor tertinggi."""
        self.score += 1
        self.hits_count += 1
        if self.score > self.high_score:
            self.high_score = self.score

    def reset_session(self) -> None:
        """Mereset skor sesi saat ini tanpa menghapus rekor skor tertinggi."""
        self.score = 0
        self.hits_count = 0
