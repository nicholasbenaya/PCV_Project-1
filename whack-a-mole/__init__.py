"""Paket Whack-A-Mole Virtual berbasis OpenCV murni.

Modul ini mengimplementasikan game interaktif Whack-A-Mole dengan pelacakan
centroid tangan menggunakan segmentasi warna kulit HSV dan moments kontur.
"""

from .config import *
from .entities import HitEffect, Hole, Mole, ScoreManager
from .game_engine import GameEngine, GameState
from .hand_tracker import HandTracker, HandTrackingResult
from .ui_renderer import UIRenderer

__all__ = [
    "GameEngine",
    "GameState",
    "HandTracker",
    "HandTrackingResult",
    "Hole",
    "Mole",
    "HitEffect",
    "ScoreManager",
    "UIRenderer",
]
