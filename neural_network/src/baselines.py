from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np


class RandomAgent:
    def act(self, game: Dict[str, Any]) -> tuple[int, int, int]:
        my_planets = [p for p in game.get("planets", []) if p["owner"] == game.get("my_id", 0) and p.get("ships", 0) >= 2]
        targets = [p for p in game.get("planets", []) if p["id"] not in {p["id"] for p in my_planets}]
        if not my_planets or not targets:
            return (-1, -1, 0)
        src = np.random.choice(my_planets)
        tgt = np.random.choice(targets)
        return int(src["id"]), int(tgt["id"]), max(1, int(float(src["ships"]) * 0.5))


class GreedyNearestWeakestAgent:
    def act(self, game: Dict[str, Any]) -> tuple[int, int, int]:
        my_id = game.get("my_id", 0)
        my_planets = [p for p in game.get("planets", []) if p["owner"] == my_id and float(p.get("ships", 0)) >= 2]
        if not my_planets:
            return (-1, -1, 0)
        best = None
        for src in my_planets:
            enemies = [p for p in game.get("planets", []) if p["id"] != src["id"] and p["owner"] != my_id]
            if not enemies:
                continue
            tgt = min(enemies, key=lambda p: (float(p.get("ships", 0.0)), np.hypot(float(src.get("x", 0.0)) - float(p.get("x", 0.0)), float(src.get("y", 0.0)) - float(p.get("y", 0.0)))))
            score = (float(tgt.get("ships", 0.0)), np.hypot(float(src.get("x", 0.0)) - float(tgt.get("x", 0.0)), float(src.get("y", 0.0)) - float(tgt.get("y", 0.0))))
            if best is None or score < best[0]:
                best = (score, src, tgt)
        if best is None:
            return (-1, -1, 0)
        _, src, tgt = best
        return int(src["id"]), int(tgt["id"]), max(1, int(float(src["ships"]) * 0.5))
