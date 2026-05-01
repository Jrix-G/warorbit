from __future__ import annotations

from typing import Any, Dict, List


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def obs_to_game_dict(obs: Any) -> Dict[str, Any]:
    planets = []
    for p in list(_get(obs, "planets", []) or []):
        planets.append({
            "id": int(p[0]),
            "owner": int(p[1]),
            "x": float(p[2]),
            "y": float(p[3]),
            "radius": float(p[4]),
            "ships": float(p[5]),
            "production": float(p[6]),
        })

    fleets = []
    for f in list(_get(obs, "fleets", []) or []):
        fleets.append({
            "id": int(f[0]),
            "owner": int(f[1]),
            "x": float(f[2]),
            "y": float(f[3]),
            "ships": float(f[4]),
            "target_id": int(f[5]),
            "eta": int(f[6]),
            "source_id": int(f[7]) if len(f) > 7 else -1,
        })

    init_planets = []
    for p in list(_get(obs, "initial_planets", []) or []):
        init_planets.append({
            "id": int(p[0]),
            "owner": int(p[1]),
            "x": float(p[2]),
            "y": float(p[3]),
            "radius": float(p[4]),
            "ships": float(p[5]),
            "production": float(p[6]),
        })

    return {
        "my_id": int(_get(obs, "player", 0) or 0),
        "turn": int(_get(obs, "step", 0) or 0),
        "planets": planets,
        "fleets": fleets,
        "initial_planets": init_planets,
        "player_ids": sorted({int(p["owner"]) for p in planets if int(p["owner"]) >= 0}),
        "is_four_player": len({int(p["owner"]) for p in planets if int(p["owner"]) >= 0}) >= 4,
        "remaining_overage_time": int(_get(obs, "remainingOverageTime", 0) or 0),
    }


def action_to_kaggle_list(action_tuple: tuple[int, int, int]) -> List[List[int]]:
    src, tgt, ships = action_tuple
    if src < 0 or tgt < 0 or ships <= 0:
        return []
    return [[int(src), int(tgt), int(ships)]]

