from __future__ import annotations

from typing import Any, Dict, List

from .trajectory import safe_plan_shot


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
        # Official observations encode fleets as:
        # [id, owner, x, y, angle, source_planet_id, ships].
        # Older code treated angle/source/ships as ships/target/eta, which
        # poisoned incoming-fleet and total-ship features.
        angle = float(f[4]) if len(f) > 4 else 0.0
        source_id = int(f[5]) if len(f) > 5 else -1
        ships = float(f[6]) if len(f) > 6 else 0.0
        fleets.append({
            "id": int(f[0]),
            "owner": int(f[1]),
            "x": float(f[2]),
            "y": float(f[3]),
            "angle": angle,
            "ships": ships,
            "target_id": -1,
            "eta": 100,
            "source_id": source_id,
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

    comets = []
    for c in list(_get(obs, "comets", []) or []):
        comets.append(list(c))

    observed_owners = {int(p["owner"]) for p in planets if int(p["owner"]) >= 0}
    observed_owners.update({int(p["owner"]) for p in init_planets if int(p["owner"]) >= 0})
    observed_owners.update({int(f["owner"]) for f in fleets if int(f["owner"]) >= 0})
    max_observed_owner = max(observed_owners, default=1)
    # Keep owner slots stable across turns. Dynamic `sorted(owners)` changes
    # one-hot meanings when a player temporarily has no visible assets.
    player_ids = list(range(max(4, max_observed_owner + 1)))[:4]

    return {
        "my_id": int(_get(obs, "player", 0) or 0),
        "turn": int(_get(obs, "step", 0) or 0),
        "planets": planets,
        "fleets": fleets,
        "initial_planets": init_planets,
        "angular_velocity": float(_get(obs, "angular_velocity", 0.0) or 0.0),
        "comets": comets,
        "comet_planet_ids": list(_get(obs, "comet_planet_ids", []) or []),
        "player_ids": player_ids,
        "is_four_player": len(observed_owners) >= 4,
        "remaining_overage_time": int(_get(obs, "remainingOverageTime", 0) or 0),
    }


def action_to_kaggle_list(action_tuple: tuple[int, int, int], game: Dict[str, Any] | None = None) -> List[List[int | float]]:
    src, tgt, ships = action_tuple
    if src < 0 or tgt < 0 or ships <= 0:
        return []
    if game is None:
        raise ValueError("game is required to convert target-id action into an official angle action")
    src_planet = next((p for p in game.get("planets", []) if int(p.get("id", -1)) == int(src)), None)
    tgt_planet = next((p for p in game.get("planets", []) if int(p.get("id", -1)) == int(tgt)), None)
    if src_planet is None or tgt_planet is None:
        return []
    angle = safe_plan_shot(src_planet, tgt_planet, game, ships=int(ships))
    if angle is None:
        return []
    return [[int(src), float(angle), int(ships)]]
