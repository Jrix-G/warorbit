from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
import numpy as np

from .utils import normalize, log_norm


@dataclass
class EncodedGame:
    features: np.ndarray
    planet_mask: np.ndarray
    fleet_mask: np.ndarray
    player_mask: np.ndarray
    metadata: Dict[str, Any]


def _owner_one_hot(owner: int, players: Sequence[int], max_players: int) -> List[float]:
    vec = [0.0] * (max_players + 2)
    if owner == -1:
        vec[0] = 1.0
    else:
        try:
            idx = players.index(owner) + 1
            vec[idx] = 1.0
        except ValueError:
            vec[-1] = 1.0
    return vec


def encode_game_state(game: Any, config: Dict[str, Any]) -> EncodedGame:
    max_planets = int(config["max_planets"])
    max_fleets = int(config["max_fleets"])
    max_players = int(config["max_players"])
    player_ids = list(game.get("player_ids", []))

    features: List[float] = []
    planet_mask = np.zeros((max_planets,), dtype=np.float32)
    fleet_mask = np.zeros((max_fleets,), dtype=np.float32)
    player_mask = np.zeros((max_players,), dtype=np.float32)

    features.extend(build_global_features(game, config))

    for i, planet in enumerate(game.get("planets", [])[:max_planets]):
        features.extend(build_planet_features(planet, game, player_ids, config))
        planet_mask[i] = 1.0

    for i, fleet in enumerate(game.get("fleets", [])[:max_fleets]):
        features.extend(build_fleet_features(fleet, game, player_ids, config))
        fleet_mask[i] = 1.0

    for i, pid in enumerate(player_ids[:max_players]):
        features.extend(build_player_features(pid, game, config))
        player_mask[i] = 1.0

    return EncodedGame(
        features=np.asarray(features, dtype=np.float32),
        planet_mask=planet_mask,
        fleet_mask=fleet_mask,
        player_mask=player_mask,
        metadata={"player_ids": player_ids, "num_players": len(player_ids)},
    )


def build_global_features(game: Any, config: Dict[str, Any]) -> List[float]:
    planets = game.get("planets", [])
    fleets = game.get("fleets", [])
    my_id = game.get("my_id", 0)
    my_planets = [p for p in planets if p["owner"] == my_id]
    enemy_planets = [p for p in planets if p["owner"] not in (-1, my_id)]
    neutral_planets = [p for p in planets if p["owner"] == -1]
    total_ships = sum(p["ships"] for p in planets) + sum(f["ships"] for f in fleets)
    my_ships = sum(p["ships"] for p in planets if p["owner"] == my_id) + sum(f["ships"] for f in fleets if f["owner"] == my_id)
    enemy_ships = total_ships - my_ships
    return [
        normalize(game.get("turn", 0), 500.0),
        normalize(len(planets), config["max_planets"]),
        normalize(len(fleets), config["max_fleets"]),
        normalize(len(my_planets), config["max_planets"]),
        normalize(len(enemy_planets), config["max_planets"]),
        normalize(len(neutral_planets), config["max_planets"]),
        normalize(my_ships, config["ship_scale"]),
        normalize(enemy_ships, config["ship_scale"]),
        normalize(sum(p["production"] for p in my_planets), config["production_scale"]),
        normalize(sum(p["production"] for p in enemy_planets), config["production_scale"]),
        1.0 if game.get("is_four_player", False) else 0.0,
    ]


def build_planet_features(planet: Dict[str, Any], game: Any, player_ids: Sequence[int], config: Dict[str, Any]) -> List[float]:
    my_id = game.get("my_id", 0)
    incoming_my = sum(f["ships"] for f in game.get("fleets", []) if f["owner"] == my_id and f["target_id"] == planet["id"])
    incoming_enemy = sum(f["ships"] for f in game.get("fleets", []) if f["owner"] != my_id and f["target_id"] == planet["id"])
    return (
        _owner_one_hot(int(planet["owner"]), player_ids, int(config["max_players"]))
        + [
            1.0 if planet["owner"] == my_id else 0.0,
            1.0 if planet["owner"] not in (-1, my_id) else 0.0,
            1.0 if planet["owner"] == -1 else 0.0,
            normalize(planet["x"], config["board_scale"]),
            normalize(planet["y"], config["board_scale"]),
            normalize(planet.get("radius", 1.0), config["radius_scale"]),
            normalize(planet.get("production", 0.0), config["production_scale"]),
            normalize(planet.get("ships", 0.0), config["ship_scale"]),
            log_norm(planet.get("ships", 0.0), config["ship_scale"]),
            normalize(incoming_my, config["ship_scale"]),
            normalize(incoming_enemy, config["ship_scale"]),
            1.0 if incoming_enemy > incoming_my else 0.0,
            1.0 if planet["owner"] == -1 else 0.0,
            1.0 if planet["owner"] == my_id else 0.0,
            1.0 if planet["owner"] not in (-1, my_id) else 0.0,
            normalize(abs(incoming_enemy - incoming_my), config["ship_scale"]),
        ]
    )


def build_fleet_features(fleet: Dict[str, Any], game: Any, player_ids: Sequence[int], config: Dict[str, Any]) -> List[float]:
    return (
        _owner_one_hot(int(fleet["owner"]), player_ids, int(config["max_players"]))
        + [
            1.0 if fleet["owner"] == game.get("my_id", 0) else 0.0,
            1.0 if fleet["owner"] != game.get("my_id", 0) else 0.0,
            normalize(fleet["x"], config["board_scale"]),
            normalize(fleet["y"], config["board_scale"]),
            normalize(fleet["ships"], config["ship_scale"]),
            log_norm(fleet["ships"], config["ship_scale"]),
            normalize(fleet["eta"], config["horizon_scale"]),
            normalize(fleet["source_id"], config["planet_id_scale"]),
            normalize(fleet["target_id"], config["planet_id_scale"]),
            1.0 if fleet["eta"] <= 3 else 0.0,
        ]
    )


def build_player_features(pid: int, game: Any, config: Dict[str, Any]) -> List[float]:
    planets = game.get("planets", [])
    fleets = game.get("fleets", [])
    alive = any(p["owner"] == pid for p in planets) or any(f["owner"] == pid for f in fleets)
    planets_count = sum(1 for p in planets if p["owner"] == pid)
    ships_total = sum(p["ships"] for p in planets if p["owner"] == pid) + sum(f["ships"] for f in fleets if f["owner"] == pid)
    production_total = sum(p["production"] for p in planets if p["owner"] == pid)
    return [
        1.0 if pid == game.get("my_id", 0) else 0.0,
        1.0 if alive else 0.0,
        normalize(planets_count, config["max_planets"]),
        normalize(ships_total, config["ship_scale"]),
        normalize(production_total, config["production_scale"]),
        normalize(sum(p["ships"] for p in planets if p["owner"] == pid), config["ship_scale"]),
        normalize(sum(f["ships"] for f in fleets if f["owner"] == pid), config["ship_scale"]),
        normalize(sum(1 for p in planets if p["owner"] == pid and p.get("production", 0) > 0), config["max_planets"]),
    ]

