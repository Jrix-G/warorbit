"""Orbit Wars V6 — NumpyEvaluator.

Small 3-layer network (~900 params) that estimates win probability
for a player from a GameState. Trained by self-play. Numpy only.

Also provides:
  - extract_features(state, me) → np.ndarray(24,)
  - evaluate_state(state, me, evaluator=None) → float [0,1]
  - encode_evaluator / decode_evaluator (base64 for embedding in bot_v6.py)
"""

from __future__ import annotations

import base64
import io
import math
from typing import Optional

import numpy as np

from sim import (
    GameState,
    CENTER, SUN_RADIUS, TOTAL_TURNS, ROTATION_RADIUS_LIMIT,
    P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD,
    F_OWNER, F_SHIPS,
)


# --- Feature extraction (24 dims) ------------------------------------------

def extract_features(state: GameState, me: int) -> np.ndarray:
    """Extract 24 normalised features from a GameState for player `me`."""
    features = np.zeros(24, dtype=np.float32)
    planets = state.planets
    fleets = state.fleets

    # --- Per-player ship totals
    total_ships = np.zeros(4, dtype=np.float64)
    for p in range(4):
        if len(planets) > 0:
            mask = planets[:, P_OWNER] == p
            if np.any(mask):
                total_ships[p] += float(np.sum(planets[mask, P_SHIPS]))
        if len(fleets) > 0:
            fmask = fleets[:, F_OWNER] == p
            if np.any(fmask):
                total_ships[p] += float(np.sum(fleets[fmask, F_SHIPS]))

    my_ships = total_ships[me]
    all_ships = total_ships.sum() + 1e-8

    features[0] = my_ships / all_ships
    features[1] = my_ships / (max(total_ships) + 1e-8)

    # --- Planet counts / production
    n_planets = max(len(planets), 1)
    my_planet_mask = planets[:, P_OWNER] == me if len(planets) > 0 else np.array([], dtype=bool)
    my_planets = planets[my_planet_mask] if len(planets) > 0 and np.any(my_planet_mask) else np.zeros((0, 7), dtype=np.float32)

    features[2] = len(my_planets) / n_planets
    features[3] = float(np.sum(my_planets[:, P_PROD])) / max(n_planets * 3.0, 1.0) if len(my_planets) > 0 else 0.0

    # --- Incoming threat (enemy fleets near my planets)
    incoming_threat = 0.0
    if len(fleets) > 0 and len(my_planets) > 0:
        enemy_mask = fleets[:, F_OWNER] != me
        enemy_fleets = fleets[enemy_mask] if np.any(enemy_mask) else np.zeros((0, 7), dtype=np.float32)
        for ef in enemy_fleets:
            for mp in my_planets:
                d = math.hypot(float(ef[2]) - float(mp[P_X]),
                               float(ef[3]) - float(mp[P_Y]))
                if d < 30.0:
                    incoming_threat += float(ef[F_SHIPS])

    features[4] = incoming_threat / (my_ships + 1e-8)

    # --- Game phase
    features[5] = state.step / 500.0
    features[6] = 1.0 if state.step < 40 else 0.0
    features[7] = 1.0 if 40 <= state.step < 150 else 0.0
    features[8] = 1.0 if state.step >= 150 else 0.0

    # --- Comets
    features[9] = 1.0 if len(state.comet_ids) > 0 else 0.0
    comet_owned = 0
    if len(state.comet_ids) > 0 and len(planets) > 0:
        for cid in state.comet_ids:
            pmask = (planets[:, P_ID] == cid) & (planets[:, P_OWNER] == me)
            if np.any(pmask):
                comet_owned += 1
    features[10] = comet_owned / max(len(state.comet_ids), 1)

    # --- Enemy ship distribution
    enemy_ships = sorted(
        [total_ships[p] for p in range(4) if p != me],
        reverse=True
    )
    for i, s in enumerate(enemy_ships[:3]):
        features[11 + i] = s / all_ships  # indices 11, 12, 13

    # --- Geographic spread
    if len(my_planets) > 1:
        cx = float(np.mean(my_planets[:, P_X]))
        cy = float(np.mean(my_planets[:, P_Y]))
        spread = float(np.mean(np.hypot(my_planets[:, P_X] - cx, my_planets[:, P_Y] - cy)))
        features[14] = spread / 70.0

    # --- Distance to sun
    if len(my_planets) > 0:
        sun_dists = np.hypot(my_planets[:, P_X] - CENTER, my_planets[:, P_Y] - CENTER)
        features[15] = float(np.min(sun_dists)) / 70.0

    # --- Time remaining
    remaining = (500 - state.step) / 500.0
    features[16] = remaining
    features[17] = 1.0 if remaining < 0.1 else 0.0

    # --- Fleet ratio (ships in flight vs total)
    my_fleet_ships = 0.0
    if len(fleets) > 0:
        fmask = fleets[:, F_OWNER] == me
        if np.any(fmask):
            my_fleet_ships = float(np.sum(fleets[fmask, F_SHIPS]))
    features[18] = my_fleet_ships / (my_ships + 1e-8)

    # --- Reachable neutral production
    neutral_mask = planets[:, P_OWNER] == -1 if len(planets) > 0 else np.array([], dtype=bool)
    neutral_planets = planets[neutral_mask] if len(planets) > 0 and np.any(neutral_mask) else np.zeros((0, 7), dtype=np.float32)
    if len(neutral_planets) > 0 and len(my_planets) > 0:
        cx = float(np.mean(my_planets[:, P_X]))
        cy = float(np.mean(my_planets[:, P_Y]))
        nd = np.hypot(neutral_planets[:, P_X] - cx, neutral_planets[:, P_Y] - cy)
        close_prod = float(np.sum(neutral_planets[nd < 40.0, P_PROD]))
        features[19] = close_prod / 20.0

    # features 20-23 reserved
    return features


# --- Network ----------------------------------------------------------------

class NumpyEvaluator:
    """3-layer MLP. Input(24)→Dense(32,relu)→Dense(16,relu)→Dense(1,sigmoid).

    ~900 parameters total. Inference ~0.05 ms.
    """

    INPUT_DIM = 24
    H1 = 32
    H2 = 16

    def __init__(self, seed: Optional[int] = None):
        rng = np.random.RandomState(seed)

        def he(fan_in, *shape):
            return rng.randn(*shape).astype(np.float32) * math.sqrt(2.0 / fan_in)

        self.W1 = he(self.INPUT_DIM, self.INPUT_DIM, self.H1)
        self.b1 = np.zeros(self.H1, dtype=np.float32)
        self.W2 = he(self.H1, self.H1, self.H2)
        self.b2 = np.zeros(self.H2, dtype=np.float32)
        self.W3 = he(self.H2, self.H2, 1)
        self.b3 = np.zeros(1, dtype=np.float32)

    def predict(self, features: np.ndarray) -> float:
        """Forward pass for a single feature vector (24,). Returns [0,1]."""
        h1 = np.maximum(0.0, features @ self.W1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)
        logit = float((h2 @ self.W3 + self.b3)[0])
        return float(1.0 / (1.0 + math.exp(-logit)))

    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Batch inference. Shape (N, 24) → (N,) in [0,1]."""
        h1 = np.maximum(0.0, features_batch @ self.W1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)
        logit = (h2 @ self.W3 + self.b3)[:, 0]
        return (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)

    def get_params(self) -> np.ndarray:
        return np.concatenate([
            self.W1.ravel(), self.b1,
            self.W2.ravel(), self.b2,
            self.W3.ravel(), self.b3,
        ])

    def set_params(self, params: np.ndarray) -> None:
        params = params.astype(np.float32)
        i = 0
        for attr, shape in [
            ("W1", (self.INPUT_DIM, self.H1)),
            ("b1", (self.H1,)),
            ("W2", (self.H1, self.H2)),
            ("b2", (self.H2,)),
            ("W3", (self.H2, 1)),
            ("b3", (1,)),
        ]:
            size = int(np.prod(shape))
            setattr(self, attr, params[i: i + size].reshape(shape))
            i += size

    def n_params(self) -> int:
        return len(self.get_params())

    def save(self, path: str) -> None:
        np.save(path, self.get_params())

    def load(self, path: str) -> None:
        self.set_params(np.load(path))


# --- Evaluate ---------------------------------------------------------------

def _heuristic_eval(state: GameState, me: int) -> float:
    """Fast heuristic fallback evaluation (no NN required)."""
    total_s = 0.0
    my_s = 0.0
    total_p = 0.0
    my_p = 0.0

    if len(state.planets) > 0:
        for row in state.planets:
            s, pr, o = float(row[P_SHIPS]), float(row[P_PROD]), int(row[P_OWNER])
            total_s += s
            total_p += pr
            if o == me:
                my_s += s
                my_p += pr

    if len(state.fleets) > 0:
        for row in state.fleets:
            s, o = float(row[F_SHIPS]), int(row[F_OWNER])
            total_s += s
            if o == me:
                my_s += s

    if total_s < 1e-6:
        return 0.5

    ship_ratio = my_s / total_s
    prod_ratio = my_p / max(total_p, 1.0)

    # Blend: early game favours production, late game favours ships
    t = state.step / 500.0
    return (1 - t) * (0.4 * prod_ratio + 0.6 * ship_ratio) + t * (0.6 * ship_ratio + 0.4 * prod_ratio)


def evaluate_state(state: GameState, me: int,
                   evaluator: Optional[NumpyEvaluator] = None) -> float:
    """Score state for player `me`. Uses NN if provided, else heuristic."""
    if evaluator is not None:
        return evaluator.predict(extract_features(state, me))
    return _heuristic_eval(state, me)


# --- Serialisation for embedding in bot_v6.py ------------------------------

def encode_evaluator(evaluator: NumpyEvaluator) -> str:
    """Return base64 string of evaluator params (for embedding in bot_v6.py)."""
    buf = io.BytesIO()
    np.save(buf, evaluator.get_params())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_evaluator(b64_string: str) -> NumpyEvaluator:
    """Reconstruct NumpyEvaluator from base64 string."""
    data = base64.b64decode(b64_string)
    params = np.load(io.BytesIO(data))
    ev = NumpyEvaluator()
    ev.set_params(params)
    return ev


def save_evaluator_b64(evaluator: NumpyEvaluator, path: str = "evaluator_b64.txt") -> None:
    b64 = encode_evaluator(evaluator)
    with open(path, "w") as f:
        f.write(b64)
    print(f"Saved {len(b64)} chars to {path}")
