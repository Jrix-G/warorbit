"""
bot_v13.py — Hybrid simulation + MLP scorer

Architecture:
  1. V12 simulation engine generates candidate swarms with computed features
  2. Small MLP (12→64→32→1) scores each candidate
  3. Greedy selection: highest-scoring candidates executed until max_actions
  4. Trained via REINFORCE on local_simulator.official_fast

Why hybrid:
  - V12 alone: heuristic ceiling, no learning
  - V11 alone: must learn physics from raw obs, slow convergence
  - V13: MLP only learns to RANK pre-validated candidates with rich features
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

# ── Game constants ────────────────────────────────────────────────────────────
_MAX_SPEED = 6.0
_SUN_X = _SUN_Y = 50.0
_SUN_R = 10.0
_BOARD = 100.0
_CENTER = 50.0
_ROT_LIMIT = 50.0

# ── Tuning ────────────────────────────────────────────────────────────────────
HORIZON = 50
MIN_GARRISON = 5
ATTACK_MARGIN = 1.05
MIN_SEND = 4
MAX_ACTIONS = 8
TOP_K_ATTACKS = 3
TOP_K_EXPANDS = 3
TOP_K_DEFENSES = 2
TOP_K_STAGING = 2
FEATURE_DIM = 12


# ═════════════════════════════ PHYSICS HELPERS ════════════════════════════════

def _get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _fleet_speed(ships: int) -> float:
    s = max(1, int(ships))
    return min(_MAX_SPEED, 1.0 + (_MAX_SPEED - 1.0) * (math.log(s) / math.log(1000)) ** 1.5)


def _dist(ax, ay, bx, by) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _seg_dist_to_point(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return _dist(px, py, ax, ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return _dist(px, py, ax + t * dx, ay + t * dy)


def _planet_pos(init_x, init_y, radius, av, step):
    dx = init_x - _CENTER
    dy = init_y - _CENTER
    r = math.sqrt(dx * dx + dy * dy)
    if r + radius >= _ROT_LIMIT:
        return init_x, init_y
    init_angle = math.atan2(dy, dx)
    angle = init_angle + av * step
    return _CENTER + r * math.cos(angle), _CENTER + r * math.sin(angle)


def _path_hits_sun(fx, fy, angle, total_dist) -> bool:
    ex = fx + math.cos(angle) * total_dist
    ey = fy + math.sin(angle) * total_dist
    return _seg_dist_to_point(_SUN_X, _SUN_Y, fx, fy, ex, ey) < _SUN_R


def _compute_eta_and_angle(
    src_x, src_y, src_radius,
    tgt_init_x, tgt_init_y, tgt_radius,
    ships, av, current_step,
):
    speed = _fleet_speed(ships)
    for eta in range(1, HORIZON + 1):
        tx, ty = _planet_pos(tgt_init_x, tgt_init_y, tgt_radius, av, current_step + eta)
        angle = math.atan2(ty - src_y, tx - src_x)
        fleet_sx = src_x + math.cos(angle) * (src_radius + 0.1)
        fleet_sy = src_y + math.sin(angle) * (src_radius + 0.1)
        d = _dist(fleet_sx, fleet_sy, tx, ty) - tgt_radius
        if d <= 0.0:
            return eta, angle
        if speed * eta >= d:
            if not _path_hits_sun(fleet_sx, fleet_sy, angle, speed * eta):
                return eta, angle
            else:
                return None
    return None


def _build_initial_map(obs):
    ip_list = _get(obs, 'initial_planets', []) or []
    return {int(p[0]): (float(p[2]), float(p[3]), float(p[4])) for p in ip_list}


def _build_arrival_table(obs, ip, av, current_step):
    obs_planets = list(_get(obs, 'planets', []) or [])
    obs_fleets = list(_get(obs, 'fleets', []) or [])
    table = {int(p[0]): [] for p in obs_planets}

    for fleet in obs_fleets:
        fx, fy = float(fleet[2]), float(fleet[3])
        angle = float(fleet[4])
        ships = int(fleet[6])
        fowner = int(fleet[1])
        speed = _fleet_speed(ships)
        cx, cy = fx, fy

        for t in range(1, HORIZON + 1):
            ocx, ocy = cx, cy
            cx += math.cos(angle) * speed
            cy += math.sin(angle) * speed
            if _seg_dist_to_point(_SUN_X, _SUN_Y, ocx, ocy, cx, cy) < _SUN_R:
                break
            if not (0.0 <= cx <= _BOARD and 0.0 <= cy <= _BOARD):
                break
            hit = False
            for planet in obs_planets:
                pid = int(planet[0])
                init_x, init_y, pradius = ip.get(
                    pid, (float(planet[2]), float(planet[3]), float(planet[4])))
                px, py = _planet_pos(init_x, init_y, pradius, av, current_step + t)
                if _seg_dist_to_point(px, py, ocx, ocy, cx, cy) < pradius:
                    table[pid].append((t, fowner, ships))
                    hit = True
                    break
            if hit:
                break
    return table


def _defenders_at_eta(planet, eta, arrival_table) -> float:
    pid = int(planet[0])
    owner = int(planet[1])
    ships = float(planet[5])
    prod = float(planet[6])
    if owner != -1:
        ships += prod * eta
    for (t, fowner, fships) in arrival_table.get(pid, []):
        if t <= eta and fowner == owner:
            ships += fships
        elif t <= eta and fowner != owner and owner != -1:
            ships -= fships
    return max(0.0, ships)


def _garrison_needed(planet, my_id, arrival_table) -> int:
    pid = int(planet[0])
    prod = float(planet[6])
    base = MIN_GARRISON
    max_threat = 0
    earliest = HORIZON + 1
    for (t, fowner, fships) in arrival_table.get(pid, []):
        if fowner != my_id and fships > max_threat:
            max_threat = fships
            earliest = t
    if max_threat > 0:
        needed = int(max_threat * 1.10 - prod * earliest)
        base = max(base, needed)
    return base


# ═════════════════════════════ CANDIDATE GENERATION ═══════════════════════════

def _make_features(
    cand_type: str,
    eta: int,
    defenders: float,
    swarm_ships: int,
    target_prod: float,
    target_is_enemy: bool,
    target_is_neutral: bool,
    my_total_ships: float,
    my_total_prod: float,
    total_prod: float,
    threat_on_src: float,
    src_ships: int,
    coordination_q: float,
    turn: int,
):
    """Return 12-dim feature vector for a candidate."""
    f = np.zeros(FEATURE_DIM, dtype=np.float32)
    f[0] = defenders / max(1.0, my_total_ships)
    f[1] = (target_prod * 10.0) / max(1.0, swarm_ships)
    f[2] = eta / 60.0
    f[3] = 1.0 if target_is_enemy else 0.0
    f[4] = 1.0 if target_is_neutral else 0.0
    f[5] = 1.0 if cand_type == 'defense' else 0.0
    f[6] = 1.0 if cand_type == 'staging' else 0.0
    f[7] = my_total_prod / max(1.0, total_prod)
    f[8] = threat_on_src / max(1.0, src_ships)
    f[9] = coordination_q
    f[10] = turn / 500.0
    f[11] = 1.0 if cand_type == 'noop' else 0.0
    return f


def _generate_attack_candidates(
    planets, my_id, ip, av, current_step, arrival_table,
    my_total_ships, my_total_prod, total_prod,
):
    """Generate up to TOP_K_ATTACKS swarm candidates against enemy planets."""
    my_planets = [p for p in planets if int(p[1]) == my_id]
    enemies = [p for p in planets if int(p[1]) not in (-1, my_id)]
    candidates = []

    for tgt in enemies:
        tid = int(tgt[0])
        tgt_ix, tgt_iy, tgt_r = ip.get(tid, (float(tgt[2]), float(tgt[3]), float(tgt[4])))

        src_options = []
        for src in my_planets:
            sid = int(src[0])
            sx, sy, sr = float(src[2]), float(src[3]), float(src[4])
            garrison = _garrison_needed(src, my_id, arrival_table)
            avail = int(src[5]) - garrison
            if avail < MIN_SEND:
                continue
            res = _compute_eta_and_angle(sx, sy, sr, tgt_ix, tgt_iy, tgt_r,
                                         avail, av, current_step)
            if res is None:
                continue
            eta, angle = res
            src_options.append((sid, eta, angle, avail, src))

        if not src_options:
            continue

        # Group by ETA — find best swarm composition
        best_eta_data = None
        best_score = -1.0
        for target_eta in sorted(set(e for (_, e, _, _, _) in src_options)):
            exact = [(sid, ang, av_s, src) for (sid, e, ang, av_s, src) in src_options if e == target_eta]
            earlier = [(sid, ang, av_s, src) for (sid, e, ang, av_s, src) in src_options if e < target_eta]

            total_exact = sum(av_s for (_, _, av_s, _) in exact)
            total_with_earlier = total_exact + sum(av_s for (_, _, av_s, _) in earlier)
            defenders = _defenders_at_eta(tgt, target_eta, arrival_table)

            if total_with_earlier < defenders * ATTACK_MARGIN:
                continue

            sources_used = exact + earlier
            total_swarm = total_with_earlier
            coord_q = total_exact / max(1.0, total_swarm)  # 1.0 = perfect simul

            prod_per_ship = float(tgt[6]) / max(1.0, total_swarm)
            score = prod_per_ship * (1.0 / max(1, target_eta)) * coord_q
            if score > best_score:
                best_score = score
                best_eta_data = (target_eta, defenders, total_swarm, coord_q, sources_used)

        if best_eta_data is None:
            continue

        target_eta, defenders, total_swarm, coord_q, sources = best_eta_data
        moves = [(sid, ang, av_s) for (sid, ang, av_s, _) in sources]
        threat_on_src = sum(_garrison_needed(src, my_id, arrival_table) for (_, _, _, src) in sources) / max(1, len(sources))
        avg_src_ships = sum(av_s for (_, _, av_s, _) in sources) / max(1, len(sources))

        feats = _make_features(
            'attack', target_eta, defenders, int(total_swarm),
            float(tgt[6]), True, False,
            my_total_ships, my_total_prod, total_prod,
            threat_on_src, int(avg_src_ships), coord_q, current_step,
        )
        candidates.append({
            'moves': moves,
            'features': feats,
            'type': 'attack',
            'score_hint': best_score,
            'sources': set(sid for (sid, _, _, _) in sources),
        })

    candidates.sort(key=lambda c: c['score_hint'], reverse=True)
    return candidates[:TOP_K_ATTACKS]


def _generate_expand_candidates(
    planets, my_id, ip, av, current_step, arrival_table,
    my_total_ships, my_total_prod, total_prod,
):
    my_planets = [p for p in planets if int(p[1]) == my_id]
    neutrals = [p for p in planets if int(p[1]) == -1]
    candidates = []

    for tgt in neutrals:
        tid = int(tgt[0])
        tgt_ix, tgt_iy, tgt_r = ip.get(tid, (float(tgt[2]), float(tgt[3]), float(tgt[4])))

        best_src_data = None
        best_dist = 1e9
        for src in my_planets:
            sid = int(src[0])
            sx, sy, sr = float(src[2]), float(src[3]), float(src[4])
            garrison = _garrison_needed(src, my_id, arrival_table)
            avail = int(src[5]) - garrison
            if avail < MIN_SEND:
                continue
            res = _compute_eta_and_angle(sx, sy, sr, tgt_ix, tgt_iy, tgt_r,
                                         avail, av, current_step)
            if res is None:
                continue
            eta, angle = res
            defenders = _defenders_at_eta(tgt, eta, arrival_table)
            if avail <= defenders:
                continue
            d = _dist(sx, sy, tgt_ix, tgt_iy)
            if d < best_dist:
                best_dist = d
                send = min(avail, int(defenders * 1.15) + int(float(tgt[6]) * eta) + MIN_SEND)
                send = max(send, MIN_SEND)
                best_src_data = (sid, eta, angle, send, defenders, src)

        if best_src_data is None:
            continue

        sid, eta, angle, send, defenders, src = best_src_data
        threat_on_src = _garrison_needed(src, my_id, arrival_table)
        feats = _make_features(
            'expand', eta, defenders, send,
            float(tgt[6]), False, True,
            my_total_ships, my_total_prod, total_prod,
            threat_on_src, int(src[5]), 1.0, current_step,
        )
        prod_per_ship = float(tgt[6]) / max(1.0, send)
        candidates.append({
            'moves': [(sid, angle, send)],
            'features': feats,
            'type': 'expand',
            'score_hint': prod_per_ship / max(1, eta),
            'sources': {sid},
        })

    candidates.sort(key=lambda c: c['score_hint'], reverse=True)
    return candidates[:TOP_K_EXPANDS]


def _generate_defense_candidates(
    planets, my_id, ip, av, current_step, arrival_table,
    my_total_ships, my_total_prod, total_prod,
):
    my_planets = [p for p in planets if int(p[1]) == my_id]
    candidates = []

    for mp in my_planets:
        mid = int(mp[0])
        threats = [(t, fships) for (t, fowner, fships) in arrival_table.get(mid, [])
                   if fowner != my_id and t <= 8]
        if not threats:
            continue
        max_threat = sum(s for (_, s) in threats)
        earliest = min(t for (t, _) in threats)
        current_def = int(mp[5])
        if current_def >= max_threat * 1.05:
            continue

        # Find nearest reinforcer
        neighbors = [p for p in my_planets if int(p[0]) != mid]
        for neighbor in sorted(neighbors,
                               key=lambda p: _dist(float(p[2]), float(p[3]),
                                                   float(mp[2]), float(mp[3]))):
            nid = int(neighbor[0])
            ngarrison = _garrison_needed(neighbor, my_id, arrival_table)
            navail = int(neighbor[5]) - ngarrison
            if navail < MIN_SEND:
                continue
            init_x, init_y, pradius = ip.get(mid, (float(mp[2]), float(mp[3]), float(mp[4])))
            res = _compute_eta_and_angle(
                float(neighbor[2]), float(neighbor[3]), float(neighbor[4]),
                init_x, init_y, pradius, navail, av, current_step,
            )
            if res is None:
                continue
            eta, angle = res
            if eta > earliest + 2:
                continue
            send = min(navail, max_threat + MIN_GARRISON * 2)
            feats = _make_features(
                'defense', eta, float(max_threat), send,
                float(mp[6]), False, False,
                my_total_ships, my_total_prod, total_prod,
                float(max_threat), int(neighbor[5]), 1.0, current_step,
            )
            candidates.append({
                'moves': [(nid, angle, send)],
                'features': feats,
                'type': 'defense',
                'score_hint': max_threat / max(1, eta),
                'sources': {nid},
            })
            break

    candidates.sort(key=lambda c: c['score_hint'], reverse=True)
    return candidates[:TOP_K_DEFENSES]


def _generate_staging_candidates(
    planets, my_id, ip, av, current_step, arrival_table,
    my_total_ships, my_total_prod, total_prod, enemies_exist,
):
    if not enemies_exist:
        return []
    my_planets = [p for p in planets if int(p[1]) == my_id]
    enemies = [p for p in planets if int(p[1]) not in (-1, my_id)]
    candidates = []

    front_ids = set()
    for mp in my_planets:
        if any(_dist(float(mp[2]), float(mp[3]), float(ep[2]), float(ep[3])) < 35.0
               for ep in enemies):
            front_ids.add(int(mp[0]))
    if not front_ids:
        return []

    rear = [p for p in my_planets if int(p[0]) not in front_ids]
    front = [p for p in my_planets if int(p[0]) in front_ids]
    if not rear or not front:
        return []

    for r in rear:
        rid = int(r[0])
        garrison = _garrison_needed(r, my_id, arrival_table)
        excess = int(r[5]) - garrison - int(r[6]) * 4
        if excess < MIN_SEND * 2:
            continue
        # nearest front
        best_f = min(front, key=lambda fp: _dist(float(r[2]), float(r[3]),
                                                  float(fp[2]), float(fp[3])))
        fid = int(best_f[0])
        fx_init, fy_init, fr = ip.get(fid, (float(best_f[2]), float(best_f[3]), float(best_f[4])))
        res = _compute_eta_and_angle(
            float(r[2]), float(r[3]), float(r[4]),
            fx_init, fy_init, fr, excess, av, current_step,
        )
        if res is None:
            continue
        eta, angle = res
        feats = _make_features(
            'staging', eta, 0.0, excess,
            float(best_f[6]), False, False,
            my_total_ships, my_total_prod, total_prod,
            0.0, int(r[5]), 1.0, current_step,
        )
        candidates.append({
            'moves': [(rid, angle, excess)],
            'features': feats,
            'type': 'staging',
            'score_hint': excess / max(1, eta),
            'sources': {rid},
        })

    candidates.sort(key=lambda c: c['score_hint'], reverse=True)
    return candidates[:TOP_K_STAGING]


def generate_all_candidates(obs, my_id, ip, av, current_step, arrival_table):
    """Return list of candidate dicts with computed features. Always includes noop."""
    planets = list(_get(obs, 'planets', []) or [])
    my_planets = [p for p in planets if int(p[1]) == my_id]
    enemies = [p for p in planets if int(p[1]) not in (-1, my_id)]

    my_total_ships = sum(float(p[5]) for p in my_planets)
    my_total_prod = sum(float(p[6]) for p in my_planets)
    total_prod = sum(float(p[6]) for p in planets if int(p[1]) != -1)

    cands = []
    cands += _generate_attack_candidates(planets, my_id, ip, av, current_step, arrival_table,
                                         my_total_ships, my_total_prod, total_prod)
    cands += _generate_expand_candidates(planets, my_id, ip, av, current_step, arrival_table,
                                         my_total_ships, my_total_prod, total_prod)
    cands += _generate_defense_candidates(planets, my_id, ip, av, current_step, arrival_table,
                                          my_total_ships, my_total_prod, total_prod)
    cands += _generate_staging_candidates(planets, my_id, ip, av, current_step, arrival_table,
                                          my_total_ships, my_total_prod, total_prod,
                                          bool(enemies))

    # Always add noop
    noop_feats = _make_features('noop', 0, 0.0, 0, 0.0, False, False,
                                 my_total_ships, my_total_prod, total_prod,
                                 0.0, 1, 1.0, current_step)
    cands.append({
        'moves': [],
        'features': noop_feats,
        'type': 'noop',
        'score_hint': 0.0,
        'sources': set(),
    })
    return cands


# ═════════════════════════════ MLP SCORER ═════════════════════════════════════

class MLPScorer:
    """Tiny MLP: 12 → 64 → 32 → 1, ReLU activations."""

    def __init__(self, weights: dict | None = None):
        if weights is not None:
            self.W1 = weights['W1'].astype(np.float32)
            self.b1 = weights['b1'].astype(np.float32)
            self.W2 = weights['W2'].astype(np.float32)
            self.b2 = weights['b2'].astype(np.float32)
            self.W3 = weights['W3'].astype(np.float32)
            self.b3 = weights['b3'].astype(np.float32)
        else:
            rng = np.random.default_rng(42)
            self.W1 = (rng.standard_normal((FEATURE_DIM, 64)) * np.sqrt(2.0 / FEATURE_DIM)).astype(np.float32)
            self.b1 = np.zeros(64, dtype=np.float32)
            self.W2 = (rng.standard_normal((64, 32)) * np.sqrt(2.0 / 64)).astype(np.float32)
            self.b2 = np.zeros(32, dtype=np.float32)
            self.W3 = (rng.standard_normal((32, 1)) * np.sqrt(2.0 / 32)).astype(np.float32)
            self.b3 = np.zeros(1, dtype=np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (N, 12) → (N,) logits."""
        h1 = np.maximum(0.0, X @ self.W1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)
        return (h2 @ self.W3 + self.b3).flatten()

    def forward_with_cache(self, X: np.ndarray):
        """Forward pass returning intermediate activations for backprop."""
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0.0, z2)
        z3 = h2 @ self.W3 + self.b3
        return z3.flatten(), {'X': X, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}

    def to_dict(self):
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2,
                'W3': self.W3, 'b3': self.b3}


_MLP_CACHE: dict[str, MLPScorer] = {}


def _load_mlp(weights_path: str | None = None) -> MLPScorer:
    if weights_path is None:
        weights_path = os.environ.get('V13_WEIGHTS', 'evaluations/scorer_v13.npz')
    weights_path = str(Path(weights_path))
    if weights_path in _MLP_CACHE:
        return _MLP_CACHE[weights_path]
    p = Path(weights_path)
    if p.exists():
        data = dict(np.load(p))
        mlp = MLPScorer(weights=data)
    else:
        mlp = MLPScorer(weights=None)
    _MLP_CACHE[weights_path] = mlp
    return mlp


def set_mlp(mlp: MLPScorer, weights_path: str = 'evaluations/scorer_v13.npz'):
    """Manually inject an MLP (used during training)."""
    _MLP_CACHE[str(Path(weights_path))] = mlp


# ═════════════════════════════ MAIN AGENT ═════════════════════════════════════

def agent(obs: Any, config: Any = None) -> list[list]:
    try:
        return _agent_inner(obs, config)
    except Exception:
        return []


def _agent_inner(obs, config):
    my_id = int(_get(obs, 'player', 0))
    current_step = int(_get(obs, 'step', 0))
    av = float(_get(obs, 'angular_velocity', 0.03))
    planets = list(_get(obs, 'planets', []) or [])
    if not planets:
        return []

    ip = _build_initial_map(obs)
    arrival_table = _build_arrival_table(obs, ip, av, current_step)

    candidates = generate_all_candidates(obs, my_id, ip, av, current_step, arrival_table)
    if not candidates:
        return []

    mlp = _load_mlp()
    feats = np.stack([c['features'] for c in candidates])
    scores = mlp.forward(feats)

    # Greedy selection: pick top scores while respecting source commitments
    order = np.argsort(-scores)
    used_sources: set[int] = set()
    actions: list[list] = []
    selected_indices: list[int] = []

    for idx in order:
        c = candidates[int(idx)]
        if c['type'] == 'noop':
            if not actions and len(selected_indices) == 0:
                selected_indices.append(int(idx))
            break
        if c['sources'] & used_sources:
            continue
        if not c['moves']:
            continue
        for (sid, angle, ships) in c['moves']:
            if sid in used_sources:
                continue
            actions.append([int(sid), float(angle), int(ships)])
            used_sources.add(sid)
            if len(actions) >= MAX_ACTIONS:
                break
        selected_indices.append(int(idx))
        if len(actions) >= MAX_ACTIONS:
            break

    return actions


def get_candidates_and_scores(obs):
    """Helper for training: returns candidates + scores without executing."""
    my_id = int(_get(obs, 'player', 0))
    current_step = int(_get(obs, 'step', 0))
    av = float(_get(obs, 'angular_velocity', 0.03))
    planets = list(_get(obs, 'planets', []) or [])
    if not planets:
        return [], np.zeros(0)
    ip = _build_initial_map(obs)
    arrival_table = _build_arrival_table(obs, ip, av, current_step)
    candidates = generate_all_candidates(obs, my_id, ip, av, current_step, arrival_table)
    if not candidates:
        return [], np.zeros(0)
    mlp = _load_mlp()
    feats = np.stack([c['features'] for c in candidates])
    scores = mlp.forward(feats)
    return candidates, scores
