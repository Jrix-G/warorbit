"""v15_search — V15.3 RCC: Recherche par Combinaisons Coordonnées.

Scientific basis (from 200 top-10 replays): strong bots do not pick the best
*individual* move — they search the space of *simultaneous coordinated
strikes*. 40.9% of their active turns hit >=2 targets at once; they stay
passive 70.9% of turns, waiting for a profitable combination window.

V7 scores each mission independently and executes them greedily — it never
asks "which COMBINATION of strikes is best". RCC does exactly that:

  A. Atomic shots   = V7's own launches (good ship-sizing) + enumerated
                      nearest-target shots (diversity V7 may miss).
  B. Baseline       = the do-nothing move, evaluated explicitly.
  C. Stage-1 prune  = score each atomic shot alone, keep the top-K.
  D. Stage-2 search = score every subset (<=MAX_COMBO, one shot per source
                      planet) of the survivors.
  E. Pick           = best combo if it beats the do-nothing baseline.

Each candidate is scored DETERMINISTICALLY: apply it, then run a fixed
deterministic continuation policy for both sides over a short horizon, then
evaluate with the Composite Static Evaluator (v15_eval). Determinism removes
the Monte-Carlo variance that made flat-MC weaker than V7.

Falls back to V7 on error / time pressure.
"""

from __future__ import annotations

import math
import time
from itertools import combinations

import numpy as np

import bot_v7
import v15_bc
import v15_eval
import v15_fast_sim as fsim

ID, OWNER, X, Y, R, SHIPS, PROD = range(7)

# --- RCC tuning -------------------------------------------------------------
_K_ATOMIC = 12          # cap on the number of atomic shots considered
_TOP_K = 6              # atomic shots surviving stage-1 prune
_MAX_COMBO = 4          # max simultaneous launches in one combination
_NEAR_TARGETS = 3       # enumerated targets per owned planet
_DET_SEND_FRAC = 0.6    # fraction sent by the deterministic continuation
_DET_MARGIN = 12.0      # min garrison before the continuation launches


def _infer_n_players(planets: np.ndarray) -> int:
    if len(planets) == 0:
        return 2
    return 4 if planets[:, OWNER].max() >= 2 else 2


def state_to_obs(fs: fsim.FastState, player: int) -> dict:
    """Convert a FastState back into an obs dict (the format agents consume).
    Used by the self-play loop to feed FastState positions to RCC / V7."""
    planets = [[int(p[ID]), int(p[OWNER]), float(p[X]), float(p[Y]),
                float(p[R]), float(p[SHIPS]), float(p[PROD])]
               for p in fs.planets]
    init = [[int(fs.planets[i][ID]), int(fs.planets[i][OWNER]),
             float(fs.p_init[i][0]), float(fs.p_init[i][1]),
             float(fs.planets[i][R]), float(fs.planets[i][SHIPS]),
             float(fs.planets[i][PROD])]
            for i in range(len(fs.planets))]
    fleets = [[int(f[0]), int(f[1]), float(f[2]), float(f[3]),
               float(f[4]), int(f[5]), float(f[6])] for f in fs.fleets]
    comet_ids = [pid for g in fs.comets for pid in g["planet_ids"]]
    return {
        "player": player, "step": fs.step,
        "angular_velocity": fs.angular_velocity,
        "planets": planets, "initial_planets": init, "fleets": fleets,
        "next_fleet_id": fs.next_fleet_id, "comets": fs.comets,
        "comet_planet_ids": comet_ids, "remainingOverageTime": 60.0,
    }


# ---------------------------------------------------------------------------
# Deterministic continuation policy
# ---------------------------------------------------------------------------

def _det_policy(fs: fsim.FastState) -> list[list]:
    """Deterministic continuation: every planet with a healthy garrison sends
    a fixed fraction toward its single NEAREST non-owned planet. No RNG — the
    same state always yields the same actions, so combo comparisons carry
    zero variance."""
    planets = fs.planets
    N = len(planets)
    actions: list[list] = [[] for _ in range(fs.n_players)]
    if N == 0:
        return actions
    px = planets[:, X]
    py = planets[:, Y]
    owners = planets[:, OWNER].astype(np.int64)
    ships = planets[:, SHIPS]
    ids = planets[:, ID].astype(np.int64)
    dxm = px[:, None] - px[None, :]
    dym = py[:, None] - py[None, :]
    dist = np.sqrt(dxm * dxm + dym * dym)
    for i in range(N):
        p = int(owners[i])
        if p < 0 or p >= fs.n_players or ships[i] <= _DET_MARGIN:
            continue
        foreign = np.where(owners != p)[0]
        if len(foreign) == 0:
            continue
        tgt = int(foreign[np.argmin(dist[i, foreign])])
        ang = math.atan2(py[tgt] - py[i], px[tgt] - px[i])
        send = int(ships[i] * _DET_SEND_FRAC)
        if send > 0:
            actions[p].append([int(ids[i]), float(ang), send])
    return actions


# ---------------------------------------------------------------------------
# Atomic shot enumeration
# ---------------------------------------------------------------------------

def _needed_ships(src_row, tgt_row, n_players: int) -> int:
    """Approximate ships required for `src` to capture `tgt` on arrival:
    current defenders + production accrued during travel + a safety margin."""
    dist = math.hypot(tgt_row[X] - src_row[X], tgt_row[Y] - src_row[Y])
    eta = max(1.0, dist / 4.0)            # ~mid-speed ETA estimate
    defenders = tgt_row[SHIPS] + tgt_row[PROD] * eta
    enemy = tgt_row[OWNER] >= 0
    margin = 5.0 if enemy else 3.0
    return int(defenders + margin) + 1


def _enumerate_shots(fs: fsim.FastState, player: int,
                     v7_move: list) -> list[list]:
    """Atomic shots: V7's own launches (good ship-sizing) plus, for each owned
    planet, capture-sized shots toward its nearest non-owned planets."""
    planets = fs.planets
    N = len(planets)
    shots: list[list] = []
    seen: set = set()

    def _add(src_id, angle, ships):
        ships = int(ships)
        if ships <= 0:
            return
        key = (int(src_id), round(float(angle), 2))
        if key in seen:
            return
        seen.add(key)
        shots.append([int(src_id), float(angle), ships])

    # V7's launches first — they carry V7's tuned ship sizing & intercept aim.
    for mv in (v7_move or []):
        if isinstance(mv, list) and len(mv) == 3:
            _add(mv[0], mv[1], mv[2])

    if N == 0:
        return shots[:_K_ATOMIC]

    owners = planets[:, OWNER].astype(np.int64)
    px = planets[:, X]
    py = planets[:, Y]
    mine = np.where(owners == player)[0]
    foreign = np.where(owners != player)[0]
    if len(foreign) == 0:
        return shots[:_K_ATOMIC]

    for i in mine:
        garrison = planets[i, SHIPS]
        if garrison < 2:
            continue
        d = np.hypot(px[foreign] - px[i], py[foreign] - py[i])
        order = foreign[np.argsort(d)][:_NEAR_TARGETS]
        for j in order:
            need = _needed_ships(planets[i], planets[j], fs.n_players)
            send = min(int(garrison), need)
            # skip hopeless attacks: too weak to matter alone
            if send < need * 0.5:
                continue
            ang = math.atan2(py[j] - py[i], px[j] - px[i])
            _add(planets[i, ID], ang, send)

    return shots[:_K_ATOMIC]


# ---------------------------------------------------------------------------
# Deterministic combo evaluation
# ---------------------------------------------------------------------------

def _eval_combo(fs: fsim.FastState, player: int, combo: list,
                horizon: int, bc_cont: bool = False,
                weights: "v15_eval.EvalWeights" = None,
                value_fn=None, value_lambda: float = 0.0) -> float:
    """Apply `combo` at step 1, then run a continuation for `horizon` steps so
    the combo's fleets travel, arrive and resolve combat; return the ESC score
    of the leaf.

    Two continuation modes, both deterministic (zero variance):
      passive (bc_cont=False) — no new launches; pure quiescence. Lets pending
        tactics play out, then evaluates a quiet position. A churning
        continuation washes out the combo signal (catastrophic in 4p).
      BC      (bc_cont=True)  — both sides continue with the behavioral-cloning
        policy (top-10 imitation). Models opponent reinforcement, so combos
        that only beat a do-nothing opponent are correctly penalised."""
    n = fs.n_players
    if bc_cont:
        actions = v15_bc.bc_policy(fs)
    else:
        actions = [[] for _ in range(n)]
    actions[player] = list(combo)
    st = fsim.step(fs, actions)
    for _ in range(horizon - 1):
        if st.done:
            break
        cont = v15_bc.bc_policy(st) if bc_cont else [[] for _ in range(n)]
        st = fsim.step(st, cont)
    esc = v15_eval.evaluate(st, player,
                            weights if weights is not None else v15_eval.ESC)
    if value_fn is None or value_lambda <= 0.0:
        return esc
    # V15++ graft: learned value [-1,1] -> [0,1], the same scale as the ESC
    # score; the blend only refines V15's ranking — lambda<1 keeps V15's eval
    # in the mix, so a wrong net can nudge but never replace it.
    net01 = 0.5 * (float(value_fn(st, player)) + 1.0)
    return (1.0 - value_lambda) * esc + value_lambda * net01


def _valid_combo(combo: list) -> bool:
    """One launch per source planet (keeps each shot within its garrison)."""
    srcs = [int(s[0]) for s in combo]
    return len(srcs) == len(set(srcs))


def search(obs, config=None, *, time_budget: float = 0.7,
           horizon: int = 24, bc_cont: bool = False,
           weights: "v15_eval.EvalWeights" = None,
           n_policy_samples: int = 0, seed: int = 0,
           use_value_fn: bool = False,
           value_fn=None, value_lambda: float = 0.0) -> list:
    """V15.3 RCC — coordinated-combination search.

    bc_cont — if True, combos are scored with a behavioral-cloning continuation
    (models opponent counter-play); if False, a passive quiescence continuation.
    weights — evaluator weight set (default ESC); a learned value function is
    supplied here by the self-play value-iteration loop.

    `n_policy_samples` / `seed` / `use_value_fn` are accepted for call-site
    compatibility but unused (RCC is deterministic)."""
    t0 = time.monotonic()
    deadline = t0 + time_budget
    try:
        if isinstance(obs, dict):
            our = int(obs.get("player", 0) or 0)
        else:
            our = int(getattr(obs, "player", 0) or 0)

        v7_move = bot_v7.agent(obs, config)
        if not isinstance(v7_move, list):
            v7_move = []

        fs = fsim.from_obs(obs, n_players=2)
        fs.n_players = _infer_n_players(fs.planets)

        atomic = _enumerate_shots(fs, our, v7_move)
        if not atomic:
            return v7_move

        h1 = max(4, horizon // 2)

        # --- baseline: the do-nothing move ---
        baseline = _eval_combo(fs, our, [], horizon, bc_cont, weights,
                               value_fn, value_lambda)

        # --- stage 1: score each atomic shot in isolation ---
        scored = []
        for shot in atomic:
            if time.monotonic() > deadline:
                break
            scored.append(
                (shot, _eval_combo(fs, our, [shot], h1, bc_cont, weights,
                                   value_fn, value_lambda)))
        if not scored:
            return v7_move
        scored.sort(key=lambda kv: kv[1], reverse=True)
        top = [shot for shot, _ in scored[:_TOP_K]]

        # --- stage 2: score every valid subset of the survivors ---
        best_combo: list = []
        best_score = baseline
        # singletons first (cheap, always informative), then larger combos
        subsets: list[list] = []
        for r in range(1, min(_MAX_COMBO, len(top)) + 1):
            for c in combinations(top, r):
                combo = list(c)
                if _valid_combo(combo):
                    subsets.append(combo)
        # evaluate larger combos first only if budget allows; keep order stable
        for combo in subsets:
            if time.monotonic() > deadline:
                break
            sc = _eval_combo(fs, our, combo, horizon, bc_cont, weights,
                             value_fn, value_lambda)
            if sc > best_score:
                best_score = sc
                best_combo = combo

        # best_combo stays [] (do nothing) if no combination beat the baseline
        return best_combo
    except Exception:
        try:
            return bot_v7.agent(obs, config)
        except Exception:
            return []
