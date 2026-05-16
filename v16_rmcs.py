"""v16_rmcs — Recursive Macro-Action Combination Search (CPU / numpy).

The V15 search is depth-1: it evaluates this turn's combos against a passive
future. RMCS instead searches a SEQUENCE of macro-decisions.

A decision = a combo. Between decisions every player follows a fast base
policy for `stride` turns. The search recurses over `depth` decision points,
so depth=3, stride=15 gives 45 turns of foresight with 3 genuine re-decision
points — versus 1 turn today. Only OUR decisions branch (branch factor B);
opponents follow the base policy (accurate — ladder bots are shallow), so the
tree is B^depth, never exploding.

Receding horizon: only the first decision's combo is returned; the search
re-runs next turn on fresh state, bounding any base-policy error to one
committed move.

CPU / numpy on the validated v15_fast_sim engine — directly deployable.
"""

from __future__ import annotations

import bot_v7
import v15_eval
import v15_fast_sim as fsim
import v15_search as rcc

ID, OWNER, X, Y, R, SHIPS, PROD = range(7)

DEPTH = 3            # number of macro-decision points
STRIDE = 15          # turns between decisions (base policy fills them)
BRANCH = 8           # candidate combos per decision (incl. do-nothing)


def _candidates(fs: fsim.FastState, player: int, branch: int) -> list[list]:
    """Up to `branch` candidate combos for `player`: do-nothing, then the
    strongest enumerated atomic shots (intercept-aimed). Combination
    coordination across TIME is provided by the recursion's depth."""
    atomic = rcc._enumerate_shots(fs, player, [])    # no V7 inside the search
    combos: list[list] = [[]]                        # do-nothing is a candidate
    for shot in atomic:
        combos.append([shot])
        if len(combos) >= branch:
            break
    return combos


def _search(fs: fsim.FastState, player: int, depth: int, stride: int,
            branch: int, weights) -> tuple[float, list | None]:
    """Return (best_value, best_combo) — best_combo is the decision at THIS
    node. Opponents (and we, between decisions) follow the base policy."""
    if fs.done or depth == 0:
        return v15_eval.evaluate(fs, player, weights), None

    best_v, best_combo = -1e18, None
    for combo in _candidates(fs, player, branch):
        # decision turn: our combo overrides the base policy
        actions = rcc._det_policy(fs)
        actions[player] = list(combo)
        st = fsim.step(fs, actions)
        # stride: base policy for everyone until the next decision point
        for _ in range(stride - 1):
            if st.done:
                break
            st = fsim.step(st, rcc._det_policy(st))
        v, _ = _search(st, player, depth - 1, stride, branch, weights)
        if v > best_v:
            best_v, best_combo = v, combo
    return best_v, best_combo


def search(obs, config=None, *, depth: int = DEPTH, stride: int = STRIDE,
           branch: int = BRANCH, weights=None) -> list:
    """RMCS agent entry point. Returns the first decision's combo."""
    try:
        if isinstance(obs, dict):
            player = int(obs.get("player", 0) or 0)
        else:
            player = int(getattr(obs, "player", 0) or 0)
        fs = fsim.from_obs(obs, n_players=2)
        fs.n_players = rcc._infer_n_players(fs.planets)
        w = weights if weights is not None else v15_eval.ESC
        _, combo = _search(fs, player, depth, stride, branch, w)
        return combo if combo else []
    except Exception:
        try:
            return bot_v7.agent(obs, config)
        except Exception:
            return []
