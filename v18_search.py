"""v18_search — combo search with a realistic opponent model + rollout-ESC.

V15 scores "my combo, then a PASSIVE rollout" — it assumes the opponent does
nothing, so it over-rates aggressive combos that a real opponent would punish.
v18 scores "my combo + the opponent's likely move, then a passive rollout":
the opponents play their deterministic nearest-target move (v15's _det_policy)
for the first step — a realistic, attacking opponent model — before the
quiescence rollout. v18 therefore prefers combos that hold up against a
counter, which V15's optimistic depth-1 cannot see.

No learned evaluator: the leaf is the fixed ESC after a quiescence rollout
(V15's own recipe). So no argmax exploit and no passive collapse.

Entry point gumbel_move(fs, player) — name kept for v18_bench compatibility.
"""

from __future__ import annotations

from itertools import combinations

import bot_v7
import v15_eval
import v15_fast_sim as fsim
import v15_search
from v15_search import _det_policy, _enumerate_shots, _valid_combo

ROLLOUT = 22          # quiescence rollout length before ESC
_TOP_SHOTS = 7        # atomic shots surviving the stage-1 prune
_OUR_MAX_COMBO = 3    # max simultaneous launches in one of our combos


def _rollout_esc(st, player: int) -> float:
    """Passive (no new launch) quiescence rollout, then ESC for `player`."""
    empty = [[] for _ in range(st.n_players)]
    for _ in range(ROLLOUT):
        if st.done:
            break
        st = fsim.step(st, empty)
    return float(v15_eval.evaluate(st, player, v15_eval.ESC))


def gumbel_move(fs, player: int, n_sims: int = 0, rng=None) -> list:
    """Best combo for `player` vs a det-policy opponent model. Deterministic."""
    try:
        n = fs.n_players
        obs = v15_search.state_to_obs(fs, player)
        try:
            v7_move = bot_v7.agent(obs, None)
        except Exception:
            v7_move = []
        if not isinstance(v7_move, list):
            v7_move = []
        our_shots = _enumerate_shots(fs, player, v7_move)
        if not our_shots:
            return []

        # opponents' likely move: deterministic nearest-target (realistic,
        # attacking opponent model — _det_policy returns moves for all seats).
        opp_moves = _det_policy(fs)

        def _score(combo):
            acts = list(opp_moves)
            acts[player] = combo
            return _rollout_esc(fsim.step(fs, acts), player)

        # stage 1: score each atomic shot alone, keep the top survivors
        scored = [(s, _score([s])) for s in our_shots]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        top = [s for s, _ in scored[:_TOP_SHOTS]]

        # do-nothing baseline + valid subsets of the survivors
        best, best_score = [], _score([])
        for r in range(1, _OUR_MAX_COMBO + 1):
            for c in combinations(top, r):
                combo = list(c)
                if not _valid_combo(combo):
                    continue
                v = _score(combo)
                if v > best_score:
                    best_score, best = v, combo
        return best
    except Exception:
        try:
            return v15_search.search(
                v15_search.state_to_obs(fs, player), None)
        except Exception:
            return []


if __name__ == "__main__":
    import time
    import v14_core
    from local_simulator.official_fast import OfficialFastGame
    for npl in (2, 4):
        g = OfficialFastGame(npl, seed=4, episode_steps=250, use_c_accel=False)
        for _ in range(45):
            g.step([[] for _ in range(npl)])
        fs = fsim.from_obs(v14_core.obs_as_dict(g.observation(0)),
                           n_players=npl, episode_steps=250)
        fs.n_players = npl
        t = time.monotonic()
        mv = gumbel_move(fs, 0)
        dt = (time.monotonic() - t) * 1000
        assert isinstance(mv, list)
        print(f"{npl}p: move={len(mv)} launches  {dt:.0f}ms")
    print("v18_search self-check passed")
