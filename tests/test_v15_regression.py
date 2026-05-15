"""M0 — V15(passthrough) must be indistinguishable from V7.

V7 itself is non-deterministic across runs (hidden state in c_engine / module
caches that random.seed + numpy.random.seed do not reset). So strict bit-identity
is unreachable. The honest contract is:

  V15(passthrough) ≡ V7  modulo V7's own self-divergence.

Procedure per seed:
  1. Play V7 once  → ref stream A.
  2. Play V7 again → control stream B. Note V7-vs-V7 divergence (noise floor).
  3. Play V15(passthrough) → treatment stream C.
  4. Assert: V15-vs-V7 divergence ≤ V7-vs-V7 divergence (signature-equal).

If V15's divergence point matches V7's self-divergence point, the wrapper is identity.
If V15 diverges *earlier* than V7's self-noise floor, the wrapper introduced a bug.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u tests/test_v15_regression.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot_v7
import bot_v12
import bot_v15
import v15_config
import v14_core
from local_simulator.official_fast import OfficialFastGame
from opponents import ZOO


def _call(fn, obs, config):
    obs = v14_core.obs_as_dict(obs)
    try:
        move = fn(obs, config)
    except TypeError:
        move = fn(obs)
    return move if isinstance(move, list) else []


def _play(p0_fn, opp_fns, n_players: int, seed: int, max_steps: int = 220):
    random.seed(seed)
    np.random.seed(seed)
    game = OfficialFastGame(
        n_players=n_players,
        seed=seed,
        episode_steps=max_steps,
        use_c_accel=True,
    )
    agents = [p0_fn, *opp_fns]
    stream = []
    while not game.done:
        actions = [
            _call(fn, game.observation(p), game.configuration)
            for p, fn in enumerate(agents)
        ]
        stream.append(actions[0])
        game.step(actions)
    return stream


def _first_diff(a, b) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1  # identical


def main() -> int:
    bot_v15.set_config(v15_config.V15Config())
    assert bot_v15._get_config().is_passthrough(), "V15 default config must be passthrough"

    opp_2p = [ZOO["notebook_distance_prioritized"]]
    opp_4p = [ZOO["notebook_distance_prioritized"], bot_v12.agent, ZOO["notebook_orbitbotnext"]]

    seeds = [101, 202, 303, 404, 505]
    configs = [
        ("2p", 2, opp_2p),
        ("4p", 4, opp_4p),
    ]

    pass_count = 0
    fail_count = 0
    for mode, n_players, opp_fns in configs:
        for seed in seeds:
            ref = _play(bot_v7.agent, opp_fns, n_players, seed)
            ctrl = _play(bot_v7.agent, opp_fns, n_players, seed)
            treat = _play(bot_v15.agent, opp_fns, n_players, seed)
            d_ctrl = _first_diff(ref, ctrl)
            d_treat = _first_diff(ref, treat)

            # M0 contract: V15 must match V7 at least as long as V7 matches itself.
            # i.e. V15 introduces no *additional* divergence vs the V7 self-noise floor.
            ok = (d_treat == -1) or (d_ctrl != -1 and d_treat >= d_ctrl)
            tag = "OK  " if ok else "FAIL"
            print(f"[{tag}] mode={mode} seed={seed} steps={len(ref)} "
                  f"V7-v-V7 diverge@={d_ctrl} V7-v-V15 diverge@={d_treat}")
            if ok:
                pass_count += 1
            else:
                fail_count += 1

    total = pass_count + fail_count
    print(f"\nSummary: {pass_count}/{total} games meet M0 contract.")
    if fail_count:
        print("M0 FAILED — V15 wrapper introduces divergence beyond V7's self-noise floor.")
        return 1
    print("M0 PASSED — V15(passthrough) is indistinguishable from V7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
