"""wo_esc_opt — fit optimised ESC linear weights by ridge regression.

For every (step, player) in replay episodes:
  1. Reconstruct the FastState from the JSON.
  2. Run a 24-step passive continuation (same as _eval_combo) -> leaf state.
  3. Compute the 11 v15_eval features of the leaf state.
  4. Label with the player's episode outcome (+1 win / -1 loss).

Then fit separate ridge regressions for 2p and 4p (same feature space) with
standardisation, and save the result as an EvalWeights file that v15_search
can load directly via the `weights` parameter.

The intuition: ESC is a linear function of features — a better linear fit
cannot amplify local noise the way a neural net can, so the argmax-exploit
problem does not apply. Even small global-correlation improvements translate
directly into better leaf rankings.

Run:
    python -u wo_esc_opt.py --out analysis/esc_optimized.npz
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import glob as _glob
import json
import math

import numpy as np

import v15_eval
import v15_fast_sim as fsim
from wo_dataset import _board_obs, _episode_id, _episode_values

GLOBS = [
    "replays/top1-05-05/episode-*.json",
    "V15LogsReplays/episode-*.json",
    "logs/logsfromVPS/episode-*.json",
]
HORIZON = 24


def _collect(episode_globs, horizon):
    """Return X2p, y2p, X4p, y4p arrays of (features, outcome) pairs."""
    files = sorted({f for g in episode_globs for f in _glob.glob(g)})
    seen = set()
    rows2p, rows4p = [], []

    for f in files:
        try:
            ep = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        eid = _episode_id(ep, f)
        if eid in seen:
            continue
        seen.add(eid)

        steps = ep.get("steps") or []
        if not steps:
            continue
        npl = len(steps[0])
        es = int((ep.get("configuration") or {}).get("episodeSteps", 500))
        val = _episode_values(ep, npl)

        for t, st in enumerate(steps):
            obs = _board_obs(st)
            if obs is None:
                continue
            fs = fsim.from_obs(obs, n_players=npl, episode_steps=es)
            fs.n_players = npl
            fs.step = t
            if len(fs.planets) == 0:
                continue

            # passive continuation -> leaf
            leaf = fs
            for _ in range(horizon):
                if leaf.done:
                    break
                leaf = fsim.step(leaf, [[] for _ in range(npl)])

            bucket = rows2p if npl == 2 else rows4p
            for p in range(npl):
                feat = v15_eval.features(leaf, p)
                bucket.append((feat, float(val[p])))

    def _to_arrays(rows):
        if not rows:
            return np.zeros((0, v15_eval.N_FEATURES)), np.zeros(0)
        X = np.stack([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        return X, y

    return _to_arrays(rows2p), _to_arrays(rows4p)


def _ridge(X, y, alpha=1.0):
    """Fit standardised ridge regression: predict y from X @ w + b.
    Returns w (N_FEATURES,), mean (N_FEATURES,), std (N_FEATURES,)."""
    if len(X) == 0:
        w = np.ones(v15_eval.N_FEATURES) / v15_eval.N_FEATURES
        m = np.zeros(v15_eval.N_FEATURES)
        s = np.ones(v15_eval.N_FEATURES)
        return w, m, s

    m = X.mean(axis=0)
    s = X.std(axis=0)
    s[s < 1e-9] = 1.0
    Xs = (X - m) / s

    # Normal equations with L2: w = (Xs^T Xs + alpha*I)^-1 Xs^T y
    A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    b = Xs.T @ y
    w = np.linalg.solve(A, b)

    pred = Xs @ w
    corr = float(np.corrcoef(pred, y)[0, 1]) if y.std() > 1e-9 else 0.0
    mse = float(((pred - y) ** 2).mean())
    print(f"  ridge fit: n={len(X)}  corr={corr:.4f}  mse={mse:.4f}  "
          f"|w|={np.abs(w).sum():.3f}")
    return w, m, s


def _play_v15_game(task):
    """Play one V15 vs V15 game; return list of (features_leaf, outcome, npl)."""
    import v14_core, v15_search
    from local_simulator.official_fast import OfficialFastGame
    n_players, seed, budget, horizon = task
    episode_steps = 250
    g = OfficialFastGame(n_players, seed=seed, episode_steps=episode_steps,
                         use_c_accel=False)
    fs = fsim.from_obs(v14_core.obs_as_dict(g.observation(0)),
                       n_players=n_players, episode_steps=episode_steps)
    fs.n_players = n_players

    history = []
    while not fs.done:
        actions = []
        for p in range(n_players):
            o = v15_search.state_to_obs(fs, p)
            m = v15_search.search(o, None, time_budget=budget)
            actions.append(m if isinstance(m, list) else [])
        history.append(fs)
        fs = fsim.step(fs, actions)

    sc = fsim.scores(fs)
    best = max(sc)
    win_set = [i for i, s in enumerate(sc) if s == best and best > 0]
    vals = [1.0 if (len(win_set) == 1 and p in win_set) else -1.0
            for p in range(n_players)]

    rows = []
    for state in history:
        if len(state.planets) == 0:
            continue
        leaf = state
        for _ in range(horizon):
            if leaf.done:
                break
            leaf = fsim.step(leaf, [[] for _ in range(n_players)])
        for p in range(n_players):
            feat = v15_eval.features(leaf, p)
            rows.append((feat, vals[p], n_players))
    return rows


def _collect_v15(n_games, modes, budget, horizon, workers, seed_offset):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    tasks = [(npl, seed_offset + i, budget, horizon)
             for npl in modes for i in range(n_games)]
    rows2p, rows4p = [], []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_play_v15_game, t): t for t in tasks}
        for fut in as_completed(futs):
            for feat, val, npl in fut.result():
                (rows2p if npl == 2 else rows4p).append((feat, val))
            done += 1
            print(f"  game {done}/{len(tasks)}", flush=True)

    def _arr(rows):
        if not rows:
            return np.zeros((0, v15_eval.N_FEATURES)), np.zeros(0)
        return np.stack([r[0] for r in rows]), np.array([r[1] for r in rows])

    return _arr(rows2p), _arr(rows4p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/esc_optimized.npz")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="ridge regularisation strength (higher = more shrinkage)")
    ap.add_argument("--horizon", type=int, default=HORIZON,
                    help="passive continuation steps before feature extraction")
    ap.add_argument("--globs", default=",".join(GLOBS),
                    help="comma-separated glob patterns for episode JSON files")
    ap.add_argument("--v15-games", type=int, default=0,
                    help="generate this many V15 vs V15 games for fitting "
                         "(0 = use replay globs instead)")
    ap.add_argument("--modes", default="2,4")
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--seed-offset", type=int, default=20_000_000)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.v15_games > 0:
        modes = [int(x) for x in args.modes.split(",") if x.strip()]
        print(f"[esc_opt] generating {args.v15_games}x{modes} V15 games, "
              f"horizon={args.horizon}")
        (X2, y2), (X4, y4) = _collect_v15(
            args.v15_games, modes, args.budget, args.horizon,
            args.workers, args.seed_offset)
    else:
        globs = [g.strip() for g in args.globs.split(",") if g.strip()]
        print(f"[esc_opt] collecting from {len(globs)} globs, horizon={args.horizon}")
        (X2, y2), (X4, y4) = _collect(globs, args.horizon)
    print(f"[esc_opt] 2p: {len(X2)} samples   4p: {len(X4)} samples")

    print("[esc_opt] 2p ridge:")
    w2, m2, s2 = _ridge(X2, y2, args.alpha)
    print("[esc_opt] 4p ridge:")
    w4, m4, s4 = _ridge(X4, y4, args.alpha)

    # compare to ESC baseline correlation
    if len(X2):
        esc_pred2 = X2 @ v15_eval.ESC.w2p
        esc_corr2 = float(np.corrcoef(esc_pred2, y2)[0, 1])
        opt_pred2 = ((X2 - m2) / s2) @ w2
        opt_corr2 = float(np.corrcoef(opt_pred2, y2)[0, 1])
        print(f"[esc_opt] 2p: ESC corr={esc_corr2:.4f}  optimised corr={opt_corr2:.4f}  "
              f"delta={opt_corr2 - esc_corr2:+.4f}")
    if len(X4):
        esc_pred4 = X4 @ v15_eval.ESC.w4p
        esc_corr4 = float(np.corrcoef(esc_pred4, y4)[0, 1])
        opt_pred4 = ((X4 - m4) / s4) @ w4
        opt_corr4 = float(np.corrcoef(opt_pred4, y4)[0, 1])
        print(f"[esc_opt] 4p: ESC corr={esc_corr4:.4f}  optimised corr={opt_corr4:.4f}  "
              f"delta={opt_corr4 - esc_corr4:+.4f}")

    weights = v15_eval.EvalWeights(
        w2p=w2, w4p=w4,
        mean2p=m2, std2p=s2,
        mean4p=m4, std4p=s4,
        tag="esc_opt",
    )
    weights.save(args.out)
    print(f"[esc_opt] saved -> {args.out}")

    # print the fitted weights for inspection
    feat_names = [
        "ship_share", "prod_share", "planet_share", "domination", "prod_margin",
        "fleet_share", "elim_share", "top_planet_prod", "ship_conc",
        "step_frac", "enemy_fleet_press",
    ]
    print("\n[esc_opt] optimised 2p weights (standardised):")
    for name, wi in zip(feat_names, w2):
        print(f"  {name:20s}: {wi:+.4f}")
    print("\n[esc_opt] optimised 4p weights (standardised):")
    for name, wi in zip(feat_names, w4):
        print(f"  {name:20s}: {wi:+.4f}")


if __name__ == "__main__":
    main()
