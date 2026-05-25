"""wo_dataset — supervised (state, policy, value) dataset from Orbit Wars
episode replays.

Foundation of the "V15++" plan. Every replay step gives, per player p: the
board state, p's actual move, and p's final game result. We convert that
into network training tensors:

  PF   [S, N_MAX, P_DIM]    per-planet features (player-relative)
  GF   [S, G_DIM]           global features
  POL  [S, N_MAX, N_MAX+1]  policy target: the move p made, one-hot per
                            owned planet over {pass} U {target planet}
  MASK [S, N_MAX]           valid-planet mask
  VAL  [S]                  value target: p's final result (+1 win / -1 loss)

Supervised on real games -> sample-efficient, GPU-light, and collapse-proof:
the targets are real outcomes/moves, not self-generated self-play visits, so
the passive-policy collapse that killed V17 cannot occur here.

Run:
    python -u wo_dataset.py --glob "V15LogsReplays/episode-*.json" \
        --out analysis/wo_replay_data.npz
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import glob as _glob
import json
import time

import numpy as np

import v15_eval
import v15_fast_sim as fsim
import v17_encode as enc

N_MAX = 48


def _pad(pf, pol):
    """Pad one sample to N_MAX planets. pf [n,P_DIM], pol [n,n+1]."""
    n = pf.shape[0]
    k = min(n, N_MAX)
    pfp = np.zeros((N_MAX, enc.P_DIM), dtype=np.float32)
    polp = np.zeros((N_MAX, N_MAX + 1), dtype=np.float32)
    mask = np.zeros(N_MAX, dtype=bool)
    pfp[:k] = pf[:k]
    mask[:k] = True
    polp[:k, 0] = pol[:k, 0]
    tk = min(pol.shape[1] - 1, N_MAX)
    polp[:k, 1:1 + tk] = pol[:k, 1:1 + tk]
    return pfp, polp, mask


def _policy_target(fs, player, targets):
    """targets[i] (from enc.action_to_targets) -> one-hot policy rows [n,n+1].
    Owned planet that passed -> column 0; that launched at j -> column j+1.
    Non-owned planets stay all-zero (the mask excludes them from the loss)."""
    n = len(fs.planets)
    pol = np.zeros((n, n + 1), dtype=np.float32)
    for i in range(n):
        if int(fs.planets[i, enc.OWNER]) != player:
            continue
        j = int(targets[i])
        pol[i, (j + 1) if 0 <= j < n else 0] = 1.0
    return pol


def _board_obs(step_records):
    """Pick a usable observation for a step: the first record that carries
    planets. Orbit Wars is perfect-information, so the board (absolute planet
    and fleet data) is identical across every player's observation."""
    for rec in step_records:
        obs = rec.get("observation") or {}
        if obs.get("planets"):
            return obs
    return None


def _episode_values(ep, npl):
    """Per-player value target: final reward (+1 win / -1 loss). Falls back
    to the final-state score (score = total ships) if rewards are absent."""
    rewards = ep.get("rewards") or []
    if len(rewards) >= npl and all(r is not None for r in rewards[:npl]):
        return [float(rewards[p]) for p in range(npl)]
    obs = _board_obs(ep["steps"][-1]) or {}
    sc = [0] * npl
    for pl in obs.get("planets", []):
        o = int(pl[1])
        if 0 <= o < npl:
            sc[o] += int(pl[5])
    for fl in obs.get("fleets", []):
        o = int(fl[1])
        if 0 <= o < npl:
            sc[o] += int(fl[6])
    best = max(sc) if sc else 0
    winners = [i for i, s in enumerate(sc) if s == best and best > 0]
    return [1.0 if (len(winners) == 1 and winners[0] == p) else -1.0
            for p in range(npl)]


def samples_from_episode(ep):
    """Yield (pf, gf, pol, mask, val) for every (step, player) of a replay."""
    steps = ep.get("steps") or []
    if not steps:
        return
    npl = len(steps[0])
    episode_steps = int((ep.get("configuration") or {}).get(
        "episodeSteps", 500))
    val = _episode_values(ep, npl)
    for t, st in enumerate(steps):
        obs = _board_obs(st)
        if obs is None:
            continue
        fs = fsim.from_obs(obs, n_players=npl, episode_steps=episode_steps)
        fs.n_players = npl
        fs.step = t                      # replay step index is authoritative
        if len(fs.planets) == 0:
            continue
        for p in range(npl):
            rec = st[p] if p < len(st) else {}
            launches = rec.get("action") or []
            pf, gf = enc.encode(fs, p)
            targets = enc.action_to_targets(fs, p, launches)
            pol = _policy_target(fs, p, targets)
            pfp, polp, mask = _pad(pf, pol)
            esc = v15_eval.evaluate(fs, p, v15_eval.ESC)
            yield pfp, gf.astype(np.float32), polp, mask, float(val[p]), float(esc)


def _episode_id(ep, fallback):
    """Stable unique id for a replay — used to drop duplicate files and to
    split train/val by whole episode (consecutive states of one game are
    near-duplicates, so a per-sample split would leak)."""
    info = ep.get("info") or {}
    return info.get("EpisodeId") or ep.get("id") or fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="V15LogsReplays/episode-*.json",
                    help="comma-separated globs of replay episode JSON files")
    ap.add_argument("--out", default="analysis/wo_replay_data.npz")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    patterns = [p.strip() for p in args.glob.split(",") if p.strip()]
    files = sorted({f for p in patterns for f in _glob.glob(p)})
    if not files:
        print(f"[wo_dataset] no files match {args.glob}")
        return

    t0 = time.time()
    PF, GF, POL, MASK, VAL, ESC_ARR, EP = [], [], [], [], [], [], []
    seen = {}                                    # episode id -> source file
    for f in files:
        base = os.path.basename(f)
        try:
            ep = json.load(open(f, encoding="utf-8"))
        except Exception as e:                   # corrupt / partial file
            print(f"[wo_dataset] skip {base}: {e}")
            continue
        eid = _episode_id(ep, base)
        if eid in seen:                          # exact-duplicate replay file
            print(f"[wo_dataset] dup {base} (== {seen[eid]}) skipped")
            continue
        ep_idx = len(seen)
        seen[eid] = base
        cnt = 0
        for s in samples_from_episode(ep):
            PF.append(s[0]); GF.append(s[1]); POL.append(s[2])
            MASK.append(s[3]); VAL.append(s[4]); ESC_ARR.append(s[5])
            EP.append(ep_idx)
            cnt += 1
        print(f"[wo_dataset] {base}: {cnt} samples")

    if not PF:
        print("[wo_dataset] no samples extracted")
        return

    PF = np.stack(PF)
    GF = np.stack(GF)
    POL = np.stack(POL)
    MASK = np.stack(MASK)
    VAL = np.array(VAL, dtype=np.float32)
    ESC_ARR = np.array(ESC_ARR, dtype=np.float32)
    EP = np.array(EP, dtype=np.int32)
    np.savez_compressed(args.out, PF=PF, GF=GF, POL=POL, MASK=MASK,
                        VAL=VAL, ESC=ESC_ARR, EP=EP)

    # sanity stats: launch-fraction is the policy-head class balance — if
    # launching is rare, supervised training needs class weighting.
    owned_rows = int((POL.sum(axis=2) > 0.5).sum())
    pass_frac = float(POL[:, :, 0].sum()) / max(owned_rows, 1)
    print(f"[wo_dataset] {len(seen)} episodes -> {len(PF)} samples "
          f"({time.time() - t0:.1f}s)")
    print(f"  PF{PF.shape} GF{GF.shape} POL{POL.shape} EP[{EP.max() + 1} eps]")
    print(f"  VAL mean={VAL.mean():+.3f}  (balanced ~ 0)")
    print(f"  owned-planet decisions={owned_rows}  "
          f"pass={pass_frac:.3f}  launch={1 - pass_frac:.3f}")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
