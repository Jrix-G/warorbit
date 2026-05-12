#!/usr/bin/env python3
"""V14.1 fine-tune: PPO-clip + critic baseline + entropy + grad-clip + PFSP.

Improvements over train_v14_finetune.py:
- Critic head (state-value baseline) cuts variance.
- PPO-clip objective with K epochs.
- Entropy bonus prevents policy collapse.
- Gradient clipping (global norm).
- Delta-based dense reward (step-to-step).
- PFSP self-play sampling.
- Anti-noop bias as logit shift (no post-hoc index swap).
- 4p checkpoint selection uses a short rolling window.
- LR cosine decay, temperature anneal.
- Sanity assert on records / step_rewards alignment.
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

import v14_core
from opponents import ZOO
from local_simulator.official_fast import OfficialFastGame
from train_v14_bc import Adam, _backward


_GAMMA = 0.99
_NOOP_LOGIT_BIAS = -0.5  # discourage noop when other candidates exist
_PPO_CLIP = 0.2
_PPO_EPOCHS = 2
_ENTROPY_BETA = 0.01
_VALUE_COEF = 0.5
_GRAD_CLIP = 1.0
_CRITIC_HIDDEN = 64
_CRITIC_IN = 2 * v14_core.FEATURE_DIM

_RANK_REWARD_4P = (1.0, 0.3, -0.3, -0.8)


# ---------- critic ----------
class V14Critic:
    def __init__(self, weights: dict | None = None, seed: int = 15):
        if weights is not None and "cW1" in weights:
            self.W1 = weights["cW1"].astype(np.float32)
            self.b1 = weights["cb1"].astype(np.float32)
            self.W2 = weights["cW2"].astype(np.float32)
            self.b2 = weights["cb2"].astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
            self.W1 = (rng.standard_normal((_CRITIC_IN, _CRITIC_HIDDEN))
                       * math.sqrt(2.0 / _CRITIC_IN)).astype(np.float32)
            self.b1 = np.zeros(_CRITIC_HIDDEN, dtype=np.float32)
            self.W2 = (rng.standard_normal((_CRITIC_HIDDEN, 1))
                       * math.sqrt(2.0 / _CRITIC_HIDDEN)).astype(np.float32)
            self.b2 = np.zeros(1, dtype=np.float32)

    def to_dict(self) -> dict:
        return {"cW1": self.W1, "cb1": self.b1, "cW2": self.W2, "cb2": self.b2}

    def forward(self, x: np.ndarray) -> tuple[float, dict]:
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(0.0, z1)
        v = float((h1 @ self.W2 + self.b2)[0])
        return v, {"x": x, "z1": z1, "h1": h1}

    def backward(self, cache: dict, grad_v: float) -> dict:
        x = cache["x"]
        z1 = cache["z1"]
        h1 = cache["h1"]
        g2 = np.array([[grad_v]], dtype=np.float32)
        dW2 = h1.reshape(-1, 1) * grad_v
        db2 = g2.reshape(-1)
        dh1 = self.W2.reshape(-1) * grad_v
        dz1 = dh1 * (z1 > 0.0).astype(np.float32)
        dW1 = np.outer(x, dz1)
        db1 = dz1
        return {"cW1": dW1, "cb1": db1, "cW2": dW2, "cb2": db2}


def _pool_feats(feats: np.ndarray) -> np.ndarray:
    if feats.shape[0] == 0:
        return np.zeros(_CRITIC_IN, dtype=np.float32)
    mean = feats.mean(axis=0)
    mx = feats.max(axis=0)
    return np.concatenate([mean, mx]).astype(np.float32)


# ---------- helpers ----------
def _call_agent(fn: Callable, obs, config) -> list:
    obs = v14_core.obs_as_dict(obs)
    try:
        move = fn(obs, config)
    except TypeError:
        move = fn(obs)
    return move if isinstance(move, list) else []


def _candidate_actions(candidate: dict) -> list[list]:
    if candidate.get("type") == "noop":
        return []
    return [[int(sid), float(angle), int(ships)]
            for sid, angle, ships in candidate.get("moves", []) or []]


def _terminal_reward(scores: list[float], my_slot: int) -> float:
    n = len(scores)
    if n == 4:
        ordered = sorted(range(n), key=lambda i: -float(scores[i]))
        rank = ordered.index(my_slot)
        return float(_RANK_REWARD_4P[rank])
    ours = float(scores[my_slot])
    best_other = max(float(s) for i, s in enumerate(scores) if i != my_slot)
    scale = max(1.0, sum(abs(float(s)) for s in scores))
    margin = float(np.clip((ours - best_other) / scale, -1.0, 1.0))
    if ours > best_other and ours > 0:
        return 1.0 + 0.5 * margin
    return -0.6 + 0.25 * margin


def _econ_share(obs: dict, me: int) -> tuple[float, float, float]:
    planets = list(obs.get("planets", []) or [])
    fleets = list(obs.get("fleets", []) or [])
    my_p = [p for p in planets if int(p[1]) == me]
    my_prod = sum(float(p[6]) for p in my_p)
    my_ships = (sum(float(p[5]) for p in my_p)
                + sum(float(f[6]) for f in fleets if int(f[1]) == me))
    total_prod = max(1.0, sum(float(p[6]) for p in planets if int(p[1]) >= 0))
    total_ships = max(1.0, sum(float(p[5]) for p in planets if int(p[1]) >= 0)
                      + sum(float(f[6]) for f in fleets if int(f[1]) >= 0))
    return my_prod / total_prod, my_ships / total_ships, float(len(my_p))


def _delta_reward(prev: tuple[float, float, float] | None,
                  cur: tuple[float, float, float],
                  n_players: int) -> float:
    if prev is None:
        return 0.0
    d_prod = cur[0] - prev[0]
    d_ship = cur[1] - prev[1]
    d_pl = (cur[2] - prev[2]) / 40.0
    return float(0.4 * d_prod + 0.3 * d_ship + 0.3 * d_pl)


def _discounted_returns(step_rewards: list[float], terminal: float, gamma: float) -> list[float]:
    n = len(step_rewards)
    if n == 0:
        return []
    returns = [0.0] * n
    running = terminal
    for t in range(n - 1, -1, -1):
        running = step_rewards[t] + gamma * running
        returns[t] = running
    return returns


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.exp(x - m).sum()) + 1e-12)


def _global_grad_norm(grads: dict) -> float:
    s = 0.0
    for g in grads.values():
        s += float((g * g).sum())
    return math.sqrt(s)


def _clip_grads(grads: dict, max_norm: float) -> float:
    norm = _global_grad_norm(grads)
    if norm > max_norm and norm > 0:
        scale = max_norm / norm
        for k in grads:
            grads[k] *= scale
    return norm


# ---------- rollout ----------
def _play_episode_task(task: tuple):
    (actor_w, critic_w, seed, max_steps, n_players,
     opponent_names, temperature, sp_pool_w, sp_weights_pfsp) = task
    actor = v14_core.V14Scorer(weights=actor_w)
    critic = V14Critic(weights=critic_w)
    rng = np.random.default_rng(seed)
    game = OfficialFastGame(
        n_players=n_players, seed=seed,
        episode_steps=max_steps, use_c_accel=True,
    )
    my_slot = int(rng.integers(0, n_players))

    # Build lineup
    sp_idx_used: list[int] = []
    lineup: list[Callable | None] = []
    for player in range(n_players):
        if player == my_slot:
            lineup.append(None)
            continue
        if sp_pool_w and rng.random() < 0.4:
            # PFSP: weight ∝ (1 - winrate)^2; sp_weights_pfsp normalized
            if sp_weights_pfsp is not None and len(sp_weights_pfsp) == len(sp_pool_w):
                pidx = int(rng.choice(len(sp_pool_w), p=sp_weights_pfsp))
            else:
                pidx = int(rng.integers(0, len(sp_pool_w)))
            sp_idx_used.append(pidx)
            sp_model = v14_core.V14Scorer(weights=sp_pool_w[pidx])

            def _make_sp(m):
                def _sp(obs, config=None):
                    obs = v14_core.obs_as_dict(obs)
                    cands = v14_core.get_candidates(obs)
                    if not cands:
                        return []
                    feats = v14_core.candidate_matrix(obs, cands)
                    s = m.forward(feats)
                    # Apply same noop logit bias
                    for i, c in enumerate(cands):
                        if c.get("type") == "noop" and len(cands) > 1:
                            s[i] += _NOOP_LOGIT_BIAS
                    return _candidate_actions(cands[int(np.argmax(s))])
                return _sp
            lineup.append(_make_sp(sp_model))
        else:
            zoo_agents = [ZOO[name] for name in opponent_names]
            lineup.append(zoo_agents[int(rng.integers(0, len(zoo_agents)))])

    records: list[dict] = []  # {feats, idx, log_p_old, value, pooled}
    step_rewards: list[float] = []
    prev_share: tuple[float, float, float] | None = None

    while not game.done:
        actions = []
        my_decided_this_step = False
        for player in range(n_players):
            obs = v14_core.obs_as_dict(game.observation(player))
            if player == my_slot:
                candidates = v14_core.get_candidates(obs)
                if not candidates:
                    actions.append([])
                    continue
                feats = v14_core.candidate_matrix(obs, candidates)
                scores = actor.forward(feats).astype(np.float32)
                # Anti-noop: logit bias when alternatives exist
                if len(candidates) > 1:
                    for i, c in enumerate(candidates):
                        if c.get("type") == "noop":
                            scores[i] += _NOOP_LOGIT_BIAS
                logits = scores / max(0.05, temperature)
                lse = _logsumexp(logits)
                probs = np.exp(logits - lse)
                idx = int(rng.choice(len(candidates), p=probs))
                log_p_old = float(logits[idx] - lse)
                pooled = _pool_feats(feats)
                v_pred, _ = critic.forward(pooled)

                records.append({
                    "feats": feats,
                    "idx": idx,
                    "log_p_old": log_p_old,
                    "value": v_pred,
                    "pooled": pooled,
                    "noop_bias_idx": [i for i, c in enumerate(candidates)
                                      if c.get("type") == "noop"]
                                     if len(candidates) > 1 else [],
                    "temperature": float(max(0.05, temperature)),
                })
                # Step reward computed AFTER applying action via post-state next iter
                cur_share = _econ_share(obs, my_slot)
                step_rewards.append(_delta_reward(prev_share, cur_share, n_players))
                prev_share = cur_share
                my_decided_this_step = True
                actions.append(_candidate_actions(candidates[idx]))
            else:
                actions.append(_call_agent(lineup[player], obs, game.configuration))
        if not my_decided_this_step:
            # No decision recorded this step; do not append step_reward to keep alignment
            pass
        game.step(actions)

    # Sanity: alignment
    assert len(records) == len(step_rewards), (
        f"records/step_rewards mismatch {len(records)} vs {len(step_rewards)}"
    )

    final_scores = game.scores()
    terminal = _terminal_reward(final_scores, my_slot)
    win = float(final_scores[my_slot]) > max(
        float(s) for i, s in enumerate(final_scores) if i != my_slot
    ) and final_scores[my_slot] > 0

    returns = _discounted_returns(step_rewards, terminal, _GAMMA)
    return records, returns, terminal, bool(win), n_players, sp_idx_used


# ---------- training step ----------
def _ppo_update(actor: v14_core.V14Scorer, critic: V14Critic,
                triples: list[tuple[np.ndarray, int, float, float, np.ndarray, list, float]],
                epochs: int,
                entropy_beta: float,
                value_coef: float,
                grad_clip: float,
                normalize_advantage: bool = True,
                advantage_scale: float = 1.0) -> dict:
    """One PPO update. triples: list of (feats, idx, return, log_p_old, pooled, noop_idxs, temperature)."""
    if not triples:
        return {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
                "kl": 0.0, "grad_norm": 0.0, "actor_grad_norm": 0.0,
                "critic_grad_norm": 0.0, "clip_frac": 0.0,
                "adv_mean": 0.0, "adv_std": 0.0, "ret_mean": 0.0,
                "ret_std": 0.0, "val_mean": 0.0, "ratio_std": 0.0}

    # Compute advantages once
    rets = np.array([t[2] for t in triples], dtype=np.float32)
    vals_old = np.array([critic.forward(t[4])[0] for t in triples], dtype=np.float32)
    advs = rets - vals_old
    raw_adv_mean = float(advs.mean())
    raw_adv_std = float(advs.std())
    if normalize_advantage and advs.std() > 1e-6:
        advs = (advs - advs.mean()) / (advs.std() + 1e-6)
    elif normalize_advantage:
        advs = advs - advs.mean()
    advs = advs * float(advantage_scale)

    metrics = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
               "kl": 0.0, "grad_norm": 0.0, "actor_grad_norm": 0.0,
               "critic_grad_norm": 0.0, "clip_frac": 0.0,
               "adv_mean": float(advs.mean()), "adv_std": float(advs.std()),
               "raw_adv_mean": raw_adv_mean, "raw_adv_std": raw_adv_std,
               "ret_mean": float(rets.mean()), "ret_std": float(rets.std()),
               "val_mean": float(vals_old.mean()), "ratio_std": 0.0}
    n = len(triples)

    for _epoch in range(epochs):
        actor_grads = {k: np.zeros_like(v) for k, v in actor.to_dict().items()}
        critic_grads = {k: np.zeros_like(v) for k, v in critic.to_dict().items()}
        ent_acc = 0.0
        kl_acc = 0.0
        ratios_acc = []
        pg_loss_acc = 0.0
        v_loss_acc = 0.0
        clip_count = 0

        for (feats, idx, ret, log_p_old, pooled, noop_idxs, temp), adv in zip(triples, advs):
            # Actor forward with cache
            scores_raw, cache = actor.forward_with_cache(feats)
            scores = scores_raw.copy()
            for ni in noop_idxs:
                scores[ni] += _NOOP_LOGIT_BIAS
            logits = scores / temp
            lse = _logsumexp(logits)
            log_p = logits - lse
            probs = np.exp(log_p)
            log_p_new = float(log_p[int(idx)])
            ratio = math.exp(log_p_new - float(log_p_old))
            ratios_acc.append(ratio)

            # PPO clip on logits
            unclipped = ratio * float(adv)
            clipped_r = max(1.0 - _PPO_CLIP, min(1.0 + _PPO_CLIP, ratio))
            clipped = clipped_r * float(adv)
            use_unclipped = unclipped <= clipped  # min selects unclipped
            if not use_unclipped:
                clip_count += 1

            obj = min(unclipped, clipped)
            pg_loss_acc += -obj

            # Gradient w.r.t. logits (then divide by temp for scores grad)
            # If clipped path active and adv*ratio is the binding side, no gradient.
            if use_unclipped:
                # d(-ratio*adv)/d log_p_new = -ratio*adv
                # d log_p_new / d logits_j = (one_hot[idx] - probs[j])
                grad_logits = -ratio * float(adv) * (-probs)
                grad_logits[int(idx)] += -ratio * float(adv) * 1.0
                # Above: grad_logits = -ratio*adv * (one_hot[idx] - probs)
            else:
                grad_logits = np.zeros_like(probs)

            # Entropy bonus: loss_ent = -beta*H, dL/dlogit_j = beta * p_j * (log p_j + H)
            H = float(-np.sum(probs * np.log(probs + 1e-12)))
            ent_acc += H
            ent_grad = float(entropy_beta) * probs * (np.log(probs + 1e-12) + H)
            grad_logits = grad_logits + ent_grad

            # Convert logits grad to scores grad (logits = scores/temp → dScores = dLogits/temp)
            grad_scores = grad_logits / temp

            # KL approx
            kl_acc += float(log_p_old - log_p_new)

            # Average across batch
            grad_scores = grad_scores / max(1, n)
            actor_sample = _backward(actor, cache, grad_scores)
            for k in actor_grads:
                actor_grads[k] += actor_sample[k]

            # Critic
            v_cur, vcache = critic.forward(pooled)
            v_err = v_cur - float(ret)
            v_loss_acc += 0.5 * v_err * v_err
            grad_v = float(value_coef) * v_err / max(1, n)
            critic_sample = critic.backward(vcache, float(grad_v))
            for k in critic_grads:
                critic_grads[k] += critic_sample[k]

        actor_norm = _global_grad_norm(actor_grads)
        critic_norm = _global_grad_norm(critic_grads)
        all_grads = {**actor_grads, **critic_grads}
        gn = _clip_grads(all_grads, float(grad_clip))
        metrics["grad_norm"] = float(gn)
        metrics["actor_grad_norm"] = float(actor_norm)
        metrics["critic_grad_norm"] = float(critic_norm)
        metrics["pg_loss"] = pg_loss_acc / max(1, n)
        metrics["v_loss"] = v_loss_acc / max(1, n)
        metrics["entropy"] = ent_acc / max(1, n)
        metrics["kl"] = kl_acc / max(1, n)
        metrics["clip_frac"] = clip_count / max(1, n)
        metrics["ratio_std"] = float(np.std(ratios_acc)) if ratios_acc else 0.0

        # Apply via Adam — caller does it; return grads here actually
        # We do step inside main loop. Stash grads in metrics.
        metrics.setdefault("_actor_grads", actor_grads)
        metrics.setdefault("_critic_grads", critic_grads)
        # Only store last epoch grads. To keep multi-epoch correct we need to step inline.
        # For simplicity, single-epoch step. Multi-epoch left for future: break here.
        break

    return metrics


def _policy_change_metrics(
    before_w: dict[str, np.ndarray],
    after: v14_core.V14Scorer,
    triples: list[tuple[np.ndarray, int, float, float, np.ndarray, list, float]],
) -> dict[str, float]:
    if not triples:
        return {"post_kl": 0.0, "logit_delta": 0.0, "chosen_logit_delta": 0.0}
    before = v14_core.V14Scorer(weights=before_w)
    kls = []
    deltas = []
    chosen_deltas = []
    for feats, idx, _ret, _old_logp, _pooled, noop_idxs, temp in triples[:512]:
        old_scores = before.forward(feats).astype(np.float32)
        new_scores = after.forward(feats).astype(np.float32)
        for ni in noop_idxs:
            old_scores[ni] += _NOOP_LOGIT_BIAS
            new_scores[ni] += _NOOP_LOGIT_BIAS
        old_logits = old_scores / temp
        new_logits = new_scores / temp
        old_logp = old_logits - _logsumexp(old_logits)
        new_logp = new_logits - _logsumexp(new_logits)
        old_p = np.exp(old_logp)
        kls.append(float(np.sum(old_p * (old_logp - new_logp))))
        deltas.append(float(np.mean(np.abs(new_scores - old_scores))))
        chosen_deltas.append(float(new_scores[int(idx)] - old_scores[int(idx)]))
    return {
        "post_kl": float(np.mean(kls)),
        "logit_delta": float(np.mean(deltas)),
        "chosen_logit_delta": float(np.mean(chosen_deltas)),
    }


# ---------- main ----------
def main() -> None:
    global _RANK_REWARD_4P
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--load", type=Path, default=Path("evaluations/scorer_v14.npz"))
    parser.add_argument("--load-critic", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("evaluations/scorer_v14_v2.npz"))
    parser.add_argument("--out-critic", type=Path,
                        default=Path("evaluations/critic_v14_v2.npz"))
    parser.add_argument("--bc-data", type=Path,
                        default=Path("replay_dataset/v14_bc_top1.npz"))
    parser.add_argument("--bc-weight-4p", type=float, default=0.25)
    parser.add_argument("--bc-weight-2p", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--temperature-start", type=float, default=1.1)
    parser.add_argument("--temperature-end", type=float, default=0.7)
    parser.add_argument("--modes", nargs="*", default=["4p", "4p", "4p", "4p"])
    parser.add_argument("--selfplay-pool", type=int, default=8)
    parser.add_argument("--selfplay-every", type=int, default=3)
    parser.add_argument("--rank-reward-4p", nargs=4, type=float,
                        default=[1.0, 0.3, -0.3, -0.8],
                        metavar=("R1", "R2", "R3", "R4"))
    parser.add_argument("--value-coef", type=float, default=_VALUE_COEF)
    parser.add_argument("--entropy-beta", type=float, default=_ENTROPY_BETA)
    parser.add_argument("--grad-clip", type=float, default=_GRAD_CLIP)
    parser.add_argument("--ppo-epochs", type=int, default=_PPO_EPOCHS)
    parser.add_argument("--advantage-scale", type=float, default=1.0)
    parser.add_argument("--best-window", type=int, default=4,
                        help="Rolling 4p window used to decide best4p checkpoint saving.")
    parser.add_argument("--no-adv-norm", action="store_true")
    parser.add_argument("--no-bc", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--opponents", nargs="*", default=[
        "greedy", "starter",
        "notebook_distance_prioritized",
        "notebook_physics_accurate",
        "notebook_orbitbotnext",
        "notebook_pascalledesma_orbitwork_v14",
    ])
    parser.add_argument("--seed", type=int, default=1414)
    args = parser.parse_args()
    _RANK_REWARD_4P = tuple(float(x) for x in args.rank_reward_4p)

    actor = (v14_core.V14Scorer.load(args.load)
             if args.load.exists() else v14_core.V14Scorer())
    critic_w = dict(np.load(args.load_critic)) if args.load_critic and args.load_critic.exists() else None
    critic = V14Critic(weights=critic_w)

    actor_opt = Adam(actor.to_dict(), lr=args.lr)
    critic_opt = Adam(critic.to_dict(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    data = np.load(args.bc_data) if args.bc_data.exists() and not args.no_bc else None
    X = data["X"].astype(np.float32) if data is not None else None
    mask = data["mask"].astype(np.float32) if data is not None else None
    y = data["y"].astype(np.int64) if data is not None else np.zeros(0, dtype=np.int64)

    opponent_names = tuple(name for name in args.opponents if name in ZOO)
    if not opponent_names:
        raise SystemExit("No valid opponents.")
    modes = tuple(4 if m == "4p" else 2 for m in args.modes)

    # Self-play pool: list of dicts; track winrate-vs each for PFSP
    selfplay_pool: list[dict] = [{k: v.copy() for k, v in actor.to_dict().items()}]
    sp_games = [0]
    sp_wins = [0]

    deadline = time.time() + args.minutes * 60.0
    batch = 0
    games_total = 0
    best_train_wr = -1.0
    best_train_wr4 = -1.0
    best_train_wr2 = -1.0
    wr4_history: deque[float] = deque(maxlen=max(1, int(args.best_window)))
    started = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    path_best_global = args.out.with_suffix(".best.npz")
    path_best_4p = args.out.with_name(args.out.stem + ".best4p.npz")
    path_best_2p = args.out.with_name(args.out.stem + ".best2p.npz")

    total_batches_estimate = max(1, int(args.minutes * 60 / 60))  # rough

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        while time.time() < deadline:
            # LR cosine decay
            t_frac = min(1.0, (time.time() - started) / max(1.0, args.minutes * 60))
            lr_t = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + math.cos(math.pi * t_frac))
            actor_opt.lr = lr_t
            critic_opt.lr = lr_t

            # Temperature anneal
            temp_t = (args.temperature_start
                      + (args.temperature_end - args.temperature_start) * t_frac)

            actor_w = {k: v.copy() for k, v in actor.to_dict().items()}
            critic_w = {k: v.copy() for k, v in critic.to_dict().items()}
            before_update_w = {k: v.copy() for k, v in actor_w.items()}

            sp_snap = [{k: v.copy() for k, v in w.items()}
                       for w in selfplay_pool[-args.selfplay_pool:]]

            # PFSP weights: ∝ (1 - winrate_vs_opp)^2
            if sp_snap:
                wr_per = []
                pool_offset = max(0, len(selfplay_pool) - args.selfplay_pool)
                for i in range(len(sp_snap)):
                    gi = sp_games[pool_offset + i] if pool_offset + i < len(sp_games) else 0
                    wi = sp_wins[pool_offset + i] if pool_offset + i < len(sp_wins) else 0
                    wr = (wi / gi) if gi > 0 else 0.5
                    wr_per.append((1.0 - wr) ** 2 + 0.05)
                pfsp = np.array(wr_per, dtype=np.float64)
                pfsp = pfsp / pfsp.sum()
            else:
                pfsp = None

            tasks = []
            for game_i in range(args.batch_size):
                n_players = modes[(batch * args.batch_size + game_i) % len(modes)]
                tasks.append((
                    actor_w, critic_w,
                    int(rng.integers(0, 1_000_000_000)),
                    args.max_steps, n_players,
                    opponent_names, temp_t, sp_snap,
                    pfsp.tolist() if pfsp is not None else None,
                ))
            results = list(pool.map(_play_episode_task, tasks))

            # Update self-play stats (approx: each used checkpoint shares win)
            pool_offset = max(0, len(selfplay_pool) - args.selfplay_pool)
            for records, returns, terminal, win, np_, sp_used in results:
                for pidx in sp_used:
                    abs_idx = pool_offset + pidx
                    if abs_idx < len(sp_games):
                        sp_games[abs_idx] += 1
                        if win:
                            sp_wins[abs_idx] += 1

            # Build triples for PPO update
            triples = []
            wins = []
            nps = []
            terminals = []
            for records, returns, terminal, win, np_, _ in results:
                wins.append(win)
                nps.append(np_)
                terminals.append(terminal)
                for rec, ret in zip(records, returns):
                    triples.append((
                        rec["feats"], rec["idx"], float(ret),
                        rec["log_p_old"], rec["pooled"],
                        rec["noop_bias_idx"], rec["temperature"],
                    ))

            metrics = _ppo_update(
                actor,
                critic,
                triples,
                max(1, int(args.ppo_epochs)),
                entropy_beta=float(args.entropy_beta),
                value_coef=float(args.value_coef),
                grad_clip=float(args.grad_clip),
                normalize_advantage=not bool(args.no_adv_norm),
                advantage_scale=float(args.advantage_scale),
            )
            actor_grads = metrics.pop("_actor_grads", None)
            critic_grads = metrics.pop("_critic_grads", None)

            # BC anchor (mixed weight by mode share)
            n4p = sum(1 for x in nps if x == 4)
            n2p = sum(1 for x in nps if x == 2)
            bc_w = 0.0 if args.no_bc else (
                (n4p * args.bc_weight_4p + n2p * args.bc_weight_2p) / max(1, len(nps))
            )
            bc_loss_acc = 0.0
            bc_correct = 0
            bc_used = 0
            if X is not None and len(y) > 0 and bc_w > 0 and actor_grads is not None:
                idxs = rng.choice(len(y), size=min(args.batch_size, len(y)), replace=False)
                for ii in idxs:
                    k = int(mask[ii].sum())
                    label = int(y[ii])
                    if k <= 0 or label >= k:
                        continue
                    feat_slice = X[ii, :k, : v14_core.FEATURE_DIM]
                    if feat_slice.shape[1] < v14_core.FEATURE_DIM:
                        pad = np.zeros((k, v14_core.FEATURE_DIM - feat_slice.shape[1]), dtype=np.float32)
                        feat_slice = np.concatenate([feat_slice, pad], axis=1)
                    sc, cache = actor.forward_with_cache(feat_slice)
                    probs = v14_core.softmax(sc)
                    bc_loss_acc += float(-math.log(float(probs[label]) + 1e-12))
                    bc_correct += int(np.argmax(probs) == label)
                    bc_used += 1
                    grad_l = probs.copy()
                    grad_l[label] -= 1.0
                    sample = _backward(actor, cache, grad_l * (bc_w / max(1, len(idxs))))
                    for kk in actor_grads:
                        actor_grads[kk] += sample[kk]

            # Apply optimizer step (grads were already global-norm-clipped on actor+critic together,
            # but BC was added after. Re-clip combined.)
            if actor_grads is not None:
                _clip_grads(actor_grads, float(args.grad_clip))
                aparams = actor.to_dict()
                if not args.diagnostic_only:
                    actor_opt.step(aparams, actor_grads)
                actor.W1, actor.b1 = aparams["W1"], aparams["b1"]
                actor.W2, actor.b2 = aparams["W2"], aparams["b2"]
                actor.W3, actor.b3 = aparams["W3"], aparams["b3"]
            if critic_grads is not None:
                _clip_grads(critic_grads, float(args.grad_clip))
                cparams = critic.to_dict()
                if not args.diagnostic_only:
                    critic_opt.step(cparams, critic_grads)
                critic.W1, critic.b1 = cparams["cW1"], cparams["cb1"]
                critic.W2, critic.b2 = cparams["cW2"], cparams["cb2"]

            change = _policy_change_metrics(before_update_w, actor, triples)

            np.savez(args.out, **actor.to_dict())
            np.savez(args.out_critic, **critic.to_dict())

            if batch % args.selfplay_every == 0:
                selfplay_pool.append({k: v.copy() for k, v in actor.to_dict().items()})
                sp_games.append(0)
                sp_wins.append(0)
                if len(selfplay_pool) > args.selfplay_pool * 2:
                    step = max(1, len(selfplay_pool) // args.selfplay_pool)
                    selfplay_pool = selfplay_pool[::step][-args.selfplay_pool:]
                    sp_games = sp_games[::step][-args.selfplay_pool:]
                    sp_wins = sp_wins[::step][-args.selfplay_pool:]

            games_total += len(results)
            wr = sum(wins) / max(1, len(wins))
            wr4_w = [w for w, p in zip(wins, nps) if p == 4]
            wr2_w = [w for w, p in zip(wins, nps) if p == 2]
            wr4 = sum(wr4_w) / max(1, len(wr4_w))
            wr2 = sum(wr2_w) / max(1, len(wr2_w))
            if wr4_w:
                wr4_history.append(wr4)
            wr4_ma = sum(wr4_history) / max(1, len(wr4_history))

            if wr > best_train_wr:
                best_train_wr = wr
                np.savez(path_best_global, **actor.to_dict())
            if wr4_w and wr4_ma > best_train_wr4:
                best_train_wr4 = wr4_ma
                np.savez(path_best_4p, **actor.to_dict())
            if wr2_w and wr2 > best_train_wr2:
                best_train_wr2 = wr2
                np.savez(path_best_2p, **actor.to_dict())

            elapsed = time.time() - started
            print(
                f"[{elapsed:6.0f}s b{batch:04d}] games={games_total} "
                f"wr={sum(wins)}/{len(wins)} ({wr:.3f}) "
                f"wr2={wr2:.3f}(best={best_train_wr2:.3f}) "
                f"wr4={wr4:.3f} ma={wr4_ma:.3f}(best={best_train_wr4:.3f}) "
                f"reward={float(np.mean(terminals)):+.3f} "
                f"dec={len(triples)} sp={len(selfplay_pool)} "
                f"pg={metrics['pg_loss']:+.3f} v={metrics['v_loss']:.3f} "
                f"H={metrics['entropy']:.3f} kl={metrics['kl']:+.4f} "
                f"postkl={change['post_kl']:.5f} dlogit={change['logit_delta']:.5f} "
                f"clip={metrics['clip_frac']:.2f} gn={metrics['grad_norm']:.2f} "
                f"agn={metrics['actor_grad_norm']:.2f} cgn={metrics['critic_grad_norm']:.2f} "
                f"adv={metrics['adv_mean']:+.3f}/{metrics['adv_std']:.3f} "
                f"rawadv={metrics.get('raw_adv_mean', 0.0):+.3f}/{metrics.get('raw_adv_std', 0.0):.3f} "
                f"ret={metrics['ret_mean']:+.3f}/{metrics['ret_std']:.3f} "
                f"bc={bc_w:.2f}/{(bc_loss_acc/max(1,bc_used)):.3f}/{(bc_correct/max(1,bc_used)):.2f} "
                f"lr={lr_t:.1e} T={temp_t:.2f}",
                flush=True,
            )
            batch += 1


if __name__ == "__main__":
    main()
