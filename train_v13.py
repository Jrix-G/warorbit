"""
train_v13.py — REINFORCE training for V13 hybrid bot

Uses c_engine.CGame to step through games, recording
(features, selected_idx) at each turn, then applying REINFORCE gradient
update on the MLP scorer.

Pool: strong notebooks from opponents.ZOO.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

import bot_v13
from c_engine import CGame
from opponents import ZOO


# ═════════════════════════════ ADAM OPTIMIZER ═════════════════════════════════

class Adam:
    def __init__(self, params: dict, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ═════════════════════════════ MLP BACKPROP ═══════════════════════════════════

def mlp_backward(mlp: bot_v13.MLPScorer, cache: dict, grad_logits: np.ndarray):
    """
    grad_logits: (N,) — gradient of loss w.r.t. final logits (z3).
    Returns dict of gradients for each parameter.
    """
    g3 = grad_logits.reshape(-1, 1)              # (N, 1)
    h2 = cache['h2']                              # (N, 32)
    h1 = cache['h1']                              # (N, 64)
    X = cache['X']                                # (N, 12)
    z2 = cache['z2']
    z1 = cache['z1']

    dW3 = h2.T @ g3                               # (32, 1)
    db3 = g3.sum(axis=0)                          # (1,)

    dh2 = g3 @ mlp.W3.T                           # (N, 32)
    dz2 = dh2 * (z2 > 0).astype(np.float32)       # ReLU grad
    dW2 = h1.T @ dz2                              # (64, 32)
    db2 = dz2.sum(axis=0)                         # (32,)

    dh1 = dz2 @ mlp.W2.T                          # (N, 64)
    dz1 = dh1 * (z1 > 0).astype(np.float32)
    dW1 = X.T @ dz1                               # (12, 64)
    db1 = dz1.sum(axis=0)                         # (64,)

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}


# ═════════════════════════════ TRAJECTORY COLLECTION ══════════════════════════

@dataclass
class StepRecord:
    features: np.ndarray              # (N_candidates, 12)
    selected_idx: int                 # index of chosen candidate
    n_candidates: int                 # for masking later

@dataclass
class Trajectory:
    steps: list[StepRecord] = field(default_factory=list)
    final_reward: float = 0.0
    won: bool = False


def _agent_obs(obs: Any) -> dict[str, Any]:
    """Normalize CGame observations for legacy agents expecting dict.get()."""
    if isinstance(obs, dict):
        return obs
    data = vars(obs).copy()
    data.setdefault("remainingOverageTime", 60.0)
    return data


def _call_agent(agent_fn: Callable, obs: Any, config: Any) -> list:
    agent_input = _agent_obs(obs)
    try:
        move = agent_fn(agent_input, config)
    except TypeError:
        move = agent_fn(agent_input)
    return move if isinstance(move, list) else []


def make_training_agent(mlp: bot_v13.MLPScorer,
                        trajectory: Trajectory,
                        rng: np.random.Generator,
                        epsilon: float = 0.05) -> Callable:
    """
    Return an agent function that records (features, selected_idx) into the trajectory.
    Sample from softmax during training (with optional epsilon-greedy).
    """
    def _agent(obs, config=None):
        my_id = int(bot_v13._get(obs, 'player', 0))
        current_step = int(bot_v13._get(obs, 'step', 0))
        av = float(bot_v13._get(obs, 'angular_velocity', 0.03))
        planets = list(bot_v13._get(obs, 'planets', []) or [])
        if not planets:
            return []

        ip = bot_v13._build_initial_map(obs)
        arrival_table = bot_v13._build_arrival_table(obs, ip, av, current_step)
        candidates = bot_v13.generate_all_candidates(
            obs, my_id, ip, av, current_step, arrival_table)
        if not candidates:
            return []

        feats = np.stack([c['features'] for c in candidates])
        scores = mlp.forward(feats)

        # Softmax sampling (with temperature 1)
        if rng.random() < epsilon:
            selected_idx = int(rng.integers(0, len(candidates)))
        else:
            shifted = scores - scores.max()
            probs = np.exp(shifted)
            probs /= probs.sum()
            selected_idx = int(rng.choice(len(candidates), p=probs))

        # Record for training
        trajectory.steps.append(StepRecord(
            features=feats,
            selected_idx=selected_idx,
            n_candidates=len(candidates),
        ))

        # Build action list from selected candidate
        c = candidates[selected_idx]
        if c['type'] == 'noop' or not c['moves']:
            return []
        actions = []
        used = set()
        for (sid, angle, ships) in c['moves']:
            if sid in used:
                continue
            actions.append([int(sid), float(angle), int(ships)])
            used.add(sid)
            if len(actions) >= bot_v13.MAX_ACTIONS:
                break
        return actions

    return _agent


def play_episode(mlp: bot_v13.MLPScorer,
                 opponents: list[Callable],
                 my_slot: int,
                 seed: int,
                 max_steps: int = 220,
                 use_c_accel: bool = True,
                 rng: np.random.Generator | None = None,
                 epsilon: float = 0.05) -> Trajectory:
    """Play one game with our trainee in slot `my_slot`. Return Trajectory."""
    if rng is None:
        rng = np.random.default_rng(seed)

    n_players = 1 + len(opponents)
    traj = Trajectory()
    trainee = make_training_agent(mlp, traj, rng, epsilon=epsilon)

    agents = list(opponents)
    agents.insert(my_slot, trainee)

    game = CGame(
        n_players=n_players,
        seed=seed,
        episode_steps=max_steps,
    )

    callables = list(agents)
    while not game.done:
        actions = []
        for player, agent_fn in enumerate(callables):
            obs = game.observation(player)
            actions.append(_call_agent(agent_fn, obs, game.configuration))
        game.step(actions)

    scores = game.scores()
    my_score = scores[my_slot]
    best_other = max(s for i, s in enumerate(scores) if i != my_slot) if len(scores) > 1 else 0

    if my_score > best_other and my_score > 0:
        traj.final_reward = 1.0
        traj.won = True
    elif my_score == best_other:
        traj.final_reward = 0.0
    else:
        traj.final_reward = -1.0

    # Dense shaping bonus: relative score margin
    total = my_score + best_other
    if total > 0:
        margin = (my_score - best_other) / total
        traj.final_reward += 0.3 * margin

    return traj


def _play_episode_worker(task: tuple[dict, str, int, int, int, float, int]) -> Trajectory:
    weights, opp_name, slot, seed, max_steps, epsilon, rng_seed = task
    mlp = bot_v13.MLPScorer(weights=weights)
    rng = np.random.default_rng(rng_seed)
    return play_episode(
        mlp,
        [ZOO[opp_name]],
        my_slot=slot,
        seed=seed,
        max_steps=max_steps,
        rng=rng,
        epsilon=epsilon,
    )


# ═════════════════════════════ REINFORCE UPDATE ═══════════════════════════════

def reinforce_update(mlp: bot_v13.MLPScorer,
                     optimizer: Adam,
                     trajectories: list[Trajectory],
                     entropy_bonus: float = 0.01):
    """
    Apply REINFORCE update across a batch of trajectories.
    Loss = -E[log_prob(selected) * advantage] - entropy_bonus * H
    """
    rewards = np.array([t.final_reward for t in trajectories], dtype=np.float32)
    if rewards.std() > 1e-6:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    else:
        advantages = rewards - rewards.mean()

    # Accumulate gradients across all (trajectory, step) pairs
    params = mlp.to_dict()
    accum_grads = {k: np.zeros_like(v) for k, v in params.items()}
    total_loss = 0.0
    total_entropy = 0.0
    total_steps = 0

    for traj, adv in zip(trajectories, advantages):
        if not traj.steps:
            continue
        for step_rec in traj.steps:
            X = step_rec.features
            scores, cache = mlp.forward_with_cache(X)
            shifted = scores - scores.max()
            exp_s = np.exp(shifted)
            probs = exp_s / exp_s.sum()
            log_probs = np.log(probs + 1e-12)

            sel = step_rec.selected_idx
            entropy = -np.sum(probs * log_probs)

            # Gradient of -log_prob[sel] * adv  w.r.t. logits = (probs - one_hot[sel]) * adv
            grad_logits = probs.copy()
            grad_logits[sel] -= 1.0
            grad_logits *= float(adv)

            # Entropy bonus gradient: H = -Σ p log p
            # dH/dz_i = p_i * (Σ_j p_j log p_j - log p_i - 1) = -p_i * (log p_i + entropy_const)
            # Simpler: grad of -H w.r.t. logits = probs * (log_probs - entropy)
            ent_grad = probs * (log_probs - entropy)
            grad_logits += entropy_bonus * ent_grad

            grads = mlp_backward(mlp, cache, grad_logits)
            for k in accum_grads:
                accum_grads[k] += grads[k]

            total_loss += -log_probs[sel] * float(adv)
            total_entropy += entropy
            total_steps += 1

    if total_steps == 0:
        return 0.0, 0.0

    # Average gradients across steps
    for k in accum_grads:
        accum_grads[k] /= total_steps

    # Gradient clipping
    total_norm = math.sqrt(sum(float((g * g).sum()) for g in accum_grads.values()))
    max_norm = 1.0
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for k in accum_grads:
            accum_grads[k] *= scale

    optimizer.step(params, accum_grads)
    # Reflect updates back to mlp object
    mlp.W1, mlp.b1 = params['W1'], params['b1']
    mlp.W2, mlp.b2 = params['W2'], params['b2']
    mlp.W3, mlp.b3 = params['W3'], params['b3']

    return total_loss / total_steps, total_entropy / total_steps


# ═════════════════════════════ EVALUATION ═════════════════════════════════════

def evaluate(mlp: bot_v13.MLPScorer,
             opponents_named: list[tuple[str, Callable]],
             games_per_opp: int = 6,
             max_steps: int = 220,
             seed_base: int = 9000,
             use_c_accel: bool = True) -> dict[str, float]:
    """Evaluate current MLP against each opponent. Returns winrate dict."""
    results = {}
    bot_v13.set_mlp(mlp)
    for name, opp in opponents_named:
        wins = 0
        for i in range(games_per_opp):
            slot = i % 2  # alternate first/second
            seed = seed_base + i * 17
            n_players = 2
            agents = [bot_v13.agent if p == slot else opp for p in range(n_players)]
            game = CGame(
                n_players=n_players,
                seed=seed,
                episode_steps=max_steps,
            )
            callables = list(agents)
            while not game.done:
                actions = []
                for p, fn in enumerate(callables):
                    obs = game.observation(p)
                    actions.append(_call_agent(fn, obs, game.configuration))
                game.step(actions)
            scores = game.scores()
            best_other = max(s for i2, s in enumerate(scores) if i2 != slot)
            if scores[slot] > best_other and scores[slot] > 0:
                wins += 1
        results[name] = wins / games_per_opp
    return results


# ═════════════════════════════ MAIN ═══════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=30.0)
    ap.add_argument('--batch-size', type=int, default=16, help='games per gradient update')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--epsilon', type=float, default=0.05)
    ap.add_argument('--entropy', type=float, default=0.01)
    ap.add_argument('--max-steps', type=int, default=220)
    ap.add_argument('--eval-every', type=int, default=20, help='eval every N batches')
    ap.add_argument('--eval-games', type=int, default=6)
    ap.add_argument('--workers', type=int, default=1, help='parallel episode workers')
    ap.add_argument('--out', type=str, default='evaluations/scorer_v13')
    ap.add_argument('--load', type=str, default=None)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument(
        '--no-c-accel',
        action='store_true',
        help='Ignored; V13 now always uses c_engine.',
    )
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.load and Path(args.load).exists():
        print(f'Loading checkpoint: {args.load}')
        mlp = bot_v13.MLPScorer(weights=dict(np.load(args.load)))
    else:
        print('Initializing fresh MLP weights')
        mlp = bot_v13.MLPScorer(weights=None)

    optimizer = Adam(mlp.to_dict(), lr=args.lr)

    # Pool: mix of weak (for warmup) + strong (for final policy)
    pool_2p_names = [
        'greedy',
        'starter',
        'notebook_distance_prioritized',
        'notebook_physics_accurate',
        'notebook_orbitbotnext',
    ]

    eval_pool = [
        ('greedy', ZOO['greedy']),
        ('notebook_distance_prioritized', ZOO['notebook_distance_prioritized']),
        ('notebook_physics_accurate', ZOO['notebook_physics_accurate']),
        ('notebook_orbitbotnext', ZOO['notebook_orbitbotnext']),
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = out_path.with_suffix('.best.npz')
    last_path = out_path.with_suffix('.npz')

    use_c_accel = not args.no_c_accel
    deadline = time.time() + args.minutes * 60.0
    batch_idx = 0
    best_avg_wr = -1.0

    workers = max(1, int(args.workers))
    print(
        f'Training for {args.minutes:.1f} min. '
        f'Pool: {len(pool_2p_names)} opponents. Workers: {workers}.'
    )

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        while time.time() < deadline:
            weights = mlp.to_dict()
            tasks = []
            for _ in range(args.batch_size):
                opp_name = pool_2p_names[int(rng.integers(0, len(pool_2p_names)))]
                slot = int(rng.integers(0, 2))
                seed = int(rng.integers(0, 1_000_000))
                rng_seed = int(rng.integers(0, 1_000_000_000))
                tasks.append((
                    weights,
                    opp_name,
                    slot,
                    seed,
                    args.max_steps,
                    args.epsilon,
                    rng_seed,
                ))

            if executor is None:
                trajectories = [_play_episode_worker(task) for task in tasks]
            else:
                trajectories = list(executor.map(_play_episode_worker, tasks))

            loss, entropy = reinforce_update(
                mlp, optimizer, trajectories, entropy_bonus=args.entropy)

            wins = sum(1 for t in trajectories if t.won)
            elapsed = time.time() - (deadline - args.minutes * 60.0)
            print(f'[{elapsed:6.0f}s b{batch_idx:4d}] '
                  f'wr={wins}/{len(trajectories)} loss={loss:+.3f} ent={entropy:.3f}')

            # Save last
            np.savez(last_path, **mlp.to_dict())

            # Eval periodically
            if (batch_idx + 1) % args.eval_every == 0:
                print('  Evaluating...')
                results = evaluate(mlp, eval_pool, games_per_opp=args.eval_games,
                                   max_steps=args.max_steps, use_c_accel=use_c_accel)
                avg_wr = sum(results.values()) / len(results)
                for name, wr in results.items():
                    print(f'    {name}: {wr:.2%}')
                print(f'    avg: {avg_wr:.2%}')
                if avg_wr > best_avg_wr:
                    best_avg_wr = avg_wr
                    np.savez(best_path, **mlp.to_dict())
                    print(f'    → new best, saved {best_path}')

            batch_idx += 1
    finally:
        if executor is not None:
            executor.shutdown()

    # Final eval
    print('Final eval:')
    results = evaluate(mlp, eval_pool, games_per_opp=args.eval_games * 2,
                       max_steps=args.max_steps, use_c_accel=use_c_accel)
    for name, wr in results.items():
        print(f'  {name}: {wr:.2%}')


if __name__ == '__main__':
    main()
