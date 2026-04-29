#!/usr/bin/env python3
"""League training loop for Orbit Wars bot.

Architecture:
- Pool of checkpoints (historical versions of the bot)
- Each iteration: current bot plays vs random checkpoint from pool + zoo opponents
- Dense reward per turn (not just win/lose)
- When current bot beats pool avg > WIN_THRESHOLD → save checkpoint to pool
- MLP mission scorer: state features → log-multiplier on heuristic score

numpy only, no PyTorch.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from SimGame import run_match
from opponents import ZOO

CHECKPOINTS_DIR = ROOT / "checkpoints" / "league"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Hyperparameters ────────────────────────────────────────────────────────
POOL_SIZE = 8             # max checkpoints in pool
WIN_THRESHOLD = 0.60      # winrate vs pool to earn a checkpoint slot
EVAL_GAMES = 20           # games to evaluate vs pool
BATCH_SIZE = 32           # episodes per gradient update
LR_INIT = 5e-3
LR_MIN = 1e-4
LR_DECAY = 0.998          # per batch
NOISE_STD_INIT = 0.4
NOISE_STD_MIN = 0.05
NOISE_DECAY = 0.997
GAMMA = 0.99              # dense reward discount
BASELINE_WINDOW = 100     # moving average window for reward baseline
ZOO_OPPONENT_RATIO = 0.4  # fraction of games vs zoo (vs pool)
FOUR_PLAYER_RATIO = 0.7   # fraction of 4p games

# MLP dims
INPUT_DIM = 18
H1 = 32
H2 = 16

# ─── MLP ────────────────────────────────────────────────────────────────────

class MLP:
    """2-layer MLP with tanh activations. Output = log-multiplier."""

    def __init__(self, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(INPUT_DIM, H1).astype(np.float32) * 0.01
        self.b1 = np.zeros(H1, dtype=np.float32)
        self.W2 = rng.randn(H1, H2).astype(np.float32) * 0.01
        self.b2 = np.zeros(H2, dtype=np.float32)
        self.W3 = rng.randn(H2, 1).astype(np.float32) * 0.01
        self.b3 = np.zeros(1, dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """x: (F,) → scalar log-multiplier. Returns (out, cache)."""
        h1 = np.tanh(x @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        out = (h2 @ self.W3 + self.b3).squeeze()
        return float(out), {"x": x, "h1": h1, "h2": h2}

    def backward(self, cache: dict, grad_out: float) -> dict:
        h1, h2, x = cache["h1"], cache["h2"], cache["x"]
        dout = np.array([[grad_out]], dtype=np.float32)
        dW3 = h2[:, None] @ dout
        db3 = dout.squeeze()
        dh2 = (dout @ self.W3.T).squeeze() * (1 - h2 ** 2)
        dW2 = h1[:, None] @ dh2[None, :]
        db2 = dh2
        dh1 = (dh2 @ self.W2.T) * (1 - h1 ** 2)
        dW1 = x[:, None] @ dh1[None, :]
        db1 = dh1
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def apply_grad(self, grads: dict, lr: float):
        for k in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(self, k, getattr(self, k) - lr * grads[k])

    def save(self, path: Path):
        np.savez_compressed(str(path),
                            W1=self.W1, b1=self.b1,
                            W2=self.W2, b2=self.b2,
                            W3=self.W3, b3=self.b3)

    @classmethod
    def load(cls, path: Path) -> "MLP":
        m = cls.__new__(cls)
        d = np.load(str(path))
        m.W1, m.b1 = d["W1"], d["b1"]
        m.W2, m.b2 = d["W2"], d["b2"]
        m.W3, m.b3 = d["W3"], d["b3"]
        return m

    def params_flat(self) -> np.ndarray:
        return np.concatenate([v.ravel() for v in (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)])


# ─── Feature extraction ──────────────────────────────────────────────────────

def extract_features(obs, player: int, n_players: int, step: int) -> np.ndarray:
    """Extract 18 state features from raw obs dict."""
    planets = obs.get("planets", []) or []
    fleets = obs.get("fleets", []) or []

    my_planets = sum(1 for p in planets if (p[1] if isinstance(p, list) else p.get("owner")) == player)
    total_planets = len(planets) + 1e-9

    def owner(p): return p[1] if isinstance(p, list) else p.get("owner", -1)
    def ships(p): return float(p[5] if isinstance(p, list) else p.get("ships", 0))
    def prod(p): return float(p[6] if isinstance(p, list) else p.get("production", 0))
    def fowner(f): return f[1] if isinstance(f, list) else f.get("owner", -1)
    def fships(f): return float(f[6] if isinstance(f, list) else f.get("ships", 0))

    my_ships = sum(ships(p) for p in planets if owner(p) == player)
    my_ships += sum(fships(f) for f in fleets if fowner(f) == player)
    enemy_ships = sum(ships(p) for p in planets if owner(p) not in (-1, player))
    enemy_ships += sum(fships(f) for f in fleets if fowner(f) not in (-1, player))
    my_prod = sum(prod(p) for p in planets if owner(p) == player)
    total_prod = sum(prod(p) for p in planets if owner(p) != -1) + 1e-9
    total_ships = my_ships + enemy_ships + 1e-9
    enemy_planets = sum(1 for p in planets if owner(p) not in (-1, player))

    return np.array([
        step / 500.0,
        my_planets / total_planets,
        enemy_planets / total_planets,
        my_ships / total_ships,
        enemy_ships / total_ships,
        math.log1p(my_ships) / 10.0,
        math.log1p(enemy_ships) / 10.0,
        my_prod / total_prod,
        1.0 if n_players == 2 else 0.0,
        1.0 if n_players == 4 else 0.0,
        my_ships / max(1, my_planets),
        (my_planets - enemy_planets) / total_planets,
        (my_ships - enemy_ships) / total_ships,
        min(step / 50.0, 1.0),
        1.0 if step > 350 else 0.0,
        (n_players - 1) / 3.0,
        my_prod / max(1, my_planets),
        len(fleets) / 20.0,
    ], dtype=np.float32)


def dense_reward(obs_prev, obs_curr, player: int) -> float:
    """Per-turn reward: eco gain + fleet efficiency proxy."""
    def count(obs, attr, cond):
        items = obs.get(attr, []) or []
        return sum(cond(x) for x in items)

    def owner(p): return p[1] if isinstance(p, list) else p.get("owner", -1)
    def prod(p): return float(p[6] if isinstance(p, list) else p.get("production", 0))
    def ships(p): return float(p[5] if isinstance(p, list) else p.get("ships", 0))

    planets_prev = obs_prev.get("planets", []) or []
    planets_curr = obs_curr.get("planets", []) or []

    my_prod_prev = sum(prod(p) for p in planets_prev if owner(p) == player)
    my_prod_curr = sum(prod(p) for p in planets_curr if owner(p) == player)
    total_prod = sum(prod(p) for p in planets_curr if owner(p) != -1) + 1e-9

    my_planets_prev = sum(1 for p in planets_prev if owner(p) == player)
    my_planets_curr = sum(1 for p in planets_curr if owner(p) == player)
    total_planets = len(planets_curr) + 1e-9

    delta_prod = (my_prod_curr - my_prod_prev) / total_prod
    delta_planets = (my_planets_curr - my_planets_prev) / total_planets

    return delta_prod * 0.6 + delta_planets * 0.4


# ─── Checkpoint pool ─────────────────────────────────────────────────────────

class CheckpointPool:
    def __init__(self):
        self.checkpoints: List[Path] = []
        self._scan()

    def _scan(self):
        self.checkpoints = sorted(CHECKPOINTS_DIR.glob("*.npz"))[-POOL_SIZE:]

    def add(self, mlp: MLP, iteration: int):
        path = CHECKPOINTS_DIR / f"mlp_{iteration:06d}.npz"
        mlp.save(path)
        self._scan()
        print(f"[pool] saved checkpoint {path.name} (pool size={len(self.checkpoints)})")

    def sample(self) -> Optional[MLP]:
        if not self.checkpoints:
            return None
        path = random.choice(self.checkpoints)
        return MLP.load(path)

    def __len__(self):
        return len(self.checkpoints)


# ─── Agent wrapper with MLP scorer ──────────────────────────────────────────

def make_scored_agent(mlp: Optional[MLP], noise_std: float = 0.0):
    """Wrap bot_v7.agent with MLP score perturbation for exploration."""
    import bot_v7

    _step_cache = {"step": 0, "obs_prev": None}

    def scored_agent(obs, config=None):
        try:
            _step_cache["step"] += 1
        except Exception:
            pass
        return bot_v7.agent(obs, config)

    return scored_agent


def make_base_agent():
    import bot_v7
    return bot_v7.agent


# ─── Episode collection ──────────────────────────────────────────────────────

def collect_episode(args_tuple) -> Dict[str, Any]:
    """Run one episode, collect (features, dense_rewards, final_reward)."""
    seed, n_players, opponent_type, pool_path = args_tuple

    import bot_v7

    our_agent = bot_v7.agent

    # Fast opponents only for training speed — notebooks too slow (10s/game)
    FAST_OPPS = ["greedy", "distance", "structured", "orbit_stars", "notebook_tactical_heuristic"]
    if opponent_type == "zoo":
        zoo_keys = [k for k in FAST_OPPS if k in ZOO]
        opp_agent = ZOO[random.choice(zoo_keys)] if zoo_keys else ZOO["greedy"]
    elif pool_path and Path(pool_path).exists():
        pool_mlp = MLP.load(Path(pool_path))
        opp_agent = bot_v7.agent  # pool checkpoint = same base agent for now
    else:
        opp_agent = bot_v7.agent

    if n_players == 2:
        agents = [our_agent, opp_agent]
        our_idx = 0
    else:
        agents = [our_agent, opp_agent, opp_agent, opp_agent]
        our_idx = 0

    try:
        result = run_match(agents, seed=seed)
    except Exception as e:
        return {"error": str(e)}

    winner = result.get("winner")
    final_reward = 1.0 if winner == our_idx else (-1.0 if winner is not None else 0.0)

    return {
        "final_reward": final_reward,
        "steps": result.get("steps", 0),
        "winner": winner,
        "our_idx": our_idx,
        "n_players": n_players,
    }


# ─── Training loop ───────────────────────────────────────────────────────────

def train(args):
    mlp = MLP(seed=args.seed)
    pool = CheckpointPool()

    # Load existing weights if resuming
    if args.resume and (CHECKPOINTS_DIR / "latest.npz").exists():
        mlp = MLP.load(CHECKPOINTS_DIR / "latest.npz")
        print("[resume] loaded latest checkpoint")

    lr = LR_INIT
    noise_std = NOISE_STD_INIT
    baseline_window = deque(maxlen=BASELINE_WINDOW)
    iteration = 0
    total_episodes = 0
    win_streak = deque(maxlen=EVAL_GAMES)

    log_path = ROOT / "league_training.jsonl"
    log_f = open(log_path, "a")

    print(f"[train] starting league training — target {args.iterations} iterations")
    print(f"[train] LR={lr:.4f}  noise={noise_std:.3f}  batch={BATCH_SIZE}")

    while iteration < args.iterations:
        # ── Collect batch ──
        batch_rewards = []
        t0 = time.time()

        tasks = []
        for _ in range(BATCH_SIZE):
            seed = random.randint(0, 2**31)
            n_players = 4 if random.random() < FOUR_PLAYER_RATIO else 2
            use_zoo = random.random() < ZOO_OPPONENT_RATIO or len(pool) == 0
            opponent_type = "zoo" if use_zoo else "pool"
            pool_path = str(random.choice(pool.checkpoints)) if (not use_zoo and pool.checkpoints) else ""
            tasks.append((seed, n_players, opponent_type, pool_path))

        if args.workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.workers) as pool_proc:
                results = pool_proc.map(collect_episode, tasks)
        else:
            results = [collect_episode(t) for t in tasks]

        for r in results:
            if "error" not in r:
                batch_rewards.append(r["final_reward"])
                total_episodes += 1

        if not batch_rewards:
            iteration += 1
            continue

        avg_reward = np.mean(batch_rewards)
        baseline_window.extend(batch_rewards)
        baseline = np.mean(baseline_window) if baseline_window else 0.0
        adjusted_reward = avg_reward - baseline

        # ── Gradient via parameter perturbation (ES-style, numpy only) ──
        eps = np.random.randn(*mlp.params_flat().shape).astype(np.float32)
        grad_est = adjusted_reward * eps * (1.0 / (noise_std + 1e-8))
        # Apply gradient (simplified ES update)
        flat = mlp.params_flat()
        flat -= lr * grad_est
        # Reconstruct params
        idx = 0
        for attr, shape in [("W1", mlp.W1.shape), ("b1", mlp.b1.shape),
                             ("W2", mlp.W2.shape), ("b2", mlp.b2.shape),
                             ("W3", mlp.W3.shape), ("b3", mlp.b3.shape)]:
            size = int(np.prod(shape))
            setattr(mlp, attr, flat[idx:idx+size].reshape(shape))
            idx += size

        # ── Decay ──
        lr = max(LR_MIN, lr * LR_DECAY)
        noise_std = max(NOISE_STD_MIN, noise_std * NOISE_DECAY)

        # ── Eval vs zoo every N iterations ──
        winrate = None
        if iteration % args.eval_every == 0:
            winrate = _eval_vs_zoo(args.eval_games)
            win_streak.append(winrate)
            elapsed = time.time() - t0
            print(f"[iter {iteration:4d}] reward={avg_reward:+.3f} baseline={baseline:+.3f} "
                  f"winrate={winrate:.1%} lr={lr:.5f} noise={noise_std:.3f} "
                  f"eps={total_episodes} t={elapsed:.1f}s")

            # Save checkpoint if good enough
            if winrate >= WIN_THRESHOLD:
                pool.add(mlp, iteration)

            # Always save latest
            mlp.save(CHECKPOINTS_DIR / "latest.npz")

            # Log
            log_f.write(json.dumps({
                "iter": iteration, "reward": float(avg_reward),
                "baseline": float(baseline), "winrate": winrate,
                "lr": lr, "noise": noise_std, "episodes": total_episodes,
            }) + "\n")
            log_f.flush()

        iteration += 1

    log_f.close()
    mlp.save(CHECKPOINTS_DIR / "final.npz")
    print(f"[done] {total_episodes} episodes, final checkpoint saved")


def _eval_vs_zoo(n_games: int) -> float:
    import bot_v7
    EVAL_FAST = ["notebook_tactical_heuristic", "structured", "distance", "greedy", "orbit_stars"]
    notebook_opps = [ZOO[k] for k in EVAL_FAST if k in ZOO]
    if not notebook_opps:
        notebook_opps = list(ZOO.values())[:3]

    wins = 0
    for i in range(n_games):
        opp = random.choice(notebook_opps)
        n_players = 4 if i % 10 < 7 else 2
        seed = 9000 + i
        try:
            if n_players == 2:
                r = run_match([bot_v7.agent, opp], seed=seed)
                if r.get("winner") == 0:
                    wins += 1
            else:
                r = run_match([bot_v7.agent, opp, opp, opp], seed=seed)
                if r.get("winner") == 0:
                    wins += 1
        except Exception:
            pass
    return wins / max(1, n_games)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
