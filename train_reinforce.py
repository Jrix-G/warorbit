"""REINFORCE self-play training for bot_v7 mission scorer.

Architecture: linear scorer (w @ features), warm-start w=0 = pure V7 behavior.
Algorithm: batched REINFORCE with baseline, self-play + ZOO mix.
"""

import sys, math, time, random, argparse, collections
import numpy as np

sys.path.insert(0, ".")
import bot_v7
from SimGame import run_match
from opponents import ZOO

FEAT_DIM = bot_v7.MISSION_FEATURE_DIM  # 15

ZOO_OPPONENTS = ["notebook_orbitbotnext", "notebook_distance_prioritized", "notebook_physics_accurate"]

# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class LinearScorer:
    """Linear log-multiplier: score_final = base * exp(w @ features)."""

    def __init__(self):
        self.w = np.zeros(FEAT_DIM, dtype=np.float64)

    def __call__(self, features):
        return float(self.w @ features.astype(np.float64))

    def update(self, delta):
        self.w += delta

    def save(self, path):
        np.save(path, self.w)

    def load(self, path):
        self.w = np.load(path).astype(np.float64)


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def collect_episode(scorer, noise_std, seed, opponent=None):
    """Play one game. Returns (log, reward, steps, won).

    opponent=None => self-play. Otherwise vs that opponent (alternating sides).
    """
    flip = (seed % 2 == 1)

    if opponent is not None:
        if not flip:
            bot_v7.set_scorer(scorer, noise_std=noise_std, log_player=0)
            bot_v7.reset_episode_log()
            result = run_match([bot_v7.agent, opponent], seed=seed)
            reward = 1.0 if result["winner"] == 0 else (-1.0 if result["winner"] == 1 else 0.0)
        else:
            bot_v7.set_scorer(scorer, noise_std=noise_std, log_player=1)
            bot_v7.reset_episode_log()
            result = run_match([opponent, bot_v7.agent], seed=seed)
            reward = 1.0 if result["winner"] == 1 else (-1.0 if result["winner"] == 0 else 0.0)
    else:
        # Self-play: log player 0 only
        bot_v7.set_scorer(scorer, noise_std=noise_std, log_player=0)
        bot_v7.reset_episode_log()
        result = run_match([bot_v7.agent, bot_v7.agent], seed=seed)
        reward = 1.0 if result["winner"] == 0 else (-1.0 if result["winner"] == 1 else 0.0)

    log = bot_v7.pop_episode_log()
    bot_v7.set_scorer(None)
    return log, reward, result.get("steps", 0), reward > 0


# ---------------------------------------------------------------------------
# REINFORCE update
# ---------------------------------------------------------------------------

def reinforce_update(scorer, batch, lr, l2):
    """Batched REINFORCE with mean-reward baseline. Returns gradient norm."""
    if not batch:
        return 0.0

    rewards = [r for (_, r, _, _) in batch]
    baseline = sum(rewards) / len(rewards)

    grad = np.zeros(FEAT_DIM, dtype=np.float64)
    n = 0

    for (log, reward, _, _) in batch:
        adj = reward - baseline
        if abs(adj) < 1e-9:
            continue
        for (features, noise) in log:
            grad += adj * noise * features.astype(np.float64)
            n += 1

    if n > 0:
        grad /= n

    grad -= l2 * scorer.w
    scorer.update(lr * grad)
    return float(np.linalg.norm(grad))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(scorer, n_games=6, seed_offset=300):
    """Benchmark vs 3 notebooks (no noise). Returns {name: (wins, total)}."""
    results = {}
    for idx, opp_name in enumerate(ZOO_OPPONENTS):
        opp = ZOO[opp_name]
        wins = 0
        for i in range(n_games):
            s = seed_offset + i * 7 + idx * 100
            if i % 2 == 0:
                bot_v7.set_scorer(scorer, noise_std=0.0, log_player=0)
                bot_v7.reset_episode_log()
                r = run_match([bot_v7.agent, opp], seed=s)
                wins += 1 if r["winner"] == 0 else 0
            else:
                bot_v7.set_scorer(scorer, noise_std=0.0, log_player=1)
                bot_v7.reset_episode_log()
                r = run_match([opp, bot_v7.agent], seed=s)
                wins += 1 if r["winner"] == 1 else 0
            bot_v7.set_scorer(None)
        results[opp_name] = (wins, n_games)
    return results


def print_benchmark(results, label=""):
    total_w = sum(w for w, _ in results.values())
    total_n = sum(n for _, n in results.values())
    tag = f" [{label}]" if label else ""
    print(f"Benchmark{tag}: {total_w}/{total_n} overall", flush=True)
    for name, (w, n) in results.items():
        short = name.replace("notebook_", "").replace("_", "-")[:22]
        print(f"  {short:<22} {w}/{n}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--noise-start", type=float, default=0.3)
    ap.add_argument("--noise-end", type=float, default=0.05)
    ap.add_argument("--zoo-ratio", type=float, default=0.4,
                    help="Fraction of episodes vs ZOO opponents (rest = self-play)")
    ap.add_argument("--benchmark-every", type=int, default=50)
    ap.add_argument("--benchmark-games", type=int, default=6)
    ap.add_argument("--out", type=str, default="evaluations/reinforce_weights.npy")
    ap.add_argument("--load", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    scorer = LinearScorer()
    if args.load:
        scorer.load(args.load)
        print(f"Loaded weights from {args.load}", flush=True)

    print(f"REINFORCE -- {args.episodes} eps  batch={args.batch_size}  "
          f"lr={args.lr}  l2={args.l2}  zoo={args.zoo_ratio}", flush=True)
    print(f"noise {args.noise_start:.2f} -> {args.noise_end:.2f}  out={args.out}", flush=True)
    print()

    # Baseline benchmark (w=0 => pure V7+AoW)
    if not args.load:
        print("=== Baseline (w=0, pure V7+AoW) ===", flush=True)
        t0 = time.time()
        res = benchmark(scorer, n_games=args.benchmark_games)
        print_benchmark(res, "baseline")
        print(f"  {time.time()-t0:.1f}s\n", flush=True)

    zoo_agents = [ZOO[n] for n in ZOO_OPPONENTS]
    batch = []
    t_start = time.time()

    for ep in range(args.episodes):
        frac = ep / max(1, args.episodes - 1)
        noise_std = args.noise_start + frac * (args.noise_end - args.noise_start)

        use_zoo = rng.random() < args.zoo_ratio
        opponent = rng.choice(zoo_agents) if use_zoo else None

        seed = args.seed * 10000 + ep
        log, reward, steps, won = collect_episode(scorer, noise_std, seed, opponent)
        batch.append((log, reward, steps, won))

        if len(batch) >= args.batch_size:
            grad_norm = reinforce_update(scorer, batch, args.lr, args.l2)
            wins_b = sum(1 for (_, _, _, w) in batch if w)
            avg_r = sum(r for (_, r, _, _) in batch) / len(batch)
            avg_steps = sum(s for (_, _, s, _) in batch) / len(batch)
            avg_log = sum(len(l) for (l, _, _, _) in batch) / len(batch)
            elapsed = time.time() - t_start
            print(f"ep {ep+1:4d}/{args.episodes}  wins={wins_b}/{len(batch)}"
                  f"  avg_r={avg_r:+.3f}  grad={grad_norm:.4f}"
                  f"  steps={avg_steps:.0f}  log/ep={avg_log:.0f}"
                  f"  noise={noise_std:.3f}  t={elapsed:.0f}s", flush=True)
            batch = []

        if (ep + 1) % args.benchmark_every == 0:
            print(f"\n=== Benchmark ep {ep+1} ===", flush=True)
            t0 = time.time()
            res = benchmark(scorer, n_games=args.benchmark_games)
            print_benchmark(res, f"ep{ep+1}")
            print(f"  {time.time()-t0:.1f}s  w={np.round(scorer.w, 3)}", flush=True)
            scorer.save(args.out)
            print(f"  saved -> {args.out}\n", flush=True)

    scorer.save(args.out)
    elapsed = time.time() - t_start
    print(f"\nDone. {args.episodes} eps in {elapsed:.0f}s. Saved -> {args.out}", flush=True)
    print(f"Final w: {np.round(scorer.w, 4)}", flush=True)

    print("\n=== Final benchmark ===", flush=True)
    res = benchmark(scorer, n_games=args.benchmark_games)
    print_benchmark(res, "final")


if __name__ == "__main__":
    main()
