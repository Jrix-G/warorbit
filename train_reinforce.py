"""ES training for bot_v7 mission scorer.

Weight-perturbation ES (OpenAI ES 2017):
  - Sample eps ~ N(0, I) per episode
  - Run game with (w + sigma*eps) @ features as scorer (deterministic)
  - Update: w += rank_normalized(R) * eps / (N*sigma)

Advantage over per-action noise:
  - Proper gradient normalization (no 1/sigma^2 issue)
  - No credit-assignment noise across 1000+ actions
  - Weights move visibly after each batch

Run: python3 train_reinforce.py [--hours 1] [--workers 4]
     python3 train_reinforce.py --resume evaluations/scorer_es.npy
"""

import sys
import os
import time
import random
import argparse
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

FEATURE_DIM = 15       # must match bot_v7.MISSION_FEATURE_DIM
SIGMA = 0.18           # perturbation magnitude (weight space)
LR = 0.015             # learning rate
BATCH_SIZE = 32        # episodes per gradient update; forced even for mirrored ES
WEIGHT_DECAY = 0.002
MAX_WEIGHT_ABS = 1.25
BENCHMARK_EVERY = 250
BENCHMARK_GAMES = 5
SAVE_EVERY = 250
OUT_DIR = "evaluations"
DEFAULT_HOURS = 1.0
ZOO_OPPONENTS = [
    "notebook_physics_accurate",
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
]

FEAT_NAMES = [
    "prod", "eta", "rem_turns", "is_2p", "domination",
    "is_enemy", "is_static", "indirect", "my_prod_ratio",
    "is_early", "is_late", "capture", "snipe", "swarm", "reinforce",
]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _shaped_reward(result, our_index):
    """Dense reward: win/loss plus score margin, bounded to roughly [-1, 1]."""
    winner = result.get("winner", -1)
    scores = list(result.get("scores", []) or [])

    if winner == our_index:
        win_term = 1.0
    elif winner == -1:
        win_term = 0.0
    else:
        win_term = -1.0

    if scores and 0 <= our_index < len(scores):
        our_score = float(scores[our_index])
        opp_best = max(float(s) for i, s in enumerate(scores) if i != our_index)
        scale = max(1.0, sum(abs(float(s)) for s in scores))
        margin_term = float(np.tanh(5.0 * (our_score - opp_best) / scale))
    else:
        margin_term = 0.0

    return 0.75 * win_term + 0.25 * margin_term


def collect_episode_worker(args):
    """Run one mirrored-ES game with perturbed weights, return (signed_eps, reward)."""
    seed, weights, sigma, eps_seed, sign, zoo_name, seat = args

    import bot_v7
    from SimGame import run_match

    rng = np.random.RandomState(eps_seed)
    eps = (float(sign) * rng.randn(FEATURE_DIM)).astype(np.float32)
    w_perturbed = np.array(weights, dtype=np.float32) + sigma * eps

    def scorer(features):
        return float(w_perturbed @ features)

    bot_v7.set_scorer(scorer, noise_std=0.0, log_player=-1)

    try:
        from opponents import ZOO
        opp = ZOO[zoo_name]
        if seat == 0:
            result = run_match([bot_v7.agent, opp], seed=seed)
            reward = _shaped_reward(result, 0)
        else:
            result = run_match([opp, bot_v7.agent], seed=seed)
            reward = _shaped_reward(result, 1)
    except Exception:
        reward = 0.0
        eps = np.zeros(FEATURE_DIM, dtype=np.float32)
    finally:
        bot_v7.set_scorer(None)

    return eps.tolist(), reward


# ---------------------------------------------------------------------------
# ES gradient update
# ---------------------------------------------------------------------------

def es_update(weights, batch_results, lr, sigma):
    """Rank-normalized mirrored ES update."""
    grad = np.zeros_like(weights)
    n = len(batch_results)
    rewards = np.array([reward for _, reward in batch_results], dtype=np.float32)
    advantages = rewards - float(np.mean(rewards))
    std = float(np.std(advantages))
    if std > 1e-6:
        advantages /= std

    for (eps_list, _), advantage in zip(batch_results, advantages):
        eps = np.array(eps_list, dtype=np.float32)
        grad += float(advantage) * eps

    updated = weights * (1.0 - WEIGHT_DECAY) + (lr / max(n * sigma, 1e-8)) * grad
    return np.clip(updated, -MAX_WEIGHT_ABS, MAX_WEIGHT_ABS).astype(np.float32)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(weights, games=BENCHMARK_GAMES):
    import bot_v7
    from SimGame import run_match
    from opponents import ZOO

    w = np.array(weights, dtype=np.float32)

    def clean_scorer(features):
        return float(w @ features)

    results = {}
    for opp_name in ZOO_OPPONENTS:
        opp = ZOO[opp_name]
        wins = 0
        print(f"  Benchmarking vs {opp_name}...", flush=True)
        for i in range(games):
            bot_v7.set_scorer(clean_scorer, noise_std=0.0, log_player=-1)
            try:
                if i % 2 == 0:
                    r = run_match([bot_v7.agent, opp], seed=2000 + i)
                    wins += 1 if r["winner"] == 0 else 0
                else:
                    r = run_match([opp, bot_v7.agent], seed=2000 + i)
                    wins += 1 if r["winner"] == 1 else 0
            except Exception:
                pass
            finally:
                bot_v7.set_scorer(None)
        results[opp_name] = (wins, games)

    return results


def benchmark_score(results):
    wins = sum(wins for wins, _ in results.values())
    games = sum(total for _, total in results.values())
    return wins / max(games, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--benchmark-games", type=int, default=BENCHMARK_GAMES)
    parser.add_argument("--skip-initial-benchmark", action="store_true")
    parser.add_argument("--out", type=str, default=f"{OUT_DIR}/scorer_es.npy")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    if args.batch % 2:
        args.batch += 1

    os.makedirs(OUT_DIR, exist_ok=True)
    deadline = time.time() + args.hours * 3600.0

    if args.resume and os.path.exists(args.resume):
        weights = np.load(args.resume).astype(np.float32)
        print(f"Resumed from {args.resume}")
    else:
        weights = np.zeros(FEATURE_DIM, dtype=np.float32)
        print("Warm-start: zero weights (pure V7 behavior)")

    print(f"\nMethod: mirrored weight-perturbation ES + dense score reward")
    print(f"Config: workers={args.workers}  lr={args.lr}  sigma={args.sigma}  batch={args.batch}  hours={args.hours}")
    print("Starting training...", flush=True)
    print()

    best_score = -1.0
    best_weights = weights.copy()
    if not args.skip_initial_benchmark:
        print("[Initial benchmark (zero weights = pure V7)]", flush=True)
        bench = run_benchmark(weights, args.benchmark_games)
        for k, (wins, total) in bench.items():
            print(f"  {k}: {wins}/{total}")
        best_score = benchmark_score(bench)
        print(f"  global: {best_score:.3f}")
        print()

    episode = 0
    t_start = time.time()
    seed_base = int(time.time() * 1000) % 100000

    with mp.Pool(args.workers) as pool:
        while time.time() < deadline:
            print(f"[batch start ep {episode:5d}]", flush=True)
            tasks = []
            pair_count = args.batch // 2
            for i in range(pair_count):
                seed = seed_base + episode + i
                eps_seed = (seed * 6364136223846793005 + 1442695040888963407) % (2**32)
                zoo_name = ZOO_OPPONENTS[i % len(ZOO_OPPONENTS)]
                if random.random() < 0.35:
                    zoo_name = random.choice(ZOO_OPPONENTS)
                seat = (episode // args.batch + i) % 2
                tasks.append((seed, weights.tolist(), args.sigma, eps_seed, +1, zoo_name, seat))
                tasks.append((seed, weights.tolist(), args.sigma, eps_seed, -1, zoo_name, seat))

            batch_results = list(pool.imap_unordered(collect_episode_worker, tasks))

            rewards = [r for _, r in batch_results]
            weights = es_update(weights, batch_results, args.lr, args.sigma)

            episode += args.batch
            elapsed = time.time() - t_start
            win_rate = sum(1 for r in rewards if r > 0) / max(len(rewards), 1)
            avg_reward = float(np.mean(rewards)) if rewards else 0.0

            print(
                f"[ep {episode:5d} | {elapsed/60:5.1f}min]  "
                f"reward={avg_reward:+.3f}  pos%={win_rate:.2f}  "
                f"|w|={np.linalg.norm(weights):.4f}  "
                f"max={np.abs(weights).max():.4f}"
            )

            if episode % BENCHMARK_EVERY < args.batch:
                print(f"  [Benchmark @ ep {episode}]")
                bench = run_benchmark(weights, args.benchmark_games)
                for k, (wins, total) in bench.items():
                    print(f"    {k}: {wins}/{total}")
                score = benchmark_score(bench)
                print(f"    global          : {score:.3f}")
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                    np.save(args.out.replace(".npy", "_best.npy"), best_weights)
                    print(f"    New best saved  : {args.out.replace('.npy', '_best.npy')}")
                for name, val in zip(FEAT_NAMES, weights):
                    print(f"    {name:16s}: {val:+.4f}")

            if episode % SAVE_EVERY < args.batch:
                path = args.out.replace(".npy", f"_ep{episode}.npy")
                np.save(path, weights)
                print(f"  Saved: {path}")

    elapsed_total = time.time() - t_start
    if args.benchmark_games > 0:
        current_bench = run_benchmark(weights, args.benchmark_games)
        current_score = benchmark_score(current_bench)
        if current_score > best_score:
            best_score = current_score
            best_weights = weights.copy()
    elif episode > 0:
        best_weights = weights.copy()

    np.save(args.out, best_weights)
    print(f"\nDone. {episode} episodes in {elapsed_total/60:.1f}min")
    print(f"Saved best: {args.out}  score={best_score:.3f}")

    print("\n[Final benchmark]")
    bench = run_benchmark(best_weights, args.benchmark_games)
    for k, (wins, total) in bench.items():
        print(f"  {k}: {wins}/{total}")
    print(f"  global: {benchmark_score(bench):.3f}")
    print("\nFinal weights:")
    for name, val in zip(FEAT_NAMES, best_weights):
        print(f"  {name:16s}: {val:+.4f}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
