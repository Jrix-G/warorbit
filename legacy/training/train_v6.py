"""Offline self-play training utilities for Orbit Wars V6."""

import argparse
import os

import numpy as np
from kaggle_environments import make

from eval import NumpyEvaluator, encode_evaluator, extract_features
from search import beam_search
from sim import state_from_obs


def make_v6_agent(evaluator, time_budget=0.85):
    """Return a Kaggle-compatible agent using V6 beam search."""
    def agent(obs, config=None):
        state = state_from_obs(obs)
        if len(state.planets) == 0:
            return []
        return beam_search(state, time_budget=time_budget, evaluator=evaluator)
    return agent


def _winner_from_env(env, n_players):
    rewards = [env.steps[-1][i].reward for i in range(n_players)]
    best = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == best]
    return winners[0] if len(winners) == 1 else -1


def generate_self_play_games(evaluator, n_games=50, n_jobs=1, time_budget=0.35):
    """Generate [(features_24, outcome_01), ...] from V6 self-play games."""
    del n_jobs  # Kept in the API for future multiprocessing.
    dataset = []

    for _ in range(n_games):
        recorded = []

        def make_recording_agent(player_eval):
            def agent(obs, config=None):
                state = state_from_obs(obs)
                if len(state.planets) > 0:
                    player = int(getattr(obs, "player", obs.get("player", 0) if isinstance(obs, dict) else 0))
                    recorded.append((state.copy(), player))
                    return beam_search(state, time_budget=time_budget, evaluator=player_eval)
                return []
            return agent

        env = make("orbit_wars", debug=False)
        env.run([make_recording_agent(evaluator), make_recording_agent(evaluator)])
        winner = _winner_from_env(env, 2)

        for state, player in recorded:
            outcome = 1.0 if winner == player else 0.0
            dataset.append((extract_features(state, player), outcome))

    return dataset


def train_evaluator_backprop(evaluator, dataset, epochs=100, lr=0.001, batch_size=64):
    """Train NumpyEvaluator with manual SGD backprop on BCE loss."""
    if not dataset:
        return evaluator

    X = np.array([d[0] for d in dataset], dtype=np.float32)
    y = np.array([d[1] for d in dataset], dtype=np.float32)

    for epoch in range(epochs):
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]

        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            yb = y[i:i + batch_size]
            if len(xb) == 0:
                continue

            h1_pre = xb @ evaluator.W1 + evaluator.b1
            h1 = np.maximum(0.0, h1_pre)
            h2_pre = h1 @ evaluator.W2 + evaluator.b2
            h2 = np.maximum(0.0, h2_pre)
            logit = (h2 @ evaluator.W3 + evaluator.b3)[:, 0]
            logit = np.clip(logit, -40.0, 40.0)
            pred = 1.0 / (1.0 + np.exp(-logit))

            loss = -np.mean(yb * np.log(pred + 1e-8) + (1.0 - yb) * np.log(1.0 - pred + 1e-8))
            total_loss += float(loss)
            n_batches += 1

            d_logit = (pred - yb) / len(xb)

            d_W3 = h2.T @ d_logit[:, np.newaxis]
            d_b3 = np.array([d_logit.sum()], dtype=np.float32)
            d_h2 = d_logit[:, np.newaxis] @ evaluator.W3.T

            d_h2_relu = d_h2 * (h2_pre > 0.0)
            d_W2 = h1.T @ d_h2_relu
            d_b2 = d_h2_relu.sum(axis=0)
            d_h1 = d_h2_relu @ evaluator.W2.T

            d_h1_relu = d_h1 * (h1_pre > 0.0)
            d_W1 = xb.T @ d_h1_relu
            d_b1 = d_h1_relu.sum(axis=0)

            evaluator.W1 -= lr * d_W1.astype(np.float32)
            evaluator.b1 -= lr * d_b1.astype(np.float32)
            evaluator.W2 -= lr * d_W2.astype(np.float32)
            evaluator.b2 -= lr * d_b2.astype(np.float32)
            evaluator.W3 -= lr * d_W3.astype(np.float32)
            evaluator.b3 -= lr * d_b3.astype(np.float32)

        if epoch % 10 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch}: loss={avg_loss:.4f}")

    return evaluator


def train_evaluator_cma(evaluator, dataset, n_generations=50):
    """Train with CMA-ES. Requires local `cma` package."""
    import cma

    if not dataset:
        return evaluator

    X = np.array([d[0] for d in dataset], dtype=np.float32)
    y = np.array([d[1] for d in dataset], dtype=np.float32)

    def loss(params):
        evaluator.set_params(np.array(params, dtype=np.float32))
        preds = evaluator.predict_batch(X)
        bce = -np.mean(y * np.log(preds + 1e-8) + (1.0 - y) * np.log(1.0 - preds + 1e-8))
        return float(bce)

    es = cma.CMAEvolutionStrategy(
        evaluator.get_params().tolist(),
        0.1,
        {"maxiter": n_generations, "verbose": -9},
    )
    es.optimize(loss)
    evaluator.set_params(np.array(es.result.xbest, dtype=np.float32))
    return evaluator


class League:
    """Historical evaluator snapshots for diversified evaluation."""

    def __init__(self, max_size=10):
        self.snapshots = []
        self.max_size = max_size

    def add_snapshot(self, evaluator, elo=1000):
        self.snapshots.append((evaluator.get_params().copy(), float(elo)))
        if len(self.snapshots) > self.max_size:
            self.snapshots.sort(key=lambda x: -x[1])
            self.snapshots = self.snapshots[:self.max_size]

    def sample_opponent(self):
        if not self.snapshots:
            return None
        params, _ = self.snapshots[np.random.randint(len(self.snapshots))]
        opp = NumpyEvaluator()
        opp.set_params(params.copy())
        return opp


def evaluate_vs_league(evaluator, league, n_games=10, time_budget=0.35):
    """Return win rate of evaluator against sampled league snapshots."""
    if not league.snapshots:
        return 1.0

    wins = 0
    played = 0
    for i in range(n_games):
        opponent = league.sample_opponent()
        if opponent is None:
            continue

        env = make("orbit_wars", debug=False)
        if i % 2 == 0:
            env.run([make_v6_agent(evaluator, time_budget), make_v6_agent(opponent, time_budget)])
            if env.steps[-1][0].reward > env.steps[-1][1].reward:
                wins += 1
        else:
            env.run([make_v6_agent(opponent, time_budget), make_v6_agent(evaluator, time_budget)])
            if env.steps[-1][1].reward > env.steps[-1][0].reward:
                wins += 1
        played += 1

    return wins / max(played, 1)


def _export_evaluator(evaluator, npy_path="best_evaluator.npy", b64_path="evaluator_b64.txt"):
    evaluator.save(npy_path)
    b64 = encode_evaluator(evaluator)
    with open(b64_path, "w", encoding="ascii") as f:
        f.write(b64)
    print(f"  Export: {npy_path} + {b64_path} ({len(b64)} chars)")


def train_v6(n_iterations=20, games_per_iter=50, n_jobs=1, epochs=50,
             lr=0.001, batch_size=64, self_play_budget=0.35,
             eval_games=10, eval_budget=0.35, save_threshold=0.6,
             seed=42):
    """Main V6 self-play loop."""
    evaluator = NumpyEvaluator(seed=seed)
    league = League()
    os.makedirs("checkpoints", exist_ok=True)

    best_win_rate = 0.0
    for iteration in range(n_iterations):
        print(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")
        print("Generation des parties self-play...")
        dataset = generate_self_play_games(
            evaluator,
            games_per_iter,
            n_jobs,
            time_budget=self_play_budget,
        )
        print(f"  {len(dataset)} etats collectes")

        print("Entrainement evaluateur...")
        train_evaluator_backprop(
            evaluator,
            dataset,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )

        win_rate = evaluate_vs_league(
            evaluator,
            league,
            n_games=eval_games,
            time_budget=eval_budget,
        )
        print(f"  Win rate vs league: {win_rate:.1%}")

        elo_estimate = 800.0 + win_rate * 800.0
        league.add_snapshot(evaluator, elo_estimate)
        evaluator.save(f"checkpoints/eval_iter_{iteration:03d}.npy")

        if win_rate > save_threshold and win_rate >= best_win_rate:
            best_win_rate = win_rate
            _export_evaluator(evaluator)
            print(f"  Nouveau meilleur sauvegarde (win_rate={win_rate:.1%})")

    return evaluator


def _profile_defaults(profile):
    if profile == "smoke":
        return {
            "n_iterations": 1,
            "games_per_iter": 1,
            "epochs": 2,
            "self_play_budget": 0.08,
            "eval_games": 0,
            "eval_budget": 0.08,
            "save_threshold": 0.0,
        }
    if profile == "rocket":
        return {
            "n_iterations": 8,
            "games_per_iter": 16,
            "epochs": 30,
            "self_play_budget": 0.25,
            "eval_games": 6,
            "eval_budget": 0.20,
            "save_threshold": 0.55,
        }
    return {
        "n_iterations": 20,
        "games_per_iter": 50,
        "epochs": 50,
        "self_play_budget": 0.35,
        "eval_games": 10,
        "eval_budget": 0.35,
        "save_threshold": 0.60,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Orbit Wars V6 evaluator.")
    parser.add_argument("--profile", choices=("smoke", "rocket", "full"), default="full")
    parser.add_argument("--rocket", action="store_true", help="Shortcut for --profile rocket.")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for --profile smoke.")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--games-per-iter", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--self-play-budget", type=float)
    parser.add_argument("--eval-games", type=int)
    parser.add_argument("--eval-budget", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-final", action="store_true",
                        help="Always export final evaluator, even if threshold was not reached.")
    args = parser.parse_args()

    profile = "rocket" if args.rocket else "smoke" if args.smoke else args.profile
    cfg = _profile_defaults(profile)
    if args.iterations is not None:
        cfg["n_iterations"] = args.iterations
    if args.games_per_iter is not None:
        cfg["games_per_iter"] = args.games_per_iter
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.self_play_budget is not None:
        cfg["self_play_budget"] = args.self_play_budget
    if args.eval_games is not None:
        cfg["eval_games"] = args.eval_games
    if args.eval_budget is not None:
        cfg["eval_budget"] = args.eval_budget

    print(f"Profile: {profile}")
    print(
        "Config: "
        f"iterations={cfg['n_iterations']}, games_per_iter={cfg['games_per_iter']}, "
        f"epochs={cfg['epochs']}, self_play_budget={cfg['self_play_budget']}, "
        f"eval_games={cfg['eval_games']}"
    )
    evaluator = train_v6(
        n_iterations=cfg["n_iterations"],
        games_per_iter=cfg["games_per_iter"],
        epochs=cfg["epochs"],
        lr=args.lr,
        batch_size=args.batch_size,
        self_play_budget=cfg["self_play_budget"],
        eval_games=cfg["eval_games"],
        eval_budget=cfg["eval_budget"],
        save_threshold=cfg["save_threshold"],
        seed=args.seed,
    )
    if args.export_final:
        _export_evaluator(evaluator, "final_evaluator.npy", "final_evaluator_b64.txt")


if __name__ == "__main__":
    main()
