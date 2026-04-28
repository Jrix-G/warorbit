"""Benchmark Kaggle Orbit Wars vs local SimGame.

Usage examples:
    python3 benchmark_simgame.py --games 10 --max-steps 300
    python3 benchmark_simgame.py --skip-kaggle --games 100
"""

import argparse
import time

from opponents.baselines import greedy_agent, passive_agent, random_agent
from SimGame import (
    benchmark,
    benchmark_state_policies,
    fast_greedy_policy,
    passive_policy,
    run_match,
)


def run_kaggle_games(agent_a, agent_b, games, max_steps):
    from kaggle_environments import make

    started = time.perf_counter()
    steps = 0
    wins_a = wins_b = ties = 0

    for i in range(games):
        env = make("orbit_wars", configuration={"episodeSteps": max_steps}, debug=False)
        if i % 2 == 0:
            env.run([agent_a, agent_b])
            reward_a = env.steps[-1][0].reward or 0
            reward_b = env.steps[-1][1].reward or 0
        else:
            env.run([agent_b, agent_a])
            reward_b = env.steps[-1][0].reward or 0
            reward_a = env.steps[-1][1].reward or 0

        steps += len(env.steps)
        if reward_a > reward_b:
            wins_a += 1
        elif reward_b > reward_a:
            wins_b += 1
        else:
            ties += 1

    seconds = time.perf_counter() - started
    return {
        "games": games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "steps": steps,
        "seconds": seconds,
        "steps_per_second": steps / max(seconds, 1e-9),
        "seconds_per_game": seconds / max(games, 1),
    }


def add_seconds_per_game(result):
    result = dict(result)
    result["seconds_per_game"] = result["seconds"] / max(result["games"], 1)
    return result


def print_result(label, result, baseline=None):
    ratio = ""
    if baseline and result["seconds"] > 0:
        ratio = f" | speedup={baseline['seconds'] / result['seconds']:.2f}x"
    print(
        f"{label:24s} games={result['games']:4d} steps={result['steps']:6d} "
        f"time={result['seconds']:8.3f}s sec/game={result['seconds_per_game']:.4f} "
        f"steps/s={result['steps_per_second']:.1f}{ratio}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-kaggle", action="store_true")
    parser.add_argument(
        "--pair",
        choices=("greedy-passive", "greedy-random"),
        default="greedy-passive",
    )
    args = parser.parse_args()

    if args.pair == "greedy-random":
        agent_a, agent_b = greedy_agent, random_agent
        policy_a, policy_b = fast_greedy_policy, fast_greedy_policy
    else:
        agent_a, agent_b = greedy_agent, passive_agent
        policy_a, policy_b = fast_greedy_policy, passive_policy

    kaggle = None
    if not args.skip_kaggle:
        kaggle = run_kaggle_games(agent_a, agent_b, args.games, args.max_steps)
        print_result("Kaggle env", kaggle)

    sim_agents = benchmark(agent_a, agent_b, games=args.games, seed=args.seed)
    sim_agents = add_seconds_per_game(sim_agents)
    print_result("SimGame agents", sim_agents, kaggle)

    sim_policies = benchmark_state_policies(policy_a, policy_b, games=args.games, seed=args.seed)
    sim_policies = add_seconds_per_game(sim_policies)
    print_result("SimGame policies", sim_policies, kaggle or sim_agents)


if __name__ == "__main__":
    main()
