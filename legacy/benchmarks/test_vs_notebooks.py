#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test bot_submit.py against extracted notebook agents
"""

import os
import sys
import io

if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from kaggle_environments import make
import bot_submit
from opponents import ZOO

TEST_OPPONENTS = [
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
    "notebook_physics_accurate",
    "notebook_tactical_heuristic",
]

def test_vs_opponent(opponent_name, n_games=5):
    """Test bot_submit against an opponent"""
    if opponent_name not in ZOO:
        print(f"  Opponent not found: {opponent_name}")
        return None

    opponent_agent = ZOO[opponent_name]
    wins = 0

    print(f"Testing vs {opponent_name}...")

    for game_num in range(n_games):
        try:
            env = make("orbit-wars", configuration={"episodeSteps": 500})

            agents = [bot_submit.agent, opponent_agent]
            for step in env.step(agents):
                pass

            # Check who won (higher final ship count wins)
            final_state = env.state[-1]
            observations = [s[0]['observation'] for s in final_state]

            our_ships = sum(p['ships'] for p in observations[0].get('planets', []))
            opp_ships = sum(p['ships'] for p in observations[1].get('planets', []))

            for fleet in observations[0].get('fleets', []):
                if fleet['owner'] == 0:
                    our_ships += fleet['ships']

            for fleet in observations[1].get('fleets', []):
                if fleet['owner'] == 1:
                    opp_ships += fleet['ships']

            if our_ships > opp_ships:
                wins += 1
                status = "WIN"
            else:
                status = "LOSS"

            print(f"    Game {game_num + 1}: {status} (us: {our_ships:.0f}, opp: {opp_ships:.0f})")

        except Exception as e:
            print(f"    Game {game_num + 1}: ERROR - {str(e)[:50]}")

    win_rate = (wins / n_games * 100) if n_games > 0 else 0
    print(f"  Result: {wins}W/{n_games} (wr={win_rate:.0f}%)")

    return win_rate

def main():
    print("\n" + "="*80)
    print("TESTING bot_submit.py VS NOTEBOOK AGENTS")
    print("="*80)

    print(f"\nAvailable opponents in ZOO: {list(ZOO.keys())}")

    print("\n" + "-"*80)
    print("TESTING...")
    print("-"*80)

    results = {}
    for opponent in TEST_OPPONENTS:
        print(f"\n{opponent}:")
        wr = test_vs_opponent(opponent, n_games=3)
        if wr is not None:
            results[opponent] = wr

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    if results:
        for opponent, wr in results.items():
            status = "GOOD" if wr >= 50 else "BAD"
            print(f"  {opponent:35s} : {wr:5.0f}% [{status}]")

        avg_wr = sum(results.values()) / len(results)
        print(f"\n  Average win rate: {avg_wr:.0f}%")
    else:
        print("  No results (agents might have import errors)")

if __name__ == "__main__":
    main()
