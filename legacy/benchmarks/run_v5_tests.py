#!/usr/bin/env python3
"""Run V5 against notebook agents using Kaggle environment"""

import sys
import os
import io

if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from kaggle_environments import make
except ImportError:
    print("ERROR: kaggle_environments not found")
    print("Install: pip install kaggle-environments --break-system-packages")
    sys.exit(1)

import bot_v5
from opponents import notebook_distance_prioritized, notebook_physics_accurate

def run_match(agent1, agent2, name1, name2, num_games=3):
    """Run matches between two agents"""
    print(f"\n{'='*80}")
    print(f"{name1} vs {name2} ({num_games} games)")
    print(f"{'='*80}")

    wins_p1 = 0
    wins_p2 = 0

    for game_num in range(num_games):
        env = make("orbit-wars", debug=False)
        agents_list = [agent1, agent2]

        try:
            result = env.run(agents_list)
            # Result: list of rewards for each player
            if result[0] > result[1]:
                wins_p1 += 1
                print(f"  Game {game_num + 1}: {name1} wins")
            else:
                wins_p2 += 1
                print(f"  Game {game_num + 1}: {name2} wins")
        except Exception as e:
            print(f"  Game {game_num + 1}: ERROR - {str(e)[:50]}")

    print(f"\nResults: {name1} {wins_p1}-{wins_p2} {name2}")
    return wins_p1, wins_p2

# === Run tests ===
print("Testing V5 Bot Against Notebook Agents")
print("="*80)

# Test 1: V5 vs Distance-Prioritized (1100 ELO)
v5_wins_dp, dp_wins = run_match(
    bot_v5.agent,
    notebook_distance_prioritized.agent,
    "V5",
    "Distance-Prioritized [1100]",
    num_games=3
)

# Test 2: V5 vs Physics-Accurate (928.7 ELO)
v5_wins_pa, pa_wins = run_match(
    bot_v5.agent,
    notebook_physics_accurate.agent,
    "V5",
    "Physics-Accurate [928.7]",
    num_games=3
)

# === Summary ===
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"V5 vs Distance-Prioritized [1100]: {v5_wins_dp}/3 ({v5_wins_dp*100//3}%)")
print(f"V5 vs Physics-Accurate [928.7]:    {v5_wins_pa}/3 ({v5_wins_pa*100//3}%)")
print(f"Total: {v5_wins_dp + v5_wins_pa}/6 games ({(v5_wins_dp + v5_wins_pa)*100//6}%)")

# Target: >45% against 900 ELO
target = 45
actual = (v5_wins_dp + v5_wins_pa) * 100 // 6
if actual >= target:
    print(f"\nSUCCESS: V5 achieved {actual}% (target: {target}%)")
else:
    print(f"\nBelow target: {actual}% (target: {target}%)")
    print("Consider tuning parameters or additional layers")
