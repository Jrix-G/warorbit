#!/usr/bin/env python3
"""Test V5 bot against notebook agents"""

import sys
import io
import os

if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import agents
import bot_v5

agents_to_test = {
    "v5": bot_v5.agent,
}

# Try loading notebook agents
try:
    from opponents import notebook_distance_prioritized
    agents_to_test["distance_prioritized"] = notebook_distance_prioritized.agent
    print("Loaded: distance_prioritized [LB 1100]")
except Exception as e:
    print(f"Could not load distance_prioritized: {str(e)[:50]}")

try:
    from opponents import notebook_physics_accurate
    agents_to_test["physics_accurate"] = notebook_physics_accurate.agent
    print("Loaded: physics_accurate [LB 928.7]")
except Exception as e:
    print(f"Could not load physics_accurate: {str(e)[:50]}")

try:
    from opponents import notebook_tactical_heuristic
    agents_to_test["tactical_heuristic"] = notebook_tactical_heuristic.agent
    print("Loaded: tactical_heuristic")
except Exception as e:
    print(f"Could not load tactical_heuristic: {str(e)[:50]}")

print(f"\nLoaded {len(agents_to_test)} agents total")
print("="*80)

# Test each agent pair
for name, agent_func in agents_to_test.items():
    print(f"\nAgent: {name}")
    print(f"  Function: {agent_func.__name__}")

    # Try calling with minimal obs
    obs = {
        'planets': [
            [0, 0, 25, 25, 2, 50, 2],
            [1, 1, 75, 75, 2, 30, 1],
        ],
        'fleets': [],
        'player': 0,
        'step': 10,
    }

    try:
        result = agent_func(obs)
        print(f"  Response: {type(result).__name__}, {len(result) if hasattr(result, '__len__') else '?'} items")
    except TypeError:
        try:
            result = agent_func(obs, config=None)
            print(f"  Response (with config): {type(result).__name__}, {len(result) if hasattr(result, '__len__') else '?'} items")
        except Exception as e2:
            print(f"  ERROR: {str(e2)[:60]}")
    except Exception as e:
        print(f"  ERROR: {str(e)[:60]}")

print("\n" + "="*80)
print("V5 bot is ready for competition testing")
print("\nNext: run train.py with V5 bot as baseline to measure win rates")
