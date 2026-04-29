#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test: Can our agents be called without errors?
"""

import os
import sys
import io

if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Testing if agents can be imported and called...")
print("="*80)

# Import bot_submit
try:
    import bot_submit
    print("bot_submit: OK")
except Exception as e:
    print(f"bot_submit: ERROR - {e}")
    sys.exit(1)

# Import notebook agents
agents = {
    "orbitbotnext": "opponents.notebook_orbitbotnext",
    "distance_prioritized": "opponents.notebook_distance_prioritized",
    "physics_accurate": "opponents.notebook_physics_accurate",
    "tactical_heuristic": "opponents.notebook_tactical_heuristic",
}

print("\nNotebook agents:")
loaded_agents = {}

for name, module_path in agents.items():
    try:
        parts = module_path.split('.')
        module = __import__(module_path, fromlist=[parts[-1]])

        if hasattr(module, 'agent'):
            loaded_agents[name] = module.agent
            print(f"  {name}: OK (callable)")
        else:
            print(f"  {name}: NO agent function")
    except Exception as e:
        print(f"  {name}: ERROR - {str(e)[:60]}")

print(f"\nLoaded {len(loaded_agents)} agents successfully!")

# Create minimal observation
sample_obs = {
    "planets": [],
    "fleets": [],
    "player": 0,
    "remaining_actions_per_player": [1000],
}

print("\nTest: Can agents respond to observation?")
print("-"*80)

for name, agent_func in loaded_agents.items():
    try:
        # Try calling with sample obs
        result = agent_func(sample_obs)
        print(f"  {name}: OK (returned {type(result).__name__})")
    except TypeError as e:
        # Agent might need config parameter
        try:
            result = agent_func(sample_obs, config=None)
            print(f"  {name}: OK (with config)")
        except Exception as e2:
            print(f"  {name}: ERROR - {str(e2)[:50]}")
    except Exception as e:
        print(f"  {name}: ERROR - {str(e)[:50]}")

print("\n" + "="*80)
print("Agents are ready for testing!")
print("="*80)
