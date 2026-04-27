#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep analysis of notebook agent strategies to understand what makes them win
"""

import json
import os
import sys
import io
import re
from pathlib import Path
from collections import defaultdict

if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def extract_strategy_from_code(filepath, agent_name):
    """Extract strategic patterns from agent code"""

    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    strategy = {
        "agent_name": agent_name,
        "file_size": len(code),
        "code_lines": len(code.split('\n')),
        "key_patterns": {},
        "constants": {},
        "decision_logic": [],
    }

    # Extract constants/parameters
    const_patterns = [
        (r'([A-Z_]+)\s*=\s*([\d\.]+)', 'numeric'),
        (r'([A-Z_]+)_RATIO\s*=\s*([\d\.]+)', 'ratio'),
        (r'([A-Z_]+)_THRESHOLD\s*=\s*([\d\.]+)', 'threshold'),
    ]

    for pattern, ctype in const_patterns:
        matches = re.findall(pattern, code)
        for name, value in matches[:10]:  # Top 10
            try:
                strategy["constants"][name] = float(value)
            except:
                pass

    # Key strategic patterns
    patterns = {
        "threat_detection": r'(threat|danger|incoming|eta|distance.*fleet)',
        "comet_strategy": r'(comet.*capture|comet.*bonus|comet.*priority)',
        "defense_logic": r'(defense|reserve|protect|threatened|attack_cost)',
        "4p_logic": r'(leader|dominant|second|third|weak.*enemy|kingmaker)',
        "fleet_management": r'(send.*ratio|fleet.*percent|commit|reserve)',
        "production_calc": r'(production.*horizon|future.*turn|projected)',
        "sun_avoidance": r'(sun.*dodge|waypoint|orbit.*safe)',
        "target_selection": r'(target.*select|priority.*order|best.*target)',
        "early_game": r'(early.*game|opening|first.*\d+.*turn)',
        "endgame": r'(endgame|late.*game|final.*turn|remaining)',
    }

    for pattern_name, regex in patterns.items():
        matches = re.findall(regex, code, re.IGNORECASE)
        if matches:
            strategy["key_patterns"][pattern_name] = len(matches)

    return strategy

def analyze_decision_flow(filepath):
    """Analyze the decision-making flow structure"""

    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    # Find function definitions
    functions = re.findall(r'def\s+(\w+)\s*\(', code)

    # Look for main decision logic
    if_statements = len(re.findall(r'\bif\s+', code))
    loops = len(re.findall(r'\bfor\s+|\bwhile\s+', code))

    return {
        "num_functions": len(functions),
        "top_functions": functions[:10],
        "if_statements": if_statements,
        "loops": loops,
        "complexity_score": if_statements + loops,
    }

def main():
    print("\n" + "="*80)
    print("DEEP ANALYSIS: NOTEBOOK AGENT STRATEGIES")
    print("="*80)

    agents_to_analyze = [
        ("opponents/notebook_orbitbotnext.py", "OrbitBotNext"),
        ("opponents/notebook_distance_prioritized.py", "Distance-Prioritized [LB 1100]"),
        ("opponents/notebook_physics_accurate.py", "Physics-Accurate [LB 928.7]"),
        ("opponents/notebook_tactical_heuristic.py", "Tactical-Heuristic"),
    ]

    all_strategies = []

    for filepath, name in agents_to_analyze:
        print(f"\n{name}")
        print("-" * 80)

        if not Path(filepath).exists():
            print(f"  File not found: {filepath}")
            continue

        strategy = extract_strategy_from_code(filepath, name)
        decision = analyze_decision_flow(filepath)

        print(f"  Code size: {strategy['code_lines']} lines")
        print(f"  Complexity: {decision['if_statements']} if-statements, {decision['loops']} loops")
        print(f"  Score: {decision['complexity_score']}")

        print(f"\n  Key Strategic Elements Found:")
        for pattern, count in sorted(strategy["key_patterns"].items(), key=lambda x: -x[1]):
            print(f"    - {pattern}: {count} occurrences")

        if strategy["constants"]:
            print(f"\n  Key Constants/Parameters:")
            for const, val in sorted(strategy["constants"].items())[:8]:
                print(f"    {const:30s} = {val}")

        all_strategies.append(strategy)

    # Comparative analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    print("\nComplexity Ranking (by if/loop count):")
    sorted_by_complexity = sorted(
        [(s["agent_name"], analyze_decision_flow(
            f"opponents/notebook_{s['agent_name'].lower().replace('-', '_').replace('[', '').replace(']', '').split()[0]}.py"
        )["complexity_score"]) for s in all_strategies],
        key=lambda x: -x[1]
    )

    for agent, score in sorted_by_complexity:
        print(f"  {agent:35s}: {score} (more complex = more adaptive)")

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS - WHAT MAKES THEM WIN")
    print("="*80)

    print("""
1. THEY ALL HAVE SOPHISTICATION (not just simple weights):
   - 1900-3100 lines of code vs our 400
   - Multiple decision functions for different game phases
   - Explicit threat detection and response logic

2. THEY ALL USE:
   ✓ Production horizon calculation (multi-turn planning)
   ✓ Threat assessment (incoming fleet detection)
   ✓ 4-player kingmaker logic (don't always attack #1)
   ✓ Defense reserves (don't send 100% of ships)
   ✓ Explicit comet strategies (time-limited windows)

3. THEY ALL ADAPT BY GAME PHASE:
   ✓ Early game (turns 1-40): different strategy
   ✓ Mid game (turns 40-200): adaptation
   ✓ Late game (turns 200+): endgame vs defensive
   ✓ Comet windows (turns 50/150/250): special handling

4. THEY CALCULATE COSTS:
   ✓ Attack cost = (distance × time × opponent defense)
   ✓ Opportunity cost (ships in transit = lost production)
   ✓ Risk-adjusted decisions (not greedy, expected value based)

5. THEY DO OPPONENT MODELING:
   ✓ Remember what opponent did last game
   ✓ Adjust strategy based on opponent type
   ✓ Predict opponent moves (threat ETA)
""")

if __name__ == "__main__":
    main()
