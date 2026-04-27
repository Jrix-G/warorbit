#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze kovi's loss game interactively
Extract and compare strategy decisions from the 220-turn replay
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Fix encoding on Windows
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Game summary (final state visible in replay)
GAME_INFO = {
    "episodeId": 75514378,
    "submissionId": 51987365,
    "totalTurns": 220,
    "winners": {
        1: {"name": "Mahdieh Rezaie", "finalShips": 737, "eloChange": 72},
        2: {"name": "kovi", "finalShips": 2578, "eloChange": -8, "LOST": True},
        3: {"name": "yuto083", "finalShips": 914, "eloChange": -5},
        4: {"name": "Arne De Brabandere", "finalShips": 538, "eloChange": 4},
    }
}

def create_turn_tracker():
    """Framework for tracking critical turns in the game"""

    tracker = {
        "key_moments": [],
        "planet_captures": defaultdict(list),  # planet_id: [(turn, player, ships)]
        "comet_events": [],  # [(turn, player_id, action)]
        "fleet_movements": [],  # [(turn, from_player, from_planet, to_planet, ships)]
        "kovi_decisions": [],  # turns where kovi made notable decisions
        "winner_strategy": [],  # Mahdieh's key moves
    }
    return tracker

def format_analysis_template():
    """Generate template for documenting the 220-turn game analysis"""

    template = """
╔════════════════════════════════════════════════════════════════════════════════╗
║             KOVI'S LOSS - COMPLETE GAME ANALYSIS (220 TURNS)                  ║
║                 Mahdieh Rezaie vs kovi (4-Player, episodeId 75514378)         ║
╚════════════════════════════════════════════════════════════════════════════════╝

CRITICAL FACT: kovi had 2578 ships (3.5× more than winner's 737) but STILL LOST

🎯 OUR ANALYSIS FOCUS:

1. WINNER'S STRATEGY (Mahdieh Rezaie - 737 ships → 1st place)
   ────────────────────────────────────────────────────────
   - Which planets captured first? (turn?)
   - Fleet send ratio (how many ships/turn sent?)
   - Target priority: neutrals first? Weak opponents? Specific planet types?
   - Sun dodging patterns (any waypoints, or direct assault?)
   - Comet capture: turns 50/150/250/350/450 - which captured?
   - Defensive strategy when outnumbered (turns 1-100?)
   - Aggressive shifts (when did aggression increase?)

2. KOVI'S DECISIONS - WHERE IT WENT WRONG (2578 ships → LOST)
   ───────────────────────────────────────────────────────────
   - Early game: planets targeted (turns 1-50?)
   - Fleet send ratio: conservative or aggressive?
   - Was Mahdieh perceived as threat from start?
   - Sun avoidance: any special logic or just direct routes?
   - Comet captures: which comets did kovi take?
   - Tactical error #1: (turn?) → consequence
   - Tactical error #2: (turn?) → consequence
   - Critical decision that cost the game: (turn?)

3. KEY TURNING POINTS
   ──────────────────
   - Turn ~50 (first comet): who captured? What changed?
   - Turn ~100: power distribution (ships per player)
   - Turn ~150 (second comet): strategic impact
   - Mid-game (turns 100-150): shift in momentum?
   - Late game (turns 150-220): final strategic phase

4. COMPARISON TO KOVI'S WINS
   ─────────────────────────
   In winning games, kovi likely:
   - Targeted weak players early
   - Built up ships faster
   - Used sun-dodging effectively
   - Captured comets strategically

   In THIS loss:
   - Did he NOT prioritize weak players?
   - Did he waste ships on unprofitable attacks?
   - Was sun-dodging inefficient?
   - Did he concede comets to Mahdieh?

════════════════════════════════════════════════════════════════════════════════

📋 INSTRUCTIONS FOR CAPTURING DATA:

Since the replay is interactive on your screen, we need to extract key game events.
You can either:

A) MANUAL TURN EXPORT (fastest):
   - For each critical turn (1, 10, 20, 50, 75, 100, 125, 150, 175, 200, 220):
     - Note who owns which planets
     - Note ship counts per player
     - Note any fleets in transit
   - Copy-paste into structured_game_data.json

B) SCREENSHOT ANALYSIS:
   - Screenshot key turns showing:
     - Board state (planet ownership)
     - Fleet status (ships by player)
     - Timeline (which turn we're at)

C) DESCRIBE EVENTS (narrative):
   - "Turn 15: Mahdieh sends 100 ships to Planet 3"
   - "Turn 42: kovi attacks neutral Planet 7, takes it with 250 ships"
   - "Turn 50 (COMET): Mahdieh captures comet, now has 340 ships total"

════════════════════════════════════════════════════════════════════════════════
"""
    print(template)
    return template

def create_data_collection_form():
    """Create JSON template for game data to be filled in"""

    form = {
        "game_meta": {
            "episode_id": 75514378,
            "submission_id": 51987365,
            "total_turns": 220,
            "players": {
                "0": "Mahdieh Rezaie (WINNER - 737 ships)",
                "1": "kovi (LOSER - 2578 ships)",
                "2": "yuto083 (914 ships)",
                "3": "Arne De Brabandere (538 ships)"
            }
        },
        "critical_turns": {
            "turn_1": {
                "note": "Starting state - all players own initial planets",
                "mahdieh_ships": None,
                "kovi_ships": None,
                "key_observation": ""
            },
            "turn_20": {
                "note": "Early game aggression patterns",
                "mahdieh_ships": None,
                "kovi_ships": None,
                "planets_captured": [],
                "key_observation": ""
            },
            "turn_50": {
                "note": "FIRST COMET APPEARS - who captures it?",
                "mahdieh_ships": None,
                "kovi_ships": None,
                "comet_captured_by": None,
                "key_observation": ""
            },
            "turn_100": {
                "note": "Mid-game inflection point",
                "mahdieh_ships": None,
                "kovi_ships": None,
                "power_distribution": {},
                "key_observation": ""
            },
            "turn_150": {
                "note": "SECOND COMET - strategic momentum",
                "mahdieh_ships": None,
                "kovi_ships": None,
                "comet_captured_by": None,
                "key_observation": ""
            },
            "turn_200": {
                "note": "End-game is visible - who's winning?",
                "mahdieh_ships": None,
                "kovi_ships": None,
                "key_observation": ""
            },
            "turn_220": {
                "note": "FINAL STATE - game over",
                "mahdieh_ships": 737,
                "kovi_ships": 2578,
                "outcome": "Mahdieh WINS with 3.5× fewer ships"
            }
        },
        "strategic_questions": {
            "mahdieh_strategy": {
                "first_planet_targeted": None,  # Which planet type?
                "fleet_send_ratio_early": None,  # Conservative or aggressive?
                "comet_strategy": "Which comets captured?",
                "sun_avoidance": "Any special dodging observed?",
                "key_insight": ""
            },
            "kovi_mistakes": {
                "early_game_error": {
                    "turn": None,
                    "description": "",
                    "consequence": ""
                },
                "mid_game_error": {
                    "turn": None,
                    "description": "",
                    "consequence": ""
                },
                "critical_mistake": {
                    "turn": None,
                    "description": "THE decision that lost the game",
                    "consequence": ""
                }
            }
        }
    }

    # Save template
    with open("game_data_template.json", "w", encoding="utf-8") as f:
        json.dump(form, f, indent=2, ensure_ascii=False)

    print("✅ Template created: game_data_template.json")
    print("\n📌 Next step: Fill in the game data from the replay you're watching")
    print("   Then run: python analyze_kovi_loss.py --analyze game_data_template.json")

    return form

def analyze_game_data(data_file):
    """Analyze filled-in game data and extract strategic insights"""

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n" + "="*80)
    print("🔍 GAME ANALYSIS - KEY FINDINGS")
    print("="*80)

    # Analyze critical turns
    print("\n📊 POWER PROGRESSION:")
    turns = data["critical_turns"]
    for turn_key in sorted(turns.keys(), key=lambda x: int(x.split("_")[1])):
        turn = turns[turn_key]
        if turn.get("mahdieh_ships") and turn.get("kovi_ships"):
            mahdieh = turn["mahdieh_ships"]
            kovi = turn["kovi_ships"]
            ratio = kovi / mahdieh if mahdieh > 0 else 0
            print(f"  Turn {turn_key.split('_')[1]:3s}: "
                  f"Mahdieh {mahdieh:5.0f} | kovi {kovi:5.0f} "
                  f"(ratio: {ratio:.2f}x)")

    # Extract strategic patterns
    print("\n🎯 MAHDIEH'S WINNING STRATEGY:")
    mahdieh_strat = data["strategic_questions"]["mahdieh_strategy"]
    for key, val in mahdieh_strat.items():
        if val:
            print(f"  {key}: {val}")

    print("\n❌ KOVI'S MISTAKES:")
    kovi_errors = data["strategic_questions"]["kovi_mistakes"]
    for error_type, error in kovi_errors.items():
        if error.get("turn"):
            print(f"  [{error_type}] Turn {error['turn']}: {error['description']}")
            print(f"    → Consequence: {error['consequence']}")

def main():
    print("\n" + "="*80)
    print("🔴 ANALYZING KOVI'S LOSS - INTERACTIVE GAME ANALYSIS")
    print("="*80)

    format_analysis_template()
    create_data_collection_form()

    print("\n" + "="*80)
    print("💡 HOW TO PROCEED:")
    print("="*80)
    print("""
Your replay URL: https://www.kaggle.com/competitions/orbit-wars/leaderboard?submissionId=51987365&episodeId=75514378

METHOD 1 - FILL JSON FORM (Recommended):
  1. Open: game_data_template.json
  2. Watch the replay turn-by-turn
  3. Fill in critical turns (1, 20, 50, 100, 150, 200, 220)
  4. Note key observations about strategies
  5. Run: python analyze_kovi_loss.py --analyze game_data_template.json

METHOD 2 - DESCRIBE KEY MOMENTS:
  1. Watch the replay
  2. Tell me: "Turn 50: [what happened]"
  3. I'll compile into analysis format

════════════════════════════════════════════════════════════════════════════════
""")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--analyze":
        analyze_game_data(sys.argv[2])
    else:
        main()
