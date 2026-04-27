#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrape Orbit Wars competition data and analyze top players
"""

import subprocess
import json
from pathlib import Path
from collections import defaultdict
import os

if os.name == 'nt':
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_cmd(cmd):
    """Execute command and return output"""
    print(f"[*] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] Error: {result.stderr}")
        return None
    return result.stdout

def download_orbit_wars_data():
    """Download Orbit Wars leaderboard and metadata"""
    print("\n" + "="*80)
    print("[+] DOWNLOADING ORBIT WARS DATA")
    print("="*80)

    data_dir = Path("orbit_data")
    data_dir.mkdir(exist_ok=True)

    # Download dataset
    cmd = f"kaggle competitions download-files orbit-wars -p {data_dir}"
    output = run_cmd(cmd)

    if output:
        print(f"[+] Downloaded to: {data_dir}")
        # List files
        files = list(data_dir.glob("*"))
        for f in files:
            print(f"    - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        return data_dir
    else:
        print("[!] Download failed - trying alternative methods...")
        return None

def scrape_leaderboard_api():
    """Get leaderboard data via Kaggle API"""
    print("\n[+] Fetching leaderboard via API...")

    # Use kaggle-cli to get competition info
    cmd = "kaggle competitions describe orbit-wars"
    output = run_cmd(cmd)

    if output:
        print(output)
        return output
    return None

def analyze_top_teams():
    """Manually define top teams for analysis"""
    print("\n" + "="*80)
    print("[+] TOP TEAMS ANALYSIS")
    print("="*80 + "\n")

    teams = [
        ("kovi", 2580.7, 1),
        ("Shun_PI", 1629.8, 2),
        ("bowwowforeach", 1593.2, 3),
        ("HY2017", 1441.3, 4),
        ("Orbital Occle", 1431.1, 5),
        ("Erfan Eshratifar", 1425.1, 6),
        ("sash", 1405.0, 7),
        ("glass_256", 1369.1, 8),
        ("ush", 1361.0, 9),
        ("fgwiebfaoish", 1340.0, 10),
    ]

    print("RANK | TEAM NAME            | ELO     | ANALYSIS PRIORITY")
    print("-" * 70)

    for name, elo, rank in teams:
        if rank == 1:
            priority = "*** CRITICAL: Find losses ***"
        elif rank <= 3:
            priority = "HIGH: Opponent strategies"
        else:
            priority = "Medium: Pattern analysis"

        print(f"{rank:4d} | {name:20s} | {elo:7.1f} | {priority}")

    return teams

def create_kovi_analysis_plan():
    """Create detailed plan to find kovi's losses"""

    plan = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                     FINDING KOVI'S LOSSES - ACTION PLAN                       ║
╚════════════════════════════════════════════════════════════════════════════════╝

OBJECTIVE: Identify and analyze the game(s) where kovi lost

WHY THIS MATTERS:
  - Winning games don't show strategy weaknesses
  - Losing games reveal what breaks the dominant strategy
  - kovi is so far ahead (2580 vs 1629), finding his loss is KEY to winning

APPROACH:
  1. Navigate to: https://www.kaggle.com/competitions/orbit-wars/leaderboard
  2. Find kovi's row (rank #1)
  3. Click "View episodes from this team's highest scoring agent" (live_tv icon)
  4. Scroll through all games to find ANY with "LOSS" or unexpected result
  5. Click that game to view the replay
  6. Analyze:
     - Who beat him?
     - What was their strategy?
     - What did they do differently?
     - What did kovi do wrong?

CRITICAL DATA POINTS TO EXTRACT:
  - Game duration (# of turns)
  - Starting positions
  - Final score differential
  - Winner's strategy observed
  - Planet priority order
  - Fleet send patterns
  - Sun dodging approach
  - Comet capture decisions

COMPARISON TO MAKE:
  - How did kovi play in WINNING games?
  - How did he play differently in LOSING game?
  - What tactical decision cost him the game?
  - Can we exploit that same weakness?

════════════════════════════════════════════════════════════════════════════════

NEXT: Open Edge browser and navigate to kovi's episodes on Leaderboard
"""

    print(plan)

def generate_code_patterns_checklist():
    """Generate checklist for code pattern analysis"""

    checklist = """
╔════════════════════════════════════════════════════════════════════════════════╗
║             CODE PATTERN ANALYSIS - TOP NOTEBOOKS CHECKLIST                   ║
╚════════════════════════════════════════════════════════════════════════════════╝

For each public notebook, extract and document:

STRATEGIC PATTERNS:
  [ ] Planet selection priority (which planets attacked first?)
  [ ] Distance vs production weighting
  [ ] Fleet send ratio (what % of ships sent per turn?)
  [ ] Neutral vs enemy target preference
  [ ] Comet capture strategy (when/how?)
  [ ] Sun dodging technique (how implemented?)

NUMERICAL PARAMETERS:
  [ ] W[0] = neutral_priority     (bonus for targeting neutral)
  [ ] W[1] = comet_bonus          (multiplier if target is comet)
  [ ] W[2] = production_horizon   (future turns estimated)
  [ ] W[3] = distance_penalty     (cost per unit distance)
  [ ] W[4] = defense_reserve      (fraction held if threatened)
  [ ] W[5] = attack_ratio         (ships_needed × this)
  [ ] W[6] = fleet_send_ratio     (fraction of ships sent)
  [ ] W[7] = leader_penalty       (attacking dominant player)
  [ ] W[8] = weak_enemy_bonus     (if enemy < 30% of our ships)
  [ ] W[9] = sun_waypoint_dist    (waypoint distance factor)
  [ ] W[10] = endgame_threshold   (ratio for defense mode)
  [ ] W[11] = threat_eta_factor   (ETA weighting)
  [ ] W[12] = reinforce_ratio     (defense reinforcement threshold)
  [ ] W[13] = neutral_ships_cap   (max neutral ships / our ships)

CONDITIONAL LOGIC:
  [ ] Early game vs late game behavior differences
  [ ] How aggressiveness changes based on position
  [ ] Self-play vs opponent response patterns
  [ ] 2P vs 4P game differences

NOTEBOOKS TO ANALYZE:
  1. Getting Started (Bovard - 557 votes) - Reference implementation
  2. OrbitBotNext (Pascal - 62 votes) - Higher performance
  3. Distance-Prioritized (ykhnkf - 49 votes, LB max 1100)
  4. Physics-Accurate (sigmaborov - 42 votes, LB 928.7)
  5. Tactical Heuristic (sigmaborov - 54 votes)

════════════════════════════════════════════════════════════════════════════════
"""

    print(checklist)

def main():
    print("\n" + "="*80)
    print("ORBIT WARS DATA SCRAPER & ANALYSIS ENGINE")
    print("="*80)

    # Try to download data
    data_dir = download_orbit_wars_data()

    # Get leaderboard info
    scrape_leaderboard_api()

    # Analyze top teams
    teams = analyze_top_teams()

    # Create kovi analysis plan
    create_kovi_analysis_plan()

    # Generate code patterns checklist
    generate_code_patterns_checklist()

    print("\n" + "="*80)
    print("READY TO PROCEED")
    print("="*80)
    print("""
IMMEDIATE ACTIONS:

1. FIND KOVI'S LOSS (Browser-based):
   - Open Edge
   - Go to: https://www.kaggle.com/competitions/orbit-wars/leaderboard
   - Click kovi's episodes → Find LOSS or unusual game
   - Analyze that replay carefully

2. DOWNLOAD NOTEBOOKS (Already done via Kaggle API):
   - Files should be in: orbit_data/
   - Extract .ipynb files and convert to readable format

3. COMPARE STRATEGIES:
   - Take notes on what makes kovi unique
   - Compare to other top players
   - Identify winning vs losing patterns

4. ADAPT BOT:
   - Modify bot.py weights based on findings
   - Test against baselines with train.py --quick
   - Iterate and improve

════════════════════════════════════════════════════════════════════════════════
""")

if __name__ == "__main__":
    main()
