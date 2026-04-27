#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orbit Wars Competition Data Analyzer
Analyse les replays pour identifier les patterns gagnants et les faiblesses des concurrents
"""

import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import os

# Fix encoding on Windows
os.chdir(Path(__file__).parent)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_cmd(cmd):
    """Execute une commande shell"""
    print(f"▶ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return None
    return result.stdout

def show_analysis_template():
    """Affiche le template d'analyse"""

    report = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                    ORBIT WARS - COMPETITIVE ANALYSIS                          ║
║                          Phase 1: Data Gathering                              ║
╚════════════════════════════════════════════════════════════════════════════════╝

📋 OBJECTIFS PRIORITAIRES:

  1. 🎯 FIND KOVI'S LOSSES (CRITICAL!)
     - Chercher les 1-2 parties perdues par kovi
     - Analyser la stratégie de l'adversaire
     - Identifier ses faiblesses

  2. 🔍 EXTRACT WINNING PATTERNS
     - Ordre de priorité des planètes?
     - Ratio d'envoi de flottes?
     - Stratégie d'esquive du soleil?
     - Timing capture des comètes?

  3. 📊 TOP 10 TEAMS ANALYSIS
     • kovi (2580.7 ELO) ← Leader, trouver sa faiblesse
     • Shun_PI (1629.8 ELO)
     • bowwowforeach (1593.2)
     • HY2017 (1441.3)
     • Orbital Occle (1431.1)
     • Erfan Eshratifar (1425.1)
     • sash (1405.0)
     • glass_256 (1369.1)
     • ush (1361.0)
     • fgwiebfaoish (1340.0)

🎲 DONNÉES À COLLECTER:

  ✓ Notebooks publics (stratégies visibles)
  ✓ Replays (comportement en jeu)
  ✓ Leaderboard (progression ELO)
  ✓ Patterns de défaites (pourquoi ils perdent)

📥 SOURCES DE DONNÉES:

  1. Kaggle Notebooks (code public):
     - Getting Started (Bovard - base officielle)
     - OrbitBotNext (62 votes)
     - Distance-Prioritized (49 votes)
     - Physics-Accurate (42 votes)

  2. Replay Analysis (si disponible):
     - Win/Loss ratios
     - Game duration patterns
     - Fleet sizes at critical moments

  3. Kaggle API (si connecté):
     kaggle competitions download-files orbit-wars

🛠️ NEXT ACTIONS:

  ① Créer script de download des notebooks
  ② Parser les codes Python publics
  ③ Extraire les heuristiques/poids
  ④ Analyser les métriques de performance
  ⑤ Chercher les losses de kovi
  ⑥ Adapter notre bot

════════════════════════════════════════════════════════════════════════════════
"""
    print(report)

def check_kaggle_auth():
    """Vérifie si kaggle-cli est configuré"""
    print("\n🔐 Vérification de l'authentification Kaggle...")

    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'

    if kaggle_json.exists():
        print("✅ kaggle.json trouvé - authentification OK")
        return True
    else:
        print("❌ kaggle.json non trouvé")
        print("   Run: kaggle auth login")
        return False

def extract_notebook_urls():
    """Liste des notebooks publics à analyser"""

    notebooks = [
        {
            "name": "Getting Started",
            "author": "Bovard",
            "url": "bovard/getting-started",
            "votes": 557,
            "badge": "gold"
        },
        {
            "name": "OrbitBotNext",
            "author": "Pascal",
            "url": "pascalledesma/orbitbotnext",
            "votes": 62,
            "badge": "silver"
        },
        {
            "name": "Distance-Prioritized Agent",
            "author": "ykhnkf",
            "url": "ykhnkf/distance-prioritized-agent-lb-max-score-1100",
            "votes": 49,
            "badge": "silver"
        },
        {
            "name": "Physics-Accurate Planner",
            "author": "sigmaborov",
            "url": "sigmaborov/lb-928-7-physics-accurate-planner",
            "votes": 42,
            "badge": "silver"
        },
        {
            "name": "Tactical Heuristic",
            "author": "sigmaborov",
            "url": "sigmaborov/orbit-wars-2026-tactical-heuristic",
            "votes": 54,
            "badge": "bronze"
        },
    ]

    print("\n📚 NOTEBOOKS À ANALYSER:")
    for i, nb in enumerate(notebooks, 1):
        badge_emoji = {"gold": "🥇", "silver": "🥈", "bronze": "🥉"}.get(nb["badge"], "")
        print(f"  {i}. {badge_emoji} {nb['name']:40s} ({nb['votes']:3d} votes)")
        print(f"     → https://kaggle.com/code/{nb['url']}")

    return notebooks

def create_download_script():
    """Crée un script de download des notebooks"""

    script = '''#!/usr/bin/env python3
"""Download Orbit Wars notebooks via Kaggle API"""

from pathlib import Path
import subprocess

notebooks = [
    "bovard/getting-started",
    "pascalledesma/orbitbotnext",
    "ykhnkf/distance-prioritized-agent-lb-max-score-1100",
    "sigmaborov/lb-928-7-physics-accurate-planner",
    "sigmaborov/orbit-wars-2026-tactical-heuristic",
]

output_dir = Path("notebooks")
output_dir.mkdir(exist_ok=True)

for notebook in notebooks:
    print(f"Downloading {notebook}...")
    cmd = f"kaggle kernels view {notebook} --metadata"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
'''

    with open("download_notebooks.py", "w") as f:
        f.write(script)

    print("\n✅ Script created: download_notebooks.py")

def main():
    print("\n" + "="*80)
    print("🚀 ORBIT WARS COMPETITIVE ANALYSIS - PHASE 1")
    print("="*80)

    show_analysis_template()

    # Check Kaggle auth
    has_kaggle = check_kaggle_auth()

    # Extract notebook URLs
    notebooks = extract_notebook_urls()

    # Create download script
    create_download_script()

    print("\n" + "="*80)
    print("📌 READY FOR NEXT PHASE")
    print("="*80)
    print("""
INSTRUCTIONS:

  1️⃣  If you have Kaggle API credentials:
      python download_notebooks.py

  2️⃣  To analyze replays manually:
      - Visit: https://www.kaggle.com/competitions/orbit-wars/code
      - Download each notebook JSON
      - Parse and extract code patterns

  3️⃣  To find kovi's loss:
      - Go to: https://www.kaggle.com/competitions/orbit-wars/leaderboard
      - Click kovi's episodes → look for LOSS or unusual results

  4️⃣  Key metrics to track:
      - Win rate
      - Average game duration
      - Planet priority order
      - Fleet send ratios
      - Sun dodging frequency

Next step: Extract actual game replays and analyze kovi's defeat!
""")

if __name__ == "__main__":
    main()
