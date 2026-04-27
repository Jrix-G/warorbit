#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and extract code from top Orbit Wars notebooks
"""

import subprocess
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Top notebooks to analyze
NOTEBOOKS = [
    {
        "name": "Getting Started (Official Base)",
        "url": "bovard/getting-started",
        "votes": 549,
        "priority": "CRITICAL"
    },
    {
        "name": "OrbitBotNext",
        "url": "pascalledesma/orbitbotnext",
        "votes": 62,
        "priority": "HIGH"
    },
    {
        "name": "Distance-Prioritized Agent [LB 1100]",
        "url": "ykhnkf/distance-prioritized-agent-lb-max-score-1100",
        "votes": 49,
        "priority": "HIGH"
    },
    {
        "name": "Physics-Accurate Planner [LB 928.7]",
        "url": "sigmaborov/lb-928-7-physics-accurate-planner",
        "votes": 42,
        "priority": "MEDIUM"
    },
    {
        "name": "Tactical Heuristic",
        "url": "sigmaborov/orbit-wars-2026-tactical-heuristic",
        "votes": 54,
        "priority": "MEDIUM"
    },
    {
        "name": "Structured Baseline",
        "url": "pilkwangkim/structured-baseline",
        "votes": 139,
        "priority": "HIGH"
    },
]

def download_notebook(notebook_url):
    """Download a single notebook from Kaggle"""
    output_file = f"notebooks/{notebook_url.split('/')[-1]}.ipynb"

    # Create notebooks directory
    Path("notebooks").mkdir(exist_ok=True)

    # Try to download using kaggle API
    cmd = f"kaggle kernels view {notebook_url} --output json > {output_file}.tmp 2>&1"
    print(f"  Downloading {notebook_url}...")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Alternative: try to get the notebook content
    cmd2 = f"kaggle datasets download -d -p notebooks --unzip 2>/dev/null || echo 'Using alternative method'"

    return output_file

def extract_code_from_notebook(notebook_path):
    """Extract Python code cells from a Jupyter notebook"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        code_cells = []
        for cell in notebook.get('cells', []):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if source.strip():
                    code_cells.append(source)

        return code_cells
    except Exception as e:
        print(f"  Error reading {notebook_path}: {e}")
        return []

def extract_strategy_patterns(code_text):
    """Extract strategy patterns from code"""
    import re

    patterns = {
        "weights": [],
        "planet_priority": [],
        "fleet_ratio": [],
        "sun_dodge": [],
        "comet_strategy": [],
        "attack_logic": [],
    }

    # Look for weight definitions
    weight_patterns = [
        r'W\s*=\s*\[(.*?)\]',
        r'weights\s*=\s*\[(.*?)\]',
        r'WEIGHTS\s*=\s*\{(.*?)\}',
    ]

    for pattern in weight_patterns:
        matches = re.findall(pattern, code_text, re.DOTALL)
        if matches:
            patterns["weights"].extend(matches)

    # Look for planet priority logic
    if 'neutral' in code_text.lower() and 'priority' in code_text.lower():
        patterns["planet_priority"].append("neutral_priority: TRUE")

    # Look for sun dodging
    if 'sun' in code_text.lower() and 'dodge' in code_text.lower():
        patterns["sun_dodge"].append("explicit_sun_dodging: TRUE")

    # Look for comet logic
    if 'comet' in code_text.lower():
        comet_matches = re.findall(r'comet.*?(capture|intercept|priority).*', code_text, re.IGNORECASE)
        patterns["comet_strategy"].extend(comet_matches[:3])

    # Look for attack ratios
    fleet_matches = re.findall(r'(\d+\.?\d*)\s*(?:\*|×).*?ships', code_text, re.IGNORECASE)
    if fleet_matches:
        patterns["fleet_ratio"].extend(fleet_matches[:5])

    return patterns

def main():
    print("\n" + "="*80)
    print("📚 DOWNLOADING TOP ORBIT WARS NOTEBOOKS")
    print("="*80)

    # Create output directory
    Path("notebooks").mkdir(exist_ok=True)

    print("\n🎯 TARGET NOTEBOOKS:")
    for nb in NOTEBOOKS:
        print(f"  [{nb['priority']}] {nb['name']}")
        print(f"       → https://kaggle.com/code/{nb['url']}")

    print("\n" + "="*80)
    print("📥 DOWNLOAD METHODS (in order of preference):")
    print("="*80)

    print("""
1️⃣ AUTOMATIC (Kaggle API):
   kaggle kernels view <url> --output json

2️⃣ MANUAL (Copy-Paste from Browser):
   1. Open: https://kaggle.com/code/[url]
   2. At top: click "Copy" → "Code as Notebook"
   3. Paste in: notebooks/[name].ipynb
   4. Save file

3️⃣ API Download (if available):
   https://www.kaggle.com/api/v1/kernels/[user]/[notebook-name]

════════════════════════════════════════════════════════════════════════════════

📋 WHAT WE'LL EXTRACT:

For each notebook:
  ✓ Complete Python code
  ✓ Weight values (W[0]-W[13] if defined)
  ✓ Planet priority logic
  ✓ Fleet send ratios
  ✓ Sun dodging implementation
  ✓ Comet capture strategy
  ✓ Attack/defense logic
  ✓ Performance benchmarks

════════════════════════════════════════════════════════════════════════════════

🚀 NEXT STEPS:

Option A - AUTOMATED:
  python3 parse_notebooks.py        # Analyze any .ipynb files in ./notebooks/

Option B - QUICK START:
  1. Open a notebook in browser (e.g., Getting Started by Bovard)
  2. Click "Code" button → "Copy as ipynb"
  3. Save to: notebooks/[name].ipynb
  4. Repeat for other top notebooks
  5. Run parse_notebooks.py

════════════════════════════════════════════════════════════════════════════════
""")

    # Try automatic download for each notebook
    print("\n🔄 ATTEMPTING AUTOMATIC DOWNLOADS...")
    for notebook in NOTEBOOKS:
        try:
            output = Path("notebooks") / f"{notebook['url'].split('/')[-1]}.ipynb"
            print(f"\n  [{notebook['priority']}] {notebook['name']}")
            print(f"       URL: {notebook['url']}")

            # Use a simple approach: list Kaggle kernels info
            cmd = f"kaggle kernels list -s {notebook['url'].split('/')[-1]} 2>&1"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)

            if "No kernels found" not in result.stdout and result.returncode == 0:
                print(f"       ✅ Found on Kaggle")
            else:
                print(f"       ⚠️  Manual download needed")

        except subprocess.TimeoutExpired:
            print(f"       ⏱️  Timeout")
        except Exception as e:
            print(f"       ❌ Error: {str(e)[:60]}")

    print("\n" + "="*80)
    print("✅ SETUP COMPLETE - Ready to parse notebooks")
    print("="*80)

if __name__ == "__main__":
    main()
