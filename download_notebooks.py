#!/usr/bin/env python3
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
