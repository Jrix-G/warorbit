#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean up extracted notebook agents - remove Jupyter magic commands
"""

import os
import sys
import io
from pathlib import Path

if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def clean_agent_file(filepath):
    """Remove Jupyter magic commands from extracted agent"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    skip_until_newline = False

    for line in lines:
        # Skip Jupyter magic commands
        if line.strip().startswith('%%'):
            skip_until_newline = True
            continue

        if skip_until_newline:
            if line.strip() == '':
                skip_until_newline = False
            continue

        # Skip empty comment lines from notebook
        if line.strip() == '#':
            continue

        cleaned_lines.append(line)

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    print(f"Cleaned: {filepath}")

# Clean all notebook agents
agents = [
    "opponents/notebook_orbitbotnext.py",
    "opponents/notebook_distance_prioritized.py",
    "opponents/notebook_physics_accurate.py",
    "opponents/notebook_tactical_heuristic.py",
]

print("Cleaning notebook agents...")
for agent_file in agents:
    if Path(agent_file).exists():
        clean_agent_file(agent_file)
    else:
        print(f"Not found: {agent_file}")

print("\nDone!")
