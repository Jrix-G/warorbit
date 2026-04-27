#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract agent code from downloaded notebooks and create testable opponents
"""

import json
import os
import sys
import io
from pathlib import Path

if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def extract_agent_from_notebook(nb_path):
    """Extract agent function code from notebook"""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    agent_code = None
    imports = []

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))

            # Collect imports
            if 'import' in source and 'def ' not in source:
                imports.append(source)

            # Find agent function
            if any(x in source for x in ['def agent', 'def make_agent', 'def my_agent']):
                agent_code = source
                break

    return agent_code, imports

def create_opponent_file(nb_name, agent_code, imports):
    """Create a Python file with the agent that can be imported"""

    filename = f"opponents/notebook_{nb_name.replace('-', '_')}.py"

    # Combine all imports
    all_imports = set()
    for imp_block in imports:
        all_imports.add(imp_block.strip())

    content = "# Generated from notebook\n"
    content += "\n".join(all_imports)
    content += "\n\n"
    content += agent_code

    # Save
    Path("opponents").mkdir(exist_ok=True)
    Path(filename).write_text(content, encoding='utf-8')

    return filename

notebooks_map = {
    "orbitbotnext": "notebooks/orbitbotnext.ipynb",
    "distance_prioritized": "notebooks/distance-prioritized-agent-lb-max-score-1100.ipynb",
    "physics_accurate": "notebooks/lb-928-7-physics-accurate-planner.ipynb",
    "tactical_heuristic": "notebooks/orbit-wars-2026-tactical-heuristic.ipynb",
}

print("Extracting notebook agents...")
print("="*80)

for name, nb_path in notebooks_map.items():
    print(f"\n{name}:")

    if not Path(nb_path).exists():
        print(f"  Not found: {nb_path}")
        continue

    agent_code, imports = extract_agent_from_notebook(nb_path)

    if agent_code:
        filename = create_opponent_file(name, agent_code, imports)
        lines = len(agent_code.split('\n'))
        print(f"  Extracted: {lines} lines")
        print(f"  Saved: {filename}")
    else:
        print(f"  No agent found")

print("\n" + "="*80)
print("Next: Update opponents/__init__.py to register these agents")
print("="*80)
