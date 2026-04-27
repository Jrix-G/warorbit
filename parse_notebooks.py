#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse Orbit Wars notebooks to extract strategies and weights
"""

import json
import os
import sys
import re
from pathlib import Path
from collections import defaultdict

if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def extract_code_from_notebook(notebook_path):
    """Extract all code cells from a Jupyter notebook"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        code_cells = []
        markdown_cells = []

        for i, cell in enumerate(notebook.get('cells', [])):
            if cell['cell_type'] == 'code':
                source = ''.join(cell.get('source', []))
                if source.strip():
                    code_cells.append({
                        'index': i,
                        'source': source,
                        'has_output': bool(cell.get('outputs'))
                    })
            elif cell['cell_type'] == 'markdown':
                source = ''.join(cell.get('source', []))
                if source.strip():
                    markdown_cells.append({
                        'index': i,
                        'source': source
                    })

        return code_cells, markdown_cells
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {notebook_path}: {e}")
        return [], []
    except Exception as e:
        print(f"❌ Error reading {notebook_path}: {e}")
        return [], []

def extract_weights(code_text):
    """Extract weight values from code"""
    weights = {}

    # Pattern 1: W = [val1, val2, val3, ...]
    match = re.search(r'W\s*=\s*\[([\d\.,\s\-e]+)\]', code_text)
    if match:
        values = match.group(1).strip()
        values = re.findall(r'[\d\.\-e]+', values)
        try:
            weights['W_array'] = [float(v) for v in values]
        except:
            pass

    # Pattern 2: Individual weight definitions
    patterns = [
        (r'W\[(\d+)\]\s*=\s*([\d\.\-e]+)', 'indexed'),
        (r'neutral.*?=\s*([\d\.\-e]+)', 'neutral_priority'),
        (r'comet.*?=\s*([\d\.\-e]+)', 'comet_bonus'),
        (r'production.*?=\s*([\d\.\-e]+)', 'production_horizon'),
        (r'distance.*?=\s*([\d\.\-e]+)', 'distance_penalty'),
        (r'defense.*?=\s*([\d\.\-e]+)', 'defense_reserve'),
        (r'attack.*?ratio.*?=\s*([\d\.\-e]+)', 'attack_ratio'),
        (r'fleet.*?ratio.*?=\s*([\d\.\-e]+)', 'fleet_send_ratio'),
    ]

    for pattern, key in patterns:
        matches = re.findall(pattern, code_text, re.IGNORECASE)
        if matches:
            if key == 'indexed':
                for idx, val in matches:
                    try:
                        weights[f'W[{idx}]'] = float(val)
                    except:
                        pass
            else:
                try:
                    weights[key] = float(matches[0])
                except:
                    weights[key] = matches[0]

    return weights

def extract_strategy_features(code_text):
    """Extract strategic features and heuristics"""
    features = {
        'has_sun_dodging': False,
        'has_comet_logic': False,
        'has_neutral_priority': False,
        'has_production_horizon': False,
        'has_threat_assessment': False,
        'has_kingmaker_logic': False,
        'target_selection_logic': [],
        'defense_strategy': [],
        'special_techniques': [],
    }

    # Check for features
    if re.search(r'sun|dodge|waypoint|orbit', code_text, re.IGNORECASE):
        features['has_sun_dodging'] = True

    if re.search(r'comet|capture', code_text, re.IGNORECASE):
        features['has_comet_logic'] = True

    if re.search(r'neutral', code_text, re.IGNORECASE):
        features['has_neutral_priority'] = True

    if re.search(r'production|future|horizon|turns?_ahead', code_text, re.IGNORECASE):
        features['has_production_horizon'] = True

    if re.search(r'threat|danger|incoming|eta', code_text, re.IGNORECASE):
        features['has_threat_assessment'] = True

    if re.search(r'leader|dominant|king|maker|second', code_text, re.IGNORECASE):
        features['has_kingmaker_logic'] = True

    # Extract comments and docstrings that explain strategy
    comments = re.findall(r'#[^\n]+', code_text)
    docstrings = re.findall(r'"""([^"]*)"""', code_text)
    docstrings += re.findall(r"'''([^']*)'''", code_text)

    for doc in docstrings[:3]:
        if 'strateg' in doc.lower() or 'algorithm' in doc.lower() or 'approach' in doc.lower():
            features['special_techniques'].append(doc[:200])

    return features

def analyze_notebook(notebook_path):
    """Analyze a single notebook"""
    filename = notebook_path.name
    print(f"\n📖 ANALYZING: {filename}")
    print("-" * 80)

    code_cells, markdown_cells = extract_code_from_notebook(notebook_path)

    if not code_cells:
        print("  ⚠️  No code cells found")
        return None

    # Combine all code
    all_code = "\n".join([cell['source'] for cell in code_cells])

    # Extract weights
    weights = extract_weights(all_code)
    print(f"\n  🔢 WEIGHTS FOUND: {len(weights)}")
    for key, val in sorted(weights.items()):
        if key != 'W_array':
            print(f"     {key}: {val}")

    if 'W_array' in weights:
        w_arr = weights['W_array']
        print(f"     W array: {len(w_arr)} values")
        if len(w_arr) <= 14:
            for i, val in enumerate(w_arr):
                print(f"       W[{i}] = {val}")

    # Extract features
    features = extract_strategy_features(all_code)
    print(f"\n  🎯 STRATEGY FEATURES:")
    for feature, value in features.items():
        if isinstance(value, bool):
            status = "✅" if value else "❌"
            print(f"     {status} {feature}")
        elif isinstance(value, list) and value:
            print(f"     • {feature}: {len(value)} items")

    # Extract key heuristics
    print(f"\n  📋 KEY CODE SECTIONS:")
    lines = all_code.split('\n')
    key_lines = [
        line for line in lines
        if any(keyword in line.lower() for keyword in ['target', 'priority', 'attack', 'defend', 'comet', 'sun', 'fleet', 'ratio'])
        and not line.strip().startswith('#')
        and len(line) > 20
        and len(line) < 120
    ]

    for line in key_lines[:5]:
        line_clean = line.strip()
        if line_clean:
            print(f"     → {line_clean[:90]}")

    return {
        'filename': filename,
        'code_cells': len(code_cells),
        'markdown_cells': len(markdown_cells),
        'weights': weights,
        'features': features,
        'code_length': len(all_code)
    }

def main():
    print("\n" + "="*80)
    print("📊 PARSING ORBIT WARS NOTEBOOKS")
    print("="*80)

    notebooks_dir = Path("notebooks")
    if not notebooks_dir.exists():
        print("❌ ./notebooks/ directory not found")
        print("   Run: python3 download_top_notebooks.py first")
        return

    notebook_files = list(notebooks_dir.glob("*.ipynb"))
    if not notebook_files:
        print("❌ No .ipynb files found in ./notebooks/")
        print("   Download notebooks from Kaggle first")
        return

    print(f"\n✅ Found {len(notebook_files)} notebooks to analyze\n")

    analysis_results = []

    for notebook_path in sorted(notebook_files):
        result = analyze_notebook(notebook_path)
        if result:
            analysis_results.append(result)

    # Summary
    print("\n" + "="*80)
    print("📈 SUMMARY")
    print("="*80)

    print(f"\nNotebooks analyzed: {len(analysis_results)}")
    print("\n🏆 Strategy Comparison:")
    print("-" * 80)

    # Create comparison table
    features_list = [
        'has_sun_dodging',
        'has_comet_logic',
        'has_neutral_priority',
        'has_production_horizon',
        'has_threat_assessment',
        'has_kingmaker_logic'
    ]

    for result in analysis_results:
        print(f"\n{result['filename']}:")
        for feature in features_list:
            status = "✅" if result['features'][feature] else "❌"
            print(f"  {status} {feature}")

    # Save detailed analysis
    output_file = "notebook_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert results to serializable format
        serializable = []
        for r in analysis_results:
            r_copy = r.copy()
            r_copy['features'] = {k: ("YES" if isinstance(v, bool) and v else ("NO" if isinstance(v, bool) else v))
                                  for k, v in r['features'].items()}
            serializable.append(r_copy)
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Detailed analysis saved: {output_file}")

    print("\n" + "="*80)
    print("🎯 NEXT STEP: Compare strategies and identify winning patterns")
    print("="*80)

if __name__ == "__main__":
    main()
