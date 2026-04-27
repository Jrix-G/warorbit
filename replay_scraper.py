#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive replay data scraper
Helps extract game state from Kaggle replay visualization
"""

import json
import os
import sys
from pathlib import Path

if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def interactive_turn_capture():
    """Interactively capture turn data from user watching replay"""

    print("\n" + "="*80)
    print("🎮 INTERACTIVE REPLAY SCRAPER")
    print("="*80)
    print("""
Regardez le replay et copiez les données suivantes pour chaque tour clé:

Pour trouver les données dans la UI Kaggle:
  1. En bas à gauche: "Turn X / 220"
  2. En haut à droite: Stats panel montre les ships par joueur
  3. Au centre: Board montre les planètes et flottes

Données à noter par tour:
  - turn_number: (nombre au bas)
  - mahdieh_ships: (total ships - cherchez "Mahdieh Rezaie")
  - kovi_ships: (total ships - cherchez "kovi")
  - yuto083_ships: (optionnel)
  - arne_ships: (optionnel)
  - key_event: (ex: "Mahdieh captures comet", "kovi attacks Planet 5")

════════════════════════════════════════════════════════════════════════════════
""")

    game_data = {
        "turns": []
    }

    # Collect turns
    while True:
        print("\n📍 Entrez les données du tour (ou 'done' pour terminer):")
        turn_input = input("Tour #: ").strip()

        if turn_input.lower() == "done":
            break

        try:
            turn_num = int(turn_input)
        except ValueError:
            print("❌ Numéro invalide")
            continue

        turn_data = {
            "turn": turn_num,
            "mahdieh_ships": input(f"  Ships Mahdieh (tour {turn_num}): ").strip(),
            "kovi_ships": input(f"  Ships kovi (tour {turn_num}): ").strip(),
            "yuto083_ships": input(f"  Ships yuto083 (tour {turn_num}) [opt]: ").strip() or None,
            "arne_ships": input(f"  Ships Arne (tour {turn_num}) [opt]: ").strip() or None,
            "key_event": input(f"  Événement clé (tour {turn_num}): ").strip(),
        }

        # Validate numeric inputs
        try:
            turn_data["mahdieh_ships"] = float(turn_data["mahdieh_ships"])
            turn_data["kovi_ships"] = float(turn_data["kovi_ships"])
            if turn_data["yuto083_ships"]:
                turn_data["yuto083_ships"] = float(turn_data["yuto083_ships"])
            if turn_data["arne_ships"]:
                turn_data["arne_ships"] = float(turn_data["arne_ships"])
        except ValueError:
            print("❌ Nombre invalide, réessayez")
            continue

        game_data["turns"].append(turn_data)
        print(f"✅ Tour {turn_num} enregistré")

    return game_data

def save_and_analyze(game_data):
    """Save collected data and run analysis"""

    output_file = "replay_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(game_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Données sauvegardées: {output_file}")

    # Quick analysis
    turns = game_data["turns"]
    if not turns:
        print("⚠️  Aucune donnée collectée")
        return

    print("\n" + "="*80)
    print("📊 ANALYSE RAPIDE DES DONNÉES")
    print("="*80)

    for turn in turns:
        turn_num = turn["turn"]
        mahdieh = turn["mahdieh_ships"]
        kovi = turn["kovi_ships"]
        ratio = kovi / mahdieh if mahdieh > 0 else 0
        event = turn["key_event"]

        print(f"\n  Tour {turn_num}: Mahdieh {mahdieh:.0f} | kovi {kovi:.0f} (ratio: {ratio:.2f}x)")
        if event:
            print(f"    → {event}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--file":
        # Read from file if provided
        try:
            with open(sys.argv[2], "r", encoding="utf-8") as f:
                game_data = json.load(f)
            save_and_analyze(game_data)
        except (FileNotFoundError, IndexError):
            print("❌ Fichier non trouvé")
    else:
        # Interactive mode
        game_data = interactive_turn_capture()
        if game_data["turns"]:
            save_and_analyze(game_data)

if __name__ == "__main__":
    main()
