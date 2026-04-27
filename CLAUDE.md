# Contexte projet warorbit

## MCP Playwright configuré

Playwright MCP installé globalement via `~/.mcp.json`:
```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"]
    }
  }
}
```

- Package: `@playwright/mcp@latest`
- Scope: global (tous projets)
- Browser: `chrome-for-testing` (installé via `npx @playwright/mcp install-browser chrome-for-testing`)
- Replays Kaggle: derrière login, browser MCP ne peut pas les lire (nouvelle session sans auth)

## Limitations browser

- Claude ne peut pas lire Firefox déjà ouvert (OS process isolation)
- Playwright lance nouveau browser séparé — session distincte
- Pour pages avec login: fournir credentials ou URL publique

---

## Projet: Orbit Wars (Kaggle competition)

URL: https://www.kaggle.com/competitions/orbit-wars

### Résumé du jeu

- Espace 2D 100×100, soleil au centre (rayon 10, centre 50,50)
- 2 ou 4 joueurs. 500 tours max. Victoire = plus de ships total (planètes + flottes)
- Planètes: statiques ou en orbite. Production 1-5 ships/tour
- Flottes: ligne droite, vitesse logarithmique (1 ship=1u/t, 1000 ships=6u/t max)
- Comètes: traversent le plateau tours 50/150/250/350/450, capturables temporairement
- Combat: plus grande force gagne, différence survit. Tie = tous morts
- **PyTorch NON DISPONIBLE dans l'env d'évaluation** → numpy/stdlib seulement

### API agent

```python
# Input
obs.planets    # [[id, owner, x, y, radius, ships, production], ...]
obs.fleets     # [[id, owner, x, y, angle, from_planet_id, ships], ...]
obs.player     # ton ID (0-3)
obs.angular_velocity    # vitesse rotation planètes (rad/tour)
obs.initial_planets     # positions planètes au début
obs.comets              # [{planet_ids, paths, path_index}, ...]
obs.comet_planet_ids    # IDs planètes = comètes
obs.remainingOverageTime  # budget temps restant (s)

# Output
return [[from_planet_id, direction_angle_radians, num_ships], ...]
```

### État compétition (27 avril 2026)

- #1 kovi: 2577 ELO (énorme écart sur #2 à 1623)
- Top 10: 1313–2577 ELO
- 1504 équipes, deadline 23 juin 2026
- ELO 1500 = top 5 actuellement (très ambitieux)
- ELO 800-1100 = objectif V1/V2 réaliste

---

## Architecture du code

### Fichiers

| Fichier | Rôle | Soumettre ? |
|---------|------|-------------|
| `bot.py` | Agent principal avec 14 poids tunables | Oui (avec WEIGHTS=None) |
| `bot_submit.py` | bot.py avec poids CMA-ES hardcodés | **Oui (celui-là en priorité)** |
| `train.py` | CMA-ES + tests vs baselines | Non |
| `best_weights.json` | Meilleurs poids sauvegardés | Non |
| `stratégie.md` | Analyse compétitive détaillée | Non |

### bot.py — Fonctionnalités V2

- **14 poids tunables** via `WEIGHTS` global (None = poids défaut `DEFAULT_W`)
- **Coordination globale** via dict `committed[target_id]` (pas de doublons)
- **Stratégie 4P**: pénalise attaque du leader (`W[7]`), bonus ennemi faible (`W[8]`)
- **Sun-dodging**: waypoint latéral autour soleil, puis fallback angle dévié
- **Prédiction orbite continue**: `predict_pos()` avec tours flottants
- **Threat ETA**: urgence = `1 / (1 + eta × W[11])` → menaces proches prioritaires
- **Mode dominant**: si `my_ships > max_enemy × W[10]`, défense conservative
- **Comet bonus**: `W[1]` multiplicateur si cible est comète

### Poids DEFAULT_W[14]

```python
W[0]  = 2.0   # neutral_priority   : bonus neutres vs ennemis
W[1]  = 1.5   # comet_bonus        : bonus comètes
W[2]  = 40.0  # production_horizon : tours futurs estimés dans gain
W[3]  = 0.3   # distance_penalty   : coût par unité de distance
W[4]  = 0.15  # defense_reserve    : fraction ships gardée si menacé
W[5]  = 1.3   # attack_ratio       : ships_needed × ce ratio
W[6]  = 0.6   # fleet_send_ratio   : fraction ships envoyée
W[7]  = 0.5   # leader_penalty     : pénalité attaque joueur dominant
W[8]  = 0.4   # weak_enemy_bonus   : bonus si ennemi <30% de nos ships
W[9]  = 0.05  # sun_waypoint_dist  : distance waypoint (× SUN_RADIUS)
W[10] = 0.8   # endgame_threshold  : ratio ships pour mode défense
W[11] = 0.25  # threat_eta_factor  : poids ETA dans urgence menace
W[12] = 1.2   # reinforce_ratio    : seuil renforts défense
W[13] = 0.5   # neutral_ships_cap  : max ships neutres / nos ships
```

---

## Entraînement CMA-ES

### Commandes

```bash
# Setup
pip install kaggle-environments numpy cma --break-system-packages

# Tests
python3 train.py --test          # poids défaut, 30 parties/adversaire
python3 train.py --test-best     # meilleurs poids + génère bot_submit.py

# Entraînement (choisir profil)
python3 train.py --quick         # ~15 min, 20 générations, 6 candidats, 2 parties
python3 train.py --medium        # ~1-2h, 80 générations, 8 candidats, 5 parties
python3 train.py --long          # ~4-6h, 300 générations, 12 candidats, 10 parties
python3 train.py --jobs 4        # forcer 4 processus parallèles (1 par CPU)

# Générer bot_submit.py sans relancer les tests
python3 train.py --generate

# Reprendre un entraînement interrompu (lit best_weights.json automatiquement)
python3 train.py --medium
```

### Profils CMA-ES

| Profil | Générations | Candidats | Parties/cand | Durée est. |
|--------|-------------|-----------|--------------|------------|
| quick  | 20          | 6         | 2            | ~15 min    |
| medium | 80          | 8         | 5            | ~1-2h      |
| long   | 300         | 12        | 10           | ~4-6h      |

### Score CMA-ES

Score = win-rate pondéré:
- vs greedy × 1.0
- vs self × 2.0 (self-play compte double)
- vs random × 0.5

Score 0.0 = perd tout, 1.0 = gagne tout. Cible: >0.65

### Workflow complet

```
1. python3 train.py --quick   # validation rapide
2. python3 train.py --medium  # entraînement sérieux
3. python3 train.py --test-best  # voir résultats + génère bot_submit.py
4. Soumettre bot_submit.py sur Kaggle
5. Analyser replays → identifier bugs → améliorer bot.py
6. Recommencer depuis 1
```

---

## Résultats connus

| Version | Poids | vs greedy | vs self | ELO estimé |
|---------|-------|-----------|---------|------------|
| V1 défaut | DEFAULT_W V1 | 95% | 60% | ~900-1100 |
| V2 défaut | DEFAULT_W V2 | 57% | 50% | ~800-1000 |
| V2 post-CMA-ES | à tuner | ~75-85% cible | - | ~1200-1500 |

V2 défaut < V1 car plus de paramètres → besoin tuning CMA-ES.

---

## Prochaines améliorations identifiées (V3+)

Pour dépasser 1500 ELO:

1. **Simulateur numpy vectorisé local** — simuler N tours en <100ms → beam search à l'inférence
2. **Beam search / MPC** — evaluer K plans candidats par tour, picker le meilleur
3. **Opponent modeling** — télécharger replays top-10 via `kaggle 2.0.2`, apprendre patterns
4. **League self-play** — garder snapshots historiques pour éviter overfitting
5. **Endgame solver** — last ~50 tours: search exhaustif possible (espace réduit)
6. **4P kingmaker logic** — attaquer #2, laisser #1 et #3 se bagarrer

Pour télécharger replays: voir discussion Kaggle "kaggle 2.0.2 is now available" (pinned).

---

## Notebooks publics clés

| Notebook | Votes | Intérêt |
|----------|-------|---------|
| Getting Started (officiel, Bovard) | 549 | Base obligatoire |
| Structured Baseline (Pilkwang Kim) | 139 | Meilleure archi rule-based publique |
| RL Tutorial (YumeNeko/kashiwaba) | 119 | PPO numpy reference |
| Orbit Star Wars \| LB: MAX 1224 | 96 | Approche atteignant 1224 ELO |
| [LB 958.1] REINFORCE (sigmaborov) | 88 | RL REINFORCE, score 958 |
| Sun-Dodging Baseline | 58 | Sun avoidance implémentation |
| Distance-Prioritized [LB 1100] | 49 | Simple mais efficace |
