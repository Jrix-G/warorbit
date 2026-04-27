# Stratégie Orbit Wars

## Contexte compétitif (27 avril 2026)

- **#1 kovi: 2577** — énorme écart sur #2 (1623)
- Top 10: scores 1313–2577, compétition très active
- 1504 équipes, deadline 23 juin 2026

## Consensus communauté: Rule-based domine MAINTENANT

- Plusieurs top-100 utilisent rule-based pur
- Un seul (Roy Wei, ~162ème) confirme PPO qui bat rule-based en 2v1
- James McGuigan a essayé RL mais n'a pas dépassé le rule-based
- **PyTorch NON DISPONIBLE dans l'env d'évaluation** → contrainte majeure pour RL

## Analogie Stockfish

Stockfish = rule-based + search (minimax alpha-beta). Ici similaire:
- Espace déterministe et simulable
- Pas de minimax naïf possible (500 tours × multi-planètes = explosion combinatoire)
- Pas de position discrète (espace continu 100×100) → MCTS classique difficile
- **Le "Stockfish" ici = planificateur rule-based très fin + poids optimisés par évolution**

## Pourquoi pas NN pur

| Problème | Détail |
|----------|--------|
| PyTorch absent | Env d'eval = numpy only |
| Coût training | ~20-30s/partie → impossible millions d'épisodes |
| Espace d'action continu | Angle en radians → discretization non triviale |
| Signal sparse | Win/lose fin de 500 tours → reward très tardif |

**Ce qui marche**: NN léger (~1000 paramètres numpy) entraîné par **CMA-ES ou OpenAI-ES**.

---

## Plan en 3 phases

### Phase 1 — Premier bot fonctionnel (rule-based greedy)

Score attendu: ~600-800

```python
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def agent(obs):
    planets = [Planet(*p) for p in obs.get('planets', [])]
    fleets  = [Fleet(*f) for f in obs.get('fleets', [])]
    me      = obs.get('player', 0)

    moves = []
    my_planets = [p for p in planets if p.owner == me]

    for src in my_planets:
        if src.ships < 10:
            continue

        targets = [p for p in planets if p.owner != me]
        targets.sort(key=lambda t: (
            t.ships / max(src.ships, 1),  # ratio force
            distance(src, t)              # distance
        ))

        if targets:
            best = targets[0]
            angle = math.atan2(best.y - src.y, best.x - src.x)
            ships = int(src.ships * 0.6)
            moves.append([src.id, angle, ships])

    return moves
```

### Phase 2 — Améliorations heuristiques (score ~1000-1200)

- **Prédiction orbite**: utiliser `angular_velocity` + `initial_planets` pour viser position future des planètes mobiles
- **Sun-dodging**: éviter trajectoires qui croisent le soleil (rayon 10, centre 50,50)
- **Défense**: détecter flottes ennemies en approche → renforcer planètes menacées
- **Comètes**: capturer si garrison faible (production gratuite pendant passage)
- **Coordination**: ne pas envoyer 3 flottes sur même cible, distribuer les attaques
- **Timing**: ne lancer que si `src.ships > enemy.ships * 1.2 + distance * 0.1`

### Phase 3 — CMA-ES pour optimiser les poids

```python
import cma
import numpy as np

# Poids = [w_distance, w_production, w_ships_ratio, w_defend, ...]
weights_init = np.zeros(20)

es = cma.CMAEvolutionStrategy(weights_init, sigma0=0.1)

while not es.stop():
    candidates = es.ask()
    scores = [evaluate_agent(w) for w in candidates]  # self-play local
    es.tell(candidates, [-s for s in scores])          # cma minimise

# evaluate_agent: joue N parties en self-play, retourne win_rate
```

Pour aller plus loin si CMA-ES plafonne:
1. **OpenAI-ES** (NES): gradient naturel → 5-10× plus rapide que (1+λ)
2. **PPO numpy**: port manuel de PPO sans PyTorch (possible mais ~200 lignes de plus)

---

## Stack technique recommandé

```
# Dépendances
numpy          # dispo partout
cma            # pip install cma, pure Python, OK en eval
kaggle-environments  # env local pour tester

# Architecture fichiers
main_agent.py  # logique rule-based pure (soumis)
scorer.py      # petit NN numpy qui score missions (soumis)
train.py       # CMA-ES local pour optimiser scorer.py (PAS soumis)
```

---

## Notebooks publics à lire

| Notebook | Votes | Intérêt |
|----------|-------|---------|
| Getting Started (officiel) | 549 | Base obligatoire |
| Structured Baseline (Pilkwang Kim) | 139 | Meilleure archi rule-based publique |
| RL Tutorial (YumeNeko/kashiwaba) | 119 | Référence PPO si on veut tenter RL |
| Orbit Star Wars \| LB: MAX 1224 | 96 | Approche ayant atteint 1224 LB |
| [LB 958.1] REINFORCE (sigmaborov) | 88 | RL avec REINFORCE, score 958 |
| Sun-Dodging Baseline | 58 | Implémentation évitement soleil |
| Distance-Prioritized Agent [LB 1100] | 49 | Simple mais efficace |
| Physics-Accurate Planner [LB 928] | 42 | Planification avec physique exacte |

---

## Observation API (référence rapide)

```python
obs.planets    # [[id, owner, x, y, radius, ships, production], ...]
obs.fleets     # [[id, owner, x, y, angle, from_planet_id, ships], ...]
obs.player     # ton ID (0-3)
obs.angular_velocity    # vitesse rotation planètes (rad/tour)
obs.initial_planets     # positions initiales planètes
obs.comets              # [{planet_ids, paths, path_index}, ...]
obs.comet_planet_ids    # IDs planètes qui sont des comètes
obs.remainingOverageTime  # budget temps restant (secondes)

# Action
return [[from_planet_id, direction_angle_radians, num_ships], ...]
```

## Config jeu (defaults)

| Param | Valeur |
|-------|--------|
| episodeSteps | 500 |
| actTimeout | 1s/tour |
| shipSpeed max | 6.0 u/tour |
| sunRadius | 10.0 |
| boardSize | 100×100 |
| cometSpeed | 4.0 u/tour |
