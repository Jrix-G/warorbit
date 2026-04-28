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

### État compétition (28 avril 2026)

- #1 kovi: 2577 ELO (énorme écart sur #2 à 1623)
- Top 10: 1313–2577 ELO
- 1504 équipes, deadline 23 juin 2026
- ELO 1500 = top 5 actuellement (très ambitieux)
- ELO ~600-800 = état actuel estimé (bot V6 0/30 vs notebooks forts)

---

## Architecture du code (état 28 avril 2026)

### Fichiers actifs

| Fichier | Rôle | Soumettre ? |
|---------|------|-------------|
| `submission.py` | **Bot actif à soumettre** | **OUI** |
| `bot_v7.py` | **Source principale V7 — WorldModel + arrival ledger** | Dev |
| `bot_v6.py` | Source V6 (beam_search + NumpyEvaluator) | Référence |
| `bot_v6_submited.py` | Sauvegarde de la version soumise V6 | Référence |
| `SimGame.py` | Simulateur local rapide (~5s/partie V7 vs notebook) | Non |
| `opponents/` | Zoo de 12 bots adversaires | Non |
| `replay_observer.py` | Capture JSON replay depuis Kaggle via Playwright | Non |
| `evaluations/` | Évaluateurs numpy sauvegardés | Non |

### bot_v7 — Architecture V7 (état 28 avril 2026)

**Changement clé vs V6**: WorldModel avec arrival ledger. Toutes les décisions basées sur état futur projeté, pas état courant.

Composants:

- **`WorldModel`**: construit chaque tour — arrival_ledger, timelines par planète, réserve défensive
- **`build_arrival_ledger`**: mappe chaque flotte en transit vers sa planète cible + ETA
- **`simulate_planet_timeline`**: simule état futur d'une planète sur HORIZON=80 tours (production + arrivées de flottes)
- **`ships_needed_to_capture(planet, eta)`**: ships nécessaires basé sur état projeté à l'arrivée
- **`keep_needed`**: garrison minimum calculé par binary search sur timeline (remplace `prod*eta` approximatif)
- **`plan_moves`**: planning global avec `planned_commitments` — évite double-comptage entre missions
- **Multi-source swarms**: coordonne 2/3 sources sur même cible avec tolérance ETA
- **Reinforcement missions**: détecte planètes menacées, envoie renforts
- **Rear staging**: transfère ships des planètes arrière vers le front
- **Tuning 2-joueurs**: `TWO_PLAYER_HOSTILE_AGGRESSION_BOOST=1.35`, `TWO_PLAYER_NEUTRAL_MARGIN_BASE=1`, `TWO_PLAYER_OPENING_TURN_LIMIT=60`
- **`NumpyEvaluator`**: MLP 24→32→16→1 conservé — sera utilisé pour REINFORCE (étape suivante)

Optimisations performance:
- `_seg_hits_sun`: comparaison dist² (évite sqrt)
- `_cos_sin`: cache trig avec dict (évite math.cos/sin répétés)
- Early-exit timeline si aucune arrivée ennemie (skip binary search)
- HORIZON=80 (vs 150 dans notebook_physics_accurate)

---

## Résultats benchmark

### V7 vs V6 vs notebooks (5 parties chacun, 28 avril 2026)

| Adversaire | V6 | V7 |
|-----------|:--:|:--:|
| notebook_orbitbotnext | 0/5 | 1/5 |
| notebook_distance_prioritized | 0/5 | 1/5 |
| notebook_physics_accurate | 0/5 | 2/5 |

**V7 > V6 sur tous les notebooks.** Gap restant = scoring policy pas encore entraîné.

### V6 vs Zoo local (10 parties chacun, référence)

| Adversaire | V6 |
|-----------|:--:|
| passive/random/starter | 10/10 |
| greedy | 6/10 |
| distance/structured/orbit_stars/sun_dodge | 5-6/10 |
| notebooks forts | 0/10 |

### Root cause gap vs notebooks

- V6: décisions sur état courant → sous-estime défenseurs à l'arrivée
- V7: décisions sur état projeté → correct, mais scoring policy encore heuristique
- Objectif 4/5: scoring policy entraîné par REINFORCE

---

## Plan REINFORCE (prochaine étape)

### Objectif

Remplacer `_target_value()` dans `bot_v7.py` par un MLP appris. Le MLP apprend les comportements émergents (agressivité 2p en early, timing d'attaque, priorité missions) sans qu'on les code à la main.

### Architecture MLP mission scorer

```python
# Features par mission (input au MLP):
features = [
    target_production / 5.0,          # valeur économique normalisée
    eta / 100.0,                       # temps de trajet
    ships_needed / 500.0,             # coût de capture
    remaining_turns / 500.0,          # urgence temporelle
    is_two_player,                     # mode 2p vs 4p
    domination_score,                  # avance/retard économique
    target_is_enemy,                   # attaque vs neutre
    target_is_static,                  # planète statique vs orbitale
    indirect_wealth_normalized,        # valeur indirecte du territoire
    my_prod_ratio,                     # ratio production totale
    # ~10-15 features total
]
# Output: scalar log-multiplier sur le score heuristique de base
# score_final = base_heuristic_score * exp(mlp(features))
# Warm-start: poids=0 → mlp(x)=0 → exp(0)=1 → comportement V7 pur au départ
```

### Algorithme REINFORCE

```
for episode in range(N):
    # Jouer une partie V7-trained vs V7-trained (self-play)
    # À chaque tour, pour chaque mission candidate:
    #   score_noisy = base_score * exp(mlp(features) + noise)  # exploration
    #   choisir missions par score_noisy (greedy)
    #   enregistrer (features, noise_used) pour chaque mission exécutée
    
    reward = +1 si victoire, -1 si défaite, 0 si nul
    
    # Mise à jour REINFORCE:
    for each (features, noise) recorded this episode:
        gradient = reward * noise  # REINFORCE policy gradient
        mlp_weights += lr * gradient * features  # backprop simplifié
```

### Comment lancer (quand 8 cores disponibles)

```bash
# Lancer training REINFORCE (à créer: train_reinforce.py)
python3 train_reinforce.py \
    --episodes 5000 \
    --workers 8 \
    --lr 0.001 \
    --save-every 500 \
    --out evaluations/mlp_v7_reinforce.npy

# Benchmark intermédiaire pendant training
python3 -c "
from SimGame import run_match
import bot_v7
from opponents import ZOO
# charger weights intermédiaires dans bot_v7._SCORER
wins = sum(run_match([bot_v7.agent, ZOO['notebook_physics_accurate']], seed=i)['winner']==0 for i in range(10))
print(f'{wins}/10')
"
```

### Fichier à créer: `train_reinforce.py`

Structure:
1. `collect_episode(agent_fn, opponent_fn, seed)` → liste de (features, actions_taken, reward)
2. `update_weights(episodes_batch, lr)` → gradient REINFORCE
3. `parallel_collect(n_workers, n_episodes)` → multiprocessing.Pool
4. Boucle principale: collect → update → benchmark every 500 eps → save weights

### Critères de succès

| Étape | Métrique |
|-------|---------|
| Warm-start OK | V7+MLP = V7 pur (0 écart) |
| Learning signal | loss décroît sur 500 épisodes |
| Amélioration locale | V7+MLP > V7 pur vs notebooks |
| Objectif | 4/5 vs notebook_physics_accurate |

### Conseils pour éviter les pièges REINFORCE

- **Variance haute**: utiliser baseline = moyenne mobile des rewards (soustrait reward moyen)
- **Self-play instable**: alterner self-play et vs ZOO (50/50) pour éviter overfitting circulaire
- **Exploration**: commencer avec noise_std=0.3, décroître vers 0.05 en fin de training
- **Batch size**: ne pas updater après chaque épisode — batcher 32-64 épisodes
- **Learning rate**: 1e-3 max, decay × 0.5 si loss diverge

---

## Système replay Kaggle

### Comment ça marche

Les replays Kaggle ne sont pas des fichiers locaux. Ils sont servis par l'API Kaggle:
- `EpisodeService/GetEpisodeReplay` → JSON complet (config, steps, observations, actions)
- URL format: `https://www.kaggle.com/competitions/orbit-wars/submissions?submissionId=X&episodeId=Y`

### Extraction via Playwright

```bash
pip install playwright --break-system-packages
playwright install chromium
python3 replay_observer.py <submissionId> <episodeId>
# ex: python3 replay_observer.py 52128366 75588964
```

Flow: browser s'ouvre → login manuel → script capture `GetEpisodeReplay` → sauvegarde `replays/episode_<id>.json`

### Soumission connue

- submissionId: `52128366`
- 13 épisodes disponibles via `ListEpisodes`

---

## Benchmark rapide

```bash
# V7 vs un adversaire, N parties
.venv/bin/python -c "
from SimGame import run_match
import bot_v7
from opponents import ZOO
opp = ZOO['notebook_physics_accurate']
wins = 0
for i in range(10):
    if i % 2 == 0:
        r = run_match([bot_v7.agent, opp], seed=42+i)
        wins += 1 if r['winner'] == 0 else 0
    else:
        r = run_match([opp, bot_v7.agent], seed=42+i)
        wins += 1 if r['winner'] == 1 else 0
print(f'{wins}/10')
"

# Voir contenu zoo
.venv/bin/python -c "from opponents import ZOO; print(list(ZOO.keys()))"
```

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
