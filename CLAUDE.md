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
- V7: décisions sur état projeté → correct, mais scoring policy heuristique + défense fragile
- Diagnostic Kaggle (1000+ parties): **21/26 pertes = mid_collapse** (garrison insuffisant, rapid loss de planètes après expansion)

---

## Stratégie ML (29 avril 2026)

### Principe: pas de REINFORCE naïf

Self-play pur → mode collapse (bot apprend à battre sa propre faiblesse, pas les vrais adversaires). Architecture choisie: **Behavioral Cloning + League Training**.

### Phase 1 — Diagnostic & Fix V7 (avant tout ML)

1. Analyser tour exact du collapse dans les parties perdues
2. Corriger `keep_needed` / garrison sous-estimé
3. Benchmark local vs 15 notebooks → baseline chiffrée

### Phase 2 — Behavioral Cloning depuis top players

Dataset: `replay_dataset/compact/episodes.jsonl.gz` (1000+ parties)

```python
# Extraire (obs_features, action) de chaque tour du GAGNANT
# MLP supervisé: état → action optimale
# Avantage: pas de variance, converge vite, warm-start sur vrais comportements gagnants
features_per_turn = [
    my_planets / total_planets,       # dominance économique
    my_ships / total_ships,           # dominance militaire
    nearest_neutral_dist,             # opportunité expansion
    nearest_enemy_dist,               # pression adverse
    turn / 500.0,                     # phase de jeu
    is_2p,                            # mode
    production_ratio,                 # eco relative
    fleet_ratio,                      # flotte relative
    # ~15-20 features globaux par tour
]
# Output: (from_planet, direction, ships_ratio) → discrétisé
```

### Phase 3 — League Training (no ceiling)

```
Pool = [V7_base, cloned_model, checkpoint_1, ...]

loop indéfini:
    opponent = random(Pool)          # jamais soi-même seul
    jouer N parties (SimGame local)
    gradient update sur dense reward
    si winrate vs Pool > 60% pendant 100 parties:
        ajouter checkpoint au Pool
    Pool = keep top 10 checkpoints par ELO interne
```

**Pourquoi pas de plafond**: adversaire change chaque génération → impossible d'overfitter. Pool grandit → challenge permanent.

### Dense reward (vs win/lose binaire)

```python
reward_t = (
    delta_production_ratio * 0.4 +   # gain eco relatif
    fleet_efficiency * 0.3 +          # ships convertis / ships envoyés
    survival_bonus * 0.3              # encore vivant à T+50
)
# Signal à chaque tour → 500x plus riche que reward final
```

### Fichiers à créer

| Fichier | Rôle |
|---------|------|
| `extract_training_data.py` | extraire (features, actions) gagnants depuis episodes.jsonl.gz |
| `train_cloning.py` | behavioral cloning supervisé numpy MLP |
| `league_training.py` | league loop: SimGame + pool + checkpoints |
| `dense_reward.py` | calcul reward par tour depuis obs |

### Critères de succès

| Étape | Métrique |
|-------|---------|
| Fix V7 défense | rapid_collapse < 5/26 (était 22/26) |
| Behavioral cloning | clone > V7_base vs notebooks |
| League gen 1 | >50% vs notebooks locaux |
| League gen N | winrate Kaggle >50% |

### Commandes clés

```bash
# Harvest données (Playwright + kaggle session)
python3 harvest_replays.py --target 2000 --top-teams 0 --headless \
  --extra-submissions 52128366 51799813 52066322 [autres IDs]

# Analyser parties
python3 analyze_replays.py --mode all --my-sub 52128366

# Watch harvest en cours
python3 watch_harvest.py 1000 <PID>
```

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
