# Orbit Wars – Neural Network Documentation

## Contexte du projet

Compétition Kaggle **Orbit Wars** : jeu de stratégie spatiale 2 joueurs. Les planètes orbitent autour d'un soleil central. Chaque tour, le bot envoie des flottes (ships) depuis ses planètes vers d'autres planètes (expand = neutre, attack = ennemi, support = allié). L'action est un triplet `[source_id, angle, ships]` — le serveur Kaggle utilise cet angle pour déplacer la flotte en ligne droite à la vitesse calculée.

**Objectif courant** : atteindre winrate ≥ 0.80 vs les 3 opponents d'éval (random, greedy, starter).

---

## Architecture du modèle

- **Entrée** : `encode_game_state()` → vecteur 2539 features (planètes, flottes, méta)
- **Backbone** : `Linear(2539→320) + ReLU + 3×ResidualBlock(320)`
- **Value head** : scalaire (baseline PPO)
- **Candidate head** : `Linear(hidden+16→1)` par candidat → logits de politique
- **Candidates** : jusqu'à ~200 actions possibles par tour, chacune avec 16 `score_features`
  - `score_features[-1]` = prior normalisé (`_candidate_prior(c, game) / 3.0`)
  - Le prior pénalise do_nothing (-2.5 si actions disponibles), favorise expand/attack proche avec bons ratios

**Algorithme** : PPO (clip_eps=0.2, 3 epochs, minibatch=96)  
**Reward** : `clip(terminal + 0.15×dense + 0.05×activity, -1.2, 1.2)`  
**Inférence** : serveur GPU avec batch_size=64, timeout=10ms, 10 workers

---

## Fichiers clés

| Fichier | Rôle |
|---|---|
| `neural_network_gpu/run_gpu_2p_simple_local.ps1` | Script de lancement — tous les hyperparamètres |
| `neural_network_gpu/scripts/run_gpu.py` | Orchestrateur principal (PPO loop, eval, rollback) |
| `neural_network_gpu/src/gpu_trainer.py` | Calcul PPO, reward shaping |
| `neural_network_gpu/src/inference_server.py` | Serveur GPU, batching, prior injection |
| `neural_network_gpu/src/vec_worker.py` | Workers (1 game/worker), collecte trajectoires |
| `neural_network/src/policy.py` | Génération candidates, prior, choose_action |
| `neural_network/src/trajectory.py` | Prédiction trajectoire + angle de tir |
| `neural_network/src/notebook_4p_training.py` | `_candidate_move` : calcul angle → action jeu |
| `local_simulator/official_fast.py` | Simulateur local = vrai moteur Kaggle (orbit_wars_official.py) |

> **SimGame.py est banni** : simulateur expérimental non conforme à Kaggle.  
> Le training utilise exclusivement `official_fast` qui wrap le vrai `orbit_wars_official.py`.

---

## Mécaniques du jeu importantes

- **Vitesse des flottes** : variable selon le nombre de ships envoyés  
  `speed = 1.0 + 5.0 × (log(ships) / log(1000))^1.5`, max = 6.0  
  Exemples : 10 ships → 1.96 u/tour, 50 ships → 3.13, 1000 ships → 6.0
- **Capture** : une flotte atterrit si son segment de déplacement passe à moins de `planet_radius` du centre de la planète
- **Planètes en orbite** : position calculée depuis `initial_planets` + `angular_velocity × turn`
- **Action envoyée** : `[src_id, angle, ships]` — l'angle est calculé localement par le bot pour intercepter la planète à sa position future

---

## Bugs diagnostiqués et corrigés

### Bug 1 — Prior do_nothing jamais appliqué (CRITIQUE)
**Fichier** : `neural_network/src/policy.py`  
**Problème** : `do_nothing_candidate` créé avec `np.zeros(16)` → `score_features[-1] = 0.0` → le prior négatif (-2.5) n'était jamais injecté dans les logits GPU.  
**Fix** : Appeler `_candidate_prior(do_nothing_cand, game)` et stocker dans `score_features[-1]` avant d'ajouter le candidat.  
**Impact** : Delta logit expand vs do_nothing passé de 0.69 → 2.45 (×3.5 avec prior_strength=0.55).

### Bug 2 — Trajectoire calculée avec vitesse fixe 6.0 (CRITIQUE)
**Fichier** : `neural_network/src/trajectory.py`  
**Problème** : `FLEET_SPEED = 6.0` utilisé pour toute prédiction d'intercept. Mais la vraie vitesse dépend du nombre de ships. Avec 25 ships (cas typique), vitesse réelle ≈ 2.6 — la flotte arrive 2.3× plus tard que prédit → la planète orbitale est ailleurs → miss systématique.  
**Fix** : Ajout de `_fleet_speed(ships)` dans `trajectory.py`. `safe_plan_shot` accepte désormais `ships: int = 0` et utilise la bonne vitesse. `_candidate_move` dans `notebook_4p_training.py` passe `ships=int(ships)`.  
**Note** : Les appels de pré-filtre dans `policy.py` (checks sun-blocking) gardent `ships=0` → speed=6.0, car c'est une heuristique, pas le calcul final.

### Bug 3 — Température non réinitialisée après rollback
**Fichier** : `neural_network_gpu/scripts/run_gpu.py`  
**Problème** : `policy_version` accumulait → température décroissante par schedule ne se réinitialisait pas → après rollback, exploration trop faible.  
**Fix** : `policy_version = 0`, `cfg["temperature_start"] = cfg["temperature_start_initial"]`, `train_history.clear()` dans le bloc rollback.

### Bug 4 — max_eval_do_nothing_rate trop strict
**Problème** : `0.55` bloquait toutes les promotions (tous les candidats avaient noop > 55%).  
**Fix** : Monté à `0.90`.

### Bug 5 — rollback_on_noop_rate trop agressif
**Problème** : `0.78` déclenchait des rollbacks spurieux sur variance.  
**Fix** : Monté à `0.97`.

---

## Paramètres actuels du run

```powershell
--device cuda
--duration-minutes 480
--workers 10                        # était 6
--train-every 32
--eval-every 512                    # était 256 → réduit rollbacks spurieux
--eval-episodes 32                  # était 16 → CI plus serrée
--batch-size 64
--batch-timeout 0.010
--ppo-minibatch-size 96
--learning-rate 0.00008
--min-lr 0.00002
--ppo-epochs 3
--n-players 2
--simple-opponents random×5,greedy×9,starter×6   # était random×6,greedy×12,starter×2
--eval-opponents random,greedy,starter
--disable-support-actions
--auto-tune-training
--policy-prior-strength 0.20
--target-winrate 0.80
--max-eval-do-nothing-rate 0.90
--rollback-on-noop-rate 0.97
--min-eval-avg-ships-sent 3.0
--rollback-margin 0.35
--max-opponent-regression 0.35
--min-ci-promotion-games 128        # était 96
--run-name gpu_2p_rg_nosupport_local
```

---

## État actuel du training (~20k épisodes)

- **best_validated** : modèle passif, noop ~95-97%, winrate agrégé ~0.70 (random=0.94, greedy=0.88-1.00, starter=0.06-0.19)
- **Stratégie dégénérée** : le bot ne tire presque jamais, profite de la production pendant que l'adversaire gaspille ses ships sur des tirs manqués
- **Problème** : le cycle rollback (rollback → candidate diverge → re-rollback) empêchait la convergence vers jeu actif
- **Avec les nouveaux paramètres** (eval-every=512, eval-episodes=32) : le candidate a maintenant ~15min de training avant jugement, et l'éval sur 32 jeux réduit la variance (moins de rollbacks spurieux)

---

## Direction attendue

### Court terme (2-4h de run)
- Le candidate doit apprendre que les tirs atterrissent maintenant (fix trajectoire)
- noop en éval devrait descendre sous 50% puis 40%
- Premier PROMOTED attendu quand le candidate bat best_validated en winrate agrégé
- **Signal positif** : `missions.expand > missions.do_nothing` dans les train logs

### Moyen terme
- Amélioration vs starter (objectif : >35%) — c'est le vrai indicateur de jeu actif
- Enchaîner plusieurs PROMOTED successifs vers winrate 0.60 → 0.70 → 0.80

### Si bloqué après 2-3h (noop reste >50% en éval)
- Envisager fresh start (épisodes=0) — les poids auraient un prior passif trop fort à désapprendre
- Sinon : augmenter `--noop-penalty` et `--action-bonus` dans auto-tune

---

## Ressources de diagnostic

- **Log principal** : `runs/gpu_2p_rg_nosupport_local/gpu_train.log`
- **Replays Kaggle** : `neural_network_gpu/replays/` (format JSON, durées par tour)
- **Checkpoints** : `runs/gpu_2p_rg_nosupport_local/latest.npz`, `candidate.npz`, `candidate_failed_*.npz`
- **Taux réel de jeux** : ~2000-2500 épisodes/heure (le `eps_per_hour` affiché est faux — il divise les épisodes totaux cumulés par l'elapsed de la session courante)

---

## Ce qu'on ne fait PAS

- Pas de SimGame.py (simulateur non conforme)
- Pas de support actions (`--disable-support-actions`)
- Pas de modification des poids directement — tout passe par PPO + rollback gate
