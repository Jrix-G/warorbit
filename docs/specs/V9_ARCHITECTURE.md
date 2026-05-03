# War Orbit V9 Architecture

Date: 2026-05-03

V9 remplace l'approche V8.5 "ranker sur plans limites" par une politique plus
robuste: generation de candidats strategiquement differents, scoring hybride,
evaluation separee et adaptation automatique quand le benchmark externe ne suit
pas l'eval interne.

Le but prioritaire n'est pas d'augmenter le score train. Le but est que
l'evaluation interne correle avec le benchmark et que le bot ne s'effondre pas
contre des adversaires notebooks non vus.

## 1. Diagnostic qui a motive V9

Symptomes observes:

- train monte vite jusqu'a 0.85-1.00;
- eval interne peut rester tres haute;
- benchmark externe peut tomber autour de 0.00-0.31;
- en 4p, les pertes arrivent surtout par dispersion multi-fronts et manque de
  consolidation avant la finition.

Conclusion: le pipeline apprenait des comportements gagnants contre son petit
pool, mais pas des strategies robustes contre des styles forts.

V9 traite donc train, eval et benchmark comme trois distributions separees.
Une promotion n'est acceptee que si le benchmark est suffisant et si l'ecart
eval-benchmark reste sous controle.

## 2. Structure de code

```text
war_orbit/
  agents/v9/
    planner.py        # candidats strategiques et simulation de plans
    evaluator.py      # extraction des features et evaluation locale
    policy.py         # scoring hybride, lock 4p, diagnostics d'action
    adaptation.py     # reaction stagnation / generalization failure
  config/
    v9_config.py      # hyperparametres, pools, timeouts, checkpoints
  evaluation/
    benchmark.py      # benchmark final et logs par adversaire
  features/
    state_features.py
    plan_features.py
  optimization/
    search.py
    tuning.py
  training/
    curriculum.py     # pools train/eval/benchmark et cross-play
    self_play.py      # execution des matchs et agregation des stats
    trainer.py        # boucle ES, promotion, sauvegarde, logs
run_v9.py             # entree unique training + evaluation
bot_v9.py             # wrapper agent
```

## 3. Separation train / eval / benchmark

Les pools sont separes dans `war_orbit/config/v9_config.py`.

- `training_opponents`: adversaires de shaping, bruit, bots simples, V7 et
  notebooks d'entrainement.
- `eval_opponents`: adversaires held-out utilises pour verifier la progression
  sans fuite directe depuis train.
- `benchmark_opponents`: pool dur, principalement notebooks forts et combos 4p.

Le benchmark par defaut est maintenant a 128 parties:

```text
benchmark_games = 128
min_promotion_benchmark_games = 128
opponent_pool_limit = 15
four_player_ratio = 0.80
benchmark_four_player_ratio = 0.80
```

Une generation peut donc etre lente. Le timeout dur garantit quand meme une
sauvegarde propre a 60 minutes maximum.

## 4. Tactiques V9 4p ajoutees

### 4p backbone

Plan: `v9_4p_backbone`

Objectif: reproduire ce que les bons agents 4p font naturellement: beaucoup de
transferts ami -> ami pour construire une colonne centrale et eviter de tout
envoyer depuis des planetes faibles.

Signal attendu dans les logs:

```text
bb >= 0.15
xfer >= 0.30
```

### Single front lock

La politique choisit un ennemi focus et un front anchor, puis garde ce focus
pendant `front_lock_turns` tours, par defaut 24.

Objectif: ne pas alterner entre trois ennemis en 4p. On veut finir ou neutraliser
un front avant d'en ouvrir un autre.

Signal attendu:

```text
lock >= 0.90
fronts <= 2.0
```

### Consolidation threshold

Plan: `v9_front_lock_consolidation`

Objectif: si le reseau possede trop de planetes fragiles, trop peu de garrison
ou trop de fronts actifs, V9 privilegie les transferts, la defense locale et la
construction du hub avant une attaque.

Ce point corrige le comportement observe ou le bot semblait "gagner" localement
mais se faisait ensuite etouffer par deux adversaires forts.

### Staged finisher

Plan: `v9_staged_finisher`

Objectif: terminer un adversaire faible sans all-in premature. Le plan stage
d'abord vers le front, puis engage seulement quand le lead ships/prod et le
timing sont suffisants.

Garde-fou actuel: pas de finisher trop tot sauf lead clair apres le midgame.

### Denial et delayed strike focus

Les plans `v9_resource_denial`, `v9_delayed_strike`, `v9_deep_staging` et les
finishers utilisent maintenant le meme focus enemy/front anchor. Ils doivent
donc produire des attaques compatibles avec la consolidation, pas des coups
opportunistes disperses.

## 5. Scoring et anti-overfitting

V9 ne s'appuie pas seulement sur un score lineaire.

Le choix final combine:

- score de base du plan;
- features d'etat et de plan;
- bonus metadata pour `backbone`, `front_lock`, `consolidation_threshold` et
  `staged_finisher`;
- penalites pour attaque dispersee en 4p midgame;
- simulation multi-step selon les parametres runtime;
- bruit et regularisation pendant training.

Pendant training, le score est regularise:

- penalite L2 contre exces de confiance;
- penalite si l'entropie de plans est trop faible;
- penalite si un plan dominant prend trop de place;
- reward noise pour eviter la memorisation exacte des seeds.

## 6. Adaptation automatique

La boucle detecte deux problemes differents:

- stagnation: pas de progression recente;
- generalization failure: train/eval hauts mais benchmark bas.

En cas de generalization failure, V9:

- augmente l'exploration;
- augmente la diversite candidats;
- pousse les familles `staging_transfer` et `defensive_consolidation`;
- applique un reset partiel des poids vers les defaults;
- refuse de promouvoir le checkpoint tant que le benchmark ne confirme pas.

Critere de promotion:

```text
selection_score = min(eval_mean, benchmark_mean)
benchmark_mean >= min_benchmark_score
benchmark_games >= min_promotion_benchmark_games
eval_mean - benchmark_mean <= max_generalization_gap
```

## 7. Logs V9

Au demarrage, V9 affiche les cibles 4p:

```text
V9 4p diag targets xfer>=0.30 bb>=0.15 lock>=0.90 fronts<=2.0
```

Chaque generation affiche maintenant un diagnostic lisible:

```text
gen=0004 train=0.900 (...) eval=0.750 (...) bench=0.312 best=1.000
grad=46.54 promo=0 explore=0.15 div=1.39
4pdiag=WARN xfer=0.34/0.30+ bb=0.29/0.15+ lock=1.00/0.90+ fronts=3.5/2.0-
elapsed_min=13.6
```

Lecture:

- `xfer`: part des moves qui sont des transferts ami -> ami.
- `bb`: part des tours ou un plan backbone/staging 4p est actif.
- `lock`: part des tours 4p avec un focus enemy/front lock actif.
- `fronts`: nombre moyen de fronts actifs.
- `OK`: les quatre cibles sont respectees.
- `WARN`: au moins une cible est hors zone.

Interpretation directe:

- `xfer < 0.30`: le bot attaque trop directement, pas assez de consolidation.
- `bb < 0.15`: le backbone 4p n'est pas assez choisi.
- `lock < 0.90`: le focus change trop souvent ou ne s'active pas.
- `fronts > 2.0`: le bot se disperse encore en multi-front.

Ces diagnostics sont aussi ecrits dans le JSONL:

```text
evaluations/v9_robust_train.jsonl
```

Champs importants:

- `train_4p_diag`
- `eval_4p_diag`
- `benchmark_4p_diag`
- `train_transfer_move_frac`
- `train_backbone_turn_frac`
- `train_front_lock_turn_frac`
- `train_active_front_avg`
- `benchmark_mean`
- `generalization_gap`
- `stop_reason`

## 8. Commande run 1h

Commande recommandee pour un run local exploitable en 1h:

```powershell
python .\run_v9.py --minutes 60 --hard-timeout-minutes 60 --pairs 5 --games-per-eval 2 --eval-games 32 --benchmark-games 16 --min-promotion-benchmark-games 16 --benchmark-progress-every 1 --max-steps 160 --eval-max-steps 220 --four-player-ratio 0.80 --pool-limit 15
```

Notes:

- `--hard-timeout-minutes 60` coupe quoi qu'il arrive autour de 1h et sauvegarde
  `latest` + `export_checkpoint`.
- V9 execute actuellement les matchs en sequence. Avec `--benchmark-games 128`
  et `benchmark_four_player_ratio=0.80`, le premier benchmark peut consommer
  l'heure entiere sur Windows/local CPU.
- `--benchmark-progress-every 1` affiche une ligne apres chaque partie benchmark
  terminee.
- Si le timeout arrive pendant l'evaluation, le script sauvegarde le dernier
  etat connu et peut sauter l'eval finale.

Benchmark complet 128 parties, a lancer separement quand on veut une mesure plus
stable:

```powershell
python .\run_v9.py --skip-training --benchmark-games 128 --benchmark-progress-every 1 --workers 8 --eval-max-steps 220 --four-player-ratio 0.80 --pool-limit 15
```

### Mode volume pur

Pour produire beaucoup plus d'updates en 1h, il faut separer apprentissage et
validation. Cette commande desactive eval/benchmark pendant le run, reduit les
parties train et lance les matchs en parallele:

```powershell
python .\run_v9.py --minutes 60 --hard-timeout-minutes 60 --train-only --workers 8 --pairs 24 --games-per-eval 8 --max-steps 80 --four-player-ratio 0.80 --train-search-width 3 --train-simulation-depth 0 --train-simulation-rollouts 0 --front-lock-turns 15 --train-opponents random noisy_greedy greedy starter distance sun_dodge structured --no-resume --checkpoint evaluations\v9_volume_latest.npz --best-checkpoint evaluations\v9_volume_best.npz --export-checkpoint evaluations\v9_volume_policy.npz --log-jsonl evaluations\v9_volume_train.jsonl
```

Lecture:

- `--train-only`: aucune evaluation couteuse dans la boucle.
- `--workers 8`: parties executees en parallele.
- `--pairs 24 --games-per-eval 8`: 384 parties train par generation.
- `--max-steps 80`: signal court, beaucoup plus de parties par heure.
- `--train-opponents ...`: adversaires legers locaux; les notebooks sont gardes
  pour le benchmark separe.

Ce mode sert a apprendre vite, pas a juger le bot. Apres le run volume:

```powershell
python .\run_v9.py --skip-training --export-checkpoint evaluations\v9_volume_policy.npz --benchmark-games 128 --benchmark-progress-every 1 --workers 8 --eval-max-steps 220 --four-player-ratio 0.80 --pool-limit 15
```

## 9. Comment juger le prochain run

Ne pas lire seulement `train`.

Ordre de priorite:

1. `benchmark_mean` doit monter et ne pas rester a 0.00-0.31.
2. `generalization_gap` doit rester <= 0.30.
3. `4pdiag` doit passer vers OK ou au moins montrer `xfer`, `bb`, `lock` bons.
4. `fronts` doit baisser au fil des generations, idealement sous 2.0.
5. Les lignes par adversaire doivent montrer moins de combos a 0.000.

Un run avec train=0.90, eval=0.75, benchmark=0.31 reste un echec de
generalisation meme si le train est bon.

Un run moins spectaculaire mais avec benchmark qui progresse et fronts qui
baisse est meilleur pour V9.

## 10. Validation locale

Commandes de validation:

```powershell
python -m compileall -q war_orbit
python -m pytest test_v9_robustness.py test_v9_smoke.py test_bot_v8_5_policy.py -q
```

Le smoke 4p attendu doit montrer une hausse claire des transferts et du
backbone. Exemple de zone correcte:

```text
xfer ~= 0.30+
bb ~= 0.15+
lock ~= 0.90+
fronts en baisse, cible <= 2.0
```

## 11. Points encore a analyser apres run

Apres le run 1h, il faut comparer les echecs par label adversaire:

- combos avec `notebook_orbitbotnext`;
- combos avec `notebook_physics_accurate`;
- combos avec `notebook_djenkivanov...sniper`;
- cas ou `xfer` est bon mais `fronts` reste eleve;
- cas ou `fronts` baisse mais le benchmark reste bas.

Si `xfer`, `bb` et `lock` sont bons mais `fronts` reste > 2.0, le prochain patch
doit durcir la selection d'un unique enemy focus et penaliser les commits hors
front anchor.

Si `fronts` est bon mais benchmark bas, le probleme sera plutot la qualite de
finition: staged finisher trop lent, mauvaise estimation des flottes en route,
ou mauvais timing contre les snipers.
