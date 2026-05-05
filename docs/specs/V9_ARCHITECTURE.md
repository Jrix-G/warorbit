# War Orbit V9 Architecture

Date: 2026-05-04

V9 est le bot experimental robuste de War Orbit. Il remplace l'approche V8.5
"ranker sur plans limites" par une politique de plans strategiques, un scoring
hybride, une evaluation separee train/eval/benchmark et un guardian 4p qui
refuse les promotions faibles meme quand le train parait bon.

Le but prioritaire n'est pas de maximiser le train. Le but est de produire un
checkpoint qui tient en 4p contre des notebooks forts non vus, sans dispersion
multi-fronts.

## 1. Diagnostic V9

Les runs precedents montraient le meme pattern:

- train/eval pouvaient monter vite;
- le benchmark externe restait faible;
- en 4p, le bot faisait beaucoup de transferts et gardait un focus logique;
- pourtant il restait expose sur trop de fronts et ne convertissait pas assez en
  victoires.

Conclusion: `lock` ne suffit pas. Un lock d'ennemi doit aussi contraindre les
moves reels: attaques, snipes, denial et finisher doivent rester compatibles avec
le front anchor.

## 2. Fichiers principaux

```text
bot_v9.py                         # wrapper public de l'agent V9
run_v9.py                         # entree training/eval/benchmark
war_orbit/config/v9_config.py     # hyperparametres et flags guardian
war_orbit/agents/v9/planner.py    # generation de plans
war_orbit/agents/v9/policy.py     # scoring, lock 4p, diagnostics runtime
war_orbit/training/curriculum.py  # schedules train/eval/benchmark
war_orbit/training/self_play.py   # execution et agregation des stats
war_orbit/training/trainer.py     # ES, promotion, guardian, logs JSONL
war_orbit/evaluation/benchmark.py # benchmark separe
```

## 3. Train, eval, benchmark

Les pools sont separes dans `V9Config`:

- `training_opponents`: shaping et apprentissage;
- `eval_opponents`: held-out interne;
- `benchmark_opponents`: pool dur notebooks, utilise pour promotion.

La promotion ne depend pas du train. Elle utilise:

```text
selection_score = min(eval_mean, benchmark_mean) + ajustement 4p
```

Et elle exige:

```text
selection_score >= best_score + min_improvement
benchmark_mean >= min_benchmark_score
benchmark_games >= min_promotion_benchmark_games
eval_mean - benchmark_mean <= max_generalization_gap
eval_mean - benchmark_mean <= guardian_max_generalization_gap
benchmark_4p >= guardian_min_benchmark_4p
benchmark_backbone >= guardian_min_benchmark_backbone
benchmark_fronts <= guardian_max_benchmark_fronts
```

Depuis le patch du 2026-05-04, si `four_player_ratio >= 1.0` dans
`build_cross_play_specs`, le schedule est vraiment 4p-only. Avant, le benchmark
commencait toujours par des matchs 2p, ce qui rendait `bench=...` trop optimiste
sur les runs guardian.

## 4. Tactiques 4p

### Backbone

Plan: `v9_4p_backbone`.

Il transfere des ships ami -> ami vers une colonne/front central. Cible:

```text
xfer >= 0.30
bb >= 0.15
```

### Front lock

V9 choisit un `focus_enemy_id` et un `front_anchor_id`, puis garde ce focus
pendant `front_lock_turns` tours. Cible:

```text
lock >= 0.90
fronts <= 2.0
```

Le patch du 2026-05-04 ajoute une penalite dure en midgame 4p si un candidat
attaque un ennemi hors focus et loin du front anchor. Cela corrige le cas
`lock=0.99` mais `fronts=3.6`.

### Consolidation

Plans: `v9_front_lock_consolidation`, `v9_defensive_consolidation`,
`v9_reserve_hold`.

Ils renforcent les planetes menacees et transferent vers le front avant de
continuer l'attaque.

### Resource denial

Plan: `v9_resource_denial`.

En 4p midgame, il est maintenant limite aux cibles du focus enemy, proches du
front anchor. Sous forte pression ou en mode strict, il vise une seule cible.

### Opportunistic snipe

Plan: `v9_opportunistic_snipe`.

En 4p midgame, il ne snipe plus globalement. Il doit rester sur le focus enemy ou
pres du front anchor. Le guardian peut aussi le desactiver via
`disable_snipe_4p`.

### Staged finisher

Plan: `v9_staged_finisher`.

Quand l'ennemi focus est affaibli, le plan passe plus vite de staging a capture:
moins de transferts supplementaires, plus de sources de front, plus d'agression.

## 5. Signal ES 4p et guardian

### Signal ES 4p

Le trainer V9 utilise OpenAI-ES avec perturbations positives/negatives:

```text
grad = mean((rank(score_pos) - rank(score_neg)) * epsilon) / sigma
```

Les parties 4p sont lentes et peu nombreuses par perturbation. Avec
`games_per_eval=2` et `four_player_ratio=0.80`, chaque candidat voit souvent
seulement une ou deux parties 4p. Le signal brut est donc tres bruite.

Pour eviter que le 4p reste noye dans la moyenne globale, `_regularized_train_score`
amplifie maintenant la composante 4p quand le winrate 4p est faible:

```text
if n_4p > 0 and wr_4p < 0.35:
    score ~= weighted_mean(wr_2p, four_p_signal_boost * wr_4p)
```

Puis un bonus 4p specifique est ajoute quand l'echantillon contient au moins deux
parties 4p:

```text
wr_4p_bonus = four_p_signal_boost * max(0, wr_4p - 0.25) * (n_4p / max(n_4p, 8))
```

Effet attendu:

- sous `0.35` de WR 4p, une petite amelioration 4p pese plus dans le ranking ES;
- le seuil random 4p `0.25` sert de base pour recompenser le vrai progres;
- le facteur `n_4p / max(n_4p, 8)` limite les faux positifs quand il n'y a que
  deux ou trois parties;
- si `n_4p == 0`, aucun bonus n'est applique et le score reste valide.

Le multiplicateur est configurable:

```bash
--four-p-signal-boost 1.4
```

Valeur par defaut: `1.4`.

### Guardian strict focus

Le guardian ajuste les poids et les flags quand le benchmark ne suit pas.

Si `benchmark_4p < guardian_min_benchmark_4p`, il augmente la pression 4p sans
reduire un run deja configure a `four_player_ratio=1.0`.

L'ajustement n'est plus fixe. Il depend du deficit au seuil:

```text
deficit = max(0, guardian_min_benchmark_4p - benchmark_4p)
step = 0.02 + 0.15 * deficit
four_player_ratio += step          # cap 1.0
benchmark_four_player_ratio += step # cap 1.0
candidate_diversity += 2 * step     # cap 1.90
```

Exemple avec `guardian_min_benchmark_4p=0.42`:

- `benchmark_4p=0.36` donne `step=0.029`;
- `benchmark_4p=0.28` donne `step=0.041`;
- `benchmark_4p=0.20` donne `step=0.053`.

Le fix strict-focus est volontairement plus tardif. Si `benchmark_4p < 0.30`
pendant au moins quatre generations et que le benchmark courant est sous `0.22`,
il active:

```text
strict_single_target_4p = True
disable_snipe_4p = True
max_focus_targets_4p = 1
```

Effet:

- `resource_denial` ne disperse plus sur plusieurs cibles;
- `opportunistic_snipe` est coupe en midgame 4p;
- la policy penalise plus fortement les attaques hors focus.

Raison: entre `0.25` et `0.35`, le plateau peut etre du bruit statistique. Activer
le strict-focus apres seulement deux generations pouvait sur-contraindre le 4p
et empecher les strategies utiles de coalition, opportunisme et punition du leader.

Ces flags sont exposables depuis `run_v9.py`:

```bash
--strict-single-target-4p 1
--disable-snipe-4p 1
--max-focus-targets-4p 1
```

## 6. Logs

La ligne de generation affiche maintenant le benchmark decompose:

```text
bench=0.417 (2p 0.833/6 4p 0.278/18) sel=0.385 gap=0.166
promo=0 block=bench4p_low
```

Les raisons possibles de refus de promotion sont:

```text
score_not_improved
benchmark_low
benchmark_games_low
gap_high
guardian_gap_high
bench4p_low
bb_low
fronts_high
```

Le diagnostic 4p reste:

```text
4pdiag=OK|WARN xfer=.../0.30+ bb=.../0.15+ lock=.../0.90+ fronts=.../2.0-
```

Lecture:

- `xfer < 0.30`: pas assez de transferts/consolidation;
- `bb < 0.15`: backbone pas assez actif;
- `lock < 0.90`: focus instable;
- `fronts > 2.0`: dispersion tactique encore trop forte.

Le JSONL contient aussi:

```text
train_focused_front_avg
train_global_front_avg
eval_focused_front_avg
eval_global_front_avg
benchmark_focused_front_avg
benchmark_global_front_avg
promotion_blockers
guardian.strict_single_target_4p
guardian.disable_snipe_4p
guardian.max_focus_targets_4p
guardian.four_p_step
```

`focused_front_avg` mesure les fronts contre le focus. `global_front_avg` mesure
l'exposition contre tous les ennemis. Si focused est bon mais global reste haut,
le bot est encore vulnerable aux adversaires non-focus.

## 7. Commandes

Run guardian 4p local:

```powershell
.\run_v9_4p_guardian_8h.ps1
```

Equivalent Python:

```powershell
python .\run_v9.py --minutes 480 --hard-timeout-minutes 480 --workers 8 --pairs 8 --games-per-eval 3 --eval-games 12 --benchmark-games 24 --min-promotion-benchmark-games 24 --benchmark-progress-every 4 --eval-every 1 --benchmark-every 1 --max-steps 120 --eval-max-steps 220 --four-player-ratio 1.0 --eval-four-player-ratio 1.0 --benchmark-four-player-ratio 1.0 --four-p-signal-boost 1.4 --train-search-width 3 --train-simulation-depth 0 --train-simulation-rollouts 0 --train-opponent-samples 1 --front-lock-turns 22 --target-active-fronts 2.0 --target-backbone-turn-frac 0.15 --front-penalty-weight 0.055 --front-penalty-cap 0.12 --front-ok-bonus 0.070 --front-partial-bonus 0.035 --backbone-penalty-weight 0.120 --backbone-bonus-weight 0.100 --front-pressure-plan-bias 0.16 --front-pressure-attack-penalty 0.14 --guardian-enabled 1 --guardian-min-benchmark-4p 0.42 --guardian-min-benchmark-backbone 0.08 --guardian-max-benchmark-fronts 2.70 --guardian-max-generalization-gap 0.18 --export-best-on-finish 1 --min-benchmark-score 0.35 --max-generalization-gap 0.18 --exploration-rate 0.08 --reward-noise 0.008 --pool-limit 15
```

Benchmark separe 4p pur:

```powershell
python .\run_v9.py --skip-training --benchmark-games 128 --benchmark-progress-every 1 --workers 8 --eval-max-steps 220 --benchmark-four-player-ratio 1.0 --four-player-ratio 1.0 --pool-limit 15
```

Validation:

```powershell
python -m compileall -q war_orbit run_v9.py
python -m pytest test_v9_robustness.py test_v9_smoke.py -q
```

## 8. Comment juger un run

Ordre de priorite:

1. `benchmark_4p` doit monter vers `0.42+`.
2. `promotion_blockers` doit perdre `bench4p_low`.
3. `fronts` doit rester sous le seuil guardian et tendre vers `2.0`.
4. `global_front_avg` ne doit pas rester beaucoup plus haut que
   `focused_front_avg`.
5. `bb`, `xfer` et `lock` doivent rester bons.

Un run avec `bench=0.417` mais `benchmark_4p=0.278` reste faible. Le score global
est acceptable seulement si le 4p progresse vraiment.
