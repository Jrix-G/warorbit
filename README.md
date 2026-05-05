# Warorbit — Orbit Wars Bot

## Fichiers

- `submission.py` — agent principal (à soumettre)
- `bot_v7.py` — source V7
- `bot_v8_5.py` — source V8.5 actuelle
- `bot_v9.py` — wrapper agent V9 experimental
- `benchmark_v8_5.py` — benchmark local V8.5 2p/4p/mixte
- `train_v8_5.py` — entraînement long du ranker V8.5
- `run_v9.py` — entraînement + benchmark V9 avec timeout dur
- `train_v8_offline.py` — entraînement offline
- `VPS/` — lanceurs conservateurs pour VPS faibles
- `docs/analysis/stratégie.md` — analyse compétitive
- `docs/specs/V9_ARCHITECTURE.md` — architecture, tactiques 4p, logs et runbook V9
- `CLAUDE.md` — config Playwright MCP

## Setup

```bash
pip install kaggle-environments numpy
```

## Tester localement

```bash
python3 -m py_compile bot_v8_5.py benchmark_v8_5.py train_v8_5.py
python3 benchmark_v8_5.py --games-per-opp 1 --mode mixed --opponents bot_v7 --max-steps 80
```

Lance un smoke test V8.5 local.

## Entraîner V8.5

Run 10h calibré (cf. `docs/specs/V8_ARCHITECTURE.md` pour la justification) :

```bash
python3 train_v8_5.py --minutes 600 --pairs 6 --games-per-eval 4 --eval-games 60 --eval-every 4 --max-steps 260 --eval-max-steps 500 --four-player-ratio 0.65 --eval-four-player-ratio 0.70 --workers 8 --pool-limit 8 --min-improvement 0.015 --min-mode-floor 0.05
```

Reprise automatique via `evaluations/v8_5_ranker_train_latest.npz`. Logs
exploitables en JSONL : `evaluations/v8_5_train.jsonl`.

Smoke (~1 min) :

```bash
python3 train_v8_5.py --minutes 1 --pairs 2 --games-per-eval 1 --eval-games 4 --max-steps 120 --workers 4 --pool-limit 2 --no-resume --checkpoint /tmp/v8_5_latest.npz --best-checkpoint /tmp/v8_5_best.npz --export-bot-checkpoint /tmp/v8_5_ranker.npz --log-jsonl /tmp/v8_5_train.jsonl
```

## Entrainer V9

Documentation complete: `docs/specs/V9_ARCHITECTURE.md`.

Etat actuel:

- V9 utilise `official_fast` par defaut.
- `official_fast` execute le vrai moteur Kaggle local `orbit_wars.py` via
  `local_simulator.official_fast`, avec le wrapper Kaggle lourd evite.
- `kaggle_fast` est un alias accepte.
- `kaggle` reste disponible pour verifier contre le wrapper officiel complet.
- L'ancien moteur rapide non officiel n'est plus disponible dans `run_v9.py`.

Smoke 10 min 4p guardian:

```bash
python3 run_v9.py --game-engine official_fast --minutes 10 --hard-timeout-minutes 10 --workers 4 --pairs 4 --games-per-eval 3 --eval-games 8 --benchmark-games 12 --min-promotion-benchmark-games 12 --benchmark-progress-every 2 --eval-every 1 --benchmark-every 1 --max-steps 120 --eval-max-steps 220 --four-player-ratio 1.0 --eval-four-player-ratio 1.0 --benchmark-four-player-ratio 1.0 --four-p-signal-boost 1.4 --train-search-width 3 --train-simulation-depth 0 --train-simulation-rollouts 0 --train-opponent-samples 1 --guardian-enabled 1 --guardian-min-benchmark-4p 0.42 --guardian-min-benchmark-backbone 0.08 --guardian-max-benchmark-fronts 2.70 --guardian-max-generalization-gap 0.18 --min-benchmark-score 0.35 --max-generalization-gap 0.18 --reward-noise 0.008 --pool-limit 15 --checkpoint evaluations/v9_10min_latest.npz --best-checkpoint evaluations/v9_10min_best.npz --export-checkpoint evaluations/v9_10min_policy.npz --log-jsonl evaluations/v9_10min_train.jsonl --no-resume
```

Run local 1h avec benchmark court et sauvegarde garantie au timeout :

```bash
python3 run_v9.py --game-engine official_fast --minutes 60 --hard-timeout-minutes 60 --pairs 5 --games-per-eval 2 --eval-games 32 --benchmark-games 16 --min-promotion-benchmark-games 16 --benchmark-progress-every 1 --max-steps 160 --eval-max-steps 220 --four-player-ratio 0.80 --pool-limit 15
```

Les logs V9 incluent maintenant le diagnostic 4p `xfer/bb/lock/fronts` et sont
écrits dans `evaluations/v9_robust_train.jsonl`.
V9 utilise le moteur officiel local optimise `official_fast` par defaut.

Pour une mesure benchmark plus stable, lancer ensuite :

```bash
python3 run_v9.py --game-engine official_fast --skip-training --benchmark-games 128 --benchmark-progress-every 1 --workers 8 --eval-max-steps 220 --four-player-ratio 0.80 --pool-limit 15
```

Mode volume pur 1h, sans eval/benchmark dans la boucle :

```bash
python3 run_v9.py --game-engine official_fast --minutes 60 --hard-timeout-minutes 60 --train-only --workers 8 --pairs 24 --games-per-eval 8 --max-steps 80 --four-player-ratio 0.80 --train-search-width 3 --train-simulation-depth 0 --train-simulation-rollouts 0 --front-lock-turns 15 --train-opponents random noisy_greedy greedy starter distance sun_dodge structured --no-resume --checkpoint evaluations/v9_volume_latest.npz --best-checkpoint evaluations/v9_volume_best.npz --export-checkpoint evaluations/v9_volume_policy.npz --log-jsonl evaluations/v9_volume_train.jsonl
```

Benchmark du checkpoint volume :

```bash
python3 run_v9.py --game-engine official_fast --skip-training --export-checkpoint evaluations/v9_volume_policy.npz --benchmark-games 128 --benchmark-progress-every 1 --workers 8 --eval-max-steps 220 --four-player-ratio 0.80 --pool-limit 15
```

## Soumettre

1. Tester localement OK
2. Aller sur https://www.kaggle.com/competitions/orbit-wars
3. Cliquer "Submit Agent"
4. Upload `submission.py`

Score initial: μ=600. Évolue selon résultats des parties.
