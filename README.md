# Warorbit — Orbit Wars Bot

## Fichiers

- `submission.py` — agent principal (à soumettre)
- `bot_v7.py` — source V7
- `bot_v8_2.py` — source V8.2 actuelle
- `benchmark_v8_2.py` — benchmark local V8.2 2p/4p/mixte
- `train_v8_2.py` — entraînement long du ranker V8.2
- `train_v8_offline.py` — entraînement offline
- `docs/analysis/stratégie.md` — analyse compétitive
- `CLAUDE.md` — config Playwright MCP

## Setup

```bash
pip install kaggle-environments numpy
```

## Tester localement

```bash
python3 -m py_compile bot_v8_2.py benchmark_v8_2.py train_v8_2.py
python3 benchmark_v8_2.py --games-per-opp 1 --mode mixed --opponents bot_v7 --max-steps 80
```

Lance un smoke test V8.2 local.

## Entraîner V8.2

Run 10h calibré (cf. `docs/specs/V8_ARCHITECTURE.md` pour la justification) :

```bash
python3 train_v8_2.py --minutes 600 --pairs 6 --games-per-eval 4 --eval-games 60 --eval-every 4 --max-steps 260 --eval-max-steps 500 --four-player-ratio 0.65 --eval-four-player-ratio 0.70 --workers 8 --pool-limit 8 --min-improvement 0.015 --min-mode-floor 0.05
```

Reprise automatique via `evaluations/v8_2_ranker_train_latest.npz`. Logs
exploitables en JSONL : `evaluations/v8_2_train.jsonl`.

Smoke (~1 min) :

```bash
python3 train_v8_2.py --minutes 1 --pairs 2 --games-per-eval 1 --eval-games 4 --max-steps 120 --workers 4 --pool-limit 2 --no-resume --checkpoint /tmp/v8_2_latest.npz --best-checkpoint /tmp/v8_2_best.npz --export-bot-checkpoint /tmp/v8_2_ranker.npz --log-jsonl /tmp/v8_2_train.jsonl
```

## Soumettre

1. Tester localement OK
2. Aller sur https://www.kaggle.com/competitions/orbit-wars
3. Cliquer "Submit Agent"
4. Upload `submission.py`

Score initial: μ=600. Évolue selon résultats des parties.
