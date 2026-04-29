# Warorbit — Orbit Wars Bot

## Fichiers

- `submission.py` — agent principal (à soumettre)
- `bot_v7.py` — source V7
- `bot_v8.py` — source V8
- `train_v8.py` — entraînement local principal
- `train_v8_offline.py` — entraînement offline
- `docs/analysis/stratégie.md` — analyse compétitive
- `CLAUDE.md` — config Playwright MCP

## Setup

```bash
pip install kaggle-environments numpy
```

## Tester localement

```bash
python train_v8.py
```

Lance les tests et benchmarks V8 locaux.

## Soumettre

1. Tester localement OK
2. Aller sur https://www.kaggle.com/competitions/orbit-wars
3. Cliquer "Submit Agent"
4. Upload `submission.py`

Score initial: μ=600. Évolue selon résultats des parties.
