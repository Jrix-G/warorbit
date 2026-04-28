# Warorbit — Orbit Wars Bot

## Fichiers

- `bot.py` — agent principal (à soumettre)
- `train.py` — tests locaux contre baselines
- `docs/analysis/stratégie.md` — analyse compétitive
- `CLAUDE.md` — config Playwright MCP

## Setup

```bash
pip install kaggle-environments numpy
```

## Tester localement

```bash
python train.py
```

Joue 20 parties contre chaque baseline (passive, random, greedy_nearest, self).
Cible: >70% win-rate vs greedy_nearest avant soumission.

## Soumettre

1. Tester localement OK
2. Aller sur https://www.kaggle.com/competitions/orbit-wars
3. Cliquer "Submit Agent"
4. Upload `bot.py`

Score initial: μ=600. Évolue selon résultats des parties.
