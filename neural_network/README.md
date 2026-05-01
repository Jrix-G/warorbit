# Neural Network

Package autonome pour expérimenter un agent neural sur Orbit Wars.

## Objectif

- encoder l'état du jeu ;
- scorer des actions candidates ;
- entraîner un modèle léger sur CPU ;
- lancer du self-play ;
- benchmarker des checkpoints ;
- sauvegarder et reprendre un run.

## Structure

- `configs/` : configuration JSON
- `src/` : code principal
- `scripts/` : points d'entrée
- `docs/` : documentation technique
- `checkpoints/` : modèles sauvegardés
- `logs/` : journaux d'entraînement
- `tests/` : tests unitaires

## Dépendances

- `numpy`
- `pytest` pour les tests

Optionnel :
- `kaggle-environments` si tu veux brancher un environnement externe plus tard

## Commandes

```bash
python3 neural_network/scripts/train.py --config neural_network/configs/default_config.json
python3 neural_network/scripts/self_play_run.py --config neural_network/configs/default_config.json
python3 neural_network/scripts/benchmark_model.py --config neural_network/configs/default_config.json
python3 neural_network/scripts/export_model.py --config neural_network/configs/default_config.json
pytest neural_network/tests
```

## Remarques

- Le package est conçu pour fonctionner sans modifier `bot_v7.py` ou `bot_v8_2.py`.
- L'agent par défaut est une base de recherche et d'itération, pas encore un bot Kaggle final.

