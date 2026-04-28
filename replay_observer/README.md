# Replay Observer

Outil expérimental pour observer automatiquement un replay Kaggle Orbit Wars.

Objectif :
- ouvrir une URL de replay ;
- capturer les requêtes réseau qui pourraient contenir des données JSON ;
- prendre des screenshots réguliers de la page et des canvas ;
- sauvegarder un rapport exploitable dans `replay_observer/output/`.

Ce dossier ne garantit pas encore que Kaggle expose les vraies données du replay.
S'il ne les expose pas, les screenshots serviront ensuite à construire un mode vision/OCR.

## Installation

```bash
python3 -m pip install -r replay_observer/requirements.txt
python3 -m playwright install chromium
```

## Test Rapide

```bash
python3 replay_observer/replay_observer.py --help
```

## Observer Un Replay

```bash
python3 replay_observer/replay_observer.py \
  --url "https://www.kaggle.com/competitions/orbit-wars/leaderboard?submissionId=51987365&episodeId=75514378" \
  --seconds 30 \
  --interval 1.0 \
  --headed
```

Sorties :
- `replay_observer/output/<run_id>/report.json`
- `replay_observer/output/<run_id>/network/*.json`
- `replay_observer/output/<run_id>/screenshots/*.png`
- `replay_observer/output/<run_id>/canvas/*.png`

## Si Kaggle Demande Login

Lance avec `--headed`, connecte-toi manuellement dans la fenêtre Chromium, puis laisse le script continuer.

Pour réutiliser une session Playwright :

```bash
python3 replay_observer/replay_observer.py --url "..." --headed --save-storage replay_observer/output/kaggle_auth.json
python3 replay_observer/replay_observer.py --url "..." --storage replay_observer/output/kaggle_auth.json
```

## Interprétation

Dans `report.json`, regarde :
- `json_responses`: réponses réseau parsées en JSON ;
- `network_hits`: URLs suspectes vues pendant le chargement ;
- `page_probe`: objets JS/storage/performance trouvés dans la page ;
- `screenshots`: captures utilisables pour vision/OCR.

