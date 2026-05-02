# Supervisor RL

## Architecture

Le système repose sur trois composants :

- `supervisor.sh` lance `scripts/run_notebook_4p_training.py`, surveille le process enfant, appelle les diagnostics et redémarre si nécessaire.
- `src/health_check.py` lit les dernières lignes du log JSONL et produit un verdict JSON `healthy` ou `unhealthy`.
- `src/autocorrect.py` lit la config et les logs, applique une correction ciblée, écrit `configs/autocorrected_config.json`, puis journalise les changements.

Le superviseur utilise `flock` sur `logs/supervisor.lock` pour éviter deux instances simultanées. Il intercepte `SIGINT` et `SIGTERM` pour tuer proprement le training enfant.

## Critères Unhealthy

`entropy_mean < 0.3` signale un collapse de politique : la distribution d'actions devient trop concentrée et l'agent n'explore plus assez.

`rank_mean > 3.7` signale une performance proche du dernier rang permanent. En FFA 4 joueurs, cela indique que l'agent est pire qu'un comportement aléatoire viable.

`do_nothing_rate > 0.80` signale une politique paralysée : l'agent évite l'exposition aux pertes en produisant trop souvent une absence d'action.

`winrate == 0.0` avec une fenêtre d'au moins 20 épisodes signale une absence totale de victoire sur une fenêtre assez grande pour être exploitable.

## Corrections

Si l'entropie est trop faible, `entropy_coef_start` est augmenté jusqu'à `0.20` et `temperature_start` passe à `1.5`. L'objectif est de rouvrir l'exploration.

Si le rang est très mauvais et que `do_nothing_rate` est élevé, l'entropie est augmentée et le learning rate est réduit. Cela évite de renforcer trop vite une politique passive.

Si aucune victoire n'apparaît sur une fenêtre de 20 épisodes, `notebook_pool_limit` passe à `2` et `entropy_coef_start` à `0.08`. L'objectif est de revenir à des adversaires plus faibles pour restaurer un signal d'apprentissage.

`train_steps` est toujours conservé à sa valeur d'origine.

## Commandes Manuelles

Lancer le superviseur depuis la racine du repo :

```bash
CHECK_INTERVAL=1800 MAX_RESTARTS=5 bash neural_network/supervisor.sh
```

Lancer uniquement le diagnostic :

```bash
python neural_network/src/health_check.py \
  --log neural_network/logs/notebook_4p_training.jsonl \
  --window 20
```

Lancer uniquement l'autocorrection :

```bash
python neural_network/src/autocorrect.py \
  --config neural_network/configs/default_config.json \
  --log neural_network/logs/notebook_4p_training.jsonl
```

## Lire supervisor.log

`CHECK_OK` indique que la fenêtre récente est saine et que le training continue.

`CHECK_UNHEALTHY` indique que le superviseur a détecté une condition de collapse ou de stagnation.

`AUTOCORRECT` contient le statut et le JSON retourné par `autocorrect.py`.

`START training` et `TRAINING_PID` permettent de tracer les redémarrages.

`FINAL_CHECK` contient le dernier diagnostic quand le training se termine naturellement.

## Limites

`MAX_RESTARTS` protège contre une boucle infinie, mais peut interrompre un run qui aurait récupéré plus tard.

Une fenêtre trop petite peut classer un run sain comme instable. Une fenêtre trop grande ralentit la détection.

Le diagnostic dépend de la qualité des champs loggés. Si `policy_entropy`, `rank`, `winner`, `our_index` ou `do_nothing_rate` sont absents, les métriques sont moins précises.

L'autocorrection ne remplace pas une bonne reward ou une bonne action policy. Elle limite les modes d'échec les plus évidents.
