# Population 4P Training

Ce document décrit le nouveau run long prévu pour entraîner l'agent neural contre le pool de notebooks.

## Objectif

Le script principal est :

```powershell
python scripts\run_90min_6agent_training.py
```

Il lance par défaut un entraînement d'environ 90 minutes avec 6 workers en parallèle.
La durée demandée est plafonnée à 8 heures maximum, même si `--duration-minutes` est plus grand.

L'objectif n'est pas de garder le dernier modèle produit, mais de faire converger progressivement le meilleur modèle disponible :

1. charger le meilleur checkpoint existant si disponible ;
2. entraîner 6 candidats en parallèle depuis cette base ;
3. faire une évaluation rapide de tous les candidats ;
4. confirmer seulement le meilleur candidat sur une évaluation plus large ;
5. promouvoir uniquement si le score composite confirmé progresse ;
6. relancer une génération depuis ce nouveau best checkpoint.

Cette approche évite de dégrader le modèle en remplaçant le best par un candidat plus récent mais moins bon.

## Curriculum automatique

Le run populationnel ne démarre plus directement contre le pool notebook complet.
Il utilise un curriculum d'adversaires persistant :

```text
basic_300      -> random / greedy / starter
heuristic_500  -> greedy / starter / distance / sun_dodge / structured / orbit_stars
mixed_700      -> heuristiques + notebooks de départ
notebook_open  -> pool notebook complet
```

Le fichier d'état est écrit dans :

```text
logs/opponent_curriculum_state.json
```

Il contient le tier courant, le pool utilisé, l'historique des promotions de
difficulté, le meilleur score du tier et le dernier record évalué. Si un run
s'arrête après 1 heure, le run suivant reprend au même niveau de curriculum
tant que `--no-resume` n'est pas utilisé.

Le passage au tier suivant est automatique. Il demande plusieurs générations
dans le tier courant et un mélange de critères :

- score composite suffisant ;
- winrate suffisant ;
- rang moyen assez bon ;
- taux de `do_nothing` raisonnable.

## Score composite

La promotion ne dépend plus du winrate seul. Le score utilisé combine :

- `winrate` ;
- `rank_mean` converti en score de rang ;
- `eval_mean` ;
- `avg_score` normalisé ;
- `eval_avg_ships_sent` ;
- pénalité sur `eval_do_nothing_rate`.

Cela évite de promouvoir un modèle qui gagne rarement par hasard mais joue
mal sur le reste des métriques.

## Fichiers ajoutés

- `scripts/run_90min_6agent_training.py` : point d'entrée du run 90 minutes.
- `src/population_4p_training.py` : boucle populationnelle avec multiprocessing, évaluation et promotion du best checkpoint.

## Configuration par défaut

Le runner force les paramètres importants suivants :

- `duration_minutes = 90`
  - plafond d'exécution: `480` minutes
- `workers = 6`
- `hidden_dim = 256`
- `notebook_pool_limit = 15`
- `train_notebook_opponents = 3`
- `four_player_ratio = 1.0`
- `eval_four_player_ratio = 1.0`

En 4 joueurs, `train_notebook_opponents = 3` signifie que notre agent occupe un slot et les trois autres slots sont remplis par des adversaires tirés du pool notebook.

## Taille du réseau

Le réseau existant utilise `hidden_dim = 256`.

Avec l'encodeur actuel :

```text
input_dim = 11 + 64 * 19 + 128 * 10 + 4 * 8 = 2539
```

Cette taille d'entrée donne déjà environ 1,31 million de paramètres avec `hidden_dim = 256`.

Il n'est donc pas nécessaire de monter `hidden_dim` à 352 pour atteindre la cible d'environ 1,3M paramètres. Une taille plus grande augmenterait le coût CPU/GPU du run sans garantir une meilleure convergence sur 90 minutes.

## Checkpoints

Les checkpoints utilisés restent ceux du projet :

- `checkpoints/best.npz` : meilleur modèle promu après évaluation.
- `checkpoints/latest.npz` : meilleur candidat de la dernière génération.
- `checkpoints/candidate.npz` : candidat courant sauvegardé.
- `checkpoints/tiers/<tier>.npz` : meilleur modèle confirmé dans le tier
  courant. Tant que le curriculum reste dans ce tier, les générations suivantes
  repartent de ce checkpoint plutôt que de revenir au meilleur global d'un tier
  plus facile.

Au démarrage, le runner reprend en priorité depuis :

1. `checkpoints/tiers/<tier>.npz` pour le tier courant, si `resume_from_tier_best`
   est actif et que le fichier existe ;
2. `resume_checkpoint` si défini et existant ;
3. `best_checkpoint` si existant ;
4. `latest_checkpoint` si existant.

## Logs

Le nouveau log principal est :

```text
logs/population_4p_training.jsonl
```

Chaque ligne contient notamment :

- `generation`
- `worker_id`
- `parameter_count`
- `pool_size`
- `train_winrate`
- `winrate`
- `score`
- `composite_score`
- `curriculum_tier`
- `tier_generation`
- `candidate_eval_episodes`
- `promotion_eval_episodes`
- `tier_best_checkpoint`
- `winrate_by_position`
- `eval_mean`
- `rank_mean`
- `eval_do_nothing_rate`
- `eval_avg_ships_sent`
- `checkpoint_promoted`
- `promotion_reason`

Pour suivre la progression pendant le run, surveiller surtout :

- `winrate`
- `checkpoint_promoted`
- `promotion_reason`
- `winrate_by_position`

## Remarque méthode

La méthode "entraîner depuis le best, évaluer, puis promouvoir seulement si meilleur" est adaptée ici. Elle garde une pression de sélection claire et réduit le risque de régression.

Le point faible reste le bruit statistique : une évaluation sur trop peu d'épisodes peut promouvoir un candidat chanceux. Si le run devient instable, augmenter `--eval-episodes` améliore la fiabilité au prix d'un entraînement plus lent.

Exemple plus fiable mais plus coûteux :

```powershell
python scripts\run_90min_6agent_training.py --eval-episodes 32
```
