# Training

Stratégie recommandée :

1. démarrer sur des épisodes synthétiques ou replay-like ;
2. valider encoder + policy + storage ;
3. activer self-play court ;
4. prolonger avec checkpoints réguliers ;
5. comparer les versions avec `benchmark.py`.

Le code fourni vise une base de recherche. Il n'essaie pas de résoudre à lui seul la totalité du problème compétitif.

