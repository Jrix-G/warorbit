# Architecture

Le package repose sur 5 blocs :

1. `encoder.py` transforme un état brut en tensors NumPy.
2. `model.py` applique un MLP partagé et produit policy + value.
3. `policy.py` transforme les sorties en action jouable.
4. `reward.py` calcule une récompense dense + terminale.
5. `trainer.py` orchestre self-play, optimisation et checkpoints.

Le choix est volontairement simple :

- compatible CPU ;
- facile à tester ;
- extensible vers un encodeur plus riche ;
- indépendant du bot rule-based existant.

