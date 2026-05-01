# Self-Play

Le self-play du package choisit un adversaire parmi :

- le modèle courant ;
- un checkpoint précédent ;
- une baseline aléatoire ;
- une baseline heuristique simple.

Le but est de produire des transitions exploitables sans dépendre de l'infrastructure Kaggle pendant les tests unitaires.

