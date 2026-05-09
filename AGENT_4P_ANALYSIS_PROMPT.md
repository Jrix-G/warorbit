# Agent Prompt — Analyse stratégique V14 (4p-first)

## Mandat

Tu es chargé d'une analyse stratégique complète du bot V14.
Objectif non négociable : **construire d'abord un bot très fort en 4p, ajuster 2p ensuite**.

Si une amélioration améliore uniquement le 2p au détriment du 4p → c'est un échec.
Si une amélioration améliore le 4p mais dégrade légèrement le 2p → acceptable à court terme.

---

## Contexte du run échoué

Run overnight `scorer_v14_6h` (38 batches, 1216 games) :

```
wr global   : 387/1216 = 31.8%
wr2 (2p)    : ~57.9%   (stable, pics à 0.917)
wr4 (4p)    : ~18.6%   (plat, jamais au-dessus de 0.35)
reward moyen: ~-0.055
```

Constat : le fine-tune a stabilisé le 2p, n'a pas du tout progressé en 4p.
Le meilleur checkpoint sauvegardé (`best_train_wr`) est probablement un pic de bruit 2p, pas une vraie percée 4p.

---

## Fichiers sources à lire

```
bot_v14.py
v14_core.py
bot_v12.py
bot_v13.py
extract_v14_bc_dataset.py
train_v14_bc.py
train_v14_finetune.py
benchmark_v14.py
gate_v14.py
V14_RUNBOOK.md
docs/reports/V14_ANALYSIS.md
docs/analysis/leaderboard_top1_deep_report.txt
docs/analysis/reverse_engineering_top1.txt
replays/top1-05-05/          (replays top1 en 4p)
imported_logs/run_10h_2p4p_20260507_123522_*
replay_dataset/
evaluations/
```

---

## Questions obligatoires

1. Comment V14 fonctionne réellement aujourd'hui (pipeline complet, pas de résumé superficiel) ?
2. Où est la frontière exacte entre heuristique tactique et politique 4p ?
3. Pourquoi le 4p ne progresse pas malgré l'entraînement ?
4. Qu'est-ce que le top1 fait en 4p que notre pipeline ne sait pas représenter ?
5. Quelles features sont indispensables pour décider correctement en 4p ?
6. Quels candidats tactiques manquent dans l'espace d'action ?
7. Quelle part du problème vient de : générateur de candidats / features / labels BC / fine-tune / fallback ?
8. Quelle stratégie d'apprentissage adopter pour gagner d'abord en 4p ?
9. Comment réordonner la feuille de route pour que le 2p devienne un raffinement, pas le centre ?

---

## Axes d'investigation

### 1. Architecture actuelle

- Pipeline de décision précis : `_load_scorer → get_candidates → score → select → fallback V12`.
- Séparer : tactique / politique / appris / heuristique.
- Identifier les dépendances cachées au 2p dans les features (FEATURE_DIM=64, `v14_core.py`).
- Vérifier si le fallback V12 masque des défauts structurels (bot_v14.py:39, :55).

### 2. Replays top1

- Lire les replays top1 comme source de **politique 4p**, pas comme dataset BC.
- Identifier motifs récurrents en 4p :
  - choix de cible selon état de la carte
  - temporisation / anti-focus
  - opportunisme quand deux adversaires se battent
  - élimination d'un joueur faible en priorité
  - redistribution front/arrière selon contexte multi-joueur
- Expliquer ce que le pipeline actuel ne peut pas capturer.
- Distinguer décisions locales (tactiques) et décisions de politique (qui attaquer, quand, pourquoi).

### 3. Espace de candidats

- Les 5 familles actuelles : attack, expand, defense, staging, noop — couvrent-elles vraiment le 4p ?
- Lister les classes absentes (ex : attaque opportuniste d'un tiers, cession de territoire, focus coordonné).
- Est-ce que le problème est l'ordonnancement ou l'absence pure des candidats ?
- Est-ce qu'un MLP sur les candidats actuels **peut en théorie** résoudre le 4p ? Si non, pourquoi.

### 4. Features

- Identifier les features trop locales (ne voient qu'une paire source-cible).
- Identifier les features qui encodent mal les adversaires individuellement.
- Expliquer pourquoi un bit `is_4p` ne suffit pas.
- Proposer une représentation du contexte politique 4p :
  - force relative par adversaire (pas juste total)
  - menace relative par adversaire
  - progression d'élimination (qui est presque mort ?)
  - opportunité de focus (est-ce que deux ennemis se battent ?)
  - capacité de contestation (qui peut me contrer si j'attaque X ?)
  - état de guerre entre adversaires (indirect, mais crucial)
  - positionnement front/back de chaque joueur
- Dire quelles features sont **bloquantes** (sans elles, le scorer ne peut pas apprendre une vraie politique 4p).

### 5. Entraînement

- BC apprend-il une imitation de surface ou une vraie politique ?
- Le fine-tune pousse-t-il vers la stabilité ou vers le bruit ? (Regarder la variance inter-batch dans le run.)
- Le signal de reward est-il aligné avec la victoire en 4p ?
  - `reward = 1.0 + 0.5 * margin` si victoire, pénalité sinon (`train_v14_finetune.py:39`).
  - Est-ce que ce reward discrimine assez l'élimination précoce vs survie tardive en 4p ?
- `best_train_wr` est sauvegardé sur winrate batch (32 games) — c'est du bruit en 4p. Quelle alternative ?
- L'anchor BC empêche-t-il la politique 4p d'émerger ?

### 6. Run récent

- Relire le log `run_10h_2p4p_20260507_123522`.
- Identifier ce qui a été atteint en 2p / ce qui a échoué en 4p batch par batch.
- b0032 : wr2=0.917, wr4=0.250 → spike 2p ou vraie amélioration ?
- Conclure clairement : le run valide-t-il ou invalide-t-il la stratégie actuelle ?

---

## Livrables attendus

1. **Diagnostic exécutif** (5-10 lignes max) — le problème central, sans détour.
2. **Analyse structurée** par section (architecture / replays / candidats / features / entraînement / run).
3. **Cartographie des manques** :
   - manques de features (lesquelles, pourquoi bloquantes)
   - manques de candidats (quelles classes, pourquoi absentes)
   - manques de labels (le BC imite quoi exactement ?)
   - manques de curriculum (ordre 2p→4p vs 4p-only)
   - manques d'objectif (le reward est-il bien défini pour le 4p ?)
4. **Stratégie 4p-first** concrète :
   - comment représenter le contexte
   - comment générer les candidats manquants
   - comment scorer les candidats avec les bonnes features
   - comment entraîner (curriculum, reward shape, anchor)
   - comment valider les progrès (métriques séparées 2p / 4p, pas mélangées)
5. **Plan d'action par étapes** :
   - court terme (cette semaine) : changements immédiats pour débloquer le 4p
   - moyen terme : refonte des features et du générateur
   - long terme : curriculum complet et validation robuste
6. **Critères de succès mesurables** :
   - wr4 cible (ex : >35% contre pool top opponents)
   - wr2 minimum acceptable (ne pas régresser sous X%)
   - stabilité inter-batch (variance max acceptable)
   - absence de régression due au fallback V12

---

## Contraintes

- Chaque conclusion doit être reliée à du code ou des replays observables. Pas d'hypothèse non vérifiée.
- Ne propose pas de solution qui améliore seulement le 2p.
- Si l'architecture actuelle est **fondamentalement mal orientée** pour le 4p, dis-le explicitement et propose la réorientation.
- Pas de généralités ("il faut plus de données"). Des causes précises et des corrections concrètes.

---

## Format de sortie

1. Diagnostic exécutif (5-10 lignes)
2. Analyse par section
3. Feuille de route priorisée (4p-first)
4. Décisions concrètes à prendre dès maintenant
