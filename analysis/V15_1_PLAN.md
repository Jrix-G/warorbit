# V15.1 — plan d'amélioration

## 0. Contrainte cadre (non négociable)

Mesuré sur Kaggle : **~1s/coup** de base + une **réserve de 60s par partie**, et **pas de numba**.
Tout ce qui suit doit tenir dans ~0,7s/coup en numpy pur. La ressource rare = le
budget de rollouts (~120 rollouts/coup au total aujourd'hui). **Améliorer V15.1 =
extraire plus de décision juste du même budget**, pas en consommer plus.

## 1. Objectif

V15.1 doit **battre V15 en tête-à-tête** (gate : winrate Wilson 95% LB ≥ 60%, n=200).
« Presque à tous les coups » (≥85%) est l'aspiration ; +60% est le seuil de réussite
honnête pour une vraie amélioration de même budget.

## 2. Les trois leviers (par ordre de levier)

### Levier A — Allocation intelligente du budget (sequential halving)

**Problème actuel.** V15 donne le *même* nombre de rollouts à chaque candidat (~15
chacun sur 8 candidats). Un candidat manifestement mauvais reçoit autant de rollouts
qu'un bon → gaspillage. C'est exactement le point « éliminer celles qui ne servent à
rien dès le début ».

**Solution scientifique.** *Sequential Halving* (Karnin-Koren-Somekh 2013) — algorithme
optimal d'identification du meilleur bras à budget fixe :
1. Round 1 : K candidats, quelques rollouts chacun → éliminer la moitié la pire.
2. Round 2 : K/2 candidats survivants, plus de rollouts chacun → éliminer la moitié.
3. … jusqu'à 1 candidat. Le meilleur reçoit la majorité du budget.

À budget égal, SH identifie le meilleur coup avec une probabilité bien supérieure à
l'allocation uniforme. **C'est le changement à plus fort levier : aucun coût de
vitesse, pur algorithme, et il permet de tester BEAUCOUP plus de candidats** (les
mauvais sont éliminés après 2-3 rollouts).

→ Gate A : V15+SH bat V15 de ≥+5% (n=200).

### Levier B — Évaluation de feuille plus précise

**Problème actuel.** La valeur d'un rollout = part de vaisseaux brute (`ship_share`)
à l'horizon H. C'est bruité et biaisé : avoir des vaisseaux ≠ gagner.

**Solution.** Évaluation pondérée multi-critères au lieu du ship-share nu :
- part de vaisseaux (court terme)
- part de **production** (prédit les vaisseaux futurs — D1 : la production est le
  facteur mi-partie décisif)
- part de **planètes** (contrôle territorial)
- bonus terminal si le rollout atteint la fin de partie (signal exact win/lose)

Une éval moins bruitée → mêmes rollouts, estimation plus juste → « prédictions plus
précises ». Les poids seront tunés ensuite par CMA-ES (run VPS).

→ Gate B : V15+B bat V15 de ≥+3% (n=200).

### Levier C — Candidats plus riches et plus diversifiés

**Problème actuel.** Candidats = sous-ensembles du coup de V7 + 3 échantillons de
politique. La recherche ne peut choisir que parmi ce qu'on génère ; si le vrai
meilleur coup n'est pas généré, elle ne le trouvera jamais.

**Solution.** Générer plus de candidats variés — possible *parce que* le Levier A
(sequential halving) élimine les mauvais à bas coût :
- variations du niveau d'engagement (envoyer 50% / 70% / 90% des vaisseaux)
- cibles alternatives (3-5 plus proches non-possédées, pas seulement la plus proche)
- coups défensifs (renfort d'une planète menacée)
- le coup vide (déjà présent)

→ Gate C : V15+C bat V15 de ≥+3% (n=200).

## 3. Ordre d'implémentation et roadmap

| phase | livrable | gate | pourquoi cet ordre |
|---|---|---|---|
| V15.1-A | sequential halving à la racine | bat V15 ≥+5% | plus fort levier, débloque C |
| V15.1-B | éval de feuille pondérée | bat (A) ≥+3% | réduit le bruit que A exploite |
| V15.1-C | candidats enrichis | bat (A+B) ≥+3% | rentable seulement une fois A en place |
| V15.1-D | CMA-ES sur les poids d'éval + params SH | bat (A+B+C) ≥+3% | run VPS, tuning fin |
| V15.1 | gate final vs V15 | bat V15 ≥+60% (LB) | — |

Chaque phase est togglable par flag et benchmarkée en tête-à-tête vs V15.
Filet de sécurité conservé : repli V7 si erreur/dépassement temps.

## 4. Ce qui N'est PAS dans V15.1 (et pourquoi)

- **Recherche en arbre profonde (MCTS depth 3+)** : trop chère en numpy sous 1s.
  Le sequential halving à la racine est le 80/20. Reporté à un éventuel V16.
- **Re-ranker / éval par réseau de neurones** : possible mais le gain incertain ne
  justifie pas le risque tant que A/B/C ne sont pas exploités. Reporté.
- **numba** : confirmé absent de Kaggle. Abandonné définitivement pour la soumission.

## 5. Honnêteté sur le plafond

Sous 1s/coup numpy, on reste un search peu profond. V15.1 vise un bot nettement
plus fort que V15, mais pas un saut d'un ordre de grandeur. Estimation : V15.1
peut ajouter ~+50 à +150 ELO sur V15. Le vrai plafond (1400+) demanderait soit
numba autorisé, soit un moteur C — hors périmètre tant que Kaggle impose .py + ~1s.

## 6. Risque principal

Le budget de rollouts est si serré (~120/coup) que le bruit Monte-Carlo domine.
Si après V15.1-A le gain est < +5%, c'est que le bruit, pas l'allocation, est le
mur → la priorité bascule sur le Levier B (réduire le bruit) avant C.
