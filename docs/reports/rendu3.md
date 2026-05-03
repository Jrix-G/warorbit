# Rendu 3 — Scraping Parties + Discussions Orbit Wars

Date: 2026-05-03  
Périmètre: collecte récente des parties + extraction des signaux des discussions publiques.

## 1) Objectif

Produire un état de situation exploitable sur:
- les nouvelles parties (cible ~200),
- les discussions actuelles (tactiques, glitches, changements de règles),
- les impacts directs sur la stratégie bot.

## 2) Méthode de collecte

### 2.1 Parties

- Script principal utilisé: `harvest_replays.py` (Playwright + endpoints Kaggle).
- Fichier final exploité pour l’analyse:
  - `replay_dataset/compact/salvage/episodes_2026-05-03_200_salvage.jsonl`
- Volume final:
  - 200 épisodes
  - 60 parties 2p
  - 140 parties 4p

### 2.2 Discussions

- Collecte via navigation Playwright sur l’onglet Discussion de la compétition.
- Discussions prioritaires inspectées:
  - 696043 (tunneling bug)
  - 695557 (décalage step production/mouvement)
  - 695715 (heuristique vs RL)
  - 696547 (types de bots top 10)
  - 694127 (visualizer local)
  - 696032 (matchmaking)
  - 694910 (symmetry update 4p)
  - 694210, 695256 (logs/replays via CLI/UI)

## 3) Qualité et limites de la collecte

Points observés:
- Le endpoint `GetEpisodeReplay` a subi du rate-limit (`429 Too many requests`) pendant les relances.
- Une relance “propre” a parfois retourné 0 replay malgré les listes d’épisodes disponibles.
- Le dataset final de 200 épisodes a été récupéré par “salvage” d’un run partiel valide.

Conclusion:
- Le volume cible est atteint et exploitable.
- La stabilité de scraping API Kaggle doit être renforcée (backoff/retry + stratégie anti-429).

## 4) Résultats quantitatifs — Parties (200 épisodes)

## Distribution globale

- 2p: 60 (30%)
- 4p: 140 (70%)

## Dynamique des parties

- Longueur moyenne: 255.65 steps
- Médiane: 217 steps
- P90: 500 steps

Lecture:
- Forte présence de parties longues, surtout en 4p.
- Le late-game reste très structurant dans le méta actuel.

## Profil d’action

- Actions moyennes par tour (agrégées): 4.69
- Taille moyenne des flottes envoyées: 15.9
- P90 taille de flotte: 30
- Max observé: 8937

Lecture:
- Base méta orientée micro/lancers modérés.
- Quelques extrêmes de volume de flotte très atypiques.

## Pression early-game

- Heuristique: proportion de parties avec au moins un envoi `>=40` avant le tour 20:
  - 9.5%

Lecture:
- Le “gros rush early” existe mais reste minoritaire.
- Le contrôle tempo/production domine plus souvent que l’all-in précoce.

## Épisodes anormaux (candidats glitch ou snowball extrême)

Exemples forts détectés:
- 75636654 (2p, 500 steps, flotte max 8937)
- 75635797 (4p, 271 steps, flotte max 1915)
- 75592408 (4p, 262 steps, flotte max 1782)
- 75653725 (4p, 500 steps, flotte max 1288)

Interprétation:
- Mélange possible de snowball structurel et d’artefacts moteur.
- Ces épisodes sont prioritaires pour revue replay ciblée.

## Soumissions les plus fréquentes dans l’échantillon

- 52066322: 99 apparitions
- 52128366: 40
- 52166772: 40
- 52165810: 19
- 51799813: 12

Impact:
- Les signaux tactiques extraits représentent surtout ce sous-ensemble de méta actif.

## 5) Résultats qualitatifs — Discussions

## Signal critique: bug de collision “tunneling”

Discussion: 696043  
Constat remonté:
- des flottes et planètes peuvent “se traverser” en un tick sans résolution de combat,
- lié à une granularité/schéma de collision par snapshots (objets mobiles traités partiellement comme statiques selon phase).

Impact stratégique:
- certains trajets proches de planètes peuvent devenir non robustes.
- les estimations de capture/combat peuvent diverger du comportement théorique attendu.

Priorité: Haute.

## Signal moteur: décalage step production vs mouvement orbital

Discussion: 695557  
Constat:
- production perçue à step 1,
- mouvement orbital visible à step 2 (effet lié à un fix d’observation/replay mentionné en discussion).

Impact stratégique:
- risque d’erreur de forecasting si le simulateur interne n’aligne pas exactement ces conventions.

Priorité: Haute (validation simulateur local vs moteur officiel).

## Changement officiel d’environnement 4p

Discussion: 694910  
Annonce:
- adoption de la symétrie rotation 90° pour des maps 4p plus équitables.

Impact stratégique:
- réduction d’asymétries structurelles,
- plus de valeur sur politiques robustes multi-spawn.

Priorité: Haute pour tuning 4p.

## Signal méta: heuristique toujours compétitive

Discussions: 695715, 696547  
Constat communautaire:
- des bots heuristiques restent compétitifs en silver, et possiblement au-delà.

Impact stratégique:
- une trajectoire non-RL reste viable si:
  - bon contrôle mid/late game,
  - robustesse aux cas limites moteur,
  - adaptation rapide aux changements de règles.

## Outillage opérationnel récent (debug/itération)

Discussions: 694210, 695256  
Signal:
- disponibilité de `kaggle 2.0.2` et options UI pour télécharger replays/logs plus facilement.

Impact:
- cycle d’itération plus court pour diagnostiquer défaites réelles.

## 6) Synthèse tactique exploitable

1. Le méta observé favorise les parties longues: investir dans des politiques de conversion/production stables paie plus qu’un early all-in systématique.
2. Les extrêmes de flotte doivent être traités explicitement (cap logique, règles anti-overcommit, réponses défensives graduées).
3. Les bugs/artefacts moteur (collision, timing step) doivent être pris en compte dans l’évaluation de trajectoires.
4. Le 4p devient plus “symétrique”: augmenter le poids des stratégies spawn-invariantes.

## 7) Risques techniques identifiés

1. Rate-limit Kaggle API replay (`429`) sans backoff robuste.
2. Couverture de scraping dépendante de sessions/cookies navigateur.
3. Biais d’échantillon (surreprésentation de quelques soumissions actives).

## 8) Plan d’actions recommandé

## Court terme (immédiat)

1. Implémenter retry exponentiel + jitter sur `GetEpisodeReplay`.
2. Ajouter persistance d’état (checkpoint) par épisode pour reprise propre.
3. Créer une “watchlist glitch” avec les épisodes anormaux détectés.

## Moyen terme

1. Construire un classifieur interne “anomalie replay” (flotte extrême, collision suspecte, mismatch capture attendu).
2. Ajouter un test de non-régression sur timings step (prod/mouvement).
3. Renforcer l’évaluation 4p avec métriques dédiées post-symmetry update.

## 9) Livrables de cette passe

- Dataset analysé:
  - `replay_dataset/compact/salvage/episodes_2026-05-03_200_salvage.jsonl`
- Script scraping ajusté:
  - `harvest_replays.py` (arrêt en cours de boucle + option d’expansion adversaires)
- Rapport:
  - `docs/reports/rendu3.md`
  - `docs/reports/V8_R3_PATCHES.md`

## 10) Conclusion

La collecte cible (200 parties) est atteinte et fournit des signaux utiles:
- méta orienté endurance/late-game,
- présence d’anomalies fortes à investiguer,
- discussions confirmant des points moteur critiques (collision/timing),
- environnement 4p modifié (symétrie 90°) à intégrer dans la stratégie.

Priorité opérationnelle: fiabiliser le pipeline de scraping (anti-429) et transformer immédiatement les signaux glitch/timing en règles de robustesse bot + tests de non-régression.

Corrections appliquées après audit: voir `docs/reports/V8_R3_PATCHES.md`.
