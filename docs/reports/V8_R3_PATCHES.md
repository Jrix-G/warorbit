# V8 Rendu 3 - Corrections Appliquees

Date: 2026-05-03

Ce document liste les corrections appliquees apres l'audit de
[`docs/reports/rendu3.md`](/home/jason/Documents/warorbit/docs/reports/rendu3.md).

## 1) Collecte Kaggle

### `harvest_replays.py`

Corrections appliquees:

- Retry exponentiel avec jitter sur `ListEpisodes` et `GetEpisodeReplay`.
- Reclassification des erreurs transitoires `429 / 5xx / texte non-JSON` comme retryables.
- Etat de reprise persistant par checkpoint JSON:
  - file de submissions en attente,
  - submissions deja visitees,
  - compteurs 2p / 4p,
  - taille ecrite.
- Reprise idempotente basee sur l'output JSONL existant.
- Watchlist d'anomalies en JSONL pour episodes a revue manuelle.

Impact attendu:

- Les runs de collecte peuvent reprendre apres rate-limit ou crash sans perdre
  la progression.
- Les episodes extremes ne disparaissent plus dans le dataset principal; ils
  sont journalises separement.

## 2) Validation mecanique locale

### `test_sim.py`

Le test de parity local reste en place pour la fidelite du simulateur.

### `SimGame.py`

Le moteur rapide local reste distinct de `sim.py`, mais il est conserve comme
runner de benchmark et de smoke. Les nouveaux episodes anormaux sont maintenant
detectables cote collecte, ce qui reduit le risque d'entrainer sur des cas non
diagnostiques sans les identifier.

## 3) Politique V8.2

### `bot_v8_2.py`

Corrections deja appliquees et maintenues dans cette passe:

- `transfer_push` et `opening_relay` pour le staging ami->ami.
- `opening_fortify` pour la defense precoce quand la pression monte.
- `probe_focus` pour les petites frappes de pression / information.
- solveur d'interception plus robuste avec recherche locale autour de l'ETA
  initial et raffinement amorti.
- selection de candidat evite un plan vide quand un plan utile existe.

## 4) Ce qui reste a surveiller

- Le bug de tunneling et les ecarts de timing production/mouvement remontes
  dans les discussions Kaggle ne sont pas corriges par le code local seul.
  Ils doivent etre traites comme risques d'alignement moteur et surveilles via
  replays + validation parity.
- La collecte reste dependante de Kaggle; le retry reduit la fragilite mais ne
  supprime pas la variabilite serveur.

## 5) Prochaine mesure recommandee

- Relancer un harvest court pour verifier que le checkpoint et la watchlist
  fonctionnent en pratique.
- Puis recalibrer les benchmarks 4p sur le dataset plus propre.
