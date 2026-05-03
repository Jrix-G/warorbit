# V8.5 Patchset - Macro Routing and Policy Bias

Date: 2026-05-03

Ce document resume la passe 8.5 appliquee au bot V8.2 apres audit des
symptomes restants:

- le bot laissait encore trop souvent la baseline absorber les situations
  qui demandaient une vraie decision macro,
- les plans de staging, fortification et conversion existaient, mais leur
  priorisation restait trop dependante du score lineaire,
- le 4p restait trop sensible au bruit de sélection dans les phases de
  pression multi-fronts.

## 1) Corrections appliquees dans `bot_v8_2.py`

### Prior macro sur le score des candidats

Ajout d'une couche de prior heuristique au-dessus du ranker lineaire:

- `opening_fortify` est favorise quand l'ouverture est sous pression.
- `opening_relay` et `transfer_push` sont favorises quand le staging ami->ami
  apporte une vraie consolidation.
- `probe_focus` est favorise quand le 4p a besoin de pression / information.
- `4p_conservation` prend le dessus quand plusieurs fronts sont actifs ou que
  le garrison ratio devient trop bas.
- `4p_conversion_push`, `4p_eliminate_weakest` et `4p_late_blitz` sont
  soutenus quand le lead permet de convertir.
- `4p_opportunistic` n'est favorise que s'il existe une vraie cible exploitable.

### Lisibilite / maintenance

- Ajout de constantes nommees pour les indices des features d'etat et de
  candidat.
- Le score lineaire reste intact, mais les decisions sont maintenant plus
  robustes aux cas de phase.

## 2) Tests ajoutes

### `test_bot_v8_5_policy.py`

Trois regressions simples verifient que:

- la fortification d'ouverture domine la baseline en situation de pression,
- le staging ami->ami est bien recompense quand il transporte du reel
  redéploiement,
- la conservation 4p l'emporte sur une baseline passive sous multi-front.

## 3) Ce que cette passe ne change pas

- Elle ne remplace pas la grammaire de plans existante.
- Elle ne change pas les dimensions de checkpoint du ranker.
- Elle ne corrige pas le moteur Kaggle lui-meme; les ecarts de timing/collision
  restent a surveiller via replays et parity tests.

## 4) Validation locale

- `py_compile` sur `bot_v8_2.py` et le nouveau test: OK
- benchmark court local:
  - en 2p, la baseline tombe sous 10% des plans choisis; `expand_focus` et
    `opening_relay` dominent les ouvertures.
  - en 4p, la selection se deplace vers `expand_focus`, `probe_focus` et
    `opening_fortify`, avec une part baseline tres faible.
  - le gain en WR n'est pas encore visible sur un echantillon de 3 parties par
    mode, mais le routage macro est bien different du comportement pre-patch.

## 5) Impact attendu

- Moins de choix "moyens" en 4p quand une phase nette impose un routage.
- Plus de chances de convertir un lead en avantage durable.
- Moins de sur-selection du plan de base lorsque le jeu demande une decision
  macro explicite.
