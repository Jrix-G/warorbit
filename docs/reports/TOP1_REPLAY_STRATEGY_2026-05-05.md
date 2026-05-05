# Analyse strategique top1 - replays du 05/05

Source principale: `replays/top1-05-05`.

## Donnees

- 27 episodes dedupliques.
- Joueur focal detecte: `bowwowforeach`, present dans 27/27 episodes.
- Resultat observe: 19 victoires / 27, dont 16/20 en 2p et 3/7 en 4p.
- Volume exploite: 3291 moves inferees, 1662 captures observees.

Limite methodologique: les actions Kaggle sont `[source_id, angle, ships]`, sans cible explicite. Les cibles sont inferees par geometrie depuis la planete source a l'etat de decision precedent. Les statistiques de distance/target sont donc des estimations, mais les tailles d'envoi, timings, ownership et courbes economie sont directes.

## Observations quantitatives

### Ouverture

- Premiere action: mediane tour 3, moyenne 3.89, p75 tour 5.
- Premiere capture: mediane tour 9, moyenne 9.74.
- Moves tour 0-30: mediane 20 ships, moyenne 22.0, p75 27, p90 35.
- Moves tour 0-50: mediane 23 ships, moyenne 32.4, p75 37, p90 68.
- En opening, `send_frac` median = 1.0: le top1 vide presque toujours la planete source disponible.
- 0-30: 186/215 moves visent des neutres; 84 moves entre 15-24 ships, 60 entre 25-49, seulement 11 sous 8.
- Distance estimee 0-30: mediane 19.7, p75 37.1, p90 52.8.

### Economie

En 2p gagne:
- t30: mediane 5.5 planetes, prod 15.5.
- t50: mediane 9.5 planetes, prod 29.
- t100: mediane 19 planetes, prod 54.5.

En 2p perd:
- t30: mediane 5 planetes, prod 13.
- t50: mediane 8 planetes, prod 19.5.
- t100: mediane 5.5 planetes, prod 17.

En 4p gagne:
- t30: mediane 5 planetes, prod 13.
- t50: mediane 3 planetes, prod 11.
- t100: mediane 8 planetes, prod 20.

En 4p perd:
- t30: mediane 5.5 planetes, prod 17.
- t50: mediane 5 planetes, prod 17.5.
- t100: mediane 4 planetes, prod 12.

Conclusion: en 4p, le top1 ne gagne pas forcement en ayant plus de planetes tres tot. Il gagne quand il survit au midgame, conserve une masse mobile, puis convertit fort apres t80/t100.

### Midgame et late

- Tour 50-120: mediane 42.5 ships par move, p75 84, p90 158.
- Tour 120+: mediane 68 ships, p75 137, p90 244.
- Captures 50-100: mediane 44 ships restants sur la planete capturee.
- Captures 100-200: mediane 75.5 ships restants.
- Le style n'est pas "petits probes partout"; c'est "vider des sources, convertir vite, puis poser des garnisons massives".

## Comparaison V9

V9 actuel:
- `min_source_ships = 8`.
- `_kovi_opening_conversion` peut descendre a `min_send=4`.
- `_wide_expansion` utilise `min_send=4`.
- `_opportunistic_snipe` utilise `min_send=6`.
- `_probe` envoie souvent 3-12 ships.
- `_shot_option` envoie `needed * aggression`, donc "juste assez" pour capturer.
- Le feature `overcommit_risk` penalise au-dessus de 58% du total ships, alors que le top1 en opening vide localement les sources et maintient 40-50% de ships en flotte.

Ce qui diverge:
- V9 optimise l'efficience marginale; top1 optimise la conversion tempo + robustesse.
- V9 cree trop de planetes fraichement capturees avec faible garnison; top1 accepte des captures faibles au tout debut, mais transitionne vite vers des captures avec 40+ ships restants.
- V9 a de bons elements 2p, mais en 4p il multiplie les fronts trop tot et n'envoie pas assez de masse par capture pour tenir.

## Recommandations de patch

Priorite 1: ajouter un plancher d'envoi opening/midgame sur les vraies captures.

- Ajouter config:
  - `opening_punch_turns = 55`
  - `opening_min_capture_send_2p = 14`
  - `opening_min_capture_send_4p = 16`
  - `midgame_min_capture_send_4p = 24`
  - `capture_garrison_margin = 0.22`
- Appliquer seulement aux familles `balanced`, `aggressive_expansion`, `kovi_opening_conversion`, `wide_expansion`, `resource_denial`, `delayed_strike`.
- Ne pas appliquer aux familles `probe`, `opportunistic_snipe`, `endgame_finisher`, `all_in_finisher`.

Priorite 2: limiter les cibles lointaines en 4p opening.

- En 4p et `step < 60`, filtrer les neutres par distance depuis la meilleure source:
  - preferer distance <= 42;
  - autoriser au-dela seulement si `production >= 5` ou `ships / production <= 4.5`.
- Penaliser fortement les attaques > p90 top1 opening, soit environ 55, sauf opportunite claire.

Priorite 3: remplacer "juste assez" par "capture qui tient".

- Pour une cible neutre early:
  - `needed = ships_needed_to_capture(...)`
  - `floor = 14/16 selon mode`
  - `garrison = ceil(target.production * turns * 0.35 + target.ships * 0.15)`
  - `send = max(needed * aggression, floor, needed + garrison)`
- En midgame 4p:
  - `send = max(send, 24, needed + 0.35 * target.ships + 2 * target.production)`

Priorite 4: proteger le 2p.

- Ne pas supprimer les snipes.
- Ne pas augmenter globalement `min_source_ships`.
- Garder l'efficience 2p, mais faire monter le floor de capture a 14-15 uniquement sur vraies captures early.

Priorite 5: changer le scoring.

- Ajouter features:
  - `opening_capture_mass`: fraction des moves early >= 15 ships.
  - `opening_close_neutral`: ratio de cibles neutres close.
  - `expected_garrison_after_capture`: marge estimee apres capture.
  - `long_opening_attack_risk`: distance > 55 avant t60.
- Ajuster poids par defaut:
  - reduire penalite `overcommit_risk` early si les moves sont des captures neutres proches.
  - bonus fort a `opening_capture_mass` et `opening_close_neutral`.
  - malus a `long_opening_attack_risk` en 4p.

## Pseudo-code

Dans `_shot_option`:

```python
def _capture_send_floor(world, target, family, turns, needed):
    if family in SMALL_MOVE_FAMILIES:
        return 1, 0
    best_dist = min(_dist(src, target) for src in world.my_planets)
    opening = world.step < config.opening_punch_turns
    if opening:
        floor = config.opening_min_capture_send_4p if world.is_four_player else config.opening_min_capture_send_2p
        garrison = ceil(target.production * turns * config.capture_garrison_margin + target.ships * 0.15)
        return floor, garrison
    if world.is_four_player and world.step < 120:
        return config.midgame_min_capture_send_4p, ceil(target.production * 2.0 + target.ships * 0.35)
    return 1, 0
```

Puis:

```python
floor, garrison = _capture_send_floor(...)
send = ceil(max(min_send, needed * aggression, floor, needed + garrison))
send = min(left, send)
```

Dans `_kovi_opening_conversion`:

```python
targets = [
    t for t in world.neutral_planets
    if nearest_distance(t) <= 42 or t.production >= 5 or t.ships / max(1, t.production) <= 4.5
]
targets.sort(key=lambda t: (
    nearest_distance(t) > 42,
    t.ships / max(1, t.production),
    nearest_distance(t),
    -t.production,
))
```

## Protocole de test

- Smoke unitaire: `python -m pytest test_v9_smoke.py test_v9_robustness.py -q`.
- Micro-benchmark 2p: verifier que le taux 2p ne chute pas sous le checkpoint actuel.
- Benchmark 4p: comparer `bench4p`, `fronts`, `conversion_t60/t100`, `bb`, `xfer`.
- Critere d'acceptation rapide:
  - `bench4p +0.05` minimum vs baseline locale.
  - `fronts <= 2.7`.
  - median move opening >= 15 ships.
  - t50 production comparable ou superieure sans explosion de fronts.

## Implementation appliquee

Parametres ajoutes a V9:

- `opening_punch_turns = 55`
- `opening_min_capture_send_2p = 14`
- `opening_min_capture_send_4p = 16`
- `midgame_min_capture_send_4p = 24`
- `capture_garrison_margin = 0.22`
- `capture_target_ship_margin = 0.15`
- `midgame_capture_target_margin_4p = 0.35`
- `opening_close_neutral_dist_4p = 42.0`
- `opening_long_attack_risk_dist_4p = 55.0`
- `opening_source_commit_frac = 1.0`

Changements appliques:

- `MoveBuilder.attack_left` autorise le drainage de source en opening si la planete n'est pas menacee.
- `_shot_option` impose un floor + une marge de garnison aux vraies captures.
- `_kovi_opening_conversion`, `_aggressive_expansion` et `_wide_expansion` filtrent/priorisent les neutres proches en 4p opening.
- `extract_plan_features` reduit le malus d'overcommit pour les conversions early robustes, et ajoute un malus implicite aux attaques longues 4p avant t55.
- `V9Policy` donne un bonus metadata aux plans `opening_punch`.
- `submission.py` reprend les memes seuils pour que la soumission Kaggle ne diverge pas de V9 source.

Validation rapide:

- `python -m py_compile run_v9.py war_orbit/config/v9_config.py war_orbit/agents/v9/planner.py war_orbit/features/plan_features.py war_orbit/agents/v9/policy.py`
- `python -m py_compile submission.py run_v9.py war_orbit/config/v9_config.py war_orbit/agents/v9/planner.py war_orbit/features/plan_features.py war_orbit/agents/v9/policy.py`
- `python -m pytest test_v9_smoke.py test_v9_robustness.py test_submission_sun_safety.py -q`
- Controle opening local source: 2p expansion sans send `<14`; 4p expansion sans send `<16`.
- Controle opening `submission.py`: 2p expansion/balanced sans send `<14`; 4p expansion/balanced sans send `<16`.
