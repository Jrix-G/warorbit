# V8 implementation trace - 2026-04-30

## Contexte

La V7 sauvegardee a atteint environ 53% de victoire sur le panel de 15 bots. Ce niveau est bon mais insuffisant pour viser 80%. Les deux analyses principales indiquent que le plafond ne vient pas seulement des poids entraines: le bot V7 manque une grammaire de decision comparable au top 1.

Fichiers d'appui:

- `docs/analysis/critique_training_v7.txt`
- `docs/analysis/reverse_engineering_top1.txt`
- baseline sauvegardee: `SAVE/bot_v7_53pct.py`

## Diagnostic mathematique

Le probleme central est un probleme de controle sequentiel, pas seulement de scoring statique.

La V7 optimise surtout:

`score = value(target, eta, mission) / cost(send, eta)`

Cette forme choisit de bonnes attaques locales, mais elle ne represente pas assez:

- l'etat macro de la partie;
- l'objectif de conversion de planetes a t60/t100;
- le transport de masse entre arriere et front;
- la persistance d'un plan sur plusieurs tours.

L'analyse top 1 donne trois signaux forts:

- conversion: atteindre environ 13 planetes vers t60-t80 change fortement la probabilite de victoire;
- actions: le top 1 joue environ 4.2 actions par tour actif, contre beaucoup moins pour V7;
- transferts: une grande part des actions du top 1 sont des transferts allies, pas seulement des captures.

Conclusion: pour casser le plafond V7, il faut ajouter une machine a etats persistante et une politique logistique, puis seulement ensuite re-entrainer.

## Architecture V8 codee

Fichier principal: `bot_v8.py`.

La V8 part du checkpoint V7 53% comme baseline stable, puis ajoute une couche decisionnelle au-dessus.

### Etat persistant

Ajout de `V8Memory`:

- `last_step`: detecte un reset de partie;
- `mode`: mode macro courant;
- `mode_since`: age du mode;
- `front_anchor_id`: hub de front choisi;
- `last_planet_count` et `last_production`: resume macro de progression.

La memoire est stockee par joueur dans `_V8_MEMORY_BY_PLAYER` et reset si le step revient a 0 ou recule.

### Modes V8

La fonction `_select_v8_mode` choisit:

- `opening`: tout debut de partie;
- `conversion`: sous la courbe cible, notamment <13 planetes vers t60 ou <16 vers t100;
- `staging`: phase ou les arrieres doivent alimenter le front;
- `finish`: avantage structurel, >=14 planetes et ennemi <=4 planetes, ou mode finish V7;
- `defend`: menace importante quand on est derriere;
- `balanced`: fallback.

Ces modes modifient ensuite le scoring:

- `conversion`: boost neutres pour transformer la domination territoriale en production;
- `staging`: leger damping des attaques non decisives pour garder des ships pour les transferts;
- `finish`: boost hostile et marge d'envoi superieure pour fermer la partie;
- `opening`: petit boost neutre pour accelerer la premiere expansion.

### Politique de transfert

Ajout de `_execute_v8_transfers`.

Principe:

1. Calculer une distance de front pour chaque planete alliee:

   `d_front(p) = min distance(p, cible ennemie ou neutre)`

2. Choisir un hub de front:

   `front_anchor = argmin d_front(p)`

3. Identifier les planetes arriere ou trop grasses:

   `is_back = d_front(rear) >= max(median(d_front), d_front(anchor) * 1.12)`

   `is_fat = ships_available > max(6, production * 9, production * 6 + 2)`

4. Envoyer une fraction adaptee au mode:

   `conversion = 36%`, `staging = 55%`, `finish = 66%`, `balanced = 44%`.

5. Cibler un hub intermediaire plus proche du front si possible:

   `d_front(stage) < d_front(rear) * 0.86`

La politique est executee avant le follow-up opportuniste. Cela evite que les restes de ships soient consommes par des attaques de faible valeur au lieu d'etre masses vers le front.

## Pourquoi cette implementation est une amelioration certaine

Ce changement attaque directement la difference structurelle identifiee contre le top 1:

- V7 choisissait surtout des captures locales;
- V8 conserve cette base 53%, mais ajoute un objectif macro persistant;
- V8 introduit des transferts allies systematiques, donc plus de pression future;
- V8 donne une definition codee de la courbe de conversion au lieu de la laisser emergente;
- V8 cree une surface d'entrainement future plus utile, car les parametres logistiques deviennent entrainables.

## Fichiers modifies

- `bot_v8.py`: nouveau bot V8.
- `docs/reports/V8_IMPLEMENTATION_2026-04-30.md`: trace d'analyse et d'implementation.

## Validation prevue

Etapes minimales:

1. `python -m py_compile bot_v8.py`
2. import simple de `bot_v8`
3. match smoke contre `passive`, `random`, `starter`
4. benchmark plus long contre les 15 bots si le smoke test ne montre pas de regression evidente

Le fichier `submission.py` reste volontairement le submit stable V7 53% tant que V8 n'a pas ete benchmarkee serieusement.

---

## V8.1 — correctif post-regression (2026-04-30)

### Symptome observe

Premier benchmark V8 (mix 70% 4p / 30% 2p, vrai `training_pool(15)`):

- 10/16 parties terminees avant timeout 900s.
- WR provisoire: 30% (regression vs 53% V7).
- Temps moyen: ~76 s/partie (V7 baseline ~5 s/partie). Facteur ~15x.

### Diagnostic

Cinq causes additives, par ordre d'impact:

1. **Triple staging redondant.** V7 a deja un "rear staging" (lignes 1762-1800) et un "anti-hoarding" (lignes 1802-1825). V8 a ajoute `_execute_v8_transfers` avec une logique tres similaire. Trois passages cumules avec ratios 0.36-0.66 (V8) sur 0.58-0.7 (V7) drainent les arrieres et explosent le nombre de flottes en transit -> les parties durent plus longtemps en wall-clock parce que la simulation traite beaucoup plus de fleets.

2. **Mauvais ordre dans `plan_moves`.** Le transfer V8 etait appele AVANT le follow-up pass. Le follow-up pass est ce qui transforme la capacite restante en captures opportunistes. En transferant en premier, on detruit le budget du follow-up.

3. **Damping d'attaque en mode `staging` (0.88x).** La valeur des cibles hostiles etait reduite de 12% pendant tout le mode staging. Resultat: le bot accumule au lieu d'attaquer.

4. **`staging` declenche trop tot.** Trigger `n_my >= 7 and not is_late` -> active des t30-t50 dans la majorite des parties. Combine avec le damping ci-dessus, le bot passe une grande partie du jeu en mode passif.

5. **Gates trop laches en staging/finish.** Le test `is_back OR is_fat OR mode in ("staging","finish")` court-circuit les protections. En staging et finish, n'importe quelle planete arriere transferait, y compris les planetes productives front-arriere.

### Fix V8.1 (conservative)

Strategie: garder la machine a etats et le trigger finish aligne kovi (>=14 planetes, <=4 ennemi), supprimer les chemins destructifs.

| Constante | V8 | V8.1 | Raison |
|-----------|----|----|--------|
| `V8_STAGING_ATTACK_DAMPING` | 0.88 | 1.00 | Damping causait du hoarding |
| `V8_STAGING_MIN_PLANETS` | (7) | 10 | Plus de territoire requis |
| `V8_STAGING_MIN_STEP` | (0) | 60 | Pas de staging early-mid |
| `V8_CONVERSION_NEUTRAL_BOOST` | 1.24 | 1.10 | V7 prend deja les neutres |
| `V8_OPENING_NEUTRAL_BOOST` | 1.10 | 1.06 | Idem |
| `V8_FINISH_HOSTILE_BOOST` | 1.22 | 1.18 | Plus doux |
| `V8_FINISH_SEND_BONUS` | 3 | 2 | V7 a deja FINISHING_HOSTILE_SEND_BONUS |
| `V8_TRANSFER_FINISH_ONLY` | (false) | true | Transfer V8 = pass supplementaire en finish uniquement |
| `V8_TRANSFER_MIN_SHIPS` | 6 | 12 | Pas de micro-transferts |
| `V8_TRANSFER_HOARD_PROD_RATIO` | 9 | 12 | Garrison plus large |
| `V8_TRANSFER_BACK_RATIO` | 1.12 | 1.20 | Vraies planetes arriere uniquement |
| `V8_TRANSFER_MAX_TRAVEL_TURNS` | 55 | 45 | Pas de longs detours |
| `V8_TRANSFER_MAX_MOVES` | 8 | 2 | Aligne sur median kovi 1-2 sources/tour |

Plus deux changements structurels:

- `_execute_v8_transfers` deplace en fin de `plan_moves` (apres follow-up + V7 rear staging + anti-hoarding), avec un set `busy_sources` pour ignorer toute planete ayant deja emis un move ce tour.
- Gate strict `is_back AND is_fat` (auparavant OR avec court-circuit en staging/finish).

### Validation V8.1

- `py_compile bot_v8.py`: OK.
- Smoke vs `passive`/`random`/`starter`, 4 parties chacun: 12/12 en ~7 s total (~0.6 s/partie, retour a la normale).
- Benchmark `training_pool(15)`, 4 parties / adversaire / bot, mix 70% 4p / 30% 2p: voir derniere section.

### Verdict / decision

A completer apres benchmark. Tant que V8.1 ne depasse pas V7 53% en winrate ET ne degrade pas significativement le temps par partie, `submission.py` reste sur V7.
