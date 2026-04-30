# V8 Defeat Analysis — submission 52195726

Source: 40 épisodes Kaggle, analysés le 2026-04-30.
Résultat global: **17/40 wins (42%)** — 2p: 13/21 (61%), 4p: 4/19 (21%).

---

## 1. Taxonomie des défaites (23 parties perdues)

| Catégorie | Count | Mode | Description |
|-----------|-------|------|-------------|
| **overextend_fast_collapse** | 9 | 2p+4p | Expansion réussie, puis perte brutale de planètes |
| **mid_collapse** | 5 | 4p | Croissance stoppée t80-120, effondrement 10-25 tours |
| **early_4p_crush** | 4 | 4p | Jamais atteint masse critique, écrasé avant t100 |
| **slow_grind_2p** | 3 | 2p | Bon début, perte territoriale progressive sur 100+ tours |
| **gradual_loss** | 2 | 4p | Perte lente, jamais dominants |

**Fait clé: 15/19 défaites 4p rentrent dans les 3 premières catégories.**

---

## 2. Bug confirmé — Escalade infinie sur cibles déjà renforcées

**Mesuré**: ep75680588 seule → **88 missions gaspillées** sur turns 81-149.
Toutes sont `ENEMY→MORE`: planète ennemie au lancement, ennemie et plus forte à l'arrivée.

### Mécanisme exact

```
Tour T:   ships_needed_to_capture(target, ETA=30) = 20  → envoie 20 ships
Tour T+1: ennemi envoie 25 renforts sur même cible (ETA=15)
Tour T+2: arrival_ledger intègre les renforts ennemis
          ships_needed_to_capture(target, ETA=29) = 30  → envoie encore 25 ships
Tour T+3: ennemi renforce encore (il produit chaque tour)...
```

`build_arrival_ledger` trace correctement mes flottes déjà en transit. Mais:
1. L'ennemi produit chaque tour sur ses planètes proches
2. Il peut envoyer des renforts APRÈS mon calcul
3. Le tour suivant, le ledger voit ces renforts → `ships_needed` monte → bot envoie encore

**Résultat**: le bot tente de capturer une planète ennemie fortement renforcée en envoyant des micro-flottes successives de 7-33 ships, chacune insuffisante, toutes perdues.

### Root cause dans le code

```python
# bot_v7.py:1295
rough_needed = world.ships_needed_to_capture(target.id, rough_turns, planned_commitments)
# planned_commitments = commits du TOUR ACTUEL seulement
# Ne contient PAS les flottes déjà en transit des tours précédents
# → sous-estime la pression déjà exercée sur la cible
```

`planned_commitments` évite le double-comptage **intra-tour** mais pas **inter-tours**. Les flottes en transit des tours passés sont dans `arrival_ledger` (donc dans `projected_state`), mais le bot ne vérifie pas "est-ce que j'ai déjà assez de flottes en route pour prendre cette planète sans en envoyer davantage?"

### Fix #0 (nouveau, priorité CRITIQUE): "already enough in flight" check

```python
def my_fleets_en_route_to(target_id, world):
    """Ships de ma flotte déjà en transit vers cette cible."""
    return sum(
        entry[2]
        for entry in world.arrival_ledger.get(target_id, [])
        if entry[1] == world.player  # owner == me
    )

# Dans la boucle de sélection de cibles (plan_moves):
already_committed = my_fleets_en_route_to(target.id, world)
projected_needed = world.ships_needed_to_capture(target.id, turns, planned_commitments)

# Si j'ai déjà assez en route → skip
if already_committed >= projected_needed:
    continue

# Si target est ennemi ET j'ai déjà envoyé > 50% du nécessaire → skip (attendre résolution)
if target.owner != -1 and already_committed > projected_needed * 0.5:
    continue
```

**Impact attendu**: élimine ~80% des 88 missions gaspillées. Ships libérés = défense renforcée + nouvelles cibles.

---

## 3. Métriques observées pendant les défaites

### 2.1 garrison_ratio chroniquement bas

Le ratio `ships_sur_planètes / ships_totaux` mesuré à chaque tour perdu:

```
overextend_fast_collapse (ep75680588, 4p):
  t100: garrison=36%, min_garrison=11
  t110: garrison=22%, min_garrison=3
  t120: garrison=26%, min_garrison=5
  t130: garrison=23%, min_garrison=3   ← pic à 1283 ships
  t140: garrison=16%, min_garrison=2   ← début collapse
  t150: garrison=24%, min_garrison=4
  t175: garrison=20%, min_garrison=1   ← 574 ships restants

slow_grind_2p (ep75676284, 2p):
  t60:  garrison=28%, min_garrison=1
  t80:  garrison=27%, min_garrison=2
  t100: garrison=21%, min_garrison=3
  t120: garrison=34%, min_garrison=6
  t140: garrison=4%,  min_garrison=4   ← CRASH (2 planètes)
  t150: garrison=27%, min_garrison=29
```

**Observation**: garrison_ratio oscille 16-38%, mais `min_garrison` reste **1-5 ships** sur la majorité des planètes pendant TOUTE la partie. Une seule flotte ennemie de 6-10 ships capture n'importe quelle planète à tout moment.

### 2.2 Asymétrie flottes ennemies vs propres

```
ep75676284 (2p, slow_grind):
  t60:  n_own_fleets=48, n_enemy_fleets=43   ← parité
  t80:  n_own_fleets=48, n_enemy_fleets=45
  t100: n_own_fleets=51, n_enemy_fleets=62   ← renversement
  t110: n_own_fleets=42, n_enemy_fleets=69
  t120: n_own_fleets=30, n_enemy_fleets=83   ← 2.7x ennemi
  t140: n_own_fleets=34, n_enemy_fleets=98   ← 2.9x ennemi
→ Collapse immédiat après t140

ep75680588 (4p, overextend):
  t95:  n_own_fleets=34, n_enemy_fleets=49
  t110: n_own_fleets=43, n_enemy_fleets=38   ← pic planètes (16pl)
  t125: n_own_fleets=48, n_enemy_fleets=32   ← pic ships (1283)
  t140: n_own_fleets=57, n_enemy_fleets=18   ← mais ships chutent déjà
```

**Observation**: le nombre de flottes ennemies devient 2-3x supérieur avant le collapse terminal. Le bot continue d'envoyer des flottes offensives même quand il perd du territoire.

### 2.3 Pattern early_4p_crush

```
ep75673784 (4p, crushed):
  t0:  10 ships, 1 planète
  t30: 48 ships, 3 planètes, 13 flottes ennemies
  t50: 66 ships, 3 planètes, 5 flottes ennemies
  t60: 89 ships, 4 planètes, 11 flottes ennemies
  t70: 114 ships, 3 planètes → peak
  t80: 114 ships, 2 planètes, 20 flottes ennemies
  t85: 57 ships, 1 planète   ← collapse brutal
  t90: 46 ships, 0 planètes  ← mort
```

**Observation**: jamais dépassé 4 planètes en 4p. Les 3 adversaires expansionnent tous en parallèle. À t70 l'ennemi a déjà 20 flottes en transit vs 4 flottes propres.

---

## 3. Causes racines identifiées

### Cause #1: min_garrison structurellement trop bas (CRITIQUE)

**Symptôme**: `min_garrison = 1-5` sur 80%+ des tours, sur toutes les planètes non-frontières.

**Mécanisme**: `keep_needed` calcule le garrison minimum via binary search sur la timeline. Mais le résultat est souvent 0-5 parce que:
- Si aucune flotte ennemie en transit détectée vers cette planète → keep=0
- Le bot envoie *tous* les ships excédentaires en offensive
- Les planètes arrière/neutres récemment capturées restent à 1-2 ships

**Conséquence**: l'adversaire peut capturer n'importe quelle planète avec 2-6 ships. En 4p, avec 3 adversaires, des dizaines de micro-flottes peuvent sonder simultanément.

**Root cause dans le code**:
```python
# bot_v7.py - keep_needed
# Binary search cherche le minimum pour survivre aux flottes *déjà connues*
# Pas de buffer contre flottes invisibles / futures
# Pas de garrison minimum absolu proportionnel à la production
```

### Cause #2: ratio flottes offensives/défensives déséquilibré (IMPORTANT)

**Symptôme**: garrison_ratio 16-28% → 72-84% des ships sont *en transit* à chaque instant.

**Mécanisme**: `plan_moves` alloue agressivement. Dès qu'une planète dépasse `keep_needed`, elle envoie le surplus. Mais `keep_needed` étant trop bas, presque tout part en offensive.

**Conséquence**: même avec 1000 ships totaux, seulement 200-280 défendent réellement les planètes. Un ennemi coordonné avec 300 ships bien placés peut prendre 10 planètes en 20 tours.

### Cause #3: aucune detection de pression multi-directionnelle (4p spécifique)

**Symptôme**: en 4p, le bot perd toujours dans les mêmes circonstances — simultanément attaqué depuis 2-3 directions.

**Mécanisme**: `WorldModel` détecte les flottes ennemies *en transit vers mes planètes* mais ne modélise pas:
- La capacité offensive totale de chaque adversaire
- Les flottes qui *pourraient* être envoyées dans N tours
- La pression coordonnée implicite (3 ennemis = 3x la pression)

**Conséquence**: le bot réagit aux menaces détectées mais pas aux menaces latentes. En 4p, même si aucun ennemi n'est en transit, 3 adversaires peuvent lancer une attaque simultanée le tour suivant.

### Cause #4: expansion trop timide en 4p early game

**Symptôme**: en early_4p_crush, jamais plus de 4 planètes à t70. Les adversaires ont souvent 6-10 planètes au même tour.

**Mécanisme**: les seuils `TWO_PLAYER_OPENING_TURN_LIMIT` et les marges de neutrals sont calibrés pour 2p. En 4p, le rythme d'expansion doit être plus rapide car:
- Les neutrals sont contestés par 3 adversaires
- La fenêtre pour capturer est plus courte
- Le retard d'expansion = retard de production = défaite assurée mid-game

---

## 4. La solution — Plan d'ingénierie

### Fix #1: Garrison minimum absolu (priorité CRITIQUE)

**Principe**: chaque planète doit avoir un garrison minimum = f(production, distance_au_front).

```python
def min_garrison_floor(planet, world_model):
    """Garrison minimum inconditiennel, indépendant de keep_needed."""
    prod = planet.production
    
    # Base: 3 tours de production (buffer contre micro-flottes)
    base = prod * 3
    
    # Bonus si planète frontier (ennemis proches)
    dist_to_nearest_enemy = world_model.nearest_enemy_dist(planet)
    if dist_to_nearest_enemy < 20:
        base = max(base, prod * 8)   # 8 tours de production
    elif dist_to_nearest_enemy < 35:
        base = max(base, prod * 5)   # 5 tours de production
    
    # Absolue: jamais moins de 5 ships
    return max(base, 5)

# Dans keep_needed:
floor = min_garrison_floor(planet, world_model)
return max(current_keep_needed, floor)
```

**Impact attendu**: min_garrison passe de 1-5 à 15-40 sur planètes arrière, 30-80 sur planètes frontières.

### Fix #2: Plafond garrison_ratio (priorité IMPORTANTE)

**Principe**: garantir que ≥35% des ships totaux restent sur les planètes à tout moment.

```python
# Dans plan_moves, avant d'envoyer:
total_my_ships = sum_all_my_ships(obs)
currently_on_planets = sum(p.ships for p in my_planets)
garrison_ratio = currently_on_planets / total_my_ships if total_my_ships > 0 else 1.0

GARRISON_FLOOR_RATIO = 0.35  # au moins 35% en défense
if garrison_ratio < GARRISON_FLOOR_RATIO:
    # Throttle les offensives: ne pas envoyer plus que
    # ce qui maintient le ratio
    max_sendable = total_my_ships * (1 - GARRISON_FLOOR_RATIO) - (total_my_ships - currently_on_planets)
    # Couper les missions les moins prioritaires
```

**Impact attendu**: garrison_ratio ne descend plus sous 30% → l'adversaire doit engager une vraie armée pour prendre des planètes.

### Fix #3: Threat score global par adversaire (priorité IMPORTANTE)

**Principe**: modéliser la *capacité offensive potentielle* de chaque adversaire, pas seulement les flottes détectées.

```python
class ThreatModel:
    def adversary_threat_score(self, enemy_id, world_model):
        """Menace totale d'un adversaire = flottes en transit + capacité de lancement."""
        in_transit = sum(f.ships for f in world_model.enemy_fleets_by_owner(enemy_id))
        
        # Capacité de lancement dans les N prochains tours
        enemy_planets = world_model.planets_by_owner(enemy_id)
        production_capacity = sum(p.production * 10 for p in enemy_planets)  # 10 tours
        garrison_available = sum(max(0, p.ships - p.production * 3) for p in enemy_planets)
        
        return in_transit + garrison_available * 0.5  # factor discounted
    
    def total_pressure_on_planet(self, planet, world_model):
        """Pression totale sur une planète = flottes détectées + menace latente des proches."""
        direct = sum(f.ships for f in world_model.fleets_targeting(planet.id))
        
        # Ennemis à portée de frappe dans 15 tours
        latent = 0
        for enemy_id in world_model.active_enemies():
            nearby_enemy_planets = [p for p in world_model.planets_by_owner(enemy_id)
                                     if world_model.travel_time(p, planet) < 15]
            latent += sum(p.ships * 0.3 for p in nearby_enemy_planets)
        
        return direct + latent
```

**Dans keep_needed**: `return max(timeline_keep, threat_score * 1.2, garrison_floor)`

### Fix #4: Mode 4p — early expansion agressif (priorité IMPORTANTE)

**Principe**: en 4p, les 60 premiers tours doivent être dédiés à l'expansion maximale.

```python
IS_4P = len(obs.initial_planets) > 8  # heuristique

if IS_4P and obs.step < 60:
    # Réduire les marges de neutrals: capturer même avec ratio 1.05 vs 1.2
    NEUTRAL_MARGIN = 1.05  # au lieu de TWO_PLAYER_NEUTRAL_MARGIN_BASE=1
    
    # Prioriser la distance absolue: captures les plus proches en premier
    # même si une planète lointaine est "plus rentable"
    
    # Ne pas garder de ships en réserve rear staging avant t60
    REAR_STAGING_MIN_TURN = 60
```

**Logique**: en 4p il faut atteindre 6+ planètes avant t60, sinon la fenêtre est fermée. Le coût d'un garrison légèrement plus bas au début est inférieur au bénéfice d'une planète de production supplémentaire.

### Fix #5: Throttle offensif en cas de pression multi-ennemis (4p spécifique)

**Principe**: détecter quand 2+ adversaires exercent une pression simultanée et passer en mode défensif.

```python
def should_throttle_offense(world_model, obs):
    """True si trop de pression pour maintenir des offensives."""
    pressures = [world_model.adversary_threat_score(eid, world_model)
                 for eid in world_model.active_enemies()]
    
    top2_pressure = sum(sorted(pressures, reverse=True)[:2])
    my_total_ships = world_model.total_my_ships()
    
    # Si les 2 adversaires les plus menaçants ont ensemble plus de ships que moi
    if top2_pressure > my_total_ships * 0.8:
        return True
    
    # Si garrison_ratio < 25% ET ennemis nombreux
    if world_model.garrison_ratio() < 0.25 and len(world_model.active_enemies()) >= 2:
        return True
    
    return False

# Dans plan_moves:
if should_throttle_offense(world_model, obs):
    # Mode consolidation: seulement des missions de reinforcement
    # Pas de nouvelles captures offensives
    # Retirer les flottes des missions non-engagées si possible
```

---

## 5. Priorisation et impact attendu

| Fix | Lignes de code | Impact 4p | Impact 2p | Risque |
|-----|---------------|-----------|-----------|--------|
| **#0 already_in_flight check** | ~15 | +++ | +++ | Très faible — skip redondant |
| #1 min_garrison_floor | ~20 | +++ | ++ | Faible — pure défense |
| #2 garrison_ratio cap | ~30 | +++ | ++ | Moyen — peut ralentir expansion |
| #3 ThreatModel latent | ~50 | +++ | + | Moyen — coût CPU négligeable |
| #4 early 4p expansion | ~20 | +++ | n/a | Faible — conditionnel |
| #5 throttle offense | ~40 | ++ | + | Élevé — risque trop passif |

**Ordre d'implémentation recommandé**: #0 → #1 → #4 → #2 → #3 → #5

**Objectif**: 4p passe de 21% → 45%+ (parité avec les bons bots publics à ~1100 ELO).

---

## 6. Validation

### Benchmark local minimal

```bash
# Avant fix:
.venv/bin/python -c "
from SimGame import run_match
import bot_v7
from opponents import ZOO
opp = ZOO['notebook_physics_accurate']
wins = sum(
    (1 if run_match([bot_v7.agent, opp, ZOO['greedy'], ZOO['distance']], seed=42+i)['winner'] == 0 else 0)
    for i in range(20)
)
print(f'4p wins: {wins}/20')
"
```

### Métriques à mesurer sur les replays

Après chaque fix, vérifier sur les 23 parties perdues (en simulation locale):
- `min_garrison` médian > 10 ✓
- `garrison_ratio` jamais < 28% ✓
- Parties `early_4p_crush` : atteindre 6 planètes avant t60 dans 80% des cas ✓
- `mid_collapse` turn > 150 dans 80% des cas ✓

---

## 7. Ce qui ne sera PAS fixé ici

- **slow_grind_2p** (3 cas) : le bot perd face à des joueurs qui ont un meilleur scoring policy. Fix ML, pas de règle.
- **gradual_loss** (2 cas) : adversaires strictement supérieurs. Nécessite behavioral cloning (Phase 2).
- **kovi-tier** (ELO 2500+) : hors scope pour V8 rule-based. Requiert league training (Phase 3).
