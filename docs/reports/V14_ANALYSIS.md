# V14 — Analyse des limites et du gap 2p/4p

## Résumé

V14 performe bien en 2p mais échoue en 4p. Ce n'est pas un problème de tuning — c'est structurel.

---

## Architecture V14

```
obs → get_candidates() [bot_v13] → build_v14_features() → V14Scorer (64→128→64→1) → select_actions()
                                                                     ↓ si scorer absent ou noop < step 160
                                                              bot_v12.agent() [fallback]
```

Le scorer est entraîné par behavioral cloning sur replays Kaggle top1 (`bowwowforeach`), puis fine-tuné par ES avec ancrage BC.

---

## Pourquoi bon en 2p

Le fallback V12 est un bot tactique solide. En 2p, quand le scorer hésite (retourne noop avant step 160), V12 prend le relais et fait le bon move. Le scorer n'a pas à être juste — il suffit qu'il ne soit pas catastrophique.

En 2p, les features sont aussi plus cohérentes avec ce que le scorer a appris : un seul ennemi, dynamique simple, candidates bien couverts par bot_v13.

---

## Pourquoi nul en 4p

### 1. BC sur replays 4p = signal biaisé

Le top1 joue avec des adversaires spécifiques dans un contexte politique précis : qui attaquer, quand lâcher un ennemi affaibli pour frapper un autre, quand temporiser pendant que deux ennemis s'affrontent.

Ces décisions dépendent du contexte global, pas juste de l'état local des planètes. Le BC apprend la *surface* du comportement (envoyer X ships vers Y) sans le *raisonnement* sous-jacent (parce que joueur 3 est distrait et joueur 2 est à 40% de sa force max).

### 2. Un seul bit binaire pour toute la dynamique 4p

```python
out[30] = 1.0 if stats["n_players"] >= 4 else 0.0
```

Toute la politique 4p est encodée dans un seul feature binaire. Le scorer ne peut pas distinguer :
- "je suis dominant, consolider"
- "je suis le plus faible, survivre"
- "deux ennemis se battent, opportunisme"
- "focus le joueur à 2 planètes pour l'éliminer"

### 3. Générateur de candidats non 4p-aware

`get_candidates()` délègue à `bot_v13.generate_all_candidates()`. Ce générateur ne produit pas de candidats politiques comme :
- "ignorer joueur X pour concentrer sur joueur Y affaibli"
- "transfer latéral pour préparer une frappe multi-source sur Z"
- "laisser joueur 1 et 2 se battre, capturer planètes neutres pendant ce temps"

Le scorer score des candidats qui n'existent pas dans le set → ne peut pas apprendre la bonne politique.

### 4. Contradiction fallback/scorer en 4p

En 4p avec 3 ennemis, V12 greedy peut attaquer le mauvais ennemi (le plus proche, pas le plus faible). Le scorer retourne noop → V12 prend le relais → mauvaise décision politique → perte d'avantage.

---

## Ce qui changerait vraiment

### Court terme (sans refonte)

Ajouter des features politiques au vecteur existant :

```python
# Par ennemi (3 ennemis max en 4p)
for i, enemy_id in enumerate(sorted_enemy_ids[:3]):
    ep = [p for p in planets if int(p[1]) == enemy_id]
    ef = [f for f in fleets if int(f[1]) == enemy_id]
    out[base + i*5 + 0] = len(ep) / 40.0           # nb planètes ennemi i
    out[base + i*5 + 1] = sum(p[5] for p in ep) / total_ships  # force relative
    out[base + i*5 + 2] = sum(p[6] for p in ep) / total_prod   # prod relative
    out[base + i*5 + 3] = sum(f[6] for f in ef) / total_ships  # flotte en vol
    out[base + i*5 + 4] = 1.0 if len(ep) <= 2 else 0.0        # éliminable ?
```

Et ajouter des candidats "focus enemy X" dans le générateur.

### Moyen terme (refonte 4p)

- Entraîner deux scorers séparés : un 2p, un 4p
- En 4p : identifier le "target priority" (ennemi le plus faible/menaçant) et conditionner tous les candidats dessus
- Générer explicitement des candidats politiques (ignore, opportunism, finish)

### Long terme

- Self-play 4p plutôt que BC sur un seul top1 — le top1 a une style, le self-play apprend la dynamique générale
- Ou : ES directement en 4p avec reward = rang final (1er=1.0, 2ème=0.5, 3ème=0.2, 4ème=0.0)

---

## Conclusion

V14 est un bot 2p avec un vernis 4p. Le BC sur un seul joueur top1 sans features politiques ne peut pas capturer la dynamique multi-joueur. La priorité pour passer le cap en 4p : générateur de candidats 4p-aware + features par-ennemi + scorer conditionné sur le contexte politique.
