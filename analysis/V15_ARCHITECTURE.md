# V15 — plan d'architecture v3 (search-based, cible 1400 ELO)

> Réécriture complète. Les versions v1/v2 partaient de "V7 immuable + couches" — ce qui plafonne par construction près de l'horizon de V7 (~900 ELO). Cette v3 vise réellement **1400+** en ajoutant ce qui manque : de la **recherche**.

---

## 1. Cible ELO et pourquoi elle est atteignable

### 1.1 Le budget ELO

| composant | gain ELO | justification |
|---|---|---|
| V7 (base) | ~700 | mesuré : 53% vs zoo notebooks |
| + CMA-ES sur les ~20 constantes 4p/multi-source non-tunées | +50 à +150 | V7 a déjà tuné ses 14 clés principales ; il en reste ~20 jamais optimisées |
| + Recherche (MCTS ~2-4 ply, rollout V7) | +300 à +500 | le levier principal — voir §1.2 |
| + Évaluation apprise sur replays top-10 | +50 à +150 | meilleur scoring des états-feuilles |
| **Total** | **~1100 à 1500** | médiane réaliste **1100-1400**, stretch **1400-1600** |

- **Cible affichée : 1400 ELO.**
- **Attendu réaliste : 1100-1400.**
- **Plancher garanti : ~700** (V7 reste le fallback — voir §3).

### 1.2 Pourquoi la recherche fonctionne — l'argument scientifique

**Le problème de V7 n'est pas sa stratégie, c'est sa myopie.**
V7 est un bot *gourmand* : à chaque tour il choisit le coup qui maximise une évaluation *statique de l'instant présent*. Il ne peut pas voir qu'un coup qui paraît bon maintenant mène à une position perdante 10 tours plus tard. C'est une limite structurelle de tout bot heuristique sans lookahead.

**La recherche corrige exactement ça.** Au lieu d'évaluer l'*apparence* d'un coup, on simule ses *conséquences* et on évalue l'état *résultant*. Principe établi en théorie des jeux :

1. **Chaque ply de recherche supplémentaire augmente la force de jeu.** Aux échecs, c'est empiriquement ~+200 ELO/ply sur les premiers niveaux (courbe de Thompson, 1982). La valeur exacte dépend du jeu, mais la direction est universelle : voir plus loin = jouer mieux.
2. **Même une recherche peu profonde avec une éval correcte écrase le pur heuristique**, parce qu'elle élimine les pires gaffes — celles que l'heuristique commet en étant myope.
3. **MCTS + une politique de rollout médiocre ≫ la politique seule.** C'est le résultat fondateur de la recherche Monte-Carlo (et la base d'AlphaGo). On utilise V7 comme politique de rollout : un bot 700-ELO produit des simulations "raisonnables", et MCTS converge vers des coups bien meilleurs que V7 nu.

**Pourquoi ça s'applique précisément ici :**
- L'audit D1 (2630 replays top-10) a montré que le jeu top-Kaggle = heuristiques minables + **consistance**. Le reviewer doctorant a confirmé : **aucune barrière "RL-only"** entre nous et le top 10. "Consistance" = ne pas commettre de gaffe. La recherche *est* le mécanisme anti-gaffe.
- V7 encode déjà de bons priors stratégiques (53% vs zoo). On ne les jette pas — on garde V7 comme **générateur de coups** et **politique de rollout**. La recherche ajoute uniquement la profondeur que V7 n'a pas.

**Pourquoi pas plus haut que ~1500 ?** Deux plafonds honnêtes :
- **Latence Kaggle** (~1 s/tour). Même à 50 µs/pas, un rollout de 30 pas = 1,5 ms → ~600 rollouts/tour : un MCTS *fin*. Aller vers 1800+ exigerait soit un moteur C (≥6000 rollouts), soit un réseau de politique guidant la recherche (style AlphaZero) — hors périmètre 6 semaines.
- **Modèle d'adversaire imparfait.** On ne connaît pas exactement le jeu des adversaires Kaggle ; notre recherche utilise un modèle approché. Cette approximation borne le gain.

---

## 2. Architecture

```
  Tour de jeu : le serveur appelle agent(obs)
        │
        ▼
  ┌────────────────────────────────────────────────┐
  │  bot_v15.agent(obs)                             │
  │                                                 │
  │  1. candidats = V7.generer_coups(obs)   ← V7    │
  │  2. SI budget temps OK :                        │
  │       coup = MCTS(obs, candidats)               │
  │              │                                  │
  │              ├── rollouts via v15_fast_sim      │  ← moteur de réflexion
  │              ├── politique de rollout = V7      │
  │              └── éval feuille = V7-score / NN   │
  │  3. SINON (ou si MCTS échoue) :                  │
  │       coup = V7.agent(obs)              ← FALLBACK
  │                                                 │
  │  return coup                                    │
  └────────────────────────────────────────────────┘
```

### Composant A — `v15_fast_sim` (le moteur de réflexion)
Réimplémentation du pas de jeu Orbit Wars (planètes, flottes, combats, comètes) en **numpy vectorisé** (puis C si besoin).
- Cible vitesse : **<50 µs/pas** (vs ~3000 µs pour `orbit_wars.py`) → facteur ~60×.
- Validé par **équivalence exacte** contre `OfficialFastGame` sur ≥10 000 scénarios aléatoires.

### Composant B — MCTS (le cœur)
Recherche arborescente Monte-Carlo sur `v15_fast_sim` :
- **Génération de coups** : les candidats de V7 (déjà bons), avec *progressive widening*.
- **Politique de rollout** : V7 (rollouts rapides et raisonnables).
- **Évaluation de feuille** : score heuristique de V7 au départ, puis éval apprise (Composant D).
- **Modèle d'adversaire** : mixte (V7 / V12 / variantes) pour éviter l'hallucination self-play.
- **Budget** : timer monotone, abort à 800 ms, fallback V7.

### Composant C — CMA-ES (tuning)
Optimise par évolution : (a) les ~20 constantes heuristiques 4p/multi-source jamais tunées, (b) les hyperparamètres de MCTS (constante d'exploration, profondeur de rollout, widening). Fitness = winrate sur panel, mesuré sur `OfficialFastGame` (désormais reproductible).

### Composant D — Évaluation apprise (optionnel)
MLP entraîné par régression sur les replays top-10 : prédit "qui gagne depuis cet état". Remplace le score statique de V7 comme évaluateur de feuille MCTS. Boost de qualité de recherche.

### Composant E — V7 fallback (filet de sécurité)
Si MCTS dépasse le budget temps ou lève une exception → on renvoie le coup de V7. **Garantit que V15 n'est jamais pire que V7 (~700 ELO).**

---

## 3. Le filet de sécurité — pourquoi on ne peut pas régresser

Tout le drame V8→V14 = des régressions silencieuses sous V7. V15 l'empêche structurellement :
- `bot_v15` flags off ≡ `bot_v7` bit-pour-bit (M0, déjà validé).
- MCTS est *opt-in* derrière un flag, et tombe sur V7 en cas de dépassement temps / erreur.
- Chaque phase a une *gate* mesurée. Une phase qui ne passe pas sa gate n'est pas livrée.
→ Le pire scénario de V15 = V7. Le scénario réaliste = V7 + recherche.

---

## 4. Roadmap par phases (avec go/no-go)

| phase | livrable | gate (critère de réussite) | si échec |
|---|---|---|---|
| **P0** ✅ | `bot_v15` wrapper + V7 fallback + déterminisme | M0 passé (fait) | — |
| **P0.5** ✅ | `OfficialFastGame` reproductible (fix `random.seed`) | même seed → partie identique (fait) | — |
| **P1** | `v15_fast_sim` (numpy) | (a) équivalence exacte vs `OfficialFastGame` sur 10k scénarios ; (b) ≥60× plus rapide | **projet plafonne ~900 ELO** : on se rabat sur CMA-ES seul (P3 sans P2) |
| **P2** | MCTS sur fast-sim, rollout V7, éval V7-score | bat V7 de ≥+15% winrate **dans le budget latence Kaggle** | améliorer l'éval feuille, ou réduire la profondeur ; si rien ne marche → CMA-ES seul |
| **P3** | CMA-ES sur 20 constantes + hyperparams MCTS | ≥+5% winrate vs P2 | garder P2 tel quel |
| **P4** | éval apprise sur replays (Composant D) | ≥+3% winrate vs P3 | livrer P3 |

Soumission Kaggle à la fin de chaque phase pour mesurer l'ELO réel.

**Le go/no-go critique du projet entier = P1.** Si le fast-sim n'atteint pas l'équivalence exacte ou la vitesse cible, la recherche est impossible et on retombe sur "+100-300 ELO". C'est pour ça que P1 est en premier et gaté durement.

---

## 5. Calendrier (6 semaines)

- **S1-2 — P1** : `v15_fast_sim`. Le morceau le plus dur. Équivalence exacte obligatoire.
- **S3 — P2** : squelette MCTS. Benchmark vs V7.
- **S4 — P3** : CMA-ES (tuning constantes + hyperparams).
- **S5 — P4** : éval apprise. Itérations de soumission.
- **S6** : gel, variantes anti-meta, post-mortem.

---

## 6. Risques (honnêtes)

1. **P1 échoue** (équivalence exacte non atteinte, ou numpy pas assez rapide) → plus de recherche → ~900 ELO. Mitigé : P1 d'abord, gate dure ; option C-engine en repli.
2. **Latence Kaggle** : MCTS trop lent → peu de rollouts → gain faible. Mitigé : budget timer + fallback V7.
3. **Modèle d'adversaire** : un modèle trop éloigné du vrai jeu Kaggle borne le gain de recherche. Mitigé : modèle mixte, calibré sur replays.
4. **`v15_fast_sim` diverge subtilement** du moteur officiel → le bot raisonne sur de fausses règles. Mitigé : équivalence exacte sur 10k scénarios = gate P1 non-négociable.

---

## 7. Fichiers

```
bot_v15.py              ✅ wrapper + fallback
v15_config.py           ✅ config
local_simulator/official_fast.py  ✅ fix random.seed (P0.5)
v15_fast_sim.py         P1 — moteur de réflexion vectorisé
tests/test_fast_sim_equivalence.py  P1 — gate équivalence
v15_search.py           P2 — MCTS
v15_eval.py             P4 — évaluation apprise
cma_es_v15.py           P3 — tuning évolutionnaire
benchmark_v15.py        ✅ harness winrate + Wilson CI
```
