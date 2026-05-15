# V15 — architecture plan v2 (post-review)

> **Révision après peer review** (`V15_REVIEW.md`). Verdict initial : *accept with major revisions*. Cette v2 intègre les corrections §3 issues 1–3, §5 algorithmiques, §6 ingénierie, et les 10 recommandations finales.

## 0. Pourquoi V7 comme base

Mesures empiriques (vs notebook zoo, 4p):
- V7 : ~53% winrate, ~700 ELO réel. Seule version qui gagne.
- V14 : 0/8 (mesuré 2026-05-15). V8–V14 ont tous régressé silencieusement.
- D1 audit (2630 replays top-10) → patterns gagnants *heuristiques*, pas de "secret RL".

**Décision** : V15 = V7 comme noyau immuable, couches ajoutées au-dessus avec flags. Si toutes les couches off → `bot_v15 ≡ bot_v7` (testé par déterminisme full-game, pas juste sample d'états).

## 1. Architecture en couches

```
                     ┌──────────────────────────────────┐
                     │  Layer 4 — MCTS / lookahead      │  recherche, opt-in, gated par M3a
                     │  flag: V15_SEARCH=1              │  fallback: skip
                     └──────────────────────────────────┘
                                  ▲ rerank
                     ┌──────────────────────────────────┐
                     │  Layer 3 — BC re-ranker (recherche)│ MLP appris par BC sur replays
                     │  flag: V15_RANKER=1              │  branche timeboxée 1 semaine
                     └──────────────────────────────────┘
                                  ▲ rescore
                     ┌──────────────────────────────────┐
                     │  Layer 2b — CMA-ES tuned configs │  **track principal d'amélioration**
                     │  flag: V15_CONFIG=<profile>      │  optimisé sur constantes V15Config
                     └──────────────────────────────────┘
                                  ▲ override constantes
                     ┌──────────────────────────────────┐
                     │  Layer 2a — D1 heuristic patches │  patchs constantes mesurés sur replays
                     │  flag: V15_PATCH_<name>=0/1      │  *bisect obligatoire à M1*
                     └──────────────────────────────────┘
                                  ▲ score adjustments
                     ┌──────────────────────────────────┐
                     │  Layer 1 — V7 core (immuable)    │  candidate gen, world model, art-of-war 2p
                     │  bot_v15_core.py = factor de V7  │  jamais modifié dans V15
                     └──────────────────────────────────┘
```

**Changement clé vs v1** : la Layer 3 (BC re-ranker) **n'est plus le track principal**. Le track principal devient **Layer 2b : CMA-ES sur les constantes V15Config**. Justification (review §7 alt-1) : le re-ranker BC ne peut, par construction, que **réordonner** les candidats de V7 — il ne peut pas dépasser l'horizon stratégique de V7. Seul un tuning des constantes (qui change *quels* candidats V7 génère) peut faire bouger ce plafond. CMA-ES sur ~30–50 scalaires avec fitness=200-game eval est mieux posé que PPO sur 5k params à reward terminale.

## 2. Couche par couche

### Layer 1 — V7 core (immuable)
- `bot_v15_core.py` = factorisation de `bot_v7.py`. Toutes les constantes V7 déplacées vers un dataclass `V15Config` (`v15_config.py`).
- Aucun monkey-patching de module. Le bot prend une `V15Config` en argument.
- **Test M0** : *full-game determinism* sur 10 parties (seed fixe, opponent fixe). V15(config=V7_DEFAULTS, all_flags=off) doit produire un flux d'actions **bit-identique** à V7. Pas juste 1000 obs — l'audit doit être trajectoire complète (review §4.4).
- **Hypothèse falsifiable** : si M0 échoue, il y a un état caché global dans V7. À corriger avant toute autre chose.

### Layer 2a — D1 heuristic patches (downgrade : soft priors)
Status épistémique corrigé : les patterns D1 sont **corrélationnels**, possiblement confondus par "position gagnante". Le contrôle ELO≤30 (`V15_FINDINGS.md` l.18) est entre-joueurs, pas intra-partie (review §4 second bullet). On les traite comme **soft priors à valider par A/B**, pas comme vérités.

| pattern D1 | constante | V7 | candidate V15 | flag | n A/B requis | gate |
|---|---|---|---|---|---|---|
| Multi-source `TOP_K` | `MULTI_SOURCE_TOP_K` | 10 | 14 | `V15_PATCH_MULTI_K` | 200 | Wilson 95% LB ≥ 0% |
| Multi-source penalty | `MULTI_SOURCE_PLAN_PENALTY` | 0.97 | 0.99 | `V15_PATCH_MULTI_PEN` | 200 | idem |
| 3-source penalty | `THREE_SOURCE_PLAN_PENALTY` | 0.94 | 0.97 | `V15_PATCH_TRI_PEN` | 200 | idem |
| Multi-source bonus early | new : +5 sur score si ≥2 sources, dès t0 | — | +5 | `V15_PATCH_MULTI_BONUS` | 200 | idem |
| Commit hard 4p | `FOUR_PLAYER_ROTATING_SEND_RATIO` | 0.72 | 0.90 | `V15_PATCH_SEND_4P` | 200 | idem |
| Commit hard 2p | `TWO_PLAYER_SEND_RATIO` (new) | — | 0.92 | `V15_PATCH_SEND_2P` | 200 | idem |
| Anti over-expand 4p | `FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT` | 0.92 | 0.86 | `V15_PATCH_4P_NEUT` | 200 | idem |
| Anti-opportunistic gate | gate `opportunistic_expand` après step 40 ou planets≥6 | — | new | `V15_PATCH_OPP_GATE` | 200 | idem |
| Late first-attack 4p | `FOUR_PLAYER_ROTATING_TURN_LIMIT` | 14 | 22 | `V15_PATCH_LATE_4P` | 200 | idem |

**Procédure M1 (mandatory bisect)** : chaque patch testé *individuellement* contre V7 baseline (V15 flags=off). 200 games × 4 modes. Garde uniquement les patchs dont **Wilson 95% lower bound ≥ 0%**. Les autres → archivés. Puis test de l'union des survivants (overlap effects).

**Note importante** : `artOfWar.txt` Rule 1 ("opening agressif t2–8") est dérivée de 95 parties 2p. Le D1 pattern #3 montre que 4p first-attack est t29–35, **contradictoire**. La règle artOfWar n'est appliquée **que en 2p** dans V15. Le flag `V15_AOW_RULE1` gate cette logique par mode.

### Layer 2b — CMA-ES tuned configs (track principal)
Nouvelle couche, élevée au statut de track principal sur recommandation review §7-alt-1.

> **Découverte M0 (2026-05-15)** : `bot_v7.py:1707` expose déjà `_TUNABLE_KEYS` (14 constantes) + `_DEFAULT_CHECKPOINT_HEURISTIC_SPECS` (specs ES mean/std). Le checkpoint `scorer_v7_kaggle.npz` (gen=273, wr=0.533) **est le résultat d'une optimisation ES déjà effectuée** sur ces 14 clés. Conséquence : re-tuner les 14 mêmes clés serait largement redondant, et l'hypothèse H1 ("V7 plafonne car constantes sous-optimales") est plus faible qu'estimée. Le levier CMA-ES réel est :
> 1. **Étendre** l'ensemble tunable aux ~20 constantes 4p / multi-source / margins absentes de `_TUNABLE_KEYS` (FOUR_PLAYER_*, MULTI_SOURCE_*, THREE/FOUR_SOURCE_*, REINFORCE_*, AOW_*).
> 2. Re-tuner contre un **panel d'adversaires différent/meilleur** que celui utilisé pour `scorer_v7_kaggle.npz`.
> 3. Tuner **conjointement** les 14 anciennes + les ~20 nouvelles (les couplages peuvent déplacer l'optimum des 14).

- **Variables d'optim** : ~34 scalaires = 14 `_TUNABLE_KEYS` existants + ~20 nouveaux (priorité aux constantes 4p/multi-source). Liste figée avant lancement, bornes hard-coded (e.g. send_ratio ∈ [0.4, 0.99]).
- **Fitness** : 200-game eval contre un panel pondéré : 25% V7 self, 25% V12, 25% `notebook_distance_prioritized`, 25% `notebook_pascalledesma_orbitwork_v14`. Modes mélangés 2p/4p 50-50.
- **Algorithme** : CMA-ES (pycma), population=24, sigma initial=0.15·range_per_param. Stop si 5 gen consécutives sans amélioration > 0.5% wr.
- **Compute** : ~300 gen × 24 individus × 200 games = 1.4M games. À ~0.5s/game c_engine → ~190 CPU-heures. Parallélisable sur 8 workers → ~24h wall-clock par run.
- **Promotion** : seul un config dont winrate **Wilson 95% LB ≥ baseline +3%** (n=500 sur eval gel) est promu en `v15_config_cma_v{N}.json`.

**Pourquoi CMA-ES > PPO ici** :
- Action space stationnaire (scalaires), pas state-dependent.
- Pas de credit-assignment problem.
- Dense fitness sur eval-set, pas reward terminale sparse.
- Empiriquement, c'est ce que les top heuristiques compétitives utilisent (review §5).

### Layer 3 — BC re-ranker (timeboxed research, 1 semaine)
Conservé mais **rétrogradé** à branche recherche. Si à fin S3 il ne bat pas CMA-ES (Layer 2b), il est **archivé** définitivement (review §8 reco 2).

#### Pré-requis avant tout training (review §6 reco 3)
1. **Audit du dataset existant `replay_corpus/imitation_4p_top10_v1/`** :
   - Investigation des **4978 unmatched actions** (~10% des samples). Stratifier par mission (expand/attack/support) et nb de sources. Si surreprésentation des multi-source → matcher biaisé contre les patterns D1. Fix avant entraînement.
   - Investigation des **340 too-many-turns**. Probablement parties très longues, peu de risque mais à confirmer.
2. **Ceiling estimation** : sur 5k turns hold-out, calculer P(action_replay ∈ V7.top_K(state)) pour K∈{4,8,16}. Si ceiling@K=8 < 30%, la cible BC top-1=25% est infaisable — il faut **augmenter K** ou abandonner Layer 3.

#### Définition métrique BC top-1 (review §3.2 fix)
**Numérateur** : nb de turns hold-out où (a) l'action replay ∈ V7.top_K=8(state) ET (b) le ranker la classe #1 parmi ces 8.
**Dénominateur** : nb de turns hold-out où l'action replay ∈ V7.top_K=8(state) (= le ceiling).
**Random baseline conditionnel** : 1/8 = 12.5%.
**Target révisé** : **top-1 ≥ 50% du ceiling** (e.g. si ceiling=40%, target=20% absolu). C'est calibré et falsifiable.

#### Architecture & training
- MLP 50 → 64 → 32 → 1, scores K candidats indépendamment puis softmax.
- BC loss : cross-entropy sur l'index du candidat replay parmi les K.
- 100 epochs, AdamW lr=3e-4, eval split 10%.
- **Sécurité top-K** : ranker ne peut sélectionner que parmi top-K=8 V7. Si Layer 3 off, score = V7 heuristique brut.

#### Pas de PPO en v2 du plan
La proposition v1 (PPO sur le re-ranker) est **supprimée** (review §3.1, §5 first bullet). Justification :
- Action space non-stationnaire (K candidats state-dependent).
- Credit assignment cassé (reward ±1 terminale + +0.001·planet_share_delta trop bruité pour 5k params).
- Plafond théorique = convex hull des candidats V7 (ne peut pas dépasser l'horizon V7).
- → Si on veut RL plus tard, il faudra un agent qui choisit aussi parmi des candidats *non-V7* (out of scope V15).

### Layer 4 — Lookahead search (opt-in, gated)
**M3a (pré-requis)** : profiler `WorldModel.simulate(8_turns)` sur un état représentatif. Cible : **≤ 80 ms / call** (review §3.4 fix). Si dépassement :
- Option A : écrire `v15_fast_sim.py` (sim simplifié sans intercept précis, sans comètes, etc.).
- Option B : abandonner rollout, garder seulement eval heuristique post-move (planet_share post-strike).
- Option C : remplacer expectimax par **MCTS avec V7 comme rollout policy** (review §7 alt-3).

**Si M3a passe** :
- Depth-1 expectimax sur top-K=5 candidats V7.
- **Modèle d'adversaire mixte** (restauré du brief original, perdu en v1) : 50% greedy V12, 30% V14, 20% top10_mimic config. Pas V7 self (review §5 second bullet — sinon hallucination self-play).
- Eval = α·V7_score + β·planet_share_post + γ·prod_share_post. Coeffs tunés par CMA-ES sur 100 games.
- Budget runtime : 800 ms max, abort hard à 1.0 s avec fallback Layer 3 ou 2b.

**Promotion** : ≥+5% wr (Wilson 95% LB) vs Layer 2b sur 500 games. Sinon Layer 4 désactivé.

## 3. Pipeline d'entraînement

### Stage A — BC ranker (week 2, conditional on §2-Layer-3 pré-requis)
Voir Layer 3 ci-dessus. Critères dépendent du ceiling estimé.

### Stage B — RETIRÉ
La proposition v1 (PPO curriculum) est supprimée. Remplacée par CMA-ES (Layer 2b) qui est lancé en parallèle de Stage A dès S2.

### Stage C — CMA-ES curriculum (week 3–5)
Plusieurs runs CMA-ES contre des panels d'adversaires de difficulté croissante. Chaque run produit un `V15Config` candidat ; on garde le meilleur globalement.

| run | adversaires panel | n_games eval | gate Wilson 95% LB |
|---|---|---|---|
| C1 | random, crazy, nearest_planet (50% chaque) | 200 | ≥ 80% |
| C2 | distance_prioritized + orbitbotnext (50/50) | 200 | ≥ 60% |
| C3 | sigmaborov + kashiwaba (50/50) | 200 | ≥ 50% |
| C4 | pascalledesma_v14 + kronos_omega (50/50) | 500 | ≥ 40% |
| C5 | mix complet (panel 4 ci-dessus, pondéré par Kaggle frequency si dispo) | 500 | ELO interne ≥ V7+200 |

**Sample sizes** : 200 pour gates ≥50% (CI ±7%), 500 pour gates ≥40% (CI ±4.4%) (review §3.5 + §8 reco 5).

### Stage D — Opening book (week 5, optionnel)
Si `analysis/v15_replay_audit.json` montre que les états des tours 1–25 du top-10 sont concentrés (faible entropie), construire un opening book :
- k-NN sur features [planets count by owner, prod_share, my_ships_total, step] des replays top.
- Pour les 25 premiers turns, action = action moyennée des K=5 voisins les plus proches (ou hard match si exact).
- Switch à V7+CMA-config dès t≥25.
- Gate : ≥+2% wr (Wilson 95% LB, n=500) vs sans opening book.

## 4. Métriques de succès — révisées avec CI

Tous les gates Wilson 95% lower bound, sample sizes spécifiés.

| milestone | semaine | critère | n | action si raté |
|---|---|---|---|---|
| **M0** | J+1 | V15(flags=off) ≡ V7 sur 10 full-game traces seedés | 10 | corriger état caché |
| **M1** | S1 | Bisect Layer 2a : ≥3 patches passent individuellement (LB ≥ 0%) | 200/patch | si 0 passe → patterns D1 = confound, archive Layer 2a |
| **M2a** | S2 | Audit dataset BC : ceiling@K=8 calculé, 4978 unmatched stratifiés | — | si ceiling<30% → abandon Layer 3 |
| **M2b** | S2 | BC ranker top-1 ≥ 50% × ceiling (LB) | 1000 holdout | data quality / matcher |
| **M3a** | S3 | WorldModel 8-turn rollout ≤ 80 ms (médiane sur 100 states) | 100 | build fast_sim ou pivot Layer 4 → MCTS / pure heuristic eval |
| **M3** | S3 | Layer 1+2a+2b bat V7 baseline LB ≥ +5% | 500 | post-mortem CMA-ES, revoir bornes |
| **M3-stretch** | S3 | Layer +3 bat sans-Layer-3 LB ≥ +3% | 500 | sinon archive Layer 3 |
| **M4** | S4 | Kaggle ELO ≥ 900 (V7 baseline +200) | — | review benchmark biais |
| **M5** | S5 | Stage C4 winrate LB ≥ 40% | 500 | rollback C4 |
| **M6** | S6 | Kaggle ELO ≥ 1200 (stretch ≥1400) | — | post-mortem |

**Git tagging mandatory** (review §8 reco 10) : chaque milestone passé → tag `v15-m{N}-passed` avec hash du config et logs benchmark associés.

## 5. Risques et garde-fous (révisés)

1. **Régression silencieuse** : M0 full-game determinism + per-patch bisect M1 + Wilson CIs partout. C'était la faille de V8–V14.
2. **Patterns D1 = confounds** : si M1 montre 0 patches passe, on accepte que D1 est correlationnel pur. Layer 2a archivé, on parie tout sur Layer 2b (CMA-ES) qui est mesuré pas inféré.
3. **Latence Kaggle** : M3a comme gate dur. Budget runtime monitoré (`time.monotonic()` en début de turn, abort 800ms).
4. **BC dataset biaisé** : audit M2a obligatoire. Si 4978 unmatched sont systématiquement multi-source → matcher cassé → Layer 3 entraîne le mauvais signal.
5. **CMA-ES overfit panel** : Stage C utilise 4 panels disjoints. Eval finale sur panel C5 (mix) jamais utilisé en training. Si gap C4→C5 > 10%, overfit confirmé → re-run avec panel mix dès C1.
6. **Notebook zoo ≠ Kaggle** : tous les benchmarks locaux sont sur le zoo, qui est plus fort que la médiane Kaggle. Calibration : V7 zoo=53% ↔ V7 Kaggle ≈ 700 ELO. Si V15 zoo=65%, on s'attend Kaggle ≈ 900–950, pas 1400.

## 6. Plan de livraison (semaines 1–6)

- **S1** : M0 (Layer 1 refactor + determinism test). M1 (Layer 2a bisect). Tag v15-m0, v15-m1. Première submission Kaggle (V15 = V7 nu, sanity check ELO).
- **S2** : Layer 2b lancé en background (24h CMA-ES run C1). M2a (audit dataset). Si ceiling ok, lancer Layer 3 BC training.
- **S3** : M3a (latency profile). M3 (combine Layer 2a+2b vs V7). Submission #2.
- **S4** : Stage C2 + C3 CMA-ES. Submission #3.
- **S5** : Stage C4. Opening book si signal. Submission #4.
- **S6** : Gel meilleur config. Variantes anti-meta. Post-mortem.

## 7. Fichiers à créer

```
bot_v15.py                     # thin wrapper, ~250 lines, lit V15Config + flags
bot_v15_core.py                # V7 logic factored, ~1500 lines
v15_config.py                  # dataclass V15Config, JSON-loadable
v15_ranker.py                  # MLP wrapper (numpy inference)
v15_search.py                  # MCTS ou expectimax selon M3a
v15_fast_sim.py                # construit si M3a échoue avec WorldModel
v15_opening_book.py            # k-NN lookup, week 5

neural_network_gpu/scripts/audit_imitation_dataset.py    # M2a
neural_network_gpu/scripts/train_v15_bc.py
neural_network_gpu/scripts/cma_es_tune_v15.py            # Layer 2b track principal

benchmark_v15.py               # wrapper avec --layers flag
gate_v15.py                    # Wilson CI, n_games configurable, refuse promotion si LB < gate

tests/test_v15_regression.py   # M0 : full-game traces
tests/test_v15_patches.py      # M1 : per-patch bisect

analysis/V15_ARCHITECTURE.md   # ce doc
analysis/V15_REVIEW.md         # peer review
analysis/V15_BC_DATASET_AUDIT.md  # M2a output
analysis/V15_LATENCY_PROFILE.md   # M3a output
```

## 8. Hypothèses critiques (révisées)

| # | hypothèse | force du support | falsifiable par |
|---|---|---|---|
| H1 | V7 plafonne à cause de constantes sous-optimales, pas de logique limitée | **Affaiblie** : V7 déjà ES-tuné sur 14 clés (M0 finding). Reste valable seulement pour les ~20 constantes non-tunées. | M3 : CMA-ES sur set étendu doit prouver ≥+5% wr |
| H2 | BC re-ranker peut atteindre 50% du ceiling et apporter ≥+3% wr | **Faible** (review : ne peut que réordonner top-K V7) | M2b et M3-stretch |
| H3 | Layer 4 dans budget Kaggle (≤1s) suffit pour +5% wr | Faible (M3a non testé) | M3a + bench Layer 4 |
| H4 | Stage C4 ≥40% wr atteignable | Modéré (CMA-ES sur eval directe = bien posé, mais panel C4 = imitations top) | M5 |
| H5 | Patterns D1 sont causaux pas correlationnels | **Hypothèse rabaissée à soft** (review §3.3) | M1 bisect (chaque patch test indépendant) |

## 9. Changes vs v1 — résumé pour audit

| changement | motivation review |
|---|---|
| PPO retiré, CMA-ES devient track principal | §3.1 (ill-posed PPO), §5 (convex hull), §7 alt-1 |
| BC métrique redéfinie (% du ceiling, pas absolu) | §3.2 (incomparable denominators) |
| Per-patch bisect M1 obligatoire | §3.3, §8 reco 4 |
| Tous les gates en Wilson 95% LB + n spécifié | §3.5, §8 reco 5 |
| M3a (latency profile) ajouté | §3.4, §8 reco 6 |
| Modèle d'adversaire mixte restauré Layer 4 | §5 second bullet, §8 reco 7 |
| M0 = full-game determinism (pas juste 1000 obs) | §4.4, §8 reco 1 |
| Audit dataset (4978 unmatched) requis avant BC | §6 third bullet, §8 reco 3 |
| artOfWar Rule 1 explicitement 2p-only | §8 reco 9 |
| Git tag obligatoire par milestone | §8 reco 10 |
| D1 patterns rétrogradés "soft priors" | §3.3, §4 second bullet |
| Sample sizes ≥200 / ≥500 selon threshold | §3.5 |
