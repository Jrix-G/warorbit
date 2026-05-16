# These d'amelioration du reseau WarOrbit

Date: 2026-05-16

## Objectif

L'objectif n'est pas seulement de rendre le bot "plus actif". L'objectif est de lui faire apprendre des actions causalement utiles: tirs qui touchent, captures rentables, defense reussie, conservation de production, et victoire robuste contre random, greedy, starter, distance puis contre une ligue plus forte.

Le probleme observe - envoi de vaisseaux dans le vide, spam de ships, absence de strategie long terme - vient de plusieurs erreurs structurelles. Les hyperparametres seuls ne peuvent pas le corriger.

## Diagnostic principal

### 1. Le signal de victoire etait trop dilue

Dans `src/gpu_trainer.py`, le reward final etait divise sur toutes les decisions de la trajectoire. Sur une partie longue, une victoire valait typiquement autour de `+0.01` par decision avant shaping. Le shaping d'activite devenait alors plus fort que le vrai objectif.

Consequence: le reseau apprend le comportement qui maximise le reward dense local, pas la strategie qui gagne.

Correction appliquee: la cible PPO est maintenant:

```text
target_t = episode_outcome + discounted_future_step_shaping_t
advantage_t = target_t - value(s_t)
```

Le terminal n'est plus divise par la longueur de partie. Le shaping reste local, mais il ne peut plus noyer le win/loss.

### 2. Le bot etait pousse a envoyer des ships sans preuve d'utilite

Plusieurs rewards donnaient un bonus a:

- faire une action reelle;
- envoyer plus de ships;
- baisser le no-op rate;
- remplir plusieurs action slots.

Ces rewards ne verifiaient pas si la flotte touchait une planete, capturait, defendait, ou ameliorait le score.

Conclusion: le comportement de spam etait une solution rationnelle pour le reseau.

Correction appliquee: le shaping est maintenant traite comme un terme local secondaire, le retour terminal garde son echelle, et le moteur rapide remonte les evenements de flotte. Les tirs qui touchent/capturent/supportent portent le bonus principal; les tirs perdus hors carte/soleil sont penalises. Les bonus plats "real action" et "ship volume" sont maintenant plafonnes quand le shaping causal est actif.

### 3. Les tirs etaient predits avec une physique incomplete

Le moteur officiel encode une flotte comme:

```text
[id, owner, x, y, angle, source_planet_id, ships]
```

L'adapter lisait auparavant ces champs comme:

```text
ships=f[4], target_id=f[5], eta=f[6]
```

Donc le modele voyait:

- ships = angle;
- target = planete source;
- eta = nombre de ships.

Consequence: features de flottes entrantes, masse en vol, menaces et total ships etaient corrompus.

Correction appliquee dans `orbit_wars_adapter.py`: decodage officiel correct, `angle`, `source_id`, `ships`, target inconnu marque `-1`.

### 4. La trajectoire depend du nombre de ships

Le moteur officiel utilise une vitesse dependante du nombre de vaisseaux:

```text
speed = 1 + (max_speed - 1) * (log(ships) / log(1000))^1.5
speed <= max_speed
```

Un petit tir est beaucoup plus lent qu'un gros tir. Si le planner predit avec une vitesse constante, il vise une planete orbitale au mauvais moment.

Correction appliquee:

- `safe_plan_shot(..., ships=amount)` utilise la vitesse officielle;
- `policy.py` valide chaque candidat avec son vrai `amount`;
- `_candidate_move` passe aussi le nombre de ships au planner.

### 5. Les owner slots pouvaient changer de sens

`player_ids` etait construit depuis les owners visibles. Si un joueur perd temporairement toutes ses planetes visibles, les indices one-hot peuvent changer.

Correction appliquee: slots stables `[0, 1, 2, 3]` dans l'adapter.

### 6. L'evaluation etait partiellement fausse

Dans `scripts/run_gpu.py`, plusieurs metriques de ships/slots/missions etaient prises depuis le dernier match de l'adversaire, pas agregees sur tous les matchs.

Correction appliquee: `_evaluate` agrege maintenant les metriques par match.

### 7. La promotion sur graines fixes favorisait l'overfit

Candidate et best doivent etre compares sur les memes seeds pour une evaluation pairee. Mais reutiliser exactement la meme fenetre de seeds pendant tout le run peut masquer des regressions.

Correction appliquee:

- candidate et best utilisent toujours les memes seeds dans une evaluation donnee;
- la fenetre de seeds tourne a chaque evaluation via `--eval-seed-stride`.

### 8. L'imitation learning n'entrainait pas le bon comportement multi-action

L'extraction replay gardait seulement le plus gros mouvement d'un tour expert. Or notre agent peut emettre plusieurs actions par tour.

Correction appliquee:

- extraction de tous les mouvements valides du tour expert, dans l'ordre;
- reservation des ships entre deux mouvements;
- metadata `action_slot`;
- config d'encodage alignee sur `planet_id_scale=64`.

### 9. Validation BC fuiteuse

Le training BC validait sur des shards ensuite utilises en training.

Correction appliquee: split train/validation par shards dans `scripts/train_imitation_4p.py`.

## Changements codes appliques

Fichiers modifies:

- `src/gpu_trainer.py`: retours non dilues, advantages value-based, logs return/value/advantage.
- `src/vec_worker.py`: attribution des evenements officiels `launch/hit/lost/combat` aux decisions du reseau.
- `src/action_metrics.py`: metriques `fleet_hit_rate`, `fleet_capture_rate`, `fleet_lost_rate`, `fleet_pending_rate`.
- `scripts/run_gpu.py`: nouveaux args `--train-return-gamma`, `--train-return-clip`, `--eval-seed-stride` et `--event-*`; eval aggregation corrigee; seed rotation; queue model-update remplacee par latest update; eval hit/capture/lost.
- `scripts/eval_checkpoints.py`: metriques d'activite compatibles avec les vrais `action_records`, plus metriques de flotte.
- `scripts/extract_4p_imitation_dataset.py`: multi-action extraction, candidate config explicite, `planet_id_scale=64`, `safe_plan_shot(... ships=...)`.
- `scripts/train_imitation_4p.py`: split train/val par shards.
- `../local_simulator/orbit_wars_official.py`: journal causal des flottes dans `env.info["fleet_events"]`.
- `../local_simulator/official_fast.py`: chargement prioritaire du moteur local instrumente, score de flotte corrige (`fleet[6]` au lieu de `fleet[4]`), copie profonde des etats initial/final.
- `../neural_network/src/notebook_4p_training.py`: rattachement des evenements aux actions et reward causal leger pour les runs notebook.
- `../neural_network/src/orbit_wars_adapter.py`: decodage officiel des flottes et owner slots stables.
- `../neural_network/src/policy.py`: validation trajectoire par amount.
- `kaggle_submission_stage/neural_network/src/*`: memes corrections critiques pour packaging/soumission.

## Ce que ces corrections changent concretement

Avant:

```text
Un tir inutile pouvait etre renforce car:
- il etait "non-noop";
- il envoyait des ships;
- il remplissait un slot;
- le terminal etait trop dilue pour contredire vite ce comportement.
```

Apres:

```text
Une decision porte un retour proche du resultat de l'episode.
Le shaping local vient maintenant surtout des consequences observees:
- hit/capture/support -> positif;
- hors carte/soleil -> negatif;
- action plate/volume -> plafonne a un niveau secondaire.
La trajectoire d'un candidat depend de son vrai nombre de ships.
Les flottes observees ne corrompent plus l'etat.
```

## Protocole de run recommande

### Phase 0 - Rebuild imitation dataset

Relancer l'extraction avec les nouveaux defaults:

```powershell
.\run_extract_4p_imitation.ps1
```

Verifier dans le report:

- `missions` contient attack/expand/support;
- `samples` augmente par rapport a l'ancien dataset si les replays ont plusieurs moves;
- `skipped.unmatched_action` ne monte pas excessivement.

### Phase 1 - Behavior cloning propre

```powershell
.\run_train_4p_imitation.ps1
```

Critere minimal:

- top-1 BC seul peut rester faible avec 2048 candidats, mais top-3/top-k doit progresser;
- validation doit etre sur `val_shards`, pas les shards de train.

Si top-1 reste vers 2-3%, il faut reduire l'espace candidat ou ajouter une loss par mission/source/target avant le cross-entropy final.

### Phase 2 - Fine-tuning PPO court

Lancer un run 60-120 min avec:

```text
--train-return-gamma 0.997
--train-return-clip 2.0
--event-capture-bonus 0.10
--event-enemy-hit-bonus 0.045
--event-lost-penalty 0.045
--per-step-ship-volume-bonus 0.0
--per-step-real-action-bonus <= 0.003
--per-step-legal-noop-penalty <= 0.006
--teacher-kl-coef 0.05 a 0.10
--eval-episodes >= 64 si possible
```

Ne pas chercher a forcer `avg_ships_sent` en premier. La bonne cible est:

```text
valid_winrate up
winrate vs greedy/random up
legal_noop stable
fleet_hit_rate up
fleet_lost_rate down
spam de ships sans hit down
```

### Phase 3 - Run long

Un run long n'a du sens que si les checks suivants sont bons sur 2 evaluations consecutives:

- `valid_winrate` non nul vs random et greedy;
- `eval_avg_ships_sent` pas artificiellement eleve;
- `return_target_std` non nul;
- `value_loss` pas bloque a zero;
- `raw_advantage_std` non nul;
- `teacher_kl` pas explosif si teacher actif;
- `fleet_hit_rate` non nul et en hausse;
- `fleet_lost_rate` pas en hausse continue;
- pas de promotion sur une seule faiblesse statistique.

## Garde-fous de run

Stopper ou rollback si:

- winrate monte mais `valid_winrate` reste plat;
- `eval_avg_ships_sent` monte fortement sans gain vs greedy;
- `legal_noop` baisse mais `starter/distance` s'effondrent;
- `return_target_mean` devient bon alors que terminal reste mauvais;
- `teacher_kl` monte continuellement: la policy oublie le BC;
- `ships_sent_max` explose: le bot sur-engage.

## Feuille de route scientifique

### Priorite A - Instrumentation officielle des evenements: appliquee

Le journal causal par flotte existe maintenant:

- launched;
- hit_planet via `hit`;
- capture/defense via `combat`;
- expired_out_of_bounds via `lost_oob`;
- crossed_sun via `lost_sun`;
- no_hit_before_end via `pending` cote worker;
- owner/source/target/ships/turn.

Le reward applique doit rester dans cette logique:

```text
+ capture neutral/enemy rentable
+ defense own planet
+ production gained
+ score delta
- miss/sun/out-of-bounds
- overcommit qui perd la source
```

Le reste du shaping d'activite doit rester plafonne: il peut aider l'exploration, mais ne doit plus redevenir l'objectif principal.

### Priorite B - Candidats avec prediction de hit

Chaque `ActionCandidate` devrait contenir:

- predicted_hit_target_id;
- predicted_turns_to_hit;
- predicted_hit_confidence;
- predicted_survivor_ships;
- source_remaining_ratio;
- target_owner/production/ships au hit estime.

Le reseau doit apprendre depuis des features de consequence, pas seulement source/target actuels.

### Priorite C - BC hierarchique

Avec beaucoup de candidats, une cross-entropy brute top-1 est dure. Meilleure decomposition:

1. head mission: noop/expand/attack/support;
2. head source;
3. head target;
4. head amount/ratio;
5. reranker candidat final.

Cela reduit la variance et donne des gradients utiles meme si le candidat exact n'est pas top-1.

### Priorite D - PPO/GAE complet

La correction actuelle est une base beaucoup plus saine que l'ancien reward par-step. La version suivante doit passer a GAE:

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = discounted_sum(lambda, delta_t)
target_t = A_t + V(s_t)
```

Cela exige des transitions plus completes, donc un worker qui stocke aussi l'etat suivant ou une valeur bootstrap par tour.

### Priorite E - Evaluation robuste

Evaluation finale recommandee:

- eval pairee candidate/best sur seeds rotatives;
- heldout seeds jamais utilises pour promotion rapide;
- rapport par adversaire;
- minimum 128 games pour promotion majeure;
- Wilson CI utilise comme gate;
- regression max par adversaire stricte.

## Conclusion

Le reseau n'etait pas fondamentalement incapable. Il optimisait un objectif mal pose:

```text
activite + volume + no-op bas
```

au lieu de:

```text
actions qui touchent + captures utiles + defense + victoire robuste
```

Les corrections appliquees remettent le terminal au centre, corrigent la physique de base, fiabilisent l'evaluation, rendent le BC coherent avec les actions multi-slots, et remplacent le coeur du shaping d'activite par des evenements causaux de flotte. La prochaine etape critique est d'ajouter des features predites de consequence directement dans les candidats pour que le reseau voie avant d'agir ce qu'un tir a des chances de produire.
