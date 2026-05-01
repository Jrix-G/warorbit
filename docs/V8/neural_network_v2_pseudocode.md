# Neural Network V2 pour Orbit Wars

## Audit court

- Fichier neural network trouvé : `docs/analysis/neural_proposal.txt`
- Idées principales du fichier :
  - encoder les planètes et les flottes avec des features locales ;
  - utiliser un encoder partagé puis une tête de type pointer pour choisir source, cible et type de mission ;
  - ajouter une value head ;
  - entraîner d’abord en behavioral cloning, puis en self-play / PPO / league training ;
  - exporter les poids en format numpy pour l’eval Kaggle.
- Idées conservées :
  - séparation nette entre représentation de l’état, policy, value et entraînement ;
  - attention à la taille du modèle et au budget d’inférence ;
  - self-play avec pool d’adversaires ;
  - sauvegarde de checkpoints et export vers une implémentation légère.
- Idées rejetées :
  - dépendance à un bot existant ou à sa grammaire d’actions ;
  - hypothèse que le bon espace d’action est déjà connu ;
  - mélange trop direct entre heuristiques codées et réseau ;
  - architecture trop dimensionnée sans plan de validité par rapport au jeu.
- Raisons :
  - le réseau doit être défini à partir du jeu, pas à partir du bot précédent ;
  - l’ancien plan mélangeait trop de choix de mise en oeuvre et de stratégie ;
  - il faut une formulation exploitable sur des entraînements longs et stables.

## 1. Objectif

Définir un système de neural network adapté à Orbit Wars, capable de :

- comprendre un état de jeu variable ;
- scorer plusieurs actions candidates par planète source ;
- apprendre par self-play sur des parties longues ;
- améliorer progressivement le winrate contre une population d’adversaires ;
- rester compatible avec une exécution rapide en production.

Ce document propose une implémentation conceptuelle complète en pseudo-code, sans dépendre du bot actuel.

## 2. Hypothèses sur le jeu

- Le jeu est simultané, à information complète.
- Les états contiennent au minimum :
  - les planètes ;
  - les flottes/vaisseaux en transit ;
  - les joueurs encore actifs ;
  - le tour courant ;
  - éventuellement des objets spéciaux du scénario si présents.
- Les actions utiles sont de la forme :
  - choisir une planète source ;
  - choisir une cible ;
  - choisir un type d’intention ;
  - choisir une quantité de vaisseaux ;
  - éventuellement choisir une priorité de planification.
- Le mode 4 joueurs change fortement la dynamique :
  - plus d’opportunités d’interactions entre adversaires ;
  - plus de fronts ;
  - plus de captures opportunistes ;
  - plus de risque d’être attaqué par un troisième joueur pendant une offensive.
- L’objectif du modèle n’est pas seulement de maximiser le nombre de planètes, mais la capacité future à produire, survivre et convertir un avantage en victoire.

## 3. Représentation de l’état

Principe :

- représenter le jeu comme un ensemble de planètes, de flottes et de joueurs ;
- construire un état global + des objets locaux ;
- conserver des dimensions fixes via padding / masking ;
- normaliser toutes les valeurs numériques.

### Schéma logique

```text
state = {
  global: features_globales,
  planets: [features_planet_0, ..., features_planet_N-1],
  fleets:  [features_fleet_0, ..., features_fleet_M-1],
  players: [features_player_0, ..., features_player_P-1],
  masks:   { planet_mask, fleet_mask, player_mask, action_mask }
}
```

### Règles de représentation

- ordonner les planètes par `id` ou par un ordre stable déterministe ;
- ordonner les flottes par `eta`, puis `owner`, puis `target_id` ;
- encoder chaque objet avec :
  - des features physiques ;
  - des features relationnelles ;
  - des features temporelles ;
  - des features de rôle ;
  - des indicateurs de validité.
- utiliser des normalisations stables :
  - `x / board_scale` ;
  - `ships / ship_scale` ;
  - `prod / prod_scale` ;
  - `eta / horizon_scale` ;
  - `turn / max_turns`.

### Pseudo-code

```python
def encode_state(obs, config):
    game = build_game_state(obs, config)

    global_feat = build_global_features(game)
    planet_feats, planet_mask = build_planet_tensor(game)
    fleet_feats, fleet_mask = build_fleet_tensor(game)
    player_feats, player_mask = build_player_tensor(game)

    return {
        "global": global_feat,
        "planets": planet_feats,
        "fleets": fleet_feats,
        "players": player_feats,
        "masks": {
            "planet": planet_mask,
            "fleet": fleet_mask,
            "player": player_mask,
        },
        "game": game,
    }
```

## 4. Features des planètes

Chaque planète doit être décrite comme une unité stratégique. Les features doivent permettre au réseau de comprendre :

- sa valeur économique ;
- son niveau de sécurité ;
- sa position dans le graphe spatial ;
- sa contestation ;
- son rôle potentiel dans les plans offensifs ou défensifs.

### Features proposées

```text
planet_features =
[
  owner_one_hot(P_MAX + 2),
  is_mine,
  is_enemy,
  is_neutral,
  x_norm,
  y_norm,
  radius_norm,
  production_norm,
  ships_norm,
  ships_log_norm,
  dist_to_center_norm,
  dist_to_my_center_norm,
  dist_to_nearest_enemy_norm,
  dist_to_nearest_ally_norm,
  incoming_my_ships_norm,
  incoming_enemy_ships_norm,
  outgoing_my_ships_norm,
  projected_owner_id_embedding_optional,
  projected_net_ships_next_horizon,
  projected_production_next_horizon,
  threat_score_norm,
  value_score_norm,
  is_contested,
  is_frontier,
  is_deep_backline,
  is_static_special,
  special_life_norm,
]
```

### Remarques

- `projected_net_ships_next_horizon` doit être calculé à plusieurs horizons courts, par exemple `h = 1, 3, 5, 10`.
- `threat_score_norm` doit intégrer la pression des flottes ennemies et la capacité de renfort locale.
- `value_score_norm` doit estimer l’intérêt économique futur d’une planète, pas seulement son état courant.
- en mode 4 joueurs, ajouter une notion de multi-contestation :
  - nombre d’ennemis pouvant atteindre la planète dans une fenêtre donnée ;
  - nombre de flottes ennemies en conflit entre elles autour de cette planète.

### Pseudo-code

```python
def build_planet_tensor(game):
    feats = []
    mask = []

    for planet in game.planets_sorted:
        f = []
        f += encode_owner_one_hot(planet.owner, game.player_ids)
        f += [
            planet.owner == game.my_id,
            planet.owner not in (-1, game.my_id),
            planet.owner == -1,
            norm(planet.x, game.board_scale),
            norm(planet.y, game.board_scale),
            norm(planet.radius, game.radius_scale),
            norm(planet.production, game.production_scale),
            norm(planet.ships, game.ship_scale),
            log_norm(planet.ships, game.ship_scale),
            norm(distance_to_center(planet, game), game.board_scale),
            norm(distance_to_my_center(planet, game), game.board_scale),
            norm(nearest_enemy_distance(planet, game), game.board_scale),
            norm(nearest_ally_distance(planet, game), game.board_scale),
            norm(incoming_my_ships(planet, game), game.ship_scale),
            norm(incoming_enemy_ships(planet, game), game.ship_scale),
            norm(outgoing_my_ships(planet, game), game.ship_scale),
            projected_owner_embedding(planet, game),
            projected_net_ships(planet, game, horizon=5),
            projected_production(planet, game, horizon=10),
            norm(threat_score(planet, game), game.threat_scale),
            norm(value_score(planet, game), game.value_scale),
            is_contested(planet, game),
            is_frontier(planet, game),
            is_backline(planet, game),
            is_special(planet, game),
            norm(special_life(planet, game), game.special_life_scale),
        ]
        feats.append(f)
        mask.append(1)

    pad_to_max_len(feats, mask, MAX_PLANETS, feature_dim=PLANET_DIM)
    return as_tensor(feats), as_tensor(mask)
```

## 5. Features des flottes

Les flottes doivent permettre de comprendre la dynamique du champ de bataille :

- qui attaque qui ;
- quand les flottes arrivent ;
- quels combats vont se produire ;
- quels renforts existent déjà ;
- quelles opportunités ou menaces ne sont pas encore résolues.

### Features proposées

```text
fleet_features =
[
  owner_one_hot(P_MAX + 2),
  is_mine,
  is_enemy,
  is_neutral,
  x_norm,
  y_norm,
  source_planet_id_norm,
  target_planet_id_norm,
  ships_norm,
  ships_log_norm,
  eta_norm,
  remaining_travel_norm,
  is_attack,
  is_reinforcement,
  is_transfer,
  projected_arrival_state,
  predicted_combat_margin,
  target_owner_at_launch_embedding_optional,
]
```

### Remarques

- si l’information de source / cible n’est pas directement donnée, elle doit être reconstruite par le moteur du jeu ou par un planificateur de trajectoires ;
- une flotte doit être encodée différemment si elle soutient une planète amie ou attaque une planète ennemie ;
- les flottes dans le futur proche sont plus importantes que celles très lointaines, mais les deux doivent rester visibles.

### Pseudo-code

```python
def build_fleet_tensor(game):
    feats = []
    mask = []

    for fleet in game.fleets_sorted:
        f = []
        f += encode_owner_one_hot(fleet.owner, game.player_ids)
        f += [
            fleet.owner == game.my_id,
            fleet.owner not in (-1, game.my_id),
            fleet.owner == -1,
            norm(fleet.x, game.board_scale),
            norm(fleet.y, game.board_scale),
            norm(fleet.source_id, game.planet_id_scale),
            norm(fleet.target_id, game.planet_id_scale),
            norm(fleet.ships, game.ship_scale),
            log_norm(fleet.ships, game.ship_scale),
            norm(fleet.eta, game.horizon_scale),
            norm(fleet.remaining_travel, game.horizon_scale),
            is_attack_fleet(fleet, game),
            is_reinforcement_fleet(fleet, game),
            is_transfer_fleet(fleet, game),
            projected_arrival_state(fleet, game),
            predicted_combat_margin(fleet, game),
            target_owner_embedding_at_launch(fleet, game),
        ]
        feats.append(f)
        mask.append(1)

    pad_to_max_len(feats, mask, MAX_FLEETS, feature_dim=FLEET_DIM)
    return as_tensor(feats), as_tensor(mask)
```

## 6. Features des joueurs

Le réseau doit raisonner sur chaque joueur actif comme sur une entité stratégique. Les features doivent capturer :

- force militaire ;
- production ;
- présence territoriale ;
- densité de front ;
- agressivité relative ;
- danger immédiat.

### Features proposées

```text
player_features =
[
  is_me,
  is_alive,
  planets_count_norm,
  ships_total_norm,
  production_total_norm,
  ships_on_planets_norm,
  ships_in_flight_norm,
  frontline_planets_norm,
  frontier_pressure_norm,
  active_front_count_norm,
  weakest_enemy_flag,
  strongest_enemy_flag,
  win_prob_proxy,
  elimination_risk_norm,
  expansion_capacity_norm,
]
```

### Pseudo-code

```python
def build_player_tensor(game):
    feats = []
    mask = []

    for pid in game.player_ids:
        p = player_summary(pid, game)
        f = [
            pid == game.my_id,
            p.alive,
            norm(p.planets_count, game.planet_count_scale),
            norm(p.ships_total, game.ship_scale),
            norm(p.production_total, game.production_scale),
            norm(p.ships_on_planets, game.ship_scale),
            norm(p.ships_in_flight, game.ship_scale),
            norm(p.frontline_planets, game.planet_count_scale),
            norm(p.frontier_pressure, game.pressure_scale),
            norm(p.active_fronts, game.front_count_scale),
            p.is_weakest_enemy,
            p.is_strongest_enemy,
            p.win_prob_proxy,
            norm(p.elimination_risk, game.risk_scale),
            norm(p.expansion_capacity, game.capacity_scale),
        ]
        feats.append(f)
        mask.append(1 if p.alive else 0)

    pad_to_max_len(feats, mask, MAX_PLAYERS, feature_dim=PLAYER_DIM)
    return as_tensor(feats), as_tensor(mask)
```

## 7. Architecture proposée du réseau

Architecture recommandée :

- encodeurs séparés pour planètes, flottes et joueurs ;
- fusion par attention ou cross-attention ;
- pooling global ;
- tête policy sur actions candidates ;
- tête value pour estimer le résultat futur ;
- éventuellement une tête auxiliaire de prévision d’état.

### Structure

```text
inputs
  -> planet_encoder
  -> fleet_encoder
  -> player_encoder
  -> shared_context_block
  -> global_pool
  -> policy_heads
  -> value_head
```

### Proposition concrète

- `PlanetEncoder`: MLP + embedding owner + embedding positionnel ;
- `FleetEncoder`: MLP + embedding owner + embedding target relation ;
- `PlayerEncoder`: MLP simple ;
- `ContextBlock`: 1 à 3 couches d’attention légère ou de message passing ;
- `PolicyHead`: score chaque action candidate ;
- `ValueHead`: score unique du state.

### Pseudo-code

```python
def forward_network(encoded_state):
    z_planet = PlanetEncoder(encoded_state["planets"], encoded_state["masks"]["planet"])
    z_fleet = FleetEncoder(encoded_state["fleets"], encoded_state["masks"]["fleet"])
    z_player = PlayerEncoder(encoded_state["players"], encoded_state["masks"]["player"])
    z_global = GlobalEncoder(encoded_state["global"], z_planet, z_fleet, z_player)

    z_context = ContextBlock(z_planet, z_fleet, z_player, z_global)

    policy_logits = PolicyHead(z_context, z_global)
    value = ValueHead(z_context, z_global)
    aux = AuxiliaryHeads(z_context, z_global)

    return {
        "policy_logits": policy_logits,
        "value": value,
        "aux": aux,
    }
```

## 8. Sorties du réseau

Le réseau doit produire des sorties séparées, afin de ne pas tout écraser dans un seul logit final.

### Sorties principales

```text
policy logits pour :
  - source planet
  - target planet
  - mission type
  - send fraction / ship amount bucket
  - optional plan mode

value head :
  - probabilité de victoire / score futur

auxiliary heads :
  - next owner distribution for key planets
  - threat level prediction
  - future production delta
```

### Choix recommandé

- `source_logits`: une valeur par planète amie ;
- `target_logits`: une valeur par cible candidate ;
- `mission_logits`: une valeur par type de mission ;
- `amount_logits`: une valeur par bucket de volume envoyé ;
- `stop_logits`: décider de ne rien faire si aucune action n’est profitable.

### Pseudo-code

```python
def decode_outputs(outputs, action_candidates):
    masked = apply_action_mask(outputs["policy_logits"], action_candidates.mask)

    source_id = argmax(masked["source"])
    target_id = argmax(masked["target"])
    mission_id = argmax(masked["mission"])
    amount_id = argmax(masked["amount"])
    stop_flag = sigmoid(outputs["aux"]["stop"])

    return {
        "source_id": source_id,
        "target_id": target_id,
        "mission_id": mission_id,
        "amount_id": amount_id,
        "stop_flag": stop_flag,
    }
```

## 9. Conversion sortie → action

Le modèle ne doit pas envoyer directement une action brute sans vérification. Il faut un convertisseur qui :

- récupère une action candidate ;
- vérifie sa validité ;
- reconstruit le nombre de vaisseaux à envoyer ;
- applique des règles de sécurité minimales ;
- rejette ou remplace les actions invalides.

### Politique d’exécution

```text
network outputs
  -> candidate selection
  -> validity check
  -> amount reconstruction
  -> game action
```

### Pseudo-code

```python
def logits_to_action(encoded_state, outputs, game):
    candidates = build_action_candidates(game)
    scored = score_candidates(outputs, candidates)
    ranked = sort_by_score_desc(scored)

    for cand in ranked:
        if not is_valid_candidate(cand, game):
            continue

        send = reconstruct_send_amount(cand, game, outputs)
        send = clip_send_to_available(send, cand.source, game)

        if send <= 0:
            continue

        if violates_safety_floor(cand.source, send, game):
            continue

        return to_game_action(cand.source, cand.target, send, cand.mission)

    return NO_OP
```

### Gestion du mode 4 joueurs

En 4 joueurs, le convertisseur doit intégrer un choix de stratégie plus large :

- si plusieurs ennemis sont actifs, augmenter le poids de la défense ;
- si un joueur est nettement plus faible, autoriser des actions d’élimination ;
- si une planète neutre est contestée par plusieurs joueurs, laisser le modèle choisir entre prise opportuniste et renforcement ;
- si le front arrière est fragile, réduire les offensives longues.

```python
def adjust_policy_for_four_players(candidates, game):
    if game.num_players < 4:
        return candidates

    if game.active_enemy_fronts >= 2:
        candidates = boost_defense_candidates(candidates)

    if game.weakest_enemy_ratio < WEAK_ENEMY_THRESHOLD:
        candidates = add_elimination_candidates(candidates)

    if game.backline_risk_high:
        candidates = suppress_long_range_offense(candidates)

    return candidates
```

## 10. Boucle de self-play

Le self-play doit produire des données stables et diversifiées. Il ne doit pas tourner uniquement contre la dernière version du modèle.

### Principes

- garder une ligue d’adversaires ;
- jouer contre soi-même, contre des checkpoints précédents et contre quelques baselines fixes ;
- varier les seeds ;
- mélanger 2p, 3p et 4p si l’environnement le permet ;
- stocker les trajectoires complètes.

### Pseudo-code

```python
def run_self_play_round(model_pool, game_sampler, n_games):
    trajectories = []

    for g in range(n_games):
        game = game_sampler.sample()
        agents = sample_opponents(model_pool, game)

        episode = play_game(game, agents)
        trajectories.append(episode)

    return trajectories
```

### Forme d’un épisode

```text
episode = [
  (state_t, action_t, reward_t, next_state_t, done_t, metadata_t),
  ...
]
```

### Politique de sélection d’adversaires

```python
def sample_opponents(model_pool, game):
    pool = []
    pool += recent_checkpoints(model_pool, k=3)
    pool += frozen_checkpoints(model_pool, k=2)
    pool += fixed_baselines(game.mode)
    pool += one_self_play_clone(model_pool)
    return weighted_sample(pool, game.mode)
```

## 11. Fonction de récompense

La récompense doit encourager :

- la victoire ;
- la survie ;
- la croissance de production ;
- la prise de planètes utiles ;
- la réduction du risque ;
- la stabilité des positions ;
- la conversion d’avantage.

Elle doit éviter les récompenses trop myopes.

### Proposition

```text
reward_t =
  w_win * terminal_win
  + w_delta_prod * delta_production_share
  + w_delta_planets * delta_planets_share
  + w_delta_ships * delta_total_ships_share
  + w_survival * survival_bonus
  + w_threat * threat_reduction_bonus
  + w_capture * useful_capture_bonus
  + w_defense * defended_planet_bonus
  - w_loss * irreversible_loss_penalty
  - w_overcommit * overcommit_penalty
```

### Règles pratiques

- le score terminal doit dominer ;
- les récompenses intermédiaires doivent être bornées ;
- les prises de planètes inutiles doivent valoir moins qu’une prise utile ;
- une action qui aggrave fortement le front défensif doit être pénalisée ;
- en 4 joueurs, la récompense doit tenir compte du fait qu’une bonne action contre un ennemi peut aider un autre joueur.

### Pseudo-code

```python
def compute_reward(prev_state, action, next_state, terminal):
    reward = 0.0
    reward += W_PROD * delta_production_share(prev_state, next_state)
    reward += W_PLANETS * delta_planet_share(prev_state, next_state)
    reward += W_SHIPS * delta_ship_share(prev_state, next_state)
    reward += W_SAFETY * safety_improvement(prev_state, next_state)
    reward += W_CAPTURE * useful_capture(prev_state, action, next_state)
    reward -= W_OVERCOMMIT * overcommitment(prev_state, action, next_state)

    if terminal:
        reward += W_WIN if did_win(next_state) else -W_LOSS

    return clip(reward, -R_MAX, R_MAX)
```

## 12. Boucle d’entraînement

L’entraînement doit être progressif et stable :

- pré-entraînement sur données de replays ;
- self-play contrôlé ;
- entraînement mixte sur anciennes et nouvelles données ;
- validation régulière ;
- conservation des meilleurs checkpoints.

### Plan recommandé

1. Behavioral cloning sur replays.
2. Fine-tuning sur self-play.
3. League training avec population de modèles.
4. Réévaluation régulière contre baselines fixes.

### Pseudo-code

```python
def train_model(model, replay_dataset, model_pool, config):
    for epoch in range(config.bc_epochs):
        batch = sample_replay_batch(replay_dataset)
        loss = supervised_loss(model, batch)
        update_weights(model, loss)

    for iteration in range(config.self_play_iterations):
        episodes = run_self_play_round(model_pool, config.game_sampler, config.games_per_iter)
        batch = build_rl_batch(episodes)
        loss = rl_loss(model, batch)
        update_weights(model, loss)

        if iteration % config.eval_every == 0:
            metrics = evaluate_model(model, config.eval_suite)
            maybe_save_checkpoint(model, metrics, config.checkpoint_manager)
            maybe_promote_best(model, metrics, config.best_manager)

    return model
```

### Loss mixte

```text
loss =
  loss_policy
  + alpha * loss_value
  + beta * loss_aux
  + gamma * entropy_bonus
```

## 13. Sauvegarde des modèles

Il faut séparer :

- le checkpoint courant ;
- le meilleur checkpoint ;
- les snapshots de ligue ;
- les modèles exportés pour l’inférence.

### Règles

- sauvegarder les poids ;
- sauvegarder la version du schéma de features ;
- sauvegarder les statistiques de normalisation ;
- sauvegarder la date, la métrique principale et les métriques secondaires ;
- sauvegarder un hash du code de preprocessing.

### Pseudo-code

```python
def save_checkpoint(model, metadata, path):
    artifact = {
        "weights": model.state_dict(),
        "feature_schema_version": metadata.feature_schema_version,
        "normalization_stats": metadata.normalization_stats,
        "training_step": metadata.training_step,
        "metrics": metadata.metrics,
        "timestamp": now(),
    }
    write_file(path, serialize(artifact))


def save_best_model(model, metrics, best_state):
    if is_better(metrics, best_state.metrics):
        save_checkpoint(model, metrics, best_state.best_path)
        best_state.metrics = metrics
```

## 14. Benchmark

Le benchmark doit mesurer autre chose que le winrate brut.

### Mesures utiles

- winrate global ;
- winrate 2 joueurs ;
- winrate 4 joueurs ;
- temps moyen par décision ;
- temps p95 ;
- nombre moyen d’actions jouées ;
- taux d’actions invalides ;
- distribution des types d’actions ;
- robustesse par seed.

### Pseudo-code

```python
def run_benchmark(model, opponents, benchmark_suite):
    results = []

    for case in benchmark_suite:
        game_metrics = run_match(model, opponents[case.mode], case.seed)
        results.append(game_metrics)

    summary = aggregate_metrics(results)
    return summary
```

## 15. Analyse des résultats

L’analyse doit identifier les régressions de comportement.

### Questions à suivre

- le modèle gagne-t-il par expansion ou par défense ?
- perd-il en 4 joueurs parce qu’il s’expose trop ?
- attaque-t-il trop tôt ou trop tard ?
- choisit-il des cibles de faible valeur ?
- maintient-il suffisamment de flotte en réserve ?
- répète-t-il les mêmes ouvertures ?

### Pseudo-code

```python
def analyze_benchmark_results(results):
    report = {}
    report["global_wr"] = compute_winrate(results)
    report["wr_2p"] = compute_winrate(filter_mode(results, 2))
    report["wr_4p"] = compute_winrate(filter_mode(results, 4))
    report["invalid_action_rate"] = compute_invalid_rate(results)
    report["avg_decision_time"] = compute_avg_time(results)
    report["action_histogram"] = compute_action_histogram(results)
    report["failure_modes"] = detect_failure_patterns(results)
    return report
```

## 16. Limites

- Le jeu est combinatoire et partiellement séquentiel, donc les labels parfaits n’existent pas toujours.
- Le self-play peut dériver vers des styles fragiles si le pool est trop étroit.
- Les replays peuvent refléter des biais de population.
- Le mode 4 joueurs augmente fortement la variance.
- Un encodeur trop gros risque de dépasser les budgets d’exécution.

## 17. Prochaines améliorations

1. Ajouter un encodeur relationnel plus fort entre planètes et flottes.
2. Ajouter une tête de prévision de combats à horizon court.
3. Ajouter un module de mémoire des adversaires par style.
4. Introduire un curriculum 2p -> 3p -> 4p.
5. Remplacer le simple MLP policy par un decoder pointer plus riche.
6. Ajouter de la distillation entre checkpoints.
7. Construire un ensemble de baselines fixes pour l’évaluation continue.

## 18. Pseudo-code complet

```python
def agent(obs, config):
    encoded = encode_state(obs, config)
    outputs = forward_network(encoded)
    action_candidates = build_action_candidates(encoded["game"])
    action_candidates = adjust_policy_for_four_players(action_candidates, encoded["game"])
    action = logits_to_action(encoded, outputs, encoded["game"])
    return action


def training_loop(config):
    model = init_model(config)
    replay_dataset = load_replays(config.replay_path)
    model_pool = init_model_pool(config)
    best_state = init_best_state()

    # Phase 1: imitation
    for epoch in range(config.bc_epochs):
        batch = sample_replay_batch(replay_dataset, config.batch_size)
        outputs = model.forward(batch.states)
        loss = supervised_loss(outputs, batch.labels)
        model.backward(loss)
        model.step()

    # Phase 2: self-play
    for iteration in range(config.self_play_iterations):
        episodes = []
        for _ in range(config.games_per_iter):
            game = sample_game(config)
            opponents = sample_opponents(model_pool, game)
            episode = play_game(model, opponents, game)
            episodes.append(episode)

        rl_batch = build_rl_batch(episodes)
        outputs = model.forward(rl_batch.states)
        loss = rl_loss(outputs, rl_batch)
        model.backward(loss)
        model.step()

        if iteration % config.eval_every == 0:
            metrics = run_benchmark(model, config.benchmark_opponents, config.benchmark_suite)
            save_checkpoint(model, metrics, config.latest_path)
            save_best_model(model, metrics, best_state)
            update_model_pool(model_pool, model, metrics)

    export_numpy_weights(model, config.export_path)
    return model


def supervised_loss(outputs, labels):
    loss_policy = cross_entropy(outputs["policy_logits"], labels.policy)
    loss_value = mse(outputs["value"], labels.value_target)
    loss_aux = auxiliary_loss(outputs["aux"], labels.aux_targets)
    return loss_policy + 0.5 * loss_value + 0.25 * loss_aux


def rl_loss(outputs, batch):
    policy_loss = policy_gradient_loss(outputs["policy_logits"], batch.actions, batch.advantages)
    value_loss = mse(outputs["value"], batch.returns)
    entropy = entropy_bonus(outputs["policy_logits"])
    return policy_loss + 0.5 * value_loss - 0.01 * entropy


def compare_model_versions(model_a, model_b, benchmark_suite):
    results_a = run_benchmark(model_a, benchmark_suite.opponents, benchmark_suite)
    results_b = run_benchmark(model_b, benchmark_suite.opponents, benchmark_suite)
    return {
        "model_a": analyze_benchmark_results(results_a),
        "model_b": analyze_benchmark_results(results_b),
        "delta": diff_metrics(results_a, results_b),
    }
```

