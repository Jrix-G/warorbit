"""Parity checks for sim.py against kaggle_environments Orbit Wars."""

import math
import random

from kaggle_environments import make

import sim


POS_TOL = 0.05


def _get(o, k, default=None):
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def _passive_agent(obs, config=None):
    return []


def _greedy_agent(obs, config=None):
    planets = _get(obs, "planets", []) or []
    player = int(_get(obs, "player", 0) or 0)
    actions = []
    my_planets = [p for p in planets if int(p[1]) == player]
    targets = [p for p in planets if int(p[1]) != player]
    if not my_planets or not targets:
        return []

    for src in my_planets:
        ships = int(src[5])
        if ships <= 5:
            continue
        best = None
        best_score = -1.0e18
        for target in targets:
            d = math.hypot(float(target[2]) - float(src[2]),
                           float(target[3]) - float(src[3]))
            score = float(target[6]) * 30.0 - d * 0.5 - float(target[5]) * 0.2
            if score > best_score:
                best_score = score
                best = target
        if best is None:
            continue
        angle = math.atan2(float(best[3]) - float(src[3]),
                           float(best[2]) - float(src[2]))
        actions.append([int(src[0]), angle, int(ships * 0.60)])
    return actions


def _rows_by_id(rows):
    return {int(r[0]): r for r in rows}


def compare_states(kg_obs, sim_state, turn):
    """Compare Kaggle observation and simulated GameState."""
    errors = []
    kg_planets = list(_get(kg_obs, "planets", []) or [])
    kg_fleets = list(_get(kg_obs, "fleets", []) or [])

    if int(_get(kg_obs, "step", turn) or 0) != int(sim_state.step):
        errors.append(f"turn {turn}: step kg={_get(kg_obs, 'step', None)} sim={sim_state.step}")

    kg_comets = set(int(x) for x in (_get(kg_obs, "comet_planet_ids", []) or []))
    if kg_comets != set(sim_state.comet_ids):
        errors.append(f"turn {turn}: comet ids kg={sorted(kg_comets)} sim={sorted(sim_state.comet_ids)}")

    if int(_get(kg_obs, "next_fleet_id", 0) or 0) != int(sim_state.next_fleet_id):
        errors.append(
            f"turn {turn}: next_fleet_id kg={_get(kg_obs, 'next_fleet_id', None)} sim={sim_state.next_fleet_id}"
        )

    if len(kg_planets) != len(sim_state.planets):
        errors.append(f"turn {turn}: planet count kg={len(kg_planets)} sim={len(sim_state.planets)}")
    if len(kg_fleets) != len(sim_state.fleets):
        errors.append(f"turn {turn}: fleet count kg={len(kg_fleets)} sim={len(sim_state.fleets)}")

    sim_planets = _rows_by_id(sim_state.planets)
    for kp in kg_planets:
        pid = int(kp[0])
        sp = sim_planets.get(pid)
        if sp is None:
            errors.append(f"turn {turn}: missing sim planet {pid}")
            continue
        if int(kp[1]) != int(sp[sim.P_OWNER]):
            errors.append(f"turn {turn}: planet {pid} owner kg={kp[1]} sim={int(sp[sim.P_OWNER])}")
        if abs(float(kp[2]) - float(sp[sim.P_X])) > POS_TOL or abs(float(kp[3]) - float(sp[sim.P_Y])) > POS_TOL:
            errors.append(
                f"turn {turn}: planet {pid} pos kg=({float(kp[2]):.4f},{float(kp[3]):.4f}) "
                f"sim=({float(sp[sim.P_X]):.4f},{float(sp[sim.P_Y]):.4f})"
            )
        if abs(float(kp[4]) - float(sp[sim.P_R])) > 1e-5:
            errors.append(f"turn {turn}: planet {pid} radius kg={kp[4]} sim={float(sp[sim.P_R])}")
        if int(kp[5]) != int(sp[sim.P_SHIPS]):
            errors.append(f"turn {turn}: planet {pid} ships kg={kp[5]} sim={int(sp[sim.P_SHIPS])}")
        if int(kp[6]) != int(sp[sim.P_PROD]):
            errors.append(f"turn {turn}: planet {pid} prod kg={kp[6]} sim={int(sp[sim.P_PROD])}")

    sim_fleets = _rows_by_id(sim_state.fleets)
    for kf in kg_fleets:
        fid = int(kf[0])
        sf = sim_fleets.get(fid)
        if sf is None:
            errors.append(f"turn {turn}: missing sim fleet {fid}")
            continue
        if int(kf[1]) != int(sf[sim.F_OWNER]):
            errors.append(f"turn {turn}: fleet {fid} owner kg={kf[1]} sim={int(sf[sim.F_OWNER])}")
        if abs(float(kf[2]) - float(sf[sim.F_X])) > POS_TOL or abs(float(kf[3]) - float(sf[sim.F_Y])) > POS_TOL:
            errors.append(
                f"turn {turn}: fleet {fid} pos kg=({float(kf[2]):.4f},{float(kf[3]):.4f}) "
                f"sim=({float(sf[sim.F_X]):.4f},{float(sf[sim.F_Y]):.4f})"
            )
        if abs(float(kf[4]) - float(sf[sim.F_ANGLE])) > 1e-5:
            errors.append(f"turn {turn}: fleet {fid} angle kg={kf[4]} sim={float(sf[sim.F_ANGLE])}")
        if int(kf[5]) != int(sf[sim.F_FROM]):
            errors.append(f"turn {turn}: fleet {fid} from kg={kf[5]} sim={int(sf[sim.F_FROM])}")
        if int(kf[6]) != int(sf[sim.F_SHIPS]):
            errors.append(f"turn {turn}: fleet {fid} ships kg={kf[6]} sim={int(sf[sim.F_SHIPS])}")

    return errors


def _run_and_compare(agents, n_games, label):
    all_errors = []
    n_players = len(agents)
    for game_idx in range(n_games):
        env = make("orbit_wars", debug=False)
        env.step([[] for _ in range(n_players)])
        if len(env.steps) < 2:
            all_errors.append(f"game {game_idx}: environment produced no initialized state")
            continue

        state = sim.state_from_obs(env.steps[1][0].observation)
        initial_errors = compare_states(env.steps[1][0].observation, state, 1)
        all_errors.extend(f"game {game_idx}: {e}" for e in initial_errors)

        while not env.done:
            current = env.steps[-1]
            actions_list = []
            actions_by_player = {}
            for player in range(n_players):
                action = agents[player](current[player].observation, env.configuration)
                if not isinstance(action, list):
                    action = []
                actions_list.append(action)
                actions_by_player[player] = action

            rng_state = random.getstate()
            sim.step_inplace(state, actions_by_player)
            random.setstate(rng_state)
            env.step(actions_list)

            turn = int(env.steps[-1][0].observation.step)
            errors = compare_states(env.steps[-1][0].observation, state, turn)
            if errors:
                all_errors.extend(f"game {game_idx}: {e}" for e in errors[:20])
                break

    if all_errors:
        print(f"{label}: FAIL ({len(all_errors)} erreurs)")
        for err in all_errors[:50]:
            print("  " + err)
    else:
        print(f"{label}: OK ({n_games} parties)")
    return all_errors


def test_passive_game(n_games=3):
    return _run_and_compare([_passive_agent, _passive_agent], n_games, "Passive vs passive")


def test_greedy_game(n_games=3):
    return _run_and_compare([_greedy_agent, _greedy_agent], n_games, "Greedy vs greedy")


if __name__ == "__main__":
    test_passive_game(3)
    test_greedy_game(3)
    print("Tests termines.")
