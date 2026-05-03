"""Self-play and mixed-opponent match utilities for V9."""

from __future__ import annotations

import math
import random
import time
from typing import Callable, Dict, Iterable, List, Sequence

from SimGame import P_OWNER, P_PROD, P_SHIPS, P_X, P_Y, SimGame

from ..agents.v9.policy import V9Agent, V9Weights, load_checkpoint
from ..config.v9_config import V9Config
from .curriculum import MatchSpec


class DeadlineExceeded(RuntimeError):
    """Raised when the global training deadline is reached."""


def _safe_agent(fn: Callable):
    def wrapped(obs, config=None):
        try:
            return fn(obs, config)
        except TypeError:
            try:
                return fn(obs)
            except Exception:
                return []
        except Exception:
            return []

    return wrapped


def _base_name(name: str) -> str:
    if name.startswith("heldout_"):
        return name[len("heldout_"):]
    if name.startswith("train_"):
        return name[len("train_"):]
    if name.startswith("eval_"):
        return name[len("eval_"):]
    if name.startswith("bench_"):
        return name[len("bench_"):]
    return name


def noisy_greedy_agent(obs, config=None):
    planets = obs.get("planets", [])
    me = obs.get("player", 0)
    my = [p for p in planets if int(p[1]) == int(me)]
    others = [p for p in planets if int(p[1]) != int(me)]
    if not my or not others:
        return []
    rng = random.Random(int(obs.get("step", 0)) * 1009 + int(me) * 9173 + len(planets))
    moves = []
    for src in sorted(my, key=lambda p: -float(p[5])):
        ships = int(src[5])
        if ships < 8 or rng.random() < 0.18:
            continue
        pool = sorted(others, key=lambda t: math.hypot(float(t[2] - src[2]), float(t[3] - src[3])))[:4]
        target = rng.choice(pool)
        angle = math.atan2(float(target[3] - src[3]), float(target[2] - src[2])) + rng.uniform(-0.10, 0.10)
        send = int(ships * rng.uniform(0.28, 0.62))
        if send >= 3:
            moves.append([int(src[0]), angle, send])
    return moves


def local_random_agent(obs, config=None):
    planets = obs.get("planets", [])
    me = obs.get("player", 0)
    my = [p for p in planets if int(p[1]) == int(me)]
    if not my:
        return []
    rng = random.Random(int(obs.get("step", 0)) * 6361 + int(me) * 1871 + len(planets))
    moves = []
    for src in my:
        ships = int(src[5])
        if ships < 8 or rng.random() > 0.38:
            continue
        moves.append([int(src[0]), rng.random() * 2.0 * math.pi, max(3, int(ships * rng.uniform(0.18, 0.45)))])
    return moves


def local_greedy_agent(obs, config=None):
    planets = obs.get("planets", [])
    me = obs.get("player", 0)
    my = [p for p in planets if int(p[1]) == int(me)]
    targets = [p for p in planets if int(p[1]) != int(me)]
    if not my or not targets:
        return []
    moves = []
    for src in sorted(my, key=lambda p: -float(p[5])):
        ships = int(src[5])
        if ships < 10:
            continue
        target = min(targets, key=lambda t: (float(t[5]) - 2.0 * float(t[6]), math.hypot(float(t[2] - src[2]), float(t[3] - src[3]))))
        angle = math.atan2(float(target[3] - src[3]), float(target[2] - src[2]))
        send = int(ships * 0.50)
        if send >= 4:
            moves.append([int(src[0]), angle, send])
    return moves


def opponent_agent(name: str, config: V9Config | None = None, current_weights: V9Weights | None = None):
    if name in ("v9_current", "self_play"):
        return V9Agent(config or V9Config(), current_weights)
    if name.startswith("v9_checkpoint:"):
        path = name.split(":", 1)[1]
        from ..agents.v9.policy import get_weights, set_weights

        old = get_weights()
        ok = load_checkpoint(path)
        weights = get_weights() if ok else old
        set_weights(old)
        return V9Agent(config or V9Config(), weights)
    base = _base_name(name)
    if base == "random":
        return _safe_agent(local_random_agent)
    if base in ("greedy", "starter", "distance", "sun_dodge", "structured", "orbit_stars"):
        return _safe_agent(local_greedy_agent)
    if base == "noisy_greedy":
        return _safe_agent(noisy_greedy_agent)
    try:
        from opponents import ZOO

        if base in ZOO:
            return _safe_agent(ZOO[base])
    except Exception:
        pass
    try:
        import bot_v7

        if base == "bot_v7":
            return _safe_agent(bot_v7.agent)
    except Exception:
        pass
    return _safe_agent(lambda obs, config=None: [])


def reward_from_result(result: dict, our_index: int) -> float:
    winner = int(result.get("winner", -1))
    if winner == our_index:
        base = 1.0
    elif winner < 0:
        base = 0.5
    else:
        base = 0.0
    scores = list(result.get("scores", []))
    if 0 <= our_index < len(scores):
        ours = float(scores[our_index])
        other = max((float(s) for i, s in enumerate(scores) if i != our_index), default=ours)
        margin = math.tanh((ours - other) / max(120.0, abs(ours) + abs(other)))
    else:
        margin = 0.0
    return max(0.0, min(1.0, base + 0.06 * margin))


def _perturb_game_state(game: SimGame, intensity: float, seed: int) -> None:
    if intensity <= 0.0 or len(game.state.planets) == 0:
        return
    rng = random.Random(seed)
    planets = game.state.planets
    for i in range(len(planets)):
        owner = int(planets[i, P_OWNER])
        if owner < 0:
            planets[i, P_SHIPS] = max(1.0, float(planets[i, P_SHIPS]) * rng.uniform(1.0 - intensity, 1.0 + intensity))
            planets[i, P_PROD] = max(1.0, round(float(planets[i, P_PROD]) * rng.uniform(1.0 - intensity * 0.5, 1.0 + intensity * 0.5)))
        if rng.random() < 0.35:
            planets[i, P_X] = min(98.0, max(2.0, float(planets[i, P_X]) + rng.uniform(-intensity, intensity) * 3.0))
            planets[i, P_Y] = min(98.0, max(2.0, float(planets[i, P_Y]) + rng.uniform(-intensity, intensity) * 3.0))
    game.state.initial_planets = planets.copy()


def _run_game_with_deadline(game: SimGame, agents: Sequence[Callable], max_steps: int, deadline: float | None) -> dict:
    if max_steps is not None:
        game.state.max_steps = int(max_steps)

    started = time.perf_counter()
    while not game.is_terminal():
        if deadline is not None and time.time() >= deadline:
            raise DeadlineExceeded("global training deadline reached")
        actions = {}
        for player, agent in enumerate(agents):
            if deadline is not None and time.time() >= deadline:
                raise DeadlineExceeded("global training deadline reached")
            obs = game.observation(player)
            try:
                move = agent(obs, None)
            except TypeError:
                move = agent(obs)
            actions[player] = move if isinstance(move, list) else []
        game.step(actions)

    elapsed = time.perf_counter() - started
    return {
        "winner": game.winner(),
        "scores": game.scores(),
        "steps": int(game.state.step),
        "seconds": elapsed,
        "steps_per_second": game.state.step / max(elapsed, 1e-9),
    }


def play_match_spec(weights: V9Weights, config: V9Config, spec: MatchSpec, deadline: float | None = None) -> dict:
    agents: List[Callable] = []
    opp_iter = iter(spec.opponent_names)
    our_agent = None
    for slot in range(spec.n_players):
        if slot == spec.our_index:
            our_agent = V9Agent(config, weights)
            agents.append(our_agent)
        else:
            agents.append(opponent_agent(next(opp_iter), config, current_weights=weights))
    game = SimGame.random_game(seed=spec.seed, n_players=spec.n_players, max_steps=spec.max_steps)
    if spec.phase == "train":
        _perturb_game_state(game, float(getattr(config, "train_state_perturbation", 0.0)), spec.seed)
    result = _run_game_with_deadline(game, agents, spec.max_steps, deadline)
    result["reward"] = reward_from_result(result, spec.our_index)
    result["mode"] = "4p" if spec.n_players >= 4 else "2p"
    result["our_index"] = spec.our_index
    result["opponents"] = list(spec.opponent_names)
    result["phase"] = spec.phase
    result["plan_history"] = list(getattr(our_agent, "plan_history", []))
    result["plan_stats"] = dict(getattr(our_agent, "plan_stats", {}) or {})
    return result


def _play_match_spec_task(task) -> dict:
    flat, config, spec, deadline = task
    return play_match_spec(V9Weights.from_flat(flat), config, spec, deadline=deadline)


def evaluate_weights(
    weights: V9Weights,
    config: V9Config,
    specs: Sequence[MatchSpec],
    deadline: float | None = None,
    *,
    progress_label: str | None = None,
    progress_every: int = 0,
    pool=None,
) -> List[dict]:
    results = []
    total = len(specs)
    started = time.perf_counter()
    flat = weights.flatten()
    if pool is not None and total > 1:
        iterator = pool.imap_unordered(
            _play_match_spec_task,
            [(flat, config, spec, deadline) for spec in specs],
            chunksize=1,
        )
    else:
        iterator = (_play_match_spec_task((flat, config, spec, deadline)) for spec in specs)
    for idx, result in enumerate(iterator, start=1):
        if deadline is not None and time.time() >= deadline:
            raise DeadlineExceeded("global training deadline reached")
        results.append(result)
        if progress_label and int(progress_every) > 0 and (idx % int(progress_every) == 0 or idx == total):
            elapsed = time.perf_counter() - started
            eta = (elapsed / max(1, idx)) * max(0, total - idx)
            summary = summarise_results(results)
            last = results[-1]
            print(
                f"{progress_label} progress={idx}/{total} "
                f"mean={summary['mean']:.3f} "
                f"2p={summary['wr_2p']:.3f}/{summary['n_2p']} "
                f"4p={summary['wr_4p']:.3f}/{summary['n_4p']} "
                f"last={last.get('mode', '?')}:{float(last.get('seconds', 0.0)):.1f}s "
                f"elapsed_min={elapsed/60.0:.1f} eta_min={eta/60.0:.1f}",
                flush=True,
            )
    return results


def summarise_results(results: Iterable[dict]) -> Dict[str, float]:
    results = list(results)
    if not results:
        return {"mean": 0.0, "wr_2p": 0.0, "wr_4p": 0.0, "n_2p": 0, "n_4p": 0}
    rewards = [float(r.get("reward", 0.0)) for r in results]
    r2 = [float(r.get("reward", 0.0)) for r in results if r.get("mode") == "2p"]
    r4 = [float(r.get("reward", 0.0)) for r in results if r.get("mode") == "4p"]
    plan_counts: Dict[str, int] = {}
    stat_totals: Dict[str, float] = {}
    total_plans = 0
    for result in results:
        for plan in result.get("plan_history", []) or []:
            plan_counts[str(plan)] = plan_counts.get(str(plan), 0) + 1
            total_plans += 1
        for key, value in (result.get("plan_stats", {}) or {}).items():
            try:
                stat_totals[str(key)] = stat_totals.get(str(key), 0.0) + float(value)
            except Exception:
                continue
    if total_plans:
        probs = [count / total_plans for count in plan_counts.values()]
        entropy = -sum(p * math.log(max(p, 1e-9)) for p in probs) / max(1e-9, math.log(max(2, len(plan_counts))))
        dominant = max(probs)
    else:
        entropy = 0.0
        dominant = 1.0
    total_moves = max(1.0, float(stat_totals.get("total_moves", 0.0)))
    total_turns = max(1.0, float(stat_totals.get("turns", 0.0)))
    four_p_turns = max(1.0, float(stat_totals.get("four_p_turns", 0.0)))
    return {
        "mean": sum(rewards) / max(1, len(rewards)),
        "wr_2p": sum(r2) / max(1, len(r2)) if r2 else 0.0,
        "wr_4p": sum(r4) / max(1, len(r4)) if r4 else 0.0,
        "n_2p": len(r2),
        "n_4p": len(r4),
        "plan_entropy": entropy,
        "dominant_plan_frac": dominant,
        "transfer_move_frac": float(stat_totals.get("transfer_moves", 0.0)) / total_moves,
        "transfer_ship_frac": float(stat_totals.get("transfer_ships", 0.0)) / max(1.0, float(stat_totals.get("total_ships_sent", 0.0))),
        "attack_move_frac": float(stat_totals.get("attack_moves", 0.0)) / total_moves,
        "expand_move_frac": float(stat_totals.get("expand_moves", 0.0)) / total_moves,
        "backbone_turn_frac": float(stat_totals.get("backbone_turns", 0.0)) / total_turns,
        "front_lock_turn_frac": float(stat_totals.get("front_lock_turns", 0.0)) / four_p_turns,
        "staged_finisher_turn_frac": float(stat_totals.get("staged_finisher_turns", 0.0)) / total_turns,
        "consolidation_threshold_turn_frac": float(stat_totals.get("consolidation_threshold_turns", 0.0)) / total_turns,
        "active_front_avg": float(stat_totals.get("active_front_sum", 0.0)) / four_p_turns,
        "focus_switches": float(stat_totals.get("focus_switches", 0.0)) / max(1.0, len(results)),
    }
