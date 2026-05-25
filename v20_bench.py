"""V20 benchmark harness against raw V15 search.

Default target is a future/current V20 module, resolved in order:
    v20_agent:agent, v20_search:search, v20_search:gumbel_move

Useful smoke run against an existing state-style agent:
    python -u v20_bench.py --candidate v18_search:gumbel_move --games 2 --modes 2 --workers 1

Real V20 run with per-game JSONL:
    python -u v20_bench.py --games 24 --modes 2,4 --seat-rotation full --log analysis/v20_run.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import inspect
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

import v14_core
import v15_fast_sim as fsim
import v15_search
from local_simulator.official_fast import OfficialFastGame


DEFAULT_CANDIDATE = "v20_agent:agent,v20_search:search,v20_search:gumbel_move"

_CANDIDATE_SPEC = ""
_CANDIDATE_FN: Any = None
_CANDIDATE_SIG: inspect.Signature | None = None
_CANDIDATE_STYLE = "auto"
_N_SIMS = 64
_CANDIDATE_BUDGET: float | None = None
_CANDIDATE_HORIZON: int | None = None
_V15_BUDGET = 0.7
_EPISODE_STEPS = 250
_FAIL_ON_AGENT_ERROR = False


class BenchConfig:
    def __init__(self, episode_steps: int, seed: int | None = None, n_players: int | None = None) -> None:
        self.episodeSteps = int(episode_steps)
        self.nPlayers = int(n_players) if n_players is not None else None
        self.actTimeout = 1.0
        self.shipSpeed = 6.0
        self.cometSpeed = 4.0
        self.remainingOverageTime = 60.0
        self.seed = seed

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = wins / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return p, centre - margin, centre + margin


def _parse_modes(raw: str) -> list[int]:
    modes: list[int] = []
    for part in raw.split(","):
        part = part.strip().lower().removesuffix("p")
        if not part:
            continue
        n_players = int(part)
        if n_players not in (2, 4):
            raise ValueError(f"unsupported mode {n_players}; expected 2 or 4")
        modes.append(n_players)
    if not modes:
        raise ValueError("at least one mode is required")
    return modes


def _parse_seeds(raw: str, seed_offset: int, games: int) -> list[int]:
    if not raw:
        return [seed_offset + i for i in range(games)]
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            step = 1 if hi >= lo else -1
            out.extend(range(lo, hi + step, step))
        else:
            out.append(int(part))
    if not out:
        raise ValueError("--seeds parsed to an empty list")
    return out


def _load_attr(spec: str) -> Any:
    if ":" in spec:
        module_name, attr_name = spec.split(":", 1)
    else:
        module_name, attr_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = module
    for attr in attr_name.split("."):
        obj = getattr(obj, attr)
    return obj


def _resolve_candidate(specs: str) -> tuple[str, Any]:
    errors: list[str] = []
    for spec in (s.strip() for s in specs.split(",")):
        if not spec:
            continue
        try:
            return spec, _load_attr(spec)
        except Exception as exc:  # pragma: no cover - exercised by CLI use
            errors.append(f"{spec}: {exc}")
    detail = "; ".join(errors) if errors else "no candidate specs were provided"
    raise RuntimeError(f"could not import candidate ({detail})")


def _safe_signature(fn: Any) -> inspect.Signature | None:
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def _infer_style(spec: str, sig: inspect.Signature | None) -> str:
    params = list(sig.parameters) if sig is not None else []
    first = params[0].lower() if params else ""
    lowered = spec.lower()
    if first in {"fs", "state", "fast_state"} or "player" in params:
        return "state"
    if first in {"obs", "observation"}:
        return "obs"
    if any(token in lowered for token in ("move", "gumbel", "mcts")):
        return "state"
    return "obs"


def _init(
    candidate_spec: str,
    candidate_style: str,
    n_sims: int,
    candidate_budget: float | None,
    candidate_horizon: int | None,
    v15_budget: float,
    episode_steps: int,
    fail_on_agent_error: bool,
) -> None:
    global _CANDIDATE_SPEC, _CANDIDATE_FN, _CANDIDATE_SIG, _CANDIDATE_STYLE
    global _N_SIMS, _CANDIDATE_BUDGET, _CANDIDATE_HORIZON
    global _V15_BUDGET, _EPISODE_STEPS, _FAIL_ON_AGENT_ERROR

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    _CANDIDATE_SPEC, _CANDIDATE_FN = _resolve_candidate(candidate_spec)
    _CANDIDATE_SIG = _safe_signature(_CANDIDATE_FN)
    _CANDIDATE_STYLE = candidate_style
    _N_SIMS = int(n_sims)
    _CANDIDATE_BUDGET = candidate_budget
    _CANDIDATE_HORIZON = candidate_horizon
    _V15_BUDGET = float(v15_budget)
    _EPISODE_STEPS = int(episode_steps)
    _FAIL_ON_AGENT_ERROR = bool(fail_on_agent_error)


def _candidate_kwargs(rng: np.random.Generator | None, seed: int) -> dict[str, Any]:
    sig = _CANDIDATE_SIG
    if sig is None:
        kwargs: dict[str, Any] = {}
        if rng is not None:
            kwargs.update({"n_sims": _N_SIMS, "rng": rng})
        if _CANDIDATE_BUDGET is not None:
            kwargs["time_budget"] = _CANDIDATE_BUDGET
        if _CANDIDATE_HORIZON is not None:
            kwargs["horizon"] = _CANDIDATE_HORIZON
        return kwargs
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs: dict[str, Any] = {}
    if rng is not None and (has_varkw or "n_sims" in params):
        kwargs["n_sims"] = _N_SIMS
    if rng is not None and "nsims" in params:
        kwargs["nsims"] = _N_SIMS
    if rng is not None and (has_varkw or "rng" in params):
        kwargs["rng"] = rng
    if "seed" in params:
        kwargs["seed"] = seed
    if _CANDIDATE_BUDGET is not None:
        if has_varkw or "time_budget" in params:
            kwargs["time_budget"] = _CANDIDATE_BUDGET
        elif "budget" in params:
            kwargs["budget"] = _CANDIDATE_BUDGET
    if _CANDIDATE_HORIZON is not None and (has_varkw or "horizon" in params):
        kwargs["horizon"] = _CANDIDATE_HORIZON
    return kwargs


def _call_state_candidate(fs: fsim.FastState, player: int, rng: np.random.Generator, seed: int) -> Any:
    kwargs = _candidate_kwargs(rng, seed)
    if _CANDIDATE_SIG is None:
        try:
            return _CANDIDATE_FN(fs, player, **kwargs)
        except TypeError:
            return _CANDIDATE_FN(fs, player)
    return _CANDIDATE_FN(fs, player, **kwargs)


def _call_obs_candidate(fs: fsim.FastState, player: int, seed: int) -> Any:
    obs = v15_search.state_to_obs(fs, player)
    cfg = BenchConfig(_EPISODE_STEPS, seed, fs.n_players)
    kwargs = _candidate_kwargs(None, seed)
    sig = _CANDIDATE_SIG
    if sig is None:
        try:
            return _CANDIDATE_FN(obs, cfg, **kwargs)
        except TypeError:
            return _CANDIDATE_FN(obs)

    params = list(sig.parameters.values())
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    if has_varargs or len(positional) >= 2:
        return _CANDIDATE_FN(obs, cfg, **kwargs)
    return _CANDIDATE_FN(obs, **kwargs)


def _call_candidate(fs: fsim.FastState, player: int, rng: np.random.Generator, seed: int) -> Any:
    style = _CANDIDATE_STYLE
    if style == "auto":
        style = _infer_style(_CANDIDATE_SPEC, _CANDIDATE_SIG)
    if style == "state":
        return _call_state_candidate(fs, player, rng, seed)
    return _call_obs_candidate(fs, player, seed)


def _normalise_move(move: Any) -> list[list[Any]]:
    if not isinstance(move, list):
        return []
    out: list[list[Any]] = []
    for item in move:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        try:
            out.append([int(item[0]), float(item[1]), int(item[2])])
        except (TypeError, ValueError):
            continue
    return out


def _call_v15(fs: fsim.FastState, player: int) -> list[list[Any]]:
    obs = v15_search.state_to_obs(fs, player)
    return _normalise_move(v15_search.search(obs, None, time_budget=_V15_BUDGET))


def _play(task: tuple[int, int, int, int]) -> dict[str, Any]:
    n_players, seed, our_seat, task_index = task
    agent_seed = (int(seed) + 1_000_003 * int(our_seat)) % (2**32 - 1)
    random.seed(agent_seed)
    np.random.seed(agent_seed)
    rng = np.random.default_rng(agent_seed)

    game = OfficialFastGame(
        n_players,
        seed=seed,
        episode_steps=_EPISODE_STEPS,
        use_c_accel=False,
    )
    fs = fsim.from_obs(
        v14_core.obs_as_dict(game.observation(0)),
        n_players=n_players,
        episode_steps=_EPISODE_STEPS,
    )
    fs.n_players = n_players

    launches_by_player = [0 for _ in range(n_players)]
    candidate_ms = 0.0
    v15_ms = 0.0
    candidate_errors = 0
    first_candidate_error = ""
    intent_counts: Counter[str] = Counter()
    steps = 0

    while not fs.done:
        actions: list[list[list[Any]]] = []
        for player in range(n_players):
            started = time.perf_counter()
            if player == our_seat:
                try:
                    move = _normalise_move(_call_candidate(fs, player, rng, seed))
                    intent_name = _last_candidate_intent_name()
                    if intent_name:
                        intent_counts[intent_name] += 1
                except Exception as exc:
                    if _FAIL_ON_AGENT_ERROR:
                        raise
                    candidate_errors += 1
                    if not first_candidate_error:
                        first_candidate_error = f"{type(exc).__name__}: {exc}"
                    move = []
                candidate_ms += (time.perf_counter() - started) * 1000.0
            else:
                move = _call_v15(fs, player)
                v15_ms += (time.perf_counter() - started) * 1000.0
            launches_by_player[player] += len(move)
            actions.append(move)
        fs = fsim.step(fs, actions)
        steps += 1

    scores = [int(x) for x in fsim.scores(fs)]
    our_score = scores[our_seat]
    best_v15_score = max(score for idx, score in enumerate(scores) if idx != our_seat)
    win = int(our_score > best_v15_score and our_score > 0)
    draw = int(our_score == best_v15_score and our_score > 0)

    return {
        "type": "game",
        "task_index": task_index,
        "candidate": _CANDIDATE_SPEC,
        "opponent": "raw_v15_search",
        "mode": f"{n_players}p",
        "n_players": n_players,
        "seed": seed,
        "our_seat": our_seat,
        "win": win,
        "draw": draw,
        "our_score": our_score,
        "best_v15_score": best_v15_score,
        "score_margin": our_score - best_v15_score,
        "all_scores": scores,
        "our_launches": launches_by_player[our_seat],
        "v15_launches": sum(v for idx, v in enumerate(launches_by_player) if idx != our_seat),
        "launches_by_player": launches_by_player,
        "candidate_ms": round(candidate_ms, 3),
        "v15_ms": round(v15_ms, 3),
        "steps": steps,
        "intent_counts": dict(intent_counts),
        "candidate_errors": candidate_errors,
        "first_candidate_error": first_candidate_error,
    }


def _last_candidate_intent_name() -> str:
    if not _CANDIDATE_SPEC.startswith("v20_agent:"):
        return ""
    try:
        import v20_agent

        intent = v20_agent.last_intent()
        return str(getattr(intent, "name", "") or "")
    except Exception:
        return ""


def _build_tasks(modes: list[int], seeds: list[int], seat_rotation: str) -> list[tuple[int, int, int, int]]:
    tasks: list[tuple[int, int, int, int]] = []
    for n_players in modes:
        for seed_index, seed in enumerate(seeds):
            if seat_rotation == "full":
                seats = range(n_players)
            elif seat_rotation == "cycle":
                seats = [seed_index % n_players]
            else:
                seats = [seed % n_players]
            for seat in seats:
                tasks.append((n_players, seed, int(seat), len(tasks)))
    return tasks


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r[key]) for r in rows) / len(rows)


def _print_summary(rows: list[dict[str, Any]], modes: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": "summary", "modes": {}, "aggregate": {}}
    agg_w = agg_n = 0
    print("\n=== FINAL: V20 candidate vs raw V15 ===", flush=True)
    for n_players in modes:
        mode = f"{n_players}p"
        sub = [r for r in rows if r["n_players"] == n_players]
        wins = sum(int(r["win"]) for r in sub)
        total = len(sub)
        agg_w += wins
        agg_n += total
        p, lo, hi = wilson_ci(wins, total)
        mode_summary = {
            "wins": wins,
            "total": total,
            "winrate": p,
            "ci_low": lo,
            "ci_high": hi,
            "avg_margin": _mean(sub, "score_margin"),
            "avg_our_launches": _mean(sub, "our_launches"),
            "avg_v15_launches": _mean(sub, "v15_launches"),
            "seats": {},
        }
        print(
            f"  {mode}: W={wins}/{total} WR={p:.3f} CI=[{lo:.3f},{hi:.3f}] "
            f"margin={mode_summary['avg_margin']:.1f} "
            f"launches us={mode_summary['avg_our_launches']:.1f} "
            f"v15={mode_summary['avg_v15_launches']:.1f}",
            flush=True,
        )
        seat_bits = []
        for seat in range(n_players):
            seat_rows = [r for r in sub if r["our_seat"] == seat]
            sw = sum(int(r["win"]) for r in seat_rows)
            sn = len(seat_rows)
            sp, slo, shi = wilson_ci(sw, sn)
            mode_summary["seats"][str(seat)] = {
                "wins": sw,
                "total": sn,
                "winrate": sp,
                "ci_low": slo,
                "ci_high": shi,
            }
            seat_bits.append(f"s{seat} {sw}/{sn}")
        print(f"       seats: {' | '.join(seat_bits)}", flush=True)
        summary["modes"][mode] = mode_summary

    p, lo, hi = wilson_ci(agg_w, agg_n)
    verdict = "BEATS_RAW_V15" if lo > 0.5 else ("INCONCLUSIVE" if hi >= 0.5 else "LOSES_TO_RAW_V15")
    summary["aggregate"] = {
        "wins": agg_w,
        "total": agg_n,
        "winrate": p,
        "ci_low": lo,
        "ci_high": hi,
        "verdict": verdict,
    }
    print(f"  AGG: W={agg_w}/{agg_n} WR={p:.3f} CI=[{lo:.3f},{hi:.3f}] -> {verdict}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=DEFAULT_CANDIDATE, help="module:attr, or comma-separated fallbacks")
    ap.add_argument("--candidate-style", choices=("auto", "state", "obs"), default="auto")
    ap.add_argument("--games", type=int, default=24, help="base seeds per mode")
    ap.add_argument("--seeds", default="", help="comma/range list, e.g. 1,2,10-19; overrides --games")
    ap.add_argument("--modes", default="2,4", help="2, 4, or 2,4")
    ap.add_argument(
        "--seat-rotation",
        choices=("cycle", "full", "seed"),
        default="cycle",
        help="cycle keeps total games fixed; full repeats each seed for every seat; seed uses seed %% players",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=30_000_000)
    ap.add_argument("--episode-steps", type=int, default=250)
    ap.add_argument("--n-sims", type=int, default=64, help="passed to state-style candidates that accept n_sims")
    ap.add_argument("--candidate-budget", type=float, default=None, help="optional time_budget/budget for candidate")
    ap.add_argument("--candidate-horizon", type=int, default=None, help="optional horizon for candidate")
    ap.add_argument("--v15-budget", type=float, default=0.7, help="raw v15_search time_budget")
    ap.add_argument("--progress-every", type=int, default=1)
    ap.add_argument("--log", default="", help="optional JSONL output path")
    ap.add_argument("--fail-on-agent-error", action="store_true")
    args = ap.parse_args()

    try:
        selected_spec, selected_fn = _resolve_candidate(args.candidate)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Provide --candidate module:attr once the V20 agent/search file exists.", file=sys.stderr)
        return 2
    selected_sig = _safe_signature(selected_fn)
    inferred_style = args.candidate_style
    if inferred_style == "auto":
        inferred_style = _infer_style(selected_spec, selected_sig)

    modes = _parse_modes(args.modes)
    seeds = _parse_seeds(args.seeds, args.seed_offset, args.games)
    tasks = _build_tasks(modes, seeds, args.seat_rotation)
    rows: list[dict[str, Any]] = []

    print(
        f"V20 bench candidate={selected_spec} style={inferred_style} "
        f"vs=raw_v15_search modes={','.join(f'{m}p' for m in modes)} "
        f"base_seeds={len(seeds)} seat_rotation={args.seat_rotation} "
        f"scheduled_games={len(tasks)} workers={args.workers}",
        flush=True,
    )

    t0 = time.time()
    init_args = (
        args.candidate,
        args.candidate_style,
        args.n_sims,
        args.candidate_budget,
        args.candidate_horizon,
        args.v15_budget,
        args.episode_steps,
        args.fail_on_agent_error,
    )
    if args.workers <= 1:
        _init(*init_args)
        for task in tasks:
            rows.append(_play(task))
            if len(rows) % max(1, args.progress_every) == 0:
                _print_progress(rows[-1], len(rows), len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=init_args) as pool:
            futs = {pool.submit(_play, task): task for task in tasks}
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                if len(rows) % max(1, args.progress_every) == 0:
                    _print_progress(row, len(rows), len(tasks))

    rows.sort(key=lambda r: (r["n_players"], r["seed"], r["our_seat"]))
    summary = _print_summary(rows, modes)
    elapsed_min = (time.time() - t0) / 60.0
    print(f"elapsed {elapsed_min:.1f} min", flush=True)

    if args.log:
        path = Path(args.log)
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "type": "meta",
            "candidate": selected_spec,
            "candidate_style": inferred_style,
            "opponent": "raw_v15_search",
            "modes": modes,
            "seeds": seeds,
            "seat_rotation": args.seat_rotation,
            "episode_steps": args.episode_steps,
            "n_sims": args.n_sims,
            "candidate_budget": args.candidate_budget,
            "candidate_horizon": args.candidate_horizon,
            "v15_budget": args.v15_budget,
        }
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(meta, sort_keys=True) + "\n")
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            f.write(json.dumps(summary, sort_keys=True) + "\n")
        print(f"per-game log -> {path}", flush=True)

    return 0


def _print_progress(row: dict[str, Any], done: int, total: int) -> None:
    err = f" errors={row['candidate_errors']}" if row.get("candidate_errors") else ""
    print(
        f"[{done}/{total}] {row['mode']} seed={row['seed']} seat={row['our_seat']} "
        f"win={row['win']} score={row['our_score']} vs{row['best_v15_score']} "
        f"margin={row['score_margin']} L={row['our_launches']}/{row['v15_launches']}{err}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
