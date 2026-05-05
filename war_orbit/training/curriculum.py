"""Adaptive opponent and mode curriculum for V9."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass
class MatchSpec:
    opponent_names: List[str]
    our_index: int
    seed: int
    max_steps: int
    phase: str = "train"

    @property
    def n_players(self) -> int:
        return len(self.opponent_names) + 1


def _available_opponents(names: Sequence[str]) -> List[str]:
    """Keep aliases and installed opponents; drop missing notebook names."""

    always = {
        "random",
        "heldout_random",
        "greedy",
        "heldout_greedy",
        "noisy_greedy",
        "starter",
        "distance",
        "sun_dodge",
        "structured",
        "orbit_stars",
        "bot_v7",
        "self_play",
        "v9_current",
    }
    needs_zoo = any(
        not (name.startswith("v9_checkpoint:") or name in always)
        for name in names
    )
    if needs_zoo:
        try:
            from opponents import ZOO
        except Exception:
            ZOO = {}
    else:
        ZOO = {}
    out: List[str] = []
    for name in names:
        if name.startswith("v9_checkpoint:") or name in always or name in ZOO:
            if name not in out:
                out.append(name)
    return out


def append_past_versions(names: Sequence[str], checkpoint_paths: Sequence[str]) -> List[str]:
    out = list(names)
    for path in checkpoint_paths:
        if path and Path(path).exists():
            label = f"v9_checkpoint:{path}"
            if label not in out:
                out.append(label)
    return out


def build_role_pools(config) -> Dict[str, List[str]]:
    """Return strict train/eval/benchmark opponent roles.

    Eval intentionally uses heldout aliases for random/greedy so tests and logs
    can detect leakage by label while retaining simple baseline behaviors.
    """

    train = list(config.training_opponents)
    eval_pool = append_past_versions(config.eval_opponents, [config.best_checkpoint])
    benchmark = list(config.benchmark_opponents)
    return {
        "train": _available_opponents(train) or ["random", "noisy_greedy"],
        "eval": _available_opponents(eval_pool) or ["heldout_random", "heldout_greedy", "bot_v7"],
        "benchmark": _available_opponents(benchmark) or ["sun_dodge", "structured", "bot_v7"],
    }


class CurriculumScheduler:
    """Build mixed 2p/4p schedules and scale difficulty from results."""

    def __init__(self, opponents: Sequence[str], *, four_player_ratio: float = 0.80, seed: int = 9009):
        self.base_opponents = list(opponents) or ["random", "greedy"]
        self.four_player_ratio = float(four_player_ratio)
        self.seed = int(seed)
        self.difficulty = 0

    def update(self, eval_score: float) -> None:
        if eval_score >= 0.62:
            self.difficulty = min(3, self.difficulty + 1)
        elif eval_score < 0.35:
            self.difficulty = max(0, self.difficulty - 1)

    def opponent_pool(self) -> List[str]:
        pool = list(self.base_opponents)
        if self.difficulty >= 1:
            pool.extend(["starter", "distance"])
        if self.difficulty >= 2:
            pool.extend(["sun_dodge", "notebook_tactical_heuristic"])
        if self.difficulty >= 3:
            pool.extend(["notebook_orbitbotnext", "notebook_distance_prioritized", "notebook_physics_accurate"])
        out = []
        for name in pool:
            if name not in out:
                out.append(name)
        return out

    def build(
        self,
        games: int,
        *,
        seed_offset: int,
        max_steps: int,
        four_player_ratio: float | None = None,
        phase: str = "train",
    ) -> List[MatchSpec]:
        rng = random.Random(self.seed + int(seed_offset))
        ratio = self.four_player_ratio if four_player_ratio is None else float(four_player_ratio)
        pool = self.opponent_pool()
        specs: List[MatchSpec] = []
        for i in range(int(games)):
            use_4p = len(pool) >= 3 and rng.random() < ratio
            if use_4p:
                opps = [pool[(i + j + rng.randrange(len(pool))) % len(pool)] for j in range(3)]
                our_index = rng.randrange(4)
            else:
                opps = [pool[rng.randrange(len(pool))]]
                our_index = rng.randrange(2)
            specs.append(MatchSpec(opps, our_index, self.seed * 100003 + seed_offset * 997 + i, int(max_steps), phase))
        return specs


def build_cross_play_specs(
    opponents: Sequence[str],
    *,
    games: int,
    seed: int,
    seed_offset: int,
    max_steps: int,
    four_player_ratio: float,
    phase: str,
) -> List[MatchSpec]:
    """Deterministic held-out schedule with rotated player slots.

    The requested 4p ratio is respected across the whole schedule. The old
    implementation front-loaded one 2p game per opponent, which distorted
    mixed benchmarks when the pool was large.
    """

    pool = _available_opponents(opponents)
    if not pool:
        pool = ["heldout_random", "heldout_greedy"]
    specs: List[MatchSpec] = []
    rng = random.Random(seed + seed_offset)
    total_games = max(0, int(games))
    if float(four_player_ratio) >= 0.999 and len(pool) >= 3:
        for i in range(total_games):
            anchor = pool[i % len(pool)]
            others = [name for name in pool if name != anchor] or [anchor]
            specs.append(MatchSpec(
                opponent_names=[anchor, others[i % len(others)], others[(i + 1) % len(others)]],
                our_index=i % 4,
                seed=seed * 100003 + seed_offset * 997 + len(specs),
                max_steps=int(max_steps),
                phase=phase,
            ))
        return specs

    ratio = max(0.0, min(1.0, float(four_player_ratio)))
    target_4p = int(round(total_games * ratio)) if len(pool) >= 3 else 0
    target_2p = max(0, total_games - target_4p)
    modes = ["4p"] * target_4p + ["2p"] * target_2p
    rng.shuffle(modes)
    for i, mode in enumerate(modes):
        use_4p = mode == "4p"
        if use_4p:
            anchor = pool[i % len(pool)]
            others = [name for name in pool if name != anchor] or [anchor]
            opps = [anchor]
            for j in range(2):
                opps.append(others[(i + j) % len(others)])
            our_index = i % 4
        else:
            opps = [pool[i % len(pool)]]
            our_index = i % 2
        specs.append(MatchSpec(
            opponent_names=opps,
            our_index=our_index,
            seed=seed * 100003 + seed_offset * 997 + len(specs),
            max_steps=int(max_steps),
            phase=phase,
        ))
    return specs
