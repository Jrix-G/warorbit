"""Thompson sampling controller over training hyperparameters.

Each arm = a config delta applied on top of base V10Config for one generation.
Reward = delta of skill posterior LCB between consecutive generations under that arm.

Modeled as Gaussian-Gaussian bandit with conjugate prior:
  reward_i ~ N(mu_a, tau^2)
  mu_a ~ N(mu0, sigma0^2)
  posterior closed-form per arm.

Usage:
    controller = ThompsonController(arms=[...])
    arm = controller.sample()
    apply(arm.deltas, config)
    ... run gen ...
    controller.observe(arm, reward)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Arm:
    name: str
    deltas: Dict[str, Any]  # {config_attr: value_or_callable(old)->new}

    # Posterior state (Gaussian on reward mean).
    mu0: float = 0.0
    sigma0: float = 0.05
    tau: float = 0.04  # observation noise std
    n: int = 0
    sum_x: float = 0.0
    sum_x2: float = 0.0

    @property
    def posterior_mean(self) -> float:
        if self.n == 0:
            return self.mu0
        prec0 = 1.0 / (self.sigma0 ** 2)
        prec_obs = self.n / (self.tau ** 2)
        x_bar = self.sum_x / self.n
        return (prec0 * self.mu0 + prec_obs * x_bar) / (prec0 + prec_obs)

    @property
    def posterior_var(self) -> float:
        prec0 = 1.0 / (self.sigma0 ** 2)
        prec_obs = self.n / (self.tau ** 2)
        return 1.0 / (prec0 + prec_obs)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.posterior_mean, math.sqrt(self.posterior_var)))

    def update(self, reward: float) -> None:
        self.n += 1
        self.sum_x += float(reward)
        self.sum_x2 += float(reward) ** 2


@dataclass
class ThompsonController:
    arms: List[Arm]
    seed: int = 1234
    rng: np.random.Generator = field(init=False)
    log_path: Optional[str] = None
    last_arm: Optional[str] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def sample(self) -> Arm:
        scores = np.array([a.sample(self.rng) for a in self.arms])
        idx = int(np.argmax(scores))
        self.last_arm = self.arms[idx].name
        return self.arms[idx]

    def observe(self, arm: Arm, reward: float) -> None:
        arm.update(reward)
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "arm": arm.name,
                    "reward": float(reward),
                    "n": arm.n,
                    "post_mean": arm.posterior_mean,
                    "post_std": math.sqrt(arm.posterior_var),
                }, sort_keys=True) + "\n")

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            a.name: {
                "n": a.n,
                "mean": a.posterior_mean,
                "std": math.sqrt(a.posterior_var),
            }
            for a in self.arms
        }


def apply_arm(config, arm: Arm) -> Dict[str, Any]:
    """Apply arm.deltas to config in-place. Returns previous values for restoration."""
    prev: Dict[str, Any] = {}
    for key, val in arm.deltas.items():
        if not hasattr(config, key):
            continue
        prev[key] = getattr(config, key)
        new_val = val(prev[key]) if callable(val) else val
        setattr(config, key, new_val)
    return prev


def restore(config, prev: Dict[str, Any]) -> None:
    for k, v in prev.items():
        setattr(config, k, v)


def default_v10_arms() -> List[Arm]:
    """Sensible default arm grid for V10 training.

    Designed for the observed pathology (bb stuck ~0.12, gate at 0.18, sel oscillating).
    Each arm targets a different hypothesis about the bottleneck.
    """
    return [
        Arm("baseline", {}),
        Arm("explore_up", {"exploration_rate": lambda x: min(0.25, float(x) * 1.6)}),
        Arm("explore_down", {"exploration_rate": lambda x: max(0.03, float(x) * 0.5)}),
        Arm("sigma_up", {"sigma": lambda x: min(0.15, float(x) * 1.4)}),
        Arm("sigma_down", {"sigma": lambda x: max(0.025, float(x) * 0.6)}),
        Arm("lr_up", {"learning_rate": lambda x: min(0.08, float(x) * 1.5)}),
        Arm("lr_down", {"learning_rate": lambda x: max(0.010, float(x) * 0.5)}),
        Arm("bb_target_down", {"target_backbone_turn_frac": lambda x: max(0.10, float(x) - 0.04),
                               "guardian_min_benchmark_backbone": lambda x: max(0.08, float(x) - 0.04)}),
        Arm("front_relax", {"front_penalty_weight": lambda x: max(0.04, float(x) * 0.6),
                             "front_penalty_cap": lambda x: max(0.10, float(x) * 0.6)}),
        Arm("front_strict", {"front_penalty_weight": lambda x: min(0.20, float(x) * 1.5),
                              "front_overlap_penalty_weight": lambda x: min(0.30, float(x) * 1.3)}),
        Arm("main_front_mass_up", {"target_main_front_ship_share_4p": lambda x: min(0.52, float(x) + 0.04),
                                   "main_front_mass_bonus": lambda x: min(0.38, float(x) + 0.06)}),
        Arm("scatter_down", {"scatter_penalty_weight_4p": lambda x: min(0.34, float(x) + 0.06),
                             "max_focus_targets_4p": lambda x: max(1, min(int(x), 2))}),
        Arm("conversion_t120_up", {"concentration_phase_end_4p": lambda x: min(150, int(x) + 18),
                                   "midgame_min_capture_send_4p": lambda x: min(34, int(x) + 4)}),
        Arm("noise_down", {"reward_noise": lambda x: max(0.002, float(x) * 0.4),
                            "train_state_perturbation": lambda x: max(0.010, float(x) * 0.6)}),
        Arm("diversity_up", {"candidate_diversity": lambda x: min(1.80, float(x) * 1.25)}),
    ]
