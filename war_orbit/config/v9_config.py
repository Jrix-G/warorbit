"""Configuration for the V9 adaptive planner and trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class V9Config:
    # Runtime planning.
    max_candidates: int = 22
    search_width: int = 8
    simulation_depth: int = 3
    simulation_rollouts: int = 2
    opponent_samples: int = 4
    train_search_width: int = 4
    train_simulation_depth: int = 1
    train_simulation_rollouts: int = 0
    train_opponent_samples: int = 1
    candidate_diversity: float = 1.15
    exploration_rate: float = 0.08
    uncertainty_penalty: float = 0.20
    rollout_weight: float = 0.42
    finisher_bias: float = 1.0
    min_source_ships: int = 8
    max_moves_per_plan: int = 12
    front_lock_turns: int = 24
    target_active_fronts: float = 2.0
    target_backbone_turn_frac: float = 0.15
    front_penalty_weight: float = 0.055
    front_penalty_cap: float = 0.12
    front_ok_bonus: float = 0.045
    front_partial_bonus: float = 0.025
    backbone_penalty_weight: float = 0.080
    backbone_bonus_weight: float = 0.060
    front_pressure_plan_bias: float = 0.12
    front_pressure_attack_penalty: float = 0.12
    seed: int = 9009

    # Training schedule.
    minutes: float = 60.0
    hard_timeout_minutes: float = 60.0
    pairs: int = 2
    games_per_eval: int = 2
    eval_games: int = 16
    max_steps: int = 160
    eval_max_steps: int = 220
    four_player_ratio: float = 0.80
    eval_four_player_ratio: Optional[float] = None
    sigma: float = 0.07
    learning_rate: float = 0.035
    l2: float = 0.0004
    confidence_l2: float = 0.0025
    reward_noise: float = 0.015
    train_state_perturbation: float = 0.035
    eval_every: int = 1
    benchmark_every: int = 1
    workers: int = 1
    train_only: bool = False
    benchmark_games: int = 128
    min_promotion_benchmark_games: int = 128
    benchmark_progress_every: int = 1
    benchmark_four_player_ratio: float = 0.80
    guardian_enabled: bool = True
    guardian_min_benchmark_4p: float = 0.42
    guardian_min_benchmark_backbone: float = 0.08
    guardian_max_benchmark_fronts: float = 2.70
    guardian_max_generalization_gap: float = 0.18
    export_best_on_finish: bool = True
    strict_single_target_4p: bool = False
    disable_snipe_4p: bool = False
    max_focus_targets_4p: int = 2
    stagnation_window: int = 4
    min_improvement: float = 0.015
    min_benchmark_score: float = 0.15
    max_generalization_gap: float = 0.30
    overfit_train_threshold: float = 0.80
    overfit_eval_threshold: float = 0.70
    benchmark_collapse_threshold: float = 0.12
    reset_fraction: float = 0.18
    target_winrate_low: float = 0.60
    target_winrate_high: float = 0.70

    # Opponents and persistence.
    opponent_pool_limit: int = 15
    training_opponents: List[str] = field(default_factory=lambda: [
        "random",
        "noisy_greedy",
        "starter",
        "distance",
        "sun_dodge",
        "structured",
        "orbit_stars",
        "bot_v7",
        "notebook_tactical_heuristic",
        "notebook_mdmahfuzsumon_how_my_ai_wins_space_wars",
        "notebook_pilkwang_orbit_wars_structured_baseline",
        "notebook_sigmaborov_orbit_wars_2026_starter",
        "notebook_sigmaborov_orbit_wars_2026_tactical_heuristic",
        "notebook_aminmahmoudalifayed_kronos_omega",
        "notebook_kashiwaba_orbit_wars_reinforcement_learning_tutorial",
    ])
    eval_opponents: List[str] = field(default_factory=lambda: [
        "heldout_random",
        "heldout_greedy",
        "notebook_johnjanson_lb_max_score_1000_agi_is_here",
        "notebook_romantamrazov_orbit_star_wars_lb_max_1224",
        "notebook_pascalledesma_orbitbotnext",
        "notebook_sigmaborov_lb_958_1_orbit_wars_2026_reinforce",
    ])
    benchmark_opponents: List[str] = field(default_factory=lambda: [
        "notebook_orbitbotnext",
        "notebook_distance_prioritized",
        "notebook_physics_accurate",
        "notebook_djenkivanov_orbit_wars_optimized_nearest_planet_sniper",
        "notebook_pascalledesma_orbitwork_v14",
    ])
    checkpoint: str = "evaluations/v9_robust_policy_latest.npz"
    best_checkpoint: str = "evaluations/v9_robust_policy_best.npz"
    export_checkpoint: str = "evaluations/v9_robust_policy.npz"
    log_jsonl: str = "evaluations/v9_robust_train.jsonl"

    def resolved_eval_four_player_ratio(self) -> float:
        return self.four_player_ratio if self.eval_four_player_ratio is None else self.eval_four_player_ratio

    def planning_kwargs(self) -> Dict[str, object]:
        keys = {
            "max_candidates",
            "search_width",
            "simulation_depth",
            "simulation_rollouts",
            "opponent_samples",
            "train_search_width",
            "train_simulation_depth",
            "train_simulation_rollouts",
            "train_opponent_samples",
            "candidate_diversity",
            "exploration_rate",
            "uncertainty_penalty",
            "rollout_weight",
            "finisher_bias",
            "min_source_ships",
            "max_moves_per_plan",
            "front_lock_turns",
            "target_active_fronts",
            "target_backbone_turn_frac",
            "front_penalty_weight",
            "front_penalty_cap",
            "front_ok_bonus",
            "front_partial_bonus",
            "backbone_penalty_weight",
            "backbone_bonus_weight",
            "front_pressure_plan_bias",
            "front_pressure_attack_penalty",
            "guardian_enabled",
            "guardian_min_benchmark_4p",
            "guardian_min_benchmark_backbone",
            "guardian_max_benchmark_fronts",
            "guardian_max_generalization_gap",
            "export_best_on_finish",
            "strict_single_target_4p",
            "disable_snipe_4p",
            "max_focus_targets_4p",
            "seed",
        }
        data = asdict(self)
        return {k: data[k] for k in keys}

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
