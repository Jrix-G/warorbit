"""Single-run self-improving trainer for V9."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..agents.v9.adaptation import AdaptationController, TrainingMetrics
from ..agents.v9.policy import V9Weights, save_checkpoint
from ..config.v9_config import V9Config
from ..optimization.tuning import apply_adaptation
from .curriculum import CurriculumScheduler, build_cross_play_specs, build_role_pools
from .self_play import DeadlineExceeded, evaluate_weights, summarise_results


V9_4P_LOG_TARGETS = {
    "xfer": 0.30,
    "bb": 0.15,
    "lock": 0.90,
    "fronts": 2.00,
}


def _rank_shape(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    shaped = order.astype(np.float32) / max(1, len(values) - 1)
    return shaped - float(np.mean(shaped))


def _save_train_checkpoint(path: str, weights: V9Weights, *, best_score: float, generation: int, meta: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(p),
        flat=weights.flatten(),
        best_score=np.asarray(best_score, dtype=np.float32),
        generation=np.asarray(generation, dtype=np.int32),
        meta_json=np.asarray(json.dumps(meta, sort_keys=True, default=str)),
    )


def _load_train_checkpoint(path: str) -> Tuple[V9Weights, float, int]:
    data = np.load(str(path), allow_pickle=False)
    weights = V9Weights.from_flat(data["flat"])
    best_score = float(data["best_score"]) if "best_score" in data else -1.0
    generation = int(data["generation"]) if "generation" in data else 0
    return weights, best_score, generation


def _load_opponents(config: V9Config):
    opponents = list(config.training_opponents)
    try:
        from opponents import ZOO, training_pool

        for name in training_pool(config.opponent_pool_limit):
            if name in ZOO and name not in opponents:
                opponents.append(name)
    except Exception:
        pass
    if Path(config.best_checkpoint).exists():
        opponents.append(f"v9_checkpoint:{config.best_checkpoint}")
    return opponents


def _regularized_train_score(summary: Dict[str, float], flat: np.ndarray, defaults: np.ndarray,
                             config: V9Config, rng: np.random.Generator) -> float:
    score = float(summary["mean"])
    confidence = float(np.linalg.norm(flat - defaults) / max(1.0, np.sqrt(flat.size)))
    low_entropy = max(0.0, 0.45 - float(summary.get("plan_entropy", 0.0)))
    repeated = max(0.0, float(summary.get("dominant_plan_frac", 1.0)) - 0.72)
    score -= float(config.confidence_l2) * confidence
    score -= 0.08 * low_entropy + 0.06 * repeated
    score += _front_pressure_adjustment(summary, config)
    if config.reward_noise > 0:
        score += float(rng.normal(0.0, config.reward_noise))
    return score


def _front_pressure_adjustment(summary: Dict[str, float], config: V9Config) -> float:
    if int(summary.get("n_4p", 0)) <= 0:
        return 0.0
    target_fronts = float(getattr(config, "target_active_fronts", 2.0))
    target_backbone = max(0.01, float(getattr(config, "target_backbone_turn_frac", 0.15)))
    fronts = float(summary.get("active_front_avg", 0.0))
    xfer = float(summary.get("transfer_move_frac", 0.0))
    backbone = float(summary.get("backbone_turn_frac", 0.0))
    lock = float(summary.get("front_lock_turn_frac", 0.0))

    excess = max(0.0, fronts - target_fronts)
    penalty = min(float(getattr(config, "front_penalty_cap", 0.12)), excess * float(getattr(config, "front_penalty_weight", 0.055)))
    backbone_shortfall = max(0.0, (target_backbone - backbone) / target_backbone)
    penalty += float(getattr(config, "backbone_penalty_weight", 0.08)) * backbone_shortfall

    backbone_score = min(1.0, backbone / target_backbone)
    front_score = min(1.0, max(0.0, 3.5 - fronts) / max(0.1, 3.5 - target_fronts))
    if backbone_score < 0.75:
        front_score *= 0.35
    progress = (
        min(1.0, xfer / 0.30)
        + backbone_score
        + min(1.0, lock / 0.90)
        + min(front_score, backbone_score)
    ) / 4.0
    bonus = float(getattr(config, "front_partial_bonus", 0.025)) * progress
    if backbone >= target_backbone and fronts <= target_fronts:
        bonus += float(getattr(config, "backbone_bonus_weight", 0.06))
    if xfer >= 0.30 and backbone >= target_backbone and lock >= 0.90 and fronts <= target_fronts:
        bonus += float(getattr(config, "front_ok_bonus", 0.045))
    return bonus - penalty


def _selection_score(eval_summary: Dict[str, float], benchmark_summary: Dict[str, float]) -> float:
    return min(float(eval_summary.get("mean", 0.0)), float(benchmark_summary.get("mean", 0.0)))


def _should_promote(selection_score: float, best_score: float, eval_summary: Dict[str, float],
                    benchmark_summary: Dict[str, float], config: V9Config) -> bool:
    eval_score = float(eval_summary.get("mean", 0.0))
    benchmark_score = float(benchmark_summary.get("mean", 0.0))
    benchmark_games = int(benchmark_summary.get("n_2p", 0)) + int(benchmark_summary.get("n_4p", 0))
    gap = eval_score - benchmark_score
    return (
        selection_score >= best_score + config.min_improvement
        and benchmark_score >= config.min_benchmark_score
        and benchmark_games >= config.min_promotion_benchmark_games
        and gap <= config.max_generalization_gap
    )


def _is_generalization_failure(train_summary: Dict[str, float], eval_summary: Dict[str, float],
                               benchmark_summary: Dict[str, float], config: V9Config) -> bool:
    train_score = float(train_summary.get("mean", 0.0))
    eval_score = float(eval_summary.get("mean", 0.0))
    benchmark_score = float(benchmark_summary.get("mean", 0.0))
    return (
        train_score >= config.overfit_train_threshold
        and eval_score >= config.overfit_eval_threshold
        and (
            benchmark_score <= config.benchmark_collapse_threshold
            or eval_score - benchmark_score > config.max_generalization_gap
        )
    )


def _partial_reset(flat: np.ndarray, defaults: np.ndarray, rng: np.random.Generator, fraction: float) -> np.ndarray:
    out = flat.copy()
    mask = rng.random(out.size) < max(0.0, min(1.0, float(fraction)))
    if np.any(mask):
        out[mask] = defaults[mask] + 0.25 * (out[mask] - defaults[mask])
        out[mask] += rng.normal(0.0, 0.01, size=int(np.sum(mask))).astype(np.float32)
    return out.astype(np.float32)


def _four_p_diag(summary: Dict[str, float]) -> Dict[str, float | str | bool]:
    xfer = float(summary.get("transfer_move_frac", 0.0))
    bb = float(summary.get("backbone_turn_frac", 0.0))
    lock = float(summary.get("front_lock_turn_frac", 0.0))
    fronts = float(summary.get("active_front_avg", 0.0))
    ok = (
        xfer >= V9_4P_LOG_TARGETS["xfer"]
        and bb >= V9_4P_LOG_TARGETS["bb"]
        and lock >= V9_4P_LOG_TARGETS["lock"]
        and fronts <= V9_4P_LOG_TARGETS["fronts"]
    )
    return {
        "xfer": xfer,
        "bb": bb,
        "lock": lock,
        "fronts": fronts,
        "ok": bool(ok),
        "status": "OK" if ok else "WARN",
        "target_xfer_min": V9_4P_LOG_TARGETS["xfer"],
        "target_backbone_min": V9_4P_LOG_TARGETS["bb"],
        "target_lock_min": V9_4P_LOG_TARGETS["lock"],
        "target_fronts_max": V9_4P_LOG_TARGETS["fronts"],
    }


def _format_four_p_diag(summary: Dict[str, float]) -> str:
    diag = _four_p_diag(summary)
    return (
        f"4pdiag={diag['status']} "
        f"xfer={diag['xfer']:.2f}/{V9_4P_LOG_TARGETS['xfer']:.2f}+ "
        f"bb={diag['bb']:.2f}/{V9_4P_LOG_TARGETS['bb']:.2f}+ "
        f"lock={diag['lock']:.2f}/{V9_4P_LOG_TARGETS['lock']:.2f}+ "
        f"fronts={diag['fronts']:.1f}/{V9_4P_LOG_TARGETS['fronts']:.1f}-"
    )


class V9Trainer:
    def __init__(self, config: V9Config, *, resume: bool = True):
        self.config = config
        self.resume = bool(resume)
        if self.resume and Path(config.checkpoint).exists():
            self.weights, self.best_score, self.generation = _load_train_checkpoint(config.checkpoint)
            print(f"Resumed {config.checkpoint} generation={self.generation} best_score={self.best_score:.4f}", flush=True)
        else:
            self.weights = V9Weights.defaults()
            self.best_score = -1.0
            self.generation = 0
            print(f"Starting V9 fresh dim={self.weights.flatten().size}", flush=True)

        self.controller = AdaptationController(
            window=config.stagnation_window,
            min_improvement=config.min_improvement,
            exploration_rate=config.exploration_rate,
            candidate_diversity=config.candidate_diversity,
            rollout_weight=config.rollout_weight,
            uncertainty_penalty=config.uncertainty_penalty,
            finisher_bias=config.finisher_bias,
        )
        self.velocity = np.zeros_like(self.weights.flatten())
        self.rng = np.random.default_rng(config.seed + self.generation)
        self.log_path = Path(config.log_jsonl)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict[str, float]:
        start = time.time()
        budget_minutes = min(float(self.config.minutes), float(self.config.hard_timeout_minutes))
        deadline = start + max(1.0, budget_minutes * 60.0)
        latest_summary: Dict[str, float] = {"mean": 0.0, "wr_2p": 0.0, "wr_4p": 0.0, "stop_reason": "running"}
        stop_reason = "completed"
        role_pools = build_role_pools(self.config)
        opponents = role_pools["train"]
        eval_opponents = role_pools["eval"]
        benchmark_opponents = role_pools["benchmark"]
        scheduler = CurriculumScheduler(opponents, four_player_ratio=self.config.four_player_ratio, seed=self.config.seed)
        params = self.weights.flatten()
        default_params = V9Weights.defaults().flatten()

        print(
            f"V9 train start minutes={self.config.minutes:.2f} pairs={self.config.pairs} "
            f"hard_timeout={budget_minutes:.2f} "
            f"games_per_eval={self.config.games_per_eval} eval_games={self.config.eval_games} "
            f"train_opponents={len(opponents)} eval_opponents={len(eval_opponents)} "
            f"benchmark_opponents={len(benchmark_opponents)} workers={self.config.workers} "
            f"train_only={int(self.config.train_only)}",
            flush=True,
        )
        print(
            "V9 4p diag targets "
            f"xfer>={V9_4P_LOG_TARGETS['xfer']:.2f} "
            f"bb>={V9_4P_LOG_TARGETS['bb']:.2f} "
            f"lock>={V9_4P_LOG_TARGETS['lock']:.2f} "
            f"fronts<={V9_4P_LOG_TARGETS['fronts']:.1f}",
            flush=True,
        )
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": "started",
                "minutes": self.config.minutes,
                "hard_timeout_minutes": budget_minutes,
                "pairs": self.config.pairs,
                "games_per_eval": self.config.games_per_eval,
                "eval_games": self.config.eval_games,
                "workers": self.config.workers,
                "train_only": self.config.train_only,
                "train_opponents": opponents,
                "eval_opponents": eval_opponents,
                "benchmark_opponents": benchmark_opponents,
                "v9_4p_diag_targets": V9_4P_LOG_TARGETS,
            }, sort_keys=True) + "\n")

        pool = None
        if int(self.config.workers) > 1:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(processes=int(self.config.workers))
            print(f"V9 worker pool started workers={self.config.workers}", flush=True)

        try:
            while time.time() < deadline:
                self.generation += 1
                active_config = apply_adaptation(self.config, self.controller)
                active_config.checkpoint = self.config.checkpoint
                active_config.best_checkpoint = self.config.best_checkpoint
                active_config.export_checkpoint = self.config.export_checkpoint
                active_config.log_jsonl = self.config.log_jsonl
                active_config.search_width = self.config.train_search_width
                active_config.simulation_depth = self.config.train_simulation_depth
                active_config.simulation_rollouts = self.config.train_simulation_rollouts
                active_config.opponent_samples = self.config.train_opponent_samples

                eps = self.rng.normal(size=(self.config.pairs, params.size)).astype(np.float32)
                rewards_pos = []
                rewards_neg = []
                all_train_results = []

                print(f"gen={self.generation:04d} train_pair_start", flush=True)

                for i in range(self.config.pairs):
                    if time.time() >= deadline:
                        raise DeadlineExceeded("global training deadline reached")
                    print(f"gen={self.generation:04d} pair={i+1}/{self.config.pairs}", flush=True)
                    specs = scheduler.build(
                        self.config.games_per_eval,
                        seed_offset=self.generation * 100 + i,
                        max_steps=self.config.max_steps,
                        four_player_ratio=self.config.four_player_ratio,
                        phase="train",
                    )
                    pos_flat = (params + active_config.sigma * eps[i]).astype(np.float32)
                    neg_flat = (params - active_config.sigma * eps[i]).astype(np.float32)
                    pos_weights = V9Weights.from_flat(pos_flat)
                    neg_weights = V9Weights.from_flat(neg_flat)
                    pos_results = evaluate_weights(pos_weights, active_config, specs, deadline=deadline, pool=pool)
                    neg_results = evaluate_weights(neg_weights, active_config, specs, deadline=deadline, pool=pool)
                    pos_summary = summarise_results(pos_results)
                    neg_summary = summarise_results(neg_results)
                    rewards_pos.append(_regularized_train_score(pos_summary, pos_flat, default_params, active_config, self.rng))
                    rewards_neg.append(_regularized_train_score(neg_summary, neg_flat, default_params, active_config, self.rng))
                    all_train_results.extend(pos_results)
                    all_train_results.extend(neg_results)

                r_pos = np.asarray(rewards_pos, dtype=np.float32)
                r_neg = np.asarray(rewards_neg, dtype=np.float32)
                shaped = _rank_shape(np.concatenate([r_pos, r_neg]))
                grad = np.mean((shaped[: self.config.pairs] - shaped[self.config.pairs :])[:, None] * eps, axis=0)
                grad /= max(1e-6, active_config.sigma)
                grad -= active_config.l2 * (params - default_params)
                grad_norm = float(np.linalg.norm(grad))
                if grad_norm > 12.0:
                    grad *= 12.0 / grad_norm
                self.velocity = 0.88 * self.velocity + 0.12 * grad
                params = (params + active_config.learning_rate * self.velocity).astype(np.float32)
                self.weights = V9Weights.from_flat(params)

                train_summary = summarise_results(all_train_results)
                eval_summary = train_summary
                benchmark_summary = {"mean": 0.0, "wr_2p": 0.0, "wr_4p": 0.0, "n_2p": 0, "n_4p": 0}
                selection_score = float(train_summary["mean"]) if self.config.train_only else 0.0
                promoted = False
                if self.config.train_only:
                    scheduler.update(float(train_summary["mean"]))
                elif self.config.eval_every > 0 and (self.generation == 1 or self.generation % self.config.eval_every == 0):
                    print(f"gen={self.generation:04d} eval_start", flush=True)
                    eval_specs = build_cross_play_specs(
                        eval_opponents,
                        games=self.config.eval_games,
                        seed=self.config.seed,
                        seed_offset=50000,
                        max_steps=self.config.eval_max_steps,
                        four_player_ratio=self.config.resolved_eval_four_player_ratio(),
                        phase="eval",
                    )
                    eval_results = evaluate_weights(self.weights, active_config, eval_specs, deadline=deadline, pool=pool)
                    eval_summary = summarise_results(eval_results)
                    run_benchmark = self.config.benchmark_every > 0 and (
                        self.generation == 1 or self.generation % self.config.benchmark_every == 0
                    )
                    if run_benchmark:
                        print(f"gen={self.generation:04d} benchmark_start games={self.config.benchmark_games}", flush=True)
                        benchmark_specs = build_cross_play_specs(
                            benchmark_opponents,
                            games=self.config.benchmark_games,
                            seed=self.config.seed,
                            seed_offset=80000,
                            max_steps=self.config.eval_max_steps,
                            four_player_ratio=self.config.benchmark_four_player_ratio,
                            phase="benchmark",
                        )
                        benchmark_results = evaluate_weights(
                            self.weights,
                            active_config,
                            benchmark_specs,
                            deadline=deadline,
                            progress_label=f"gen={self.generation:04d} benchmark",
                            progress_every=max(1, int(self.config.benchmark_progress_every)),
                            pool=pool,
                        )
                        benchmark_summary = summarise_results(benchmark_results)
                    selection_score = _selection_score(eval_summary, benchmark_summary)
                    if _should_promote(selection_score, self.best_score, eval_summary, benchmark_summary, self.config):
                        self.best_score = float(selection_score)
                        promoted = True
                        _save_train_checkpoint(
                            self.config.best_checkpoint,
                            self.weights,
                            best_score=self.best_score,
                            generation=self.generation,
                            meta={"config": self.config.to_dict(), "eval": eval_summary, "benchmark": benchmark_summary},
                        )
                        save_checkpoint(
                            self.config.export_checkpoint,
                            self.weights,
                            meta={"generation": self.generation, "score": self.best_score, "eval": eval_summary, "benchmark": benchmark_summary},
                        )
                        past_name = f"v9_checkpoint:{self.config.best_checkpoint}"
                        if past_name not in scheduler.base_opponents:
                            scheduler.base_opponents.append(past_name)
                    if _is_generalization_failure(train_summary, eval_summary, benchmark_summary, self.config):
                        print(
                            f"gen={self.generation:04d} generalization_failure "
                            f"train={train_summary['mean']:.3f} eval={eval_summary['mean']:.3f} "
                            f"benchmark={benchmark_summary['mean']:.3f}",
                            flush=True,
                        )
                        params = _partial_reset(params, default_params, self.rng, self.config.reset_fraction)
                        self.weights = V9Weights.from_flat(params)
                        self.controller.observe_generalization_failure()
                    scheduler.update(float(eval_summary["mean"]))

                _save_train_checkpoint(
                    self.config.checkpoint,
                    self.weights,
                    best_score=self.best_score,
                    generation=self.generation,
                    meta={"config": self.config.to_dict(), "eval": eval_summary, "benchmark": benchmark_summary},
                )

                metrics = TrainingMetrics(
                    generation=self.generation,
                    train_score=float(train_summary["mean"]),
                    eval_score=float(selection_score),
                    eval_2p=float(eval_summary["wr_2p"]),
                    eval_4p=float(eval_summary["wr_4p"]),
                    promoted=promoted,
                )
                state = self.controller.observe(metrics)

                elapsed = (time.time() - start) / 60.0
                latest_summary = {
                    "generation": self.generation,
                    "train_mean": float(train_summary["mean"]),
                    "train_2p": float(train_summary["wr_2p"]),
                    "train_4p": float(train_summary["wr_4p"]),
                    "eval_mean": float(eval_summary["mean"]),
                    "eval_2p": float(eval_summary["wr_2p"]),
                    "eval_4p": float(eval_summary["wr_4p"]),
                    "benchmark_mean": float(benchmark_summary["mean"]),
                    "benchmark_2p": float(benchmark_summary["wr_2p"]),
                    "benchmark_4p": float(benchmark_summary["wr_4p"]),
                    "selection_score": float(selection_score),
                    "generalization_gap": float(eval_summary["mean"] - benchmark_summary["mean"]),
                    "best": float(self.best_score),
                    "grad_norm": grad_norm,
                    "promoted": promoted,
                    "elapsed_min": elapsed,
                    "explore": state.exploration_rate,
                    "diversity": state.candidate_diversity,
                    "rollout_weight": state.rollout_weight,
                    "train_transfer_move_frac": float(train_summary.get("transfer_move_frac", 0.0)),
                    "train_transfer_ship_frac": float(train_summary.get("transfer_ship_frac", 0.0)),
                    "train_backbone_turn_frac": float(train_summary.get("backbone_turn_frac", 0.0)),
                    "train_front_lock_turn_frac": float(train_summary.get("front_lock_turn_frac", 0.0)),
                    "train_staged_finisher_turn_frac": float(train_summary.get("staged_finisher_turn_frac", 0.0)),
                    "train_active_front_avg": float(train_summary.get("active_front_avg", 0.0)),
                    "train_front_pressure_adjustment": float(_front_pressure_adjustment(train_summary, self.config)),
                    "eval_transfer_move_frac": float(eval_summary.get("transfer_move_frac", 0.0)),
                    "eval_backbone_turn_frac": float(eval_summary.get("backbone_turn_frac", 0.0)),
                    "eval_front_lock_turn_frac": float(eval_summary.get("front_lock_turn_frac", 0.0)),
                    "benchmark_transfer_move_frac": float(benchmark_summary.get("transfer_move_frac", 0.0)),
                    "benchmark_backbone_turn_frac": float(benchmark_summary.get("backbone_turn_frac", 0.0)),
                    "benchmark_front_lock_turn_frac": float(benchmark_summary.get("front_lock_turn_frac", 0.0)),
                    "train_4p_diag": _four_p_diag(train_summary),
                    "eval_4p_diag": _four_p_diag(eval_summary),
                    "benchmark_4p_diag": _four_p_diag(benchmark_summary),
                    "stop_reason": "running",
                }
                line = (
                    f"gen={self.generation:04d} "
                    f"train={train_summary['mean']:.3f} (2p {train_summary['wr_2p']:.3f}/{train_summary['n_2p']} "
                    f"4p {train_summary['wr_4p']:.3f}/{train_summary['n_4p']}) "
                    f"eval={eval_summary['mean']:.3f} (2p {eval_summary['wr_2p']:.3f} 4p {eval_summary['wr_4p']:.3f}) "
                    f"bench={benchmark_summary['mean']:.3f} "
                    f"best={self.best_score:.3f} grad={grad_norm:.2f} promo={int(promoted)} "
                    f"explore={state.exploration_rate:.2f} div={state.candidate_diversity:.2f} "
                    f"{_format_four_p_diag(train_summary)} "
                    f"front_adj={_front_pressure_adjustment(train_summary, self.config):+.3f} "
                    f"elapsed_min={elapsed:.1f}"
                )
                print(line, flush=True)
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(latest_summary, sort_keys=True) + "\n")
        except DeadlineExceeded:
            stop_reason = "timeout"
            print(f"V9 timeout reached after {budget_minutes:.2f} minutes; saving latest checkpoint", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        latest_summary["stop_reason"] = stop_reason
        latest_summary["elapsed_min"] = (time.time() - start) / 60.0
        _save_train_checkpoint(
            self.config.checkpoint,
            self.weights,
            best_score=self.best_score,
            generation=self.generation,
            meta={"config": self.config.to_dict(), "latest": latest_summary, "stop_reason": stop_reason},
        )
        save_checkpoint(
            self.config.export_checkpoint,
            self.weights,
            meta={"generation": self.generation, "score": self.best_score, "latest": latest_summary},
        )
        print(
            f"Saved latest={self.config.checkpoint} best={self.config.best_checkpoint} "
            f"bot_checkpoint={self.config.export_checkpoint} stop_reason={stop_reason}",
            flush=True,
        )
        return latest_summary
