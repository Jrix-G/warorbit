"""GPU vectorized training: N parallel env workers + GPU inference server + GPU trainer."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGE_DIR = Path(__file__).resolve().parents[2]

from neural_network.src.model import ModelConfig, NeuralNetworkModel, load_compatible_state_dict, count_parameters
from neural_network.src.notebook_4p_training import _infer_input_dim
from neural_network.src.storage import load_checkpoint, save_checkpoint
from neural_network.src.utils import ensure_dir, save_json
from neural_network.src.population_4p_training import configure_run_logging

from neural_network_gpu.src.vec_worker import worker_fn
from neural_network_gpu.src.inference_server import inference_server_fn
from neural_network_gpu.src.gpu_trainer import train_on_episodes


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(path: Path, msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    with path.open("a") as f:
        f.write(line + "\n")


def _run_tag() -> str:
    return datetime.now(timezone.utc).strftime("gpu_%Y%m%d_%H%M%S")


def _build_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        # Game
        "game_engine": "official_fast",
        "official_fast_c_accel": True,
        "max_turns": 100,
        "max_actions_per_turn": 4,
        "min_expand_attack_ships": 6,
        "send_ratios": [0.45, 0.65, 0.85],
        "policy_prior_strength": 1.30,
        # Encoder
        "max_planets": 64,
        "max_fleets": 128,
        "max_players": 4,
        "board_scale": 100.0,
        "ship_scale": 2000.0,
        "production_scale": 10.0,
        "radius_scale": 10.0,
        "horizon_scale": 100.0,
        "planet_id_scale": 64.0,
        # Model
        "hidden_dim": 320,
        # Training
        "learning_rate": 0.00012,
        "gamma": 0.99,
        "value_loss_coef": 0.25,
        "entropy_coef_start": 0.065,
        "baseline_momentum": 0.10,
        "max_grad_norm": 1.0,
        # Activity shaping
        "dense_reward_enabled": True,
        "dense_planet_coef": 0.05,
        "dense_production_coef": 0.04,
        "dense_ship_share_coef": 0.14,
        "dense_score_coef": 0.10,
        "dense_survival_coef": 0.05,
        "dense_reward_clip": 0.25,
        "train_target_do_nothing_rate": 0.12,
        "train_noop_penalty_coef": 1.5,
        "train_action_bonus_coef": 0.28,
        "train_ships_sent_bonus_coef": 0.18,
        "train_activity_reward_clip": 0.55,
        # Temperature
        "temperature_start": 1.05,
        "temperature_end": 0.18,
        # Pool
        "stage1_pool": ["random", "greedy", "starter", "starter", "starter", "distance"],
        # GPU
        "n_workers": args.workers,
        "train_every": args.train_every,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "max_batch_size": args.batch_size,
        "device": args.device,
        # Run
        "duration_minutes": args.duration_minutes,
        "target_winrate": args.target_winrate,
        "seed": 42,
    }


def _evaluate(
    model: NeuralNetworkModel,
    config: Dict[str, Any],
    device: torch.device,
    episodes: int,
    seed_offset: int,
    pool: List[str],
    n_players: int = 2,
) -> Dict[str, Any]:
    from neural_network.src.notebook_4p_training import (
        _build_agents_n, run_match, _action_summary,
    )
    from neural_network.src.notebook_4p_training import _rank_from_scores
    from neural_network.src.population_4p_training import _composite_score

    model.eval()
    wins = []
    do_nothing_rates = []

    eval_cfg = dict(config)
    eval_cfg["temperature_end"] = 0.0

    for i in range(episodes):
        seed = seed_offset + i
        our_index = i % n_players
        log_probs: List = []
        action_records: List = []
        agents, log_probs, action_records, _ = _build_agents_n(
            model, eval_cfg, seed, our_index, n_players,
            temperature=0.0, pool=pool, explore=False,
        )
        result = run_match(
            agents, seed=seed, n_players=n_players,
            max_steps=int(config.get("max_turns", 100)),
            stop_player=our_index,
            game_engine=str(config.get("game_engine", "official_fast")),
            use_c_accel=bool(config.get("official_fast_c_accel", True)),
        )
        winner = int(result.get("winner", -1))
        wins.append(float(winner == our_index))
        metrics = _action_summary(action_records)
        do_nothing_rates.append(float(metrics.get("do_nothing_rate", 1.0)))

    winrate = float(np.mean(wins))
    return {
        "winrate": winrate,
        "eval_do_nothing_rate": float(np.mean(do_nothing_rates)),
        "eval_episodes": episodes,
    }


def main() -> None:
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-minutes", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--train-every", type=int, default=64, help="Train after N episodes")
    parser.add_argument("--eval-every", type=int, default=256, help="Eval after N episodes")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-winrate", type=float, default=0.85)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_name = args.run_name or _run_tag()
    cfg = _build_config(args)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    run_dir = PACKAGE_DIR / "runs" / run_name
    ensure_dir(run_dir)
    log_path = run_dir / "gpu_train.log"
    checkpoint_path = run_dir / "best.npz"
    latest_path = run_dir / "latest.npz"

    save_json(run_dir / "config.json", cfg)
    _log(log_path, f"run_start name={run_name} device={device} workers={cfg['n_workers']} target={cfg['target_winrate']}")

    # Build model
    model = NeuralNetworkModel(ModelConfig(
        input_dim=_infer_input_dim(cfg),
        hidden_dim=int(cfg["hidden_dim"]),
    ))
    if args.resume_checkpoint and Path(args.resume_checkpoint).exists():
        state, meta = load_checkpoint(args.resume_checkpoint)
        load_compatible_state_dict(model, state)
        _log(log_path, f"resumed from {args.resume_checkpoint}")
    model = model.to(device)
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    _log(log_path, f"params={count_parameters(model)} lr={cfg['learning_rate']}")

    # Queues
    obs_queue: mp.Queue = mp.Queue(maxsize=cfg["n_workers"] * 8)
    action_queues: Dict[int, mp.Queue] = {i: mp.Queue(maxsize=8) for i in range(cfg["n_workers"])}
    result_queue: mp.Queue = mp.Queue(maxsize=cfg["n_workers"] * 4)
    model_update_queue: mp.Queue = mp.Queue(maxsize=2)
    stop_event: mp.Event = mp.Event()

    pool = list(cfg["stage1_pool"])
    n_players = 2

    # Start inference server
    initial_state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    inf_proc = mp.Process(
        target=inference_server_fn,
        args=(initial_state, cfg, obs_queue, action_queues, model_update_queue, stop_event),
        kwargs={"device_str": str(device), "max_batch_size": cfg["max_batch_size"]},
        daemon=True,
    )
    inf_proc.start()
    _log(log_path, "inference_server started")

    # Start env workers
    worker_procs = []
    for wid in range(cfg["n_workers"]):
        p = mp.Process(
            target=worker_fn,
            args=(
                wid, cfg, pool, n_players,
                obs_queue, action_queues[wid],
                result_queue, stop_event,
                int(cfg["seed"]) + wid * 1000,
            ),
            daemon=True,
        )
        p.start()
        worker_procs.append(p)
    _log(log_path, f"{cfg['n_workers']} workers started")

    started_at = time.time()
    deadline = started_at + float(cfg["duration_minutes"]) * 60.0

    baseline = 0.0
    best_winrate = -1.0
    total_episodes = 0
    last_eval_episode = 0
    pending_episodes: List[Dict[str, Any]] = []

    try:
        while time.time() < deadline:
            # Collect episode results
            try:
                ep = result_queue.get(timeout=1.0)
                pending_episodes.append(ep)
                total_episodes += 1
            except Empty:
                continue

            # Train every N episodes
            if len(pending_episodes) >= cfg["train_every"]:
                baseline, metrics = train_on_episodes(
                    model, optimizer, pending_episodes, cfg, device,
                    baseline, float(cfg["baseline_momentum"]),
                )
                elapsed = (time.time() - started_at) / 60.0
                wins = [ep["win"] for ep in pending_episodes]
                _log(
                    log_path,
                    f"train episodes={total_episodes} elapsed={elapsed:.1f}m "
                    f"winrate={np.mean(wins):.3f} "
                    f"policy_loss={metrics.get('policy_loss', 0):.3f} "
                    f"entropy={metrics.get('entropy', 0):.3f} "
                    f"baseline={baseline:.3f}",
                )
                # Push updated weights to inference server
                new_state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
                try:
                    model_update_queue.put_nowait(new_state)
                except Exception:
                    pass

                save_checkpoint(latest_path, {k: v.cpu().numpy() for k, v in model.state_dict().items()}, {"winrate": best_winrate})
                pending_episodes = []

            # Eval every N episodes
            if total_episodes - last_eval_episode >= cfg["eval_every"]:
                last_eval_episode = total_episodes
                eval_result = _evaluate(
                    model, cfg, device,
                    episodes=cfg["eval_episodes"],
                    seed_offset=total_episodes + 900000,
                    pool=pool,
                    n_players=n_players,
                )
                elapsed = (time.time() - started_at) / 60.0
                _log(
                    log_path,
                    f"eval episodes={total_episodes} elapsed={elapsed:.1f}m "
                    f"winrate={eval_result['winrate']:.3f} "
                    f"noop={eval_result['eval_do_nothing_rate']:.3f} "
                    f"best={best_winrate:.3f} target={cfg['target_winrate']}",
                )
                if eval_result["winrate"] > best_winrate:
                    best_winrate = eval_result["winrate"]
                    save_checkpoint(checkpoint_path, {k: v.cpu().numpy() for k, v in model.state_dict().items()}, eval_result)
                    _log(log_path, f"new best winrate={best_winrate:.4f} saved to {checkpoint_path}")
                if best_winrate >= cfg["target_winrate"]:
                    _log(log_path, f"TARGET REACHED winrate={best_winrate:.4f}")
                    break

    finally:
        stop_event.set()
        for p in worker_procs:
            p.join(timeout=5)
        inf_proc.join(timeout=5)

    elapsed = (time.time() - started_at) / 60.0
    result = {
        "run_name": run_name,
        "elapsed_minutes": elapsed,
        "total_episodes": total_episodes,
        "best_winrate": best_winrate,
        "target_winrate": cfg["target_winrate"],
        "target_reached": best_winrate >= cfg["target_winrate"],
        "best_checkpoint": str(checkpoint_path) if checkpoint_path.exists() else "",
    }
    save_json(run_dir / "result.json", result)
    _log(log_path, f"run_done best_winrate={best_winrate:.4f} episodes={total_episodes}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
