from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_network.src.notebook_4p_training import run_notebook_4p_training
from neural_network.src.utils import ensure_dir, load_json


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    duration_minutes = float(os.environ.get("WARORBIT_SMOKE_MINUTES", "20.0"))
    eval_episodes = int(os.environ.get("WARORBIT_SMOKE_EVAL_EPISODES", "32"))
    eval_slices = int(os.environ.get("WARORBIT_SMOKE_EVAL_SLICES", "20"))

    print(f"Launching {duration_minutes:g}-minute 4P smoke test...", flush=True)
    cfg = load_json(ROOT / "configs" / "default_config.json")
    cfg["train_steps"] = max(1, int(duration_minutes * 10))
    cfg["eval_episodes"] = eval_episodes
    cfg["benchmark_games"] = eval_episodes
    cfg["eval_every"] = max(1, int(cfg["train_steps"] // max(1, eval_slices)))
    cfg["max_wall_seconds"] = duration_minutes * 60.0
    cfg["worker_train_steps"] = 4
    cfg["train_notebook_opponents"] = 3
    cfg["temperature_start"] = float(cfg.get("temperature_start", 1.2))
    cfg["temperature_end"] = float(cfg.get("temperature_end", 0.35))
    cfg["log_dir"] = str((ROOT / "logs").resolve())
    cfg["checkpoint_dir"] = str((ROOT / "checkpoints").resolve())
    cfg["candidate_checkpoint"] = str((ROOT / "checkpoints" / "candidate.npz").resolve())
    cfg["best_checkpoint"] = str((ROOT / "checkpoints" / "best.npz").resolve())
    cfg["latest_checkpoint"] = str((ROOT / "checkpoints" / "latest.npz").resolve())
    cfg["export_path"] = str((ROOT / "checkpoints" / "export.npz").resolve())
    cfg["resume_checkpoint"] = cfg.get("best_checkpoint")
    cfg["send_ratios"] = [0.25, 0.5, 0.75]
    cfg["policy_prior_strength"] = 0.8

    ensure_dir(cfg["checkpoint_dir"])
    ensure_dir(cfg["log_dir"])
    print(f"Logs: {cfg['log_dir']}", flush=True)
    print(f"Checkpoints: {cfg['checkpoint_dir']}", flush=True)
    print(f"Eval episodes: {cfg['eval_episodes']}", flush=True)
    print(f"Train steps: {cfg['train_steps']}", flush=True)
    print(f"Wall limit: {cfg['max_wall_seconds']} seconds", flush=True)
    result = run_notebook_4p_training(cfg, resume=True)
    print("Run finished.", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
