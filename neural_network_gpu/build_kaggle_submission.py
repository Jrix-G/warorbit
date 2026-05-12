from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent


MAIN_PY = """from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from neural_network.src.model import ModelConfig, NeuralNetworkModel, load_compatible_state_dict
from neural_network.src.notebook_4p_training import _make_policy_agent
from neural_network.src.storage import load_checkpoint
import neural_network.src.storage as storage_module


ROOT = Path(storage_module.__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "best_validated.npz"
CONFIG_PATH = ROOT / "config.json"


def _load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}
    config.update(
        {
            "allow_support_actions": False,
            "board_scale": 100.0,
            "game_engine": "official_fast",
            "hidden_dim": 320,
            "horizon_scale": 100.0,
            "max_actions_per_turn": 4,
            "max_fleets": 128,
            "max_planets": 64,
            "max_players": 4,
            "max_turns": 100,
            "min_expand_attack_ships": 6,
            "official_fast_c_accel": True,
            "planet_id_scale": 64.0,
            "policy_prior_strength": 0.55,
            "production_scale": 10.0,
            "radius_scale": 10.0,
            "seed": 42,
            "send_ratios": [0.25, 0.35, 0.5, 0.65, 0.8, 0.95],
            "ship_scale": 2000.0,
            "simple_2p_only": True,
        }
    )
    return config


@lru_cache(maxsize=1)
def _build_agent():
    config = _load_config()
    state, _metadata = load_checkpoint(CHECKPOINT_PATH)
    input_weight = state["input_proj.0.weight"]
    hidden_dim, input_dim = int(input_weight.shape[0]), int(input_weight.shape[1])
    model = NeuralNetworkModel(ModelConfig(input_dim=input_dim, hidden_dim=hidden_dim))
    load_compatible_state_dict(model, state)
    model.eval()
    return _make_policy_agent(model, config, temperature=0.0, explore=False)


def agent(obs, config=None):
    return _build_agent()(obs, config)
"""


NEURAL_NETWORK_FILES = [
    "neural_network/__init__.py",
    "neural_network/src/__init__.py",
    "neural_network/src/autocorrect.py",
    "neural_network/src/baselines.py",
    "neural_network/src/benchmark.py",
    "neural_network/src/diagnostics.py",
    "neural_network/src/encoder.py",
    "neural_network/src/health_check.py",
    "neural_network/src/model.py",
    "neural_network/src/notebook_4p_training.py",
    "neural_network/src/orbit_wars_adapter.py",
    "neural_network/src/policy.py",
    "neural_network/src/population_4p_training.py",
    "neural_network/src/reward.py",
    "neural_network/src/self_play.py",
    "neural_network/src/storage.py",
    "neural_network/src/torch_compat.py",
    "neural_network/src/trainer.py",
    "neural_network/src/trajectory.py",
    "neural_network/src/utils.py",
]


def build_submission(
    output_zip: Path,
    checkpoint: Path,
    config_path: Path,
    source_root: Path,
) -> Path:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source root: {source_root}")

    with tempfile.TemporaryDirectory(prefix="kaggle_submission_") as tmp_dir:
        stage = Path(tmp_dir)
        (stage / "neural_network" / "src").mkdir(parents=True, exist_ok=True)

        for rel_path in NEURAL_NETWORK_FILES:
            src = source_root / rel_path
            dst = stage / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dst)
            elif rel_path == "neural_network/src/__init__.py":
                dst.write_text("", encoding="utf-8")
            else:
                raise FileNotFoundError(f"Missing required source file: {src}")

        shutil.copy2(checkpoint, stage / "best_validated.npz")
        if config_path.exists():
            shutil.copy2(config_path, stage / "config.json")
        else:
            (stage / "config.json").write_text("{}", encoding="utf-8")

        (stage / "main.py").write_text(MAIN_PY, encoding="utf-8")

        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in stage.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    zf.write(path, path.relative_to(stage).as_posix())

    return output_zip


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Kaggle Orbit Wars submission zip.")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "kaggle_submission_best_validated_main_v4.zip",
        help="Path to the output zip file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs" / "gpu_2p_rg_nosupport_local" / "best_validated.npz",
        help="Path to the validated checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "runs" / "gpu_2p_rg_nosupport_local" / "config.json",
        help="Path to the run config.json.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT,
        help="Path to the repo root that contains neural_network/.",
    )
    args = parser.parse_args()

    out = build_submission(
        output_zip=args.output.resolve(),
        checkpoint=args.checkpoint.resolve(),
        config_path=args.config.resolve(),
        source_root=args.source_root.resolve(),
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
