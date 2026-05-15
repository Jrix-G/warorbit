"""Evaluate multiple checkpoints against simple opponents and report winrate/valid/noop/ships."""
from __future__ import annotations

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.model import ModelConfig, NeuralNetworkModel, load_compatible_state_dict
from neural_network.src.notebook_4p_training import _build_agents, run_match


def _load_raw(ckpt_path: Path):
    """Load npz checkpoint, return (model_state_dict, metadata_dict)."""
    raw = np.load(ckpt_path, allow_pickle=False)
    state = {k: torch.as_tensor(raw[k]) for k in raw.files if k != "metadata"}
    try:
        meta = json.loads(str(raw["metadata"]))
    except KeyError:
        meta = {}
    return state, meta


def _load_fallback_config(config_path: Path | None):
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Fallback config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def eval_checkpoint(
    ckpt_path: Path,
    opponent: str,
    n_games: int,
    n_players: int,
    device: torch.device,
    fallback_config: dict | None = None,
):
    model_state, meta = _load_raw(ckpt_path)
    input_dim  = model_state["input_proj.0.weight"].shape[1]
    hidden_dim = model_state["input_proj.0.weight"].shape[0]
    cfg = meta.get("adaptive_config", meta.get("config", {}))
    if not cfg:
        cfg = dict(fallback_config or {})
    if not cfg:
        raise KeyError(
            f"No config found in {ckpt_path.name} metadata and no fallback config provided."
        )
    model_cfg = ModelConfig(input_dim=input_dim, hidden_dim=hidden_dim)
    model = NeuralNetworkModel(model_cfg).to(device)
    load_compatible_state_dict(model, model_state)
    model.eval()

    eval_cfg = dict(cfg)
    eval_cfg["temperature_start"] = 0.0
    eval_cfg["temperature_end"] = 0.0

    valid_win_max_legal_noop = float(cfg.get("valid_win_max_legal_noop_rate", 0.35))
    valid_win_min_ships = float(cfg.get("valid_win_min_avg_ships_sent", 4.0))
    valid_win_min_real_moves = float(cfg.get("valid_win_min_real_moves_turn", 0.90))

    wins, valid_wins, noops, ships_list = [], [], [], []

    for i in range(n_games):
        our_index = i % n_players
        agents, _, action_records, _ = _build_agents(
            model, eval_cfg, seed=42000 + i, our_index=our_index,
            temperature=0.0, pool=[opponent], explore=False,
            n_players=n_players,
        )
        result = run_match(agents, seed=42000 + i, n_players=n_players)
        won = 1.0 if result.get("winner") == our_index else 0.0
        wins.append(won)

        ar = [a for a in action_records if a.get("player") == our_index]
        total = len(ar) or 1
        noop_r = sum(1 for a in ar if a.get("action_type") in ("do_nothing", "legal_noop")) / total
        s_sent = [a.get("ships_sent", 0) for a in ar if a.get("action_type") not in ("do_nothing", "legal_noop")]
        avg_s = float(np.mean(s_sent)) if s_sent else 0.0
        real_m = sum(1 for a in ar if a.get("action_type") not in ("do_nothing", "legal_noop", "forced_noop")) / total

        noops.append(noop_r)
        ships_list.append(avg_s)

        is_valid = (
            won == 1.0
            and noop_r <= valid_win_max_legal_noop
            and avg_s >= valid_win_min_ships
            and real_m >= valid_win_min_real_moves
        )
        valid_wins.append(1.0 if is_valid else 0.0)

        if (i + 1) % 16 == 0:
            print(f"    progress {i+1}/{n_games} wins={int(sum(wins))}", flush=True)

    return {
        "winrate": float(np.mean(wins)),
        "valid": float(np.mean(valid_wins)),
        "noop": float(np.mean(noops)),
        "ships": float(np.mean(ships_list)),
        "wins": int(sum(wins)),
        "games": n_games,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--opponents", type=str, default="random,greedy")
    parser.add_argument("--n-players", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Fallback config.json to use when a checkpoint has no embedded metadata.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    opponents = args.opponents.split(",")
    fallback_config = _load_fallback_config(Path(args.config)) if args.config else None

    checkpoints = {
        "bc_4p_top10": ROOT / "runs/imitation_4p_top10_v1/bc_4p_top10_best.npz",
        "overnight_9h_best": ROOT / "runs/gpu_2p_overnight_9h/best_validated.npz",
    }

    for name, path in checkpoints.items():
        if not path.exists():
            print(f"\nSKIP {name} — not found: {path}")
            continue
        print(f"\n{'='*50}")
        print(f"Checkpoint: {name}")
        print(f"Path: {path}")
        print(f"{'='*50}")
        for opp in opponents:
            print(f"  vs {opp} ({args.games} games)...", flush=True)
            r = eval_checkpoint(path, opp, args.games, args.n_players, device, fallback_config)
            print(f"  => winrate={r['winrate']:.3f}  valid={r['valid']:.3f}  noop={r['noop']:.3f}  ships={r['ships']:.2f}  ({r['wins']}/{r['games']} wins)")

    print("\nDone.")


if __name__ == "__main__":
    main()
