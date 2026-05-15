from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "kaggle_submission_stage") not in sys.path:
    sys.path.insert(0, str(ROOT / "kaggle_submission_stage"))

from neural_network.src.model import ModelConfig, NeuralNetworkModel, load_compatible_state_dict  # noqa: E402


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    npz = np.load(path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def _infer_hidden_dim(state: dict[str, Any], default: int = 320) -> int:
    weight = state.get("input_proj.0.weight")
    if weight is None:
        return default
    return int(np.asarray(weight).shape[0])


def _masked_ce(logits: torch.Tensor, masks: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~masks, -1.0e9)
    loss = F.cross_entropy(masked, labels, reduction="none")
    weights = weights / weights.mean().clamp_min(1.0e-6)
    return (loss * weights).mean()


def _batch_indices(n: int, batch_size: int, shuffle: bool) -> list[np.ndarray]:
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _sample_validation(shards: list[Path], max_samples: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    chunks: dict[str, list[np.ndarray]] = {"states": [], "candidates": [], "masks": [], "labels": [], "weights": []}
    remaining = int(max_samples)
    for shard in shards:
        if remaining <= 0:
            break
        data = _load_npz(shard)
        n = int(data["labels"].shape[0])
        take = min(n, remaining)
        idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
        for key in chunks:
            chunks[key].append(data[key][idx])
        remaining -= take
    return {key: np.concatenate(vals, axis=0) for key, vals in chunks.items() if vals}


@torch.no_grad()
def _evaluate(model: NeuralNetworkModel, val: dict[str, np.ndarray], batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct1 = 0
    correct3 = 0
    losses: list[float] = []
    for idx in _batch_indices(int(val["labels"].shape[0]), batch_size, shuffle=False):
        states = torch.from_numpy(val["states"][idx]).to(device)
        candidates = torch.from_numpy(val["candidates"][idx]).to(device)
        masks = torch.from_numpy(val["masks"][idx]).to(device=device, dtype=torch.bool)
        labels = torch.from_numpy(val["labels"][idx]).to(device=device, dtype=torch.long)
        weights = torch.from_numpy(val["weights"][idx]).to(device=device, dtype=torch.float32)
        logits = model(states, candidates)["policy_logits"]
        losses.append(float(_masked_ce(logits, masks, labels, weights).item()))
        masked = logits.masked_fill(~masks, -1.0e9)
        pred1 = torch.argmax(masked, dim=-1)
        top3 = torch.topk(masked, k=min(3, masked.shape[-1]), dim=-1).indices
        total += labels.numel()
        correct1 += int((pred1 == labels).sum().item())
        correct3 += int((top3 == labels.unsqueeze(-1)).any(dim=-1).sum().item())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "top1": correct1 / max(1, total),
        "top3": correct3 / max(1, total),
        "samples": float(total),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    shards = sorted(dataset_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz found in {dataset_dir}")

    first = _load_npz(shards[0])
    input_dim = int(first["states"].shape[1])
    checkpoint_state = _load_checkpoint(Path(args.init_checkpoint)) if args.init_checkpoint else {}
    hidden_dim = int(args.hidden_dim or _infer_hidden_dim(checkpoint_state))
    model = NeuralNetworkModel(ModelConfig(input_dim=input_dim, hidden_dim=hidden_dim))
    load_report = load_compatible_state_dict(model, checkpoint_state) if checkpoint_state else {"loaded": [], "partial": {}, "skipped": {}}

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    val = _sample_validation(shards, int(args.val_samples), int(args.seed) + 11)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_top1 = -1.0
    best_report: dict[str, Any] = {}
    global_step = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        random.shuffle(shards)
        epoch_losses: list[float] = []
        for shard in shards:
            data = _load_npz(shard)
            for idx in _batch_indices(int(data["labels"].shape[0]), int(args.batch_size), shuffle=True):
                states = torch.from_numpy(data["states"][idx]).to(device)
                candidates = torch.from_numpy(data["candidates"][idx]).to(device)
                masks = torch.from_numpy(data["masks"][idx]).to(device=device, dtype=torch.bool)
                labels = torch.from_numpy(data["labels"][idx]).to(device=device, dtype=torch.long)
                weights = torch.from_numpy(data["weights"][idx]).to(device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                logits = model(states, candidates)["policy_logits"]
                loss = _masked_ce(logits, masks, labels, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()
                global_step += 1
                epoch_losses.append(float(loss.item()))
                if global_step % int(args.log_every) == 0:
                    print(f"train step={global_step} epoch={epoch} loss={np.mean(epoch_losses[-args.log_every:]):.4f}", flush=True)

        metrics = _evaluate(model, val, int(args.batch_size), device)
        metrics.update({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0})
        print(json.dumps(metrics), flush=True)
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
            np.savez_compressed(out_dir / "bc_4p_top10_best.npz", **state)
            best_report = dict(metrics)

    final_state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(out_dir / "bc_4p_top10_latest.npz", **final_state)
    report = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(out_dir),
        "shards": len(shards),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "device": str(device),
        "init_checkpoint": str(args.init_checkpoint),
        "load_report": load_report,
        "best": best_report,
        "args": vars(args),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior-clone the candidate policy on compact Orbit Wars replay shards.")
    parser.add_argument("--dataset-dir", default=str(ROOT / "replay_corpus" / "imitation_4p_top10_v1"))
    parser.add_argument("--output-dir", default=str(ROOT.parent / "runs" / "imitation_4p_top10_v1"))
    parser.add_argument("--init-checkpoint", default=str(ROOT.parent / "runs" / "gpu_2p_top1_distance_guarded_local_v2" / "best_validated.npz"))
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
