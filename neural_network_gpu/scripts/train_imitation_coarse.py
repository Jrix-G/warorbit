"""Coarse behavior cloning: imitate the expert (source->target) DIRECTION.

The fine-grained dataset labels the exact (source, angle, ships) candidate among
~1400 per state, which is too noisy/hard to learn (val top1 ~3%).  Here we relax
the target to the set of candidates that share the expert's (source_id,target_id)
pair -- i.e. "fire from S toward T", letting the ship ratio float.  Trained with
group-marginal cross-entropy:  loss = -log sum_{j in group} softmax(logit_j).

Candidate features (see policy.build_action_candidates):
    feat[0] = source_id / n_planets,  feat[1] = target_id / n_planets.
So group membership is recovered directly from the stored candidate tensors; no
raw replay archive is required.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "kaggle_submission_stage") not in sys.path:
    sys.path.insert(0, str(ROOT / "kaggle_submission_stage"))

from neural_network.src.model import ModelConfig, NeuralNetworkModel, load_compatible_state_dict  # noqa: E402


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _group_mask(candidates: np.ndarray, masks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Bool [B, C]: candidates sharing the expert's (src,tgt) pair, within mask."""
    b = np.arange(labels.shape[0])
    src = candidates[:, :, 0]
    tgt = candidates[:, :, 1]
    exp_src = src[b, labels][:, None]
    exp_tgt = tgt[b, labels][:, None]
    group = (np.isclose(src, exp_src) & np.isclose(tgt, exp_tgt)) & masks
    group[b, labels] = True  # always include the exact expert candidate
    return group


def _infer_hidden_dim(state: dict[str, Any], default: int = 320) -> int:
    w = state.get("input_proj.0.weight")
    return default if w is None else int(np.asarray(w).shape[0])


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _group_ce(logits: torch.Tensor, masks: torch.Tensor, group: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    neg = torch.finfo(logits.dtype).min / 4
    lse_all = torch.logsumexp(logits.masked_fill(~masks, neg), dim=-1)
    lse_grp = torch.logsumexp(logits.masked_fill(~group, neg), dim=-1)
    loss = lse_all - lse_grp  # -log P(group)
    weights = weights / weights.mean().clamp_min(1e-6)
    return (loss * weights).mean()


def _batches(n: int, bs: int, shuffle: bool) -> list[np.ndarray]:
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    return [idx[i : i + bs] for i in range(0, n, bs)]


@torch.no_grad()
def _evaluate(model, val, bs, device) -> dict[str, float]:
    model.eval()
    total = correct = grp_correct = 0
    losses: list[float] = []
    for idx in _batches(val["labels"].shape[0], bs, False):
        states = torch.from_numpy(val["states"][idx]).to(device)
        cands = torch.from_numpy(val["candidates"][idx]).to(device)
        masks = torch.from_numpy(val["masks"][idx]).to(device, torch.bool)
        group = torch.from_numpy(val["group"][idx]).to(device, torch.bool)
        labels = torch.from_numpy(val["labels"][idx]).to(device, torch.long)
        weights = torch.from_numpy(val["weights"][idx]).to(device, torch.float32)
        logits = model(states, cands)["policy_logits"]
        losses.append(float(_group_ce(logits, masks, group, weights).item()))
        neg = torch.finfo(logits.dtype).min / 4
        pred = torch.argmax(logits.masked_fill(~masks, neg), dim=-1)
        total += labels.numel()
        correct += int((pred == labels).sum().item())
        grp_correct += int(group[torch.arange(len(pred)), pred].sum().item())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "exact_top1": correct / max(1, total),
        "direction_top1": grp_correct / max(1, total),
        "samples": float(total),
    }


def _prep(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    data = dict(data)
    data["group"] = _group_mask(data["candidates"], data["masks"], data["labels"])
    return data


def _sample_val(shards: list[Path], max_samples: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in ["states", "candidates", "masks", "labels", "weights", "group"]}
    remaining = max_samples
    for shard in shards:
        if remaining <= 0:
            break
        d = _prep(_load_npz(shard))
        n = d["labels"].shape[0]
        take = min(n, remaining)
        idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
        for k in chunks:
            chunks[k].append(d[k][idx])
        remaining -= take
    return {k: np.concatenate(v, axis=0) for k, v in chunks.items() if v}


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    shards = sorted(dataset_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz in {dataset_dir}")
    rng = random.Random(args.seed + 101)
    shuffled = list(shards)
    rng.shuffle(shuffled)
    val_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * args.val_shard_ratio)))
    val_shards = sorted(shuffled[:val_count])
    train_shards = sorted(shuffled[val_count:])

    first = _load_npz(train_shards[0])
    input_dim = int(first["states"].shape[1])
    ckpt = _load_checkpoint(Path(args.init_checkpoint)) if args.init_checkpoint else {}
    hidden_dim = int(args.hidden_dim or _infer_hidden_dim(ckpt))
    model = NeuralNetworkModel(ModelConfig(input_dim=input_dim, hidden_dim=hidden_dim))
    if ckpt:
        load_compatible_state_dict(model, ckpt)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    val = _sample_val(val_shards, args.val_samples, args.seed + 11)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = -1.0
    best_report: dict[str, Any] = {}
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_shards)
        ep_losses: list[float] = []
        for shard in train_shards:
            d = _prep(_load_npz(shard))
            for idx in _batches(d["labels"].shape[0], args.batch_size, True):
                states = torch.from_numpy(d["states"][idx]).to(device)
                cands = torch.from_numpy(d["candidates"][idx]).to(device)
                masks = torch.from_numpy(d["masks"][idx]).to(device, torch.bool)
                group = torch.from_numpy(d["group"][idx]).to(device, torch.bool)
                labels = torch.from_numpy(d["labels"][idx]).to(device, torch.long)
                weights = torch.from_numpy(d["weights"][idx]).to(device, torch.float32)
                opt.zero_grad(set_to_none=True)
                logits = model(states, cands)["policy_logits"]
                loss = _group_ce(logits, masks, group, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                step += 1
                ep_losses.append(float(loss.item()))
        metrics = _evaluate(model, val, args.batch_size, device)
        metrics.update({"epoch": epoch, "train_loss": float(np.mean(ep_losses)) if ep_losses else 0.0})
        print(json.dumps(metrics), flush=True)
        if metrics["direction_top1"] > best:
            best = metrics["direction_top1"]
            state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
            np.savez_compressed(out_dir / "bc_coarse_best.npz", **state)
            best_report = dict(metrics)

    final = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(out_dir / "bc_coarse_latest.npz", **final)
    report = {"best": best_report, "input_dim": input_dim, "hidden_dim": hidden_dim, "device": str(device), "args": vars(args)}
    (out_dir / "train_coarse_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, default=str), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=str(ROOT / "replay_corpus" / "imitation_4p_top10_v1"))
    p.add_argument("--output-dir", default=str(ROOT.parent / "runs" / "imitation_4p_coarse_v1"))
    p.add_argument("--init-checkpoint", default=str(ROOT.parent / "runs" / "gpu_2p_top1_distance_guarded_local_v2" / "best_validated.npz"))
    p.add_argument("--hidden-dim", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-samples", type=int, default=8192)
    p.add_argument("--val-shard-ratio", type=float, default=0.20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=19)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
