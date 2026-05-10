from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.model import ModelConfig, NeuralNetworkModel, load_compatible_state_dict
from neural_network.src.notebook_4p_training import _infer_input_dim


def _scheduled_temperature(config: Dict[str, Any], policy_version: int) -> float:
    start = float(config.get("temperature_start", config.get("temperature_end", 0.18)))
    end = float(config.get("temperature_end", 0.18))
    decay_updates = max(1, int(config.get("temperature_decay_updates", 200)))
    frac = min(1.0, max(0.0, float(policy_version) / float(decay_updates)))
    return float(start + (end - start) * frac)


def inference_server_fn(
    model_state: Dict[str, Any],
    config: Dict[str, Any],
    obs_queue: mp.Queue,
    action_queues: Dict[int, mp.Queue],
    model_update_queue: mp.Queue,
    stop_event: mp.Event,
    device_str: str = "cuda",
    max_batch_size: int = 64,
    batch_timeout: float = 0.005,
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model = NeuralNetworkModel(ModelConfig(
        input_dim=_infer_input_dim(config),
        hidden_dim=int(config.get("hidden_dim", 320)),
    ))
    load_compatible_state_dict(model, {k: torch.as_tensor(v) for k, v in model_state.items()})
    model = model.to(device)
    model.eval()
    policy_version = int(config.get("policy_version", 0))

    pending: List[Dict] = []

    while not stop_event.is_set():
        # Check for model weight update from trainer
        try:
            new_state = model_update_queue.get_nowait()
            if isinstance(new_state, dict) and "state" in new_state:
                policy_version = int(new_state.get("policy_version", policy_version + 1))
                new_state = new_state["state"]
            else:
                policy_version += 1
            load_compatible_state_dict(model, {k: torch.as_tensor(v) for k, v in new_state.items()})
            model.eval()
        except Empty:
            pass

        # Collect batch
        deadline = time.monotonic() + batch_timeout
        while len(pending) < max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                msg = obs_queue.get(timeout=remaining)
                pending.append(msg)
            except Empty:
                break

        if not pending:
            continue

        n_cands = [m["n_candidates"] for m in pending]
        max_n = max(n_cands)
        cand_dim = pending[0]["candidates"].shape[-1]

        states = np.stack([m["state"] for m in pending])
        cands_padded = np.zeros((len(pending), max_n, cand_dim), dtype=np.float32)
        mask = np.zeros((len(pending), max_n), dtype=bool)
        for i, (m, n) in enumerate(zip(pending, n_cands)):
            cands_padded[i, :n] = m["candidates"]
            mask[i, :n] = True

        state_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        cand_t = torch.as_tensor(cands_padded, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)

        with torch.no_grad():
            outputs = model(state_t, cand_t)
            logits = outputs["policy_logits"]
            logits = logits.masked_fill(~mask_t, float("-inf"))

            temp = _scheduled_temperature(config, policy_version)
            if temp > 0.0:
                action_logits = logits / max(temp, 1e-6)
                probs = torch.softmax(action_logits, dim=-1)
                probs = probs.nan_to_num(0.0).clamp(min=1e-8)
                action_idxs = torch.multinomial(probs, 1).squeeze(-1)
            else:
                action_logits = logits
                action_idxs = logits.argmax(dim=-1)
            log_probs_all = torch.log_softmax(action_logits, dim=-1)
            probs_all = log_probs_all.exp()
            safe_log_probs = log_probs_all.masked_fill(~mask_t, 0.0)
            entropy_terms = probs_all * safe_log_probs
            entropies = -entropy_terms.sum(dim=-1)
            selected_log_probs = log_probs_all.gather(1, action_idxs.unsqueeze(-1)).squeeze(-1)

        action_idxs_np = action_idxs.cpu().numpy()
        log_probs_np = selected_log_probs.cpu().numpy()
        entropies_np = entropies.cpu().numpy()
        for msg, action_idx, old_log_prob, entropy in zip(pending, action_idxs_np, log_probs_np, entropies_np):
            action_queues[msg["worker_id"]].put({
                "action_idx": int(action_idx),
                "old_log_prob": float(old_log_prob),
                "entropy": float(entropy),
                "temperature": float(temp),
                "policy_version": int(policy_version),
            })

        pending = []
