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

    pending: List[Dict] = []

    while not stop_event.is_set():
        # Check for model weight update from trainer
        try:
            new_state = model_update_queue.get_nowait()
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

            temp = float(config.get("temperature_end", 0.18))
            if temp > 0.0:
                probs = torch.softmax(logits / max(temp, 1e-6), dim=-1)
                probs = probs.nan_to_num(0.0).clamp(min=1e-8)
                action_idxs = torch.multinomial(probs, 1).squeeze(-1)
            else:
                action_idxs = logits.argmax(dim=-1)

        action_idxs_np = action_idxs.cpu().numpy()
        for msg, action_idx in zip(pending, action_idxs_np):
            action_queues[msg["worker_id"]].put(int(action_idx))

        pending = []
