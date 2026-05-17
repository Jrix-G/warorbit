"""v17_infer — batched inference server for V17 self-play.

Self-play workers each run MCTS sequentially, so a worker cannot batch its
own forwards. But the N workers are independent: at any instant several are
blocked on a net evaluation. This module centralises those evaluations in one
server process that collects up to `max_batch` pending requests and runs a
SINGLE padded forward, then scatters the results back.

Numerically identical to per-state forwards: every op in V17Net is per-row
(per-planet encoder, masked mean/max pooling, per-row attention, heads); the
padding mask excludes padded planets from pooling/attention exactly as the
unbatched path does. Batching changes throughput, never the result.

Linux/fork only (workers inherit the queues as module globals). On other
platforms keep the server off and use the in-process fallback in v17_mcts.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import queue
import time

import numpy as np
import torch

import v17_encode as enc
from v17_net import V17Net, policy_probs

_REQ_Q = None
_RES_QS = None
_WID_COUNTER = None


def _batch_forward(net, device, reqs):
    """reqs: list of (pf [n,P_DIM], gf [G_DIM]).
    Returns list of (probs [n,n+1], value float), one per request."""
    B = len(reqs)
    maxN = max(r[0].shape[0] for r in reqs)
    pf = np.zeros((B, maxN, enc.P_DIM), dtype=np.float32)
    gf = np.zeros((B, enc.G_DIM), dtype=np.float32)
    mask = np.zeros((B, maxN), dtype=bool)
    ns = []
    for i, (p, g) in enumerate(reqs):
        n = p.shape[0]
        ns.append(n)
        if n:
            pf[i, :n] = p
            mask[i, :n] = True
        gf[i] = g
    with torch.no_grad():
        logits, value = net(torch.as_tensor(pf, device=device),
                            torch.as_tensor(gf, device=device),
                            torch.as_tensor(mask, device=device))
        probs = policy_probs(logits).cpu().numpy()
        value = value.cpu().numpy()
    out = []
    for i, n in enumerate(ns):
        out.append((probs[i, :n, :n + 1].copy(), float(value[i])))
    return out


def _server_loop(req_q, res_qs, state_dict, d, device, max_batch, timeout):
    torch.set_num_threads(1)
    net = V17Net(d=d)
    net.load_state_dict(state_dict)
    net.eval().to(device)
    while True:
        first = req_q.get()
        if first is None:
            break
        batch = [first]
        deadline = time.monotonic() + timeout
        stop = False
        while len(batch) < max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = req_q.get(timeout=remaining)
            except queue.Empty:
                break
            if item is None:
                stop = True
                break
            batch.append(item)
        reqs = [(pf, gf) for (_, pf, gf) in batch]
        outs = _batch_forward(net, device, reqs)
        for (wid, _, _), o in zip(batch, outs):
            res_qs[wid].put(o)
        if stop:
            break


class ServerEvaluator:
    """Worker-side client: same interface as a local net eval."""

    __slots__ = ("req_q", "res_q", "wid")

    def __init__(self, req_q, res_q, wid):
        self.req_q = req_q
        self.res_q = res_q
        self.wid = wid

    def eval(self, fs, player):
        pf, gf = enc.encode(fs, player)
        self.req_q.put((self.wid, pf, gf))
        return self.res_q.get()


def start_server(state_dict, d, n_workers, *, device="cpu",
                 max_batch=None, timeout=0.003):
    """Start the inference server process. Linux/fork only.

    Stores the queues as module globals so forked workers inherit them.
    Returns the server Process (pass it to stop_server)."""
    import multiprocessing as mp
    global _REQ_Q, _RES_QS, _WID_COUNTER
    ctx = mp.get_context("fork")
    _REQ_Q = ctx.Queue()
    _RES_QS = [ctx.Queue() for _ in range(n_workers)]
    _WID_COUNTER = ctx.Value("i", 0)
    mb = max_batch or n_workers
    cpu_sd = {k: v.cpu() for k, v in state_dict.items()}
    p = ctx.Process(target=_server_loop,
                    args=(_REQ_Q, _RES_QS, cpu_sd, d, device, mb, timeout),
                    daemon=True)
    p.start()
    return p


def attach_worker():
    """Called once in each self-play worker; installs a ServerEvaluator."""
    import v17_mcts
    with _WID_COUNTER.get_lock():
        wid = _WID_COUNTER.value
        _WID_COUNTER.value += 1
    if wid >= len(_RES_QS):
        raise RuntimeError(f"worker id {wid} exceeds {len(_RES_QS)} queues")
    v17_mcts.set_evaluator(ServerEvaluator(_REQ_Q, _RES_QS[wid], wid))


def stop_server(p):
    try:
        if _REQ_Q is not None:
            _REQ_Q.put(None)
    except Exception:
        pass
    if p is not None:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    # Numerical-identity test: batched padded forward == per-state forward.
    torch.manual_seed(0)
    net = V17Net(d=64)
    net.eval()

    def _single(pf, gf):
        n = pf.shape[0]
        with torch.no_grad():
            logits, value = net(torch.as_tensor(pf[None]),
                                torch.as_tensor(gf[None]),
                                torch.ones(1, n, dtype=torch.bool))
            return policy_probs(logits)[0].numpy(), float(value[0])

    rng = np.random.default_rng(1)
    sizes = [3, 12, 7, 20, 1, 15]
    reqs = []
    for n in sizes:
        pf = rng.random((n, enc.P_DIM), dtype=np.float32)
        gf = rng.random(enc.G_DIM, dtype=np.float32)
        reqs.append((pf, gf))

    batched = _batch_forward(net, "cpu", reqs)
    worst_p = worst_v = 0.0
    for (pf, gf), (bp, bv) in zip(reqs, batched):
        sp, sv = _single(pf, gf)
        assert bp.shape == sp.shape, (bp.shape, sp.shape)
        worst_p = max(worst_p, float(np.abs(bp - sp).max()))
        worst_v = max(worst_v, abs(bv - sv))
    assert worst_p < 1e-4, f"policy mismatch {worst_p}"
    assert worst_v < 1e-4, f"value mismatch {worst_v}"
    print(f"v17_infer: batched==single  policy_err={worst_p:.2e} "
          f"value_err={worst_v:.2e}  ({len(sizes)} states) OK")
