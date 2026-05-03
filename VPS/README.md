# VPS Runbook

This folder contains the low-load launcher for long VPS training runs.

Install the CPU limiter first if needed:

```bash
sudo apt-get update
sudo apt-get install -y cpulimit
```

## Recommended launches

```bash
bash VPS/run_v8_5_train_10h.sh
```

```bash
bash VPS/run_v9_train_10h.sh
```

What it does:

- runs `train_v8_5.py` for 10 hours
- keeps the process detached with `nohup`
- pins NumPy/BLAS thread pools to 1 thread
- prefers `cpulimit -l 80` if available
- exits with a clear message if `cpulimit` is missing
- writes logs and checkpoints under `VPS/logs/` and `VPS/evaluations/`

V9 uses the same safety pattern, plus a stricter training profile:

- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `workers=1` by default in the VPS launcher
- `cpulimit -l 80` when available
- optional RAM cap via `VPS_MAX_VMEM_KB` and `ulimit -Sv`

Practical rule:

- if the VPS has 2 vCPU and 2 to 4 GB RAM, keep `workers=1`
- if it has 4 vCPU and 8 GB RAM, `workers=2` is usually still safe
- if it has 8 vCPU and 16 GB+ RAM, `workers=4` to `8` is reasonable, but increase gradually

For V9, the real gain comes mostly from `--workers`, not from the raw clock speed alone.
Because training has serial parts, the speedup is usually not linear:

- 2 workers: about 1.6x to 1.8x
- 4 workers: about 2.7x to 3.5x
- 8 workers: about 4x to 6x

So a run that takes 10 hours on 1 worker can often fall to about 2 to 4 hours on a
good 8-worker VPS, but only if the machine stays memory-stable and the benchmark is
kept out of the inner loop.

If you want a hard memory ceiling, set something like:

```bash
export VPS_MAX_VMEM_KB=$((6 * 1024 * 1024))
```

That example caps virtual memory around 6 GB before launching the VPS script.
If you prefer a cleaner isolation boundary, use a cgroup or `systemd-run` with
`MemoryMax=`.

If you want to inspect the run:

```bash
tail -f VPS/logs/*.log
```
