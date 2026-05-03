# VPS Runbook

This folder contains the low-load launcher for long VPS training runs.

Install the CPU limiter first if needed:

```bash
sudo apt-get update
sudo apt-get install -y cpulimit
```

## Recommended launch

```bash
bash VPS/run_v8_5_train_10h.sh
```

What it does:

- runs `train_v8_5.py` for 10 hours
- keeps the process detached with `nohup`
- pins NumPy/BLAS thread pools to 1 thread
- prefers `cpulimit -l 80` if available
- exits with a clear message if `cpulimit` is missing
- writes logs and checkpoints under `VPS/logs/` and `VPS/evaluations/`

If you want to inspect the run:

```bash
tail -f VPS/logs/*.log
```
