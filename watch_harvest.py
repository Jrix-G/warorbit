#!/usr/bin/env python3
import gzip, json, time, sys, os

path = "replay_dataset/compact/episodes.jsonl.gz"
target = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
pid = int(sys.argv[2]) if len(sys.argv) > 2 else None

prev = 0
while True:
    try:
        n, c2, c4 = 0, 0, 0
        try:
            with gzip.open(path, "rt") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        n += 1
                        if e["n_players"] == 2: c2 += 1
                        else: c4 += 1
                    except Exception:
                        pass
        except EOFError:
            pass  # file mid-write, use partial count
        bar_len = 40
        filled = int(bar_len * n / target)
        bar = "█" * filled + "░" * (bar_len - filled)
        mb = os.path.getsize(path) / 1e6
        running = pid and os.path.exists(f"/proc/{pid}")
        status = "⏳" if running else "✅ DONE"
        print(f"\r[{bar}] {n}/{target}  2p={c2} 4p={c4}  {mb:.1f}MB  {status}   ", end="", flush=True)
        if n == prev and not running:
            print()
            break
        prev = n
    except Exception:
        pass  # file mid-write
    time.sleep(5)
