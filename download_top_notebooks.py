#!/usr/bin/env python3
"""Discover and download Orbit Wars notebook kernels from Kaggle.

This is now a thin discovery utility that uses Kaggle's kernels list instead of
the old hardcoded notebook shortlist.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from notebook_harvest import (
    DEFAULT_BROWSER_AUTH,
    DEFAULT_COMPETITION,
    discover_competition_kernels,
    download_kernel_notebook,
    safe_module_name,
)


if os.name == "nt":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and download top Orbit Wars notebooks.")
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--download-dir", default="notebooks/harvested")
    parser.add_argument("--auth-file", default=str(DEFAULT_BROWSER_AUTH),
                        help="browser auth export containing Kaggle cookies and client token")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.download_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("📚 DOWNLOADING TOP ORBIT WARS NOTEBOOKS")
    print("=" * 80)

    kernels = discover_competition_kernels(
        args.competition,
        limit=args.limit,
        auth_file=Path(args.auth_file),
    )
    if not kernels:
        raise SystemExit("Could not discover kernels. Check Kaggle auth and connectivity.")

    print(f"Found {len(kernels)} competition kernels")
    print("\n🎯 TARGET NOTEBOOKS:")
    for kernel in kernels:
        ref = kernel.get("ref")
        if not ref:
            continue
        print(f"  - {kernel.get('title') or ref} ({kernel.get('totalVotes', '?')} votes)")
        print(f"       → https://kaggle.com/code/{ref}")

    print("\n🔄 DOWNLOADS:")
    for kernel in kernels:
        ref = kernel.get("ref")
        if not ref:
            continue
        module_name = safe_module_name(ref)
        try:
            nb_path = download_kernel_notebook(
                ref,
                out_dir,
                refresh=args.refresh,
                auth_file=Path(args.auth_file),
            )
            if nb_path is None:
                print(f"  {module_name}: download failed")
            else:
                print(f"  {module_name}: {nb_path}")
        except Exception as exc:
            print(f"  {module_name}: error {exc}")

    print("\n" + "=" * 80)
    print("✅ SETUP COMPLETE - Use extract_notebook_agents.py --discover next")
    print("=" * 80)


if __name__ == "__main__":
    main()
