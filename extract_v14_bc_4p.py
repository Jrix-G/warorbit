#!/usr/bin/env python3
"""Convenience wrapper to extract a 4p-only V14 BC dataset."""

from __future__ import annotations

import sys

from extract_v14_bc_dataset import main


def _rewrite_argv() -> None:
    argv = sys.argv
    rewritten: list[str] = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--input":
            rewritten.append("--inputs")
            if i + 1 < len(argv):
                rewritten.append(argv[i + 1])
                i += 2
                continue
        if arg == "--mode":
            mode = argv[i + 1] if i + 1 < len(argv) else ""
            if mode == "all":
                rewritten.extend(["--all-players"])
            elif mode != "4p":
                raise SystemExit("extract_v14_bc_4p.py only supports 4p mode.")
            i += 2
            continue
        rewritten.append(arg)
        i += 1
    sys.argv = rewritten


if __name__ == "__main__":
    _rewrite_argv()
    main()
