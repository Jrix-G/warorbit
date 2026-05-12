#!/usr/bin/env python3
"""Convenience wrapper to extract a 4p-only V14 BC dataset."""

from __future__ import annotations

import sys

from extract_v14_bc_compact import main


def _ensure_default_mode() -> None:
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "4p"])


if __name__ == "__main__":
    _ensure_default_mode()
    main()
