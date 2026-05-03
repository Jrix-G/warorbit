"""Backward-compatible shim for the V8.5 trainer."""

from train_v8_5 import *  # noqa: F401,F403

if __name__ == "__main__":
    from train_v8_5 import main

    main()
