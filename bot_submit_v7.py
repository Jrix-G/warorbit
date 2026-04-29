"""Submission entry point for the current V7 AoW bot.

This is intentionally thin: it delegates directly to `bot_v7.agent` so the
submission uses the exact same policy as the local V7 AoW benchmark.
"""

from __future__ import annotations

import bot_v7


def agent(obs, config=None):
    return bot_v7.agent(obs, config)


__all__ = ["agent"]
