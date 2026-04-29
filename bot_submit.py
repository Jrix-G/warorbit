"""Orbit Wars submission entry point for the current V7 AoW bot.

Keep this file minimal so it can be uploaded directly as the competition
submission without dragging in unrelated tuning code.
"""

from __future__ import annotations

import bot_v7


def agent(obs, config=None):
    return bot_v7.agent(obs, config)


__all__ = ["agent"]
