"""Thin public wrapper for the War Orbit V10 agent."""

from war_orbit.agents.v10 import agent, load_checkpoint, save_checkpoint

__all__ = ["agent", "load_checkpoint", "save_checkpoint"]

