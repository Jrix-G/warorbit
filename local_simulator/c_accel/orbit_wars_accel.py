"""Activation helpers for optional Orbit Wars C accelerators."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


_ORIGINALS: dict[str, Any] = {}


def load_c_module() -> ModuleType | None:
    try:
        return importlib.import_module("local_simulator.c_accel.orbit_wars_c")
    except Exception:
        return None


def enable(orbit_wars_module: ModuleType) -> bool:
    """Patch supported hotspots on an imported orbit_wars module.

    Returns True when the C extension is available and installed.  The patch is
    intentionally small and reversible.
    """

    c_module = load_c_module()
    if c_module is None:
        return False
    if "generate_comet_paths" not in _ORIGINALS:
        _ORIGINALS["generate_comet_paths"] = orbit_wars_module.generate_comet_paths
    orbit_wars_module.generate_comet_paths = c_module.generate_comet_paths
    return True


def disable(orbit_wars_module: ModuleType) -> None:
    original = _ORIGINALS.get("generate_comet_paths")
    if original is not None:
        orbit_wars_module.generate_comet_paths = original


def is_enabled(orbit_wars_module: ModuleType) -> bool:
    c_module = load_c_module()
    return c_module is not None and orbit_wars_module.generate_comet_paths is c_module.generate_comet_paths

