"""V15Config — opt-in overrides for bot_v7 constants.

Defaults are empty: no override means bot_v15 behaves bit-identically to bot_v7.
Each field corresponds to a flag in `analysis/V15_ARCHITECTURE.md` §2 Layer 2a.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict
from typing import Any


# Patchable V7 constants (Layer 2a candidates from D1 audit).
# Tuple of (config_field, bot_v7_attr, default_v7_value) for reference / sanity check.
PATCHABLE = (
    ("multi_source_top_k",                 "MULTI_SOURCE_TOP_K",                       10),
    ("multi_source_plan_penalty",          "MULTI_SOURCE_PLAN_PENALTY",                0.97),
    ("three_source_plan_penalty",          "THREE_SOURCE_PLAN_PENALTY",                0.94),
    ("four_player_rotating_send_ratio",    "FOUR_PLAYER_ROTATING_SEND_RATIO",          0.72),
    ("four_player_neutral_score_mult",     "FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT",  0.92),
    ("four_player_rotating_turn_limit",    "FOUR_PLAYER_ROTATING_TURN_LIMIT",          14),
)


@dataclass
class V15Config:
    """Per-field None means "leave V7 default untouched"."""

    multi_source_top_k: int | None = None
    multi_source_plan_penalty: float | None = None
    three_source_plan_penalty: float | None = None
    four_player_rotating_send_ratio: float | None = None
    four_player_neutral_score_mult: float | None = None
    four_player_rotating_turn_limit: int | None = None

    # Logic-level toggles (control code paths, not just constants). M1 will wire these in.
    enable_multi_source_early_bonus: bool = False   # V15_PATCH_MULTI_BONUS
    enable_opportunistic_expand_gate: bool = False  # V15_PATCH_OPP_GATE

    # Layer toggles
    enable_layer3_ranker: bool = False              # V15_RANKER
    enable_layer4_search: bool = False              # V15_SEARCH

    def is_passthrough(self) -> bool:
        """True iff this config will not modify V7 behavior in any way."""
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool):
                if v:
                    return False
            elif v is not None:
                return False
        return True

    def overrides_dict(self) -> dict[str, Any]:
        """Map bot_v7 attribute name → override value, only for non-None constant fields."""
        out: dict[str, Any] = {}
        for cfg_field, v7_attr, _default in PATCHABLE:
            v = getattr(self, cfg_field)
            if v is not None:
                out[v7_attr] = v
        return out


def from_env() -> V15Config:
    """Build a V15Config from V15_* environment variables.

    Convention:
      V15_MULTI_SOURCE_TOP_K=14          → multi_source_top_k=14
      V15_PATCH_MULTI_BONUS=1            → enable_multi_source_early_bonus=True
      V15_PATCH_OPP_GATE=1               → enable_opportunistic_expand_gate=True
      V15_RANKER=1                       → enable_layer3_ranker=True
      V15_SEARCH=1                       → enable_layer4_search=True
    Unset / "0" / "" → no effect.
    """
    cfg = V15Config()
    env_map = {
        "V15_MULTI_SOURCE_TOP_K":           ("multi_source_top_k",                int),
        "V15_MULTI_SOURCE_PLAN_PENALTY":    ("multi_source_plan_penalty",         float),
        "V15_THREE_SOURCE_PLAN_PENALTY":    ("three_source_plan_penalty",         float),
        "V15_FOUR_PLAYER_SEND_RATIO":       ("four_player_rotating_send_ratio",   float),
        "V15_FOUR_PLAYER_NEUTRAL_MULT":     ("four_player_neutral_score_mult",    float),
        "V15_FOUR_PLAYER_TURN_LIMIT":       ("four_player_rotating_turn_limit",   int),
    }
    for env_name, (attr, caster) in env_map.items():
        raw = os.environ.get(env_name)
        if raw:
            setattr(cfg, attr, caster(raw))

    bool_map = {
        "V15_PATCH_MULTI_BONUS": "enable_multi_source_early_bonus",
        "V15_PATCH_OPP_GATE":    "enable_opportunistic_expand_gate",
        "V15_RANKER":            "enable_layer3_ranker",
        "V15_SEARCH":            "enable_layer4_search",
    }
    for env_name, attr in bool_map.items():
        raw = os.environ.get(env_name, "").strip()
        if raw and raw != "0":
            setattr(cfg, attr, True)

    return cfg


def assert_v7_defaults_match(bot_v7_module) -> None:
    """Sanity check : the PATCHABLE table records what V7 currently uses.

    If V7 is updated upstream, this raises and reminds us to update PATCHABLE.
    """
    mismatches = []
    for _cfg, v7_attr, expected in PATCHABLE:
        actual = getattr(bot_v7_module, v7_attr, None)
        if actual != expected:
            mismatches.append(f"{v7_attr}: PATCHABLE={expected!r} v7={actual!r}")
    if mismatches:
        raise RuntimeError(
            "V15Config PATCHABLE drifted from bot_v7. Update v15_config.PATCHABLE:\n  "
            + "\n  ".join(mismatches)
        )
