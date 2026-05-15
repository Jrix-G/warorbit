"""bot_v15 — V7 noyau + opt-in overrides (Layer 2a) + futur Layer 2b/3/4.

Contract (M0): si V15Config est passthrough (tous les overrides None / flags False),
bot_v15.agent doit produire un flux d'actions bit-identique à bot_v7.agent sur la
même obs / config. Vérifié par tests/test_v15_regression.py.

Stratégie : snapshot + monkey-patch + restore autour de chaque appel agent.
Idempotent côté bot_v7 : après retour, bot_v7 est byte-identique à avant.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import bot_v7
import v15_config


_CONFIG: v15_config.V15Config | None = None
_VALIDATED = False


def set_config(cfg: v15_config.V15Config) -> None:
    """Override config used by agent(). If never called, env-based config is used."""
    global _CONFIG
    _CONFIG = cfg


def _get_config() -> v15_config.V15Config:
    global _CONFIG, _VALIDATED
    if not _VALIDATED:
        v15_config.assert_v7_defaults_match(bot_v7)
        _VALIDATED = True
    if _CONFIG is None:
        _CONFIG = v15_config.from_env()
    return _CONFIG


@contextmanager
def _patched_v7(overrides: dict[str, Any]) -> Iterator[None]:
    """Snapshot, override, yield, restore. Idempotent."""
    if not overrides:
        yield
        return
    snapshot = {k: getattr(bot_v7, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(bot_v7, k, v)
        yield
    finally:
        for k, v in snapshot.items():
            setattr(bot_v7, k, v)


def agent(obs, config=None):
    cfg = _get_config()

    if cfg.is_passthrough():
        # Pas de mutation → V15 ≡ V7 par construction.
        return bot_v7.agent(obs, config)

    overrides = cfg.overrides_dict()
    with _patched_v7(overrides):
        result = bot_v7.agent(obs, config)

    # Logic-level patches (Layer 2a §M1, à câbler progressivement).
    # Pour l'instant : aucune modification du résultat. Les flags sont
    # acceptés mais no-op tant que la logique n'est pas implémentée.
    # TODO M1 : enable_multi_source_early_bonus, enable_opportunistic_expand_gate.

    return result
