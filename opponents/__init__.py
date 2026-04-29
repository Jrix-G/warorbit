"""Zoo d'adversaires pour entraînement et tournois.

Chaque module expose `agent(obs, config=None)` compatible Kaggle.
`ZOO` regroupe tous les adversaires utilisables en training, y compris les
notebooks scrappés quand ils sont présents localement.
"""

from __future__ import annotations

import pkgutil
from importlib import import_module
from pathlib import Path

from .baselines import greedy_agent, passive_agent, random_agent, starter_agent
from .heuristics import distance_priority_agent, sun_dodging_agent
from .placeholders import orbit_star_wars_agent, structured_baseline_agent


def _load_notebook_agent(module_name: str):
    import io, sys
    try:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = io.StringIO()
        try:
            module = import_module(f".{module_name}", __name__)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        agent = getattr(module, "agent", None)
        if not callable(agent):
            return None

        def safe_agent(obs, config=None):
            try:
                return agent(obs, config)
            except TypeError:
                try:
                    return agent(obs)
                except Exception:
                    return []
            except Exception:
                return []

        return safe_agent
    except Exception:
        return None


ZOO = {
    "passive": passive_agent,
    "random": random_agent,
    "greedy": greedy_agent,
    "starter": starter_agent,
    "distance": distance_priority_agent,
    "sun_dodge": sun_dodging_agent,
    "structured": structured_baseline_agent,
    "orbit_stars": orbit_star_wars_agent,
}

_SKIP_NOTEBOOKS = {"notebook_bovard_getting_started"}

_pkg_dir = Path(__file__).resolve().parent
for module_info in sorted(pkgutil.iter_modules([str(_pkg_dir)]), key=lambda m: m.name):
    if not module_info.name.startswith("notebook_"):
        continue
    if module_info.name in _SKIP_NOTEBOOKS:
        continue
    agent = _load_notebook_agent(module_info.name)
    if agent is not None:
        ZOO[module_info.name] = agent


NOTEBOOK_POOL_ORDER = [
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
    "notebook_physics_accurate",
    "notebook_tactical_heuristic",
]


def get(name):
    if name not in ZOO:
        raise ValueError(f"Adversaire inconnu: {name}. Disponibles: {list(ZOO)}")
    return ZOO[name]


def training_pool(limit: int = 15):
    pool = [name for name in NOTEBOOK_POOL_ORDER if name in ZOO]
    if len(pool) < limit:
        extras = [name for name in sorted(ZOO) if name not in pool]
        pool.extend(name for name in extras if name.startswith("notebook_"))
    if limit <= 0:
        return pool
    return pool[:limit]
