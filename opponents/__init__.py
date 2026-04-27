"""Zoo d'adversaires pour entraînement et tournois.

Chaque module expose `agent(obs, config=None)` compatible kaggle_environments.
ZOO regroupe tous les adversaires utilisables en training.
"""

from .baselines import passive_agent, random_agent, greedy_agent, starter_agent
from .heuristics import distance_priority_agent, sun_dodging_agent
from .placeholders import structured_baseline_agent, orbit_star_wars_agent

# Import notebook agents (extracted from top player notebooks)
try:
    from .notebook_orbitbotnext import agent as orbitbotnext_agent
except:
    orbitbotnext_agent = None

try:
    from .notebook_distance_prioritized import agent as distance_prioritized_agent
except:
    distance_prioritized_agent = None

try:
    from .notebook_physics_accurate import agent as physics_accurate_agent
except:
    physics_accurate_agent = None

try:
    from .notebook_tactical_heuristic import agent as tactical_heuristic_agent
except:
    tactical_heuristic_agent = None

ZOO = {
    "passive":     passive_agent,
    "random":      random_agent,
    "greedy":      greedy_agent,
    "starter":     starter_agent,
    "distance":    distance_priority_agent,
    "sun_dodge":   sun_dodging_agent,
    "structured":  structured_baseline_agent,
    "orbit_stars": orbit_star_wars_agent,
}

# Add notebook agents if available
if orbitbotnext_agent:
    ZOO["notebook_orbitbotnext"] = orbitbotnext_agent
if distance_prioritized_agent:
    ZOO["notebook_distance_prioritized"] = distance_prioritized_agent
if physics_accurate_agent:
    ZOO["notebook_physics_accurate"] = physics_accurate_agent
if tactical_heuristic_agent:
    ZOO["notebook_tactical_heuristic"] = tactical_heuristic_agent


def get(name):
    if name not in ZOO:
        raise ValueError(f"Adversaire inconnu: {name}. Disponibles: {list(ZOO)}")
    return ZOO[name]
