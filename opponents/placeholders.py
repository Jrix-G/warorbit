"""Placeholders pour adversaires scrapés.

Tant que le scraping Kaggle n'est pas fait, ces fonctions retombent sur des
heuristiques honnêtes (distance_priority + sun_dodging combinés). Quand
analysis/scrape.py aura récupéré les notebooks, on remplace l'implémentation
ici sans toucher au reste du code.
"""

from .heuristics import distance_priority_agent, sun_dodging_agent


_TODO_SCRAPE = """
TODO scrape :
  - Structured Baseline (Pilkwang Kim, 139 votes) → structured_baseline_agent
  - Orbit Star Wars LB MAX 1224                    → orbit_star_wars_agent
Voir analysis/scrape.py.
"""


def structured_baseline_agent(obs, config=None):
    return distance_priority_agent(obs, config)


def orbit_star_wars_agent(obs, config=None):
    """Combine sun-dodge + distance-priority. À remplacer par scraping."""
    moves = sun_dodging_agent(obs, config)
    if not moves:
        return distance_priority_agent(obs, config)
    return moves
