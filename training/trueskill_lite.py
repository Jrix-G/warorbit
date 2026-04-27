"""TrueSkill-lite : ranking adaptatif de candidats.

Implémentation minimale (≈ Glicko simplifié, suffit pour notre usage):
- chaque agent: (mu, sigma)
- victoire de A sur B → mu_A monte, mu_B descend, proportionnel au prior et
  à la surprise; sigma diminue à chaque match (on est plus sûr).
- on peut décider de "stopper" un matchup quand sigma_A < TOL.

Pas de vraie inférence bayésienne : on veut juste arrêter les matchups
inutiles et ranker les candidats. Suffisamment précis pour CMA-ES.
"""

import math


MU0 = 25.0
SIGMA0 = 25.0 / 3.0
BETA = SIGMA0 / 2.0     # incertitude perf
TAU = SIGMA0 / 100.0    # dérive
DRAW_PROB = 0.05


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def update_match(rating_a, rating_b, score_a):
    """Met à jour (mu, sigma) après une partie.

    score_a = 1 si A gagne, 0 si B gagne, 0.5 = nul.
    Retourne (new_a, new_b).
    """
    mu_a, sigma_a = rating_a
    mu_b, sigma_b = rating_b

    c2 = 2.0 * BETA * BETA + sigma_a * sigma_a + sigma_b * sigma_b
    c = math.sqrt(c2)
    t = (mu_a - mu_b) / c

    if score_a == 1:
        v = _phi_pdf(t) / max(_phi(t), 1e-9)
        w = v * (v + t)
        new_mu_a = mu_a + (sigma_a * sigma_a / c) * v
        new_mu_b = mu_b - (sigma_b * sigma_b / c) * v
    elif score_a == 0:
        v = _phi_pdf(-t) / max(_phi(-t), 1e-9)
        w = v * (v - t)
        new_mu_a = mu_a - (sigma_a * sigma_a / c) * v
        new_mu_b = mu_b + (sigma_b * sigma_b / c) * v
    else:  # nul
        w = 0.5
        new_mu_a, new_mu_b = mu_a, mu_b

    new_sigma_a = math.sqrt(max(sigma_a * sigma_a * (1.0 - sigma_a * sigma_a / c2 * w), TAU * TAU))
    new_sigma_b = math.sqrt(max(sigma_b * sigma_b * (1.0 - sigma_b * sigma_b / c2 * w), TAU * TAU))

    return (new_mu_a, new_sigma_a), (new_mu_b, new_sigma_b)


def conservative_skill(rating):
    """μ - 3σ : borne basse de skill avec ~99% de confiance."""
    return rating[0] - 3.0 * rating[1]


class RatingTable:
    """Table de ratings pour un set d'agents."""

    def __init__(self):
        self.ratings = {}

    def get(self, name):
        if name not in self.ratings:
            self.ratings[name] = (MU0, SIGMA0)
        return self.ratings[name]

    def record(self, name_a, name_b, score_a):
        ra = self.get(name_a)
        rb = self.get(name_b)
        new_a, new_b = update_match(ra, rb, score_a)
        self.ratings[name_a] = new_a
        self.ratings[name_b] = new_b

    def converged(self, name, sigma_target=2.0):
        return self.get(name)[1] < sigma_target

    def leaderboard(self):
        return sorted(
            [(n, r[0], r[1], conservative_skill(r)) for n, r in self.ratings.items()],
            key=lambda x: x[3],
            reverse=True,
        )
