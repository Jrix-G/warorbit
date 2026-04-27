"""Adversarial league : démon discovery.

Toutes les N générations, on lance un mini-CMA-ES inversé : on cherche un
vecteur de poids `D` qui *maximise le win-rate contre best courant*. Ce
`D` devient un démon ajouté au pool d'adversaires. Force le bot à devenir
robuste contre ses propres faiblesses.

Stockage : league.json = liste de dicts {weights, win_rate_vs_best, gen}.
"""

import json
import math
from pathlib import Path


class League:
    def __init__(self, path="league.json", max_demons=8):
        self.path = Path(path)
        self.max_demons = max_demons
        self.demons = []
        if self.path.exists():
            try:
                self.demons = json.loads(self.path.read_text())
            except Exception:
                self.demons = []

    def add(self, weights, win_rate, gen):
        self.demons.append({
            "weights": list(weights),
            "win_rate_vs_best": float(win_rate),
            "gen": int(gen),
        })
        # Garde les meilleurs démons (plus fort win-rate contre best)
        self.demons.sort(key=lambda d: d["win_rate_vs_best"], reverse=True)
        self.demons = self.demons[: self.max_demons]
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.demons, indent=2))

    def all_weights(self):
        return [d["weights"] for d in self.demons]

    def __len__(self):
        return len(self.demons)


def discover_demon(target_weights, evaluate_vs_target, sigma0=0.4,
                   maxgen=10, popsize=8, seed_weights=None):
    """Mini-CMA-ES qui cherche poids battant target_weights.

    evaluate_vs_target(candidate_weights) -> win_rate ∈ [0, 1] de candidate
    contre target. On veut maximiser ce win_rate.
    """
    try:
        import cma
    except ImportError:
        print("  cma absent — skip demon discovery")
        return None, 0.0

    start = list(seed_weights) if seed_weights is not None else list(target_weights)
    es = cma.CMAEvolutionStrategy(
        start, sigma0,
        {"maxiter": maxgen, "popsize": popsize, "verbose": -9, "tolx": 1e-3},
    )

    best_w, best_wr = None, 0.0
    while not es.stop():
        cand = es.ask()
        scores = [evaluate_vs_target(c) for c in cand]
        es.tell(cand, [-s for s in scores])
        i = max(range(len(scores)), key=lambda k: scores[k])
        if scores[i] > best_wr:
            best_wr = scores[i]
            best_w = list(cand[i])
    return best_w, best_wr
