"""Entraînement CMA-ES pour optimiser les poids du bot.

Usage:
    python3 train.py              # lance CMA-ES, sauvegarde meilleurs poids
    python3 train.py --quick      # entraînement court pour itérer
    python3 train.py --medium     # entraînement intermédiaire
    python3 train.py --long       # entraînement long
    python3 train.py --jobs 4     # force 4 processus parallèles
    python3 train.py --test       # teste bot avec poids par défaut vs baselines
    python3 train.py --test-best  # teste bot avec meilleurs poids sauvegardés

Principe:
    CMA-ES explore l'espace des 14 poids. Chaque candidat joue N parties
    en self-play + vs baselines. Le win-rate devient le score à maximiser.
    CMA-ES ajuste sa distribution de recherche selon les résultats.

Durée estimée: dépend du profil et de --jobs.
"""

import sys
import json
import math
import random
import argparse
import os
from multiprocessing import get_context
from pathlib import Path

from kaggle_environments import make
import bot


WEIGHTS_FILE = Path("best_weights.json")
N_GAMES_EVAL = 10   # parties par candidat pendant entraînement (rapide)
N_GAMES_TEST = 30   # parties pour test final (précis)
SIGMA0       = 0.3  # bruit initial CMA-ES (à réduire si déjà bon point de départ)
MAX_GEN      = 300  # générations max
POPSIZE      = 12   # candidats par génération
JOBS         = max(1, min(os.cpu_count() or 1, POPSIZE))

PROFILES = {
    "quick": {
        "N_GAMES_EVAL": 2,
        "N_GAMES_TEST": 10,
        "MAX_GEN": 20,
        "POPSIZE": 6,
        "SIGMA0": 0.3,
    },
    "medium": {
        "N_GAMES_EVAL": 5,
        "N_GAMES_TEST": 20,
        "MAX_GEN": 80,
        "POPSIZE": 8,
        "SIGMA0": 0.25,
    },
    "long": {
        "N_GAMES_EVAL": 10,
        "N_GAMES_TEST": 30,
        "MAX_GEN": 300,
        "POPSIZE": 12,
        "SIGMA0": 0.3,
    },
}


def apply_profile(profile):
    """Applique un profil de vitesse/précision à la config globale."""
    global N_GAMES_EVAL, N_GAMES_TEST, SIGMA0, MAX_GEN, POPSIZE, JOBS

    if profile is None:
        return

    cfg = PROFILES[profile]
    N_GAMES_EVAL = cfg["N_GAMES_EVAL"]
    N_GAMES_TEST = cfg["N_GAMES_TEST"]
    MAX_GEN = cfg["MAX_GEN"]
    POPSIZE = cfg["POPSIZE"]
    SIGMA0 = cfg["SIGMA0"]
    JOBS = max(1, min(os.cpu_count() or 1, POPSIZE))


def set_jobs(jobs):
    """Configure le nombre de processus worker."""
    global JOBS
    if jobs is not None:
        JOBS = max(1, jobs)


# ── Baselines ────────────────────────────────────────────────────────────────

def passive_agent(obs, config=None):
    return []


def random_agent(obs, config=None):
    planets = obs.get("planets", [])
    me = obs.get("player", 0)
    moves = []
    for p in planets:
        if p[1] == me and p[5] > 10:
            angle = random.uniform(0, 2 * math.pi)
            moves.append([p[0], angle, p[5] // 2])
    return moves


def greedy_agent(obs, config=None):
    """Greedy: attaque planète la plus proche."""
    planets = obs.get("planets", [])
    me = obs.get("player", 0)
    my = [p for p in planets if p[1] == me]
    others = [p for p in planets if p[1] != me]
    moves = []
    for src in my:
        if src[5] < 10 or not others:
            continue
        tgt = min(others, key=lambda t: math.hypot(t[2] - src[2], t[3] - src[3]))
        angle = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
        moves.append([src[0], angle, src[5] // 2])
    return moves


# ── Évaluation ───────────────────────────────────────────────────────────────

def make_agent_with_weights(weights):
    """Retourne un agent qui utilise des poids spécifiques."""
    def _agent(obs, config=None):
        old = bot.WEIGHTS
        bot.WEIGHTS = weights
        result = bot.agent(obs, config)
        bot.WEIGHTS = old
        return result
    return _agent


def play_game(agent_a, agent_b):
    """Joue 1 partie, retourne (reward_a, reward_b)."""
    env = make("orbit_wars", debug=False)
    env.run([agent_a, agent_b])
    last = env.steps[-1]
    ra = last[0].get("reward", 0) or 0
    rb = last[1].get("reward", 0) or 0
    return ra, rb


def evaluate_weights(weights, n_games=N_GAMES_EVAL, verbose=False):
    """
    Évalue un vecteur de poids.
    Retourne score entre 0 et 1 (win-rate pondéré).
    """
    candidate = make_agent_with_weights(weights)
    opponents = [
        ("greedy",   greedy_agent,   1.0),  # (nom, agent, poids dans score)
        ("self",     bot.agent,      2.0),  # self-play compte double
        ("random",   random_agent,   0.5),
    ]

    wins, total_weight = 0.0, 0.0
    for opp_name, opp, w in opponents:
        w_count = 0
        for i in range(n_games):
            if i % 2 == 0:
                ra, rb = play_game(candidate, opp)
            else:
                rb, ra = play_game(opp, candidate)
            if ra > rb:
                w_count += 1
        wr = w_count / n_games
        if verbose:
            print(f"  vs {opp_name:8s}: {w_count}/{n_games}  wr={wr:.0%}")
        wins += wr * w
        total_weight += w

    score = wins / total_weight
    return score


def evaluate_candidate_worker(args):
    """Worker multiprocessing: évalue un candidat CMA-ES."""
    weights, n_games = args
    return evaluate_weights(list(weights), n_games=n_games)


def evaluate_candidates(candidates):
    """Évalue les candidats en parallèle si JOBS > 1."""
    if JOBS <= 1 or len(candidates) <= 1:
        return [evaluate_weights(c, n_games=N_GAMES_EVAL) for c in candidates]

    worker_args = [(list(c), N_GAMES_EVAL) for c in candidates]
    ctx = get_context("fork")
    with ctx.Pool(processes=min(JOBS, len(candidates))) as pool:
        return pool.map(evaluate_candidate_worker, worker_args)


def get_opponent(name):
    if name == "passive":
        return passive_agent
    if name == "random":
        return random_agent
    if name == "greedy":
        return greedy_agent
    if name == "self":
        return bot.agent
    raise ValueError(f"Adversaire inconnu: {name}")


def play_test_game_worker(args):
    """Worker multiprocessing: joue une partie de test."""
    weights, opp_name, game_idx = args
    candidate = make_agent_with_weights(weights)
    opponent = get_opponent(opp_name)

    if game_idx % 2 == 0:
        ra, rb = play_game(candidate, opponent)
    else:
        rb, ra = play_game(opponent, candidate)
    return 1 if ra > rb else 0


# ── CMA-ES ───────────────────────────────────────────────────────────────────

def run_cma_es():
    try:
        import cma
    except ImportError:
        print("Installe cma: pip install cma")
        sys.exit(1)

    # Point de départ: meilleurs poids sauvegardés ou défaut
    if WEIGHTS_FILE.exists():
        with open(WEIGHTS_FILE) as f:
            data = json.load(f)
        start = data["weights"]
        best_score = data.get("score", 0.0)
        print(f"Reprise depuis {WEIGHTS_FILE} (score={best_score:.3f})")
    else:
        start = list(bot.DEFAULT_W)
        best_score = 0.0
        print("Départ depuis poids par défaut")

    best_weights = list(start)

    es = cma.CMAEvolutionStrategy(
        start,
        SIGMA0,
        {
            "maxiter":    MAX_GEN,
            "popsize":    POPSIZE,  # candidats par génération
            "tolx":       1e-4,
            "verbose":    1,
            "bounds":     [
                # bornes min/max pour chaque poids
                [0.1, 0.1, 5.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.01, 0.3, 0.0, 0.5, 0.0],
                [5.0, 5.0, 100.0, 2.0, 0.5, 3.0, 1.0, 1.0, 2.0, 0.5, 2.0, 1.0, 3.0, 1.0],
            ],
        }
    )

    gen = 0
    print(
        f"Config: gen={MAX_GEN}, popsize={POPSIZE}, "
        f"games_eval={N_GAMES_EVAL}, jobs={JOBS}"
    )
    while not es.stop():
        gen += 1
        candidates = es.ask()

        # Évaluer tous les candidats. Chaque candidat est indépendant.
        scores = evaluate_candidates(candidates)
        for i, score in enumerate(scores):
            print(f"  gen={gen:3d} cand={i+1:2d}/{len(candidates)} score={score:.3f}")

        # CMA-ES minimise → on passe -score
        es.tell(candidates, [-s for s in scores])

        # Sauvegarder si amélioré
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best_idx] > best_score:
            best_score   = scores[best_idx]
            best_weights = list(candidates[best_idx])
            save_weights(best_weights, best_score)
            print(f"  ✓ Nouveau meilleur: score={best_score:.3f}")
            print(f"  Poids: {[round(w, 3) for w in best_weights]}")

        print(f"Gen {gen}: best_this_gen={scores[best_idx]:.3f} overall_best={best_score:.3f}")

    print(f"\nEntraînement terminé. Meilleur score: {best_score:.3f}")
    print(f"Poids sauvegardés dans {WEIGHTS_FILE}")
    return best_weights, best_score


def save_weights(weights, score):
    with open(WEIGHTS_FILE, "w") as f:
        json.dump({"weights": weights, "score": score}, f, indent=2)


def load_best_weights():
    if not WEIGHTS_FILE.exists():
        print(f"Pas de fichier {WEIGHTS_FILE}. Lance d'abord python3 train.py")
        sys.exit(1)
    with open(WEIGHTS_FILE) as f:
        data = json.load(f)
    return data["weights"], data.get("score", 0.0)


# ── Tests ─────────────────────────────────────────────────────────────────────

def run_test(weights=None, label="défaut"):
    if weights is None:
        weights = bot.DEFAULT_W

    print(f"\nTest bot avec poids {label} ({N_GAMES_TEST} parties/adversaire, jobs={JOBS})\n")

    candidate = make_agent_with_weights(weights)
    opponents = [
        "passive",
        "random",
        "greedy",
        "self",
    ]

    for opp_name in opponents:
        if JOBS <= 1 or N_GAMES_TEST <= 1:
            opp = get_opponent(opp_name)
            wins = 0
            for i in range(N_GAMES_TEST):
                if i % 2 == 0:
                    ra, rb = play_game(candidate, opp)
                else:
                    rb, ra = play_game(opp, candidate)
                if ra > rb:
                    wins += 1
        else:
            worker_args = [(list(weights), opp_name, i) for i in range(N_GAMES_TEST)]
            ctx = get_context("fork")
            with ctx.Pool(processes=min(JOBS, N_GAMES_TEST)) as pool:
                wins = sum(pool.map(play_test_game_worker, worker_args))

        wr = wins / N_GAMES_TEST
        print(f"vs {opp_name:12s}: {wins:2d}W/{N_GAMES_TEST}  win-rate={wr:.0%}")


# ── Générer bot_submit.py avec poids hardcodés ───────────────────────────────

def generate_submit_bot(weights):
    """Copie bot.py en remplaçant WEIGHTS = None par les poids appris."""
    with open("bot.py") as f:
        src = f.read()

    weights_str = "[" + ", ".join(f"{w:.4f}" for w in weights) + "]"
    src = src.replace("WEIGHTS = None  # sera remplacé par CMA-ES",
                      f"WEIGHTS = {weights_str}  # poids CMA-ES")

    with open("bot_submit.py", "w") as f:
        f.write(src)
    print("bot_submit.py généré — c'est ce fichier qu'on soumet à Kaggle.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",        action="store_true", help="Test poids par défaut")
    parser.add_argument("--test-best",   action="store_true", help="Test meilleurs poids")
    parser.add_argument("--generate",    action="store_true", help="Génère bot_submit.py")
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument("--quick",  action="store_const", const="quick",  dest="profile",
                               help="Profil rapide: peu de parties/générations")
    profile_group.add_argument("--medium", action="store_const", const="medium", dest="profile",
                               help="Profil intermédiaire")
    profile_group.add_argument("--long",   action="store_const", const="long",   dest="profile",
                               help="Profil long: config historique")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Nombre de processus parallèles (défaut: min(CPU, popsize))")
    args = parser.parse_args()

    apply_profile(args.profile)
    set_jobs(args.jobs)

    if args.test:
        run_test(weights=None, label="défaut")

    elif args.test_best:
        w, score = load_best_weights()
        print(f"Poids chargés (score entraînement={score:.3f})")
        run_test(weights=w, label="best")
        generate_submit_bot(w)

    elif args.generate:
        w, score = load_best_weights()
        generate_submit_bot(w)
        print(f"Score entraînement: {score:.3f}")

    else:
        # Lance CMA-ES
        best_w, best_s = run_cma_es()
        print("\nTest final des meilleurs poids:")
        run_test(weights=best_w, label="best")
        generate_submit_bot(best_w)
