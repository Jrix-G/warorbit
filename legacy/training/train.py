"""Entraînement CMA-ES — V3.

Nouveautés vs V2:
    --zoo          : utilise opponents/ZOO complet (8 adversaires) au lieu de 3
    --surrogate    : Gaussian Process accéléré, sample 4× la popsize, eval top-k
    --league       : demon discovery toutes les N générations (défaut N=10)
    --profile      : cProfile sur 5 parties, sort un rapport
    --opponents A,B,C : restreint le pool à une liste

Flags conservés:
    --quick / --medium / --long
    --jobs N
    --test / --test-best / --generate

Principe identique à V2 : CMA-ES sur 14 poids, score = win-rate pondéré,
sauvegarde dans best_weights.json.
"""

import sys
import json
import math
import random
import argparse
import os
import time
from multiprocessing import get_context
from pathlib import Path

from kaggle_environments import make
import bot
from opponents import ZOO, get as get_opp
from training.surrogate import Surrogate
from training.league import League, discover_demon


WEIGHTS_FILE = Path("best_weights.json")
N_GAMES_EVAL = 10
N_GAMES_TEST = 30
SIGMA0       = 0.3
MAX_GEN      = 300
POPSIZE      = 12
JOBS         = max(1, min(os.cpu_count() or 1, POPSIZE))

# Pool d'adversaires par défaut (compat V2)
DEFAULT_OPPONENT_NAMES = ["greedy", "self", "random"]
DEFAULT_OPPONENT_WEIGHTS = {"greedy": 1.0, "self": 2.0, "random": 0.5,
                             "passive": 0.3, "starter": 1.0,
                             "distance": 1.0, "sun_dodge": 1.0,
                             "structured": 1.5, "orbit_stars": 1.5}

# Pool étendu si --zoo
ZOO_OPPONENT_NAMES = ["greedy", "self", "starter", "distance", "sun_dodge",
                       "structured", "orbit_stars"]

PROFILES = {
    "rocket": {"N_GAMES_EVAL": 2,  "N_GAMES_TEST": 10, "MAX_GEN": 5,   "POPSIZE": 6,  "SIGMA0": 0.3},
    "quick":  {"N_GAMES_EVAL": 2,  "N_GAMES_TEST": 10, "MAX_GEN": 20,  "POPSIZE": 6,  "SIGMA0": 0.3},
    "medium": {"N_GAMES_EVAL": 5,  "N_GAMES_TEST": 20, "MAX_GEN": 80,  "POPSIZE": 8,  "SIGMA0": 0.25},
    "long":   {"N_GAMES_EVAL": 10, "N_GAMES_TEST": 30, "MAX_GEN": 300, "POPSIZE": 12, "SIGMA0": 0.3},
    "smoke":  {"N_GAMES_EVAL": 1,  "N_GAMES_TEST": 2,  "MAX_GEN": 2,   "POPSIZE": 4,  "SIGMA0": 0.3},
}


def apply_profile(profile):
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
    global JOBS
    if jobs is not None:
        JOBS = max(1, jobs)


# ── Évaluation ───────────────────────────────────────────────────────────────

def make_agent_with_weights(weights):
    def _agent(obs, config=None):
        old = bot.WEIGHTS
        bot.WEIGHTS = weights
        result = bot.agent(obs, config)
        bot.WEIGHTS = old
        return result
    return _agent


def play_game(agent_a, agent_b):
    env = make("orbit_wars", debug=False)
    env.run([agent_a, agent_b])
    last = env.steps[-1]
    ra = last[0].get("reward", 0) or 0
    rb = last[1].get("reward", 0) or 0
    return ra, rb


def _resolve_opponent(name):
    """Retourne agent callable. 'self' → bot.agent (avec poids défaut)."""
    if name == "self":
        return bot.agent
    return get_opp(name)


def evaluate_weights(weights, n_games=None, opponent_names=None, verbose=False):
    """Score = win-rate pondéré contre une liste d'adversaires."""
    if n_games is None:
        n_games = N_GAMES_EVAL
    if opponent_names is None:
        opponent_names = DEFAULT_OPPONENT_NAMES

    candidate = make_agent_with_weights(weights)
    wins, total_w = 0.0, 0.0
    for name in opponent_names:
        opp = _resolve_opponent(name)
        w = DEFAULT_OPPONENT_WEIGHTS.get(name, 1.0)
        won = 0
        for i in range(n_games):
            if i % 2 == 0:
                ra, rb = play_game(candidate, opp)
            else:
                rb, ra = play_game(opp, candidate)
            if ra > rb:
                won += 1
        wr = won / max(n_games, 1)
        if verbose:
            print(f"  vs {name:12s}: {won}/{n_games}  wr={wr:.0%}")
        wins += wr * w
        total_w += w
    return wins / max(total_w, 1e-9)


def evaluate_candidate_worker(args):
    weights, n_games, opp_names = args
    return evaluate_weights(list(weights), n_games=n_games, opponent_names=opp_names)


def evaluate_candidates(candidates, opp_names):
    if JOBS <= 1 or len(candidates) <= 1:
        return [evaluate_weights(c, n_games=N_GAMES_EVAL, opponent_names=opp_names)
                for c in candidates]
    args = [(list(c), N_GAMES_EVAL, opp_names) for c in candidates]
    # Sur Windows, fork n'existe pas → spawn. Plus lent au démarrage.
    ctx = get_context("spawn") if os.name == "nt" else get_context("fork")
    with ctx.Pool(processes=min(JOBS, len(candidates))) as pool:
        return pool.map(evaluate_candidate_worker, args)


# ── Demon discovery ─────────────────────────────────────────────────────────

def evaluate_vs_target_factory(target_weights, n_games=2):
    """Retourne fonction qui évalue win-rate de candidate vs target."""
    target_agent = make_agent_with_weights(target_weights)

    def _eval(candidate_weights):
        cand = make_agent_with_weights(list(candidate_weights))
        won = 0
        for i in range(n_games):
            if i % 2 == 0:
                ra, rb = play_game(cand, target_agent)
            else:
                rb, ra = play_game(target_agent, cand)
            if ra > rb:
                won += 1
        return won / n_games

    return _eval


# ── CMA-ES principal ────────────────────────────────────────────────────────

def run_cma_es(opp_names, use_surrogate=False, use_league=False, league_period=10):
    try:
        import cma
    except ImportError:
        print("Installe cma: pip install cma")
        sys.exit(1)

    if WEIGHTS_FILE.exists():
        data = json.loads(WEIGHTS_FILE.read_text())
        start = data["weights"]
        best_score = data.get("score", 0.0)
        print(f"Reprise depuis {WEIGHTS_FILE} (score={best_score:.3f})")
    else:
        start = list(bot.DEFAULT_W)
        best_score = 0.0
        print("Départ depuis poids par défaut")

    best_weights = list(start)

    es = cma.CMAEvolutionStrategy(
        start, SIGMA0,
        {
            "maxiter": MAX_GEN, "popsize": POPSIZE, "tolx": 1e-4, "verbose": 1,
            "bounds": [
                [0.1, 0.1, 5.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.01, 0.3, 0.0, 0.5, 0.0],
                [5.0, 5.0, 100.0, 2.0, 0.5, 3.0, 1.0, 1.0, 2.0, 0.5, 2.0, 1.0, 3.0, 1.0],
            ],
        }
    )

    surrogate = Surrogate() if use_surrogate else None
    league = League() if use_league else None

    print(f"Config: gen={MAX_GEN}, popsize={POPSIZE}, games={N_GAMES_EVAL}, "
          f"jobs={JOBS}, surrogate={use_surrogate}, league={use_league}")
    print(f"Adversaires: {opp_names}")
    if league is not None and len(league):
        print(f"  + {len(league)} demons preexistants")

    gen = 0
    while not es.stop():
        gen += 1
        t0 = time.perf_counter()

        # Étendre le pool avec les démons (1-2 sélectionnés au hasard par gen)
        active_opps = list(opp_names)
        if league is not None and len(league):
            demons = league.all_weights()
            picked = random.sample(demons, min(2, len(demons)))
            for i, dw in enumerate(picked):
                # Enregistre temporairement le démon comme agent custom
                name = f"_demon_{i}"
                ZOO[name] = make_agent_with_weights(dw)
                active_opps.append(name)
                DEFAULT_OPPONENT_WEIGHTS.setdefault(name, 1.5)

        # ── Sampling
        if surrogate and len(surrogate) >= 5:
            # Sample 4x la popsize, évalue top-popsize via surrogate
            big = es.ask(POPSIZE * 4)
            keep_idx = surrogate.select_promising(big, k_keep=POPSIZE - 2, k_explore=2)
            candidates = [big[i] for i in keep_idx]
        else:
            candidates = es.ask()

        scores = evaluate_candidates(candidates, active_opps)

        # ── Reporting
        for i, score in enumerate(scores):
            print(f"  gen={gen:3d} cand={i+1:2d}/{len(candidates)} score={score:.3f}")

        # CMA-ES a besoin de POPSIZE candidats. Si surrogate filtre, on remplit
        # avec interpolation : on ré-évalue les autres via prédiction du
        # surrogate pour donner un signal cohérent à CMA.
        if surrogate and len(candidates) != POPSIZE:
            # Fallback : ne donner que les évalués réels
            es.tell(candidates, [-s for s in scores])
        else:
            es.tell(candidates, [-s for s in scores])

        # Mémoire surrogate
        if surrogate:
            for w, s in zip(candidates, scores):
                surrogate.add(list(w), s)
            surrogate.fit()

        # Best update
        i_best = max(range(len(scores)), key=lambda i: scores[i])
        if scores[i_best] > best_score:
            best_score = scores[i_best]
            best_weights = list(candidates[i_best])
            save_weights(best_weights, best_score)
            print(f"  ✓ Nouveau meilleur: score={best_score:.3f}")

        elapsed = time.perf_counter() - t0
        print(f"Gen {gen}: best_gen={scores[i_best]:.3f} overall={best_score:.3f} "
              f"({elapsed:.1f}s)")

        # Demon discovery
        if league is not None and gen % league_period == 0 and gen > 0:
            print(f"  >> Demon discovery (vs current best)...")
            t1 = time.perf_counter()
            demon_w, demon_wr = discover_demon(
                best_weights,
                evaluate_vs_target_factory(best_weights, n_games=2),
                sigma0=0.4, maxgen=5, popsize=6,
            )
            if demon_w and demon_wr > 0.5:
                league.add(demon_w, demon_wr, gen)
                print(f"    + demon added: wr={demon_wr:.0%} "
                      f"({time.perf_counter()-t1:.1f}s)")
            else:
                print(f"    no demon found (wr max={demon_wr:.0%})")

        # Cleanup démons temporaires du ZOO
        for name in [n for n in ZOO if n.startswith("_demon_")]:
            del ZOO[name]
            DEFAULT_OPPONENT_WEIGHTS.pop(name, None)

    print(f"\nFini. Meilleur score: {best_score:.3f}")
    return best_weights, best_score


def save_weights(weights, score):
    WEIGHTS_FILE.write_text(json.dumps(
        {"weights": weights, "score": score}, indent=2))


def load_best_weights():
    if not WEIGHTS_FILE.exists():
        print(f"Pas de fichier {WEIGHTS_FILE}.")
        sys.exit(1)
    data = json.loads(WEIGHTS_FILE.read_text())
    return data["weights"], data.get("score", 0.0)


# ── Tests ───────────────────────────────────────────────────────────────────

def play_test_game_worker(args):
    weights, opp_name, game_idx = args
    candidate = make_agent_with_weights(weights)
    opponent = _resolve_opponent(opp_name)
    if game_idx % 2 == 0:
        ra, rb = play_game(candidate, opponent)
    else:
        rb, ra = play_game(opponent, candidate)
    return 1 if ra > rb else 0


def run_test(weights=None, label="défaut", opp_names=None):
    if weights is None:
        weights = bot.DEFAULT_W
    if opp_names is None:
        opp_names = ["passive", "random", "greedy", "starter", "self"]

    print(f"\nTest poids {label} ({N_GAMES_TEST} parties/adversaire, jobs={JOBS})\n")
    candidate = make_agent_with_weights(weights)
    for name in opp_names:
        if JOBS <= 1 or N_GAMES_TEST <= 1:
            opp = _resolve_opponent(name)
            wins = 0
            for i in range(N_GAMES_TEST):
                if i % 2 == 0:
                    ra, rb = play_game(candidate, opp)
                else:
                    rb, ra = play_game(opp, candidate)
                if ra > rb:
                    wins += 1
        else:
            args = [(list(weights), name, i) for i in range(N_GAMES_TEST)]
            ctx = get_context("spawn") if os.name == "nt" else get_context("fork")
            with ctx.Pool(processes=min(JOBS, N_GAMES_TEST)) as pool:
                wins = sum(pool.map(play_test_game_worker, args))
        wr = wins / N_GAMES_TEST
        print(f"vs {name:12s}: {wins:2d}W/{N_GAMES_TEST}  wr={wr:.0%}")


def run_profile():
    """cProfile sur N parties pour identifier hot paths."""
    import cProfile
    import pstats
    n_games = 5
    print(f"Profile sur {n_games} parties bot vs bot…")
    candidate = make_agent_with_weights(bot.DEFAULT_W)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n_games):
        play_game(candidate, bot.agent)
    pr.disable()
    stats = pstats.Stats(pr).sort_stats("cumulative")
    stats.print_stats(25)


# ── Génération bot_submit.py ────────────────────────────────────────────────

def generate_submit_bot(weights):
    src = Path("bot.py").read_text(encoding='utf-8')
    weights_str = "[" + ", ".join(f"{w:.4f}" for w in weights) + "]"
    src = src.replace("WEIGHTS = None  # sera remplacé par CMA-ES",
                      f"WEIGHTS = {weights_str}  # poids CMA-ES")
    Path("bot_submit.py").write_text(src, encoding='utf-8')
    print("bot_submit.py généré.")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",      action="store_true")
    parser.add_argument("--test-best", action="store_true")
    parser.add_argument("--generate",  action="store_true")
    parser.add_argument("--profile",   action="store_true",
                        help="cProfile sur 5 parties et exit")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--rocket", action="store_const", const="rocket", dest="profile_name",
                   help="5 gens ultra-rapide (~3-5 min)")
    g.add_argument("--quick",  action="store_const", const="quick",  dest="profile_name")
    g.add_argument("--medium", action="store_const", const="medium", dest="profile_name")
    g.add_argument("--long",   action="store_const", const="long",   dest="profile_name")
    g.add_argument("--smoke",  action="store_const", const="smoke",  dest="profile_name",
                   help="2 gens × 4 cand × 1 game (test scaffolding)")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--zoo",        action="store_true",
                        help="Pool étendu (8 adversaires) au lieu de 3")
    parser.add_argument("--surrogate",  action="store_true",
                        help="GP surrogate pour accélérer CMA-ES")
    parser.add_argument("--league",     action="store_true",
                        help="Demon discovery toutes les N gens")
    parser.add_argument("--league-period", type=int, default=10)
    parser.add_argument("--opponents", type=str, default=None,
                        help="Liste personnalisée: greedy,self,starter,…")
    args = parser.parse_args()

    apply_profile(args.profile_name)
    set_jobs(args.jobs)

    # Profile shortcut
    if args.profile:
        run_profile()
        sys.exit(0)

    # Pool d'adversaires
    if args.opponents:
        opp_names = [s.strip() for s in args.opponents.split(",")]
    elif args.zoo:
        opp_names = ZOO_OPPONENT_NAMES
    else:
        opp_names = DEFAULT_OPPONENT_NAMES

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
    else:
        best_w, best_s = run_cma_es(
            opp_names=opp_names,
            use_surrogate=args.surrogate,
            use_league=args.league,
            league_period=args.league_period,
        )
        print("\nTest final:")
        run_test(weights=best_w, label="best")
        generate_submit_bot(best_w)
