"""v15_selfplay — RCC self-play games on v15_fast_sim (no official harness).

Generates training data for the self-play value-iteration loop. A game is
played entirely on the validated v15_fast_sim engine; only the initial map is
drawn from OfficialFastGame (one cheap reset — the slow official step loop is
never used). At sampled steps the position features of every player are
recorded; at game end each sample is labelled with whether that player
finished strictly first.

This is the same supervised target as build_value_dataset.py, but on RCC's
own self-play distribution — so a value function fitted on it has no
distribution shift when RCC later uses it to score quiet leaf positions.
"""

from __future__ import annotations

import v14_core
import v15_eval
import v15_fast_sim as fsim
import v15_search
from local_simulator.official_fast import OfficialFastGame


def initial_state(n_players: int, seed: int,
                  episode_steps: int = 500) -> fsim.FastState:
    """Draw a fresh Orbit Wars map and return it as a FastState at step 0."""
    g = OfficialFastGame(n_players, seed=seed, episode_steps=episode_steps,
                         use_c_accel=False)
    obs = v14_core.obs_as_dict(g.observation(0))
    fs = fsim.from_obs(obs, n_players=n_players, episode_steps=episode_steps)
    fs.n_players = n_players
    return fs


def play_game(n_players: int, seed: int, weights_by_player: list, *,
              episode_steps: int = 500, time_budget: float = 0.7,
              horizon: int = 24, sample_every: int = 6,
              skip_head: int = 8, skip_tail: int = 12):
    """Play one RCC self-play game.

    weights_by_player[p] — the EvalWeights player p searches with.
    Returns (samples, scores) where samples is a list of
    (n_players, features[5], win_label) and scores is the final ship score.
    """
    fs = initial_state(n_players, seed, episode_steps)
    raw: list = []   # (n_players, features, player, step)
    while not fs.done:
        if (fs.step >= skip_head
                and fs.step < episode_steps - skip_tail
                and fs.step % sample_every == 0):
            for p in range(n_players):
                raw.append((n_players, v15_eval.features(fs, p), p))
        actions = []
        for p in range(n_players):
            obs = v15_search.state_to_obs(fs, p)
            mv = v15_search.search(obs, None, time_budget=time_budget,
                                   horizon=horizon,
                                   weights=weights_by_player[p])
            actions.append(mv if isinstance(mv, list) else [])
        fs = fsim.step(fs, actions)

    sc = fsim.scores(fs)
    best = max(sc) if sc else 0
    winners = [p for p in range(n_players) if sc[p] == best and best > 0]
    sole = winners[0] if len(winners) == 1 else -1

    samples = [(n, feat, 1.0 if p == sole else 0.0)
               for (n, feat, p) in raw]
    return samples, sc


if __name__ == "__main__":
    # quick self-check: one 2p and one 4p game
    import time
    for n in (2, 4):
        w = [v15_eval.ESC] * n
        t = time.monotonic()
        samples, sc = play_game(n, seed=99, weights_by_player=w,
                                episode_steps=200)
        dt = time.monotonic() - t
        wins = sum(s[2] for s in samples)
        print(f"{n}p: {len(samples)} samples ({wins:.0f} positive), "
              f"scores={sc}, {dt:.0f}s")
