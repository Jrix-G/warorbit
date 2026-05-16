"""v15_gpu_selfplay — batched self-play + paired benchmark on the GPU.

Plays B Orbit Wars games at once on v15_gpu_sim, every player driven by the
GPU-native RCC search (v15_gpu_search). Used for two jobs of the self-play
value-iteration loop:

  * data generation — every player uses the same weight set; at sampled steps
    the position features of every live game/player are recorded and later
    labelled with the game winner.
  * paired benchmark — players use a mix of weight sets (genN vs gen N-1);
    the final scores decide which set won each game.

torch.compile (Triton) fuses the engine step — ~9 ms per game-turn for a
batch, vs ~9 s naive. Planet slots are padded to a fixed N so the kernel is
compiled once.
"""

from __future__ import annotations

import torch

import v14_core
import v15_fast_sim as fsim
import v15_gpu_sim as gsim
import v15_gpu_search as gsearch
from local_simulator.official_fast import OfficialFastGame

N_FIXED = 48          # planet-slot padding (>= any real map) -> one compile
M_MAX = 96            # fleet-slot capacity
SAMPLE_EVERY = 6
SKIP_HEAD = 8
SKIP_TAIL = 12

_COMPILED = False


def _ensure_compiled():
    global _COMPILED
    if not _COMPILED:
        gsim.step = torch.compile(gsim.step, dynamic=False)
        _COMPILED = True


def initial_states(n_players, n_games, seed_offset, warm_steps=0):
    """Draw n_games fresh maps as comet-free FastStates at step 0."""
    states = []
    for k in range(n_games):
        seed = seed_offset + k
        g = OfficialFastGame(n_players, seed=seed, episode_steps=500,
                             use_c_accel=False)
        obs = v14_core.obs_as_dict(g.observation(0))
        fs = fsim.from_obs(obs, n_players=n_players)
        fs.n_players = n_players
        for _ in range(warm_steps):
            fs = fsim.step(fs, [[] for _ in range(n_players)])
        states.append(fs)
    return states


def play_batch(states, weights_by_player, *, horizon=24, device="cuda",
               collect=True, max_steps=500, explore=0.0):
    """Play one batch of games to the end.

    weights_by_player[p] — EvalWeights player p searches with.
    explore — per-game random-move probability (data generation only; 0 for
              benchmarking).
    Returns (samples, scores):
      samples — list of (n_players, features[11], win_label) if collect else []
      scores  — [B, n_players] final ship scores (numpy).
    """
    _ensure_compiled()
    batch = gsim.from_faststates(states, device=device, m_max=M_MAX,
                                 dtype=torch.float32, n_fixed=N_FIXED)
    P = batch.n_players
    B = batch.B
    recorded = []   # (features[B,11] cpu, player, alive_mask[B] cpu)

    for t in range(max_steps):
        if bool(batch.done.all()):
            break
        if collect and t >= SKIP_HEAD and t % SAMPLE_EVERY == 0 \
                and t < max_steps - SKIP_TAIL:
            alive = (~batch.done).cpu()
            for p in range(P):
                feats = gsearch.batch_features(batch, p).cpu()
                recorded.append((feats, p, alive))
        moves = [gsearch.gpu_search(batch, p, weights_by_player[p],
                                    horizon=horizon, explore=explore)
                 for p in range(P)]
        actions = torch.stack(moves, dim=1)            # [B,P,A,3]
        batch = gsim.step(batch, actions)

    sc = gsim.scores(batch).cpu().numpy()              # [B,P]
    best = sc.max(axis=1)
    winner = []
    for b in range(B):
        w = [p for p in range(P) if sc[b, p] == best[b] and best[b] > 0]
        winner.append(w[0] if len(w) == 1 else -1)

    samples = []
    if collect:
        for feats, p, alive in recorded:
            fa = feats.numpy()
            for b in range(B):
                if alive[b]:
                    samples.append((P, fa[b],
                                    1.0 if winner[b] == p else 0.0))
    return samples, sc
