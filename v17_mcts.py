"""v17_mcts — single-agent PUCT MCTS for the V17 AlphaZero bot.

Orbit Wars has SIMULTANEOUS moves, so a classic alternating game tree does not
apply. Design: the tree plans only OUR moves; opponents are folded into the
environment, modelled by the network's own greedy policy (self-consistent —
they play the same net). This makes the search single-agent: plain PUCT, plain
averaging backup, no negamax / maxⁿ — and the SAME code serves 2p and 4p.

Per node:
  * the net gives a value and a per-planet policy,
  * children = a handful of candidate full-moves sampled from that policy
    (plus the greedy move and the do-nothing move, always),
  * a child edge is taken by simulating one engine step (our move + the
    opponents' net-greedy moves),
  * leaves are evaluated by the net value (no rollout); terminal leaves by the
    true game result.

Re-planning every turn bounds the opponent-model error to one committed move.

`mcts_move` returns the chosen action AND the per-planet visit-marginal
policy — the training target for the network's policy head.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import v15_fast_sim as fsim
import v17_encode as enc
from v17_net import policy_probs

C_PUCT = 1.5
K_CHILDREN = 10
DIRICHLET_ALPHA = 0.6
DIRICHLET_FRAC = 0.25

# Optional batched-inference evaluator (set per worker by v17_infer).
# When set, `_net_eval` delegates to it instead of running a local forward.
_EVALUATOR = None


def set_evaluator(ev) -> None:
    """Install (or clear with None) the inference evaluator for this process."""
    global _EVALUATOR
    _EVALUATOR = ev


def _net_eval(net, fs, player, device):
    """Net forward on a single state -> (policy [N,N+1], value scalar)."""
    if _EVALUATOR is not None:
        return _EVALUATOR.eval(fs, player)
    pf, gf = enc.encode(fs, player)
    pft = torch.as_tensor(pf[None], device=device)
    gft = torch.as_tensor(gf[None], device=device)
    mask = torch.ones(1, len(pf), dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, value = net(pft, gft, mask)
    return policy_probs(logits)[0].cpu().numpy(), float(value[0])


def _greedy_targets(probs, fs, player):
    """Per owned planet, the argmax action -> target index (or -1 = pass)."""
    n = len(fs.planets)
    targets = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if int(fs.planets[i, enc.OWNER]) == player:
            a = int(probs[i].argmax())
            targets[i] = a - 1                    # 0=pass -> -1
    return targets


def _candidate_moves(probs, fs, player, k, rng):
    """k+2 candidate moves: greedy, all-pass, and k policy samples.
    A move is a target array [N]; returns list of (targets, prior_logprob)."""
    n = len(fs.planets)
    owned = [i for i in range(n)
             if int(fs.planets[i, enc.OWNER]) == player]
    moves = {}

    def _add(targets):
        key = targets.tobytes()
        if key in moves:
            return
        lp = 0.0
        for i in owned:
            a = int(targets[i]) + 1
            lp += math.log(probs[i, a] + 1e-9)
        moves[key] = (targets, lp)

    _add(_greedy_targets(probs, fs, player))      # greedy
    _add(np.full(n, -1, dtype=np.int64))          # do-nothing
    for _ in range(k):
        t = np.full(n, -1, dtype=np.int64)
        for i in owned:
            p = probs[i]
            a = int(rng.choice(len(p), p=p / p.sum()))
            t[i] = a - 1
        _add(t)
    return list(moves.values())


class _Node:
    __slots__ = ("fs", "player", "value", "moves", "actions",
                 "P", "N", "W", "child", "opp_actions")

    def __init__(self, fs, player, net, device, rng, root=False):
        self.fs = fs
        self.player = player
        if fs.done:
            sc = fsim.scores(fs)
            best = max(sc) if sc else 0
            win = [p for p in range(fs.n_players) if sc[p] == best and best > 0]
            self.value = (1.0 if (len(win) == 1 and win[0] == player)
                          else (-1.0 if win and player not in win else 0.0))
            self.moves = []
            self.opp_actions = None
            return
        probs, val = _net_eval(net, fs, player, device)
        self.value = val
        cands = _candidate_moves(probs, fs, player, K_CHILDREN, rng)
        self.moves = [t for (t, _) in cands]
        lp = np.array([l for (_, l) in cands], dtype=np.float64)
        pri = np.exp(lp - lp.max())
        pri = pri / pri.sum()
        if root:                                  # exploration noise at root
            noise = rng.dirichlet([DIRICHLET_ALPHA] * len(pri))
            pri = (1 - DIRICHLET_FRAC) * pri + DIRICHLET_FRAC * noise
        self.P = pri
        self.N = np.zeros(len(pri))
        self.W = np.zeros(len(pri))
        self.actions = [enc.decode_move(fs, player, t) for t in self.moves]
        self.child = [None] * len(pri)
        # G1: opponents' greedy moves are deterministic on this (fixed) state,
        # so compute them once here instead of once per child expansion.
        self.opp_actions = _opponent_actions(fs, player, net, device)


def _opponent_actions(fs, our_player, net, device):
    """Net-greedy move for every player except ours (the opponent model)."""
    acts = [[] for _ in range(fs.n_players)]
    for q in range(fs.n_players):
        if q == our_player:
            continue
        probs, _ = _net_eval(net, fs, q, device)
        acts[q] = enc.decode_move(fs, q, _greedy_targets(probs, fs, q))
    return acts


def _simulate(node, net, device, rng):
    """One PUCT simulation from `node`; return the value (our perspective)."""
    if node.fs.done or not node.moves:
        return node.value
    total = node.N.sum()
    u = (node.W / np.maximum(node.N, 1)
         + C_PUCT * node.P * math.sqrt(total + 1) / (1 + node.N))
    a = int(np.argmax(u))
    if node.child[a] is None:
        actions = list(node.opp_actions)        # G1: reuse cached opp moves
        actions[node.player] = node.actions[a]
        nxt = fsim.step(node.fs, actions)
        node.child[a] = _Node(nxt, node.player, net, device, rng)
        v = node.child[a].value
    else:
        v = _simulate(node.child[a], net, device, rng)
    node.N[a] += 1
    node.W[a] += v
    return v


def mcts_move(net, fs, player, *, n_sims=100, device="cpu", rng=None,
              temperature=1.0):
    """Run MCTS; return (action, visit_policy [N,N+1]).

    visit_policy is the per-planet visit-marginal — the training target for
    the network's policy head."""
    if rng is None:
        rng = np.random.default_rng()
    root = _Node(fs, player, net, device, rng, root=True)
    if not root.moves:
        return [], _uniform_policy(fs, player)
    for _ in range(n_sims):
        _simulate(root, net, device, rng)

    visits = root.N
    if temperature <= 1e-3:
        pick = int(np.argmax(visits))
    else:
        w = visits ** (1.0 / temperature)
        s = w.sum()
        pick = int(rng.choice(len(w), p=w / s) if s > 0 else 0)

    # per-planet visit-marginal policy target
    n = len(fs.planets)
    pol = np.zeros((n, n + 1), dtype=np.float32)
    vs = visits.sum()
    if vs > 0:
        for mi, targets in enumerate(root.moves):
            wv = visits[mi] / vs
            for i in range(n):
                if int(fs.planets[i, enc.OWNER]) == player:
                    pol[i, int(targets[i]) + 1] += wv
    return root.actions[pick], pol


def _uniform_policy(fs, player):
    n = len(fs.planets)
    pol = np.zeros((n, n + 1), dtype=np.float32)
    pol[:, 0] = 1.0
    return pol


if __name__ == "__main__":
    import random
    import v14_core
    from local_simulator.official_fast import OfficialFastGame
    from v17_net import V17Net

    net = V17Net(d=64)
    net.eval()
    for npl in (2, 4):
        random.seed(3)
        np.random.seed(3)
        g = OfficialFastGame(npl, seed=3, episode_steps=300, use_c_accel=False)
        for _ in range(50):
            g.step([[] for _ in range(npl)])
        obs = v14_core.obs_as_dict(g.observation(0))
        fs = fsim.from_obs(obs, n_players=npl)
        fs.n_players = npl
        rng = np.random.default_rng(0)
        import time
        t = time.monotonic()
        action, pol = mcts_move(net, fs, 0, n_sims=80, rng=rng)
        dt = time.monotonic() - t
        assert isinstance(action, list)
        assert pol.shape == (len(fs.planets), len(fs.planets) + 1)
        owned = [i for i in range(len(fs.planets))
                 if int(fs.planets[i, 1]) == 0]
        for i in owned:
            assert abs(pol[i].sum() - 1.0) < 1e-4, pol[i].sum()
        print(f"{npl}p: 80 sims {dt*1000:.0f}ms  action={len(action)} launches"
              f"  policy rows OK")
    print("v17_mcts self-check passed")
