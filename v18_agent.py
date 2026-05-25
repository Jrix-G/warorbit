"""v18_agent — search-at-inference agent: MCTS driven by the SUPERVISED nets.

The V17 self-play loop collapsed (the net learned to pass). But the supervised
nets are trained on real data and do not collapse:
  * wo_policy  (WOPolicyNet)  — imitation of strong players' targets
  * wo_value   (WOValueNet)   — regression on real game outcomes, [-1,1]

This module wraps both into a single evaluator with the interface v17_mcts
expects (`.eval(fs, player) -> (policy_probs[N,N+1], value)`), so the existing
MCTS can search with a non-collapsed prior + value. No training required —
this is the plan's component 3 (search at inference).
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

import v15_eval
import v15_fast_sim as fsim
import v17_encode as enc
from wo_policy_net import WOPolicyNet
from wo_net import WOValueNet


class SupervisedEvaluator:
    """Policy + value evaluator backed by the two supervised nets.

    eval(fs, player) -> (probs [N,N+1] float32, value float in [-1,1]).
    probs[i,0] = pass prob for planet i; probs[i,j+1] = attack-planet-j prob.
    """

    def __init__(self, policy_ckpt: str = "analysis/wo_policy.pt",
                 value_ckpt: str = "analysis/wo_value.pt",
                 device: str = "cpu"):
        torch.set_num_threads(1)
        self.device = device

        cp = torch.load(policy_ckpt, map_location=device, weights_only=False)
        self.policy = WOPolicyNet(d=int(cp["d"]))
        self.policy.load_state_dict(cp["state_dict"])
        self.policy.eval().to(device)

        cv = torch.load(value_ckpt, map_location=device, weights_only=False)
        if bool(cv.get("residual", False)):
            raise ValueError(
                f"{value_ckpt} is a residual net; MCTS needs an absolute "
                "outcome net (train/use a non-residual one).")
        self.value = WOValueNet(d=int(cv["d"]))
        self.value.load_state_dict(cv["state_dict"])
        self.value.eval().to(device)

    def eval(self, fs, player: int):
        pf, gf = enc.encode(fs, player)
        n = pf.shape[0]
        if n == 0:
            return np.zeros((0, 1), dtype=np.float32), 0.0
        pft = torch.as_tensor(pf[None], dtype=torch.float32, device=self.device)
        gft = torch.as_tensor(gf[None], dtype=torch.float32, device=self.device)
        mask = torch.ones(1, n, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits = self.policy(pft, gft, mask)          # [1,N,N+1]
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            v = float(self.value(pft, gft, mask)[0])
        return probs.astype(np.float32), v


class RolloutESCEvaluator:
    """Policy prior from wo_policy; value from a det-policy rollout + ESC.

    The leaf value CANNOT collapse: it is not learned. From the leaf state we
    roll `rollout` steps with v15's deterministic continuation (both sides),
    then score with the fixed ESC. A passive line rolls out to the opponent
    visibly gaining planets/production -> low ESC -> the MCTS avoids it. This
    is v15's `_eval_combo` recipe applied at every MCTS leaf, adding tree
    depth on top of v15-grade evaluation.

    pass_weight < 1 de-emphasises the 'pass' action in the policy prior so the
    MCTS samples launch candidates (the prior from imitation is pass-heavy).
    """

    def __init__(self, policy_ckpt: str = "analysis/wo_policy.pt",
                 device: str = "cpu", rollout: int = 22,
                 pass_weight: float = 0.35):
        torch.set_num_threads(1)
        self.device = device
        self.rollout = rollout
        self.pass_weight = pass_weight
        cp = torch.load(policy_ckpt, map_location=device, weights_only=False)
        self.policy = WOPolicyNet(d=int(cp["d"]))
        self.policy.load_state_dict(cp["state_dict"])
        self.policy.eval().to(device)

    def eval(self, fs, player: int):
        pf, gf = enc.encode(fs, player)
        n = pf.shape[0]
        if n == 0:
            return np.zeros((0, 1), dtype=np.float32), 0.0
        pft = torch.as_tensor(pf[None], dtype=torch.float32, device=self.device)
        gft = torch.as_tensor(gf[None], dtype=torch.float32, device=self.device)
        mask = torch.ones(1, n, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits = self.policy(pft, gft, mask)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        probs = probs.astype(np.float32)
        if self.pass_weight != 1.0:
            probs[:, 0] *= self.pass_weight
            probs /= np.maximum(probs.sum(-1, keepdims=True), 1e-9)
        value = self._rollout_esc(fs, player)
        return probs, value

    def _rollout_esc(self, fs, player: int) -> float:
        # Passive continuation (no new launches) — v15's _eval_combo recipe.
        # An aggressive rollout washes out the move's signal and systematically
        # punishes attacking (depleted garrison shows immediately, the captured
        # planet only ~25 steps later). Quiescence lets pending fleets land.
        st = fs
        empty = [[] for _ in range(fs.n_players)]
        for _ in range(self.rollout):
            if st.done:
                break
            st = fsim.step(st, empty)
        esc = v15_eval.evaluate(st, player, v15_eval.ESC)
        return 2.0 * float(esc) - 1.0


if __name__ == "__main__":
    import v14_core
    from local_simulator.official_fast import OfficialFastGame

    ev = SupervisedEvaluator()
    rev = RolloutESCEvaluator()
    for npl in (2, 4):
        g = OfficialFastGame(npl, seed=2, episode_steps=250, use_c_accel=False)
        for _ in range(40):
            g.step([[] for _ in range(npl)])
        fs = fsim.from_obs(v14_core.obs_as_dict(g.observation(0)),
                           n_players=npl, episode_steps=250)
        fs.n_players = npl
        probs, val = ev.eval(fs, 0)
        assert probs.shape == (len(fs.planets), len(fs.planets) + 1)
        assert np.allclose(probs.sum(-1), 1.0, atol=1e-4)
        assert -1.0 <= val <= 1.0
        rprobs, rval = rev.eval(fs, 0)
        assert np.allclose(rprobs.sum(-1), 1.0, atol=1e-4)
        assert -1.0 <= rval <= 1.0
        print(f"{npl}p: supervised value={val:+.3f}  rollout-ESC value={rval:+.3f}")
    print("v18_agent self-check passed")
