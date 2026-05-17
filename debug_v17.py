"""Reverse-engineering debug: trace exactly what V17 does each turn."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import traceback
import numpy as np
import torch

import v14_core
import v15_fast_sim as fsim
import v15_search
import v17_encode as enc
from v17_mcts import mcts_move, _candidate_moves, _net_eval, _greedy_targets
from v17_net import V17Net, policy_probs
from local_simulator.official_fast import OfficialFastGame

CKPT = "analysis/v17_warmstart.pt"

print(f"=== Loading {CKPT} ===")
c = torch.load(CKPT, map_location="cpu")
net = V17Net(d=c["d"])
net.load_state_dict(c["state_dict"])
net.eval()
print(f"Net loaded: d={c['d']}")

rng = np.random.default_rng(42)

for seed in [9100000, 9100001, 9100002]:
    our = seed % 2
    print(f"\n=== seed={seed}  our=player{our} ===")
    g = OfficialFastGame(2, seed=seed, episode_steps=250, use_c_accel=False)
    obs0 = v14_core.obs_as_dict(g.observation(0))
    fs = fsim.from_obs(obs0, n_players=2, episode_steps=250)
    fs.n_players = 2

    for step in range(8):
        # 1) Raw network policy
        pf, gf = enc.encode(fs, our)
        pft = torch.as_tensor(pf[None])
        gft = torch.as_tensor(gf[None])
        mask = torch.ones(1, len(pf), dtype=torch.bool)
        with torch.no_grad():
            logits, val = net(pft, gft, mask)
        probs = policy_probs(logits)[0].numpy()  # [N, N+1]

        owned = [i for i in range(len(fs.planets))
                 if int(fs.planets[i, enc.OWNER]) == our]
        ships_owned = [int(fs.planets[i, enc.SHIPS]) for i in owned]
        n_attack_argmax = sum(1 for i in owned if probs[i].argmax() > 0)
        pass_probs = [float(probs[i, 0]) for i in owned[:5]]

        # 2) Candidate moves
        cands = _candidate_moves(probs, fs, our, k=10, rng=rng)
        greedy_targets = _greedy_targets(probs, fs, our)
        n_greedy_attacks = int((greedy_targets >= 0).sum())

        # 3) Direct decode_move on greedy targets
        greedy_action = enc.decode_move(fs, our, greedy_targets)

        # 4) Call mcts_move directly (NO try-except) to expose any exception
        try:
            action, pol = mcts_move(net, fs, our, n_sims=1, device="cpu",
                                    rng=rng, temperature=0.0)
        except Exception:
            print(f"  step {step}: EXCEPTION in mcts_move:")
            traceback.print_exc()
            break

        print(f"  step {step:2d}: owned={len(owned)} ships={ships_owned[:4]} "
              f"p(pass)[0:5]={[f'{p:.2f}' for p in pass_probs]} "
              f"net_atk={n_attack_argmax} greedy_atk={n_greedy_attacks} "
              f"greedy_decode={len(greedy_action)} "
              f"mcts_action={len(action)} val={float(val[0]):.3f}")
        if action:
            print(f"    launches: {action[:2]}")
        if greedy_action and not action:
            print(f"    *** greedy produces launches but mcts returns []!")

        # step with both passing to progress game
        v15_a = v15_search.search(v15_search.state_to_obs(fs, 1 - our), None)
        actions = [[], []]
        actions[our] = action if isinstance(action, list) else []
        actions[1 - our] = v15_a if isinstance(v15_a, list) else []
        fs = fsim.step(fs, actions)
        if fs.done:
            print(f"  game over at step {step}")
            break

print("\n=== DONE ===")
