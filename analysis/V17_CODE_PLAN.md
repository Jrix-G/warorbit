# V17 — AlphaZero bot: code plan

Self-play RL bot. A neural net (policy + value) guides an MCTS; the net is
trained on the MCTS's own self-play games; iterate. Warm-started by cloning
V15 so it begins at ~975 ELO instead of from random.

## Modules

### Reused (built & validated)
- `v15_gpu_sim.py`   — batched torch engine (training). Bit-exact, compiled.
- `v15_fast_sim.py`  — numpy engine (CPU deployment).
- `v15_search.py`    — V15 (RCC+V7), the warm-start teacher and the benchmark.

### New — V17

**`v17_encode.py`** — state <-> tensors.
- `encode(state)` -> planet features [N,~12], fleet features, global features.
- `decode(policy_head, state)` -> a real action: per owned planet a target ->
  `[src_id, intercept_angle, ships]` (ships = capture-sized heuristic).
- Handles the variable planet count and the combinatorial action space.

**`v17_net.py`** — the neural network (small, torch).
- Per-planet encoder (shared MLP) + deep-sets context (pool -> concat) so each
  planet embedding sees the global situation.
- POLICY head: per owned planet, a softmax over candidate targets (+ "pass").
  This is what makes the variable/combinatorial action space tractable.
- VALUE head: pooled embedding -> value. 2p: scalar. 4p: per-player P(1st).
- ~few hundred K params — fits 6 GB, fast to evaluate millions of times.

**`v17_mcts.py`** — Monte-Carlo Tree Search.
- PUCT selection (policy prior + value), expand one node/sim, evaluate the
  leaf with `v17_net` (no rollouts), backup.
- VECTORISED across many games at once (the GPU-throughput key).
- Dirichlet noise at the root, temperature for move selection.
- 2p: minimax-style. 4p: maxⁿ (each node maximises the mover's value).

**`v17_selfplay.py`** — self-play data generation.
- Run B games; every move chosen by `v17_mcts`.
- Record per move: (state, MCTS visit distribution pi, eventual outcome z).

**`v17_train.py`** — network training step.
- Loss = cross-entropy(policy, pi) + MSE(value, z), on a replay buffer.

**`v17_warmstart.py`** — behavioural cloning.
- Generate V15 self-play games, train `v17_net`'s policy to imitate V15's
  moves -> the net starts at ~V15 strength, not random. Saves ~2-3 weeks.

**`v17_loop.py`** — the orchestrator.
- warm-start -> iterate { self-play -> train -> gate (new net vs old/V15) }.
- Checkpoint/resume every iteration (the run is days long).

**`v17_agent.py`** — deployment.
- Load the trained net, run `v17_mcts` on CPU (numpy engine) within ~1 s/move
  (simulation budget tuned). The Kaggle submission entry point.
- Packaging: bundle agent + net weights into one submission file.

## Data flow

```
            state
              | v17_encode.encode
              v
   planet / fleet / global tensors
              | v17_net
              v
     policy pi_prior ,  value v
              |
   v17_mcts: PUCT search, pi_prior as prior, v at leaves
              v
       visit counts  --v17_encode.decode-->  action
```

Training:
```
v17_selfplay  ->  (state, MCTS visit-dist pi, outcome z)  records
              ->  v17_train: policy->pi (CE), value->z (MSE)
              ->  new net  ->  v17_loop gates it  ->  next iteration
```

## Build order & gates

| Phase | Modules built | Gate |
|-------|---------------|------|
| 0 | v17_encode, v17_net, v17_warmstart | cloned net plays ~= V15 (+/-50 ELO) |
| 1 | v17_mcts, v17_selfplay, v17_train, v17_loop (2p) | after 5 iterations, AZ-2p > V15 |
| 2 | — (run the 2p loop) | 2p ELO plateau |
| 3 | v17_mcts maxⁿ (4p), 4p run | AZ-4p >> V15 in 4p |
| 4 | v17_agent, packaging | <1 s/move, runs on Kaggle |

Each gate is a measured benchmark — a failed gate stops the work before the
multi-day compute is spent.

## Key design choices (and why)

- **Per-planet policy head** — the only tractable way to handle a variable,
  combinatorial action space without a giant fixed softmax.
- **No rollouts, value-net leaves** — the AlphaZero choice; faster, lower
  variance than rollouts.
- **Warm-start from V15** — start the climb at 975, not at random play.
- **Vectorised MCTS** — batch the tree search across games so the GPU is fed.
- **MCTS averages the value (visit counts), never argmaxes it** — this is why
  the value net does NOT Goodhart, unlike the failed supervised-VF approach.
