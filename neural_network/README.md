# Neural Network Training

This project now uses PyTorch end-to-end.

## Migration PyTorch

- `src/model.py` contains a real `torch.nn.Module`.
- Checkpoints use `state_dict()` / `load_state_dict()`.
- Training uses autograd, `loss.backward()`, and `optimizer.step()`.

## REINFORCE

- Episodes are collected in `src/self_play.py`.
- `src/trainer.py` stores `log_prob` and rewards.
- Returns are computed with `gamma`.
- Loss is `-sum(log_prob * return)`.

## Tests

```bash
python -m pytest neural_network/tests/ -v
```

## Short training

```bash
python neural_network/scripts/run_30min_analysis.py --duration-minutes 1
```

## 30 minute run

```bash
python neural_network/scripts/run_30min_analysis.py --duration-minutes 30
```

## 4P Notebook Training

```bash
python neural_network/scripts/run_notebook_4p_training.py --duration-minutes 30 --eval-episodes 20
```

This runs the model in 4-player matches against the extracted notebook zoo in
`opponents/`, with 4p forced on for both training and evaluation.

## Benchmark

```bash
python neural_network/scripts/benchmark_model.py --episodes 20
```

## Logs

- `reward_total`
- `avg_reward`
- `loss`
- `grad_norm`
- `actions_count`
- `real_actions_count`
- `do_nothing_rate`
- `invalid_action_rate`
- `winner`
- `episode_length`
- `winrate_vs_random`
- `winrate_vs_greedy`

## Success criteria

- `grad_norm > 0`
- `loss != 0`
- `real_actions_count > 0`
- `do_nothing_rate < 80%`
- reward is not always `-1.0`
- checkpoints are written
