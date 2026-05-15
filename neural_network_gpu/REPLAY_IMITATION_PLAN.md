# Replay Imitation Plan

## Goal

Use high-level Kaggle Orbit Wars replays as a supervised prior without reintroducing the previous failure mode where the policy wins noisy evaluations by doing nothing or sending almost no ships.

## Dataset Construction

The raw Kaggle archive stays outside Git at `D:\warorbit_kaggle_raw`. The repo only stores compact numerical shards under `replay_corpus/imitation_4p_top10_v1`.

Filtering rules:

- Keep 4-player replays only.
- Reject games longer than 250 turns.
- Learn from winner actions by default.
- Drop no-op actions by default.
- For a replay action `(source, angle, ships)`, find the legal model candidate whose source and firing angle best match the replay.
- Reject ambiguous labels when angular error is above `0.28` radians.
- Store padded candidate tensors plus masks, labels, sample weights, and metadata.

This makes the supervised objective:

`min_theta E[-w_t log pi_theta(a*_t | s_t, C_t)]`

where `C_t` is the legal candidate set from the current policy code and `a*_t` is the replay-matched expert action.

## Training

The imitation script fine-tunes the current candidate policy with masked cross-entropy:

- logits for invalid padded candidates are masked to `-inf`;
- labels are candidate indices, not raw replay moves;
- checkpoints stay compatible with the existing model format;
- validation reports top-1/top-3 candidate imitation accuracy.

## RL Handoff

After behavior cloning, use the resulting `bc_4p_top10_best.npz` as the initialization checkpoint for the normal self-play/evaluation loop. The replay phase is only a prior; promotion still has to pass the anti-noop, avg-ship, valid-winrate, and regression gates.

## Commands

Extract compact shards:

```powershell
.\run_extract_4p_imitation.ps1
```

Train imitation checkpoint:

```powershell
.\run_train_4p_imitation.ps1
```
