"""v16_eval — non-linear evaluator with Standing-Conditioned Risk (SCR).

Two design ideas, in one module:

P2 — non-linear evaluator. A small MLP on the 11 position features lifts the
representational ceiling of the linear ESC. To make regression impossible the
MLP is a *residual correction*: value = ESC_linear(feat) + MLP(feat), with the
MLP weights initialised to ~0, so theta_0 reproduces the ESC exactly.

P3 — Standing-Conditioned Risk. Orbit Wars is winner-take-all (+1 to 1st only).
A safe 2nd place is a guaranteed loss, so when behind one must seek variance;
when ahead one must lock the win down. The MLP therefore has a SECOND head, a
positive `spread` (outcome uncertainty of a leaf). The search scores a leaf:

    score = value + z_gain * (0.5 - root_value) * spread

`root_value` is our win-prob estimate at the current position. Behind
(root<0.5) -> positive multiplier -> optimistic, seeks the high-variance combo;
ahead -> negative -> pessimistic, picks the safe combo. One rollout per combo —
no extra cost. ES tunes the whole vector (MLP weights + z_gain); if SCR does
not help, ES drives z_gain -> 0 (graceful, self-validating).

theta is a flat float64 vector (the ES parameter); the torch side unpacks it
into batched tensors for the GPU search.
"""

from __future__ import annotations

import numpy as np
import torch

N_FEATURES = 11


def n_params(hidden: int) -> int:
    """Length of the flat theta vector for a given hidden width."""
    # W1[F,H] b1[H]  Wv[H] bv[1]  Ws[H] bs[1]  z_gain[1]
    return N_FEATURES * hidden + hidden + hidden + 1 + hidden + 1 + 1


def initial_theta(hidden: int) -> np.ndarray:
    """theta_0 — MLP correction ~0, z_gain 0 => the evaluator == ESC exactly.

    A tiny W1 is used (not exactly 0) only so ES perturbations have a non-zero
    gradient to work with; with Wv=Ws=0 the hidden layer still has no effect on
    the output at theta_0."""
    rng = np.random.default_rng(0)
    H = hidden
    W1 = rng.standard_normal(N_FEATURES * H) * 0.05
    b1 = np.zeros(H)
    Wv = np.zeros(H)            # value-correction head: 0 -> value == ESC
    bv = np.zeros(1)
    Ws = np.zeros(H)            # spread head
    bs = np.full(1, -2.0)       # softplus(-2) ~ 0.13 : small baseline spread
    zg = np.zeros(1)            # z_gain 0 -> SCR term off at theta_0
    return np.concatenate([W1, b1, Wv, bv, Ws, bs, zg]).astype(np.float64)


def unpack(theta: np.ndarray, hidden: int, device, dtype):
    """Flat theta -> dict of batched torch tensors for the GPU search."""
    H = hidden
    t = torch.as_tensor(theta, dtype=dtype, device=device)
    i = 0

    def take(n, shape):
        nonlocal i
        chunk = t[i:i + n].reshape(shape)
        i += n
        return chunk

    return {
        "W1": take(N_FEATURES * H, (N_FEATURES, H)),
        "b1": take(H, (H,)),
        "Wv": take(H, (H, 1)),
        "bv": take(1, (1,)),
        "Ws": take(H, (H, 1)),
        "bs": take(1, (1,)),
        "z_gain": take(1, (1,)),
    }


def _softplus(x):
    return torch.nn.functional.softplus(x)


def value_and_spread(feats: torch.Tensor, esc_w: torch.Tensor, p):
    """feats [.,11], esc_w [11] -> (value [.], spread [.]).

    value  = ESC linear term + MLP residual correction
    spread = softplus(MLP spread head)  (strictly positive)
    """
    esc_lin = feats @ esc_w                           # [.]
    hidden = torch.tanh(feats @ p["W1"] + p["b1"])    # [.,H]
    value = esc_lin + (hidden @ p["Wv"]).squeeze(-1) + p["bv"]
    spread = _softplus((hidden @ p["Ws"]).squeeze(-1) + p["bs"])
    return value, spread


def scr_score(leaf_feats: torch.Tensor, root_value: torch.Tensor,
              esc_w: torch.Tensor, p) -> torch.Tensor:
    """Standing-Conditioned Risk score of leaf positions.

    leaf_feats  [.,11] — features of the rollout leaf for each combo,
    root_value  [.]    — our win-prob estimate at the CURRENT position,
                         broadcast per combo.
    Returns the score the search maximises."""
    value, spread = value_and_spread(leaf_feats, esc_w, p)
    z = p["z_gain"] * (0.5 - root_value)
    return value + z * spread


if __name__ == "__main__":
    # --- unit checks -------------------------------------------------------
    H = 8
    dev, dt = "cpu", torch.float64
    th = initial_theta(H)
    assert len(th) == n_params(H), (len(th), n_params(H))
    p = unpack(th, H, dev, dt)
    esc_w = torch.tensor(
        [0.40, 0.30, 0.05, 0.15, 0.10, 0, 0, 0, 0, 0, 0], dtype=dt)

    feats = torch.rand((20, N_FEATURES), dtype=dt)
    value, spread = value_and_spread(feats, esc_w, p)
    esc_lin = feats @ esc_w

    # theta_0: value must equal the ESC linear value exactly
    assert torch.allclose(value, esc_lin, atol=1e-9), \
        f"theta_0 value != ESC: max diff {(value-esc_lin).abs().max()}"
    # spread strictly positive
    assert (spread > 0).all(), "spread must be positive"
    # theta_0: z_gain 0 -> SCR score == ESC value regardless of standing
    for root in (0.1, 0.5, 0.9):
        rv = torch.full((20,), root, dtype=dt)
        sc = scr_score(feats, rv, esc_w, p)
        assert torch.allclose(sc, esc_lin, atol=1e-9), \
            f"theta_0 SCR score != ESC at root={root}"

    # a non-zero z_gain must tilt the score by standing
    th2 = th.copy()
    th2[-1] = 1.0                                    # z_gain = 1
    p2 = unpack(th2, H, dev, dt)
    behind = scr_score(feats, torch.full((20,), 0.1, dtype=dt), esc_w, p2)
    ahead = scr_score(feats, torch.full((20,), 0.9, dtype=dt), esc_w, p2)
    assert (behind > ahead).all(), \
        "behind should score a leaf higher than ahead (seeks variance)"
    print(f"v16_eval: n_params(H={H})={n_params(H)}  all unit checks passed")
