"""v15_gpu_search — GPU-native batched RCC search (no V7 in the hot loop).

V7's per-move call is 1870 lines of pure-Python heuristics: not batchable, so
it bottlenecks any batched self-play. This module reimplements RCC as pure
tensor ops on v15_gpu_sim, so the WHOLE search runs batched on GPU:

  * enumerate one intercept-aimed atomic shot per (top-8 owned) planet,
  * stage-1: score each atomic shot by a passive-continuation rollout,
  * stage-2: score every subset (<=4 shots) of the top-6 survivors,
  * pick the best combo per game, or the do-nothing baseline.

It searches one player across all B games at once. The engine dtype (float32
for fast self-play on consumer GPUs) is followed throughout.
"""

from __future__ import annotations

from itertools import combinations

import torch

import v15_gpu_sim as gsim

ID, OWNER, X, Y, R, SHIPS, PROD = range(7)
F_OWNER, F_SHIPS = 1, 6
F_X, F_Y, F_ANGLE = 2, 3, 4
BOARD_SIZE = 100.0
CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
_EPS = 1e-9
_DIAG = float((BOARD_SIZE ** 2 + BOARD_SIZE ** 2) ** 0.5)
_TAU = 10.0

K_SRC = 8        # atomic shots = top-K_SRC owned planets, one shot each
T_TOP = 6        # atomic survivors entering the subset search
MAX_COMBO = 4    # max simultaneous launches in a combination
A_SLOTS = 4      # action slots per player


def _subset_patterns(t: int, max_combo: int):
    """All non-empty subsets (size<=max_combo) of range(t), padded with -1."""
    pats = []
    for r in range(1, min(max_combo, t) + 1):
        for c in combinations(range(t), r):
            pats.append(list(c) + [-1] * (A_SLOTS - r))
    return torch.tensor(pats, dtype=torch.int64)


_PATTERNS = _subset_patterns(T_TOP, MAX_COMBO)   # [S, A_SLOTS]


# ---------------------------------------------------------------------------
# Batched evaluator (mirror of v15_eval, 11 features, on GpuBatch tensors)
# ---------------------------------------------------------------------------

def _player_totals(batch: gsim.GpuBatch):
    """garrison, fleet, prod, planets — each [B, P]."""
    B = batch.B
    P = batch.n_players
    dev = batch.device
    dt = batch.planets.dtype
    owner = batch.planets[:, :, OWNER]
    ships = batch.planets[:, :, SHIPS]
    prod = batch.planets[:, :, PROD]
    fowner = batch.fleets[:, :, F_OWNER]
    fships = batch.fleets[:, :, F_SHIPS]
    garrison = torch.zeros((B, P), dtype=dt, device=dev)
    fleet = torch.zeros((B, P), dtype=dt, device=dev)
    prodt = torch.zeros((B, P), dtype=dt, device=dev)
    planets = torch.zeros((B, P), dtype=dt, device=dev)
    for p in range(P):
        pm = (owner == p) & batch.pmask
        garrison[:, p] = (ships * pm).sum(dim=1)
        prodt[:, p] = (prod * pm).sum(dim=1)
        planets[:, p] = pm.sum(dim=1).to(dt)
        fm = (fowner == p) & batch.fmask
        fleet[:, p] = (fships * fm).sum(dim=1)
    return garrison, fleet, prodt, planets


def _extra_batch_features(batch: gsim.GpuBatch, player: int,
                          ships: torch.Tensor) -> torch.Tensor:
    """[B, 13] V18 signals — batched mirror of v15_eval._extra_features.

    `ships` is [B, P] garrison+fleet power per player."""
    dev, dt = batch.device, batch.planets.dtype
    P = batch.n_players
    pl = batch.planets
    o, x, y = pl[:, :, OWNER], pl[:, :, X], pl[:, :, Y]
    r, sh, pr = pl[:, :, R], pl[:, :, SHIPS], pl[:, :, PROD]
    pmask = batch.pmask
    mine = (o == player) & pmask
    enemy = (o >= 0) & (o != player) & pmask
    neutral = (o < 0) & pmask
    INF = torch.full((), 1e9, dtype=dt, device=dev)
    zero = torch.zeros((), dtype=dt, device=dev)
    nm = mine.sum(dim=1).clamp(min=1).to(dt)
    has_my = mine.any(dim=1)
    has_en = enemy.any(dim=1)

    D = torch.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
    dmin_enemy = torch.where(enemy[:, None, :], D, INF).min(dim=2).values        # [B,N]
    frontier_mean = torch.where(mine, dmin_enemy, zero).sum(dim=1) / nm
    frontier = 1.0 - (frontier_mean / _DIAG).clamp(max=1.0)
    frontier = torch.where(has_my & has_en, frontier, torch.full_like(frontier, 0.5))

    cx = (x * mine).sum(dim=1) / nm
    cy = (y * mine).sum(dim=1) / nm
    disp = torch.where(mine, torch.hypot(x - cx[:, None], y - cy[:, None]), zero).sum(dim=1) / nm
    dispersion = 1.0 - (disp / (_DIAG * 0.5)).clamp(max=1.0)
    dispersion = torch.where(has_my, dispersion, torch.full_like(dispersion, 0.5))

    d_to_my = torch.where(mine[:, None, :], D, INF).min(dim=2).values            # [B,N]
    contested_k = neutral & (d_to_my < dmin_enemy)
    nn = neutral.sum(dim=1)
    contested = torch.where(nn > 0, contested_k.sum(dim=1).to(dt) / nn.clamp(min=1).to(dt), zero)

    fl, fmask = batch.fleets, batch.fmask
    fo = fl[:, :, F_OWNER]
    fx, fy = fl[:, :, F_X], fl[:, :, F_Y]
    fang, fsh = fl[:, :, F_ANGLE], fl[:, :, F_SHIPS]
    efl = (fo >= 0) & (fo != player) & fmask                                     # [B,M]
    has_thr = has_my & efl.any(dim=1)   # CPU computes threat/tempo only here
    dcos, dsin = torch.cos(fang), torch.sin(fang)
    wx = x[:, None, :] - fx[:, :, None]                                          # [B,M,N]
    wy = y[:, None, :] - fy[:, :, None]
    t = wx * dcos[:, :, None] + wy * dsin[:, :, None]
    perp = (-wx * dsin[:, :, None] + wy * dcos[:, :, None]).abs()
    hit = (t > 0) & (perp <= (r[:, None, :] + 2.0)) & efl[:, :, None] & mine[:, None, :]
    threat = (fsh[:, :, None] * hit).sum(dim=1)                                  # [B,N]
    garr_my = torch.where(mine, sh, zero)
    incoming = (threat.sum(dim=1) / (garr_my.sum(dim=1) + _EPS)).clamp(max=1.0)
    incoming = torch.where(has_thr, incoming, zero)
    at_risk = torch.where(has_thr, ((threat > sh) & mine).sum(dim=1).to(dt) / nm, zero)
    diff = torch.where(mine, (threat - sh) / (threat + sh + _EPS),
                       torch.full_like(sh, -1.0))
    worst = 0.5 * (diff.max(dim=1).values + 1.0)
    worst = torch.where(has_thr, worst, torch.full_like(worst, 0.5))

    speed = max(float(batch.ship_speed), 1.0)
    eta = torch.where(mine[:, None, :], torch.hypot(wx, wy), INF).min(dim=2).values / speed
    tot_s = ships.sum(dim=1).clamp(min=_EPS)
    eta_enemy = ((fsh * torch.exp(-eta / _TAU) * efl).sum(dim=1) / tot_s).clamp(max=1.0)
    eta_enemy = torch.where(has_thr, eta_enemy, zero)

    targets = (~mine) & pmask
    near_idx = torch.where(mine[:, None, :], D, INF).min(dim=2).indices          # [B,N]
    force = torch.gather(sh, 1, near_idx) * 0.5
    cap = (force > sh) & targets
    tot_p = pr.sum(dim=1).clamp(min=_EPS)
    free_cap = torch.where(has_my, (torch.where(cap, pr, zero).sum(dim=1) / tot_p).clamp(max=1.0), zero)
    expansion = torch.where(
        has_my, (cap & neutral).sum(dim=1).to(dt) / pmask.sum(dim=1).clamp(min=1).to(dt), zero)

    me_pow = ships[:, player]
    others = [q for q in range(P) if q != player]
    max_opp = torch.stack([ships[:, q] for q in others], dim=1).max(dim=1).values
    tot = ships.sum(dim=1)
    am_leader = (me_pow >= max_opp).to(dt)
    gap = ((max_opp - me_pow).clamp(min=0.0) / (tot + _EPS)).clamp(max=1.0)
    lead = ((me_pow - max_opp).clamp(min=0.0) / (tot + _EPS)).clamp(max=1.0) * am_leader
    gangup = ((tot - me_pow) / (me_pow + _EPS)).clamp(max=1.0)

    return torch.stack([frontier, dispersion, contested, incoming, at_risk, worst,
                        eta_enemy, free_cap, expansion, am_leader, gap, lead,
                        gangup], dim=1).clamp(0.0, 1.0)


def batch_features(batch: gsim.GpuBatch, player: int) -> torch.Tensor:
    """[B, 24] — the v15_eval feature vector for `player`, batched."""
    garrison, fleet, prod, planets = _player_totals(batch)
    P = batch.n_players
    dt = batch.planets.dtype
    ships = garrison + fleet
    tot_s = ships.sum(dim=1, keepdim=True).clamp(min=_EPS)
    tot_p = prod.sum(dim=1, keepdim=True).clamp(min=_EPS)
    tot_pl = planets.sum(dim=1, keepdim=True).clamp(min=_EPS)

    ship_share = ships[:, player] / tot_s[:, 0]
    prod_share = prod[:, player] / tot_p[:, 0]
    planet_share = planets[:, player] / tot_pl[:, 0]

    others = [q for q in range(P) if q != player]
    opp_s = torch.stack([ships[:, q] for q in others], dim=1)
    opp_p = torch.stack([prod[:, q] for q in others], dim=1)
    opp_f = torch.stack([fleet[:, q] for q in others], dim=1)
    max_opp_s = opp_s.max(dim=1).values
    max_opp_p = opp_p.max(dim=1).values

    denom_s = (ships[:, player] + max_opp_s).clamp(min=_EPS)
    domination = 0.5 * ((ships[:, player] - max_opp_s) / denom_s + 1.0)
    prod_margin = 0.5 * ((prod[:, player] - max_opp_p) / tot_p[:, 0] + 1.0)

    my_ships = ships[:, player].clamp(min=_EPS)
    fleet_share = fleet[:, player] / my_ships
    elim = sum((((garrison[:, q] + fleet[:, q]) <= _EPS)
                & (planets[:, q] <= _EPS)).to(dt) for q in others)
    elim_share = elim / max(1, P - 1)

    owner = batch.planets[:, :, OWNER]
    mine = (owner == player) & batch.pmask
    zero = torch.zeros_like(batch.planets[:, :, SHIPS])
    big_garr = torch.where(mine, batch.planets[:, :, SHIPS], zero).max(dim=1).values
    top_prod = torch.where(mine, batch.planets[:, :, PROD], zero).max(dim=1).values
    top_planet_prod = top_prod / tot_p[:, 0]
    ship_conc = big_garr / my_ships
    enemy_fleet_press = opp_f.sum(dim=1) / tot_s[:, 0]
    step_frac = (batch.step.to(dt) / 500.0).clamp(max=1.0)

    base = torch.stack([
        ship_share, prod_share, planet_share, domination, prod_margin,
        fleet_share, elim_share, top_planet_prod, ship_conc,
        step_frac, enemy_fleet_press], dim=1).clamp(0.0, 1.0)
    extra = _extra_batch_features(batch, player, ships)
    return torch.cat([base, extra], dim=1)


def batch_eval(batch: gsim.GpuBatch, player: int, w, m, s) -> torch.Tensor:
    """[B] position score for `player`. w/m/s are [11] tensors."""
    feats = batch_features(batch, player)
    return (((feats - m) / s) * w).sum(dim=1)


# ---------------------------------------------------------------------------
# Enumeration + rollout
# ---------------------------------------------------------------------------

def _intercept_angle(batch, src_idx, tgt_idx, ship_speed):
    """Intercept-aimed launch angle from src planet to tgt planet, [B]."""
    B = batch.B
    dt = batch.planets.dtype
    bidx = torch.arange(B, device=batch.device)
    sx = batch.planets[bidx, src_idx, X]
    sy = batch.planets[bidx, src_idx, Y]
    tix = batch.p_init[bidx, tgt_idx, 0] - CENTER
    tiy = batch.p_init[bidx, tgt_idx, 1] - CENTER
    orb_r = torch.hypot(tix, tiy)
    base_ang = torch.atan2(tiy, tix)
    tr = batch.planets[bidx, tgt_idx, R]
    rotating = orb_r + tr < ROTATION_RADIUS_LIMIT
    tx = batch.planets[bidx, tgt_idx, X]
    ty = batch.planets[bidx, tgt_idx, Y]
    ang = torch.atan2(ty - sy, tx - sx)
    step_f = batch.step.to(dt)
    for _ in range(3):
        dist = torch.hypot(tx - sx, ty - sy)
        eta = (dist / max(ship_speed, 1.0)).clamp(min=1.0)
        fut = base_ang + batch.ang_vel * (step_f + eta)
        ptx = CENTER + orb_r * torch.cos(fut)
        pty = CENTER + orb_r * torch.sin(fut)
        tx = torch.where(rotating, ptx, tx)
        ty = torch.where(rotating, pty, ty)
        ang = torch.atan2(ty - sy, tx - sx)
    return ang


def enumerate_shots(batch: gsim.GpuBatch, player: int):
    """One intercept-aimed atomic shot per top-K_SRC owned planet.
    Returns shots [B, K, 3] (src_id, angle, ships) and valid mask [B, K]."""
    B, N, _ = batch.planets.shape
    dev = batch.device
    dt = batch.planets.dtype
    owner = batch.planets[:, :, OWNER]
    ships = batch.planets[:, :, SHIPS]
    pmask = batch.pmask
    mine = (owner == player) & pmask
    foreign = (owner != player) & pmask

    garr = torch.where(mine, ships, torch.full_like(ships, -1.0))
    k = min(K_SRC, N)
    src_val, src_idx = torch.topk(garr, k=k, dim=1)         # [B,k]
    src_valid = src_val > 0

    px = batch.planets[:, :, X]
    py = batch.planets[:, :, Y]
    shots = torch.zeros((B, k, 3), dtype=dt, device=dev)
    valid = torch.zeros((B, k), dtype=torch.bool, device=dev)
    bidx = torch.arange(B, device=dev)
    for j in range(k):
        si = src_idx[:, j]
        sx = px[bidx, si]
        sy = py[bidx, si]
        d = torch.hypot(px - sx.unsqueeze(1), py - sy.unsqueeze(1))
        d = torch.where(foreign, d, torch.full_like(d, 1e9))
        ti = torch.argmin(d, dim=1)
        has_tgt = foreign.any(dim=1)
        ang = _intercept_angle(batch, si, ti, batch.ship_speed)
        t_ships = batch.planets[bidx, ti, SHIPS]
        t_prod = batch.planets[bidx, ti, PROD]
        dist = torch.hypot(px[bidx, ti] - sx, py[bidx, ti] - sy)
        eta = (dist / 4.0).clamp(min=1.0)
        enemy = batch.planets[bidx, ti, OWNER] >= 0
        margin = torch.where(enemy,
                             torch.tensor(5.0, dtype=dt, device=dev),
                             torch.tensor(3.0, dtype=dt, device=dev))
        need = torch.round(t_ships + t_prod * eta + margin) + 1
        garrison = batch.planets[bidx, si, SHIPS]
        send = torch.minimum(garrison, need)
        ok = src_valid[:, j] & has_tgt & (send >= need * 0.5) & (send > 0)
        shots[:, j, 0] = batch.planets[bidx, si, ID]
        shots[:, j, 1] = ang
        shots[:, j, 2] = torch.where(ok, send, torch.zeros_like(send))
        valid[:, j] = ok
    return shots, valid


def _expand(batch: gsim.GpuBatch, factor: int) -> gsim.GpuBatch:
    """Repeat each game `factor` times -> batch of size B*factor."""
    ri = torch.repeat_interleave
    return gsim.GpuBatch(
        planets=ri(batch.planets, factor, dim=0),
        p_init=ri(batch.p_init, factor, dim=0),
        pmask=ri(batch.pmask, factor, dim=0),
        fleets=ri(batch.fleets, factor, dim=0),
        fmask=ri(batch.fmask, factor, dim=0),
        step=ri(batch.step, factor, dim=0),
        ang_vel=ri(batch.ang_vel, factor, dim=0),
        next_fid=ri(batch.next_fid, factor, dim=0),
        done=ri(batch.done, factor, dim=0),
        n_players=batch.n_players, episode_steps=batch.episode_steps,
        ship_speed=batch.ship_speed)


def _rollout_eval(states: gsim.GpuBatch, first_actions: torch.Tensor,
                  horizon: int, player: int, w, m, s) -> torch.Tensor:
    """Apply first_actions, then `horizon-1` PASSIVE steps; return [G] score.
    Passive steps pass has_launch=False so the launch loop is skipped."""
    G = states.B
    P = states.n_players
    # first step applies the combo (eager, with launch); the passive
    # continuation uses the compiled step_passive (the hot path).
    st = gsim.step(states, first_actions, has_launch=True)
    empty = torch.zeros((G, P, A_SLOTS, 3), dtype=states.planets.dtype,
                        device=states.device)
    for _ in range(horizon - 1):
        st = gsim.step_passive(st, empty)
    return batch_eval(st, player, w, m, s)


def _combo_actions(combos: torch.Tensor, player: int,
                   n_players: int) -> torch.Tensor:
    """combos [G, A_SLOTS, 3] -> first-actions tensor [G, P, A_SLOTS, 3]."""
    G = combos.shape[0]
    out = torch.zeros((G, n_players, A_SLOTS, 3), dtype=combos.dtype,
                      device=combos.device)
    out[:, player, :, :] = combos
    return out


def gpu_search(batch: gsim.GpuBatch, player: int, weights, *,
               horizon: int = 24, explore: float = 0.0) -> torch.Tensor:
    """Best move for `player` across all B games. Returns [B, A_SLOTS, 3].

    explore — with this per-game probability, a random combo is played
    instead of the best one. Needed in self-play data generation: identical
    deterministic policies on a symmetric map make mirror moves and the game
    ends in a tie (no winner, no training label). Exploration breaks the
    symmetry and diversifies the value-function training distribution. Keep
    it 0 for benchmarking (measure true strength)."""
    B = batch.B
    P = batch.n_players
    dev = batch.device
    dt = batch.planets.dtype
    if P >= 4:
        w = torch.as_tensor(weights.w4p, dtype=dt, device=dev)
        m = torch.as_tensor(weights.mean4p, dtype=dt, device=dev)
        s = torch.as_tensor(weights.std4p, dtype=dt, device=dev)
    else:
        w = torch.as_tensor(weights.w2p, dtype=dt, device=dev)
        m = torch.as_tensor(weights.mean2p, dtype=dt, device=dev)
        s = torch.as_tensor(weights.std2p, dtype=dt, device=dev)

    shots, valid = enumerate_shots(batch, player)            # [B,K,3],[B,K]
    K = shots.shape[1]

    # --- baseline: the do-nothing move ---
    empty1 = torch.zeros((B, P, A_SLOTS, 3), dtype=dt, device=dev)
    baseline = _rollout_eval(batch, empty1, horizon, player, w, m, s)

    # --- stage 1: score each atomic shot alone ---
    big = _expand(batch, K)
    single = torch.zeros((B, K, A_SLOTS, 3), dtype=dt, device=dev)
    single[:, :, 0, :] = shots
    single = single.reshape(B * K, A_SLOTS, 3)
    fa = _combo_actions(single, player, P)
    s1 = _rollout_eval(big, fa, horizon, player, w, m, s).reshape(B, K)
    s1 = torch.where(valid, s1, torch.full_like(s1, -1e9))

    # --- top-T atomic survivors ---
    t = min(T_TOP, K)
    top_val, top_idx = torch.topk(s1, k=t, dim=1)            # [B,t]
    top_shots = torch.gather(
        shots, 1, top_idx.unsqueeze(-1).expand(-1, -1, 3))   # [B,t,3]
    top_valid = top_val > -1e8

    # --- stage 2: score every subset of the survivors ---
    pats = _PATTERNS.to(dev)                                 # [S,A]
    S = pats.shape[0]
    gidx = pats.clamp(min=0)                                 # [S,A]
    combos = top_shots[:, gidx, :]                           # [B,S,A,3]
    slot_on = (pats >= 0)
    surv_ok = torch.gather(
        top_valid, 1, gidx.reshape(-1).unsqueeze(0).expand(B, -1)
    ).reshape(B, S, A_SLOTS)
    active = slot_on.unsqueeze(0) & surv_ok                  # [B,S,A]
    combos = combos * active.unsqueeze(-1)
    combo_valid = active.any(dim=2)

    big2 = _expand(batch, S)
    fa2 = _combo_actions(combos.reshape(B * S, A_SLOTS, 3), player, P)
    s2 = _rollout_eval(big2, fa2, horizon, player, w, m, s).reshape(B, S)
    s2 = torch.where(combo_valid, s2, torch.full_like(s2, -1e9))

    # --- pick: best combo if it beats the do-nothing baseline ---
    best_val, best_pat = torch.max(s2, dim=1)
    bidx = torch.arange(B, device=dev)
    explored = torch.zeros(B, dtype=torch.bool, device=dev)
    if explore > 0.0:
        explored = torch.rand(B, device=dev) < explore
        rand_pat = torch.randint(0, S, (B,), device=dev)
        best_pat = torch.where(explored, rand_pat, best_pat)
    chosen = combos[bidx, best_pat]                          # [B,A,3]
    # explored games always play their (random) combo; others play it only
    # if it beats doing nothing
    take = (explored | (best_val > baseline)).view(B, 1, 1)
    return torch.where(take, chosen, torch.zeros_like(chosen))
