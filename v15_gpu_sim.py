"""v15_gpu_sim — batched Orbit Wars engine (torch), B games stepped at once.

Mirrors v15_fast_sim.step for the COMET-FREE case. This is sufficient because
a self-play game starts at step 0 (comets spawn only at step >= 50) and
v15_fast_sim never spawns new comets — so every self-play trajectory is
comet-free and the planet count is fixed for the whole game.

State is a GpuBatch of fixed-shape tensors:
  planets [B,N,7]  id, owner, x, y, r, ships, prod
  p_init  [B,N,2]  initial x, y (rotation reference)
  pmask   [B,N]    valid planet slot (games may have different planet counts)
  fleets  [B,M,7]  id, owner, x, y, angle, from_id, ships
  fmask   [B,M]    valid fleet slot
  step / ang_vel / next_fid / done : per-game [B]

float64 throughout, so results match the numpy engine bit-for-bit. Validated
against v15_fast_sim by tests/test_gpu_sim_equivalence.py.

Runs on CPU or CUDA (set device); the win is batching — one step advances all
B games as a single set of tensor ops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

import v15_fast_sim as fsim

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
_LOG1000 = math.log(1000.0)

ID, OWNER, X, Y, R, SHIPS, PROD = range(7)
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)


@dataclass
class GpuBatch:
    planets: torch.Tensor   # [B,N,7] f64
    p_init: torch.Tensor    # [B,N,2] f64
    pmask: torch.Tensor     # [B,N]  bool
    fleets: torch.Tensor    # [B,M,7] f64
    fmask: torch.Tensor     # [B,M]  bool
    step: torch.Tensor      # [B] i64
    ang_vel: torch.Tensor   # [B] f64
    next_fid: torch.Tensor  # [B] i64
    done: torch.Tensor      # [B] bool
    n_players: int
    episode_steps: int
    ship_speed: float

    @property
    def B(self):
        return self.planets.shape[0]

    @property
    def device(self):
        return self.planets.device

    def clone(self) -> "GpuBatch":
        return GpuBatch(
            planets=self.planets.clone(), p_init=self.p_init.clone(),
            pmask=self.pmask.clone(), fleets=self.fleets.clone(),
            fmask=self.fmask.clone(), step=self.step.clone(),
            ang_vel=self.ang_vel.clone(), next_fid=self.next_fid.clone(),
            done=self.done.clone(), n_players=self.n_players,
            episode_steps=self.episode_steps, ship_speed=self.ship_speed)


def from_faststates(states: list[fsim.FastState], *, device="cpu",
                    m_max: int = 256, dtype=torch.float32,
                    n_fixed: int = 0) -> GpuBatch:
    """Pack a list of FastState (all same n_players, comet-free) into a batch.
    Planet/fleet slots are padded to the batch maxima; masks track validity.

    dtype  — float32 for fast self-play on consumer GPUs (their float64 path
             is ~32x slower); float64 for the bit-exact equivalence test.
    n_fixed — pad the planet dimension to (at least) this many slots, so every
             chunk has identical shape and torch.compile compiles only once."""
    B = len(states)
    n_players = states[0].n_players
    episode_steps = states[0].episode_steps
    ship_speed = states[0].ship_speed
    N = max(n_fixed, max(len(s.planets) for s in states))
    M = m_max

    planets = torch.zeros((B, N, 7), dtype=dtype)
    p_init = torch.zeros((B, N, 2), dtype=dtype)
    pmask = torch.zeros((B, N), dtype=torch.bool)
    fleets = torch.zeros((B, M, 7), dtype=dtype)
    fmask = torch.zeros((B, M), dtype=torch.bool)
    step = torch.zeros(B, dtype=torch.int64)
    ang_vel = torch.zeros(B, dtype=dtype)
    next_fid = torch.zeros(B, dtype=torch.int64)
    done = torch.zeros(B, dtype=torch.bool)

    for b, s in enumerate(states):
        np_ = len(s.planets)
        if np_:
            planets[b, :np_] = torch.from_numpy(np.asarray(s.planets)).to(dtype)
            p_init[b, :np_] = torch.from_numpy(np.asarray(s.p_init)).to(dtype)
            pmask[b, :np_] = True
        mf = len(s.fleets)
        if mf:
            if mf > M:
                raise ValueError(f"fleet count {mf} exceeds m_max {M}")
            fleets[b, :mf] = torch.from_numpy(np.asarray(s.fleets)).to(dtype)
            fmask[b, :mf] = True
        step[b] = s.step
        ang_vel[b] = s.angular_velocity
        next_fid[b] = s.next_fleet_id
        done[b] = s.done

    return GpuBatch(
        planets.to(device), p_init.to(device), pmask.to(device),
        fleets.to(device), fmask.to(device), step.to(device),
        ang_vel.to(device), next_fid.to(device), done.to(device),
        n_players, episode_steps, ship_speed)


def to_faststate(batch: GpuBatch, b: int) -> fsim.FastState:
    """Extract game `b` of a batch back into a FastState (for comparison)."""
    pm = batch.pmask[b].cpu().numpy()
    fm = batch.fmask[b].cpu().numpy()
    planets = batch.planets[b].cpu().numpy().astype(np.float64)[pm]
    p_init = batch.p_init[b].cpu().numpy().astype(np.float64)[pm]
    fleets = batch.fleets[b].cpu().numpy().astype(np.float64)[fm]
    return fsim.FastState(
        planets=planets.copy(), p_init=p_init.copy(),
        p_comet=np.zeros(len(planets), dtype=np.bool_),
        fleets=fleets.copy(), comets=[],
        step=int(batch.step[b].item()),
        angular_velocity=float(batch.ang_vel[b].item()),
        next_fleet_id=int(batch.next_fid[b].item()),
        episode_steps=batch.episode_steps, ship_speed=batch.ship_speed,
        n_players=batch.n_players, done=bool(batch.done[b].item()))


def _seg_point_dist(px, py, ax, ay, bx, by):
    """Vectorised point-to-segment distance (torch, broadcasts)."""
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    t = ((px - ax) * dx + (py - ay) * dy) / torch.where(
        l2 == 0.0, torch.ones_like(l2), l2)
    t = torch.where(l2 == 0.0, torch.zeros_like(t), t)
    t = torch.clamp(t, 0.0, 1.0)
    projx = ax + t * dx
    projy = ay + t * dy
    return torch.hypot(px - projx, py - projy)


def step(batch: GpuBatch, actions: torch.Tensor,
         has_launch: bool = True) -> GpuBatch:
    """Advance every game one turn.

    actions : [B, n_players, A, 3] float — (from_planet_id, angle, ships).
              Unused action slots must be all-zero (ships<=0 is ignored).
    has_launch : set False when actions are all-zero (passive rollout steps)
              to skip the launch loop entirely — a large saving.
    Returns a NEW GpuBatch (input untouched). Games already `done` are frozen.
    """
    s = batch.clone()
    B, N, _ = s.planets.shape
    M = s.fleets.shape[1]
    P = s.n_players
    dev = s.device
    dt = s.planets.dtype

    prev = batch                      # to freeze games that are already done
    active = ~s.done                  # [B]

    planets = s.planets
    pmask = s.pmask
    fleets = s.fleets
    fmask = s.fmask

    # ---- 0. fleet launch -------------------------------------------------
    if has_launch:
        A = actions.shape[2]
        pid = planets[:, :, ID]                              # [B,N]
        bidx = torch.arange(B, device=dev)
        for p in range(P):
            for a in range(A):
                mv = actions[:, p, a, :]                     # [B,3]
                from_id = mv[:, 0:1]                         # [B,1]
                angle = mv[:, 1]                             # [B]
                ships = torch.round(mv[:, 2])                # [B]
                match = (pid == from_id) & pmask             # [B,N]
                has = match.any(dim=1)                       # [B]
                idx = torch.argmax(match.to(torch.int64), dim=1)
                owner_ok = planets[bidx, idx, OWNER] == p
                garrison = planets[bidx, idx, SHIPS]
                valid = (has & owner_ok & active
                         & (ships > 0) & (garrison >= ships))
                planets[bidx, idx, SHIPS] = torch.where(
                    valid, garrison - ships, garrison)
                slot = torch.argmax((~fmask).to(torch.int64), dim=1)
                has_slot = (~fmask).any(dim=1)
                place = valid & has_slot
                r = planets[bidx, idx, R]
                sx = planets[bidx, idx, X] + torch.cos(angle) * (r + 0.1)
                sy = planets[bidx, idx, Y] + torch.sin(angle) * (r + 0.1)
                newf = torch.stack([
                    s.next_fid.to(dt),
                    torch.full((B,), p, dtype=dt, device=dev),
                    sx, sy, angle, mv[:, 0], ships], dim=1)
                cur = fleets[bidx, slot, :]
                fleets[bidx, slot, :] = torch.where(
                    place.unsqueeze(1), newf, cur)
                fmask[bidx, slot] = fmask[bidx, slot] | place
                s.next_fid = s.next_fid + place.to(torch.int64)

    # ---- 1. production ---------------------------------------------------
    owned = (planets[:, :, OWNER] != -1) & pmask
    planets[:, :, SHIPS] += torch.where(
        owned, planets[:, :, PROD], torch.zeros_like(planets[:, :, PROD]))

    # ---- 2. fleet movement + collision ----------------------------------
    caught = torch.full((B, M), -1, dtype=torch.int64, device=dev)
    removed = torch.zeros((B, M), dtype=torch.bool, device=dev)

    old_x = fleets[:, :, F_X].clone()
    old_y = fleets[:, :, F_Y].clone()
    sh = fleets[:, :, F_SHIPS]
    safe_sh = torch.clamp(sh, min=1.0)
    speeds = 1.0 + (s.ship_speed - 1.0) * (torch.log(safe_sh) / _LOG1000) ** 1.5
    speeds = torch.minimum(speeds, torch.tensor(s.ship_speed, dtype=dt,
                                                device=dev))
    new_x = old_x + torch.cos(fleets[:, :, F_ANGLE]) * speeds
    new_y = old_y + torch.sin(fleets[:, :, F_ANGLE]) * speeds
    fleets[:, :, F_X] = new_x
    fleets[:, :, F_Y] = new_y

    oob = ~((new_x >= 0) & (new_x <= BOARD_SIZE) &
            (new_y >= 0) & (new_y <= BOARD_SIZE))
    cen = torch.full((B, M), CENTER, dtype=dt, device=dev)
    sun = _seg_point_dist(cen, cen, old_x, old_y, new_x, new_y) < SUN_RADIUS
    removed = (oob | sun) & fmask

    # fleet [B,M] vs planet [B,N]: distance of each fleet segment to planet
    D = _seg_point_dist(
        planets[:, :, X].unsqueeze(1), planets[:, :, Y].unsqueeze(1),
        old_x.unsqueeze(2), old_y.unsqueeze(2),
        new_x.unsqueeze(2), new_y.unsqueeze(2))                # [B,M,N]
    hit = (D < planets[:, :, R].unsqueeze(1)) & pmask.unsqueeze(1)
    hit = hit & fmask.unsqueeze(2) & (~removed).unsqueeze(2)
    mhit = hit.any(dim=2)                                      # [B,M]
    first = torch.argmax(hit.to(torch.int64), dim=2)           # [B,M] slot idx
    caught = torch.where(mhit, first, caught)
    removed = removed | mhit

    # ---- 3. planet rotation + sweep -------------------------------------
    dx = s.p_init[:, :, 0] - CENTER
    dy = s.p_init[:, :, 1] - CENTER
    orb_r = torch.hypot(dx, dy)
    rotating = pmask & (orb_r + planets[:, :, R] < ROTATION_RADIUS_LIMIT)
    old_px = planets[:, :, X].clone()
    old_py = planets[:, :, Y].clone()
    cur_angle = torch.atan2(dy, dx) + s.ang_vel.unsqueeze(1) * s.step.unsqueeze(1)
    rot_x = CENTER + orb_r * torch.cos(cur_angle)
    rot_y = CENTER + orb_r * torch.sin(cur_angle)
    planets[:, :, X] = torch.where(rotating, rot_x, planets[:, :, X])
    planets[:, :, Y] = torch.where(rotating, rot_y, planets[:, :, Y])

    moved = (old_px != planets[:, :, X]) | (old_py != planets[:, :, Y])
    # planet [B,N] sweep vs fleet [B,M]
    DS = _seg_point_dist(
        fleets[:, :, F_X].unsqueeze(1), fleets[:, :, F_Y].unsqueeze(1),
        old_px.unsqueeze(2), old_py.unsqueeze(2),
        planets[:, :, X].unsqueeze(2), planets[:, :, Y].unsqueeze(2))  # [B,N,M]
    hitS = (DS < planets[:, :, R].unsqueeze(2)) & moved.unsqueeze(2)
    hitS = hitS & pmask.unsqueeze(2) & fmask.unsqueeze(1) & (~removed).unsqueeze(1)
    shit = hitS.any(dim=1)                                     # [B,M]
    firstS = torch.argmax(hitS.to(torch.int64), dim=1)         # [B,M] slot idx
    caught = torch.where(shit, firstS, caught)
    removed = removed | shit

    # ---- 4. combat resolution -------------------------------------------
    # accumulate caught-fleet ships per (game, planet slot, owner) with a
    # single scatter_add (no Python loop over N or P) — `caught` holds the
    # planet slot index a fleet hit, or -1.
    valid_c = caught >= 0                                      # [B,M]
    owner_f = fleets[:, :, F_OWNER].to(torch.int64).clamp(min=0)
    slot_c = caught.clamp(min=0)
    flat_idx = slot_c * P + owner_f                            # [B,M]
    flat_idx = torch.where(valid_c, flat_idx, torch.zeros_like(flat_idx))
    contrib = torch.where(valid_c, fleets[:, :, F_SHIPS],
                          torch.zeros_like(fleets[:, :, F_SHIPS]))
    acc = torch.zeros((B, N * P), dtype=dt, device=dev)
    acc.scatter_add_(1, flat_idx, contrib)
    acc = acc.view(B, N, P)

    top2, _ = torch.topk(acc, k=min(2, P), dim=2)               # [B,N,<=2]
    top1 = top2[:, :, 0]
    second = top2[:, :, 1] if top2.shape[2] > 1 else torch.zeros_like(top1)
    top_owner = torch.argmax(acc, dim=2)                        # [B,N]
    any_fleet = acc.sum(dim=2) > 0
    survivor_ships = top1 - second
    tie = top1 == second
    survivor_ships = torch.where(tie, torch.zeros_like(survivor_ships),
                                 survivor_ships)
    has_survivor = any_fleet & (survivor_ships > 0)

    cur_owner = planets[:, :, OWNER]
    reinforce = has_survivor & (cur_owner == top_owner)
    attack = has_survivor & (cur_owner != top_owner)
    new_ships = planets[:, :, SHIPS].clone()
    new_ships = torch.where(reinforce, new_ships + survivor_ships, new_ships)
    after = planets[:, :, SHIPS] - survivor_ships
    flipped = attack & (after < 0)
    new_ships = torch.where(attack & ~flipped, after, new_ships)
    new_ships = torch.where(flipped, torch.abs(after), new_ships)
    new_owner = torch.where(flipped, top_owner.to(dt), cur_owner)
    planets[:, :, SHIPS] = torch.where(pmask, new_ships, planets[:, :, SHIPS])
    planets[:, :, OWNER] = torch.where(pmask, new_owner, planets[:, :, OWNER])

    # drop removed fleets
    fmask = fmask & (~removed)

    # ---- terminal --------------------------------------------------------
    alive = torch.zeros((B, P), dtype=torch.bool, device=dev)
    for p in range(P):
        has_planet = ((planets[:, :, OWNER] == p) & pmask).any(dim=1)
        has_fleet = ((fleets[:, :, F_OWNER] == p) & fmask).any(dim=1)
        alive[:, p] = has_planet | has_fleet
    n_alive = alive.to(torch.int64).sum(dim=1)
    terminated = (s.step >= s.episode_steps - 2) | (n_alive <= 1)

    s.planets = planets
    s.pmask = pmask
    s.fleets = fleets
    s.fmask = fmask
    new_done = s.done | terminated
    s.step = s.step + 1

    # freeze games that were already done before this step
    froz = batch.done
    if froz.any():
        s.planets = torch.where(froz[:, None, None], prev.planets, s.planets)
        s.pmask = torch.where(froz[:, None], prev.pmask, s.pmask)
        s.fleets = torch.where(froz[:, None, None], prev.fleets, s.fleets)
        s.fmask = torch.where(froz[:, None], prev.fmask, s.fmask)
        s.step = torch.where(froz, prev.step, s.step)
        s.next_fid = torch.where(froz, prev.next_fid, s.next_fid)
        new_done = torch.where(froz, prev.done, new_done)
    s.done = new_done
    return s


def step_passive(batch: GpuBatch, actions: torch.Tensor) -> GpuBatch:
    """A step with no fleet launch (rollout continuation). Compiled separately
    from `step` so torch.compile sees only this hot path — avoids recompiling
    for the `has_launch` guard variant."""
    return step(batch, actions, has_launch=False)


def scores(batch: GpuBatch) -> torch.Tensor:
    """[B, n_players] integer ship totals (planets + fleets)."""
    B = batch.B
    P = batch.n_players
    out = torch.zeros((B, P), dtype=torch.int64, device=batch.device)
    for p in range(P):
        ps = ((batch.planets[:, :, OWNER] == p) & batch.pmask) \
            * batch.planets[:, :, SHIPS]
        fsh = ((batch.fleets[:, :, F_OWNER] == p) & batch.fmask) \
            * batch.fleets[:, :, F_SHIPS]
        out[:, p] = (ps.sum(dim=1) + fsh.sum(dim=1)).to(torch.int64)
    return out
