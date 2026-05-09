"""
cgame.py — Python wrapper around the C engine.

Mirrors the OfficialFastGame interface so it's a drop-in replacement.
Comet path generation stays in Python (via the official orbit_wars module)
to preserve RNG determinism. Per-step interpreter logic runs in C.
"""

from __future__ import annotations

import ctypes as ct
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

# Use the official Python module ONLY for initial state generation and
# comet path generation. Step execution goes through C.
import sys
_LOCAL_SIM = Path(__file__).resolve().parent.parent / "local_simulator"
if str(_LOCAL_SIM) not in sys.path:
    sys.path.insert(0, str(_LOCAL_SIM))

from local_simulator.official_fast import OfficialFastGame, FastConfig  # noqa: E402
import local_simulator.orbit_wars_official as orbit_wars_py  # noqa: E402


_LIB_PATH = Path(__file__).resolve().parent / "liborbit_wars.so"
if not _LIB_PATH.exists():
    raise RuntimeError(
        f"C library not found at {_LIB_PATH}. Run `make` in c_engine/."
    )

_lib = ct.CDLL(str(_LIB_PATH))

# Match constants from the C header (orbit_wars_consts.h)
_MAX_COMET_PATH_LEN = 64
_MAX_PLANETS = 80
_MAX_FLEETS = 4096
COMET_SPAWN_STEPS = orbit_wars_py.COMET_SPAWN_STEPS

# ── ctypes signatures ─────────────────────────────────────────────────────

_lib.gs_create.argtypes = [ct.c_int, ct.c_double, ct.c_double, ct.c_int, ct.c_double]
_lib.gs_create.restype = ct.c_void_p

_lib.gs_destroy.argtypes = [ct.c_void_p]
_lib.gs_destroy.restype = None

_lib.gs_add_planet.argtypes = [
    ct.c_void_p, ct.c_int, ct.c_int, ct.c_double, ct.c_double,
    ct.c_double, ct.c_int, ct.c_int]
_lib.gs_add_planet.restype = ct.c_int

_lib.gs_inject_comet_group.argtypes = [
    ct.c_void_p, ct.c_int,
    ct.POINTER(ct.c_int),         # planet_ids
    ct.POINTER(ct.c_int),         # path_lengths
    ct.POINTER(ct.c_double),      # paths_x
    ct.POINTER(ct.c_double),      # paths_y
    ct.c_int]
_lib.gs_inject_comet_group.restype = ct.c_int

_lib.gs_step.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int),         # action_counts
    ct.POINTER(ct.c_int),         # action_offsets
    ct.POINTER(ct.c_double)]      # action_data
_lib.gs_step.restype = ct.c_int

_lib.gs_count_active_planets.argtypes = [ct.c_void_p]
_lib.gs_count_active_planets.restype = ct.c_int

_lib.gs_count_active_fleets.argtypes = [ct.c_void_p]
_lib.gs_count_active_fleets.restype = ct.c_int

_lib.gs_copy_planets.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
_lib.gs_copy_planets.restype = None

_lib.gs_copy_fleets.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
_lib.gs_copy_fleets.restype = None

_lib.gs_copy_initial_planets.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
_lib.gs_copy_initial_planets.restype = None

_lib.gs_copy_comet_planet_ids.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
_lib.gs_copy_comet_planet_ids.restype = ct.c_int

_lib.gs_get_done.argtypes = [ct.c_void_p]
_lib.gs_get_done.restype = ct.c_int
_lib.gs_get_next_fleet_id.argtypes = [ct.c_void_p]
_lib.gs_get_next_fleet_id.restype = ct.c_int
_lib.gs_is_terminated.argtypes = [ct.c_void_p]
_lib.gs_is_terminated.restype = ct.c_int


# ── Direct field access via Structure mirror ─────────────────────────────
# We expose state via accessor functions; reading scalar fields is also
# needed. For simplicity we set up ad-hoc reads through small helpers.

# We need to read step, done, scores, next_fleet_id, angular_velocity.
# These live in the GameState struct. Easiest: add read accessors? For now
# track step/done/next_fleet_id/av in Python after each call too.


# ═════════════════════════════ CGame wrapper ═════════════════════════════

class CGame:
    """Drop-in replacement for OfficialFastGame using the C engine."""

    def __init__(
        self,
        n_players: int = 2,
        *,
        seed: int | None = None,
        episode_steps: int = 500,
        ship_speed: float = 6.0,
        comet_speed: float = 4.0,
        remaining_overage_time: float = 60.0,
    ):
        self.n_players = int(n_players)
        self.seed = seed
        self.episode_steps = int(episode_steps)
        self.ship_speed = float(ship_speed)
        self.comet_speed = float(comet_speed)
        self.remaining_overage_time = float(remaining_overage_time)

        # Generate initial state in Python using the official RNG path
        if seed is None:
            seed = random.randrange(2**31)
        self._episode_seed = int(seed)
        init_rng = random.Random(self._episode_seed)
        angular_velocity = init_rng.uniform(0.025, 0.05)
        planets = orbit_wars_py.generate_planets(init_rng)
        # Snapshot BEFORE home assignment — matches Python interpreter line 348
        # (obs0.initial_planets = [p.copy() for p in obs0.planets] runs before
        #  home assignment).
        initial_planets_snapshot = [list(p) for p in planets]

        # Assign home planets — same logic as Python interpreter
        num_groups = len(planets) // 4
        if num_groups > 0:
            home_group = init_rng.randint(0, num_groups - 1)
            base = home_group * 4
            if n_players == 2:
                planets[base][1] = 0
                planets[base][5] = 10
                planets[base + 3][1] = 1
                planets[base + 3][5] = 10
            elif n_players == 4:
                for j in range(4):
                    planets[base + j][1] = j
                    planets[base + j][5] = 10

        # Allocate C state
        self._handle = _lib.gs_create(
            n_players, ship_speed, comet_speed, episode_steps, angular_velocity)
        if not self._handle:
            raise RuntimeError("gs_create failed")

        # Push initial planets to C
        for p in planets:
            rc = _lib.gs_add_planet(
                self._handle,
                int(p[0]), int(p[1]),
                float(p[2]), float(p[3]), float(p[4]),
                int(p[5]), int(p[6]))
            if rc != 0:
                raise RuntimeError(f"gs_add_planet overflow at id={p[0]}")

        # Python-side mirror state
        self._step = 0
        self._done = False
        self._angular_velocity = angular_velocity
        self._next_fleet_id = 0
        # initial_planets snapshot (matches Python's pre-home-assignment snapshot)
        self._initial_planets = initial_planets_snapshot
        # Comet bookkeeping (Python-side; C also tracks)
        self._comet_planet_ids: list[int] = []
        # ── Performance caches (logic-neutral) ────────────────────────────
        self._p_buf = (ct.c_double * (_MAX_PLANETS * 7))()
        self._f_buf = (ct.c_double * (_MAX_FLEETS * 7))()
        self._ids_buf = (ct.c_int * _MAX_PLANETS)()
        self._obs_cache: list = [None] * self.n_players
        self._obs_shared: tuple | None = None
        self._initial_planets_cache: list | None = None
        # The configuration mirror
        self.configuration = FastConfig(
            episodeSteps=episode_steps,
            shipSpeed=ship_speed,
            cometSpeed=comet_speed,
            seed=None,
            remainingOverageTime=remaining_overage_time,
        )

    @classmethod
    def from_planets(
        cls,
        n_players: int,
        planets: list,
        angular_velocity: float,
        initial_planets: list | None = None,
        *,
        episode_steps: int = 500,
        ship_speed: float = 6.0,
        comet_speed: float = 4.0,
        remaining_overage_time: float = 60.0,
    ) -> "CGame":
        """Init from existing planet config (e.g. compact replay dataset)."""
        obj = object.__new__(cls)
        obj.n_players = int(n_players)
        obj.seed = None
        obj.episode_steps = int(episode_steps)
        obj.ship_speed = float(ship_speed)
        obj.comet_speed = float(comet_speed)
        obj.remaining_overage_time = float(remaining_overage_time)
        obj._episode_seed = 0
        obj._handle = _lib.gs_create(
            n_players, ship_speed, comet_speed, episode_steps, angular_velocity)
        if not obj._handle:
            raise RuntimeError("gs_create failed")
        for p in planets:
            rc = _lib.gs_add_planet(
                obj._handle,
                int(p[0]), int(p[1]),
                float(p[2]), float(p[3]), float(p[4]),
                int(p[5]), int(p[6]))
            if rc != 0:
                raise RuntimeError(f"gs_add_planet overflow at id={p[0]}")
        obj._step = 0
        obj._done = False
        obj._angular_velocity = float(angular_velocity)
        obj._next_fleet_id = 0
        obj._initial_planets = [list(p) for p in (initial_planets if initial_planets is not None else planets)]
        obj._comet_planet_ids = []
        obj._p_buf = (ct.c_double * (_MAX_PLANETS * 7))()
        obj._f_buf = (ct.c_double * (_MAX_FLEETS * 7))()
        obj._ids_buf = (ct.c_int * _MAX_PLANETS)()
        obj._obs_cache = [None] * obj.n_players
        obj._obs_shared = None
        obj._initial_planets_cache = None
        obj.configuration = FastConfig(
            episodeSteps=episode_steps,
            shipSpeed=ship_speed,
            cometSpeed=comet_speed,
            seed=None,
            remainingOverageTime=remaining_overage_time,
        )
        return obj

    def __del__(self):
        if getattr(self, '_handle', None):
            _lib.gs_destroy(self._handle)
            self._handle = None

    @property
    def done(self) -> bool:
        return self._done

    # ── Observation ──────────────────────────────────────────────────────

    def _build_observation(self, player_id: int) -> SimpleNamespace:
        """Construct the per-player observation Struct."""
        # Per-player obs cache — same step, same player → already built
        cached = self._obs_cache[player_id]
        if cached is not None:
            return cached
        # Cross-player shared marshalling — C state common to both players
        shared = self._obs_shared
        if shared is None:
            n_p = _lib.gs_count_active_planets(self._handle)
            n_f = _lib.gs_count_active_fleets(self._handle)

            if n_p > 0:
                _lib.gs_copy_planets(self._handle, self._p_buf)
                p_buf = self._p_buf
                planets = []
                for i in range(n_p):
                    o = i * 7
                    planets.append([
                        int(p_buf[o + 0]), int(p_buf[o + 1]),
                        float(p_buf[o + 2]), float(p_buf[o + 3]),
                        float(p_buf[o + 4]),
                        int(p_buf[o + 5]), int(p_buf[o + 6])])
            else:
                planets = []

            if n_f > 0:
                _lib.gs_copy_fleets(self._handle, self._f_buf)
                f_buf = self._f_buf
                fleets = []
                for i in range(n_f):
                    o = i * 7
                    fleets.append([
                        int(f_buf[o + 0]), int(f_buf[o + 1]),
                        float(f_buf[o + 2]), float(f_buf[o + 3]),
                        float(f_buf[o + 4]),
                        int(f_buf[o + 5]), int(f_buf[o + 6])])
            else:
                fleets = []

            n_ids = _lib.gs_copy_comet_planet_ids(self._handle, self._ids_buf)
            comet_ids = [int(self._ids_buf[i]) for i in range(n_ids)]

            # initial_planets list — cached, rebuilt only on comet inject
            if self._initial_planets_cache is None:
                initial = [list(p) for p in self._initial_planets]
                self._initial_planets_cache = initial
            else:
                initial = self._initial_planets_cache

            shared = (planets, fleets, comet_ids, initial)
            self._obs_shared = shared
        else:
            planets, fleets, comet_ids, initial = shared

        # MATCH PYTHON BUG: OfficialFastGame._set_step only updates state[0],
        # so player>0 always observes step=0. Reproduce that behavior here.
        observed_step = self._step if player_id == 0 else 0
        ns = SimpleNamespace(
            player=player_id,
            step=observed_step,
            planets=planets,
            fleets=fleets,
            initial_planets=initial,
            angular_velocity=self._angular_velocity,
            next_fleet_id=self._next_fleet_id,
            remainingOverageTime=self.remaining_overage_time,
            comet_planet_ids=comet_ids,
            comets=[],
        )
        self._obs_cache[player_id] = ns
        return ns

    def observation(self, player_id: int) -> SimpleNamespace:
        return self._build_observation(int(player_id))

    # ── Comet injection (Python-driven) ──────────────────────────────────

    def _injected_comets(self) -> list[list[list]]:
        return getattr(self, '_injected_comet_records', [])

    def _maybe_inject_comets(self):
        """Called before each step. If next step is a spawn step, generate
        comet paths in Python and inject into C."""
        if (self._step + 1) not in COMET_SPAWN_STEPS:
            return
        comet_rng = random.Random(
            f"orbit_wars-comet-{self._episode_seed}-{self._step + 1}")

        # Build the initial_planets list directly (avoid triggering obs cache
        # mid-step — _build_observation would cache a pre-step snapshot).
        ip = [list(p) for p in self._initial_planets]

        comet_paths = orbit_wars_py.generate_comet_paths(
            ip,
            self._angular_velocity,
            self._step + 1,
            self._comet_planet_ids,
            self.comet_speed,
            rng=comet_rng,
        )
        if not comet_paths:
            return

        # Get current max planet id from C (reuse shared buffer)
        n_p = _lib.gs_count_active_planets(self._handle)
        if n_p > 0:
            _lib.gs_copy_planets(self._handle, self._p_buf)
            max_id = -1
            p_buf = self._p_buf
            for i in range(n_p):
                pid = int(p_buf[i * 7])
                if pid > max_id:
                    max_id = pid
        else:
            max_id = -1
        next_id = max_id + 1

        comet_ships = min(
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
        )

        # Pack into ctypes buffers
        n_paths = len(comet_paths)
        planet_ids = (ct.c_int * n_paths)()
        path_lens = (ct.c_int * n_paths)()
        paths_x = (ct.c_double * (n_paths * _MAX_COMET_PATH_LEN))()
        paths_y = (ct.c_double * (n_paths * _MAX_COMET_PATH_LEN))()
        for i, p_path in enumerate(comet_paths):
            pid = next_id + i
            planet_ids[i] = pid
            self._comet_planet_ids.append(pid)
            plen = min(len(p_path), _MAX_COMET_PATH_LEN)
            path_lens[i] = plen
            for j in range(plen):
                paths_x[i * _MAX_COMET_PATH_LEN + j] = p_path[j][0]
                paths_y[i * _MAX_COMET_PATH_LEN + j] = p_path[j][1]

        rc = _lib.gs_inject_comet_group(
            self._handle, n_paths, planet_ids, path_lens, paths_x, paths_y,
            comet_ships)
        if rc != 0:
            raise RuntimeError("gs_inject_comet_group failed")

        # Mirror to Python initial_planets list
        records = []
        for i in range(n_paths):
            pid = next_id + i
            records.append([pid, -1, -99.0, -99.0,
                            orbit_wars_py.COMET_RADIUS, comet_ships,
                            orbit_wars_py.COMET_PRODUCTION])
        self._initial_planets.extend(records)
        self._initial_planets_cache = None

    # ── Step ────────────────────────────────────────────────────────────

    def step(self, actions_per_player: Sequence[Sequence[Sequence]]):
        """actions_per_player[p] = list of [from_id, angle, ships] triples."""
        if self._done:
            raise RuntimeError("Game done")
        # Invalidate observation caches — state about to change
        self._obs_cache = [None] * self.n_players
        self._obs_shared = None
        # Comet injection happens BEFORE the C interpreter (matches Python order)
        self._maybe_inject_comets()

        n = self.n_players
        counts = [len(a) if a else 0 for a in actions_per_player]
        # Sanitize: ensure each entry is a [from_id, angle, ships] triple
        flat = []
        offsets = [0] * n
        running = 0
        for i in range(n):
            offsets[i] = running
            a = actions_per_player[i] or []
            valid = []
            for move in a:
                if isinstance(move, (list, tuple)) and len(move) == 3:
                    valid.append(move)
            counts[i] = len(valid)
            for move in valid:
                flat.extend([float(move[0]), float(move[1]), float(move[2])])
            running += counts[i]

        c_counts = (ct.c_int * n)(*counts)
        c_offsets = (ct.c_int * n)(*offsets)
        if running == 0:
            c_data = (ct.c_double * 1)()  # dummy non-null
        else:
            c_data = (ct.c_double * (running * 3))(*flat)

        rc = _lib.gs_step(self._handle, c_counts, c_offsets, c_data)
        if rc != 0:
            raise RuntimeError(f"gs_step failed rc={rc}")

        # next_fleet_id from C (single scalar call, no buffer copy)
        self._next_fleet_id = _lib.gs_get_next_fleet_id(self._handle)

        # Increment step (matches OfficialFastGame._set_step)
        self._step += 1

        # Done detection (match interpreter logic): we mirror via direct check
        self._update_done_flag()

        # Defensive: invalidate caches after step in case any internal path
        # repopulated them between gs_step and now.
        self._obs_cache = [None] * self.n_players
        self._obs_shared = None

    def _update_done_flag(self):
        """Detect terminal state via C accessor (avoids buffer copy)."""
        if self._step >= self.episode_steps - 1:
            self._done = True
            return
        if _lib.gs_is_terminated(self._handle):
            self._done = True

    def scores(self) -> list[int]:
        n_p = _lib.gs_count_active_planets(self._handle)
        n_f = _lib.gs_count_active_fleets(self._handle)
        scores = [0] * self.n_players
        if n_p > 0:
            _lib.gs_copy_planets(self._handle, self._p_buf)
            p_buf = self._p_buf
            for i in range(n_p):
                o = int(p_buf[i * 7 + 1])
                if 0 <= o < self.n_players:
                    scores[o] += int(p_buf[i * 7 + 5])
        if n_f > 0:
            _lib.gs_copy_fleets(self._handle, self._f_buf)
            f_buf = self._f_buf
            for i in range(n_f):
                o = int(f_buf[i * 7 + 1])
                if 0 <= o < self.n_players:
                    scores[o] += int(f_buf[i * 7 + 6])
        return scores

    def winner(self) -> int:
        s = self.scores()
        m = max(s) if s else 0
        winners = [i for i, sc in enumerate(s) if sc == m and m > 0]
        return winners[0] if len(winners) == 1 else -1
