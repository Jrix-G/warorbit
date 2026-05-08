/*
 * orbit_wars.c — C port of local_simulator/orbit_wars_official.py
 *
 * The interpreter() step is ported faithfully line-by-line. Generation of
 * planets and comet paths remains in Python (RNG determinism), and Python
 * injects them via gs_add_planet() / gs_inject_comet_group().
 *
 * Determinism notes:
 *  - Python iteration order over `obs0.planets` and `obs0.fleets` is preserved
 *    (we use insertion-order arrays, no hash maps).
 *  - Python `combat_lists` is a dict ordered by planet creation; we mirror
 *    via a per-planet vector of fleet indices.
 *  - `fleets_to_remove` membership in Python is a list `not in` check; we use
 *    a per-fleet `active` flag, equivalent semantics, faster.
 *  - `sorted(player_ships.items(), key=ships, reverse=True)` is stable on
 *    ties; we use the same — qsort is not stable so we add an insertion-index
 *    tiebreaker to reproduce Python order exactly.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "orbit_wars.h"
#include "geom.h"

/* ═════════════════════════════ INTERNAL HELPERS ═════════════════════════ */

/* Find a planet by id; returns index in gs->planets[] or -1. */
static int find_planet_by_id(const GameState* gs, int id) {
    for (int i = 0; i < gs->num_planets; ++i) {
        if (gs->planets[i].active && gs->planets[i].id == id) return i;
    }
    return -1;
}

/* Compact arrays: remove inactive entries, preserve relative order. */
static void compact_planets(GameState* gs) {
    int w = 0;
    for (int r = 0; r < gs->num_planets; ++r) {
        if (gs->planets[r].active) {
            if (w != r) gs->planets[w] = gs->planets[r];
            /* If this planet belongs to a comet group, refresh comet_index
             * after possible reordering. We do this in a second pass. */
            ++w;
        }
    }
    gs->num_planets = w;
}

static void compact_fleets(GameState* gs) {
    int w = 0;
    for (int r = 0; r < gs->num_fleets; ++r) {
        if (gs->fleets[r].active) {
            if (w != r) gs->fleets[w] = gs->fleets[r];
            ++w;
        }
    }
    gs->num_fleets = w;
}

/* ═════════════════════════════ LIFECYCLE ════════════════════════════════ */

GameState* gs_create(int n_agents,
                     double ship_speed,
                     double comet_speed,
                     int episode_steps,
                     double angular_velocity)
{
    GameState* gs = (GameState*)calloc(1, sizeof(GameState));
    if (!gs) return NULL;
    gs->num_agents = n_agents;
    gs->ship_speed = ship_speed;
    gs->comet_speed = comet_speed;
    gs->episode_steps = episode_steps;
    gs->angular_velocity = angular_velocity;
    gs->step = 0;
    gs->done = 0;
    gs->next_fleet_id = 0;
    gs->num_planets = 0;
    gs->num_fleets = 0;
    gs->num_comet_groups = 0;
    return gs;
}

void gs_destroy(GameState* gs) {
    free(gs);
}

int gs_add_planet(GameState* gs,
                  int id, int owner,
                  double x, double y, double radius,
                  int ships, int production)
{
    if (gs->num_planets >= MAX_PLANETS) return -1;
    Planet* p = &gs->planets[gs->num_planets];
    p->id = id;
    p->owner = owner;
    p->x = x;
    p->y = y;
    p->radius = radius;
    p->ships = ships;
    p->production = production;
    p->init_x = x;          /* For non-comets, init_x == x at creation. */
    p->init_y = y;
    p->is_comet = 0;
    p->comet_group = -1;
    p->comet_index = -1;
    p->active = 1;
    gs->num_planets++;
    return 0;
}

/* Inject a comet group. Mirrors Python lines 436-453:
 *   - Adds N (typically 4) new planets with off-board placeholder coords
 *     (Python uses -99,-99). On first advancement they jump to path[0].
 *   - Records each new planet ID in obs0.comet_planet_ids.
 *   - The CometGroup tracks paths and path_index (-1 initially).
 */
int gs_inject_comet_group(GameState* gs,
                          int n_paths,
                          const int* planet_ids,
                          const int* path_lengths,
                          const double* paths_x,
                          const double* paths_y,
                          int comet_ships)
{
    if (gs->num_comet_groups >= MAX_COMET_GROUPS) return -1;
    if (n_paths < 1 || n_paths > 4) return -1;
    if (gs->num_planets + n_paths > MAX_PLANETS) return -1;

    CometGroup* g = &gs->comets[gs->num_comet_groups];
    g->n_paths = n_paths;
    g->path_index = -1;
    g->active = 1;

    for (int k = 0; k < n_paths; ++k) {
        int pid = planet_ids[k];
        int plen = path_lengths[k];
        if (plen > MAX_COMET_PATH_LEN) plen = MAX_COMET_PATH_LEN;
        g->planet_ids[k] = pid;
        g->path_lengths[k] = plen;
        for (int i = 0; i < plen; ++i) {
            g->paths_x[k][i] = paths_x[k * MAX_COMET_PATH_LEN + i];
            g->paths_y[k][i] = paths_y[k * MAX_COMET_PATH_LEN + i];
        }

        /* Add comet planet record */
        Planet* p = &gs->planets[gs->num_planets];
        p->id = pid;
        p->owner = -1;
        p->x = -99.0;
        p->y = -99.0;
        p->radius = COMET_RADIUS;
        p->ships = comet_ships;
        p->production = COMET_PRODUCTION;
        p->init_x = -99.0;
        p->init_y = -99.0;
        p->is_comet = 1;
        p->comet_group = gs->num_comet_groups;
        p->comet_index = k;
        p->active = 1;
        gs->num_planets++;
    }

    gs->num_comet_groups++;
    return 0;
}

/* ═════════════════════════════ STEP HELPERS ═════════════════════════════ */

/* Python lines 388-408: remove expired comets at start of step. */
static void expire_comets_pre_step(GameState* gs) {
    for (int g = 0; g < gs->num_comet_groups; ++g) {
        CometGroup* group = &gs->comets[g];
        if (!group->active) continue;
        int idx = group->path_index;
        for (int k = 0; k < group->n_paths; ++k) {
            int pid = group->planet_ids[k];
            if (pid < 0) continue;  /* already removed */
            int p_idx = find_planet_by_id(gs, pid);
            if (p_idx < 0) continue;
            if (idx >= group->path_lengths[k]) {
                gs->planets[p_idx].active = 0;
                group->planet_ids[k] = -1;
            }
        }
        /* If all paths in group expired, mark group inactive */
        int any_alive = 0;
        for (int k = 0; k < group->n_paths; ++k) {
            if (group->planet_ids[k] >= 0) { any_alive = 1; break; }
        }
        if (!any_alive) group->active = 0;
    }
    compact_planets(gs);
}

/* Python lines 455-488: process player actions. */
static void process_moves(GameState* gs,
                          int player_id,
                          const double* actions,  /* triples */
                          int n_triples)
{
    for (int t = 0; t < n_triples; ++t) {
        int from_id = (int)actions[t * 3 + 0];
        double angle = actions[t * 3 + 1];
        int ships = (int)actions[t * 3 + 2];

        int p_idx = find_planet_by_id(gs, from_id);
        if (p_idx < 0) continue;
        Planet* from_planet = &gs->planets[p_idx];
        if (from_planet->owner != player_id) continue;
        if (ships <= 0 || from_planet->ships < ships) continue;

        /* Subtract from planet, create fleet */
        from_planet->ships -= ships;
        if (gs->num_fleets >= MAX_FLEETS) continue;
        Fleet* f = &gs->fleets[gs->num_fleets++];
        f->id = gs->next_fleet_id++;
        f->owner = player_id;
        /* Start fleet just outside the planet's radius (matches Python) */
        f->x = from_planet->x + cos(angle) * (from_planet->radius + 0.1);
        f->y = from_planet->y + sin(angle) * (from_planet->radius + 0.1);
        f->angle = angle;
        f->from_planet_id = from_id;
        f->ships = ships;
        f->active = 1;
    }
}

/* ── Combat list: per-planet fleet index list, in insertion order ──────── */

typedef struct {
    int fleet_indices[MAX_FLEETS];
    int n;
} PlanetCombatList;

/* Combat sort with stable tiebreak.
 * Python `sorted(..., reverse=True)` is stable; equal-ship players keep
 * original insertion order. We attach an insertion index to each entry. */
typedef struct {
    int owner;
    int ships;
    int insert_idx;
} PlayerShipEntry;

static int compare_player_ships(const void* a, const void* b) {
    const PlayerShipEntry* ea = (const PlayerShipEntry*)a;
    const PlayerShipEntry* eb = (const PlayerShipEntry*)b;
    if (ea->ships != eb->ships) {
        return (eb->ships > ea->ships) - (eb->ships < ea->ships);
    }
    /* Stable tiebreak: lower insert_idx first (matches Python order). */
    return (ea->insert_idx > eb->insert_idx) - (ea->insert_idx < eb->insert_idx);
}

/* ═════════════════════════════ MAIN STEP ════════════════════════════════ */

int gs_step(GameState* gs,
            const int* action_counts,
            const int* action_offsets,
            const double* action_data)
{
    if (gs->done) return 0;

    /* Python lines 388-408: pre-step comet expiration */
    expire_comets_pre_step(gs);

    /* Note: Python's interpreter handles comet *spawn* INSIDE the same call,
     * before move processing. In our hybrid design Python orchestrator must
     * call gs_inject_comet_group() BEFORE gs_step() on the relevant turn.
     * The injection happens against the post-expiration state, which matches
     * Python's order (expiration block runs before the spawn block).
     */

    /* 0. Fleet Launch (Python line 455-488) */
    for (int i = 0; i < gs->num_agents; ++i) {
        process_moves(gs, i,
                      action_data + action_offsets[i] * 3,
                      action_counts[i]);
    }

    /* 1. Production (Python lines 490-493) */
    for (int i = 0; i < gs->num_planets; ++i) {
        Planet* p = &gs->planets[i];
        if (!p->active) continue;
        if (p->owner != -1) {
            p->ships += p->production;
        }
    }

    /* Per-planet combat lists, indexed by current planet array index.
     * Insertion order is preserved (Python dict is insertion-ordered). */
    static PlanetCombatList combat_lists[MAX_PLANETS];
    for (int i = 0; i < gs->num_planets; ++i) combat_lists[i].n = 0;

    /* 2. Fleet Movement (Python lines 495-533) */
    double max_speed = gs->ship_speed;
    double log1000 = log(1000.0);

    for (int fi = 0; fi < gs->num_fleets; ++fi) {
        Fleet* fleet = &gs->fleets[fi];
        if (!fleet->active) continue;
        double angle = fleet->angle;
        int ships = fleet->ships;
        double speed = 1.0 + (max_speed - 1.0) * pow(log((double)ships) / log1000, 1.5);
        if (speed > max_speed) speed = max_speed;

        double old_x = fleet->x;
        double old_y = fleet->y;
        fleet->x += cos(angle) * speed;
        fleet->y += sin(angle) * speed;
        double new_x = fleet->x;
        double new_y = fleet->y;

        /* Continuous collision: check planets first (Python line 514-523) */
        int hit_planet = 0;
        for (int pi = 0; pi < gs->num_planets; ++pi) {
            Planet* planet = &gs->planets[pi];
            if (!planet->active) continue;
            double d = geom_point_to_segment_distance(
                planet->x, planet->y, old_x, old_y, new_x, new_y);
            if (d < planet->radius) {
                if (combat_lists[pi].n < MAX_FLEETS) {
                    combat_lists[pi].fleet_indices[combat_lists[pi].n++] = fi;
                }
                fleet->active = 0;
                hit_planet = 1;
                break;
            }
        }
        if (hit_planet) continue;

        /* Out of bounds (Python line 526-528) */
        if (!(fleet->x >= 0.0 && fleet->x <= BOARD_SIZE &&
              fleet->y >= 0.0 && fleet->y <= BOARD_SIZE)) {
            fleet->active = 0;
            continue;
        }

        /* Sun collision (Python line 531-533) */
        double sd = geom_point_to_segment_distance(
            CENTER, CENTER, old_x, old_y, new_x, new_y);
        if (sd < SUN_RADIUS) {
            fleet->active = 0;
            continue;
        }
    }

    /* 3. Planet Movement & Sweep (Python lines 535-572)
     *
     * Python uses: initial_angle = atan2(dy, dx) where dx,dy = init_p[2]-CENTER
     *              current_angle = initial_angle + angular_velocity * step
     * We use init_x, init_y stored per planet. NOTE: Python references step
     * = obs0.step, which is the CURRENT step BEFORE the interpreter increments
     * it. Looking at Python: obs0.step is set after interpreter via core.py
     * (each step appends new state). When interpreter() reads `step = get(obs0, "step", 1)`
     * here, it reads the value set BEFORE this interpreter call.
     *
     * In our C, gs->step represents the same "current step". Python's
     * `obs0.step` is incremented by the wrapper outside the interpreter.
     * We mirror by reading gs->step (which the wrapper sets to t+1 before
     * calling step? No — match Python behavior: read current step here).
     *
     * Actually Python's behavior:
     *   - obs0.step starts at 0
     *   - interpreter is called; it reads step = obs0.step (e.g. 0)
     *   - planet rotation uses angular_velocity * step (= 0)
     *   - At end of interpreter, core.py sets new_state[0].observation.step = len(steps)
     *   - So next call's step is 1, etc.
     *
     * For OfficialFastGame the step is set in _set_step() AFTER interpreter
     * returns. So when interpreter runs at index 0, step is 0; at index 1,
     * step is 1; ... at index 220, step is 220.
     *
     * In our C wrapper: caller increments gs->step AFTER gs_step() returns.
     * That means inside gs_step(), gs->step holds the value as Python sees it.
     */
    int step_for_rotation = gs->step;
    double av = gs->angular_velocity;

    for (int pi = 0; pi < gs->num_planets; ++pi) {
        Planet* planet = &gs->planets[pi];
        if (!planet->active) continue;
        if (planet->is_comet) continue;  /* Python: skip comets here (line 556) */

        double dx = planet->init_x - CENTER;
        double dy = planet->init_y - CENTER;
        double r = sqrt(dx * dx + dy * dy);
        double old_x = planet->x;
        double old_y = planet->y;

        if (r + planet->radius < ROTATION_RADIUS_LIMIT) {
            double initial_angle = atan2(dy, dx);
            double current_angle = initial_angle + av * step_for_rotation;
            planet->x = CENTER + r * cos(current_angle);
            planet->y = CENTER + r * sin(current_angle);
        }

        /* sweep_fleets — Python line 541-552 */
        double new_x = planet->x;
        double new_y = planet->y;
        if (old_x != new_x || old_y != new_y) {
            for (int fi = 0; fi < gs->num_fleets; ++fi) {
                Fleet* fleet = &gs->fleets[fi];
                if (!fleet->active) continue;
                double sd = geom_point_to_segment_distance(
                    fleet->x, fleet->y, old_x, old_y, new_x, new_y);
                if (sd < planet->radius) {
                    if (combat_lists[pi].n < MAX_FLEETS) {
                        combat_lists[pi].fleet_indices[combat_lists[pi].n++] = fi;
                    }
                    fleet->active = 0;
                }
            }
        }
    }

    /* Comet movement (Python lines 574-592) */
    for (int g = 0; g < gs->num_comet_groups; ++g) {
        CometGroup* group = &gs->comets[g];
        if (!group->active) continue;
        group->path_index += 1;
        int idx = group->path_index;
        for (int k = 0; k < group->n_paths; ++k) {
            int pid = group->planet_ids[k];
            if (pid < 0) continue;
            int p_idx = find_planet_by_id(gs, pid);
            if (p_idx < 0) continue;
            Planet* planet = &gs->planets[p_idx];
            int plen = group->path_lengths[k];
            if (idx >= plen) {
                /* Mark for expiration; Python defers actual removal */
                /* We delay the deactivation until after sweep to match order */
                /* Actually Python adds to expired_comet_pids and removes after */
                /* Apply now (matches Python order: removal between movement+sweep block and combat) */
                continue;  /* will be handled below */
            }
            double old_x = planet->x;
            double old_y = planet->y;
            planet->x = group->paths_x[k][idx];
            planet->y = group->paths_y[k][idx];
            if (old_x >= 0.0) {  /* skip first placement (Python line 591) */
                double new_x = planet->x;
                double new_y = planet->y;
                if (old_x != new_x || old_y != new_y) {
                    for (int fi = 0; fi < gs->num_fleets; ++fi) {
                        Fleet* fleet = &gs->fleets[fi];
                        if (!fleet->active) continue;
                        double sd = geom_point_to_segment_distance(
                            fleet->x, fleet->y, old_x, old_y, new_x, new_y);
                        if (sd < planet->radius) {
                            if (combat_lists[p_idx].n < MAX_FLEETS) {
                                combat_lists[p_idx].fleet_indices[
                                    combat_lists[p_idx].n++] = fi;
                            }
                            fleet->active = 0;
                        }
                    }
                }
            }
        }
    }

    /* Now apply post-advancement comet expirations (Python lines 594-608) */
    for (int g = 0; g < gs->num_comet_groups; ++g) {
        CometGroup* group = &gs->comets[g];
        if (!group->active) continue;
        int idx = group->path_index;
        int any_alive = 0;
        for (int k = 0; k < group->n_paths; ++k) {
            int pid = group->planet_ids[k];
            if (pid < 0) continue;
            if (idx >= group->path_lengths[k]) {
                int p_idx = find_planet_by_id(gs, pid);
                if (p_idx >= 0) gs->planets[p_idx].active = 0;
                group->planet_ids[k] = -1;
            } else {
                any_alive = 1;
            }
        }
        if (!any_alive) group->active = 0;
    }

    /* 4. Combat Resolution (Python lines 612-651) */
    for (int pi = 0; pi < gs->num_planets; ++pi) {
        Planet* planet = &gs->planets[pi];
        if (!planet->active) continue;
        PlanetCombatList* cl = &combat_lists[pi];
        if (cl->n == 0) continue;

        /* Sum ships per player, in insertion order. Python uses dict. */
        PlayerShipEntry entries[MAX_AGENTS + 16];
        int n_entries = 0;
        for (int j = 0; j < cl->n; ++j) {
            int fi = cl->fleet_indices[j];
            Fleet* fleet = &gs->fleets[fi];
            int owner = fleet->owner;
            int found = -1;
            for (int e = 0; e < n_entries; ++e) {
                if (entries[e].owner == owner) { found = e; break; }
            }
            if (found >= 0) {
                entries[found].ships += fleet->ships;
            } else {
                entries[n_entries].owner = owner;
                entries[n_entries].ships = fleet->ships;
                entries[n_entries].insert_idx = n_entries;
                n_entries++;
            }
        }

        if (n_entries == 0) continue;

        qsort(entries, n_entries, sizeof(PlayerShipEntry), compare_player_ships);

        int top_player = entries[0].owner;
        int top_ships = entries[0].ships;
        int survivor_owner;
        int survivor_ships;

        if (n_entries > 1) {
            int second_ships = entries[1].ships;
            survivor_ships = top_ships - second_ships;
            if (entries[0].ships == entries[1].ships) {
                survivor_ships = 0;
            }
            survivor_owner = (survivor_ships > 0) ? top_player : -1;
        } else {
            survivor_owner = top_player;
            survivor_ships = top_ships;
        }

        if (survivor_ships > 0) {
            if (planet->owner == survivor_owner) {
                planet->ships += survivor_ships;
            } else {
                planet->ships -= survivor_ships;
                if (planet->ships < 0) {
                    planet->owner = survivor_owner;
                    planet->ships = -planet->ships;  /* abs */
                }
            }
        }
    }

    /* Compact arrays for next step */
    compact_fleets(gs);
    compact_planets(gs);

    /* Termination check (Python lines 661-693) */
    int terminated = 0;
    if (gs->step >= gs->episode_steps - 2) {
        terminated = 1;
    }

    /* alive_players check */
    int alive[MAX_AGENTS] = {0};
    int n_alive = 0;
    for (int i = 0; i < gs->num_planets; ++i) {
        Planet* p = &gs->planets[i];
        if (!p->active) continue;
        if (p->owner != -1 && p->owner < gs->num_agents) {
            if (!alive[p->owner]) { alive[p->owner] = 1; n_alive++; }
        }
    }
    for (int i = 0; i < gs->num_fleets; ++i) {
        Fleet* f = &gs->fleets[i];
        if (!f->active) continue;
        if (f->owner >= 0 && f->owner < gs->num_agents) {
            if (!alive[f->owner]) { alive[f->owner] = 1; n_alive++; }
        }
    }
    if (n_alive <= 1) terminated = 1;

    /* Increment step at end (matches OfficialFastGame._set_step after interpreter).
     * Inside this gs_step call, gs->step held the pre-step value (Python's
     * obs.step at start of interpreter). Now bump it for next call. */
    gs->step += 1;

    if (terminated) {
        gs->done = 1;
        for (int i = 0; i < gs->num_agents; ++i) gs->scores[i] = 0;
        for (int i = 0; i < gs->num_planets; ++i) {
            Planet* p = &gs->planets[i];
            if (!p->active) continue;
            if (p->owner != -1 && p->owner < gs->num_agents) {
                gs->scores[p->owner] += p->ships;
            }
        }
        for (int i = 0; i < gs->num_fleets; ++i) {
            Fleet* f = &gs->fleets[i];
            if (!f->active) continue;
            if (f->owner >= 0 && f->owner < gs->num_agents) {
                gs->scores[f->owner] += f->ships;
            }
        }
        int max_score = 0;
        for (int i = 0; i < gs->num_agents; ++i)
            if (gs->scores[i] > max_score) max_score = gs->scores[i];
        for (int i = 0; i < gs->num_agents; ++i) {
            gs->rewards[i] = (gs->scores[i] == max_score && max_score > 0) ? 1 : -1;
        }
    }

    return 0;
}

/* ═════════════════════════════ STATE READBACK ═══════════════════════════ */

int gs_count_active_planets(const GameState* gs) {
    int n = 0;
    for (int i = 0; i < gs->num_planets; ++i)
        if (gs->planets[i].active) ++n;
    return n;
}

int gs_count_active_fleets(const GameState* gs) {
    int n = 0;
    for (int i = 0; i < gs->num_fleets; ++i)
        if (gs->fleets[i].active) ++n;
    return n;
}

void gs_copy_planets(const GameState* gs, double* out_buf) {
    int o = 0;
    for (int i = 0; i < gs->num_planets; ++i) {
        const Planet* p = &gs->planets[i];
        if (!p->active) continue;
        out_buf[o++] = (double)p->id;
        out_buf[o++] = (double)p->owner;
        out_buf[o++] = p->x;
        out_buf[o++] = p->y;
        out_buf[o++] = p->radius;
        out_buf[o++] = (double)p->ships;
        out_buf[o++] = (double)p->production;
    }
}

void gs_copy_fleets(const GameState* gs, double* out_buf) {
    int o = 0;
    for (int i = 0; i < gs->num_fleets; ++i) {
        const Fleet* f = &gs->fleets[i];
        if (!f->active) continue;
        out_buf[o++] = (double)f->id;
        out_buf[o++] = (double)f->owner;
        out_buf[o++] = f->x;
        out_buf[o++] = f->y;
        out_buf[o++] = f->angle;
        out_buf[o++] = (double)f->from_planet_id;
        out_buf[o++] = (double)f->ships;
    }
}

void gs_copy_initial_planets(const GameState* gs, double* out_buf) {
    int o = 0;
    for (int i = 0; i < gs->num_planets; ++i) {
        const Planet* p = &gs->planets[i];
        if (!p->active) continue;
        out_buf[o++] = (double)p->id;
        out_buf[o++] = -1.0;       /* Python: initial_planets owner reset is not preserved here */
        out_buf[o++] = p->init_x;
        out_buf[o++] = p->init_y;
        out_buf[o++] = p->radius;
        out_buf[o++] = 0.0;        /* ships unused for orbit calcs */
        out_buf[o++] = (double)p->production;
    }
}

int gs_copy_comet_planet_ids(const GameState* gs, int* out_buf) {
    int n = 0;
    for (int i = 0; i < gs->num_planets; ++i) {
        const Planet* p = &gs->planets[i];
        if (!p->active) continue;
        if (p->is_comet) out_buf[n++] = p->id;
    }
    return n;
}
