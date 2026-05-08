/*
 * orbit_wars.h — Public C API for the Orbit Wars engine port.
 *
 * The C engine is a faithful port of local_simulator/orbit_wars_official.py
 * `interpreter()` step logic. Initial state generation and comet path
 * generation remain in Python (RNG determinism is too fragile to port).
 *
 * Lifecycle:
 *   1. Caller allocates GameState via gs_create()
 *   2. Caller populates initial planets via gs_init_planets()
 *   3. Each turn:
 *        - If (step+1) is a comet spawn step, caller injects pre-computed
 *          comet paths via gs_inject_comet_group()
 *        - Caller submits actions via gs_step()
 *   4. Caller reads state back via gs_get_*() functions
 *   5. Caller frees with gs_destroy()
 *
 * Determinism: the engine produces results identical to the Python reference
 * up to libm floating-point precision. The parity test suite verifies this.
 */

#ifndef ORBIT_WARS_H
#define ORBIT_WARS_H

#include <stddef.h>
#include "orbit_wars_consts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int id;
    int owner;          /* -1 = neutral */
    double x, y;
    double radius;
    int ships;
    int production;
    /* Initial position used for orbit calc; matches initial_planets[i] in Python */
    double init_x, init_y;
    int is_comet;       /* 1 if comet planet */
    int comet_group;    /* index into comets[] if is_comet, else -1 */
    int comet_index;    /* position in group->planet_ids, else -1 */
    int active;         /* 0 if removed (expired comet) */
} Planet;

typedef struct {
    int id;
    int owner;
    double x, y;
    double angle;
    int from_planet_id;
    int ships;
    int active;         /* 0 if hit/OOB/sun-destroyed */
} Fleet;

typedef struct {
    /* Up to 4 quadrant-symmetric paths */
    int planet_ids[4];          /* planet IDs assigned to each path */
    int n_paths;                /* always 4 for normal spawn */
    int path_lengths[4];
    double paths_x[4][MAX_COMET_PATH_LEN];
    double paths_y[4][MAX_COMET_PATH_LEN];
    int path_index;             /* -1 before first advance, then increments */
    int active;                 /* 0 if all comet planets removed */
} CometGroup;

typedef struct {
    /* Planets — kept in insertion order. Holes (active==0) NOT compacted to
     * preserve indexing during a single step; compacted at start of next step. */
    Planet planets[MAX_PLANETS];
    int num_planets;            /* including inactive holes */

    /* Fleets — same: order preserved, holes compacted between steps */
    Fleet fleets[MAX_FLEETS];
    int num_fleets;

    CometGroup comets[MAX_COMET_GROUPS];
    int num_comet_groups;

    int next_fleet_id;
    double angular_velocity;
    int step;
    int num_agents;
    double ship_speed;
    double comet_speed;
    int episode_steps;
    int done;

    /* Scoring (computed at termination) */
    int scores[MAX_AGENTS];
    int rewards[MAX_AGENTS];
} GameState;

/* ── Lifecycle ────────────────────────────────────────────────────────────── */

GameState* gs_create(int n_agents,
                     double ship_speed,
                     double comet_speed,
                     int episode_steps,
                     double angular_velocity);

void gs_destroy(GameState* gs);

/* Append a planet (initial state setup). Call before first gs_step(). */
int gs_add_planet(GameState* gs,
                  int id, int owner,
                  double x, double y, double radius,
                  int ships, int production);

/* ── Per-turn operations ──────────────────────────────────────────────────── */

/* Inject a pre-computed comet group. Called by Python wrapper when
 * step+1 is in COMET_SPAWN_STEPS. After this, gs_step() will treat the new
 * comet planets as part of the world. */
int gs_inject_comet_group(GameState* gs,
                          int n_paths,
                          const int* planet_ids,
                          const int* path_lengths,
                          const double* paths_x,
                          const double* paths_y,
                          int comet_ships);

/* Submit per-player action arrays and run one full interpreter step.
 * action_data layout: [from_id, angle, ships] triples, packed flat.
 * action_offsets[i] = starting triple index for player i's actions.
 * action_counts[i]  = number of triples for player i.
 *
 * Returns 0 on success, non-zero on error (invalid bounds).
 */
int gs_step(GameState* gs,
            const int* action_counts,
            const int* action_offsets,
            const double* action_data);

/* ── State readback ──────────────────────────────────────────────────────── */

/* Return number of currently-active planets. */
int gs_count_active_planets(const GameState* gs);

/* Return number of currently-active fleets. */
int gs_count_active_fleets(const GameState* gs);

/* Copy active planet records into out_buf as flat doubles
 * [id, owner, x, y, radius, ships, production] per planet, in original order.
 * out_buf must have capacity >= 7 * gs_count_active_planets().
 */
void gs_copy_planets(const GameState* gs, double* out_buf);

/* Same for fleets: [id, owner, x, y, angle, from_planet_id, ships]. */
void gs_copy_fleets(const GameState* gs, double* out_buf);

/* Copy initial planet positions (init_x, init_y per active planet). */
void gs_copy_initial_planets(const GameState* gs, double* out_buf);

/* Copy comet planet IDs (active ones). out_buf must have capacity. */
int gs_copy_comet_planet_ids(const GameState* gs, int* out_buf);

#ifdef __cplusplus
}
#endif

#endif
