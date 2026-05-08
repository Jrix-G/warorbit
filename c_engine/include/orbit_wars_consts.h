/*
 * orbit_wars_consts.h — Game constants matching local_simulator/orbit_wars_official.py
 * KEEP IN SYNC WITH PYTHON. Any change here must mirror Python.
 */
#ifndef ORBIT_WARS_CONSTS_H
#define ORBIT_WARS_CONSTS_H

#define BOARD_SIZE 100.0
#define CENTER 50.0
#define SUN_RADIUS 10.0
#define ROTATION_RADIUS_LIMIT 50.0
#define COMET_RADIUS 1.0
#define COMET_PRODUCTION 1
#define PLANET_CLEARANCE 7

/* Hard caps — chosen safely above realistic max:
 *   Realistic planets per game: ~28 (4p) to 36 (2p), comets up to 5*4=20 added
 *   Realistic fleets: tested up to ~200 in heavy games
 */
#define MAX_PLANETS 80
#define MAX_FLEETS 4096
#define MAX_COMET_GROUPS 8
#define MAX_COMET_PATH_LEN 64
#define MAX_AGENTS 4
#define MAX_ACTIONS_PER_TURN 4096

#endif
