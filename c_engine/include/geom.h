#ifndef OW_GEOM_H
#define OW_GEOM_H

#ifdef __cplusplus
extern "C" {
#endif

double geom_distance(double ax, double ay, double bx, double by);

double geom_point_to_segment_distance(
    double px, double py,
    double vx, double vy,
    double wx, double wy);

/* Continuous collision: fleet moving A->B, planet moving P0->P1.
 * Returns 1 iff they come within radius r at any t in [0, 1]. */
int geom_swept_pair_hit(
    double ax, double ay, double bx, double by,
    double p0x, double p0y, double p1x, double p1y,
    double r);

#ifdef __cplusplus
}
#endif

#endif
