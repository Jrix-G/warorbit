/*
 * geom.c — Geometric primitives matching Python orbit_wars_official.py
 */
#include <math.h>
#include "geom.h"

double geom_distance(double ax, double ay, double bx, double by) {
    double dx = ax - bx;
    double dy = ay - by;
    return sqrt(dx * dx + dy * dy);
}

/* Mirrors Python's point_to_segment_distance(p, v, w):
 *   l2 = |w-v|^2
 *   if l2 == 0: return distance(p, v)
 *   t = clamp01(((p-v) . (w-v)) / l2)
 *   proj = v + t*(w-v)
 *   return distance(p, proj)
 *
 * Critical: must match the FP behavior of the Python version exactly,
 * which uses standard arithmetic and math.sqrt. C with libm matches.
 */
double geom_point_to_segment_distance(
    double px, double py,
    double vx, double vy,
    double wx, double wy)
{
    double dx = vx - wx;
    double dy = vy - wy;
    double l2 = dx * dx + dy * dy;
    if (l2 == 0.0) {
        return geom_distance(px, py, vx, vy);
    }
    double t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    double proj_x = vx + t * (wx - vx);
    double proj_y = vy + t * (wy - vy);
    return geom_distance(px, py, proj_x, proj_y);
}
