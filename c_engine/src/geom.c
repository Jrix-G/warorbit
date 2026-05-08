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

/* Mirrors Python's swept_pair_hit(A, B, P0, P1, r):
 *   d0 = A - P0
 *   dv = (B - A) - (P1 - P0)
 *   a = dv.dv ; b = 2 * d0.dv ; c = d0.d0 - r*r
 *   if a < 1e-12: return c <= 0
 *   disc = b*b - 4*a*c
 *   if disc < 0: return False
 *   sq = sqrt(disc); t1 = (-b - sq)/(2a); t2 = (-b + sq)/(2a)
 *   return t2 >= 0 and t1 <= 1
 */
int geom_swept_pair_hit(
    double ax, double ay, double bx, double by,
    double p0x, double p0y, double p1x, double p1y,
    double r)
{
    double d0x = ax - p0x;
    double d0y = ay - p0y;
    double dvx = (bx - ax) - (p1x - p0x);
    double dvy = (by - ay) - (p1y - p0y);
    double a = dvx * dvx + dvy * dvy;
    double b = 2.0 * (d0x * dvx + d0y * dvy);
    double c = d0x * d0x + d0y * d0y - r * r;
    if (a < 1e-12) {
        return c <= 0.0 ? 1 : 0;
    }
    double disc = b * b - 4.0 * a * c;
    if (disc < 0.0) return 0;
    double sq = sqrt(disc);
    double t1 = (-b - sq) / (2.0 * a);
    double t2 = (-b + sq) / (2.0 * a);
    return (t2 >= 0.0 && t1 <= 1.0) ? 1 : 0;
}
