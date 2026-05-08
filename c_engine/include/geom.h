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

#ifdef __cplusplus
}
#endif

#endif
