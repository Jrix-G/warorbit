"""Regression checks for the fast local simulator geometry."""

from SimGame import CENTER, SUN_RADIUS, segment_point_dist_sq


def test_sun_collision_detected():
    # Straight line through the sun center must be a collision.
    d2 = segment_point_dist_sq(CENTER, CENTER, 0.0, CENTER, 100.0, CENTER)
    assert d2 < SUN_RADIUS * SUN_RADIUS


def test_near_miss_not_colliding():
    # A path far from the sun should remain outside the collision radius.
    d2 = segment_point_dist_sq(CENTER + 40.0, CENTER + 31.0, 0.0, CENTER + 20.0, 100.0, CENTER + 20.0)
    assert d2 > SUN_RADIUS * SUN_RADIUS
