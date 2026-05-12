"""Tests for capsule target geometry.

Covers:
- Targets: check_target_overlap, get_dist_to_targets, get_percep_angles
- PerceptionModel: _get_target_signals with capsule targets
- Zero-width (w=0) and finite-width (w>0) capsules
- End-on viewing, wrapping around +-pi, partial occlusion
- Both scalar and array l/w/theta parameters
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from decision_model import Targets, PerceptionModel, convert_angles, _smallest_enclosing_arc

pi = np.pi
passed = 0
failed = 0


def check(name, result, expected, tol=1e-10):
    global passed, failed
    result = np.asarray(result, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if result.shape != expected.shape:
        print(f"FAIL {name}: shape {result.shape} != {expected.shape}")
        print(f"  got:      {result}")
        print(f"  expected: {expected}")
        failed += 1
        return
    if np.allclose(result, expected, atol=tol):
        passed += 1
        print(f"  ok {name}")
    else:
        print(f"FAIL {name}:")
        print(f"  got:      {result}")
        print(f"  expected: {expected}")
        failed += 1


def check_bool(name, result, expected):
    global passed, failed
    result = np.asarray(result)
    expected = np.asarray(expected)
    if result.shape != expected.shape:
        print(f"FAIL {name}: shape {result.shape} != {expected.shape}")
        failed += 1
        return
    if np.array_equal(result, expected):
        passed += 1
        print(f"  ok {name}")
    else:
        print(f"FAIL {name}: got {result}, expected {expected}")
        failed += 1


# ========================================================
print("=== _smallest_enclosing_arc ===")
# ========================================================

# Four angles clustered in a small range
check("small arc", _smallest_enclosing_arc(np.array([-0.1, 0.05, 0.1, -0.05])),
      [-0.1, 0.1])

# Arc that wraps around +-pi
check("wrapping arc", _smallest_enclosing_arc(np.array([2.9, 3.0, -3.0, -2.9])),
      [2.9, -2.9])

# Two points
check("two points", _smallest_enclosing_arc(np.array([0.5, 1.5])),
      [0.5, 1.5])

# All same angle
check("degenerate", _smallest_enclosing_arc(np.array([1.0, 1.0, 1.0, 1.0])),
      [1.0, 1.0])


# ========================================================
print("\n=== check_target_overlap (capsule w=0) ===")
# ========================================================

# Zero-width capsule = line segment
tgt = Targets(locs=np.array([[5, 5]]), geom_name='capsule', l=4, w=0, theta=0)

# Point exactly on the segment spine
check_bool("on spine midpoint w=0", tgt.check_target_overlap(np.array([5, 5])), [True])
check_bool("on spine near endpt w=0", tgt.check_target_overlap(np.array([4, 5])), [True])
# Point off the segment
check_bool("above spine w=0", tgt.check_target_overlap(np.array([5, 6])), [False])


# ========================================================
print("\n=== check_target_overlap (capsule w>0) ===")
# ========================================================

# Capsule at (5,5), l=4, w=2, theta=0: spine from (3,5) to (7,5), radius 1
tgt_w = Targets(locs=np.array([[5, 5]]), geom_name='capsule', l=4, w=2, theta=0)

# On the spine
check_bool("on spine w=2", tgt_w.check_target_overlap(np.array([5, 5])), [True])
# Inside endcap (within radius 1 of endpoint (7,5))
check_bool("in endcap", tgt_w.check_target_overlap(np.array([7.5, 5])), [True])
# Above spine within width
check_bool("above within width", tgt_w.check_target_overlap(np.array([5, 5.9])), [True])
# Outside
check_bool("above outside", tgt_w.check_target_overlap(np.array([5, 6.1])), [False])
check_bool("far away", tgt_w.check_target_overlap(np.array([0, 0])), [False])

# Multiple capsules with array params
tgt_multi = Targets(locs=np.array([[5, 5], [10, 10]]), geom_name='capsule',
                    l=np.array([4, 6]), w=np.array([2, 1]), theta=np.array([0, pi/2]))
check_bool("multi: inside first", tgt_multi.check_target_overlap(np.array([5, 5.5])), [True, False])
check_bool("multi: inside second", tgt_multi.check_target_overlap(np.array([10.4, 10])), [False, True])
check_bool("multi: neither", tgt_multi.check_target_overlap(np.array([0, 0])), [False, False])


# ========================================================
print("\n=== get_dist_to_targets (capsule) ===")
# ========================================================

# w=0: distance is to the spine
tgt0 = Targets(locs=np.array([[5, 5]]), geom_name='capsule', l=4, w=0, theta=0)
check("dist w=0 above midpoint", tgt0.get_dist_to_targets(np.array([5, 8])), [3.0])
check("dist w=0 past endpoint", tgt0.get_dist_to_targets(np.array([1, 5])), [2.0])

# w=2: distance is spine_dist - 1
tgt_w = Targets(locs=np.array([[5, 5]]), geom_name='capsule', l=4, w=2, theta=0)
check("dist w=2 above midpoint", tgt_w.get_dist_to_targets(np.array([5, 8])), [2.0])
check("dist w=2 inside", tgt_w.get_dist_to_targets(np.array([5, 5.5])), [0.0])

# Multiple with array params
tgt_m = Targets(locs=np.array([[5, 5], [10, 10]]), geom_name='capsule',
                l=np.array([4, 6]), w=np.array([2, 1]), theta=np.array([0, pi/2]))
dists = tgt_m.get_dist_to_targets(np.array([5, 8]))
# First: spine dist = 3.0, w/2 = 1.0, so dist = 2.0
check("multi dist first", dists[0:1], [2.0])
# Second: spine (10,7)-(10,13), closest point to (5,8) is (10,8), spine dist = 5.0,
# w/2 = 0.5, so dist = 4.5
check("multi dist second", dists[1:2], [4.5])


# ========================================================
print("\n=== get_percep_angles (capsule w=0, same as old segment) ===")
# ========================================================

# Perpendicular segment at (10,0), theta=pi/2, l=2, w=0
# Endpoints: (10,-1) and (10,1)
tgt_s = Targets(locs=np.array([[10, 0]]), geom_name='capsule', l=2, w=0, theta=pi/2)
angles = tgt_s.get_percep_angles(np.array([0, 0]), angle=0)
expected_lo = np.arctan2(-1, 10)
expected_hi = np.arctan2(1, 10)
check("w=0 perp segment lo", angles[0, 0:1], [expected_lo])
check("w=0 perp segment hi", angles[0, 1:2], [expected_hi])


# ========================================================
print("\n=== get_percep_angles (capsule w>0) ===")
# ========================================================

# Capsule at (10, 0), theta=pi/2, l=2, w=1 (endcap radius 0.5)
# Endpoints: (10,-1) and (10,1), each with a circle of radius 0.5
tgt_c = Targets(locs=np.array([[10, 0]]), geom_name='capsule', l=2, w=1, theta=pi/2)
angles_c = tgt_c.get_percep_angles(np.array([0, 0]), angle=0)
# The outer tangent angles:
# Endpoint (10, 1): center angle = arctan2(1,10), dist = sqrt(101),
#   half = arcsin(0.5/sqrt(101))
# Endpoint (10, -1): center angle = arctan2(-1,10), dist = sqrt(101),
#   half = arcsin(0.5/sqrt(101))
d_ep = np.sqrt(101)
half_ep = np.arcsin(0.5 / d_ep)
exp_lo = np.arctan2(-1, 10) - half_ep
exp_hi = np.arctan2(1, 10) + half_ep
check("w=1 capsule lo", angles_c[0, 0:1], [exp_lo], tol=1e-8)
check("w=1 capsule hi", angles_c[0, 1:2], [exp_hi], tol=1e-8)

# Capsule extent should be wider than zero-width segment
assert angles_c[0, 1] - angles_c[0, 0] > angles[0, 1] - angles[0, 0], \
    "Capsule w>0 should subtend wider angle than w=0"
print("  ok capsule wider than segment")
passed += 1


# ========================================================
print("\n=== End-on viewing ===")
# ========================================================

# Observer looking directly along a zero-width capsule's axis: angular extent -> 0
# Capsule at (10, 0), theta=0, l=4, w=0: endpoints (8,0) and (12,0)
tgt_end = Targets(locs=np.array([[10, 0]]), geom_name='capsule', l=4, w=0, theta=0)
angles_end = tgt_end.get_percep_angles(np.array([0, 0]), angle=0)
# Both endpoints are on the x-axis, so both angles are 0 -> degenerate arc
check("end-on w=0: zero extent", angles_end[0, 1] - angles_end[0, 0], 0.0, tol=1e-10)

# Same but with w=1: capsule has endcap radius 0.5
# Now it should have nonzero angular extent (like a circle of radius 0.5 at distance ~8-12)
tgt_end_w = Targets(locs=np.array([[10, 0]]), geom_name='capsule', l=4, w=1, theta=0)
angles_end_w = tgt_end_w.get_percep_angles(np.array([0, 0]), angle=0)
# Closer endpoint at (8,0): circle r=0.5, center angle=0, half = arcsin(0.5/8)
# Farther endpoint at (12,0): circle r=0.5, center angle=0, half = arcsin(0.5/12)
# The wider of the two dominates: lo = -arcsin(0.5/8), hi = arcsin(0.5/8)
exp_half = np.arcsin(0.5 / 8)
check("end-on w=1 lo", angles_end_w[0, 0:1], [-exp_half], tol=1e-8)
check("end-on w=1 hi", angles_end_w[0, 1:2], [exp_half], tol=1e-8)
assert angles_end_w[0, 1] - angles_end_w[0, 0] > 0, "Capsule w>0 should not vanish end-on"
print("  ok end-on w>0 has nonzero extent")
passed += 1


# ========================================================
print("\n=== Wrapping around +-pi ===")
# ========================================================

# Capsule behind observer straddling +-pi
tgt_behind = Targets(locs=np.array([[-5, 0]]), geom_name='capsule',
                     l=20, w=1, theta=pi/2)
angles_behind = tgt_behind.get_percep_angles(np.array([0, 0]), angle=0)
# This should be a wrapping interval (lo > hi)
lo_b, hi_b = angles_behind[0]
# Verify that the arc wraps
ccw_dist = (hi_b - lo_b) % (2*pi)
assert ccw_dist < pi, f"Behind capsule should subtend less than pi, got {ccw_dist}"
print(f"  ok wrapping capsule: lo={lo_b:.3f}, hi={hi_b:.3f}, arc={ccw_dist:.3f}")
passed += 1


# ========================================================
print("\n=== _get_target_signals: capsule w=0 ===")
# ========================================================

# Two non-overlapping capsules
tgt_two = Targets(locs=np.array([[10, 3], [10, -3]]), geom_name='capsule',
                  l=2, w=0, theta=pi/2)
pm = PerceptionModel(tgt_two, focal_loc=(0, 0), focal_angle=0,
                     neural_weight=None, neural_angle=None)
c_angles, rho = pm._get_target_signals()
check("two capsules: equal rho", rho, [0.5, 0.5], tol=1e-6)

# Full blocking: front capsule blocks one behind
tgt_block = Targets(locs=np.array([[5, 0], [10, 0]]), geom_name='capsule',
                    l=np.array([4, 2]), w=0, theta=np.array([pi/2, pi/2]))
pm_block = PerceptionModel(tgt_block, focal_loc=(0, 0), focal_angle=0,
                           neural_weight=None, neural_angle=None)
c_angles_b, rho_b = pm_block._get_target_signals()
check("blocking w=0: only front visible", len(rho_b), 1)

# Partial blocking
tgt_partial = Targets(locs=np.array([[5, 0], [10, 0]]), geom_name='capsule',
                      l=np.array([1, 10]), w=0, theta=np.array([pi/2, pi/2]))
pm_partial = PerceptionModel(tgt_partial, focal_loc=(0, 0), focal_angle=0,
                             neural_weight=None, neural_angle=None)
c_angles_p, rho_p = pm_partial._get_target_signals()
check("partial block w=0: both visible", len(rho_p), 2)


# ========================================================
print("\n=== _get_target_signals: capsule w>0 ===")
# ========================================================

# Two capsules with finite width, symmetric
tgt_w2 = Targets(locs=np.array([[10, 3], [10, -3]]), geom_name='capsule',
                 l=2, w=1, theta=pi/2)
pm_w = PerceptionModel(tgt_w2, focal_loc=(0, 0), focal_angle=0,
                       neural_weight=None, neural_angle=None)
c_angles_w, rho_w = pm_w._get_target_signals()
check("two capsules w=1: equal rho", rho_w, [0.5, 0.5], tol=1e-6)

# End-on capsule with w>0 should still be visible (not zero weight)
tgt_end_sig = Targets(locs=np.array([[10, 0]]), geom_name='capsule',
                      l=4, w=1, theta=0)
pm_end = PerceptionModel(tgt_end_sig, focal_loc=(0, 0), focal_angle=0,
                         neural_weight=None, neural_angle=None)
c_end, rho_end = pm_end._get_target_signals()
check("end-on w=1: visible", len(rho_end), 1)
if len(rho_end) == 1:
    check("end-on w=1: full weight", rho_end, [1.0])


# ========================================================
print("\n=== _get_target_signals: capsule with cutoff weight ===")
# ========================================================

tgt_cut = Targets(locs=np.array([[10, 3], [10, -3]]), geom_name='capsule',
                  l=2, w=1, theta=pi/2)
pm_cut = PerceptionModel(tgt_cut, focal_loc=(0, 0), focal_angle=0,
                         neural_weight='cutoff', neural_angle='integral')
c_angles_c, rho_c = pm_cut._get_target_signals()
check("cutoff capsule: equal rho", rho_c, [0.5, 0.5], tol=1e-6)


# ========================================================
print("\n=== Capsule w>0 blocking vs w=0 blocking ===")
# ========================================================

# A wide capsule in front should block more than a zero-width one
tgt_wide_front = Targets(locs=np.array([[5, 0], [10, 0]]), geom_name='capsule',
                         l=np.array([1, 10]), w=np.array([2, 0]),
                         theta=np.array([pi/2, pi/2]))
pm_wide = PerceptionModel(tgt_wide_front, focal_loc=(0, 0), focal_angle=0,
                          neural_weight=None, neural_angle=None)
c_wide, rho_wide = pm_wide._get_target_signals()

tgt_thin_front = Targets(locs=np.array([[5, 0], [10, 0]]), geom_name='capsule',
                         l=np.array([1, 10]), w=np.array([0, 0]),
                         theta=np.array([pi/2, pi/2]))
pm_thin = PerceptionModel(tgt_thin_front, focal_loc=(0, 0), focal_angle=0,
                          neural_weight=None, neural_angle=None)
c_thin, rho_thin = pm_thin._get_target_signals()

# Both should have 2 visible targets
check("wide front: both visible", len(rho_wide), 2)
check("thin front: both visible", len(rho_thin), 2)
# The wide front capsule should claim more weight (blocks more of back)
if len(rho_wide) == 2 and len(rho_thin) == 2:
    # rho[0] is front target - should be larger when front is wide
    assert rho_wide[0] > rho_thin[0], \
        f"Wide front should claim more weight: {rho_wide[0]} vs {rho_thin[0]}"
    print(f"  ok wide blocks more: front rho {rho_wide[0]:.4f} > {rho_thin[0]:.4f}")
    passed += 1


# ========================================================
print("\n=== Observer on capsule ===")
# ========================================================

tgt_on = Targets(locs=np.array([[0, 0], [10, 0]]), geom_name='capsule',
                 l=np.array([4, 2]), w=np.array([2, 1]),
                 theta=np.array([0, pi/2]))
pm_on = PerceptionModel(tgt_on, focal_loc=(0, 0), focal_angle=0,
                        neural_weight=None, neural_angle=None)
c_on, rho_on = pm_on._get_target_signals()
# First capsule overlaps observer -> [-pi, pi], blocks everything
check("on capsule: only overlapping visible", len(rho_on), 1)


# ========================================================
print("\n=== Capsule degenerates to circle when l=0 ===")
# ========================================================

# A capsule with l=0 and w=2r should look like a circle of radius r
r_test = 0.5
tgt_cap = Targets(locs=np.array([[10, 3]]), geom_name='capsule',
                  l=0, w=2*r_test, theta=0)
tgt_cir = Targets(locs=np.array([[10, 3]]), geom_name='circle', r=r_test)

angles_cap = tgt_cap.get_percep_angles(np.array([0, 0]), angle=0)
angles_cir = tgt_cir.get_percep_angles(np.array([0, 0]), angle=0)
check("l=0 capsule matches circle: lo", angles_cap[0, 0:1], angles_cir[0, 0:1], tol=1e-8)
check("l=0 capsule matches circle: hi", angles_cap[0, 1:2], angles_cir[0, 1:2], tol=1e-8)

dist_cap = tgt_cap.get_dist_to_targets(np.array([0, 0]))
dist_cir = tgt_cir.get_dist_to_targets(np.array([0, 0]))
check("l=0 capsule dist matches circle", dist_cap, dist_cir, tol=1e-10)


# ========================================================
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'='*40}")
