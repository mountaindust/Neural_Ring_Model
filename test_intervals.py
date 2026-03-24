"""Exhaustive tests for _subtract_intervals_circle and helper methods."""

import numpy as np
from decision_model import PerceptionModel as PM

sub = PM._subtract_intervals_circle
unwrap = PM._unwrap_interval
sub_pair = PM._subtract_interval_pair

pi = np.pi
passed = 0
failed = 0


def check(name, result, expected, tol=1e-12):
    global passed, failed
    # Sort both for comparison
    result_s = sorted(result)
    expected_s = sorted(expected)
    if len(result_s) != len(expected_s):
        print(f"FAIL {name}: got {result_s}, expected {expected_s}")
        failed += 1
        return
    for (a, b), (c, d) in zip(result_s, expected_s):
        if abs(a - c) > tol or abs(b - d) > tol:
            print(f"FAIL {name}: got {result_s}, expected {expected_s}")
            failed += 1
            return
    passed += 1
    print(f"  ok {name}")


# ========================================================
print("=== _unwrap_interval tests ===")
# ========================================================
check("unwrap non-wrapping", unwrap((0.5, 1.5)), [(0.5, 1.5)])
check("unwrap non-wrapping negative", unwrap((-1.0, 0.5)), [(-1.0, 0.5)])
check("unwrap full range", unwrap((-pi, pi)), [(-pi, pi)])
check("unwrap wrapping", unwrap((2.5, -2.5)), [(2.5, pi), (-pi, -2.5)])
check("unwrap wrapping near pi", unwrap((3.0, -3.0)), [(3.0, pi), (-pi, -3.0)])
check("unwrap point", unwrap((1.0, 1.0)), [(1.0, 1.0)])

# ========================================================
print("\n=== _subtract_interval_pair tests (non-wrapping only) ===")
# ========================================================

# No overlap cases
check("no overlap: hole right", sub_pair((0.0, 1.0), (1.5, 2.0)), [(0.0, 1.0)])
check("no overlap: hole left", sub_pair((1.0, 2.0), (0.0, 0.5)), [(1.0, 2.0)])
check("no overlap: hole at right boundary", sub_pair((0.0, 1.0), (1.0, 2.0)), [(0.0, 1.0)])
check("no overlap: hole at left boundary", sub_pair((1.0, 2.0), (0.0, 1.0)), [(1.0, 2.0)])

# Full overlap
check("full overlap: hole contains interval", sub_pair((0.5, 1.5), (0.0, 2.0)), [])
check("full overlap: hole equals interval", sub_pair((0.5, 1.5), (0.5, 1.5)), [])
check("full overlap: hole slightly larger", sub_pair((0.5, 1.5), (0.4, 1.6)), [])

# Left bite
check("left bite", sub_pair((0.0, 2.0), (-0.5, 1.0)), [(1.0, 2.0)])
check("left bite exact start", sub_pair((0.0, 2.0), (0.0, 1.0)), [(1.0, 2.0)])

# Right bite
check("right bite", sub_pair((0.0, 2.0), (1.0, 2.5)), [(0.0, 1.0)])
check("right bite exact end", sub_pair((0.0, 2.0), (1.0, 2.0)), [(0.0, 1.0)])

# Middle bite (split)
check("middle bite", sub_pair((0.0, 3.0), (1.0, 2.0)), [(0.0, 1.0), (2.0, 3.0)])
check("middle bite small hole", sub_pair((-2.0, 2.0), (-0.1, 0.1)),
      [(-2.0, -0.1), (0.1, 2.0)])

# Tiny remainder (should be discarded by eps threshold)
check("left bite tiny remainder", sub_pair((0.0, 1.0), (-0.5, 1.0 - 1e-15)), [])
check("middle bite one side tiny", sub_pair((0.0, 1.0), (0.0 + 1e-15, 0.5)),
      [(0.5, 1.0)])

# ========================================================
print("\n=== TWO NON-OVERLAPPING TARGETS ===")
# ========================================================
check("no blocking: separated targets",
      sub([(0.2, 0.8)], (1.3, 1.7)), [(0.2, 0.8)])
check("no blocking: opposite sides",
      sub([(-2.0, -1.0)], (1.0, 2.0)), [(-2.0, -1.0)])

# ========================================================
print("\n=== PARTIAL BLOCKING (side) ===")
# ========================================================
check("partial block from right",
      sub([(0.0, 2.0)], (1.0, 3.0)), [(0.0, 1.0)])
check("partial block from left",
      sub([(0.0, 2.0)], (-1.0, 1.0)), [(1.0, 2.0)])

# ========================================================
print("\n=== SMALLER TARGET BLOCKS MIDDLE OF LARGER ===")
# ========================================================
check("middle block by smaller target",
      sub([(-1.5, 1.5)], (-0.3, 0.3)), [(-1.5, -0.3), (0.3, 1.5)])
check("asymmetric middle block",
      sub([(-2.0, 2.0)], (0.5, 1.0)), [(-2.0, 0.5), (1.0, 2.0)])

# ========================================================
print("\n=== FULL BLOCKING ===")
# ========================================================
check("full block: hole contains target",
      sub([(0.5, 1.0)], (0.0, 1.5)), [])
check("full block: hole equals target",
      sub([(0.5, 1.0)], (0.5, 1.0)), [])

# ========================================================
print("\n=== WRAPPING TARGET, NON-WRAPPING HOLE ===")
# ========================================================
# Target wraps: (2.5, -2.5) = [2.5, pi] U [-pi, -2.5]
check("wrapping target, hole in positive piece",
      sub([(2.5, -2.5)], (2.8, 3.0)),
      [(2.5, 2.8), (3.0, pi), (-pi, -2.5)])

check("wrapping target, hole in negative piece",
      sub([(2.5, -2.5)], (-pi, -2.8)),
      [(2.5, pi), (-2.8, -2.5)])

check("wrapping target, hole covers positive piece",
      sub([(2.5, -2.5)], (2.0, pi)),
      [(-pi, -2.5)])

check("wrapping target, hole covers negative piece",
      sub([(2.5, -2.5)], (-pi, -2.0)),
      [(2.5, pi)])

check("wrapping target, hole covers all",
      sub([(2.5, -2.5)], (2.0, -2.0)), [])

# ========================================================
print("\n=== NON-WRAPPING TARGET, WRAPPING HOLE ===")
# ========================================================
check("non-wrapping target, wrapping hole no overlap",
      sub([(0.5, 1.5)], (2.5, -2.5)), [(0.5, 1.5)])

# Target near -pi: [-3.0, -2.5]
# Hole wraps: (2.5, -2.8) = [2.5,pi]U[-pi,-2.8] -> overlaps [-3.0,-2.5] on left
check("non-wrapping target near -pi, wrapping hole partial",
      sub([(-3.0, -2.5)], (2.5, -2.8)), [(-2.8, -2.5)])

# Target near pi: [2.5, 3.0]
# Hole wraps: (2.8, -2.8) = [2.8,pi]U[-pi,-2.8] -> overlaps [2.5,3.0] on right
check("non-wrapping target near pi, wrapping hole partial",
      sub([(2.5, 3.0)], (2.8, -2.8)), [(2.5, 2.8)])

check("non-wrapping target, wrapping hole full cover",
      sub([(-3.0, -2.8)], (2.5, -2.5)), [])

# ========================================================
print("\n=== BOTH WRAPPING ===")
# ========================================================
# Target: (2.0, -2.0) = [2.0,pi]U[-pi,-2.0]
# Hole:   (2.5, -2.5) = [2.5,pi]U[-pi,-2.5]
# Result: [2.0,2.5] and [-2.5,-2.0]
check("both wrapping, partial overlap",
      sub([(2.0, -2.0)], (2.5, -2.5)), [(2.0, 2.5), (-2.5, -2.0)])

check("both wrapping, hole contains target",
      sub([(2.5, -2.5)], (2.0, -2.0)), [])

check("both wrapping, identical",
      sub([(2.5, -2.5)], (2.5, -2.5)), [])

# Target: (2.0, -2.0) = [2.0,pi]U[-pi,-2.0]
# Hole:   (2.5, -2.8) = [2.5,pi]U[-pi,-2.8]
# Positive: [2.0,pi] - [2.5,pi] = [2.0,2.5]
# Negative: [-pi,-2.0] - [-pi,-2.8] = [-2.8,-2.0]
check("both wrapping, asymmetric overlap",
      sub([(2.0, -2.0)], (2.5, -2.8)), [(2.0, 2.5), (-2.8, -2.0)])

# ========================================================
print("\n=== MULTIPLE INPUT INTERVALS ===")
# ========================================================
check("two intervals, hole hits one",
      sub([(0.0, 1.0), (2.0, 3.0)], (0.5, 1.5)),
      [(0.0, 0.5), (2.0, 3.0)])

check("two intervals, hole spans gap",
      sub([(0.0, 1.0), (1.5, 2.5)], (0.5, 2.0)),
      [(0.0, 0.5), (2.0, 2.5)])

# Chained subtraction: first split, then bite one piece
iv1 = sub([(0.0, 3.0)], (1.0, 2.0))  # -> [(0.0, 1.0), (2.0, 3.0)]
check("chained subtraction",
      sub(iv1, (0.5, 0.8)), [(0.0, 0.5), (0.8, 1.0), (2.0, 3.0)])

# ========================================================
print("\n=== EDGE CASES ===")
# ========================================================
check("nearly full circle, small hole",
      sub([(-pi, pi)], (-0.1, 0.1)), [(-pi, -0.1), (0.1, pi)])

check("full circle minus full circle",
      sub([(-pi, pi)], (-pi, pi)), [])

check("empty input", sub([], (0.0, 1.0)), [])

check("point-like hole (degenerate)",
      sub([(0.0, 2.0)], (1.0, 1.0)), [(0.0, 2.0)])

check("target at pi boundary",
      sub([(pi - 0.5, pi)], (pi - 0.3, pi)),
      [(pi - 0.5, pi - 0.3)])

check("target at -pi boundary",
      sub([(-pi, -pi + 0.5)], (-pi, -pi + 0.3)),
      [(-pi + 0.3, -pi + 0.5)])

# ========================================================
print("\n=== REALISTIC SCENARIO: two circle targets ===")
# ========================================================
# Simulate two circle targets as seen from an observer.
# Target A at egocentric angle 0.3, half-width 0.5 -> extent [-0.2, 0.8]
# Target B at egocentric angle 0.1, half-width 0.2 -> extent [-0.1, 0.3]
# B is closer and partially blocks A from [-0.1, 0.3]
check("realistic: B partially blocks A",
      sub([(-0.2, 0.8)], (-0.1, 0.3)),
      [(-0.2, -0.1), (0.3, 0.8)])

# Target A at 2.9 (near pi), half-width 0.4 -> extent [2.5, -2.8] (wraps!)
# Target B at -2.9 (near -pi), half-width 0.3 -> extent [-pi, -2.6] (at boundary)
# B is closer, blocks from [-pi, -2.6].
# A unwrapped is [2.5, pi] U [-pi, -2.8]
# After subtracting [-pi, -2.6]: positive piece untouched.
# Negative piece [-pi, -2.8] is fully contained in [-pi, -2.6] since -2.8 < -2.6.
check("realistic: wrapping target, boundary blocking (full neg block)",
      sub([(2.5, -2.8)], (-pi, -2.6)),
      [(2.5, pi)])

# Same but with larger negative piece that survives partial blocking
# Target wraps: (2.5, -2.0) = [2.5, pi] U [-pi, -2.0]
# Hole: [-pi, -2.6] blocks part of negative piece
# Negative piece [-pi, -2.0] minus [-pi, -2.6] = [-2.6, -2.0]
check("realistic: wrapping target, boundary blocking (partial neg block)",
      sub([(2.5, -2.0)], (-pi, -2.6)),
      [(2.5, pi), (-2.6, -2.0)])

# Complete blocking scenario: small far target behind a close big target
# Far target: extent [0.0, 0.3]
# Close target: extent [-0.5, 0.5]
check("realistic: complete blocking",
      sub([(0.0, 0.3)], (-0.5, 0.5)), [])

# Three targets chained blocking
# Far target: [-1.0, 1.0]
# Medium target blocks [-0.5, 0.0]
# Close target blocks [0.3, 0.8]
iv = sub([(-1.0, 1.0)], (-0.5, 0.0))   # -> [(-1.0, -0.5), (0.0, 1.0)]
iv = sub(iv, (0.3, 0.8))                 # -> [(-1.0, -0.5), (0.0, 0.3), (0.8, 1.0)]
check("realistic: three targets, two blockers",
      iv, [(-1.0, -0.5), (0.0, 0.3), (0.8, 1.0)])


# ========================================================
print("\n=== STRESS TESTS: near-boundary floating point ===")
# ========================================================

# Two wrapping intervals that barely overlap
# Target: (pi - 0.01, -pi + 0.01)  very thin wrap around pi
# Hole:   (pi - 0.005, -pi + 0.005) even thinner wrap, contained in target
# Target unwraps: [pi-0.01, pi] U [-pi, -pi+0.01]
# Hole unwraps:   [pi-0.005, pi] U [-pi, -pi+0.005]
# Positive: [pi-0.01, pi] - [pi-0.005, pi] = [pi-0.01, pi-0.005]
# Negative: [-pi, -pi+0.01] - [-pi, -pi+0.005] = [-pi+0.005, -pi+0.01]
check("thin wrapping intervals",
      sub([(pi - 0.01, -pi + 0.01)], (pi - 0.005, -pi + 0.005)),
      [(pi - 0.01, pi - 0.005), (-pi + 0.005, -pi + 0.01)])

# Hole that exactly matches one unwrapped piece of a wrapping target
# Target: (2.0, -2.0) = [2.0, pi] U [-pi, -2.0]
# Hole: (2.0, pi) — exactly the positive piece
check("hole matches one unwrapped piece exactly",
      sub([(2.0, -2.0)], (2.0, pi)), [(-pi, -2.0)])

# Multiple sequential blocking: simulate 4 targets blocking a large far target
# Far target: [-2.5, 2.5] (covers 5 radians)
# Blocker 1: [-2.5, -1.5] -> removes left chunk
# Blocker 2: [-0.5, 0.5] -> removes middle
# Blocker 3: [1.5, 2.5] -> removes right chunk
iv = sub([(-2.5, 2.5)], (-2.5, -1.5))
check("sequential blocking step 1", iv, [(-1.5, 2.5)])
iv = sub(iv, (-0.5, 0.5))
check("sequential blocking step 2", iv, [(-1.5, -0.5), (0.5, 2.5)])
iv = sub(iv, (1.5, 2.5))
check("sequential blocking step 3", iv, [(-1.5, -0.5), (0.5, 1.5)])

# Verify total remaining arc length
total_arc = sum(hi - lo for lo, hi in iv)
expected_arc = 1.0 + 1.0  # two intervals of width 1.0
assert abs(total_arc - expected_arc) < 1e-12, (
    f"Arc length mismatch: {total_arc} != {expected_arc}")
print(f"  ok sequential blocking arc length = {total_arc:.1f}")

# Degenerate: interval of width exactly eps (should be kept)
tiny = 2e-14
check("interval barely above eps", sub([(0.0, tiny)], (1.0, 2.0)), [(0.0, tiny)])

# Degenerate: interval of width below eps after subtraction (should be discarded)
# Interval [0.0, 1.0], hole [0.0, 1.0 - 5e-15] -> remainder of width 5e-15 < eps
check("remainder below eps discarded",
      sub([(0.0, 1.0)], (0.0, 1.0 - 5e-15)), [])

# ========================================================
print("\n=== _integrate_neural_weight tests ===")
# ========================================================

# We need a PerceptionModel instance. Minimal setup with no targets needed
# for the integration method itself, but the constructor requires targets.
# We'll create two instances: one with uniform weight, one with cutoff weight.
from decision_model import Targets

# Minimal targets (not used by _integrate_neural_weight, just needed for PM init)
tgts = Targets(locs=np.array([[10.0, 10.0]]))

# Uniform weight model
pm_uniform = PM(tgts, neural_weight=None)

# Cutoff weight model (uses default a=pi/3, b=4*pi/5)
pm_cutoff = PM(tgts, neural_weight='cutoff')


def check_scalar(name, result, expected, tol=1e-10):
    global passed, failed
    if abs(result - expected) > tol:
        print(f"FAIL {name}: got {result:.15g}, expected {expected:.15g}, "
              f"diff {abs(result - expected):.2e}")
        failed += 1
    else:
        passed += 1
        print(f"  ok {name}")


# --- Uniform weight: integral = arc length ---

check_scalar("uniform: single interval",
             pm_uniform._integrate_neural_weight([(0.0, 1.0)]), 1.0)

check_scalar("uniform: two intervals",
             pm_uniform._integrate_neural_weight([(0.0, 1.0), (2.0, 2.5)]), 1.5)

check_scalar("uniform: full circle",
             pm_uniform._integrate_neural_weight([(-pi, pi)]), 2*pi)

check_scalar("uniform: empty",
             pm_uniform._integrate_neural_weight([]), 0.0)

check_scalar("uniform: symmetric pair",
             pm_uniform._integrate_neural_weight([(-1.0, -0.5), (0.5, 1.0)]), 1.0)

check_scalar("uniform: tiny interval",
             pm_uniform._integrate_neural_weight([(0.0, 1e-8)]), 1e-8, tol=1e-20)

# --- Cutoff weight: F(hi) - F(lo) via antiderivative ---

a_val = pm_cutoff.a  # pi/3
b_val = pm_cutoff.b  # 4*pi/5

# Single interval in the flat region (|theta| < a => weight=1)
# F(x) = x * 2*pi/(a+b) in flat region, so integral = (hi-lo) * 2*pi/(a+b)
norm = 2*pi / (a_val + b_val)
check_scalar("cutoff: flat region [0, a/2]",
             pm_cutoff._integrate_neural_weight([(0.0, a_val/2)]),
             (a_val/2) * norm)

# Symmetric interval [-a/2, a/2]: both in flat region
F_pos = PM._smooth_cutoff_integral(a_val/2, a_val, b_val)
F_neg = PM._smooth_cutoff_integral(-a_val/2, a_val, b_val)
check_scalar("cutoff: symmetric flat [-a/2, a/2]",
             pm_cutoff._integrate_neural_weight([(-a_val/2, a_val/2)]),
             F_pos - F_neg)

# Interval spanning flat and transition region [0, (a+b)/2]
mid = (a_val + b_val) / 2
F_mid = PM._smooth_cutoff_integral(mid, a_val, b_val)
F_zero = PM._smooth_cutoff_integral(0.0, a_val, b_val)
check_scalar("cutoff: flat+transition [0, (a+b)/2]",
             pm_cutoff._integrate_neural_weight([(0.0, mid)]),
             F_mid - F_zero)

# Full support [-b, b]
F_b = PM._smooth_cutoff_integral(b_val, a_val, b_val)
F_nb = PM._smooth_cutoff_integral(-b_val, a_val, b_val)
check_scalar("cutoff: full support [-b, b]",
             pm_cutoff._integrate_neural_weight([(-b_val, b_val)]),
             F_b - F_nb)

# Two disjoint intervals: should equal sum of individual integrals
iv1_val = pm_cutoff._integrate_neural_weight([(0.1, 0.5)])
iv2_val = pm_cutoff._integrate_neural_weight([(1.0, 1.5)])
combined = pm_cutoff._integrate_neural_weight([(0.1, 0.5), (1.0, 1.5)])
check_scalar("cutoff: additivity of disjoint intervals",
             combined, iv1_val + iv2_val)

# Interval in zero region (|theta| > b => weight=0)
check_scalar("cutoff: zero region [b, pi]",
             pm_cutoff._integrate_neural_weight([(b_val, pi)]),
             PM._smooth_cutoff_integral(pi, a_val, b_val) -
             PM._smooth_cutoff_integral(b_val, a_val, b_val))

# Negative side interval
F_na = PM._smooth_cutoff_integral(-a_val, a_val, b_val)
F_nb2 = PM._smooth_cutoff_integral(-b_val, a_val, b_val)
check_scalar("cutoff: negative transition [-b, -a]",
             pm_cutoff._integrate_neural_weight([(-b_val, -a_val)]),
             F_na - F_nb2)

# --- Cross-check: cutoff integral vs high-resolution mesh sum ---
# _smooth_cutoff_integral includes a normalization factor 2*pi/(a+b), so
# F(hi) - F(lo) = norm * ∫_lo^hi cutoff(x) dx. The mesh sum gives the raw
# integral without norm, so we scale it up for comparison.
mesh_fine = np.linspace(-pi, pi, 100001)
dx = mesh_fine[1] - mesh_fine[0]

# Test interval [0.2, 1.5] - spans flat and transition regions
lo_test, hi_test = 0.2, 1.5
mask = (mesh_fine >= lo_test) & (mesh_fine <= hi_test)
mesh_integral = np.sum(pm_cutoff.get_neural_weight(mesh_fine[mask])) * dx * norm
analytic_integral = pm_cutoff._integrate_neural_weight([(lo_test, hi_test)])
check_scalar("cutoff vs mesh: [0.2, 1.5]",
             analytic_integral, mesh_integral, tol=1e-4)

# Test two disjoint intervals [-1.5, -0.3] and [0.5, 2.0]
mask1 = (mesh_fine >= -1.5) & (mesh_fine <= -0.3)
mask2 = (mesh_fine >= 0.5) & (mesh_fine <= 2.0)
mesh_integral2 = np.sum(pm_cutoff.get_neural_weight(mesh_fine[mask1 | mask2])) * dx * norm
analytic_integral2 = pm_cutoff._integrate_neural_weight([(-1.5, -0.3), (0.5, 2.0)])
check_scalar("cutoff vs mesh: two intervals",
             analytic_integral2, mesh_integral2, tol=1e-4)


# ========================================================
print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
# ========================================================
if failed > 0:
    exit(1)
