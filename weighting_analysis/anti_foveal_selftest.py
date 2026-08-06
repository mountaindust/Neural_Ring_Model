"""Numerics self-test for the retired anti-foveal families (see anti_foveal.py).

These were `tests/test_lin_dip.py` and `tests/test_lin_ring.py` while `lin_dip`
and `lin_ring` lived in `PerceptionModel`. The families were removed from the
model when the outward-bias result came back negative (outward_bias.md), so
their tests moved here with them: they are no longer part of `pytest tests/`,
which covers the shipped model only.

Run deliberately:  python weighting_analysis/anti_foveal_selftest.py

NOTE this file calls `anti_foveal.register()`, which patches `PerceptionModel`
for the whole process. That is why it is not named `test_*.py` -- it must not be
swept up by a `pytest` run that also exercises the real model.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import quad
from decision_model import PerceptionModel as PM, Targets

import anti_foveal
from anti_foveal import (lin_dip, lin_dip_integral, lin_dip_int_inverse,
                         lin_ring, lin_ring_integral, lin_ring_int_inverse)

anti_foveal.register()

pi = np.pi

passed = 0
failed = 0


def check_scalar(name, result, expected, tol=1e-12):
    global passed, failed
    diff = abs(result - expected)
    if np.isnan(diff) or diff > tol:
        print(f"FAIL {name}: got {result!r}, expected {expected!r}, "
              f"diff {diff:.3e}")
        failed += 1
    else:
        passed += 1
        print(f"  ok {name} (diff {diff:.2e})")


def check_array(name, result, expected, tol=1e-12):
    global passed, failed
    result = np.asarray(result)
    expected = np.asarray(expected)
    if result.shape != expected.shape:
        print(f"FAIL {name}: shape {result.shape} != {expected.shape}")
        failed += 1
        return
    diff = np.max(np.abs(result - expected))
    if np.isnan(diff) or diff > tol:
        argmax = np.argmax(np.abs(result - expected))
        print(f"FAIL {name}: max diff {diff:.3e} at flat index {argmax}, "
              f"got {result.flat[argmax]:.15g}, expected "
              f"{expected.flat[argmax]:.15g}")
        failed += 1
    else:
        passed += 1
        print(f"  ok {name} (max diff {diff:.2e})")


def ok(cond, name):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok {name}")
    else:
        failed += 1
        print(f"FAIL {name}")


def raises(fn, frag, name):
    global passed, failed
    try:
        fn()
    except (ValueError, NotImplementedError, TypeError) as e:
        if frag.lower() in str(e).lower():
            passed += 1
            print(f"  ok {name}")
        else:
            failed += 1
            print(f"FAIL {name}: wrong message {str(e)!r}")
    else:
        failed += 1
        print(f"FAIL {name}: no error raised")


# ======================================================================
# lin_dip
# ======================================================================

def F_ref_dip(theta, m, b):
    """Reference forward map: norm * quad of the dip density from 0.

    Hand quad the corner at b as an explicit breakpoint so it resolves the
    kink to machine precision.
    """
    norm = 2 * pi / (2 * pi - b * (1 - m))
    ub = min(abs(theta), pi)
    pts = [b] if 0 < b < ub else None
    val, _ = quad(lin_dip, 0.0, ub, args=(m, b), points=pts)
    return np.sign(theta) * norm * val


rng_dip = np.random.default_rng(seed=20260806)

# Parameter sets: the family defaults, a zero-floor dip, a deep narrow dip, a
# shallow near-uniform dip, and a full-circle ramp (b = pi, peak at the rear).
PARAMS_dip = [(0.25, pi / 2), (0.0, pi / 2), (0.05, 0.3), (0.9, 2.0), (0.5, pi)]

# ========================================================
print("=== Density _lin_dip: floor, ramp, peripheral plateau ===")
# ========================================================
for m, b in PARAMS_dip:
    xs = np.array([0.0, 0.5 * b, b, 0.5 * (b + pi), pi,
                   -0.5 * b, -b, -pi])
    g = lin_dip(xs, m, b)
    mid = m + (1 - m) * 0.5
    expected = np.array([m, mid, 1.0, 1.0, 1.0, mid, 1.0, 1.0])
    check_array(f"density values m={m:.3g} b={b:.3g}", g, expected)
    check_array(f"density even m={m:.3g} b={b:.3g}",
                lin_dip(xs, m, b), lin_dip(-xs, m, b))
    # Anti-foveal: strictly increasing in |theta| on the ramp, flat after.
    ts = np.linspace(0.0, pi, 501)
    d = lin_dip(ts, m, b)
    ok(np.all(np.diff(d) >= -1e-15) and d[0] == m and d[-1] == 1.0,
       f"density non-decreasing outward, floor m, peak 1 (m={m:.3g})")
    # Everywhere positive whenever m > 0 (m = 0 vanishes only at theta = 0).
    if m > 0:
        ok(np.all(lin_dip(np.linspace(-pi, pi, 1001), m, b) > 0),
           f"density strictly positive everywhere (m={m:.3g})")

# The whole point: uniform weight is the m -> 1 limit.
check_array("m -> 1 recovers uniform density",
            lin_dip(np.linspace(-pi, pi, 101), 1 - 1e-12, 1.0),
            np.ones(101), tol=1e-11)

# ========================================================
print("\n=== Forward map vs numeric quad reference ===")
# ========================================================
for m, b in PARAMS_dip:
    theta_fixed = np.array([0.0, b, -b, pi, -pi, 0.5 * b, -0.5 * b,
                            0.5 * (b + pi), 1.0, -2.0])
    theta_rand = rng_dip.uniform(-pi, pi, size=2000)
    theta = np.concatenate([theta_fixed, theta_rand])
    got = lin_dip_integral(theta, m, b)
    ref = np.array([F_ref_dip(t, m, b) for t in theta])
    check_array(f"forward vs quad m={m:.3g} b={b:.3g}", got, ref, tol=1e-11)

# ========================================================
print("\n=== Forward map conventions (endpoints, oddness, monotonicity) ===")
# ========================================================
for m, b in PARAMS_dip:
    norm = 2 * pi / (2 * pi - b * (1 - m))
    check_scalar(f"F(0)=0 m={m:.3g}", lin_dip_integral(0.0, m, b), 0.0)
    check_scalar(f"F(pi)=pi m={m:.3g}", lin_dip_integral(pi, m, b), pi)
    check_scalar(f"F(-pi)=-pi m={m:.3g}", lin_dip_integral(-pi, m, b), -pi)
    check_scalar(f"F(b)=norm*b*(1+m)/2 m={m:.3g}",
                 lin_dip_integral(b, m, b), norm * b * (1 + m) / 2)
    # Beyond the branch cut the map saturates (|theta| is clipped to pi).
    check_scalar(f"F saturates past pi m={m:.3g}",
                 lin_dip_integral(4.0, m, b), pi)
    ts = rng_dip.uniform(-pi, pi, size=500)
    check_array(f"F odd m={m:.3g}", lin_dip_integral(ts, m, b),
                -lin_dip_integral(-ts, m, b))
    grid = np.linspace(-pi, pi, 2001)
    ok(np.all(np.diff(lin_dip_integral(grid, m, b)) > 0),
       f"F strictly increasing m={m:.3g}")
    check_scalar(f"F scalar==vector m={m:.3g}",
                 lin_dip_integral(0.7, m, b),
                 float(lin_dip_integral(np.array([0.7]), m, b)[0]))

# The dip compresses front angles: |F(theta)| < |theta| inside the dip.
m, b = 0.25, pi / 2
ok(abs(lin_dip_integral(0.3, m, b)) < 0.3,
   "dip compresses frontal angles (|F(theta)| < |theta| inside the dip)")

# ========================================================
print("\n=== Inverse: round-trips to machine precision ===")
# ========================================================
for m, b in PARAMS_dip:
    theta = np.concatenate([
        np.array([0.0, b, -b, pi, -pi, 0.5 * b, -0.5 * b]),
        rng_dip.uniform(-pi, pi, size=4000)])
    theta_rt = lin_dip_int_inverse(lin_dip_integral(theta, m, b), m, b)
    check_array(f"Finv(F(theta))=theta m={m:.3g} b={b:.3g}",
                theta_rt, theta, tol=1e-12)

    yvals = np.concatenate([np.array([0.0, pi, -pi]),
                            rng_dip.uniform(-pi, pi, size=4000)])
    y_rt = lin_dip_integral(lin_dip_int_inverse(yvals, m, b), m, b)
    check_array(f"F(Finv(y))=y m={m:.3g} b={b:.3g}", y_rt, yvals, tol=1e-12)

    check_scalar(f"Finv(pi)=pi m={m:.3g}",
                 lin_dip_int_inverse(pi, m, b), pi)
    check_scalar(f"Finv(-pi)=-pi m={m:.3g}",
                 lin_dip_int_inverse(-pi, m, b), -pi)
    check_scalar(f"Finv(0)=0 m={m:.3g}",
                 lin_dip_int_inverse(0.0, m, b), 0.0)

# The cancellation-free ramp inverse must hold up in the near-uniform limit,
# where the textbook (-m + sqrt(D)) * b / (1-m) form loses all its digits.
for m in [0.99, 0.999, 0.999999]:
    ts = np.linspace(-pi, pi, 401)
    check_array(f"inverse stable as m -> 1 (m={m})",
                lin_dip_int_inverse(lin_dip_integral(ts, m, 1.0),
                                        m, 1.0), ts, tol=1e-12)

# ========================================================
print("\n=== Relationship to lin_cutoff and to uniform ===")
# ========================================================
# m -> 1 makes the integral map the identity (uniform density).
ts = np.linspace(-pi, pi, 401)
check_array("m -> 1 forward map is the identity",
            lin_dip_integral(ts, 1 - 1e-13, 2.0), ts, tol=1e-11)
# lin_dip and lin_cutoff are complementary in shape: where one has its plateau
# the other has its floor.
ok(lin_dip(0.0, 0.25, pi / 2) < lin_dip(pi, 0.25, pi / 2)
   and PM._lin_cutoff(0.0, 0.5, 2.0) > PM._lin_cutoff(1.9, 0.5, 2.0),
   "lin_dip peaks outward where lin_cutoff peaks forward")

# ========================================================
print("\n=== Validation and inverse domain ===")
# ========================================================
raises(lambda: lin_dip(0.5, 1.0, 1.0), "0 <= m < 1",
       "density rejects m == 1")
raises(lambda: lin_dip_integral(0.5, -0.1, 1.0), "0 <= m < 1",
       "forward rejects m < 0")
raises(lambda: lin_dip_integral(0.5, 0.5, 4.0), "0 < b <= pi",
       "forward rejects b > pi")
raises(lambda: lin_dip_integral(0.5, 0.5, 0.0), "0 < b <= pi",
       "forward rejects b == 0")
raises(lambda: lin_dip_int_inverse(4.0, 0.25, 1.0), "-pi <= y <= pi",
       "inverse rejects y > pi")

# ========================================================
print("\n=== End-to-end as WARP (no spline built) ===")
# ========================================================
pm_warp = PM(neural_angle_dist='lin_dip', a_warp=0.25, b_warp=pi / 2)
ok(pm_warp._warp_forward_spline is None and pm_warp._warp_inverse_spline is None,
   "lin_dip warp builds no spline (analytic)")
ok(pm_warp.warp_params == {'m': 0.25, 'b': pi / 2},
   "lin_dip warp_params view correct")
theta = rng_dip.uniform(-pi, pi, size=1000)
check_array("warp get_neural_angle round-trips",
            pm_warp.get_neural_angle_inverse(pm_warp.get_neural_angle(theta)),
            theta, tol=1e-12)
pm_warp.a_warp = 0.6
pm_warp.b_warp = 1.0
ok(pm_warp.warp_params == {'m': 0.6, 'b': 1.0},
   "warp a_warp/b_warp setters take effect")

# ========================================================
print("\n=== End-to-end as WEIGHT: the outward bias ===")
# ========================================================
# Locust three-target layout: distance 3, bearings {0, +-35 deg}, observer at
# the origin facing the centre target. This is the configuration the family was
# added for, so pin its defining behaviour.
sep = np.radians(35.0)
TLOCS = np.array([[3.0, 0.0],
                  [3.0 * np.cos(sep), 3.0 * np.sin(sep)],
                  [3.0 * np.cos(sep), -3.0 * np.sin(sep)]])
tgts = Targets(locs=TLOCS, geom_name='circle', r=0.1)

pm_unif = PM(tgts, (0.0, 0.0), 0.0, neural_angle_dist=None, angle_weight=None)
_, rho_unif = pm_unif.get_neural_signals(0.0, np.array([0.0, 0.0]))
check_scalar("uniform weight is symmetric 1/3 each (equal extents)",
             float(rho_unif[0]), 1.0 / 3.0, tol=1e-6)

centre_frac = []
for m in [0.9, 0.5, 0.25, 0.05]:
    pm_dip = PM(tgts, (0.0, 0.0), 0.0, neural_angle_dist=None,
                angle_weight='lin_dip', a_weight=m, b_weight=pi / 2)
    _, rho = pm_dip.get_neural_signals(0.0, np.array([0.0, 0.0]))
    ok(np.all(np.isfinite(rho)) and abs(rho.sum() - 1.0) < 1e-12,
       f"lin_dip rho finite & sums to 1 (m={m})")
    ok(abs(rho[1] - rho[2]) < 1e-12, f"lin_dip preserves y-symmetry (m={m})")
    centre_frac.append(float(rho[0]))
ok(all(c < 1.0 / 3.0 for c in centre_frac),
   "lin_dip weight puts LESS mass on the dead-ahead target than uniform")
ok(all(centre_frac[i] > centre_frac[i + 1] for i in range(len(centre_frac) - 1)),
   "centre mass decreases monotonically as the dip deepens (m down)")
print(f"     centre rho by m=[0.9,0.5,0.25,0.05]: "
      f"{np.round(centre_frac, 4).tolist()} (uniform = 0.3333)")

# The arc integral must agree with a direct numeric integral of the density.
pm_dip = PM(tgts, (0.0, 0.0), 0.0, neural_angle_dist=None,
            angle_weight='lin_dip', a_weight=0.25, b_weight=pi / 2)
norm = 2 * pi / (2 * pi - (pi / 2) * (1 - 0.25))
for lo, hi in [(-0.5, 0.5), (0.2, 2.0), (-pi, -1.0), (1.4, 1.8)]:
    ref = norm * quad(lin_dip, lo, hi, args=(0.25, pi / 2),
                      points=[-pi / 2, pi / 2])[0]
    check_scalar(f"_integrate_neural_weight on [{lo}, {hi}]",
                 pm_dip._integrate_neural_weight([(lo, hi)]), ref, tol=1e-11)

# Tied weight: warp == lin_dip and weight tied to it.
pm_tied = PM(tgts, neural_angle_dist='lin_dip', a_warp=0.25, b_warp=pi / 2,
             angle_weight='neural_angle_dist')
pm_explicit = PM(tgts, neural_angle_dist=None, angle_weight='lin_dip',
                 a_weight=0.25, b_weight=pi / 2)
same = True
for fl in [(0.0, 0.0), (1.0, 0.4), (2.0, -0.8)]:
    _, r0 = pm_tied._get_target_signals(focal_angle=0.0, focal_loc=fl)
    _, r1 = pm_explicit._get_target_signals(focal_angle=0.0, focal_loc=fl)
    same = same and np.allclose(r0, r1, atol=1e-12)
ok(same, "tied lin_dip weight matches explicit lin_dip weight")

# Delta targets take the pointwise-weight path rather than the arc integral.
tg_delta = Targets(locs=TLOCS, geom_name=None)
pm_delta = PM(tg_delta, (0.0, 0.0), 0.0, neural_angle_dist=None,
              angle_weight='lin_dip', a_weight=0.25, b_weight=pi / 2)
_, rho_d = pm_delta.get_neural_signals(0.0, np.array([0.0, 0.0]))
ok(rho_d.size == 3 and rho_d[0] < rho_d[1] and abs(rho_d.sum() - 1) < 1e-12,
   "delta targets also get the outward bias (pointwise weight path)")

# Constructor-level validation.
raises(lambda: PM(tgts, angle_weight='lin_dip', a_weight=1.0),
       "0 <= m < 1", "constructor validates lin_dip weight params")
raises(lambda: PM(neural_angle_dist='lin_dip', b_warp=5.0),
       "0 < b <= pi", "constructor validates lin_dip warp params")

# ========================================================
print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
# ========================================================


# ======================================================================
# lin_ring
# ======================================================================

def F_ref_ring(theta, m, p):
    """Reference forward map: norm * quad of the ring density from 0, with the
    peak handed to quad as an explicit breakpoint."""
    norm = 2 * pi / (pi + p * m)
    ub = min(abs(theta), pi)
    pts = [p] if 0 < p < ub else None
    val, _ = quad(lin_ring, 0.0, ub, args=(m, p), points=pts)
    return np.sign(theta) * norm * val


rng_ring = np.random.default_rng(seed=20260806)

# Family defaults, a zero-floor ring, a narrow forward ring, a shallow ring, and
# a rear-shifted peak.
PARAMS_ring = [(0.25, pi/2), (0.0, 1.0), (0.4, pi/4), (0.9, 2.0), (0.25, 2.6)]

# ========================================================
print("=== Density _lin_ring: floor, peak, rear falloff ===")
# ========================================================
for m, p in PARAMS_ring:
    xs = np.array([0.0, 0.5*p, p, 0.5*(p + pi), pi, -0.5*p, -p, -pi])
    inner_mid = m + (1 - m) * 0.5
    outer_mid = (pi - 0.5*(p + pi)) / (pi - p)
    expected = np.array([m, inner_mid, 1.0, outer_mid, 0.0,
                         inner_mid, 1.0, 0.0])
    check_array(f"density values m={m:.3g} p={p:.3g}",
                lin_ring(xs, m, p), expected)
    check_array(f"density even m={m:.3g} p={p:.3g}",
                lin_ring(xs, m, p), lin_ring(-xs, m, p))
    # Unimodal in |theta| with its maximum exactly at p. Put p on the grid --
    # it is a kink, so an off-grid p makes the sampled max fall short of 1.
    ts = np.unique(np.concatenate([np.linspace(0.0, pi, 1001), [p]]))
    d = lin_ring(ts, m, p)
    ok(abs(d.max() - 1.0) < 1e-12 and abs(ts[d.argmax()] - p) < 1e-12,
       f"peak value 1 at |theta| = p (m={m:.3g}, p={p:.3g})")
    up, down = ts <= p, ts >= p
    ok(np.all(np.diff(d[up]) >= -1e-15) and np.all(np.diff(d[down]) <= 1e-15),
       f"rises to the peak then falls monotonically (m={m:.3g})")
    # Anti-foveal AND rear-shedding: this is the defining shape property.
    ok(lin_ring(0.0, m, p) < 1.0 and lin_ring(pi, m, p) == 0.0,
       f"dip in the middle, zero at the rear cut (m={m:.3g})")
    # Positive on the open interval (only the branch cut itself vanishes).
    if m > 0:
        ok(np.all(lin_ring(np.linspace(-pi + 1e-9, pi - 1e-9, 1001),
                               m, p) > 0),
           f"strictly positive on (-pi, pi) (m={m:.3g})")

# p -> pi degenerates to the monotone lin_dip with b = pi -- on the open
# interval. At |theta| = pi itself the outer ramp is a vanishingly narrow cliff
# from 1 down to 0, so lin_ring keeps its rear zero however small (pi - p) is;
# exclude a neighbourhood of the branch cut.
ts = np.linspace(-pi + 1e-3, pi - 1e-3, 401)
check_array("p -> pi degenerates to lin_dip(b=pi) away from the cut",
            lin_ring(ts, 0.3, pi - 1e-9), lin_dip(ts, 0.3, pi),
            tol=1e-5)
ok(lin_ring(pi, 0.3, pi - 1e-9) == 0.0 and lin_dip(pi, 0.3, pi) == 1.0,
   "...but lin_ring still vanishes exactly at the cut, lin_dip does not")

# ========================================================
print("\n=== Forward map vs numeric quad reference ===")
# ========================================================
for m, p in PARAMS_ring:
    theta_fixed = np.array([0.0, p, -p, pi, -pi, 0.5*p, -0.5*p,
                            0.5*(p + pi), 1.0, -2.0])
    theta = np.concatenate([theta_fixed, rng_ring.uniform(-pi, pi, size=2000)])
    ref = np.array([F_ref_ring(t, m, p) for t in theta])
    check_array(f"forward vs quad m={m:.3g} p={p:.3g}",
                lin_ring_integral(theta, m, p), ref, tol=1e-11)

# ========================================================
print("\n=== Forward map conventions ===")
# ========================================================
for m, p in PARAMS_ring:
    norm = 2 * pi / (pi + p * m)
    check_scalar(f"F(0)=0 m={m:.3g}", lin_ring_integral(0.0, m, p), 0.0)
    check_scalar(f"F(pi)=pi m={m:.3g}", lin_ring_integral(pi, m, p), pi)
    check_scalar(f"F(-pi)=-pi m={m:.3g}",
                 lin_ring_integral(-pi, m, p), -pi)
    check_scalar(f"F(p)=norm*p*(1+m)/2 m={m:.3g}",
                 lin_ring_integral(p, m, p), norm * p * (1 + m) / 2)
    check_scalar(f"F saturates past pi m={m:.3g}",
                 lin_ring_integral(4.0, m, p), pi)
    ts = rng_ring.uniform(-pi, pi, size=500)
    check_array(f"F odd m={m:.3g}", lin_ring_integral(ts, m, p),
                -lin_ring_integral(-ts, m, p))
    grid = np.linspace(-pi, pi, 2001)
    ok(np.all(np.diff(lin_ring_integral(grid, m, p)) > 0),
       f"F strictly increasing m={m:.3g}")
    check_scalar(f"F scalar==vector m={m:.3g}",
                 lin_ring_integral(0.7, m, p),
                 float(lin_ring_integral(np.array([0.7]), m, p)[0]))

# The normalization is independent of p (the shape's mean is (1+m)/2 either
# way) -- a non-obvious identity worth pinning.
check_scalar("normalization independent of p",
             lin_ring_integral(pi, 0.3, 0.5),
             lin_ring_integral(pi, 0.3, 2.9))

# ========================================================
print("\n=== Inverse: round-trips ===")
# ========================================================
for m, p in PARAMS_ring:
    theta = np.concatenate([
        np.array([0.0, p, -p, pi, -pi, 0.5*p, -0.5*p]),
        rng_ring.uniform(-pi, pi, size=4000)])
    # The density vanishes at +-pi, so dF/dtheta -> 0 there and the sqrt
    # inverse is condition-limited at the fold, exactly as for 'cutoff'.
    check_array(f"Finv(F(theta))=theta m={m:.3g} p={p:.3g}",
                lin_ring_int_inverse(lin_ring_integral(theta, m, p),
                                         m, p), theta, tol=1e-7)
    yvals = np.concatenate([np.array([0.0, pi, -pi]),
                            rng_ring.uniform(-pi, pi, size=4000)])
    check_array(f"F(Finv(y))=y m={m:.3g} p={p:.3g}",
                lin_ring_integral(
                    lin_ring_int_inverse(yvals, m, p), m, p),
                yvals, tol=1e-12)
    check_scalar(f"Finv(pi)=pi m={m:.3g}",
                 lin_ring_int_inverse(pi, m, p), pi)
    check_scalar(f"Finv(-pi)=-pi m={m:.3g}",
                 lin_ring_int_inverse(-pi, m, p), -pi)
    check_scalar(f"Finv(0)=0 m={m:.3g}",
                 lin_ring_int_inverse(0.0, m, p), 0.0)

# The inner ramp shares lin_dip's cancellation-free form; check the m -> 1 limit.
for m in [0.99, 0.999, 0.999999]:
    ts = np.linspace(-1.0, 1.0, 401)      # stay on the inner ramp (p = 1.0)
    check_array(f"inner-ramp inverse stable as m -> 1 (m={m})",
                lin_ring_int_inverse(lin_ring_integral(ts, m, 1.0),
                                         m, 1.0), ts, tol=1e-10)

# ========================================================
print("\n=== Validation and inverse domain ===")
# ========================================================
raises(lambda: lin_ring(0.5, 1.0, 1.0), "0 <= m < 1",
       "density rejects m == 1")
raises(lambda: lin_ring_integral(0.5, -0.1, 1.0), "0 <= m < 1",
       "forward rejects m < 0")
raises(lambda: lin_ring_integral(0.5, 0.5, pi), "0 < p < pi",
       "forward rejects p == pi")
raises(lambda: lin_ring_integral(0.5, 0.5, 0.0), "0 < p < pi",
       "forward rejects p == 0")
raises(lambda: lin_ring_int_inverse(4.0, 0.25, 1.0), "-pi <= y <= pi",
       "inverse rejects y > pi")

# ========================================================
print("\n=== End-to-end as WARP (no spline built) ===")
# ========================================================
pm_warp = PM(neural_angle_dist='lin_ring', a_warp=0.25, b_warp=pi/2)
ok(pm_warp._warp_forward_spline is None and pm_warp._warp_inverse_spline is None,
   "lin_ring warp builds no spline (analytic)")
ok(pm_warp.warp_params == {'m': 0.25, 'p': pi/2},
   "lin_ring warp_params view correct")
theta = rng_ring.uniform(-3.0, 3.0, size=1000)
check_array("warp get_neural_angle round-trips",
            pm_warp.get_neural_angle_inverse(pm_warp.get_neural_angle(theta)),
            theta, tol=1e-8)
pm_warp.a_warp = 0.5
pm_warp.b_warp = 1.2
ok(pm_warp.warp_params == {'m': 0.5, 'p': 1.2},
   "warp a_warp/b_warp setters take effect")

# ========================================================
print("\n=== End-to-end as WEIGHT: dip AND rear falloff ===")
# ========================================================
sep = np.radians(35.0)
TLOCS = np.array([[3.0, 0.0],
                  [3.0*np.cos(sep), 3.0*np.sin(sep)],
                  [3.0*np.cos(sep), -3.0*np.sin(sep)]])
tgts = Targets(locs=TLOCS, geom_name='circle', r=0.1)

pm_ring = PM(tgts, (0.0, 0.0), 0.0, neural_angle_dist=None,
             angle_weight='lin_ring', a_weight=0.25, b_weight=pi/2)
_, rho = pm_ring.get_neural_signals(0.0, np.array([0.0, 0.0]))
ok(np.all(np.isfinite(rho)) and abs(rho.sum() - 1.0) < 1e-12,
   "lin_ring rho finite & sums to 1")
ok(rho[0] < 1.0/3.0, "lin_ring under-weights the dead-ahead target")
ok(abs(rho[1] - rho[2]) < 1e-12, "lin_ring preserves y-symmetry")

# The distinguishing property vs lin_dip: at an OUTER commitment the third
# target sits far to the rear, where lin_ring has shed weight but lin_dip has
# not. Observer at (2, 0) facing the upper target -- the configuration analysed
# in weighting_analysis/outward_bias.md.
obs = np.array([2.0, 0.0])
bear = np.arctan2(TLOCS[:, 1] - obs[1], TLOCS[:, 0] - obs[0])
pm_dip = PM(tgts, (0.0, 0.0), 0.0, neural_angle_dist=None,
            angle_weight='lin_dip', a_weight=0.25, b_weight=pi/2)
_, r_ring = pm_ring.get_neural_signals(bear[1], obs)
_, r_dip = pm_dip.get_neural_signals(bear[1], obs)
ok(r_ring[2] < r_dip[2],
   "lin_ring gives the rear target less mass than lin_dip does")
# ...and inside the dip the two families are literally identical (b = p).
_, rc_ring = pm_ring.get_neural_signals(0.0, obs)
_, rc_dip = pm_dip.get_neural_signals(0.0, obs)
check_array("identical to lin_dip when every target is inside |theta| <= b = p",
            rc_ring, rc_dip)

# Tied weight.
pm_tied = PM(tgts, neural_angle_dist='lin_ring', a_warp=0.25, b_warp=pi/2,
             angle_weight='neural_angle_dist')
pm_explicit = PM(tgts, neural_angle_dist=None, angle_weight='lin_ring',
                 a_weight=0.25, b_weight=pi/2)
same = all(np.allclose(
    pm_tied._get_target_signals(focal_angle=0.0, focal_loc=fl)[1],
    pm_explicit._get_target_signals(focal_angle=0.0, focal_loc=fl)[1],
    atol=1e-12) for fl in [(0.0, 0.0), (1.0, 0.4), (2.0, -0.8)])
ok(same, "tied lin_ring weight matches explicit lin_ring weight")

# Arc integral vs a direct numeric integral of the density.
norm = 2 * pi / (pi + (pi/2) * 0.25)
for lo, hi in [(-0.5, 0.5), (0.2, 2.0), (-pi, -1.0), (1.4, 2.9)]:
    ref = norm * quad(lin_ring, lo, hi, args=(0.25, pi/2),
                      points=[-pi/2, pi/2])[0]
    check_scalar(f"_integrate_neural_weight on [{lo}, {hi}]",
                 pm_ring._integrate_neural_weight([(lo, hi)]), ref, tol=1e-11)

raises(lambda: PM(tgts, angle_weight='lin_ring', a_weight=1.0),
       "0 <= m < 1", "constructor validates lin_ring weight params")
raises(lambda: PM(neural_angle_dist='lin_ring', b_warp=pi),
       "0 < p < pi", "constructor validates lin_ring warp params")

# ========================================================
print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
# ========================================================


print(f"\n=== TOTAL: {passed} passed, {failed} failed ===")

if __name__ == '__main__' and failed > 0:
    exit(1)
