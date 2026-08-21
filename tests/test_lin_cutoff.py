"""Verify the analytic 'lin_cutoff' (trapezoidal) perception family.

lin_cutoff is the piecewise-linear analog of 'cutoff': density 1 on [-a, a],
a linear ramp to 0 on a < |theta| < b, 0 outside. Unlike 'cutoff' it has a
closed-form integral map and inverse (no spline), so this file checks the
analytic functions directly against a numeric quad reference and against the
saturation / normalization conventions it shares with 'cutoff', then exercises
it end-to-end as both a warp and a weight.

Run: python tests/test_lin_cutoff.py  (also importable by pytest).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import quad
from decision_model import PerceptionModel as PM, Targets
from decision_model import angle_distributions as ad

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


def F_ref(theta, a, b):
    """Reference forward map: norm * quad of the trapezoid density from 0.

    The density is zero beyond b, so clamp the upper limit to b (integrating
    the flat-zero tail just adds quad noise). Hand quad the corner at a as an
    explicit breakpoint so it resolves the kink to machine precision.
    """
    norm = 2 * pi / (a + b)
    ub = min(abs(theta), b)
    pts = [a] if 0 < a < ub else None
    val, _ = quad(ad.lin_cutoff, 0.0, ub, args=(a, b), points=pts)
    return np.sign(theta) * norm * val


rng = np.random.default_rng(seed=20260609)

# Parameter sets: defaults, a triangle (a=0), and a narrow plateau.
PARAMS = [(pi / 3, 4 * pi / 5), (0.0, 2.5), (0.5, 1.2), (0.2, pi)]

# ========================================================
print("=== Density _lin_cutoff: shape, plateau, support ===")
# ========================================================
for a, b in PARAMS:
    # Plateau == 1 inside [-a, a]; ramp value at midpoint; 0 outside [-b, b].
    xs = np.array([0.0, 0.5 * a, a, 0.5 * (a + b), b, 0.5 * (b + pi + 1e-9),
                   -a, -0.5 * (a + b), -b])
    g = ad.lin_cutoff(xs, a, b)
    expected = np.array([
        1.0, 1.0, 1.0, (b - 0.5 * (a + b)) / (b - a), 0.0, 0.0,
        1.0, (b - 0.5 * (a + b)) / (b - a), 0.0])
    check_array(f"density values a={a:.3g} b={b:.3g}", g, expected)
    # Even function.
    check_array(f"density even a={a:.3g} b={b:.3g}",
                ad.lin_cutoff(xs, a, b), ad.lin_cutoff(-xs, a, b))

# ========================================================
print("\n=== Forward map vs numeric quad reference ===")
# ========================================================
for a, b in PARAMS:
    theta_fixed = np.array([0.0, a, -a, b, -b, 0.5 * (a + b), -0.5 * (a + b),
                            pi, -pi, 0.99 * b, 1.0, -2.0])
    theta_rand = rng.uniform(-pi, pi, size=4000)
    theta = np.concatenate([theta_fixed, theta_rand])
    got = ad.lin_cutoff_integral(theta, a, b)
    ref = np.array([F_ref(t, a, b) for t in theta])
    check_array(f"forward vs quad a={a:.3g} b={b:.3g}", got, ref, tol=1e-11)

# ========================================================
print("\n=== Forward map conventions (endpoints, saturation, oddness) ===")
# ========================================================
for a, b in PARAMS:
    norm = 2 * pi / (a + b)
    check_scalar(f"F(0)=0 a={a:.3g}", ad.lin_cutoff_integral(0.0, a, b), 0.0)
    check_scalar(f"F(b)=pi a={a:.3g}", ad.lin_cutoff_integral(b, a, b), pi)
    check_scalar(f"F(-b)=-pi a={a:.3g}", ad.lin_cutoff_integral(-b, a, b), -pi)
    check_scalar(f"F(a)=norm*a a={a:.3g}",
                 ad.lin_cutoff_integral(a, a, b), norm * a)
    # Saturation beyond b (use b<pi sets only).
    if b < pi:
        check_scalar(f"F(pi)=pi (b<pi) a={a:.3g}",
                     ad.lin_cutoff_integral(pi, a, b), pi)
    # Oddness.
    ts = rng.uniform(-pi, pi, size=500)
    check_array(f"F odd a={a:.3g}", ad.lin_cutoff_integral(ts, a, b),
                -ad.lin_cutoff_integral(-ts, a, b))
    # Scalar vs vector consistency.
    check_scalar(f"F scalar==vector a={a:.3g}",
                 ad.lin_cutoff_integral(0.7, a, b),
                 float(ad.lin_cutoff_integral(np.array([0.7]), a, b)[0]))

# ========================================================
print("\n=== Inverse: round-trips to machine precision ===")
# ========================================================
for a, b in PARAMS:
    # F^{-1}(F(theta)) == theta on the invertible region [-b, b].
    theta = np.concatenate([
        np.array([0.0, a, -a, b, -b, 0.5 * (a + b), -0.5 * (a + b)]),
        rng.uniform(-b, b, size=4000)])
    y = ad.lin_cutoff_integral(theta, a, b)
    theta_rt = ad.lin_cutoff_int_inverse(y, a, b)
    # The sqrt inverse is condition-limited at the fold (theta -> b, disc -> 0),
    # so allow ~1e-10 there; F(Finv(y))=y below is exact (~1e-16) everywhere.
    check_array(f"Finv(F(theta))=theta on [-b,b] a={a:.3g} b={b:.3g}",
                theta_rt, theta, tol=1e-10)

    # F(F^{-1}(y)) == y on the full domain [-pi, pi].
    yvals = np.concatenate([
        np.array([0.0, pi, -pi]),
        rng.uniform(-pi, pi, size=4000)])
    y_rt = ad.lin_cutoff_integral(ad.lin_cutoff_int_inverse(yvals, a, b), a, b)
    check_array(f"F(Finv(y))=y on [-pi,pi] a={a:.3g} b={b:.3g}",
                y_rt, yvals, tol=1e-12)

    # Inverse endpoints.
    check_scalar(f"Finv(pi)=b a={a:.3g}",
                 ad.lin_cutoff_int_inverse(pi, a, b), b)
    check_scalar(f"Finv(-pi)=-b a={a:.3g}",
                 ad.lin_cutoff_int_inverse(-pi, a, b), -b)
    check_scalar(f"Finv(0)=0 a={a:.3g}",
                 ad.lin_cutoff_int_inverse(0.0, a, b), 0.0)

# ========================================================
print("\n=== Shared conventions with smooth 'cutoff' ===")
# ========================================================
a, b = pi / 3, 4 * pi / 5
# Agree exactly at 0, a, b (same plateau + normalization); differ on the ramp.
for t in [0.0, a, b]:
    check_scalar(f"lin vs smooth agree at t={t:.3g}",
                 ad.lin_cutoff_integral(t, a, b),
                 ad.smooth_cutoff_integral(t, a, b), tol=1e-9)
mid = 0.5 * (a + b)
ok(abs(ad.lin_cutoff_integral(mid, a, b)
       - ad.smooth_cutoff_integral(mid, a, b)) > 1e-3,
   "lin and smooth cutoff differ on the ramp interior")

# ========================================================
print("\n=== Validation and inverse domain ===")
# ========================================================
raises(lambda: ad.lin_cutoff_integral(0.5, 1.0, 1.0), "0 <= a < b",
       "forward rejects a == b")
raises(lambda: ad.lin_cutoff_integral(0.5, -0.1, 1.0), "0 <= a < b",
       "forward rejects a < 0")
raises(lambda: ad.lin_cutoff_int_inverse(0.5, 2.0, 1.0), "0 <= a < b",
       "inverse rejects a > b")
raises(lambda: ad.lin_cutoff_int_inverse(4.0, 0.5, 1.2), "-pi <= y <= pi",
       "inverse rejects y > pi")

# ========================================================
print("\n=== End-to-end as WARP (no spline built) ===")
# ========================================================
pm_warp = PM(neural_angle_dist='lin_cutoff', a_warp=0.5, b_warp=2.0)
ok(pm_warp._warp_forward_spline is None and pm_warp._warp_inverse_spline is None,
   "lin_cutoff warp builds no spline (analytic)")
ok(pm_warp.warp_params == {'a': 0.5, 'b': 2.0},
   "lin_cutoff warp_params view correct")
theta = rng.uniform(-2.0, 2.0, size=1000)  # inside [-b, b]
na = pm_warp.get_neural_angle(theta)
check_array("warp get_neural_angle round-trips",
            pm_warp.get_neural_angle_inverse(na), theta, tol=1e-12)
# Property-setter rebuild path (analytic -> just re-resolves params, no spline).
pm_warp.a_warp = 0.2
pm_warp.b_warp = 1.5
check_scalar("warp a_warp/b_warp setter takes effect",
             pm_warp.get_neural_angle(1.5), pi)

# ========================================================
print("\n=== End-to-end as WEIGHT and tied weight ===")
# ========================================================
TLOCS = np.array([[4.33, 2.5], [4.33, -2.5], [6.0, 0.0]])
tgts = Targets(locs=TLOCS, geom_name='circle', r=0.5)

pm_wt = PM(tgts, neural_angle_dist=None, angle_weight='lin_cutoff',
           a_weight=0.0, b_weight=pi)
for fl in [(3.0, 0.0), (3.0, 1.0), (2.0, -1.5)]:
    ang, rho = pm_wt.get_neural_signals(focal_angle=0.0, focal_loc=fl)
    if rho.size:
        ok(np.all(np.isfinite(rho)) and abs(rho.sum() - 1.0) < 1e-12,
           f"lin_cutoff weight rho finite & sums to 1 at {fl}")
    else:
        ok(True, f"lin_cutoff weight empty (no visible) at {fl}")

# Tied weight: warp == lin_cutoff, weight tied to it; rho must match an
# explicit lin_cutoff weight with the same params and identity-equivalent warp.
pm_tied = PM(tgts, neural_angle_dist='lin_cutoff', a_warp=0.0, b_warp=pi,
             angle_weight='neural_angle_dist')
pm_explicit = PM(tgts, neural_angle_dist=None, angle_weight='lin_cutoff',
                 a_weight=0.0, b_weight=pi)
same = True
for fl in [(3.0, 0.0), (3.0, 1.0), (2.0, -1.5)]:
    _, r0 = pm_tied._get_target_signals(focal_angle=0.0, focal_loc=fl)
    _, r1 = pm_explicit._get_target_signals(focal_angle=0.0, focal_loc=fl)
    same = same and np.allclose(r0, r1, atol=1e-12)
ok(same, "tied lin_cutoff weight matches explicit lin_cutoff weight")

# Registered for both roles and listed by the family-name error helper.
raises(lambda: PM(neural_angle_dist='lin_cutoff', a_warp=1.0, b_warp=0.5),
       "0 <= a < b", "constructor validates lin_cutoff warp params")

# ========================================================
print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
# ========================================================


def test_lin_cutoff():
    """Pytest entry point: the checks run at import; fail if any failed."""
    assert failed == 0, f"{failed} lin_cutoff checks failed"


if __name__ == '__main__' and failed > 0:
    exit(1)
