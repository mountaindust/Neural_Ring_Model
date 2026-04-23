"""Verify the precomputed integral splines in PerceptionModel.

Compares the spline-based neural-angle path against the reference
quad/brentq/scipy.stats path for both the 'cutoff' and 'vonmises'
weighting functions, and sanity-checks the end-to-end routing into
_integrate_neural_weight and the a/b/k property-setter rebuild path.
"""

import numpy as np
from scipy.stats import vonmises
from decision_model import PerceptionModel as PM, Targets

pi = np.pi

passed = 0
failed = 0


def check_scalar(name, result, expected, tol=1e-10):
    global passed, failed
    diff = abs(result - expected)
    if np.isnan(diff) or diff > tol:
        print(f"FAIL {name}: got {result!r}, expected {expected!r}, "
              f"diff {diff:.3e}")
        failed += 1
    else:
        passed += 1
        print(f"  ok {name}")


def check_array(name, result, expected, tol=1e-10):
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


rng = np.random.default_rng(seed=20260423)

# ========================================================
print("=== Cutoff forward spline vs reference ===")
# ========================================================

pm_cutoff = PM(neural_weight='cutoff', neural_angle='integral')
a = pm_cutoff.a
b = pm_cutoff.b

# Random samples in [-pi, pi]
theta_rand = rng.uniform(-pi, pi, size=5000)
# Fixed stress points
theta_fixed = np.array([
    0.0, a, -a, b, -b,
    0.99*a, -0.99*a, 0.99*b, -0.99*b,
    1.01*b, -1.01*b, pi, -pi,
    (a + b)/2, -(a + b)/2,
])
theta_all = np.concatenate([theta_rand, theta_fixed])

spline_vals = pm_cutoff._neural_angle_cutoff(theta_all)
ref_vals = np.array([PM._smooth_cutoff_integral(t, a, b) for t in theta_all])
check_array("cutoff forward: spline vs quad", spline_vals, ref_vals, tol=1e-10)

# ========================================================
print("\n=== Cutoff inverse spline vs reference ===")
# ========================================================

norm = 2*pi / (a + b)
# Sample y away from the ±pi endpoints. F(x) saturates to ±pi in floating
# point once |b - |x|| < ~0.05, so for y within machine-eps of ±pi, brentq
# can return any x in the saturated plateau — there is no unique inverse to
# compare against. Stay in the well-conditioned region.
y_rand = rng.uniform(-pi + 1e-3, pi - 1e-3, size=5000)
y_fixed = np.array([
    0.0, a*norm, -a*norm,
    pi - 1e-3, -(pi - 1e-3),
    0.5*pi, -0.5*pi,
])
y_all = np.concatenate([y_rand, y_fixed])

spline_inv = pm_cutoff._neural_angle_cutoff_inverse(y_all)
ref_inv = np.array([PM._smooth_cutoff_int_inverse(y, a, b) for y in y_all])
# Inverse accuracy is condition-limited: x-error = y-error / |dF/dx|, and
# dF/dx -> 0 as |x| -> b. Forward (F) is accurate to ~5e-11, so inverse at
# y near ±pi amplifies this to ~1e-8. Forward accuracy is what the
# _integrate_neural_weight hot path actually uses; inverse is only used for
# display mapping in get_neural_angle_inverse.
check_array("cutoff inverse: spline vs brentq", spline_inv, ref_inv, tol=1e-8)

# ========================================================
print("\n=== Cutoff roundtrip ===")
# ========================================================

# Stay inside the interior (well away from x=±b) — F(x) saturates to ±pi in
# floating point once b - |x| drops below ~0.05 (the essential singularity
# exp(-norm/(b-x)) underflows), so F^-1 of a saturated value collapses back
# to ±b and cannot recover the original x. Even short of saturation, dF/dx
# vanishes near the boundary so the inverse is ill-conditioned.
x_rt = rng.uniform(-0.85*b, 0.85*b, size=5000)
y_rt = pm_cutoff._neural_angle_cutoff(x_rt)
x_back = pm_cutoff._neural_angle_cutoff_inverse(y_rt)
check_array("cutoff roundtrip x -> y -> x", x_back, x_rt, tol=1e-8)

# y -> x -> y picks up two cubic-spline interpolation errors stacked on top
# of brentq's quad tolerance; 1e-9 is the realistic achievable precision.
y_dir = rng.uniform(-pi + 1e-3, pi - 1e-3, size=5000)
x_mid = pm_cutoff._neural_angle_cutoff_inverse(y_dir)
y_back = pm_cutoff._neural_angle_cutoff(x_mid)
check_array("cutoff roundtrip y -> x -> y", y_back, y_dir, tol=1e-9)

# ========================================================
print("\n=== Cutoff symmetry and endpoints ===")
# ========================================================

check_scalar("cutoff F(0) == 0", pm_cutoff._neural_angle_cutoff(0.0), 0.0,
             tol=0.0)
check_scalar("cutoff F(b) == pi", pm_cutoff._neural_angle_cutoff(b), pi,
             tol=0.0)
check_scalar("cutoff F(-b) == -pi", pm_cutoff._neural_angle_cutoff(-b), -pi,
             tol=0.0)
check_scalar("cutoff F(1.5*b) saturates to pi",
             pm_cutoff._neural_angle_cutoff(1.5*b), pi, tol=0.0)
check_scalar("cutoff F(-1.5*b) saturates to -pi",
             pm_cutoff._neural_angle_cutoff(-1.5*b), -pi, tol=0.0)
check_scalar("cutoff Finv(pi) == b",
             pm_cutoff._neural_angle_cutoff_inverse(pi), b, tol=0.0)
check_scalar("cutoff Finv(-pi) == -b",
             pm_cutoff._neural_angle_cutoff_inverse(-pi), -b, tol=0.0)

# Antisymmetry F(-x) = -F(x)
theta_sym = rng.uniform(-b, b, size=500)
f_pos = pm_cutoff._neural_angle_cutoff(theta_sym)
f_neg = pm_cutoff._neural_angle_cutoff(-theta_sym)
check_array("cutoff antisymmetry F(-x) = -F(x)", f_neg, -f_pos, tol=1e-12)

# ========================================================
print("\n=== Vonmises forward spline vs reference ===")
# ========================================================

pm_vm = PM(neural_weight='vonmises', neural_angle='integral')
k_val = pm_vm.k

theta_rand = rng.uniform(-pi, pi, size=5000)
# Stay within [-pi, pi]: outside that, the spline saturates to ±pi by design
# while scipy.stats.vonmises.cdf extends past 1.0, so the two intentionally
# differ. Saturation is checked separately below.
theta_fixed = np.array([
    0.0, pi, -pi, pi/2, -pi/2, pi/4, -pi/4,
    0.99*pi, -0.99*pi,
])
theta_all = np.concatenate([theta_rand, theta_fixed])

spline_vals = pm_vm._neural_angle_vonmises(theta_all)
ref_vals = PM._vonmises_integral(theta_all, k_val)
check_array("vonmises forward: spline vs scipy cdf", spline_vals, ref_vals,
            tol=1e-10)

# ========================================================
print("\n=== Vonmises inverse spline vs reference ===")
# ========================================================

y_rand = rng.uniform(-pi, pi, size=5000)
y_fixed = np.array([
    0.0, pi, -pi, 0.5*pi, -0.5*pi,
    pi - 1e-12, -pi + 1e-12,
])
y_all = np.concatenate([y_rand, y_fixed])

spline_inv = pm_vm._neural_angle_vonmises_inverse(y_all)
ref_inv = PM._vonmises_int_inverse(y_all, k_val)
check_array("vonmises inverse: spline vs scipy ppf", spline_inv, ref_inv,
            tol=1e-10)

# ========================================================
print("\n=== Vonmises roundtrip ===")
# ========================================================

theta_rt = rng.uniform(-pi, pi, size=5000)
y_rt = pm_vm._neural_angle_vonmises(theta_rt)
theta_back = pm_vm._neural_angle_vonmises_inverse(y_rt)
check_array("vonmises roundtrip theta -> y -> theta", theta_back, theta_rt,
            tol=1e-10)

y_dir = rng.uniform(-pi, pi, size=5000)
theta_mid = pm_vm._neural_angle_vonmises_inverse(y_dir)
y_back = pm_vm._neural_angle_vonmises(theta_mid)
check_array("vonmises roundtrip y -> theta -> y", y_back, y_dir, tol=1e-10)

# ========================================================
print("\n=== Vonmises symmetry and endpoints ===")
# ========================================================

check_scalar("vonmises G(0) == 0", pm_vm._neural_angle_vonmises(0.0), 0.0,
             tol=0.0)
check_scalar("vonmises G(pi) == pi", pm_vm._neural_angle_vonmises(pi), pi,
             tol=0.0)
check_scalar("vonmises G(-pi) == -pi", pm_vm._neural_angle_vonmises(-pi), -pi,
             tol=0.0)
check_scalar("vonmises G(1.5*pi) saturates to pi",
             pm_vm._neural_angle_vonmises(1.5*pi), pi, tol=0.0)

theta_sym = rng.uniform(-pi, pi, size=500)
g_pos = pm_vm._neural_angle_vonmises(theta_sym)
g_neg = pm_vm._neural_angle_vonmises(-theta_sym)
check_array("vonmises antisymmetry G(-x) = -G(x)", g_neg, -g_pos, tol=1e-12)

# ========================================================
print("\n=== _integrate_neural_weight invariance (cutoff) ===")
# ========================================================

# End-to-end check: the routing swap should preserve rho to within spline
# accuracy. Build a small Targets scenario with circles and compare rho
# against a temporarily monkey-patched reference path.
tgts = Targets(locs=np.array([[10.0, 8.0], [12.0, 12.0], [8.0, 13.0]]),
               geom_name='circle', r=0.5)
pm_e2e = PM(targets=tgts, focal_loc=(5.0, 10.0), focal_angle=0.1,
            neural_weight='cutoff', neural_angle='integral')

c_spline, rho_spline = pm_e2e._get_target_signals()

# Monkey-patch to force the reference path
a_e, b_e = pm_e2e.a, pm_e2e.b
orig_fn = pm_e2e._neural_angle_cutoff
pm_e2e._neural_angle_cutoff = lambda t: PM._smooth_cutoff_integral(
    float(t), a_e, b_e)
c_ref, rho_ref = pm_e2e._get_target_signals()
pm_e2e._neural_angle_cutoff = orig_fn

check_array("e2e cutoff: c_angles unchanged", c_spline, c_ref, tol=1e-14)
check_array("e2e cutoff: rho matches reference path", rho_spline, rho_ref,
            tol=1e-10)

# ========================================================
print("\n=== _integrate_neural_weight invariance (vonmises) ===")
# ========================================================

pm_e2e_vm = PM(targets=tgts, focal_loc=(5.0, 10.0), focal_angle=0.1,
               neural_weight='vonmises', neural_angle='integral')
c_spline_vm, rho_spline_vm = pm_e2e_vm._get_target_signals()

k_e = pm_e2e_vm.k
orig_fn = pm_e2e_vm._neural_angle_vonmises
pm_e2e_vm._neural_angle_vonmises = lambda t: PM._vonmises_integral(
    np.asarray(t, dtype=float), k_e)
c_ref_vm, rho_ref_vm = pm_e2e_vm._get_target_signals()
pm_e2e_vm._neural_angle_vonmises = orig_fn

check_array("e2e vonmises: c_angles unchanged", c_spline_vm, c_ref_vm,
            tol=1e-14)
check_array("e2e vonmises: rho matches reference path", rho_spline_vm,
            rho_ref_vm, tol=1e-10)

# ========================================================
print("\n=== Parameter sweep (cutoff) ===")
# ========================================================

for a_test, b_test in [(0.2, 1.0), (0.5, 1.5), (pi/4, 3*pi/4),
                       (0.01, pi - 0.1), (1.0, 2.8)]:
    pm_sweep = PM(neural_weight='cutoff', neural_angle='integral')
    pm_sweep.a = a_test
    pm_sweep.b = b_test
    theta_sweep = rng.uniform(-b_test, b_test, size=500)
    sp = pm_sweep._neural_angle_cutoff(theta_sweep)
    rf = np.array([PM._smooth_cutoff_integral(t, a_test, b_test)
                   for t in theta_sweep])
    check_array(f"cutoff sweep a={a_test:.3f}, b={b_test:.3f}", sp, rf,
                tol=1e-9)

# ========================================================
print("\n=== Parameter sweep (vonmises) ===")
# ========================================================

for k_test in [0.5, 1.0, 3.0, 5.0, 10.0]:
    pm_sweep = PM(neural_weight='vonmises', neural_angle='integral')
    pm_sweep.k = k_test
    theta_sweep = rng.uniform(-pi, pi, size=500)
    sp = pm_sweep._neural_angle_vonmises(theta_sweep)
    rf = PM._vonmises_integral(theta_sweep, k_test)
    check_array(f"vonmises sweep k={k_test}", sp, rf, tol=1e-10)

# ========================================================
print("\n=== Property-setter rebuild ===")
# ========================================================

# The a/b/k setters must trigger a spline rebuild; the accuracy checks after
# each setter verify this by comparing against the freshly parameterised
# reference. (id()-based identity checks are unreliable — CPython can reuse
# freed memory slots immediately for the new CubicSpline instance.)
pm_rb = PM(neural_weight='cutoff', neural_angle='integral')
pm_rb.a = 0.8
check_scalar(
    "cutoff a-setter accuracy",
    pm_rb._neural_angle_cutoff(0.5),
    PM._smooth_cutoff_integral(0.5, 0.8, pm_rb.b),
    tol=1e-10,
)

pm_rb.b = 2.5
check_scalar(
    "cutoff b-setter accuracy",
    pm_rb._neural_angle_cutoff(1.2),
    PM._smooth_cutoff_integral(1.2, pm_rb.a, 2.5),
    tol=1e-10,
)

pm_rb_vm = PM(neural_weight='vonmises', neural_angle='integral')
pm_rb_vm.k = 4.0
check_scalar(
    "vonmises k-setter accuracy",
    pm_rb_vm._neural_angle_vonmises(0.7),
    float(PM._vonmises_integral(0.7, 4.0)),
    tol=1e-10,
)

# Single build during __init__: patch the counter and confirm exactly one call.
build_count = {'n': 0}
orig_build = PM._build_integral_splines


def counting_build(self):
    build_count['n'] += 1
    return orig_build(self)


PM._build_integral_splines = counting_build
try:
    build_count['n'] = 0
    _ = PM(neural_weight='cutoff', neural_angle='integral')
    check_scalar("cutoff __init__ builds splines exactly once",
                 build_count['n'], 1, tol=0.0)

    build_count['n'] = 0
    _ = PM(neural_weight='vonmises', neural_angle='integral')
    check_scalar("vonmises __init__ builds splines exactly once",
                 build_count['n'], 1, tol=0.0)

    build_count['n'] = 0
    _ = PM(neural_weight=None, neural_angle=None)
    check_scalar("plain __init__ builds splines exactly once",
                 build_count['n'], 1, tol=0.0)
finally:
    PM._build_integral_splines = orig_build


# ========================================================
print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
# ========================================================
if failed > 0:
    exit(1)
