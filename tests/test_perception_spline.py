"""Verify the precomputed integral splines in PerceptionModel.

Compares the spline-based neural-angle path against the reference
quad/brentq/scipy.stats path for both the 'cutoff' and 'vonmises'
weighting functions, and sanity-checks the end-to-end routing into
_integrate_neural_weight and the a/b/k property-setter rebuild path.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import vonmises
from scipy.stats import beta as beta_dist
from decision_model import PerceptionModel as PM, Targets
from decision_model import angle_distributions as ad

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

pm_cutoff = PM(neural_angle_dist='cutoff')
a = pm_cutoff.warp_params['a']
b = pm_cutoff.warp_params['b']

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

spline_vals = pm_cutoff.get_neural_angle(theta_all)
ref_vals = np.array([ad.smooth_cutoff_integral(t, a, b) for t in theta_all])
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

spline_inv = pm_cutoff.get_neural_angle_inverse(y_all)
ref_inv = np.array([ad.smooth_cutoff_int_inverse(y, a, b) for y in y_all])
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
y_rt = pm_cutoff.get_neural_angle(x_rt)
x_back = pm_cutoff.get_neural_angle_inverse(y_rt)
check_array("cutoff roundtrip x -> y -> x", x_back, x_rt, tol=1e-8)

# y -> x -> y picks up two cubic-spline interpolation errors stacked on top
# of brentq's quad tolerance; 1e-9 is the realistic achievable precision.
y_dir = rng.uniform(-pi + 1e-3, pi - 1e-3, size=5000)
x_mid = pm_cutoff.get_neural_angle_inverse(y_dir)
y_back = pm_cutoff.get_neural_angle(x_mid)
check_array("cutoff roundtrip y -> x -> y", y_back, y_dir, tol=1e-9)

# ========================================================
print("\n=== Cutoff symmetry and endpoints ===")
# ========================================================

check_scalar("cutoff F(0) == 0", pm_cutoff.get_neural_angle(0.0), 0.0,
             tol=0.0)
check_scalar("cutoff F(b) == pi", pm_cutoff.get_neural_angle(b), pi,
             tol=0.0)
check_scalar("cutoff F(-b) == -pi", pm_cutoff.get_neural_angle(-b), -pi,
             tol=0.0)
check_scalar("cutoff F(1.5*b) saturates to pi",
             pm_cutoff.get_neural_angle(1.5*b), pi, tol=0.0)
check_scalar("cutoff F(-1.5*b) saturates to -pi",
             pm_cutoff.get_neural_angle(-1.5*b), -pi, tol=0.0)
check_scalar("cutoff Finv(pi) == b",
             pm_cutoff.get_neural_angle_inverse(pi), b, tol=0.0)
check_scalar("cutoff Finv(-pi) == -b",
             pm_cutoff.get_neural_angle_inverse(-pi), -b, tol=0.0)

# Antisymmetry F(-x) = -F(x)
theta_sym = rng.uniform(-b, b, size=500)
f_pos = pm_cutoff.get_neural_angle(theta_sym)
f_neg = pm_cutoff.get_neural_angle(-theta_sym)
check_array("cutoff antisymmetry F(-x) = -F(x)", f_neg, -f_pos, tol=1e-12)

# ========================================================
print("\n=== Vonmises forward spline vs reference ===")
# ========================================================

pm_vm = PM(neural_angle_dist='vonmises')
k_val = pm_vm.warp_params['k']

theta_rand = rng.uniform(-pi, pi, size=5000)
# Stay within [-pi, pi]: outside that, the spline saturates to ±pi by design
# while scipy.stats.vonmises.cdf extends past 1.0, so the two intentionally
# differ. Saturation is checked separately below.
theta_fixed = np.array([
    0.0, pi, -pi, pi/2, -pi/2, pi/4, -pi/4,
    0.99*pi, -0.99*pi,
])
theta_all = np.concatenate([theta_rand, theta_fixed])

spline_vals = pm_vm.get_neural_angle(theta_all)
ref_vals = ad.vonmises_integral(theta_all, k_val)
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

spline_inv = pm_vm.get_neural_angle_inverse(y_all)
ref_inv = ad.vonmises_int_inverse(y_all, k_val)
check_array("vonmises inverse: spline vs scipy ppf", spline_inv, ref_inv,
            tol=1e-10)

# ========================================================
print("\n=== Vonmises roundtrip ===")
# ========================================================

theta_rt = rng.uniform(-pi, pi, size=5000)
y_rt = pm_vm.get_neural_angle(theta_rt)
theta_back = pm_vm.get_neural_angle_inverse(y_rt)
check_array("vonmises roundtrip theta -> y -> theta", theta_back, theta_rt,
            tol=1e-10)

y_dir = rng.uniform(-pi, pi, size=5000)
theta_mid = pm_vm.get_neural_angle_inverse(y_dir)
y_back = pm_vm.get_neural_angle(theta_mid)
check_array("vonmises roundtrip y -> theta -> y", y_back, y_dir, tol=1e-10)

# ========================================================
print("\n=== Vonmises symmetry and endpoints ===")
# ========================================================

check_scalar("vonmises G(0) == 0", pm_vm.get_neural_angle(0.0), 0.0,
             tol=0.0)
check_scalar("vonmises G(pi) == pi", pm_vm.get_neural_angle(pi), pi,
             tol=0.0)
check_scalar("vonmises G(-pi) == -pi", pm_vm.get_neural_angle(-pi), -pi,
             tol=0.0)
check_scalar("vonmises G(1.5*pi) saturates to pi",
             pm_vm.get_neural_angle(1.5*pi), pi, tol=0.0)

theta_sym = rng.uniform(-pi, pi, size=500)
g_pos = pm_vm.get_neural_angle(theta_sym)
g_neg = pm_vm.get_neural_angle(-theta_sym)
check_array("vonmises antisymmetry G(-x) = -G(x)", g_neg, -g_pos, tol=1e-12)

# ========================================================
print("\n=== Symmetric Beta roundtrip ===")
# ========================================================

# symmetric_beta uses scipy.stats.beta.cdf / .ppf directly (no spline cache),
# so the only error source is scipy's internal accuracy. Tight tolerance.
pm_sb = PM(neural_angle_dist='symmetric_beta')
alpha_val = pm_sb.warp_params['alpha']
b_sb = pm_sb.warp_params['b']

# Stay inside the well-conditioned interior. Near +/-b, scipy.beta.cdf
# saturates to 0.0/1.0 in floating point (for alpha = 5, this happens
# within ~1e-3 of the boundary), losing the information needed to roundtrip
# back exactly. This is intrinsic to the floating-point cdf, not a wrapper bug.
theta_rt = rng.uniform(-0.95*b_sb, 0.95*b_sb, size=5000)
y_rt = pm_sb.get_neural_angle(theta_rt)
theta_back = pm_sb.get_neural_angle_inverse(y_rt)
check_array("symmetric_beta roundtrip theta -> y -> theta", theta_back,
            theta_rt, tol=1e-11)

y_dir = rng.uniform(-pi + 1e-3, pi - 1e-3, size=5000)
theta_mid = pm_sb.get_neural_angle_inverse(y_dir)
y_back = pm_sb.get_neural_angle(theta_mid)
check_array("symmetric_beta roundtrip y -> theta -> y", y_back, y_dir,
            tol=1e-11)

# ========================================================
print("\n=== Symmetric Beta symmetry and endpoints ===")
# ========================================================

check_scalar("symmetric_beta G(0) == 0",
             pm_sb.get_neural_angle(0.0), 0.0, tol=0.0)
check_scalar("symmetric_beta G(b) == pi",
             pm_sb.get_neural_angle(b_sb), pi, tol=0.0)
check_scalar("symmetric_beta G(-b) == -pi",
             pm_sb.get_neural_angle(-b_sb), -pi, tol=0.0)
check_scalar("symmetric_beta G(1.5*b) saturates to pi",
             pm_sb.get_neural_angle(1.5*b_sb), pi, tol=0.0)
check_scalar("symmetric_beta G(-1.5*b) saturates to -pi",
             pm_sb.get_neural_angle(-1.5*b_sb), -pi, tol=0.0)
check_scalar("symmetric_beta Ginv(pi) == b",
             pm_sb.get_neural_angle_inverse(pi), b_sb, tol=0.0)
check_scalar("symmetric_beta Ginv(-pi) == -b",
             pm_sb.get_neural_angle_inverse(-pi), -b_sb, tol=0.0)

theta_sym = rng.uniform(-b_sb, b_sb, size=500)
g_pos = pm_sb.get_neural_angle(theta_sym)
g_neg = pm_sb.get_neural_angle(-theta_sym)
check_array("symmetric_beta antisymmetry G(-x) = -G(x)", g_neg, -g_pos,
            tol=1e-12)

# ========================================================
print("\n=== Symmetric Beta validation ===")
# ========================================================

# alpha < 1 must raise; alpha = 1 must NOT raise; b <= 0 must raise.
try:
    ad.symmetric_beta(0.0, alpha=0.5, b=pi)
except ValueError:
    passed += 1
    print("  ok symmetric_beta rejects alpha=0.5")
else:
    failed += 1
    print("FAIL symmetric_beta should reject alpha=0.5")

try:
    ad.symmetric_beta(0.0, alpha=1.0, b=pi)
except ValueError:
    failed += 1
    print("FAIL symmetric_beta should accept alpha=1.0")
else:
    passed += 1
    print("  ok symmetric_beta accepts alpha=1.0")

try:
    ad.symmetric_beta(0.0, alpha=2.0, b=0.0)
except ValueError:
    passed += 1
    print("  ok symmetric_beta rejects b=0.0")
else:
    failed += 1
    print("FAIL symmetric_beta should reject b=0.0")

# ========================================================
print("\n=== Reg_power forward spline vs reference ===")
# ========================================================

pm_rp = PM(neural_angle_dist='reg_power')
d_val = pm_rp.warp_params['d']
e_val = pm_rp.warp_params['e']

# Random samples plus stress points near 0 (where the integrand peaks).
theta_rand = rng.uniform(-pi, pi, size=5000)
theta_fixed = np.array([
    0.0, pi, -pi, 0.99*pi, -0.99*pi,
    1e-6, -1e-6, 1e-3, -1e-3, 0.1, -0.1,
    pi/4, -pi/4, pi/2, -pi/2,
])
theta_all = np.concatenate([theta_rand, theta_fixed])

spline_vals = pm_rp.get_neural_angle(theta_all)
ref_vals = ad.reg_power_integral(theta_all, d_val, e_val)
# 2001-node cubic-power-stretched mesh has its accuracy floor set by the
# u^3 concentration near 0; empirical max error on a random 5000-point
# grid is ~5e-7 across d in [0.3, 1.0] and e in [1e-3, 1e-1].
check_array("reg_power forward: spline vs quad", spline_vals, ref_vals,
            tol=1e-6)

# ========================================================
print("\n=== Reg_power inverse spline vs reference ===")
# ========================================================

# brentq reference is set to xtol=1e-8, so the inverse-spline tolerance
# matches the brentq accuracy floor. Stay away from +-pi (forward saturation).
y_rand = rng.uniform(-pi + 1e-3, pi - 1e-3, size=2000)
y_fixed = np.array([
    0.0, 0.5*pi, -0.5*pi,
    pi - 1e-3, -(pi - 1e-3),
])
y_all = np.concatenate([y_rand, y_fixed])

spline_inv = pm_rp.get_neural_angle_inverse(y_all)
ref_inv = ad.reg_power_int_inverse(y_all, d_val, e_val)
# Inverse spline error is dominated by the forward-spline interpolation
# error (~5e-7), amplified mildly by the inverse Jacobian away from 0.
check_array("reg_power inverse: spline vs brentq", spline_inv, ref_inv,
            tol=1e-5)

# ========================================================
print("\n=== Reg_power roundtrip ===")
# ========================================================

theta_rt = rng.uniform(-pi, pi, size=5000)
y_rt = pm_rp.get_neural_angle(theta_rt)
theta_back = pm_rp.get_neural_angle_inverse(y_rt)
# Two stacked cubic-spline interpolations: ~2x the forward floor.
check_array("reg_power roundtrip theta -> y -> theta", theta_back, theta_rt,
            tol=1e-5)

y_dir = rng.uniform(-pi + 1e-3, pi - 1e-3, size=5000)
theta_mid = pm_rp.get_neural_angle_inverse(y_dir)
y_back = pm_rp.get_neural_angle(theta_mid)
check_array("reg_power roundtrip y -> theta -> y", y_back, y_dir, tol=1e-5)

# ========================================================
print("\n=== Reg_power symmetry and endpoints ===")
# ========================================================

check_scalar("reg_power F(0) == 0",
             pm_rp.get_neural_angle(0.0), 0.0, tol=0.0)
check_scalar("reg_power F(pi) == pi",
             pm_rp.get_neural_angle(pi), pi, tol=0.0)
check_scalar("reg_power F(-pi) == -pi",
             pm_rp.get_neural_angle(-pi), -pi, tol=0.0)
check_scalar("reg_power F(1.5*pi) saturates to pi",
             pm_rp.get_neural_angle(1.5*pi), pi, tol=0.0)
check_scalar("reg_power F(-1.5*pi) saturates to -pi",
             pm_rp.get_neural_angle(-1.5*pi), -pi, tol=0.0)
check_scalar("reg_power Finv(pi) == pi",
             pm_rp.get_neural_angle_inverse(pi), pi, tol=0.0)
check_scalar("reg_power Finv(-pi) == -pi",
             pm_rp.get_neural_angle_inverse(-pi), -pi, tol=0.0)

theta_sym = rng.uniform(-pi, pi, size=500)
f_pos = pm_rp.get_neural_angle(theta_sym)
f_neg = pm_rp.get_neural_angle(-theta_sym)
check_array("reg_power antisymmetry F(-x) = -F(x)", f_neg, -f_pos, tol=1e-12)

# ========================================================
print("\n=== Reg_power validation ===")
# ========================================================

for bad_d in [-0.5, 0.0]:
    try:
        ad.reg_power(0.0, d=bad_d, e=1e-3)
    except ValueError:
        passed += 1
        print(f"  ok reg_power rejects d={bad_d}")
    else:
        failed += 1
        print(f"FAIL reg_power should reject d={bad_d}")

for bad_e in [-1e-3, 0.0]:
    try:
        ad.reg_power(0.0, d=0.5, e=bad_e)
    except ValueError:
        passed += 1
        print(f"  ok reg_power rejects e={bad_e}")
    else:
        failed += 1
        print(f"FAIL reg_power should reject e={bad_e}")

# ========================================================
print("\n=== Reg_power approximation of _power (regression pin) ===")
# ========================================================

# Pinned tolerance: at the default (d=0.5, e=1e-3), the normalized integral
# matches _power(theta, c=1-d=0.5) to ~8e-3 (see analyze_reg_power_e.py).
# This guards against accidental changes to the default e or normalization
# convention; bumping the bound here is fine if the default genuinely changes.
theta_grid = np.linspace(-pi, pi, 2001)
F_default = pm_rp.get_neural_angle(theta_grid)
P_target = ad.direct_power(theta_grid, 1.0 - d_val)
limit_err = float(np.max(np.abs(F_default - P_target)))
if limit_err < 1e-2:
    passed += 1
    print(f"  ok reg_power matches _power(c=1-d) to {limit_err:.3e} "
          f"at default d={d_val}, e={e_val}")
else:
    failed += 1
    print(f"FAIL reg_power deviates from _power(c=1-d) by {limit_err:.3e} "
          f"at default d={d_val}, e={e_val}")

# ========================================================
print("\n=== _integrate_neural_weight invariance (cutoff) ===")
# ========================================================

# End-to-end check: the weight-side arc-integral (rho) should match a
# reference path built from the quad/brentq/scipy antiderivatives. Build a
# small Targets scenario with circles. The weight is an INDEPENDENT family
# (angle_weight=<family>) so this exercises the decoupled weight spline; with
# an identity warp the warp role does not interfere.
tgts = Targets(locs=np.array([[10.0, 8.0], [12.0, 12.0], [8.0, 13.0]]),
               geom_name='circle', r=0.5)
pm_e2e = PM(targets=tgts, focal_loc=(5.0, 10.0), focal_angle=0.1,
            neural_angle_dist=None, angle_weight='cutoff')

c_spline, rho_spline = pm_e2e._get_target_signals()

# Monkey-patch the weight forward map to the quad reference antiderivative.
# The dispatcher is a module attribute, so this is global for the three
# lines it is in force; each block restores it immediately.
a_e = pm_e2e.weight_params['a']
b_e = pm_e2e.weight_params['b']
orig_eval = ad.eval_forward_map
ad.eval_forward_map = lambda name, params, fwd, t: \
    ad.smooth_cutoff_integral(float(t), a_e, b_e)
c_ref, rho_ref = pm_e2e._get_target_signals()
ad.eval_forward_map = orig_eval

check_array("e2e cutoff: c_angles unchanged", c_spline, c_ref, tol=1e-14)
check_array("e2e cutoff: rho matches reference path", rho_spline, rho_ref,
            tol=1e-10)

# ========================================================
print("\n=== _integrate_neural_weight invariance (vonmises) ===")
# ========================================================

pm_e2e_vm = PM(targets=tgts, focal_loc=(5.0, 10.0), focal_angle=0.1,
               neural_angle_dist=None, angle_weight='vonmises')
c_spline_vm, rho_spline_vm = pm_e2e_vm._get_target_signals()

k_e = pm_e2e_vm.weight_params['k']
orig_eval = ad.eval_forward_map
ad.eval_forward_map = lambda name, params, fwd, t: \
    ad.vonmises_integral(np.asarray(t, dtype=float), k_e)
c_ref_vm, rho_ref_vm = pm_e2e_vm._get_target_signals()
ad.eval_forward_map = orig_eval

check_array("e2e vonmises: c_angles unchanged", c_spline_vm, c_ref_vm,
            tol=1e-14)
check_array("e2e vonmises: rho matches reference path", rho_spline_vm,
            rho_ref_vm, tol=1e-10)

# ========================================================
print("\n=== _integrate_neural_weight invariance (reg_power) ===")
# ========================================================

pm_e2e_rp = PM(targets=tgts, focal_loc=(5.0, 10.0), focal_angle=0.1,
               neural_angle_dist=None, angle_weight='reg_power')
c_spline_rp, rho_spline_rp = pm_e2e_rp._get_target_signals()

d_e = pm_e2e_rp.weight_params['d']
e_e = pm_e2e_rp.weight_params['e']
orig_eval = ad.eval_forward_map
ad.eval_forward_map = lambda name, params, fwd, t: \
    ad.reg_power_integral(np.asarray(t, dtype=float), d_e, e_e)
c_ref_rp, rho_ref_rp = pm_e2e_rp._get_target_signals()
ad.eval_forward_map = orig_eval

check_array("e2e reg_power: c_angles unchanged", c_spline_rp, c_ref_rp,
            tol=1e-14)
check_array("e2e reg_power: rho matches reference path", rho_spline_rp,
            rho_ref_rp, tol=1e-9)

# ========================================================
print("\n=== _get_target_signals dispatch (symmetric_beta) ===")
# ========================================================

# symmetric_beta has no slow-vs-fast path split (no spline cache), so we
# only need to verify that the dispatch in _integrate_neural_weight reaches
# the symmetric_beta branch and produces a normalized rho of the right shape.
pm_e2e_sb = PM(targets=tgts, focal_loc=(5.0, 10.0), focal_angle=0.1,
               neural_angle_dist=None, angle_weight='symmetric_beta')
c_sb, rho_sb = pm_e2e_sb._get_target_signals()
check_scalar("e2e symmetric_beta: rho.sum() == 1", float(rho_sb.sum()), 1.0,
             tol=1e-12)
check_scalar("e2e symmetric_beta: rho length matches target count",
             len(rho_sb), tgts.locs.shape[0], tol=0.0)
check_scalar("e2e symmetric_beta: c_angles length matches target count",
             len(c_sb), tgts.locs.shape[0], tol=0.0)

# ========================================================
print("\n=== Parameter sweep (cutoff) ===")
# ========================================================

for a_test, b_test in [(0.2, 1.0), (0.5, 1.5), (pi/4, 3*pi/4),
                       (0.01, pi - 0.1), (1.0, 2.8)]:
    pm_sweep = PM(neural_angle_dist='cutoff')
    pm_sweep.a_warp = a_test
    pm_sweep.b_warp = b_test
    theta_sweep = rng.uniform(-b_test, b_test, size=500)
    sp = pm_sweep.get_neural_angle(theta_sweep)
    rf = np.array([ad.smooth_cutoff_integral(t, a_test, b_test)
                   for t in theta_sweep])
    check_array(f"cutoff sweep a={a_test:.3f}, b={b_test:.3f}", sp, rf,
                tol=1e-9)

# ========================================================
print("\n=== Parameter sweep (vonmises) ===")
# ========================================================

for k_test in [0.5, 1.0, 3.0, 5.0, 10.0]:
    pm_sweep = PM(neural_angle_dist='vonmises')
    pm_sweep.a_warp = k_test
    theta_sweep = rng.uniform(-pi, pi, size=500)
    sp = pm_sweep.get_neural_angle(theta_sweep)
    rf = ad.vonmises_integral(theta_sweep, k_test)
    check_array(f"vonmises sweep k={k_test}", sp, rf, tol=1e-10)

# ========================================================
print("\n=== Parameter sweep (symmetric_beta) ===")
# ========================================================

# Single regime: direct scipy evaluation is machine-precision for any alpha,
# including the formerly marginal regime 1 < alpha < 3 where a cubic-spline
# cache could not resolve the limited boundary smoothness.
for alpha_test, b_test in [(1.0, pi), (1.25, pi), (1.5, pi), (1.75, pi),
                           (2.0, pi), (3.0, pi), (5.0, pi), (10.0, pi),
                           (3.0, 0.8*pi), (5.0, 0.5*pi)]:
    pm_sweep = PM(neural_angle_dist='symmetric_beta')
    pm_sweep.a_warp = alpha_test
    pm_sweep.b_warp = b_test
    theta_sweep = rng.uniform(-b_test, b_test, size=500)
    sp = pm_sweep.get_neural_angle(theta_sweep)
    rf = ad.symmetric_beta_integral(theta_sweep, alpha_test, b_test)
    check_array(f"symmetric_beta sweep alpha={alpha_test:.3f}, b={b_test:.3f}",
                sp, rf, tol=1e-12)

# ========================================================
print("\n=== Parameter sweep (reg_power) ===")
# ========================================================

# Cover a range of d (front bias steepness) and e (regularization). At small
# e the spline build runs more quad calls per node, but the spline-vs-quad
# comparison is bounded by the cubic-power-stretched-mesh interpolation error.
for d_test, e_test in [(0.3, 1e-2), (0.5, 1e-3), (0.5, 1e-2),
                       (0.7, 1e-3), (0.7, 1e-2), (1.0, 1e-3),
                       (0.5, 1e-1)]:
    pm_sweep = PM(neural_angle_dist='reg_power')
    pm_sweep.a_warp = d_test
    pm_sweep.b_warp = e_test
    theta_sweep = rng.uniform(-pi, pi, size=500)
    sp = pm_sweep.get_neural_angle(theta_sweep)
    rf = ad.reg_power_integral(theta_sweep, d_test, e_test)
    check_array(f"reg_power sweep d={d_test:.3f}, e={e_test:.0e}",
                sp, rf, tol=1e-6)

# ========================================================
print("\n=== Property-setter rebuild ===")
# ========================================================

# assigning a_warp/b_warp must trigger a warp-spline rebuild; the accuracy checks
# after each call verify this by comparing against the freshly parameterised
# reference. (id()-based identity checks are unreliable — CPython can reuse
# freed memory slots immediately for the new CubicSpline instance.)
pm_rb = PM(neural_angle_dist='cutoff')
pm_rb.a_warp = 0.8
check_scalar(
    "cutoff a-setter accuracy",
    pm_rb.get_neural_angle(0.5),
    ad.smooth_cutoff_integral(0.5, 0.8, pm_rb.warp_params['b']),
    tol=1e-10,
)

pm_rb.b_warp = 2.5
check_scalar(
    "cutoff b-setter accuracy",
    pm_rb.get_neural_angle(1.2),
    ad.smooth_cutoff_integral(1.2, pm_rb.warp_params['a'], 2.5),
    tol=1e-10,
)

pm_rb_vm = PM(neural_angle_dist='vonmises')
pm_rb_vm.a_warp = 4.0
check_scalar(
    "vonmises k-setter accuracy",
    pm_rb_vm.get_neural_angle(0.7),
    float(ad.vonmises_integral(0.7, 4.0)),
    tol=1e-10,
)

pm_rb_sb = PM(neural_angle_dist='symmetric_beta')
pm_rb_sb.a_warp = 3.0
check_scalar(
    "symmetric_beta alpha-setter accuracy",
    pm_rb_sb.get_neural_angle(0.7),
    float(ad.symmetric_beta_integral(0.7, 3.0, pm_rb_sb.warp_params['b'])),
    tol=0.0,
)

pm_rb_sb.b_warp = 0.7*pi
check_scalar(
    "symmetric_beta b-setter accuracy",
    pm_rb_sb.get_neural_angle(0.4),
    float(ad.symmetric_beta_integral(0.4, pm_rb_sb.warp_params['alpha'],
                                      0.7*pi)),
    tol=0.0,
)

pm_rb_rp = PM(neural_angle_dist='reg_power')
pm_rb_rp.a_warp = 0.7
check_scalar(
    "reg_power d-setter accuracy",
    pm_rb_rp.get_neural_angle(0.7),
    float(ad.reg_power_integral(0.7, 0.7, pm_rb_rp.warp_params['e'])),
    tol=1e-10,
)

pm_rb_rp.b_warp = 1e-2
check_scalar(
    "reg_power e-setter accuracy",
    pm_rb_rp.get_neural_angle(0.4),
    float(ad.reg_power_integral(0.4, pm_rb_rp.warp_params['d'], 1e-2)),
    tol=1e-10,
)

# Single warp-spline build during __init__: patch the counter and confirm
# exactly one call. (angle_weight=None here, so no weight spline is built.)
build_count = {'n': 0}
orig_build = PM._build_warp_splines


def counting_build(self):
    build_count['n'] += 1
    return orig_build(self)


PM._build_warp_splines = counting_build
try:
    for fam in ['cutoff', 'vonmises', 'symmetric_beta', 'reg_power']:
        build_count['n'] = 0
        _ = PM(neural_angle_dist=fam)
        check_scalar(f"{fam} __init__ builds warp spline exactly once",
                     build_count['n'], 1, tol=0.0)

    build_count['n'] = 0
    _ = PM(neural_angle_dist=None, angle_weight=None)
    check_scalar("plain __init__ builds warp spline exactly once",
                 build_count['n'], 1, tol=0.0)
finally:
    PM._build_warp_splines = orig_build


# ========================================================
print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
# ========================================================


def test_perception_spline():
    """Pytest entry point: the checks run at import; fail if any failed."""
    assert failed == 0, f"{failed} perception-spline checks failed"


if __name__ == '__main__' and failed > 0:
    exit(1)
