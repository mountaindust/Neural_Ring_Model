"""New-capability tests for the decoupled warp/weight PerceptionModel API.

Covers behaviors that did not exist before the decouple: warp and weight from
different families, the same family with different per-role parameters, the
tied-vs-uniform weighting equivalences, respline isolation, and the error
paths. Numerical reproduction of the old configs was verified during the
decouple via a (since-removed) golden-master baseline; this file covers the new
degrees of freedom and cross-checks against analytic references.

Run: python tests/test_warp_weight_decouple.py  (also importable by pytest).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from decision_model import PerceptionModel as PM, Targets

pi = np.pi
TLOCS = np.array([[4.33, 2.5], [4.33, -2.5]])
R = 0.5

passed = 0
failed = 0


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
    except (ValueError, NotImplementedError, TypeError, AttributeError) as e:
        if frag.lower() in str(e).lower():
            passed += 1
            print(f"  ok {name}")
        else:
            failed += 1
            print(f"FAIL {name}: wrong message {str(e)!r}")
    else:
        failed += 1
        print(f"FAIL {name}: no error raised")


def tg(geom="circle"):
    return Targets(locs=TLOCS, geom_name=(None if geom == "delta" else "circle"),
                   r=R)


# ---------------------------------------------------------------------------
print("=== Warp and weight from different families ===")
# ---------------------------------------------------------------------------
# warp = vonmises(k=0.55); weight = cutoff(a=0, b=pi). The warp map must match
# a vonmises-only model; the rho must match a cutoff-weight model.
pm = PM(tg(), neural_angle_dist="vonmises", a_warp=0.55,
        angle_weight="cutoff", a_weight=0.0, b_weight=pi)

warp_ref = PM(tg(), neural_angle_dist="vonmises", a_warp=0.55, angle_weight=None)
th = np.linspace(-pi, pi, 257)
ok(np.allclose(pm.get_neural_angle(th), warp_ref.get_neural_angle(th), atol=1e-13),
   "mixed: warp map matches vonmises-only")

# rho must match a model whose WEIGHT is cutoff(0, pi) and whose warp does not
# matter for rho (rho only uses the weight arc-integral + visible geometry).
# Use identity warp + cutoff weight as the rho reference.
rho_ref = PM(tg(), neural_angle_dist=None, angle_weight="cutoff",
             a_weight=0.0, b_weight=pi)
same = True
for fl in [(1.2, 0.0), (3.0, 1.0), (5.0, 2.0), (2.0, 1.5)]:
    _, r0 = pm._get_target_signals(focal_angle=0, focal_loc=fl)
    _, r1 = rho_ref._get_target_signals(focal_angle=0, focal_loc=fl)
    same = same and r0.shape == r1.shape and np.allclose(r0, r1, atol=1e-13)
ok(same, "mixed: rho matches cutoff-weight-only (warp-independent)")


# ---------------------------------------------------------------------------
print("\n=== Same family, different params per role ===")
# ---------------------------------------------------------------------------
pm2 = PM(tg(), neural_angle_dist="vonmises", a_warp=0.55,
         angle_weight="vonmises", a_weight=0.9)
ok(pm2.warp_params == {"k": 0.55} and pm2.weight_params == {"k": 0.9},
   "same-family: independent k per role stored")
# warp uses k=0.55, weight uses k=0.9: weight density at pi/2 must equal a
# vonmises(0.9) pdf there, not vonmises(0.55).
w_ref_09 = PM._vonmises(pi / 2, 0.9)
ok(abs(pm2.get_neural_weight(pi / 2) - w_ref_09) < 1e-13,
   "same-family: weight pdf uses its own k=0.9")
warp_ref_055 = PM(tg(), neural_angle_dist="vonmises", a_warp=0.55,
                  angle_weight=None)
ok(np.allclose(pm2.get_neural_angle(th), warp_ref_055.get_neural_angle(th),
               atol=1e-13),
   "same-family: warp map uses its own k=0.55")


# ---------------------------------------------------------------------------
print("\n=== Tied weighting reproduces full weighting; None reproduces uniform ===")
# ---------------------------------------------------------------------------
# 'neural_angle_dist' weight ties weight to warp -> the weight density equals
# the warp family pdf with the warp params.
pm_tied = PM(tg(), neural_angle_dist="cutoff", a_warp=0.0, b_warp=pi,
             angle_weight="neural_angle_dist")
pm_wexplicit = PM(tg(), neural_angle_dist="cutoff", a_warp=0.0, b_warp=pi,
                  angle_weight="cutoff", a_weight=0.0, b_weight=pi)
same = True
for fl in [(1.2, 0.0), (3.0, 1.0), (5.0, 2.0)]:
    _, r0 = pm_tied._get_target_signals(focal_angle=0, focal_loc=fl)
    _, r1 = pm_wexplicit._get_target_signals(focal_angle=0, focal_loc=fl)
    same = same and np.allclose(r0, r1, atol=1e-13)
ok(same, "tied weight == explicit same-family weight")

# angle_weight=None -> uniform: rho depends only on visible arc length, so it
# matches a model with an entirely different warp but also uniform weight.
pm_uniform_a = PM(tg(), neural_angle_dist="cutoff", a_warp=0.0, b_warp=pi,
                  angle_weight=None)
pm_uniform_b = PM(tg(), neural_angle_dist="vonmises", a_warp=2.0,
                  angle_weight=None)
same = True
for fl in [(1.2, 0.0), (3.0, 1.0), (5.0, 2.0)]:
    _, r0 = pm_uniform_a._get_target_signals(focal_angle=0, focal_loc=fl)
    _, r1 = pm_uniform_b._get_target_signals(focal_angle=0, focal_loc=fl)
    same = same and np.allclose(r0, r1, atol=1e-13)
ok(same, "uniform weight is warp-independent (rho from arc length only)")


# ---------------------------------------------------------------------------
print("\n=== Default config == old weight_angle_only=True cutoff ===")
# ---------------------------------------------------------------------------
pm_default = PM(tg())
ok(pm_default.warp_name == "cutoff" and pm_default.weight_name is None,
   "default: cutoff warp + uniform weight")
ok(pm_default.warp_params == {"a": pi / 3, "b": 4 * pi / 5},
   "default: cutoff warp params are the documented defaults")


# ---------------------------------------------------------------------------
print("\n=== Respline isolation ===")
# ---------------------------------------------------------------------------
pm3 = PM(tg(), neural_angle_dist="vonmises", a_warp=0.55,
         angle_weight="cutoff", a_weight=0.0, b_weight=pi)
w_before = pm3.get_neural_angle(th).copy()
pm3.a_weight = pi / 4
ok(np.allclose(pm3.get_neural_angle(th), w_before, atol=0.0),
   "a_weight assignment leaves warp map byte-identical")

_, rho_before = pm3._get_target_signals(focal_angle=0, focal_loc=(3.0, 1.0))
rho_before = rho_before.copy()
pm3.a_warp = 1.5
_, rho_after = pm3._get_target_signals(focal_angle=0, focal_loc=(3.0, 1.0))
ok(np.allclose(rho_before, rho_after, atol=0.0),
   "a_warp assignment leaves independent-weight rho unchanged")

# tied: a_warp moves rho; a_weight errors.
pm4 = PM(tg(), neural_angle_dist="cutoff", angle_weight="neural_angle_dist")
_, rb0 = pm4._get_target_signals(focal_angle=0, focal_loc=(3.0, 1.0))
rb0 = rb0.copy()
pm4.a_warp = 0.0
pm4.b_warp = pi
_, rb1 = pm4._get_target_signals(focal_angle=0, focal_loc=(3.0, 1.0))
ok(not np.allclose(rb0, rb1), "tied: a_warp assignment moves rho too")


# ---------------------------------------------------------------------------
print("\n=== Error paths ===")
# ---------------------------------------------------------------------------
raises(lambda: PM(tg(), angle_weight="direct_power"),
       "not allowed", "direct_power disallowed as weight")
raises(lambda: PM(tg(), neural_angle_dist="cutoff",
                  angle_weight="neural_angle_dist", a_weight=0.1),
       "a_weight/b_weight must be None", "a_weight with tied weight rejected")
raises(lambda: PM(tg(), neural_angle_dist=None,
                  angle_weight="neural_angle_dist"),
       "requires neural_angle_dist", "tied weight needs density warp")
raises(lambda: PM(tg(), angle_weight=None, a_weight=0.1),
       "a_weight/b_weight must be None", "a_weight with uniform weight rejected")
raises(lambda: PM(tg(), neural_angle_dist="vonmises", a_warp=-1),
       "k) must be > 0", "vonmises k>0 enforced (named param)")
raises(lambda: PM(tg(), neural_angle_dist="cutoff", a_warp=2.0, b_warp=1.0),
       "0 <= a < b", "cutoff 0<=a<b enforced")
raises(lambda: PM(tg(), neural_angle_dist="symmetric_beta", a_warp=0.5),
       "alpha) must be >= 1", "beta alpha>=1 enforced (named param)")
raises(lambda: PM(tg(), neural_angle_dist="reg_power", b_warp=-1.0),
       "e) must be > 0", "reg_power e>0 enforced (named param)")
raises(lambda: PM(tg(), neural_angle_dist="vonmises", b_warp=3.0),
       "b_warp is not used", "unused slot rejected")
raises(lambda: PM(tg(), neural_angle_dist="bogus"),
       "must be one of", "unknown warp family rejected")
raises(lambda: PM(tg(), angle_weight="bogus"),
       "must be one of", "unknown weight family rejected")
# Post-init parameter mutation now goes through the a_warp/b_warp/a_weight/
# b_weight properties; their strict setters reject invalid targets. (setattr
# wrappers because a lambda body can't contain an assignment statement.)
raises(lambda: setattr(PM(tg(), neural_angle_dist=None), "a_warp", 1),
       "identity warp", "a_warp set errors on identity warp")
raises(lambda: setattr(PM(tg(), angle_weight=None), "a_weight", 1),
       "uniform weight", "a_weight set errors on uniform weight")
raises(lambda: setattr(PM(tg(), neural_angle_dist="vonmises",
                          angle_weight="neural_angle_dist"), "a_weight", 1),
       "tied to the warp", "a_weight set errors when tied")
raises(lambda: setattr(PM(tg(), neural_angle_dist="vonmises"), "b_warp", 3.0),
       "not used", "b_warp set rejects unused slot")
raises(lambda: setattr(PM(tg(), neural_angle_dist="cutoff"), "a_warp", 5.0),
       "0 <= a < b", "a_warp set re-validates (a < b)")

# Read-only parameter views: both write paths blocked with a helpful message.
raises(lambda: PM(tg(), neural_angle_dist="vonmises").warp_params.__setitem__("k", 9),
       "read-only", "warp_params item assignment blocked")
raises(lambda: setattr(PM(tg(), neural_angle_dist="vonmises"), "warp_params", {"k": 9}),
       "read-only", "warp_params rebinding blocked")
ok(PM(tg(), neural_angle_dist="vonmises", a_warp=0.55).warp_params == {"k": 0.55},
   "warp_params read-only view compares equal to a plain dict")


def test_decouple():
    assert failed == 0, f"{failed} decouple-capability checks failed"


print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
if __name__ == "__main__" and failed:
    sys.exit(1)
