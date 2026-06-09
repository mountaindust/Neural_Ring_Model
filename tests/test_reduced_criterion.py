"""Tests for the timescale-separated ('reduced') stability criterion.

The 'reduced' criterion declares a self-consistent equilibrium stable iff
  (1) the fast gamma block A = d(dgamma)/dgamma is Hurwitz, and
  (2) the slow Schur complement lam_slow = d - c A^{-1} b < 0,
where the coupled Jacobian is partitioned J = [[A, b], [c, d]] with
b = d(dgamma)/dtheta, c = d(dtheta)/dgamma, d = d(dtheta)/dtheta.

These tests verify the criterion is *correct*, not merely self-consistent:

 - test_schur_block_determinant_identity_*:
       det(J) == det(A) * (d - c A^{-1} b)  -- the block-determinant identity,
       validating the Schur partition against the full Jacobian.

 - test_schur_equals_slaved_slow_flow_nbm:
       the Schur complement equals the linearization of the genuinely slaved
       heading flow dtheta/dt = g(h(theta)), computed independently by
       root-finding the gamma-equilibrium branch h(theta) (continued from the
       self-consistent point) and finite-differencing in theta. This is the
       physical content of the criterion.

 - test_reduced_reproduces_documented_vonmises_disagreement:
       at the documented (1.5, 0) vonmises k=0.55 config, reduced=3 stable,
       coupled=3, discrim_a=5; the two extra discrim_a-stable equilibria are
       fast-stable but slow-unstable (positive Schur complement).

 - test_reduced_equals_coupled_on_standard_grid:
       in a non-Hopf regime (smooth cutoff a=0, b=pi, uniform weight) reduced
       and coupled agree everywhere.

 - test_fast_unstable_is_reduced_unstable / test_defaults_are_reduced.
"""
import sys, os, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import root
import warnings
warnings.filterwarnings('ignore')

from decision_model import (Targets, PerceptionModel, NeuralBandModel,
                            IsingExtModel)

TWO_CIRCLE = np.array([[4.33, 2.5], [4.33, -2.5]])


def _nbm_vonmises():
    """The documented Hopf-island setup: vonmises k=0.55, full weighting."""
    t = Targets(locs=TWO_CIRCLE, geom_name='circle', r=0.5)
    pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist='vonmises',
                         angle_weight='neural_angle_dist', a_warp=0.55)
    return NeuralBandModel(pm)


def _nbm_cutoff():
    """Standard non-Hopf setup: smooth cutoff a=0, b=pi, uniform weight."""
    t = Targets(locs=TWO_CIRCLE, geom_name='circle', r=0.5)
    pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist='cutoff',
                         a_warp=0.0, b_warp=np.pi, angle_weight=None)
    return NeuralBandModel(pm)


def _iem_plain():
    t = Targets(locs=TWO_CIRCLE, geom_name='circle', r=0.5)
    pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist=None, angle_weight=None)
    return IsingExtModel(pm)


def _blocks(J):
    return J[:2, :2], J[:2, 2], J[2, :2], J[2, 2]


def _schur(J):
    A, b, c, d = _blocks(J)
    return d - c @ np.linalg.solve(A, b)


def _nbm_sc_eqs(nbm, fl):
    """Self-consistent equilibria of NBM as (gamma*, theta*) with gamma* real.

    sc_equilib returns only the headings; we polish each on the 2D
    self-consistent system _self_consistent_eq([theta, R]) = 0 to recover the
    precise (theta*, R*) and hence gamma* = R* + 0j (the SC equilibrium has
    neural consensus angle 0, i.e. gamma real positive)."""
    angles, _ = nbm.sc_equilib(focal_loc=fl)
    eqs = []
    for th in angles:
        sol = root(nbm._self_consistent_eq, [th, 0.5], args=(fl,),
                   method='hybr', tol=1e-12)
        if not sol.success:
            continue
        th_s, R_s = sol.x
        if R_s < 0.05:
            continue
        eqs.append((R_s + 0j, float(th_s)))
    return eqs


def _iem_sc_eqs(iem, fl):
    """IEM self-consistent equilibria: gamma is already the equilibrium in
    allocentric coordinates, with focal_angle = angle(gamma)."""
    return [(g, float(np.angle(g))) for g in iem.sc_equilib(focal_loc=fl)]


def _branch_gamma(model, theta, focal_loc, warm):
    """High-precision gamma-equilibrium at fixed heading theta on the branch
    continued from `warm` (the slaved sheet h(theta)), via hybr."""
    sol = root(model.dgamma_dt_vec, [warm.real, warm.imag],
               args=(theta, focal_loc), method='hybr', tol=1e-12)
    assert sol.success, f"branch root find failed at theta={theta}"
    return sol.x[0] + 1j * sol.x[1]


# --------------------------------------------------------------------------
# 1. Block-determinant identity: det(J) = det(A) * Schur complement.
# --------------------------------------------------------------------------
def _check_block_det_identity(model, eqs_fn, focal_locs):
    max_rel, n = 0.0, 0
    for fl in focal_locs:
        fl = np.array(fl, float)
        for g, th in eqs_fn(model, fl):
            J = model._coupled_jacobian(g, th, fl)
            A, b, c, d = _blocks(J)
            lhs = np.linalg.det(J)
            rhs = np.linalg.det(A) * (d - c @ np.linalg.solve(A, b))
            rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
            max_rel = max(max_rel, rel)
            n += 1
    return max_rel, n


def test_schur_block_determinant_identity_nbm():
    nbm = _nbm_vonmises()
    locs = [(1.0, 0.0), (1.5, 0.0), (2.0, 0.0), (2.5, 0.0), (1.5, 0.5)]
    max_rel, n = _check_block_det_identity(nbm, _nbm_sc_eqs, locs)
    assert n >= 8, f"too few NBM equilibria sampled ({n})"
    assert max_rel < 1e-4, f"NBM det(J) != det(A)*Schur, max rel err {max_rel:.2e}"


def test_schur_block_determinant_identity_iem():
    iem = _iem_plain()
    locs = [(1.0, 0.0), (1.5, 0.0), (2.0, 0.0), (2.5, 0.0), (1.5, 0.5)]
    max_rel, n = _check_block_det_identity(iem, _iem_sc_eqs, locs)
    assert n >= 4, f"too few IEM equilibria sampled ({n})"
    assert max_rel < 1e-4, f"IEM det(J) != det(A)*Schur, max rel err {max_rel:.2e}"


# --------------------------------------------------------------------------
# 2. The Schur complement IS the slaved slow-flow eigenvalue.
# --------------------------------------------------------------------------
def test_schur_equals_slaved_slow_flow_nbm():
    nbm = _nbm_vonmises()
    dth = 1e-4
    checked = 0
    for fl in [(1.0, 0.0), (1.5, 0.0), (2.0, 0.0), (2.5, 0.0)]:
        fl = np.array(fl, float)
        for g_star, th in _nbm_sc_eqs(nbm, fl):
            J = nbm._coupled_jacobian(g_star, th, fl)
            A = J[:2, :2]
            # The slaved reduction is only defined on an attracting fast sheet.
            if not np.all(np.real(np.linalg.eigvals(A)) < 0):
                continue
            lam_schur = _schur(J)

            # Slaved slow flow g(h(theta)), continuing the branch from gamma*.
            def g_of(theta):
                ge = _branch_gamma(nbm, theta, fl, g_star)
                return nbm.K * abs(ge) * np.sin(np.angle(ge) / 2)

            lam_direct = (g_of(th + dth) - g_of(th - dth)) / (2 * dth)
            assert np.isclose(lam_schur, lam_direct, rtol=0.03, atol=3e-3), (
                f"Schur {lam_schur:.5f} != slaved slow-flow slope "
                f"{lam_direct:.5f} at focal_loc={fl}, theta={th:.4f}")
            checked += 1
    assert checked >= 6, f"too few stable equilibria checked ({checked})"


# --------------------------------------------------------------------------
# 3. Documented disagreement at (1.5, 0), vonmises k=0.55.
# --------------------------------------------------------------------------
def test_reduced_reproduces_documented_vonmises_disagreement():
    nbm = _nbm_vonmises()
    fl = np.array([1.5, 0.0])
    _, s_red = nbm.sc_equilib(focal_loc=fl, stability_criterion='reduced')
    _, s_cpl = nbm.sc_equilib(focal_loc=fl, stability_criterion='coupled')
    _, s_dsc = nbm.sc_equilib(focal_loc=fl, stability_criterion='discrim_a')
    assert sum(s_dsc) == 5, f"expected discrim_a=5 stable, got {sum(s_dsc)}"
    assert sum(s_red) == 3, f"expected reduced=3 stable, got {sum(s_red)}"
    assert sum(s_cpl) == 3, f"expected coupled=3 stable, got {sum(s_cpl)}"

    # The discrim_a-stable-but-reduced-unstable equilibria must be exactly the
    # two near theta = +-0.16: fast-stable (A Hurwitz) yet slow-unstable
    # (positive Schur complement).
    extra = []
    for g, th in _nbm_sc_eqs(nbm, fl):
        if bool(nbm._discrim_A(g, th, fl)) and not nbm._discrim_reduced(g, th, fl):
            J = nbm._coupled_jacobian(g, th, fl)
            assert np.all(np.real(np.linalg.eigvals(J[:2, :2])) < 0)  # fast-stable
            assert _schur(J) > 0                                      # slow-unstable
            extra.append(th)
    assert len(extra) == 2, f"expected 2 fast-stable/slow-unstable eqs, got {extra}"
    assert all(abs(abs(th) - 0.1606) < 0.02 for th in extra), extra


# --------------------------------------------------------------------------
# 4. In a non-Hopf regime reduced and coupled agree everywhere.
# --------------------------------------------------------------------------
def test_reduced_equals_coupled_on_standard_grid():
    nbm = _nbm_cutoff()
    disagreements = []
    for x in (1.0, 2.0, 3.0, 4.0):
        for y in (-1.0, 0.0, 1.0):
            fl = np.array([x, y])
            _, s_red = nbm.sc_equilib(focal_loc=fl, stability_criterion='reduced')
            _, s_cpl = nbm.sc_equilib(focal_loc=fl, stability_criterion='coupled')
            if sum(s_red) != sum(s_cpl):
                disagreements.append((x, y, sum(s_red), sum(s_cpl)))
    assert not disagreements, f"reduced != coupled at: {disagreements}"


# --------------------------------------------------------------------------
# 5. Fast-unstable equilibria are reduced-unstable regardless of slow mode.
# --------------------------------------------------------------------------
def test_fast_unstable_is_reduced_unstable():
    nbm = _nbm_vonmises()
    fl = np.array([2.0, 0.0])
    found_fast_unstable = False
    for g, th in _nbm_sc_eqs(nbm, fl):
        J = nbm._coupled_jacobian(g, th, fl)
        if np.any(np.real(np.linalg.eigvals(J[:2, :2])) > 0):
            found_fast_unstable = True
            assert not nbm._discrim_reduced(g, th, fl)
            assert not bool(nbm._discrim_A(g, th, fl))
    assert found_fast_unstable, "expected a fast-unstable equilibrium at (2,0)"


# --------------------------------------------------------------------------
# 5b. Near a gamma-fold (eig(A) -> 0) the slow test stays correct and well-
#     conditioned: A^{-1} blows up the Schur complement, but the det(J) sign
#     test the criterion uses stays bounded and agrees with sign(lam_slow).
# --------------------------------------------------------------------------
def test_reduced_robust_near_gamma_fold():
    nbm = _nbm_vonmises()
    # The fold arc in the upper island; these locations have a near-singular A.
    checked = 0
    for fl in [(2.05, 2.40), (2.25, 2.52), (2.35, 2.57), (1.88, 2.28)]:
        fl = np.array(fl, float)
        for g, th in _nbm_sc_eqs(nbm, fl):
            J = nbm._coupled_jacobian(g, th, fl)
            A = J[:2, :2]
            eA = np.linalg.eigvals(A)
            if not np.all(eA.real < 0):        # need A Hurwitz (gate passed)
                continue
            condA = np.linalg.cond(A)
            if condA < 100:                    # only care about near-fold cases
                continue
            lam_slow = _schur(J)               # blows up here
            detJ = np.linalg.det(J)            # stays bounded
            # det(J) is well-conditioned where the Schur complement is not:
            assert abs(lam_slow) > 10, (condA, lam_slow)
            assert abs(detJ) < 10, detJ
            # sign(det J) == sign(lam_slow) == the criterion's verdict:
            assert (detJ < 0) == (lam_slow < 0)
            assert nbm._discrim_reduced(g, th, fl) == bool(detJ < 0)
            checked += 1
    assert checked >= 1, "no near-fold A-Hurwitz equilibrium found to test"


# --------------------------------------------------------------------------
# 6. 'reduced' is the default everywhere it is a parameter.
# --------------------------------------------------------------------------
def test_defaults_are_reduced():
    targets = [
        (NeuralBandModel, 'sc_equilib'),
        (NeuralBandModel, 'gamma_equilib'),
        (NeuralBandModel, 'plot_bifurcation_diagram'),
        (NeuralBandModel, 'plot_direction_mesh'),
        (IsingExtModel, 'plot_bifurcation_diagram'),
        (IsingExtModel, 'plot_direction_mesh'),
    ]
    for cls, name in targets:
        sig = inspect.signature(getattr(cls, name))
        default = sig.parameters['stability_criterion'].default
        assert default == 'reduced', f"{cls.__name__}.{name} default={default!r}"


if __name__ == '__main__':
    test_schur_block_determinant_identity_nbm()
    test_schur_block_determinant_identity_iem()
    test_schur_equals_slaved_slow_flow_nbm()
    test_reduced_reproduces_documented_vonmises_disagreement()
    test_reduced_equals_coupled_on_standard_grid()
    test_fast_unstable_is_reduced_unstable()
    test_reduced_robust_near_gamma_fold()
    test_defaults_are_reduced()
    print("all reduced-criterion tests passed")
