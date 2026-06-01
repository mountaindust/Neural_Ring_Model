"""Tests for the half-angle heading-torque law dtheta/dt = K*R*sin(ego/2).

Covers:
  - NBM torque shape: zero only at ego=0, monotone, +-K*R at ego->+-pi, and the
    intentional 2*K*R jump across the +-pi branch cut.
  - IEM torque shape + the convert_angles wrapping regression (sin(x/2) is
    4*pi-periodic, so the egocentric argument MUST be wrapped before halving).
  - Bifurcation invariance: doubling K exactly cancels the 1/2 from sin(ego/2)
    in the coupled 3x3 Jacobian at self-consistent equilibria, so eigenvalues
    (and hence stability / Hopf / SN structure) are unchanged vs the old
    K=1 * sin(ego) law.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decision_model as dm


def _blind_model():
    """Single target due north; cutoff WEIGHT with b_weight=pi/2 (blind cone).

    A walker facing away from the lone target sees nothing (blind spot).
    """
    targets = dm.Targets(locs=np.array([(0.0, 3.0)]), geom_name='circle', r=1.0)
    pm = dm.PerceptionModel(targets=targets, focal_loc=(0.0, 0.0),
                            focal_angle=0.0,
                            neural_angle_dist='cutoff', a_warp=0.0, b_warp=np.pi,
                            angle_weight='cutoff', a_weight=0.0,
                            b_weight=np.pi/2)
    return dm.NeuralBandModel(percep_model=pm, T=0.2, K=2)


def _identity_nbm(K=2):
    """NBM with identity warp so ego == np.angle(gamma)."""
    targets = dm.Targets(locs=np.array([(4.33, 2.5), (4.33, -2.5)]),
                         geom_name='circle', r=0.5)
    pm = dm.PerceptionModel(targets=targets, focal_loc=(0.0, 0.0),
                            focal_angle=0.0,
                            neural_angle_dist=None, angle_weight=None)
    return dm.NeuralBandModel(percep_model=pm, T=0.2, K=K)


def _identity_iem(K=2):
    targets = dm.Targets(locs=np.array([(4.33, 2.5), (4.33, -2.5)]),
                         geom_name='circle', r=0.5)
    pm = dm.PerceptionModel(targets=targets, focal_loc=(0.0, 0.0),
                            focal_angle=0.0,
                            neural_angle_dist=None, angle_weight=None)
    return dm.IsingExtModel(percep_model=pm, T=0.2, K=K)


# ---------------------------------------------------------------------------
# NBM torque shape
# ---------------------------------------------------------------------------
def test_nbm_default_K_is_2():
    assert dm.NeuralBandModel().K == 2
    assert dm.IsingExtModel().K == 2


def test_nbm_torque_shape():
    K, R = 2.0, 0.7
    nbm = _identity_nbm(K=K)
    # gamma passed explicitly -> no ODE solve; ego == angle(gamma) (identity warp)
    # zero only at ego = 0
    assert nbm.dtheta_dt(theta=0.0, gamma=R + 0j) == pytest.approx(0.0, abs=1e-12)
    # sign matches ego
    assert nbm.dtheta_dt(theta=0.0, gamma=R*np.exp(1j*0.5)) > 0
    assert nbm.dtheta_dt(theta=0.0, gamma=R*np.exp(-1j*0.5)) < 0
    # explicit value at ego = pi/2
    ego = np.pi/2
    assert nbm.dtheta_dt(theta=0.0, gamma=R*np.exp(1j*ego)) == \
        pytest.approx(K*R*np.sin(ego/2))
    # magnitude grows monotonically toward the facing-away point
    egos = np.linspace(0.05, np.pi - 0.05, 12)
    vals = [nbm.dtheta_dt(theta=0.0, gamma=R*np.exp(1j*e)) for e in egos]
    assert np.all(np.diff(vals) > 0)


def test_nbm_torque_max_at_facing_away_and_jump():
    K, R = 2.0, 0.8
    nbm = _identity_nbm(K=K)
    eps = 1e-7
    # just below +pi  -> +K*R ; just above -pi -> -K*R
    t_plus = nbm.dtheta_dt(theta=0.0, gamma=R*np.exp(1j*(np.pi - eps)))
    t_minus = nbm.dtheta_dt(theta=0.0, gamma=R*np.exp(1j*(-np.pi + eps)))
    assert t_plus == pytest.approx(K*R, abs=1e-4)
    assert t_minus == pytest.approx(-K*R, abs=1e-4)
    # the intentional jump discontinuity at the facing-away point
    assert (t_plus - t_minus) == pytest.approx(2*K*R, abs=1e-4)


# ---------------------------------------------------------------------------
# IEM torque shape + wrapping regression
# ---------------------------------------------------------------------------
def test_iem_torque_value_and_wrapping():
    K, R = 2.0, 0.6
    iem = _identity_iem(K=K)
    # pick phi, theta so the raw egocentric arg leaves (-pi, pi]
    phi, theta = 2.0, -2.0          # phi - theta = 4.0 > pi
    gamma = R*np.exp(1j*phi)
    val = iem.dtheta_dt(theta=theta, gamma=gamma)
    ego_wrapped = dm.convert_angles(phi - theta)       # 4.0 -> 4.0 - 2pi
    assert val == pytest.approx(K*R*np.sin(ego_wrapped/2))
    # regression: must NOT use the unwrapped argument
    assert val != pytest.approx(K*R*np.sin((phi - theta)/2))


def test_iem_torque_2pi_invariance():
    iem = _identity_iem(K=2.0)
    gamma = 0.6*np.exp(1j*1.1)
    base = iem.dtheta_dt(theta=0.3, gamma=gamma)
    # invariant under theta -> theta + 2pi (would fail without convert_angles)
    assert iem.dtheta_dt(theta=0.3 + 2*np.pi, gamma=gamma) == pytest.approx(base)


# ---------------------------------------------------------------------------
# Bifurcation invariance: K-doubling cancels the 1/2 at SC equilibria
# ---------------------------------------------------------------------------
def _coupled_jac(nbm, gamma_star, theta, half_angle, h=1e-6):
    """Build the 3x3 coupled Jacobian using either sin(ego) or sin(ego/2)."""
    gr0, gi0, th0 = gamma_star.real, gamma_star.imag, float(theta)

    def rhs(gr, gi, th):
        gamma = gr + 1j*gi
        dg = nbm.dgamma_dt(gamma=gamma, focal_angle=th,
                           focal_loc=nbm.percep_model.focal_loc)
        ego, R = nbm.convert_gamma(gamma)
        s = np.sin(ego/2) if half_angle else np.sin(ego)
        return np.array([dg.real, dg.imag, nbm.K * R * s])

    J = np.zeros((3, 3))
    for k, (dr, di, dt) in enumerate([(h, 0, 0), (0, h, 0), (0, 0, h)]):
        J[:, k] = (rhs(gr0+dr, gi0+di, th0+dt)
                   - rhs(gr0-dr, gi0-di, th0-dt)) / (2*h)
    return J


def test_jacobian_invariance_under_K_doubling():
    """At SC equilibria, (K=1, sin) and (K=2, sin/2) give identical eigenvalues."""
    targets = dm.Targets(locs=np.array([(4.33, 2.5), (4.33, -2.5)]),
                         geom_name='circle', r=0.5)
    pm_old = dm.PerceptionModel(targets=targets, focal_loc=(0.0, 0.0),
                                neural_angle_dist='cutoff', a_warp=0.0,
                                b_warp=np.pi, angle_weight='neural_angle_dist')
    pm_new = dm.PerceptionModel(targets=targets, focal_loc=(0.0, 0.0),
                                neural_angle_dist='cutoff', a_warp=0.0,
                                b_warp=np.pi, angle_weight='neural_angle_dist')
    nbm_old = dm.NeuralBandModel(percep_model=pm_old, T=0.2, K=1)
    nbm_new = dm.NeuralBandModel(percep_model=pm_new, T=0.2, K=2)

    checked = 0
    for fl in [(0.2, 0.0), (1.0, 0.5), (1.65, -0.8)]:
        nbm_old.percep_model.focal_loc = np.array(fl, dtype=float)
        nbm_new.percep_model.focal_loc = np.array(fl, dtype=float)
        angles, _ = nbm_old.sc_equilib(focal_loc=fl,
                                       stability_criterion='coupled')
        for th in angles:
            gamma_star = nbm_old.run_dgamma_dt(
                focal_angle=th, focal_loc=fl,
                init_gamma=0.5*np.exp(1j*0.0))
            # SC equilibrium has gamma = R + 0j with heading = consensus
            gamma_star = abs(gamma_star) + 0j
            J_old = _coupled_jac(nbm_old, gamma_star, th, half_angle=False)
            J_new = _coupled_jac(nbm_new, gamma_star, th, half_angle=True)
            e_old = np.sort_complex(np.linalg.eigvals(J_old))
            e_new = np.sort_complex(np.linalg.eigvals(J_new))
            assert np.allclose(e_old, e_new, atol=1e-6), \
                f"eigvals differ at fl={fl}, theta={th}: {e_old} vs {e_new}"
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# Blind-spot search (Fix 2)
# ---------------------------------------------------------------------------
def test_blind_spot_detection():
    nbm = _blind_model()
    pm = nbm.percep_model
    # facing the target -> visible
    pm.focal_angle = np.pi/2
    neur, _ = pm.get_neural_signals()
    assert neur.size == 1
    # facing directly away -> blind (zero total weight, empty signals)
    pm.focal_angle = -np.pi/2
    neur, _ = pm.get_neural_signals()
    assert neur.size == 0


def _single_track(nbm, blind_search_std, *, seed=0, max_steps=300):
    """Run one walker from a blind start and return its (x, y) trajectory."""
    nbm.rng = np.random.default_rng(seed)
    fig, ax = plt.subplots()
    nbm.plot_walkers(dt=0.1, v=1, std=0, repetitions=1, max_steps=max_steps,
                     start_loc=(0.0, 0.0), start_angle=-np.pi/2,
                     plot_tracks=True, ax=ax, blind_search_std=blind_search_std)
    line = ax.get_lines()[-1]
    xd, yd = line.get_xdata(), line.get_ydata()
    plt.close(fig)
    return np.asarray(xd), np.asarray(yd)


def test_blind_spot_frozen_drifts_straight():
    """With blind_search_std=0 a blind walker keeps its heading and walks off."""
    xd, yd = _single_track(_blind_model(), blind_search_std=0.0)
    # heading frozen pointing south: x stays put, y marches negative
    assert np.std(xd) < 1e-6
    assert yd[-1] < -10            # walked far away from the (0,3) target


def test_blind_spot_search_escapes():
    """With blind_search_std>0 the blind walker reorients, re-acquires the
    target, and is captured (instead of marching off to infinity)."""
    max_steps = 400
    xd, yd = _single_track(_blind_model(), blind_search_std=np.pi/2,
                           max_steps=max_steps)
    # reorientation happened (frozen drift keeps x identically 0)
    assert np.std(xd) > 1e-3
    # terminated early -> found the target (track shorter than max_steps)
    assert len(xd) - 1 < max_steps
    # ended at the target (surface at y = 3 - 1 = 2), not drifting away south
    assert yd[-1] > 1.0
