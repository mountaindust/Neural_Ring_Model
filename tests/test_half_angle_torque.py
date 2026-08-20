"""Tests for the half-angle heading-torque law
dtheta/dt = K*R*sin(arg(gamma)/2) in the neural consensus angle.

Covers:
  - Torque shape: zero only at arg(gamma)=0, monotone, +-K*R at
    arg(gamma)->+-pi, and the intentional 2*K*R jump across the +-pi branch cut.
    (The shape tests use the identity warp, where ego == arg(gamma).)
  - convert_angles is odd at the +-pi branch cut, so the fork direction is
    inherited from the sign of its argument.
  - Bifurcation invariance: doubling K exactly cancels the 1/2 from
    sin(arg(gamma)/2) in the coupled 3x3 Jacobian at self-consistent equilibria,
    so eigenvalues (and hence stability / Hopf / SN structure) are unchanged vs
    a K=1 * sin(arg(gamma)) full-angle law.
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
    # One target, so beta = 1/0.2 = 5 matches the old T=0.2 parameterization
    # (whose effective coupling was N_targets/T).
    return dm.NeuralBandModel(percep_model=pm, beta=5.0, K=2)


def _identity_nbm(K=2):
    """NBM with identity warp so ego == np.angle(gamma)."""
    targets = dm.Targets(locs=np.array([(4.33, 2.5), (4.33, -2.5)]),
                         geom_name='circle', r=0.5)
    pm = dm.PerceptionModel(targets=targets, focal_loc=(0.0, 0.0),
                            focal_angle=0.0,
                            neural_angle_dist=None, angle_weight=None)
    return dm.NeuralBandModel(percep_model=pm, beta=10.0, K=K)



# ---------------------------------------------------------------------------
# NBM torque shape
# ---------------------------------------------------------------------------
def test_nbm_default_K_is_2():
    assert dm.NeuralBandModel().K == 2


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
# convert_angles is odd at the +-pi branch cut
# ---------------------------------------------------------------------------
def test_convert_angles_is_odd_including_the_cut():
    """The wrap must satisfy ca(-x) == -ca(x) everywhere, INCLUDING +-pi.

    A plain floor-wrap gives the half-open [-pi, pi) and collapses both +pi and
    -pi to -pi, which hardcodes the left/right fork direction at the
    facing-away branch cut and breaks mirror equivariance at that point.
    """
    assert dm.convert_angles(np.pi) == pytest.approx(np.pi)
    assert dm.convert_angles(-np.pi) == pytest.approx(-np.pi)
    # and after a full turn, the sign of the input still selects the endpoint
    assert dm.convert_angles(3*np.pi) == pytest.approx(np.pi)
    assert dm.convert_angles(-3*np.pi) == pytest.approx(-np.pi)
    for x in [0.0, 0.5, 2.0, np.pi, 2*np.pi, 3*np.pi, 4.0, 100.0]:
        assert dm.convert_angles(-x) == pytest.approx(-dm.convert_angles(x))
    # array form agrees with the scalar form and stays in the closed range
    arr = np.array([np.pi, -np.pi, 3*np.pi, -3*np.pi, 4.0, -4.0])
    assert np.allclose(dm.convert_angles(arr),
                       [dm.convert_angles(float(x)) for x in arr])
    wide = dm.convert_angles(np.linspace(-20.0, 20.0, 4001))
    assert np.all(np.abs(wide) <= np.pi + 1e-15)
    # idempotent: wrapping an already-wrapped value is a no-op
    assert np.array_equal(dm.convert_angles(wide), wide)



# ---------------------------------------------------------------------------
# Bifurcation invariance: K-doubling cancels the 1/2 at SC equilibria
# ---------------------------------------------------------------------------
def _coupled_jac(nbm, gamma_star, theta, half_angle, h=1e-6):
    """Build the 3x3 coupled Jacobian using either sin(arg(gamma)) or
    sin(arg(gamma)/2) (the neural consensus angle, matching the current
    NBM torque)."""
    gr0, gi0, th0 = gamma_star.real, gamma_star.imag, float(theta)

    def rhs(gr, gi, th):
        gamma = gr + 1j*gi
        dg = nbm.dgamma_dt(gamma=gamma, focal_angle=th,
                           focal_loc=nbm.percep_model.focal_loc)
        arg, R = np.angle(gamma), np.abs(gamma)
        s = np.sin(arg/2) if half_angle else np.sin(arg)
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
    nbm_old = dm.NeuralBandModel(percep_model=pm_old, beta=10.0, K=1)
    nbm_new = dm.NeuralBandModel(percep_model=pm_new, beta=10.0, K=2)

    checked = 0
    for fl in [(0.2, 0.0), (1.0, 0.5), (1.65, -0.8)]:
        nbm_old.percep_model.focal_loc = np.array(fl, dtype=float)
        nbm_new.percep_model.focal_loc = np.array(fl, dtype=float)
        # criterion is irrelevant here: only the equilibrium LOCATIONS are
        # used (the stability flags are discarded).
        angles, _ = nbm_old.sc_equilib(focal_loc=fl)
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


def _single_track(nbm, std, *, walk_std=0.75*np.pi, noise_exp=0, seed=0,
                  max_steps=300, start_angle=-np.pi/2):
    """Run one walker and return its (x, y) trajectory."""
    nbm.rng = np.random.default_rng(seed)
    fig, ax = plt.subplots()
    nbm.plot_walkers(dt=0.1, v=1, std=std, walk_std=walk_std, noise_exp=noise_exp,
                     repetitions=1, max_steps=max_steps, start_loc=(0.0, 0.0),
                     start_angle=start_angle, ax=ax)
    line = ax.get_lines()[-1]
    xd, yd = line.get_xdata(), line.get_ydata()
    plt.close(fig)
    return np.asarray(xd), np.asarray(yd)


def test_blind_spot_frozen_drifts_straight():
    """walk_std=0: a blind walker keeps its heading and walks off (frozen)."""
    xd, yd = _single_track(_blind_model(), std=0.0, walk_std=0.0)
    # heading frozen pointing south: x stays put, y marches negative
    assert np.std(xd) < 1e-6
    assert yd[-1] < -10            # walked far away from the (0,3) target


def test_blind_spot_search_escapes():
    """A blind walker searches at walk_std -- set independently of the committed
    std -- reorients, re-acquires, and is captured. Uses the DEFAULT constant-
    noise mode (std=None -> 0.1, noise_exp=0): without the independent walk_std
    the blind search would be a feeble 0.1 and the walker would march off."""
    max_steps = 400
    xd, yd = _single_track(_blind_model(), std=None, noise_exp=0,
                           max_steps=max_steps)
    # reorientation happened (frozen drift keeps x identically 0)
    assert np.std(xd) > 1e-3
    # terminated early -> found the target (track shorter than max_steps)
    assert len(xd) - 1 < max_steps
    # ended at the target (surface at y = 3 - 1 = 2), not drifting away south
    assert yd[-1] > 1.0


def test_walk_std_orthogonal_to_std():
    """walk_std and std are independent: walk_std=0 freezes the blind search
    even when the committed std>0 (the walker, always blind here, marches off);
    and a blind walker still searches under the default-constant mode regardless
    of the small committed std."""
    # std>0 but walk_std=0 -> blind frozen (marches straight off)
    xd, yd = _single_track(_blind_model(), std=0.5, walk_std=0.0)
    assert np.std(xd) < 1e-6
    assert yd[-1] < -10
    # std small (gentle constant) but walk_std>0 -> blind searches and escapes
    xd2, yd2 = _single_track(_blind_model(), std=0.1, walk_std=0.75*np.pi,
                             max_steps=400)
    assert np.std(xd2) > 1e-3 and yd2[-1] > 1.0


# ---------------------------------------------------------------------------
# State-gated noise sigma*(1-R)^noise_exp
# ---------------------------------------------------------------------------
def _final_spread(make, std, noise_exp, *, n=4, start_angle=0.2):
    """Sum of per-axis std of the final walker position across n seeds."""
    finals = []
    for s in range(n):
        xd, yd = _single_track(make(), std, noise_exp=noise_exp, seed=s,
                               max_steps=200, start_angle=start_angle)
        finals.append([xd[-1], yd[-1]])
    return float(np.std(np.array(finals), axis=0).sum())


def test_std_zero_is_deterministic():
    """std=0 => no angular noise => trajectories are seed-independent."""
    xa, ya = _single_track(_identity_nbm(), std=0.0, seed=0, start_angle=0.2,
                           max_steps=200)
    xb, yb = _single_track(_identity_nbm(), std=0.0, seed=9, start_angle=0.2,
                           max_steps=200)
    assert xa.shape == xb.shape
    assert np.allclose(xa, xb) and np.allclose(ya, yb)


def test_default_injects_noise():
    """Default (std=None, noise_exp=0) resolves to a gentle constant 0.1 noise,
    so trajectories are seed-dependent (unlike std=0)."""
    assert _final_spread(_identity_nbm, std=None, noise_exp=0) > 1e-6


def test_gate_quiets_committed_walker():
    """noise_exp>0 closes the gate as R rises, so at the same sigma a homing
    walker is quieter than under the constant (noise_exp=0) law."""
    s_const = _final_spread(_identity_nbm, std=0.75*np.pi, noise_exp=0)
    s_gated = _final_spread(_identity_nbm, std=0.75*np.pi, noise_exp=2)
    assert s_gated < s_const


# ---------------------------------------------------------------------------
# cos(Theta/2) heading-aligned noise modulation (noise_exp != 0)
# ---------------------------------------------------------------------------
def _offaxis_model():
    """Two symmetric delta targets due east, identity warp (neural == ego)."""
    targets = dm.Targets(locs=np.array([(3.0, 1.0), (3.0, -1.0)]), geom_name=None)
    pm = dm.PerceptionModel(targets=targets, focal_loc=(0.0, 0.0),
                            focal_angle=0.0, neural_angle_dist=None,
                            angle_weight=None)
    return dm.NeuralBandModel(percep_model=pm, beta=10.0, K=2)


def _one_step_loc(nbm, std, noise_exp, seed, start_angle, R_exp=1, dt=0.1, v=1):
    """Walker's actual (x,y) after a single step (the second track point).
    R_exp=None omits the kwarg so plot_walkers' own default is exercised."""
    nbm.rng = np.random.default_rng(seed); nbm.gamma = 0 + 0j
    fig, ax = plt.subplots()
    kw = {} if R_exp is None else {'R_exp': R_exp}
    nbm.plot_walkers(dt=dt, v=v, std=std, walk_std=0.5*np.pi, noise_exp=noise_exp,
                     repetitions=1, max_steps=1, start_loc=(0.0, 0.0),
                     start_angle=start_angle, ax=ax, **kw)
    line = ax.get_lines()[-1]
    loc = np.array([line.get_xdata()[1], line.get_ydata()[1]])
    plt.close(fig)
    return loc


def _predict_loc(nbm, std, noise_exp, seed, start_angle, with_cos, R_exp=1,
                 dt=0.1, v=1):
    """Predicted (x,y) after one step from the KNOWN first RNG draw, with drift
    K*R^R_exp*sin(Theta/2) and sigma_eff = std*(1-R)^noise_exp*[cos(Theta/2)]."""
    nbm.gamma = 0 + 0j
    dth = nbm.dtheta_dt(theta=start_angle)              # relaxes nbm.gamma
    R = abs(nbm.gamma); Theta = np.angle(nbm.gamma)
    s = dth / (nbm.K * R) if R > 0 else 0.0             # = sin(Theta/2)
    drift = nbm.K * R ** R_exp * s if R_exp != 1 else dth
    sig = std * max(0.0, 1.0 - R) ** noise_exp
    if with_cos and noise_exp != 0 and R > 0.0:
        sig *= np.cos(Theta / 2)
    # Mirror plot_walkers' per-walk seeding: it draws one int from self.rng
    # (= default_rng(seed), set by _one_step_loc) and spawns a child
    # SeedSequence per repetition; the walk's RNG is default_rng(child[0]).
    base = int(np.random.default_rng(seed).integers(0, 2**63 - 1))
    child = np.random.SeedSequence(base).spawn(1)[0]
    z0 = np.random.default_rng(child).normal()
    th = start_angle + drift * dt + sig * z0 * np.sqrt(dt)
    return Theta, R, v * dt * np.array([np.cos(th), np.sin(th)])


def test_cos_factor_applied_for_nonzero_noise_exp():
    """At noise_exp!=0 the visible noise carries an exact cos(Theta/2) factor.
    Off-axis heading => Theta!=0, R<1, so cos(Theta/2)<1 with the gate open."""
    nbm = _offaxis_model()
    ang = np.pi / 2                       # face north; consensus (east) is off-axis
    Theta, R, pred = _predict_loc(nbm, 2.0, 2, 1, ang, with_cos=True)
    assert abs(Theta) > 0.5 and R < 0.99          # meaningful cos<1, gate open
    act = _one_step_loc(nbm, 2.0, 2, 1, ang)
    assert np.allclose(pred, act, atol=1e-12)     # sigma_eff includes cos, exactly
    # omitting the cos factor must NOT reproduce the step
    _, _, pred_nocos = _predict_loc(nbm, 2.0, 2, 1, ang, with_cos=False)
    assert not np.allclose(pred_nocos, act, atol=1e-9)


def test_no_cos_factor_in_constant_mode():
    """noise_exp=0 (constant mode) has NO cos factor: the plain sigma*dW step."""
    nbm = _offaxis_model()
    ang = np.pi / 2
    _, _, pred = _predict_loc(nbm, 2.0, 0, 1, ang, with_cos=False)
    act = _one_step_loc(nbm, 2.0, 0, 1, ang)
    assert np.allclose(pred, act, atol=1e-12)


def test_walk_std_default_is_half_pi():
    """The walk_std signature default moved 0.75pi -> 0.5pi."""
    import inspect
    d = inspect.signature(
        dm.NeuralBandModel.plot_walkers).parameters['walk_std'].default
    assert d == pytest.approx(0.5 * np.pi)


def test_R_exp_default_is_one():
    """R_exp default is a flat 1 (the model torque) for every noise_exp -- the
    old regime-aware None -> 1/noise_exp coupling was removed."""
    import inspect
    assert inspect.signature(
        dm.NeuralBandModel.plot_walkers).parameters['R_exp'].default == 1
    nbm = _offaxis_model(); ang = np.pi / 2
    # noise_exp=2 with std=0 (noise off -> RNG-independent, location set purely by
    # drift): the default-driven step is the model torque (R_exp=1), NOT the old
    # 1/noise_exp = 0.5 boost.
    act = _one_step_loc(nbm, 0.0, 2, 1, ang, R_exp=None)   # omit -> plot_walkers default
    _, _, p_unit = _predict_loc(nbm, 0.0, 2, 1, ang, with_cos=True, R_exp=1)
    assert np.allclose(p_unit, act, atol=1e-12)
    _, _, p_half = _predict_loc(nbm, 0.0, 2, 1, ang, with_cos=True, R_exp=0.5)
    assert not np.allclose(p_half, act, atol=1e-9)


def test_R_exp_scales_walker_drift():
    """R_exp!=1 makes the walker's drift K*R^R_exp*sin(Theta/2): the step matches
    the R_exp-included prediction exactly, and NOT the R_exp=1 one."""
    nbm = _offaxis_model(); ang = np.pi / 2
    _, R, pred = _predict_loc(nbm, 2.0, 2, 1, ang, with_cos=True, R_exp=0.5)
    assert 0.0 < R < 1.0                                   # R^0.5 != R
    act = _one_step_loc(nbm, 2.0, 2, 1, ang, R_exp=0.5)
    assert np.allclose(pred, act, atol=1e-12)
    _, _, pred_unit = _predict_loc(nbm, 2.0, 2, 1, ang, with_cos=True, R_exp=1)
    assert not np.allclose(pred_unit, act, atol=1e-9)
