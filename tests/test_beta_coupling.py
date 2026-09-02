# Neural Ring Model: Ising-type dynamics of spatial decision-making.
# Copyright (C) 2026 Christopher Strickland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Neural Boltzmann factor `beta` in NeuralBandModel — coupling semantics.

`NeuralBandModel.beta` is the Boltzmann factor E/(k_B*temp) of the Glauber
dynamics. It is a property of the neural ring, so the logistic argument in
`dgamma_dt` is `2*beta*R*cos(theta_j - Theta)` with no dependence on how many
targets happen to be in the scene or in view.

These tests pin three things:

  1. the logistic argument really is 2*beta*R*cos(...), against an independent
     reference implementation;
  2. that argument does not scale with the target count, and in particular does
     not change when a target drops out of perception entirely;
  3. the analytic free-energy Hessian in `_discrim_A` uses the same beta as
     `dgamma_dt` -- verified against the numerically-differenced fast block of
     `_coupled_jacobian`, which depends on beta only through `dgamma_dt`.

Test (3) is the one that would catch a mis-substituted beta in `_discrim_A`:
that Hessian is written analytically, so nothing else in the codebase cross-
checks its temperature factor.
"""
import numpy as np
import pytest

from decision_model import Targets, PerceptionModel, NeuralBandModel

pi = np.pi

TWO = np.array([[4.33, 2.5], [4.33, -2.5]])
FOUR = np.array([[4.33, 2.25], [4.33, -2.25], [4.33, 0.75], [4.33, -0.75]])


def _nbm(locs, beta, *, weight=None, b_weight=None, r=0.5):
    t = Targets(locs=locs, geom_name='circle', r=r)
    kw = dict(neural_angle_dist='lin_cutoff', a_warp=0.25*pi, b_warp=0.9*pi,
              angle_weight=weight)
    if weight is not None and b_weight is not None:
        kw.update(a_weight=0.4*pi, b_weight=b_weight)
    pm = PerceptionModel(t, (0, 0), 0, **kw)
    return NeuralBandModel(pm, beta=beta)


def _reference_dgamma_dt(nbm, gamma, focal_angle, focal_loc):
    """Independent transcription of the mean-field Glauber dgamma/dt."""
    neur_angles, rho = nbm.percep_model.get_neural_signals(focal_angle,
                                                          focal_loc)
    if neur_angles.size == 0:
        return -gamma
    R, Theta = np.abs(gamma), np.angle(gamma)
    logistic = 1.0/(1.0 + np.exp(-2.0*nbm.beta*R*np.cos(neur_angles - Theta)))
    return np.sum(rho*np.exp(1j*neur_angles)*logistic) - gamma


STATES = [(0.35+0.2j, 0.0, (1.5, 0.0)),
          (0.8-0.1j, 0.7, (3.0, 1.0)),
          (0.05+0.0j, -2.0, (0.5, -2.0)),
          (0.6+0.6j, pi, (4.0, 0.5))]


def test_default_beta_is_10():
    assert NeuralBandModel().beta == 10


@pytest.mark.parametrize('beta', [1.0, 10.0, 30.0])
def test_dgamma_dt_matches_reference(beta):
    nbm = _nbm(TWO, beta)
    for gamma, ang, loc in STATES:
        got = nbm.dgamma_dt(None, gamma, ang, np.array(loc, dtype=float))
        want = _reference_dgamma_dt(nbm, gamma, ang, np.array(loc, dtype=float))
        assert got == want


def test_coupling_does_not_scale_with_target_count():
    """Same beta, different N: the per-target logistic weight is unchanged.

    Compare the logistic factor applied to a target sitting at a given neural
    angle in a 2-target scene and in a 4-target scene. Under the old N/T
    coupling these differed by a factor of two; under beta they are identical.
    """
    beta, R = 10.0, 0.5
    nbm2, nbm4 = _nbm(TWO, beta), _nbm(FOUR, beta)
    loc, ang = np.array([1.0, 0.0]), 0.0

    def logistic_factors(nbm):
        a, _ = nbm.percep_model.get_neural_signals(ang, loc)
        return a, 1.0/(1.0 + np.exp(-2.0*nbm.beta*R*np.cos(a)))

    a2, f2 = logistic_factors(nbm2)
    a4, f4 = logistic_factors(nbm4)
    assert a2.size == 2 and a4.size == 4          # both scenes fully visible

    # The factor depends only on the neural angle, not on how many targets
    # share the ring: evaluate both scenes' formula at a common angle.
    probe = 0.3
    g2 = 1.0/(1.0 + np.exp(-2.0*nbm2.beta*R*np.cos(probe)))
    g4 = 1.0/(1.0 + np.exp(-2.0*nbm4.beta*R*np.cos(probe)))
    assert g2 == g4

    # And the model's own dgamma_dt agrees with the reference in both.
    for nbm in (nbm2, nbm4):
        gamma = R + 0j
        assert nbm.dgamma_dt(None, gamma, ang, loc) == \
            _reference_dgamma_dt(nbm, gamma, ang, loc)


def test_coupling_unchanged_when_a_target_leaves_view():
    """Losing a target from perception must not re-temper the neural ring.

    With a restricted weight cone some targets contribute nothing at all. The
    remaining targets' logistic factors must be the same as they would be with
    every target in view -- beta is a property of the ring, not of the scene.
    """
    beta, R = 10.0, 0.5
    nbm = _nbm(np.array([[3.0, 0.0], [-3.0, 0.3], [0.0, 3.0]]), beta,
               weight='lin_cutoff', b_weight=0.5*pi, r=0.3)
    loc = np.array([0.0, 0.0])

    seen_counts = set()
    for ang in np.linspace(-pi, pi, 24, endpoint=False):
        a, _ = nbm.percep_model.get_neural_signals(ang, loc)
        seen_counts.add(a.size)
        gamma = R + 0j
        # The reference uses a fixed 2*beta*R*cos(...) regardless of a.size.
        assert nbm.dgamma_dt(None, gamma, ang, loc) == \
            _reference_dgamma_dt(nbm, gamma, ang, loc)
    # The scenario must actually exercise a varying visible count, or the test
    # is checking nothing.
    assert len(seen_counts) > 1, f'visible count never varied: {seen_counts}'


@pytest.mark.parametrize('beta', [10.0, 30.0])
def test_discrim_A_hessian_uses_same_beta_as_dgamma_dt(beta):
    """The analytic fast block in `_discrim_A` == the numerical one.

    `_coupled_jacobian` differences `dgamma_dt`, so its 2x2 gamma block carries
    whatever temperature factor `dgamma_dt` uses. `_discrim_A` builds the same
    block analytically (A_block = -H_Fhat). Requiring the two to agree pins the
    beta in the Hessian to the beta in the flow.
    """
    nbm = _nbm(TWO, beta)
    checked = 0
    for xy in [(0.5, 0.0), (1.5, 0.0), (2.5, 0.8), (3.5, -1.2), (4.5, 0.0)]:
        fl = np.array(xy, dtype=float)
        angles, _, Rs = nbm.sc_equilib(focal_loc=fl, return_R=True)
        for th, R in zip(angles, Rs):
            gamma = complex(R)
            A_num = nbm._coupled_jacobian(gamma, th, fl)[:2, :2]

            # Analytic block, transcribed from _discrim_A: in the frame rotated
            # to gamma, A_block = -(I - sum_j w_j vhat_j vhat_j^T).
            neur, rho = nbm.percep_model.get_neural_signals(th, fl)
            Theta = np.angle(gamma)
            w = (nbm.beta/2*rho
                 / np.cosh(nbm.beta*R*np.cos(Theta - neur))**2)
            cc, ss = np.cos(Theta - neur), np.sin(Theta - neur)
            H_rr = 1 - np.sum(w*cc**2)
            H_tt = 1 - np.sum(w*ss**2)
            H_rt = -np.sum(w*cc*ss)

            # Compare rotation-invariant scalars (trace and det) so the frame
            # convention does not matter.
            assert np.trace(A_num) == pytest.approx(-(H_rr + H_tt), abs=1e-5)
            assert np.linalg.det(A_num) == pytest.approx(
                H_rr*H_tt - H_rt**2, abs=1e-5)
            checked += 1
    assert checked > 0, 'no self-consistent equilibria found to check'


def test_legacy_N_over_T_equivalence_at_full_visibility():
    """beta = N/T reproduces the old per-target-temperature coupling.

    Where every target contributes to perception, the old logistic argument
    2*N*R*cos(...)/T equals 2*beta*R*cos(...) with beta = N/T. Delta targets
    never occlude and a uniform weight has full support, so N is constant here.

    Agreement is to a few ULP rather than bit-exact: folding N/T into a single
    beta re-associates the arithmetic (one rounded constant times R*cos,
    instead of N*R*cos divided by T), and N/T is not generally representable.
    The tolerance below is ~1e-12 relative, four orders tighter than the
    solver tolerances anything downstream runs at.
    """
    T, locs = 0.2, np.array([[4.33, 2.5], [5.0, 0.0], [4.33, -2.5]])
    n = len(locs)
    t = Targets(locs=locs, geom_name=None)
    pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist='lin_cutoff',
                         a_warp=0.25*pi, b_warp=0.9*pi, angle_weight=None)
    nbm = NeuralBandModel(pm, beta=n/T)

    rng = np.random.default_rng(4)
    for _ in range(50):
        gamma = rng.uniform(0, 1)*np.exp(1j*rng.uniform(-pi, pi))
        ang = rng.uniform(-pi, pi)
        loc = np.array([rng.uniform(-1, 6), rng.uniform(-3.5, 3.5)])

        neur, rho = pm.get_neural_signals(ang, loc)
        assert neur.size == n                      # full visibility by design
        R, Theta = np.abs(gamma), np.angle(gamma)
        legacy = np.sum(
            rho*np.exp(1j*neur)
            / (1 + np.exp(-2*neur.size*R*np.cos(neur - Theta)/T))) - gamma

        got = nbm.dgamma_dt(None, gamma, ang, loc)
        assert got.real == pytest.approx(legacy.real, rel=1e-12, abs=1e-14)
        assert got.imag == pytest.approx(legacy.imag, rel=1e-12, abs=1e-14)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
