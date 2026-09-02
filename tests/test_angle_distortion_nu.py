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

"""Tests for `NeuralBandModel(angle_distortion_nu=...)` -- the distorted cosine
coupling kernel J(x) = cos(pi*(|x|/pi)**nu) of Sridhar et al. (2021) Eq. [2].

This kernel came in with the fold of the allocentric-frame IsingExtModel into
NBM (retired 2026-08-20). That class was NBM in a rotated frame: with an
identity warp, uniform weight and beta = N_visible/T,

    dgamma_IEM(gamma*exp(i*theta), theta) == exp(i*theta) * dgamma_NBM(gamma, theta)

so the two differed only by an advection term that vanishes at every
equilibrium. The equivalence tests that verified this retired with the class;
see theory/iem_nbm_fold.md for the measurements they produced.

What remains here are the intrinsic properties of the folded model:
  - nu=None and nu=1 leave the model bit-identical to the plain cosine one
    (same kernel, same sc_equilib probe radii, same analytic _discrim_A).
  - the constructor and the property setter both reject a warped or weighted
    PerceptionModel when nu is set, and non-positive/non-finite nu.
  - nu_cosine's zero crossing sits at pi*(1/2)**(1/nu).
  - dgamma_dt wraps the kernel argument into [-pi, pi]: the nu kernel is NOT
    2*pi-periodic, so omitting the wrap silently changes the coupling for any
    target more than pi away from the consensus.
  - _discrim_A's numerical fallback agrees with the analytic free-energy
    Hessian wherever both apply.
  - plot_dtheta_dt's neutral-seed sweep does not leak self.gamma.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from decision_model import (Targets, PerceptionModel, NeuralBandModel,
                            convert_angles)

# Legacy per-target temperature: the earlier parameterization's effective
# coupling was N_targets/T, so beta = N/T reproduces it.
T_LEGACY = 0.2
TWO_TARGETS = np.array([[0., 1.5], [0., -1.5]])
GEOMS = [('delta', dict(geom_name=None)),
         ('circle', dict(geom_name='circle', r=0.5)),
         ('capsule', dict(geom_name='capsule', l=0.8, w=0.3, theta=0.4))]


def _identity_pm(geom_kwargs=None):
    """The perception model the nu kernel requires: identity warp, uniform
    weight."""
    t = Targets(locs=TWO_TARGETS, **(geom_kwargs or dict(geom_name='circle',
                                                         r=0.5)))
    return PerceptionModel(t, (3, 0), 0., neural_angle_dist=None,
                           angle_weight=None)


# =====================================================================
# 1. INTRINSIC
# =====================================================================

@pytest.mark.parametrize('nu', [None, 1, 1.0])
def test_plain_cosine_when_nu_is_absent_or_one(nu):
    """nu=None and nu=1 are the same model as no nu at all: the flag that
    gates every nu-specific branch stays off, so the kernel, the sc_equilib
    probe radii and the analytic _discrim_A are untouched."""
    m = NeuralBandModel(_identity_pm(), angle_distortion_nu=nu)
    assert m._nu_active is False
    x = np.linspace(-np.pi, np.pi, 101)
    assert np.array_equal(m.nu_cosine(x), np.cos(x))


def test_nu_none_matches_explicit_nu_one_everywhere():
    """The whole deterministic stack agrees bit-for-bit between nu=None and
    nu=1 -- equilibria, coherence and stability under both criteria."""
    pm = _identity_pm()
    plain = NeuralBandModel(pm, beta=10)
    one = NeuralBandModel(pm, beta=10, angle_distortion_nu=1)
    for loc in ([2., 0.], [1.5, 0.8], [4., -1.2]):
        for crit in ('reduced', 'discrim_a'):
            a = plain.sc_equilib(focal_loc=loc, return_R=True,
                                 stability_criterion=crit)
            b = one.sc_equilib(focal_loc=loc, return_R=True,
                               stability_criterion=crit)
            assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


@pytest.mark.parametrize('pm_kwargs', [
    dict(),                                              # default lin_cutoff
    dict(neural_angle_dist='vonmises', a_warp=0.55),
    dict(neural_angle_dist='direct_power', a_warp=0.5),
    dict(neural_angle_dist=None, angle_weight='cutoff'),
    dict(neural_angle_dist='vonmises', a_warp=0.55,
         angle_weight='neural_angle_dist'),
])
def test_nu_rejects_warped_or_weighted_perception(pm_kwargs):
    """nu bends the coupling strength; a warp bends the angles themselves.
    Setting nu on a model that does either is refused rather than silently
    composing two distortions."""
    t = Targets(locs=TWO_TARGETS, geom_name='circle', r=0.5)
    pm = PerceptionModel(t, (3, 0), 0., **pm_kwargs)
    with pytest.raises(ValueError, match='angle_distortion_nu'):
        NeuralBandModel(pm, angle_distortion_nu=0.5)
    # ... and the same perception model is fine with no nu.
    assert NeuralBandModel(pm)._nu_active is False


def test_nu_guard_fires_on_nu_one_too():
    """The perception-model requirement keys on nu being SET, not on it being
    non-trivial: nu=1 with a warp is refused as well, so the configuration
    means the same thing at every nu."""
    t = Targets(locs=TWO_TARGETS, geom_name='circle', r=0.5)
    with pytest.raises(ValueError):
        NeuralBandModel(PerceptionModel(t, (3, 0), 0.),
                        angle_distortion_nu=1)


def test_nu_is_a_validating_property():
    """angle_distortion_nu validates on assignment, not only at construction,
    so sweeping it on a built model cannot slip past the perception-model
    requirement."""
    m = NeuralBandModel(_identity_pm(), angle_distortion_nu=0.5)
    for v, active in ((0.5, True), (1, False), (2, True), (None, False)):
        m.angle_distortion_nu = v
        assert m._nu_active is active
        assert v is None or isinstance(m.angle_distortion_nu, float)

    # Assigning onto a warped/weighted model is refused the same way the
    # constructor refuses it...
    w = NeuralBandModel(PerceptionModel(
        Targets(locs=TWO_TARGETS, geom_name='circle', r=0.5), (3, 0), 0.))
    with pytest.raises(ValueError, match='angle_distortion_nu'):
        w.angle_distortion_nu = 0.5
    assert w.angle_distortion_nu is None
    w.angle_distortion_nu = None      # ... but clearing it is always allowed


@pytest.mark.parametrize('bad', [0, -1, -0.5, float('nan'), float('inf')])
def test_nu_rejects_nonpositive_and_nonfinite(bad):
    """nu <= 0 is degenerate rather than sharp: (|x|/pi)**nu -> 1 for every
    x != 0 as nu -> 0, collapsing the kernel to the constant -1, and it
    diverges at x = 0 for nu < 0."""
    m = NeuralBandModel(_identity_pm())
    with pytest.raises(ValueError):
        m.angle_distortion_nu = bad


@pytest.mark.parametrize('bad', ['sharp', [0.5], None.__class__])
def test_nu_rejects_non_numbers(bad):
    # A numeric STRING is accepted -- the setter coerces with float(), the
    # same latitude every other numeric argument on this class has.
    m = NeuralBandModel(_identity_pm())
    with pytest.raises(TypeError):
        m.angle_distortion_nu = bad


def test_nu_assignment_changes_the_kernel():
    """The setter is not merely a guard -- the assigned value drives the
    kernel and the equilibrium finder."""
    m = NeuralBandModel(_identity_pm(dict(geom_name=None)),
                        beta=TWO_TARGETS.shape[0]/T_LEGACY)
    x = 1.0
    assert m.nu_cosine(x) == pytest.approx(np.cos(x))
    m.angle_distortion_nu = 0.5
    assert m.nu_cosine(x) == pytest.approx(
        np.cos(np.pi*(abs(x)/np.pi)**0.5))
    loc = np.array([1.75, -1.0])
    angles, Rs, _ = m.sc_equilib(focal_loc=loc, return_R=True)
    assert any(r > 0.7 for r in Rs), (
        'the nu-gated high probe radius should be in force after assignment')


def test_nu_default_perception_model_is_identity_uniform():
    m = NeuralBandModel(angle_distortion_nu=0.5)
    assert m.percep_model.warp_name is None
    assert m.percep_model.weight_name is None
    # No nu: the class default (a warped model) is still what you get.
    assert NeuralBandModel().percep_model.warp_name == 'lin_cutoff'


@pytest.mark.parametrize('nu', [0.5, 1, 2, 3])
def test_nu_cosine_zero_crossing(nu):
    """J(x) = cos(pi*(|x|/pi)**nu) crosses zero at pi*(1/2)**(1/nu): lower nu
    is a sharper, more locally excitatory kernel."""
    m = NeuralBandModel(_identity_pm(), angle_distortion_nu=nu)
    x0 = np.pi*(0.5)**(1/nu)
    assert m.nu_cosine(0.0) == pytest.approx(1.0)
    assert m.nu_cosine(x0) == pytest.approx(0.0, abs=1e-12)
    assert m.nu_cosine(-x0) == pytest.approx(0.0, abs=1e-12)
    assert m.nu_cosine(np.pi) == pytest.approx(-1.0)
    assert m.nu_cosine(-np.pi) == pytest.approx(-1.0)


def test_nu_cosine_is_not_2pi_periodic():
    """The reason dgamma_dt must wrap before calling it. cos is periodic;
    this kernel reads |x| literally."""
    m = NeuralBandModel(_identity_pm(), angle_distortion_nu=0.5)
    x = 3.283
    assert m.nu_cosine(x) != pytest.approx(m.nu_cosine(x - 2*np.pi), abs=1e-6)


@pytest.mark.parametrize('nu', [0.5, 2])
def test_dgamma_dt_wraps_the_kernel_argument(nu):
    """Regression: dgamma_dt must feed convert_angles(neural_angle - Theta)
    to the kernel. Reproduce the sum by hand, with and without the wrap, at a
    state where some target sits more than pi from the consensus; only the
    wrapped reference may match."""
    pm = _identity_pm()
    m = NeuralBandModel(pm, beta=10, angle_distortion_nu=nu)
    loc, th, gamma = np.array([2.0, 0.4]), 0.0, 0.6*np.exp(1j*2.9)
    angles, rho = pm.get_neural_signals(th, loc)
    assert np.abs(convert_angles(angles - np.angle(gamma))).max() > 0.0
    assert (np.abs(angles - np.angle(gamma)) > np.pi).any(), \
        'test state must exercise the wrap'

    R, Theta = np.abs(gamma), np.angle(gamma)

    def reference(arg):
        s = rho*np.exp(1j*angles)/(1+np.exp(-2*m.beta*R*m.nu_cosine(arg)))
        return np.sum(s) - gamma

    got = m.dgamma_dt(gamma=gamma, focal_angle=th, focal_loc=loc)
    assert got == pytest.approx(reference(convert_angles(angles - Theta)))
    assert got != pytest.approx(reference(angles - Theta))


@pytest.mark.parametrize('geom_name,geom_kwargs', GEOMS)
def test_discrim_A_fallback_matches_analytic_hessian(geom_name, geom_kwargs):
    """_discrim_A uses the analytic free-energy Hessian for the plain cosine
    kernel and a numerically differenced fast block otherwise. Where both are
    valid (nu=1) they must return the same verdict at every equilibrium."""
    pm = _identity_pm(geom_kwargs)
    analytic = NeuralBandModel(pm, beta=10)
    numeric = NeuralBandModel(pm, beta=10, angle_distortion_nu=1)
    # Nudge nu off 1 to select the numerical branch without changing the
    # kernel to anything the analytic form could not also describe.
    numeric.angle_distortion_nu = 1 + 1e-12
    assert numeric._nu_active is True
    n_eq = 0
    for x in np.linspace(0.5, 4.5, 7):
        for y in np.linspace(-2.0, 2.0, 5):
            loc = np.array([x, y])
            angles, Rs, _ = analytic.sc_equilib(focal_loc=loc, return_R=True)
            for th, R in zip(angles, Rs):
                n_eq += 1
                assert (analytic._discrim_A(R+0j, th, loc)
                        == numeric._discrim_A(R+0j, th, loc))
    assert n_eq > 20, 'expected a meaningful number of equilibria to compare'


def test_sc_equilib_adds_a_high_probe_only_under_nu():
    """The distorted kernel pushes equilibria out to R ~ 0.8, past the plain
    model's probe radii, so sc_equilib scans an extra high radius when (and
    only when) the kernel is distorted."""
    import inspect
    src = inspect.getsource(NeuralBandModel.sc_equilib)
    assert '(0.3, 0.5, 0.7, 0.85)' in src and '(0.3, 0.5, 0.7)' in src
    # Behavioural side: at nu=0.5 the extra probe surfaces a genuine root that
    # the three plain radii miss. Its residual must be at machine precision.
    pm = _identity_pm(dict(geom_name=None))
    m = NeuralBandModel(pm, beta=2/T_LEGACY, angle_distortion_nu=0.5)
    loc = np.array([1.75, -1.0])
    angles, Rs, _ = m.sc_equilib(focal_loc=loc, return_R=True)
    high = [(t, r) for t, r in zip(angles, Rs) if r > 0.7]
    assert high, 'expected an equilibrium above the plain probe radii'
    for th, R in high:
        assert abs(m.dgamma_dt(gamma=R+0j, focal_angle=th,
                               focal_loc=loc)) < 1e-10


def test_plot_dtheta_dt_neutral_seed_restores_gamma():
    """gamma=False re-seeds each heading from a neutral gamma instead of
    sweeping the carried one, and must leave self.gamma where it found it."""
    m = NeuralBandModel(_identity_pm(), beta=10, angle_distortion_nu=0.5)
    m.gamma = 0.37 + 0.11j
    before = m.gamma
    m.plot_dtheta_dt(focal_loc=(3, 0), gamma=False)
    plt.close('all')
    assert m.gamma == before
    # The swept form is allowed to move it (it is the model's running state).
    m.plot_dtheta_dt(focal_loc=(3, 0))
    plt.close('all')
    assert m.gamma != before


def test_plot_nu_cosine_runs():
    NeuralBandModel(_identity_pm(), angle_distortion_nu=0.5).plot_nu_cosine()
    plt.close('all')
    NeuralBandModel().plot_nu_cosine()      # plain cosine, nu label 1
    plt.close('all')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
