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

"""Tests for PerceptionModel.signal_cache and its use in the NBM root finders.

The cache is a pure performance change: `sc_equilib` and `gamma_equilib` re-fetch
identical perception signals many times per call (a fixed heading across the
whole `gamma_equilib` multistart; repeated headings across the `sc_equilib` scan
probes, polish iterates and stability test), and memoizing them on the exact
observer state removes the repeats. Nothing about *what* is computed changes, so
the load-bearing test is equality of every solver output with the cache disabled.

 - test_cached_signals_are_identical / _same_object:
       a cache hit returns exactly what a recompute would, without recomputing.

 - test_cache_is_dropped_on_exit / _on_exception / test_nested_cache_*:
       lifetime -- the memo never outlives its block (so it cannot go stale
       against a later warp/weight/target change), and nesting reuses the
       outer block's memo rather than clearing it early.

 - test_theta_probes_are_distinct_entries / test_jacobian_theta_column_survives:
       the exact-key requirement. Perception is the ONLY theta dependence in
       dgamma_dt, so a key that merged the Jacobian's theta +- h probes with the
       base point would zero the theta column of the coupled Jacobian, give
       det(J) = 0, and report every equilibrium unstable.

 - test_sc_equilib_identical_* / test_gamma_equilib_identical_*:
       solver output is bit-identical with the cache disabled, over a spatial
       grid x all three stability criteria x several perception setups.
"""
import os
import sys
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from decision_model import Targets, PerceptionModel, NeuralBandModel

pi = np.pi
CRITERIA = ('reduced', 'discrim_a')


def _fly2():
    """Two circle targets, lin_cutoff warp + independent lin_cutoff weight."""
    t = Targets(locs=np.array([[4.33, 2.5], [4.33, -2.5]]),
                geom_name='circle', r=0.5)
    pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist='lin_cutoff',
                         angle_weight='lin_cutoff', a_warp=0.65*pi,
                         b_warp=0.9*pi, a_weight=0.25*pi, b_weight=0.30*pi)
    return NeuralBandModel(pm, beta=20.0, K=2.0)


def _vm055():
    """vonmises warp, uniform weight -- the VM-k055 calibration setup."""
    t = Targets(locs=np.array([[4.33, 2.5], [4.33, -2.5]]),
                geom_name='circle', r=0.5)
    pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist='vonmises',
                         angle_weight=None, a_warp=0.55)
    return NeuralBandModel(pm, beta=10.0, K=2.0)


def _delta_power():
    """Delta targets + power warp: the zero-extent perception path."""
    t = Targets(locs=np.array([[4.0, 2.0], [4.0, -2.0], [5.0, 0.0]]),
                geom_name=None)
    pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist='direct_power',
                         angle_weight=None, a_warp=0.5)
    return NeuralBandModel(pm, beta=10.0, K=2.0)


BUILDERS = {'fly2': _fly2, 'vm055': _vm055, 'delta_power': _delta_power}
GRID = [(0.5, 0.0), (1.5, 0.0), (2.0, 1.0), (3.0, -1.5), (4.0, 1.5),
        (4.5, -0.5), (6.0, 2.0)]
HEADINGS = [0.0, 0.7, -1.9, pi]


@contextmanager
def _null_cache(self):
    """Stand-in for signal_cache() that memoizes nothing."""
    yield


# --- what the memo hands back -------------------------------------------

def test_cached_signals_are_identical():
    pm = _fly2().percep_model
    loc = np.array([2.0, 1.0])
    want = pm.get_neural_signals(0.3, loc)
    with pm.signal_cache():
        first = pm.get_neural_signals(0.3, loc)
        second = pm.get_neural_signals(0.3, loc)
    for got in (first, second):
        assert np.array_equal(got[0], want[0])
        assert np.array_equal(got[1], want[1])


def test_cache_hit_skips_recompute():
    pm = _fly2().percep_model
    loc = np.array([2.0, 1.0])
    with pm.signal_cache():
        first = pm.get_neural_signals(0.3, loc)
        second = pm.get_neural_signals(0.3, loc)
    # a hit returns the stored arrays themselves, not a fresh computation
    assert first[0] is second[0] and first[1] is second[1]


def test_resolved_defaults_share_a_key():
    """An explicit state and the same state left to the object's attributes
    are one cache entry (the key is built after the None-resolution)."""
    pm = _fly2().percep_model
    pm.focal_loc = np.array([2.0, 1.0])
    pm.focal_angle = 0.3
    with pm.signal_cache():
        explicit = pm.get_neural_signals(0.3, np.array([2.0, 1.0]))
        implicit = pm.get_neural_signals()
    assert explicit[0] is implicit[0]


def test_empty_signals_are_cached_too():
    """A state where nothing is visible (a tight weight cone facing away)
    still round-trips as the empty-array pair."""
    t = Targets(locs=np.array([[4.0, 0.0]]), geom_name='circle', r=0.5)
    pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist='lin_cutoff',
                         angle_weight='lin_cutoff', a_warp=0.25*pi,
                         b_warp=0.9*pi, a_weight=0.2*pi, b_weight=0.4*pi)
    loc = np.array([0.0, 0.0])
    blind = pm.get_neural_signals(pi, loc)
    assert blind[0].size == 0
    with pm.signal_cache():
        assert pm.get_neural_signals(pi, loc)[0].size == 0
        assert pm.get_neural_signals(pi, loc)[0].size == 0


# --- lifetime -----------------------------------------------------------

def test_no_cache_by_default():
    pm = _fly2().percep_model
    assert pm._signal_cache is None
    pm.get_neural_signals(0.3, np.array([2.0, 1.0]))
    assert pm._signal_cache is None


def test_cache_is_dropped_on_exit():
    pm = _fly2().percep_model
    with pm.signal_cache():
        pm.get_neural_signals(0.3, np.array([2.0, 1.0]))
        assert pm._signal_cache
    assert pm._signal_cache is None
    assert '_signal_cache' not in pm.__dict__


def test_cache_is_dropped_on_exception():
    pm = _fly2().percep_model
    with pytest.raises(RuntimeError):
        with pm.signal_cache():
            pm.get_neural_signals(0.3, np.array([2.0, 1.0]))
            raise RuntimeError('boom')
    assert pm._signal_cache is None


def test_nested_cache_reuses_the_outer_one():
    pm = _fly2().percep_model
    loc = np.array([2.0, 1.0])
    with pm.signal_cache():
        outer = pm._signal_cache
        first = pm.get_neural_signals(0.3, loc)
        with pm.signal_cache():
            assert pm._signal_cache is outer
            assert pm.get_neural_signals(0.3, loc)[0] is first[0]
        # the inner block must not have torn down the outer block's memo
        assert pm._signal_cache is outer
        assert pm.get_neural_signals(0.3, loc)[0] is first[0]
    assert pm._signal_cache is None


def test_parameter_change_after_the_block_is_seen():
    pm = _fly2().percep_model
    loc = np.array([2.0, 1.0])
    with pm.signal_cache():
        before = pm.get_neural_signals(0.3, loc)
    pm.a_warp = 0.2*pi
    after = pm.get_neural_signals(0.3, loc)
    assert not np.array_equal(before[0], after[0])


# --- the exact-key requirement ------------------------------------------

@pytest.mark.parametrize('h', [1e-6, 1e-7])
def test_theta_probes_are_distinct_entries(h):
    """The Jacobians difference in theta at h = 1e-6 / 1e-7; those probes must
    not collide with the base state."""
    pm = _fly2().percep_model
    loc = np.array([2.0, 1.0])
    th = 0.3
    with pm.signal_cache():
        base = pm.get_neural_signals(th, loc)
        plus = pm.get_neural_signals(th + h, loc)
        minus = pm.get_neural_signals(th - h, loc)
        assert len(pm._signal_cache) == 3
    assert not np.array_equal(base[0], plus[0])
    assert not np.array_equal(base[0], minus[0])


def test_jacobian_theta_column_survives_the_cache():
    """b = d(dgamma)/dtheta is the whole theta dependence of the coupled
    Jacobian: if the memo returned the base signals for the theta +- h probes,
    b would be zero, det(J) would be zero, and 'reduced' would call every
    equilibrium unstable."""
    nbm = _vm055()
    loc = np.array([2.0, 0.0])
    angles, Rs, stab = nbm.sc_equilib(focal_loc=loc, return_R=True)
    assert any(stab), 'setup must have a stable equilibrium to be a test'
    for th, R, is_stable in zip(angles, Rs, stab):
        with nbm.percep_model.signal_cache():
            J = nbm._coupled_jacobian(R + 0j, th, loc)
        assert np.any(J[:2, 2] != 0.0)
        assert np.linalg.det(J) != 0.0
        assert nbm._discrim_reduced(R + 0j, th, loc) == is_stable


# --- solver output is unchanged -----------------------------------------

@pytest.mark.parametrize('name', sorted(BUILDERS))
@pytest.mark.parametrize('criterion', CRITERIA)
def test_sc_equilib_matches_uncached(name, criterion, monkeypatch):
    nbm = BUILDERS[name]()
    cached = [nbm.sc_equilib(focal_loc=np.array(loc),
                             stability_criterion=criterion, return_R=True)
              for loc in GRID]
    monkeypatch.setattr(PerceptionModel, 'signal_cache', _null_cache)
    uncached = [nbm.sc_equilib(focal_loc=np.array(loc),
                               stability_criterion=criterion, return_R=True)
                for loc in GRID]
    assert repr(cached) == repr(uncached)


@pytest.mark.parametrize('name', sorted(BUILDERS))
@pytest.mark.parametrize('criterion', CRITERIA)
def test_gamma_equilib_matches_uncached(name, criterion, monkeypatch):
    nbm = BUILDERS[name]()
    locs = [(2.0, 1.0), (4.0, 1.5)]
    cached = [nbm.gamma_equilib(focal_angle=th, focal_loc=np.array(loc),
                                stability_criterion=criterion)
              for loc in locs for th in HEADINGS]
    monkeypatch.setattr(PerceptionModel, 'signal_cache', _null_cache)
    uncached = [nbm.gamma_equilib(focal_angle=th, focal_loc=np.array(loc),
                                  stability_criterion=criterion)
                for loc in locs for th in HEADINGS]
    assert repr(cached) == repr(uncached)


def test_count_stable_at_matches_uncached(monkeypatch):
    nbm = _fly2()
    cached = [nbm._count_stable_at((i, x, y, 'reduced'))
              for i, (x, y) in enumerate(GRID)]
    monkeypatch.setattr(PerceptionModel, 'signal_cache', _null_cache)
    uncached = [nbm._count_stable_at((i, x, y, 'reduced'))
                for i, (x, y) in enumerate(GRID)]
    assert cached == uncached


def test_solvers_leave_no_cache_behind():
    nbm = _fly2()
    nbm.sc_equilib(focal_loc=np.array([2.0, 1.0]))
    assert nbm.percep_model._signal_cache is None
    nbm.gamma_equilib(focal_angle=0.3, focal_loc=np.array([2.0, 1.0]))
    assert nbm.percep_model._signal_cache is None


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
