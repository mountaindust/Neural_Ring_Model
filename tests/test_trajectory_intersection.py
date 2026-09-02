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

"""Tests for Targets.check_trajectory_intersection and _min_dist_segments."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from decision_model import Targets

pi = np.pi


def test_circle_passthrough():
    tgt = Targets(locs=np.array([[5.0, 5.0]]), geom_name='circle', r=1.0)
    # Trajectory passes through the circle
    result = tgt.check_trajectory_intersection(np.array([3.0, 5.0]),
                                                np.array([7.0, 5.0]))
    assert result[0], "should detect pass-through of circle"


def test_circle_miss():
    tgt = Targets(locs=np.array([[5.0, 5.0]]), geom_name='circle', r=1.0)
    # Trajectory passes above the circle (miss)
    result = tgt.check_trajectory_intersection(np.array([3.0, 6.5]),
                                                np.array([7.0, 6.5]))
    assert not result[0], "should not detect intersection for a miss"


def test_circle_graze():
    tgt = Targets(locs=np.array([[5.0, 5.0]]), geom_name='circle', r=1.0)
    # Trajectory just touches the circle boundary (tangent)
    result = tgt.check_trajectory_intersection(np.array([3.0, 6.0]),
                                                np.array([7.0, 6.0]))
    assert result[0], "should detect tangent intersection"


def test_circle_endpoint_inside():
    tgt = Targets(locs=np.array([[5.0, 5.0]]), geom_name='circle', r=1.0)
    # Start outside, end inside
    result = tgt.check_trajectory_intersection(np.array([3.0, 5.0]),
                                                np.array([5.0, 5.0]))
    assert result[0], "should detect endpoint-inside intersection"


def test_circle_multi_target():
    tgt = Targets(locs=np.array([[5.0, 5.0], [15.0, 5.0]]),
                  geom_name='circle', r=np.array([1.0, 1.0]))
    # Trajectory hits first target, misses second
    result = tgt.check_trajectory_intersection(np.array([3.0, 5.0]),
                                                np.array([7.0, 5.0]))
    assert result[0], "should hit first target"
    assert not result[1], "should miss second target"


def test_circle_zero_length_step():
    tgt = Targets(locs=np.array([[5.0, 5.0]]), geom_name='circle', r=1.0)
    result = tgt.check_trajectory_intersection(np.array([3.0, 5.0]),
                                                np.array([3.0, 5.0]))
    assert not result[0], "zero-length step should return False"


def test_delta_always_false():
    tgt = Targets(locs=np.array([[5.0, 5.0]]))
    result = tgt.check_trajectory_intersection(np.array([3.0, 5.0]),
                                                np.array([7.0, 5.0]))
    assert not result[0], "delta targets should always return False"


def test_capsule_passthrough():
    tgt = Targets(locs=np.array([[5.0, 5.0]]), geom_name='capsule',
                  l=4.0, w=2.0, theta=0.0)
    # Vertical trajectory through horizontal capsule
    result = tgt.check_trajectory_intersection(np.array([5.0, 0.0]),
                                                np.array([5.0, 10.0]))
    assert result[0], "should detect pass-through of capsule"


def test_capsule_miss():
    tgt = Targets(locs=np.array([[5.0, 5.0]]), geom_name='capsule',
                  l=4.0, w=2.0, theta=0.0)
    # Trajectory well above the capsule
    result = tgt.check_trajectory_intersection(np.array([0.0, 8.0]),
                                                np.array([10.0, 8.0]))
    assert not result[0], "should not detect intersection for a miss"


def test_capsule_endcap_hit():
    tgt = Targets(locs=np.array([[5.0, 5.0]]), geom_name='capsule',
                  l=4.0, w=2.0, theta=0.0)
    # Trajectory through the endcap (beyond spine endpoint)
    result = tgt.check_trajectory_intersection(np.array([7.5, 3.0]),
                                                np.array([7.5, 7.0]))
    assert result[0], "should detect pass-through of capsule endcap"


def test_capsule_zero_width():
    tgt = Targets(locs=np.array([[5.0, 5.0]]), geom_name='capsule',
                  l=4.0, w=0.0, theta=0.0)
    # Zero-width capsule: crossing trajectory should still be detected (dist=0=w/2)
    result = tgt.check_trajectory_intersection(np.array([5.0, 4.0]),
                                                np.array([5.0, 6.0]))
    assert result[0], "zero-width capsule crossing should be detected"
    # Parallel trajectory offset from spine should miss
    result = tgt.check_trajectory_intersection(np.array([3.0, 5.1]),
                                                np.array([7.0, 5.1]))
    assert not result[0], "trajectory parallel but offset from zero-width spine should miss"


def test_min_dist_segments_parallel():
    dist = Targets._min_dist_segments(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]),
        np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    assert abs(dist - 1.0) < 1e-12, f"parallel segments distance should be 1.0, got {dist}"


def test_min_dist_segments_crossing():
    dist = Targets._min_dist_segments(
        np.array([0.0, 0.0]), np.array([2.0, 2.0]),
        np.array([2.0, 0.0]), np.array([0.0, 2.0]))
    assert abs(dist) < 1e-12, f"crossing segments distance should be 0, got {dist}"


def test_min_dist_segments_endpoint():
    dist = Targets._min_dist_segments(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]),
        np.array([2.0, 0.0]), np.array([3.0, 0.0]))
    assert abs(dist - 1.0) < 1e-12, f"collinear separated segments distance should be 1.0, got {dist}"


def test_min_dist_segments_degenerate():
    # Both segments are points
    dist = Targets._min_dist_segments(
        np.array([0.0, 0.0]), np.array([0.0, 0.0]),
        np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert abs(dist - np.sqrt(2)) < 1e-12


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
