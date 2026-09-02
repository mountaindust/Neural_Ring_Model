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

"""Angle wrapping and arc helpers shared across the model.

These are frame-agnostic: they operate on plain angles and know nothing about
allocentric / egocentric / neural coordinates. See the coordinate-system
section of CLAUDE.md for which frame a given caller is working in.
"""

import numpy as np


def convert_angles(theta):
    '''Given a scalar or array of angles, convert to angles in the closed
    interval [-np.pi, np.pi].

    The map is odd: convert_angles(-x) == -convert_angles(x) for every x. At
    the endpoints this means +pi and -pi are both fixed points, so the interval
    is closed rather than half-open: they are two representations of the same
    (facing-away) direction, and which one is returned is inherited from the
    sign of the argument.

    That sign is meaningful. Facing-away is the branch cut of the heading
    torque, where dtheta/dt jumps between +K*R and -K*R; carrying the argument's
    sign through lets the caller's approach direction select the branch, and
    makes the map commute with the mirror theta -> -theta. It is the same
    convention as np.angle, which selects the branch by the sign of the
    imaginary part (including IEEE signed zero).
    '''
    wrapped = theta - (theta+np.pi)//(2*np.pi)*2*np.pi
    # The floor division lands the +pi endpoint on -pi; take the sign from theta.
    flip = (wrapped == -np.pi) & (theta > 0)
    if np.ndim(wrapped) == 0:
        return np.pi if flip else wrapped
    wrapped = np.array(wrapped, copy=True)
    wrapped[flip] = np.pi
    return wrapped


def _smallest_enclosing_arc(angles):
    """Return (lo, hi) for the shortest arc on [-pi, pi] containing all angles.

    The arc goes counter-clockwise from lo to hi. If lo > hi, the arc wraps
    around ±pi. Input angles must be in [-pi, pi].

    Parameters
    ----------
    angles : 1D array
        Angles in [-pi, pi].

    Returns
    -------
    (lo, hi) : tuple of float
    """
    s = np.sort(angles % (2*np.pi))  # sort on [0, 2pi)
    n = len(s)
    # Compute the gap between consecutive sorted angles (including wrap-around)
    gaps = np.empty(n)
    gaps[:-1] = s[1:] - s[:-1]
    gaps[-1] = s[0] + 2*np.pi - s[-1]
    # The largest gap is the one NOT covered by the arc.
    # The arc starts just after the largest gap and ends just before it.
    k = np.argmax(gaps)
    # lo is the angle just after the largest gap, hi is the angle just before it
    lo = s[(k + 1) % n]
    hi = s[k]
    return (convert_angles(lo), convert_angles(hi))
