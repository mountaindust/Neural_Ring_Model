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

"""Perception: the observer's view of the scene, in neural coordinates.

``PerceptionModel`` turns a ``Targets`` scene plus an observer pose
(``focal_loc``, ``focal_angle``) into the two arrays the neural band consumes:
the neural angle of each visible target and its rho (neural group size).  It
owns the interval arithmetic that resolves occlusion, and it holds -- but does
not implement -- the two distribution roles:

  WARP   ``neural_angle_dist`` : egocentric angle -> neural angle
  WEIGHT ``angle_weight``      : density integrated over a visible arc -> rho

The families themselves, and all dispatch on family name, live in
``angle_distributions``; this class only decides which family fills which
role, caches its splines, and applies it.  Family functions are reached as
``ad.<name>`` rather than imported by value so that
``weighting_analysis/anti_foveal.py`` can register extra families by patching
the module.
"""

from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt

from . import angle_distributions as ad
from .targets import Targets


class _ReadOnlyParams(dict):
    """A read-only dict view of a PerceptionModel role's parameters.

    Reads behave exactly like a dict (indexing, iteration, ``==`` against a
    plain dict, ``dict(...)``); any attempt to mutate raises with a message
    pointing the user at the ``a_warp``/``b_warp`` / ``a_weight``/``b_weight``
    properties, which are the supported way to change parameters (they rebuild
    the affected splines). Subclasses ``dict`` so equality and read access are
    free; only the mutators are overridden.
    """
    _MSG = ("PerceptionModel parameter dicts are read-only; set parameters via "
            "the a_warp/b_warp (or a_weight/b_weight) properties, which rebuild "
            "the splines.")

    def __setitem__(self, *args):
        raise TypeError(self._MSG)

    def __delitem__(self, *args):
        raise TypeError(self._MSG)

    def update(self, *args, **kwargs):
        raise TypeError(self._MSG)

    def setdefault(self, *args):
        raise TypeError(self._MSG)

    def pop(self, *args):
        raise TypeError(self._MSG)

    def popitem(self):
        raise TypeError(self._MSG)

    def clear(self):
        raise TypeError(self._MSG)


class PerceptionModel:
    '''This class takes in a Targets object and a focal location and angle for an
    observer, and then translates that into a neural angular position and a neural 
    spin group size for each target based on the perceived angular extents of the 
    target, its signal strength, and a weighting function that describes the 
    density of neurons in the ring as a function of angle.
    
    Following mathematics conventions, egocentric angles increase counterclockwise;
    e.g., positive egocentric angles are to the left of the observer and 
    negative egocentric angles are to the right.'''

    # Memo installed by the signal_cache() context manager; None means
    # "no caching" (the normal state). A class attribute rather than an
    # instance one so objects unpickled from before this existed still read
    # None instead of raising.
    _signal_cache = None

    def __init__(self, targets=None, focal_loc=(5,10), focal_angle=0,
                 neural_angle_dist='lin_cutoff', angle_weight=None,
                 a_warp=None, b_warp=None, a_weight=None, b_weight=None,
                 theta_mesh=2000):
        '''Establishes an observer at location focal_loc, looking in a direction
        given by focal_angle, at targets given by the targets object.

        The perception model has two independent roles:

          1. WARPING (neural_angle_dist): a distribution that is integrated in a
             CDF-like fashion to produce the egocentric -> neural angle map.
          2. WEIGHTING (angle_weight): a distribution integrated over each
             target's visible angular extent to set rho (the relative neural
             group size / attractiveness driving gamma).

        The two roles were a single coupled function in earlier versions; they
        are now decoupled so warp shape and weight shape/parameters can be tuned
        independently. The default (angle_weight=None) gives uniform weighting.

        Mutable post-init: focal_loc, focal_angle, targets. Warp and weight
        parameters are changed by assigning the a_warp/b_warp/a_weight/b_weight
        properties (same slot names as the constructor args below), which
        rebuild only the affected role's splines automatically. The
        warp_params/weight_params attributes are read-only views of the current
        parameters (keyed by canonical name); assign the a_*/b_* properties to
        change them. The roles (neural_angle_dist, angle_weight) themselves are
        not mutable post-init.

        Parameters
        ----------
        targets : Targets object, optional.
            the targets around the observer as a Targets object. If no targets
            object is given, a default target object will be set.
        focal_loc : array-like of length 2
            (x,y) location of observer in Euclidean space. Will be stored as an
            ndarray.
        focal_angle : float
            direction observer is facing in Euclidean space from [-pi,pi).
        neural_angle_dist : {'cutoff', 'lin_cutoff', 'vonmises', 'symmetric_beta', 'reg_power', 'direct_power', None} (default = 'lin_cutoff')
            WARP role. A named distribution is integrated CDF-like to map
            egocentric angles to neural angles (denser neural representation
            where the distribution is concentrated). The families:
                - 'cutoff' : smooth cutoff, 1 in front and 0 in back with a
                             smooth transition. Params a (inner), b (outer);
                             0 <= a < b. See smooth_cutoff.
                - 'lin_cutoff' : trapezoidal cutoff -- the piecewise-linear
                             analog of 'cutoff' (1 on [-a, a], linear ramp to 0
                             on a < |theta| < b). Same params/support; closed-form
                             integral and inverse (no spline). See lin_cutoff.
                - 'vonmises' : von Mises pdf exp(k*cos(theta))/(2*pi*I0(k)).
                             Param k > 0 (larger k = narrower peak). See vonmises.
                - 'symmetric_beta' : Beta(alpha, alpha) rescaled to [-b, b].
                             Params alpha >= 1, b > 0 (alpha = 1 is uniform on
                             [-b, b]). See symmetric_beta.
                - 'reg_power' : regularized power weight 1/(|theta|^d + e),
                             d, e > 0. The normalized integral converges to
                             direct_power(theta, c=1-d) as e -> 0. See reg_power.
                - 'direct_power' : the power angle map directly (NOT a
                             CDF-integral of a density):
                             f(theta) = pi * sign(theta) * (|theta|/pi)^c, c > 0.
                             c = 1 is identity, c < 1 expands front angles, c > 1
                             compresses them. WARP ONLY -- not valid as a weight
                             (it is a signed angle map, not a density).
                - None : identity warp (neural angle = egocentric angle).
        angle_weight : {'cutoff', 'lin_cutoff', 'vonmises', 'symmetric_beta', 'reg_power', 'neural_angle_dist', None} (default = None)
            WEIGHT role. A named density family (same families as the warp,
            EXCEPT 'direct_power', which is disallowed) integrated over each
            target's visible arc to set rho. Plus:
                - 'neural_angle_dist' : tie the weight to the warp -- use the
                             same family and parameters as neural_angle_dist.
                             Reproduces the old full-weighting behavior.
                - None : uniform weight (rho from visible arc length / visible
                             count only). Reproduces the old weight_angle_only
                             behavior. This is the default.
        a_warp, b_warp : float, optional
            Parameters for the warp family, by slot. The real meaning of each
            slot depends on the family (see angle_distributions.FAMILY_INFO):
                cutoff/lin_cutoff: a_warp=a, b_warp=b;  vonmises: a_warp=k;
                symmetric_beta: a_warp=alpha, b_warp=b;
                reg_power: a_warp=d, b_warp=e;  direct_power: a_warp=c.
            If left None, the family default is used. Unused slots (e.g. b_warp
            for vonmises/direct_power) must be left None.
        a_weight, b_weight : float, optional
            Parameters for the weight family, same slot convention as the warp
            params. Must be None when angle_weight is None or
            'neural_angle_dist' (no independent weight family to parameterize).
        theta_mesh : float or 1D ndarray
            the number of equally spaced mesh points on [-pi,pi) to evaluate at
            or a mesh of theta values to evaluate at
        '''

        self.focal_loc = np.array(focal_loc, dtype=float)
        self.focal_angle = focal_angle

        # --- Resolve and validate the WARP role. ---
        if neural_angle_dist is not None and neural_angle_dist not in ad.FAMILY_INFO:
            raise ValueError(
                f"neural_angle_dist must be one of "
                f"{sorted(ad.FAMILY_INFO)} or None, got {neural_angle_dist!r}.")
        self.warp_name = neural_angle_dist
        self._warp_params = ad.resolve_params(
            neural_angle_dist, a_warp, b_warp, role='warp')

        # --- Resolve and validate the WEIGHT role. ---
        # 'direct_power' is a signed angle map, not a density: disallow.
        if angle_weight == 'direct_power':
            raise ValueError(
                "angle_weight='direct_power' is not allowed: the power map is a "
                "signed angle map, not a non-negative density, so it cannot be "
                "integrated over visible arcs as a weight. Use 'reg_power' "
                "(whose density is the power-map derivative) instead.")
        self._weight_tied_to_warp = (angle_weight == 'neural_angle_dist')
        if self._weight_tied_to_warp:
            if a_weight is not None or b_weight is not None:
                raise ValueError(
                    "a_weight/b_weight must be None when "
                    "angle_weight='neural_angle_dist' (the weight reuses the "
                    "warp family and parameters). Set the parameters via "
                    "a_warp/b_warp instead.")
            if self.warp_name == 'direct_power' or self.warp_name is None:
                raise ValueError(
                    "angle_weight='neural_angle_dist' requires neural_angle_dist "
                    f"to be a density family, got {self.warp_name!r}. "
                    "'direct_power' and None have no associated density to "
                    "weight with.")
            # Weight uses the warp family + params.
            self.weight_name = self.warp_name
            self._weight_params = dict(self._warp_params)
        elif angle_weight is None:
            if a_weight is not None or b_weight is not None:
                raise ValueError(
                    "a_weight/b_weight must be None when angle_weight is None "
                    "(uniform weighting has no parameters).")
            self.weight_name = None
            self._weight_params = {}
        else:
            if angle_weight not in ad.FAMILY_INFO:
                raise ValueError(
                    f"angle_weight must be one of "
                    f"{sorted(k for k in ad.FAMILY_INFO if k != 'direct_power')}, "
                    f"'neural_angle_dist', or None, got {angle_weight!r}.")
            self.weight_name = angle_weight
            self._weight_params = ad.resolve_params(
                angle_weight, a_weight, b_weight, role='weight')

        if targets is None:
            self.targets = Targets()
        else:
            assert isinstance(targets,Targets), "targets must be a Targets object."
            self.targets = targets
        if isinstance(theta_mesh, int):
            self.theta_mesh = np.linspace(-np.pi, np.pi, theta_mesh+1)[:-1]
        else:
            self.theta_mesh = theta_mesh

        # Build spline lookups for each role exactly once. The weight splines
        # are skipped when the weight is uniform or tied to the warp (the warp
        # antiderivative is reused in that case).
        self._build_warp_splines()
        self._build_weight_splines()


    # --- read-only views of the live parameter dicts ---

    @property
    def warp_params(self):
        """Read-only view of the warp family's current parameters (keyed by
        canonical name, e.g. {'k': 0.55}). To change a parameter, assign the
        a_warp/b_warp properties (they rebuild the splines)."""
        return _ReadOnlyParams(dict(self._warp_params))

    @warp_params.setter
    def warp_params(self, value):
        raise AttributeError(_ReadOnlyParams._MSG)

    @property
    def weight_params(self):
        """Read-only view of the weight family's current parameters. To change a
        parameter, assign the a_weight/b_weight properties (they rebuild the
        splines)."""
        return _ReadOnlyParams(dict(self._weight_params))

    @weight_params.setter
    def weight_params(self, value):
        raise AttributeError(_ReadOnlyParams._MSG)

    # --- assignable two-slot parameter properties (auto-respline on write) ---
    #
    # a_warp/b_warp/a_weight/b_weight map a role's generic slots onto its
    # family's canonical parameter (see angle_distributions.FAMILY_INFO and the constructor
    # docstring for the slot->name table). The getter is permissive (returns
    # None for an unused slot / identity warp / uniform weight); the setter is
    # strict and rebuilds the affected role's splines.

    @property
    def a_warp(self):
        return self._get_slot('warp', 0)

    @a_warp.setter
    def a_warp(self, value):
        self._set_slot('warp', 0, value)

    @property
    def b_warp(self):
        return self._get_slot('warp', 1)

    @b_warp.setter
    def b_warp(self, value):
        self._set_slot('warp', 1, value)

    @property
    def a_weight(self):
        return self._get_slot('weight', 0)

    @a_weight.setter
    def a_weight(self, value):
        self._set_slot('weight', 0, value)

    @property
    def b_weight(self):
        return self._get_slot('weight', 1)

    @b_weight.setter
    def b_weight(self, value):
        self._set_slot('weight', 1, value)

    def _role_state(self, role):
        """Return (name, params_dict) for 'warp' or 'weight'. For a tied weight
        the warp's name + live params are returned (the weight mirrors them)."""
        if role == 'warp':
            return self.warp_name, self._warp_params
        if self._weight_tied_to_warp:
            return self.warp_name, self._warp_params
        return self.weight_name, self._weight_params

    def _get_slot(self, role, slot_idx):
        """Permissive read of a role's slot value. Returns None when the role is
        identity/uniform (no family) or the slot is unused by the family."""
        name, params = self._role_state(role)
        if name is None:
            return None
        key = ad.FAMILY_INFO[name]['slots'][slot_idx]
        if key is None:
            return None
        return params.get(key)

    def _set_slot(self, role, slot_idx, value):
        """Strict write of a role's slot value, then rebuild that role's splines
        (mirroring + rebuilding the weight too when a tied warp is changed).
        Raises when there is no parameter to set for this slot."""
        slot_name = f'{"a" if slot_idx == 0 else "b"}_{role}'
        if role == 'weight' and self._weight_tied_to_warp:
            raise ValueError(
                f"cannot set {slot_name}: the weight is tied to the warp "
                "(angle_weight='neural_angle_dist'); set a_warp/b_warp instead.")
        name, params = self._role_state(role)
        if name is None:
            if role == 'warp':
                raise ValueError(
                    f"cannot set {slot_name}: neural_angle_dist is None "
                    "(identity warp has no parameters).")
            raise ValueError(
                f"cannot set {slot_name}: angle_weight is None "
                "(uniform weight has no parameters).")
        key = ad.FAMILY_INFO[name]['slots'][slot_idx]
        if key is None:
            raise ValueError(
                f"{slot_name} is not used by {name!r} "
                f"(it has a single parameter {ad.FAMILY_INFO[name]['slots'][0]!r}).")
        # Validate against a trial copy so a bad value leaves state unchanged.
        trial = dict(params)
        trial[key] = value
        ad.validate_params(name, trial, role)
        params[key] = value
        if role == 'warp':
            self._build_warp_splines()
            if self._weight_tied_to_warp:
                self._weight_params = dict(self._warp_params)
                self._build_weight_splines()
        else:
            self._build_weight_splines()


    def _build_warp_splines(self):
        """Build the warp role's forward+inverse integral splines (left None for
        analytic symmetric_beta, identity None, or the direct_power map)."""
        self._warp_forward_spline, self._warp_inverse_spline = \
            ad.make_integral_spline(self.warp_name, self._warp_params)

    def _build_weight_splines(self):
        """Build the weight role's forward integral spline (the antiderivative
        used by the rho arc-integral). Left None when the weight is uniform
        (weight_name is None), tied to the warp (the warp antiderivative is
        reused), or analytic (symmetric_beta)."""
        if self.weight_name is None or self._weight_tied_to_warp:
            self._weight_forward_spline = None
            return
        self._weight_forward_spline, _ = \
            ad.make_integral_spline(self.weight_name, self._weight_params)


    @staticmethod
    def _subtract_interval_pair(interval, hole):
        """Subtract a single non-wrapping hole from a single non-wrapping interval.

        Both interval and hole must satisfy lo <= hi (non-wrapping).

        Parameters
        ----------
        interval : (float, float)
            Non-wrapping interval [lo, hi] with lo <= hi.
        hole : (float, float)
            Non-wrapping hole [lo, hi] with lo <= hi.

        Returns
        -------
        list of (float, float)
            Remaining non-wrapping intervals after subtraction.
        """
        a, b = interval
        h_lo, h_hi = hole
        eps = 1e-14

        # Degenerate hole (zero width)
        if h_hi - h_lo <= eps:
            return [(a, b)]

        # No overlap
        if h_hi <= a or h_lo >= b:
            return [(a, b)]
        # Full overlap
        if h_lo <= a and h_hi >= b:
            return []
        # Left bite
        if h_lo <= a and h_hi < b:
            if b - h_hi > eps:
                return [(h_hi, b)]
            return []
        # Right bite
        if h_lo > a and h_hi >= b:
            if h_lo - a > eps:
                return [(a, h_lo)]
            return []
        # Middle bite: h_lo > a and h_hi < b
        result = []
        if h_lo - a > eps:
            result.append((a, h_lo))
        if b - h_hi > eps:
            result.append((h_hi, b))
        return result

    @staticmethod
    def _unwrap_interval(interval):
        """Decompose an angular interval into non-wrapping pieces.

        If lo <= hi, the interval is already non-wrapping and returned as-is.
        If lo > hi, the interval wraps around ±pi and is split into
        (lo, pi) and (-pi, hi).

        Parameters
        ----------
        interval : (float, float)
            Angular interval (lo, hi) on [-pi, pi].

        Returns
        -------
        list of (float, float)
            One or two non-wrapping intervals with lo <= hi.
        """
        lo, hi = interval
        if lo <= hi:
            return [(lo, hi)]
        else:
            pieces = []
            if np.pi - lo > 1e-14:
                pieces.append((lo, np.pi))
            if hi - (-np.pi) > 1e-14:
                pieces.append((-np.pi, hi))
            return pieces

    @staticmethod
    def _subtract_intervals_circle(intervals, hole):
        """Subtract an angular interval (hole) from a list of angular intervals
        on the circle [-pi, pi].

        Both the input intervals and the hole may wrap around ±pi (lo > hi).
        All returned intervals are non-wrapping (lo <= hi).

        Parameters
        ----------
        intervals : list of (float, float)
            Angular intervals on [-pi, pi]. Each (lo, hi) with lo <= hi is
            the arc [lo, hi]. If lo > hi, the interval wraps around ±pi.
        hole : (float, float)
            The angular interval to subtract. May wrap around ±pi.

        Returns
        -------
        list of (float, float)
            Remaining non-wrapping intervals after subtraction.
        """
        # Decompose all input intervals into non-wrapping pieces
        unwrapped = []
        for iv in intervals:
            unwrapped.extend(PerceptionModel._unwrap_interval(iv))

        # Decompose the hole into non-wrapping pieces
        hole_pieces = PerceptionModel._unwrap_interval(hole)

        # Subtract each hole piece from each interval piece
        current = unwrapped
        for hp in hole_pieces:
            next_intervals = []
            for iv in current:
                next_intervals.extend(
                    PerceptionModel._subtract_interval_pair(iv, hp))
            current = next_intervals

        return current


    def _integrate_neural_weight(self, intervals):
        """Integrate the neural weight function over a union of angular intervals.

        Computes the exact integral of get_neural_weight(theta) over the
        given intervals. For the 'cutoff' weight, this uses the existing
        smooth_cutoff_integral antiderivative. For uniform weight (None),
        the integral is simply the total arc length.

        Parameters
        ----------
        intervals : list of (float, float)
            Non-wrapping intervals [lo, hi] with lo <= hi, all in [-pi, pi].

        Returns
        -------
        float
            The integral value. A shared constant factor (from the
            smooth_cutoff_integral normalization) is present but cancels
            in rho = G / G.sum(), so the result is suitable for relative
            group size computation.
        """
        if not intervals:
            return 0.0

        name = self.weight_name
        if name is None:
            # Uniform weight: integral is arc length.
            return sum(hi - lo for lo, hi in intervals)

        # Weighted: integrate the weight density via its CDF-like antiderivative
        # F. The constant normalization factor in F cancels in rho = G/G.sum(),
        # so F(hi) - F(lo) is the relevant quantity. When the weight is tied to
        # the warp, reuse the warp's forward spline (no separate weight spline
        # was built); otherwise use the weight's own forward spline.
        if self._weight_tied_to_warp:
            fwd = self._warp_forward_spline
        else:
            fwd = self._weight_forward_spline
        F = lambda x: ad.eval_forward_map(name, self._weight_params, fwd, x)
        return sum(F(hi) - F(lo) for lo, hi in intervals)


    def get_neural_weight(self, theta):
        '''Returns the neural weight density for given angles theta based on the
        weighting function (angle_weight). This is a proxy for the relative
        attractiveness / neural group size contributed per unit visible arc, and
        (for front-biased families) weights things in front more highly than in
        back. Returns ones for uniform weighting (angle_weight=None).

        Parameters
        ----------
        theta : float or 1D ndarray
            angle(s) to evaluate the neural weight at

        Returns
        -------
        neural weight(s) corresponding to input theta value(s)
        '''

        return ad.eval_density(self.weight_name, self._weight_params, theta)


    def get_neural_angle(self, theta):
        '''Returns the neural position for a given angle theta based on the
        warp (neural_angle_dist). This maps the perceived center of each target
        to the neural position of the corresponding spin group, via the CDF-like
        integral of a density family, the direct power map, or identity.

        Parameters
        ----------
        theta : float or 1D ndarray
            angle(s) to evaluate the neural position transformation at

        Returns
        -------
        neural position(s) corresponding to input theta value(s)
        '''

        name = self.warp_name
        if name is None:
            return theta
        elif name == 'direct_power':
            return ad.direct_power(theta, self._warp_params['c'])
        else:
            return ad.eval_forward_map(
                name, self._warp_params, self._warp_forward_spline, theta)


    def get_neural_angle_inverse(self, theta):
        '''Returns the angle corresponding to a given neural position theta,
        the inverse of the warp (neural_angle_dist). Maps neural position back to
        the perceived center of each target.

        Parameters
        ----------
        theta : float or 1D ndarray
            neural position(s) to evaluate the inverse transformation at

        Returns
        -------
        angle(s) corresponding to input neural position value(s)
        '''

        name = self.warp_name
        if name is None:
            return theta
        elif name == 'direct_power':
            return ad.direct_power_inverse(theta, self._warp_params['c'])
        else:
            return ad.eval_inverse_map(
                name, self._warp_params, self._warp_inverse_spline, theta)


    def _get_target_signals(self, focal_angle=None, focal_loc=None, mesh_signal=False):
        '''Returns the egocentric angular location of the center of each VISIBLE
        target (closer targets that are not delta functions block ones behind) as
        a length N array, and a normalized neural group size (rho) for each.

        For circle (and capsule) targets, blocking and neural group sizes are
        computed using exact interval arithmetic rather than a discrete mesh.

        If mesh_signal is True, instead returns the neural weighting
        (including attractiveness) for each target evaluated on the theta mesh,
        as an Nxlen(theta_mesh) array. This is used for plotting the perception
        signal (see plot_blocked_signals).

        If all targets are fully blocked or have zero neural weight, returns
        empty arrays (length-0 c_angles and rho).

        Parameters
        ----------
        focal_angle : float, optional
            the focal angle for egocentric perception. If None, uses the object's
            focal_angle attribute.
        focal_loc : array-like, optional
            the (x,y) focal location for egocentric perception. If None, uses the
            object's focal_loc attribute.
        mesh_signal : bool
            if True, return the neural weighting for each target on the theta
            mesh (for plotting). Otherwise return integrated group sizes (rho).

        Returns
        -------
        c_angles : length N ndarray
            angles to the visual centers of visible targets
        rho : length N ndarray (mesh_signal=False)
            normalized neural group size for each visible target, sums to 1.
        signals : Nxlen(theta_mesh) ndarray (mesh_signal=True)
            neural weighting times attractiveness on the theta mesh.
        '''

        if focal_angle is None:
            focal_angle = self.focal_angle
        if focal_loc is None:
            focal_loc = self.focal_loc
        else:
            focal_loc = np.array(focal_loc, dtype=float)

        dists = self.targets.get_dist_to_targets(focal_loc)
        c_angles = self.targets.get_angles_to_targets(focal_loc, focal_angle)
        angles = self.targets.get_percep_angles(focal_loc, focal_angle)

        if self.targets.geom_name is None:
            ##### Delta function targets (no blocking) #####
            # Each target is a single angle; evaluate neural weight there directly.
            s_values = self.targets.values
            G = self.get_neural_weight(angles) * s_values

            if mesh_signal:
                theta_supp = np.zeros((angles.shape[0], self.theta_mesh.size))
                for n, theta in enumerate(angles):
                    idx = np.searchsorted(self.theta_mesh, theta)
                    if idx == len(self.theta_mesh):
                        idx = 0
                    elif idx != 0 and \
                    theta-self.theta_mesh[idx-1] < self.theta_mesh[idx]-theta:
                        idx = idx - 1
                    theta_supp[n, idx] = 1
                weighted_signals = theta_supp*self.get_neural_weight(self.theta_mesh)
                return c_angles, weighted_signals*s_values[:, np.newaxis]
            else:
                G_total = G.sum()
                if G_total == 0:
                    return np.array([]), np.array([])
                return c_angles, G/G_total

        elif self.targets.geom_name == 'circle' or self.targets.geom_name == 'capsule':
            ##### Extended targets: exact interval arithmetic #####
            # Sort by closest-point distance (closest first for blocking).
            # For circles this is exact. For capsules, two capsules can
            # mutually occlude each other at different angles (the one with
            # the farther closest-point can still cross in front at some
            # angles). A fully correct solution would require per-angle
            # depth comparison; this closest-point sort is a practical
            # approximation.
            arg_srt = dists.argsort()
            angles_sorted = angles[arg_srt]
            c_angles_sorted = c_angles[arg_srt]

            # Build visible intervals for each target using interval subtraction.
            # Each target starts with its full angular extent, then all closer
            # targets' original extents are subtracted (they block it).
            num_targets = len(arg_srt)
            # original_extents[n] = (lo, hi) as returned by get_percep_angles
            original_extents = [(float(angles_sorted[n, 0]),
                                 float(angles_sorted[n, 1]))
                                for n in range(num_targets)]
            visible_intervals = []  # list of lists of (lo, hi) tuples
            for n in range(num_targets):
                # Unwrap up front: get_percep_angles encodes an extent that
                # straddles +-pi as a wrapping pair (lo > hi), but both
                # _integrate_neural_weight and the mesh_signal masking below
                # require non-wrapping pieces. _subtract_intervals_circle
                # unwraps its inputs, so targets with a closer blocker were
                # already safe -- but the CLOSEST target (n == 0) never enters
                # that loop, and a raw wrapping pair reaching the integrator
                # yields a negative arc length, which the G > 0 visibility
                # filter then silently discards. That dropped the nearest
                # target for the whole angular window in which it straddles
                # the rear branch cut.
                intervals = self._unwrap_interval(original_extents[n])
                for closer in range(n):
                    intervals = self._subtract_intervals_circle(
                        intervals, original_extents[closer])
                    if not intervals:
                        break
                visible_intervals.append(intervals)

            # Compute neural group sizes by integrating neural weight over
            # visible intervals, weighted by target attractiveness.
            G_sorted = np.empty(num_targets)
            for n in range(num_targets):
                G_sorted[n] = self._integrate_neural_weight(
                    visible_intervals[n]) * self.targets.values[arg_srt[n]]

            # Undo sorting to restore original target order
            inv_arg_srt = np.empty_like(arg_srt)
            inv_arg_srt[arg_srt] = np.arange(num_targets)
            G = G_sorted[inv_arg_srt]
            c_angles = c_angles_sorted[inv_arg_srt]
            visible_intervals_orig = [visible_intervals[inv_arg_srt[n]]
                                      for n in range(num_targets)]

            # Remove completely blocked targets (G == 0)
            vis = G > 0
            c_angles = c_angles[vis]
            s_values = self.targets.values[vis]
            G = G[vis]
            visible_intervals_vis = [visible_intervals_orig[n]
                                     for n in range(num_targets) if vis[n]]

            if mesh_signal:
                # Build mesh representation from exact intervals for plotting
                theta_supp = np.zeros((len(c_angles), self.theta_mesh.size))
                for n, ivs in enumerate(visible_intervals_vis):
                    for lo, hi in ivs:
                        mask = (self.theta_mesh >= lo) & (self.theta_mesh <= hi)
                        theta_supp[n, mask] = 1
                weighted_signals = theta_supp*self.get_neural_weight(self.theta_mesh)
                return c_angles, weighted_signals*s_values[:, np.newaxis]
            else:
                G_total = G.sum()
                if G_total == 0:
                    return np.array([]), np.array([])
                return c_angles, G/G_total

        else:
            raise NotImplementedError("Unknown target geometry name.")


    def plot_neural_weight(self, polar=False, ax=None, wb_plot=False):
        '''Plot the neural weighting function on [-pi, pi], with the max weight
        normalized to 1. For non-polar plots, also overlays the physical-to-
        neural angle mapping (theta -> hat{theta}) on a twin y-axis (radians),
        using a dashed line in the same color as the corresponding weight
        curve.

        Parameters
        ----------
        polar : bool, optional
            If True, use a polar plot. Default False (linear plot). The angle
            mapping overlay is only drawn for non-polar plots.
        ax : matplotlib axis, optional
            If provided, the curve is added to this axis (no figure created,
            no plt.show). For polar=True, the axis must have been created
            with projection='polar'. If None, a new figure and axis are
            created and plt.show() is called. When plotting multiple models
            on the same non-polar axis, a single twin axis is reused so all
            angle-mapping curves share one right-hand y-axis.
        wb_plot : bool
            whether or not plotting in a Jupyter notebook

        Returns
        -------
        ax : matplotlib axis, if axis was provided as an argument.
            Otherwise, None.
        '''

        theta = np.linspace(-np.pi, np.pi, 361)
        weights = self.get_neural_weight(theta)
        weights = weights/weights.max()

        if ax is None:
            local_plot = True
            if wb_plot:
                fig = plt.figure(figsize=(8, 5))
            else:
                fig = plt.figure(figsize=(6, 5))
            if polar:
                ax = plt.subplot(1, 1, 1, projection='polar')
            else:
                ax = plt.subplot(1, 1, 1)
        else:
            local_plot = False

        weight_label = f'{self.weight_name or "uniform"} weight'
        weight_line, = ax.plot(theta, weights, label=weight_label)

        if polar:
            # Match the tick scheme used in plot_blocked_signals, but with
            # labels in [-pi, pi] on the back half so it mirrors the data range.
            angles_deg = np.linspace(0, 360, 8, endpoint=False)
            labels = [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',
                      r'$\frac{3\pi}{4}$', r'$\pm\pi$', r'$-\frac{3\pi}{4}$',
                      r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$']
            ax.set_thetagrids(angles_deg, labels)
            if local_plot:
                ax.set_title('Neural Weighting Function')
                ax.legend()
                plt.show()
            else:
                return ax
            return

        tick_locs = np.array([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0,
                              np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        tick_labels = [r'$-\pi$', r'$-\frac{3\pi}{4}$', r'$-\frac{\pi}{2}$',
                       r'$-\frac{\pi}{4}$', r'$0$', r'$\frac{\pi}{4}$',
                       r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel('Neural weight')
        ax.set_xlim(-np.pi, np.pi)

        # Reuse a previously-attached twin so multiple models share one
        # right-hand axis when plotted on the same ax.
        ax2 = getattr(ax, '_neural_angle_twin', None)
        if ax2 is None:
            ax2 = ax.twinx()
            ax._neural_angle_twin = ax2
            ax2.set_ylabel(r'$\hat{\theta}$')
            ax2.set_yticks(tick_locs)
            ax2.set_yticklabels(tick_labels)
            ax2.set_ylim(-np.pi, np.pi)

        neural_theta = self.get_neural_angle(theta)
        angle_label = f'{self.warp_name or "identity"} angle map'
        ax2.plot(theta, neural_theta, color=weight_line.get_color(),
                 linestyle='--', label=angle_label)

        # Build a combined legend so callers do not need to know about the
        # twin axis. Calling this each time means the legend stays in sync
        # as additional models are added to the same ax.
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left')

        if local_plot:
            ax.set_title('Neural Weighting Function')
            plt.show()
        else:
            return ax


    def plot_blocked_signals(self, wb_plot=False, ax=None):
        '''Plots visible targets, their angular direction from the observer,
        their associated neural angles, and also the signal distribution from
        the point of view of the observer normalized so that the maximum signal
        strength is 1.

        Use as a test for _get_target_signals and get_neural_angle.

        Set wb_plot to True if plotting in a Jupyber notebook

        Parameters
        ----------
        wb_plot : bool
            whether or not plotting in a Jupyter notebook
        ax : matplotlib axis, optional
            If provided, only the target-geometry panel (targets plus the
            black perception lines and red dashed neural-direction lines)
            is drawn onto this axis; the polar perception-signal panel is
            skipped and no figure is created or shown. If None (default),
            a new figure is created with both the target-geometry and the
            polar perception-signal subplots, and plt.show() is called.

        Returns
        -------
        ax : matplotlib axis, if ax was provided as an argument.
            Otherwise, None.
        '''

        vis_angles, signals = self._get_target_signals(mesh_signal=True)
        neur_angles = self.get_neural_angle(vis_angles)

        if ax is None:
            local_plot = True
            if wb_plot:
                plt.figure(figsize=(8,5))
            else:
                plt.figure(figsize=(12,6))
            ax1 = plt.subplot(121)
        else:
            local_plot = False
            ax1 = ax

        ###### Target Geometry Plot ######
        # First, plot the targets themselves
        self.targets.plot_targets_to_axis(ax1)

        # Now plot perception lines. Requires adding back angle of focal locust
        #   to get allocentric angles.
        neur_angles_allo = neur_angles + self.focal_angle
        for n, theta in enumerate(vis_angles+self.focal_angle):
            r = self.targets.get_dist_to_targets(self.focal_loc)[n]
            x = (self.focal_loc[0],self.focal_loc[0] + r*np.cos(theta))
            y = (self.focal_loc[1],self.focal_loc[1] + r*np.sin(theta))
            ax1.plot(x,y,'k')
            x_neur = (self.focal_loc[0],self.focal_loc[0] + r*np.cos(neur_angles_allo[n]))
            y_neur = (self.focal_loc[1],self.focal_loc[1] + r*np.sin(neur_angles_allo[n]))
            ax1.plot(x_neur, y_neur, 'r--', alpha=0.5)
        ax1.arrow(self.focal_loc[0],self.focal_loc[1],
                  0.5*np.cos(self.focal_angle),0.5*np.sin(self.focal_angle),
                width=0.1, head_length=0.25)
        ax1.set_aspect('equal')
        ax1.set_title('Target Geometry and\n Neural Directions')

        if not local_plot:
            return ax1

        ###### Perception Signal Plot ######
        ax2 = plt.subplot(122, projection='polar')

        if signals.max() == 0:
            p_func = signals.sum(axis=0)
        else:
            p_func = signals.sum(axis=0)/signals.max()

        ax2.plot(self.theta_mesh,p_func)
        ax2.arrow(0,-0.5,0,0.25, width=0.2, head_length=0.15)
        ax2.set_rmin(-0.5)
        ax2.set_rmax(1)
        ax2.set_rlabel_position(10)
        ax2.set_rticks([0, 0.5, 1])
        # Define positions in degrees (0 to 360)
        angles_deg = np.linspace(0, 360, 8, endpoint=False)
        # Define corresponding labels in radians (0 to 2π)
        labels = [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
                r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']

        ax2.set_thetagrids(angles_deg, labels)
        ax2.set_title('Perception Signal')

        plt.show()


    @contextmanager
    def signal_cache(self):
        '''Memoize get_neural_signals on (focal_angle, focal_loc) within a block.

        The perception signals are a pure function of the observer state
        (heading, location) and the perception geometry, so a root finder that
        evaluates the model repeatedly at states it has already visited pays
        for identical interval arithmetic and spline evaluations again and
        again. Inside this context manager each distinct state is computed once
        and reused; outside it (the default) nothing is cached, and no caller
        that does not opt in is affected.

        The cache lives only for the duration of the block, so it cannot go
        stale against a later warp/weight/target change. Nesting is safe: an
        inner block reuses the cache an outer block already installed. The
        cached arrays are shared between callers, so treat them as read-only.

        The key is the *exact* float state, deliberately not a tolerance: the
        Jacobians probe theta +- h (h = 1e-6 in _coupled_jacobian, 1e-7 in
        _self_consistent_jac), and a key that merged those probes with the base
        point would hand back the same signals for all three. Since perception
        is the ONLY theta dependence in dgamma_dt, that would zero the theta
        column of the coupled Jacobian, making det(J) = 0 and reporting every
        equilibrium unstable.
        '''
        if self.__dict__.get('_signal_cache') is not None:
            yield                      # an enclosing block already owns one
            return
        self._signal_cache = {}
        try:
            yield
        finally:
            del self._signal_cache     # falls back to the class attribute None


    def get_neural_signals(self, focal_angle=None, focal_loc=None):
        '''Returns the neural angles and neural group sizes for each target based 
        on the perceived angles and the neural position transformation function. 
        This is a mapping from perceived center of each target to the neural 
        position of the corresponding spin group.

        Parameters
        ----------
        focal_angle : float, optional
            the focal angle for egocentric perception. If None, uses the object's 
            focal_angle attribute.
        focal_loc : array-like, optional
            the (x,y) focal location for egocentric perception. If None, uses the 
            object's focal_loc attribute.

        Returns
        -------
        neural_angles : length N ndarray
            neural angles corresponding to visible targets
        rho : length N ndarray
            normalized neural group size for each visible target

        See Also
        --------
        signal_cache : context manager that memoizes this call on the observer
            state, for solvers that revisit the same states many times.
        '''

        if focal_angle is None:
            focal_angle = self.focal_angle
        if focal_loc is None:
            focal_loc = self.focal_loc

        # Inside a signal_cache() block, hand back the arrays already computed
        # for this exact observer state (see signal_cache for the exact-key
        # requirement).
        cache = self._signal_cache
        if cache is not None:
            key = (float(focal_angle), float(focal_loc[0]), float(focal_loc[1]))
            cached = cache.get(key)
            if cached is not None:
                return cached

        angles, rho = self._get_target_signals(focal_angle=focal_angle,
                                              focal_loc=focal_loc)
        if angles.size == 0:
            signals = (angles, rho)
        else:
            signals = (self.get_neural_angle(angles), rho)

        if cache is not None:
            cache[key] = signals
        return signals
