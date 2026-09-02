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

"""The warp / weight distribution family library.

Every angular distribution family the model can use lives here, together with
the whole name-based dispatch layer. ``PerceptionModel`` uses these families
in two independent roles (see CLAUDE.md):

  WARP   -- the density is integrated CDF-like to map egocentric angles to
            neural angles (``neural_angle_dist``).
  WEIGHT -- the density is integrated over each target's visible arc to set
            its rho (``angle_weight``).

Each family supplies up to three functions: the density ``f(theta, *params)``,
its antiderivative ``f_integral`` normalized so that ``F(+-pi) = +-pi``, and
the inverse ``f_int_inverse``. ``direct_power`` is the exception -- it is a
direct angle map rather than a CDF integral, so it has only the map and its
inverse, and is warp-only.

ADDING A FAMILY
---------------
Everything a new family touches is in this file:

  1. ``FAMILY_INFO``          -- slot names and defaults for the generic
                                 two-slot constructor kwargs (a_warp/b_warp,
                                 a_weight/b_weight).
  2. ``validate_params``      -- its parameter constraints.
  3. ``eval_density``         -- the density, for the WEIGHT role.
  4. ``eval_forward_map``     -- the CDF-like angle map, for the WARP role.
  5. ``eval_inverse_map``     -- its inverse.
  6. ``make_integral_spline`` -- only if the family needs a spline; families
                                 with closed-form integrals fall through its
                                 final ``else`` and get ``(None, None)``,
                                 which is correct for them.

Then write the family's own functions in the implementations section below.
``weighting_analysis/anti_foveal.py`` registers two retired families from
outside by patching exactly these module attributes, which is why
``PerceptionModel`` reaches them as ``angle_distributions.<name>`` rather than
importing them by value.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
from scipy.special import i0
from scipy.stats import vonmises as vonmises_dist
from scipy.stats import beta as beta_dist


# Per-family metadata for PerceptionModel's two roles (warp + weight). Maps the
# generic two-slot constructor params (a_warp/b_warp, a_weight/b_weight) onto
# each distribution family's real parameter names, with defaults. 'slots' gives
# the real name for the (a_*, b_*) slots; None means that slot is unused by the
# family. The same families serve as warp (CDF-integrated angle map) and as
# weight (rho attractiveness), except 'direct_power' which is warp-only.
FAMILY_INFO = {
    'cutoff':         {'slots': ('a', 'b'),     'defaults': {'a': np.pi/3, 'b': 4*np.pi/5}},
    'lin_cutoff':     {'slots': ('a', 'b'),     'defaults': {'a': np.pi/3, 'b': 4*np.pi/5}},
    'vonmises':       {'slots': ('k', None),    'defaults': {'k': 1.0}},
    'symmetric_beta': {'slots': ('alpha', 'b'), 'defaults': {'alpha': 5.0, 'b': np.pi}},
    'reg_power':      {'slots': ('d', 'e'),     'defaults': {'d': 0.5, 'e': 1e-3}},
    'direct_power':   {'slots': ('c', None),    'defaults': {'c': 0.5}},
}


# ------------------------------------------------------------------
# Dispatch layer: everything PerceptionModel reaches by family name.
# ------------------------------------------------------------------

def resolve_params(name, a_slot, b_slot, role):
    """Map the generic (a_slot, b_slot) constructor params onto a family's
    real parameter dict, filling family defaults for any slot left None and
    validating per-family constraints.

    Parameters
    ----------
    name : str or None
        family name (key of FAMILY_INFO) or None (identity/uniform).
    a_slot, b_slot : float or None
        the generic slot values (a_warp/b_warp or a_weight/b_weight).
    role : {'warp', 'weight'}
        used only for clearer error messages.

    Returns
    -------
    dict : real parameter names -> values (empty dict for name=None).
    """
    if name is None:
        if a_slot is not None or b_slot is not None:
            raise ValueError(
                f"a_{role}/b_{role} must be None when the {role} family is "
                "None (identity warp / uniform weight has no parameters).")
        return {}

    info = FAMILY_INFO[name]
    a_key, b_key = info['slots']
    params = dict(info['defaults'])

    if a_slot is not None:
        if a_key is None:
            raise ValueError(
                f"a_{role} is not used by {name!r}; leave it None.")
        params[a_key] = a_slot
    if b_slot is not None:
        if b_key is None:
            raise ValueError(
                f"b_{role} is not used by {name!r} (it has a single "
                f"parameter {a_key!r}); leave b_{role} None.")
        params[b_key] = b_slot

    validate_params(name, params, role)
    return params


def validate_params(name, params, role):
    """Validate a family's resolved parameter dict, naming the real
    parameter (and the generic slot) in any error message."""
    if name in ('cutoff', 'lin_cutoff'):
        a, b = params['a'], params['b']
        if not (0 <= a < b):
            raise ValueError(
                f"for {name} {role}, a_{role} (a) and b_{role} (b) must "
                f"satisfy 0 <= a < b (got a={a}, b={b}).")
    elif name == 'vonmises':
        if not (params['k'] > 0):
            raise ValueError(
                f"for vonmises {role}, a_{role} (k) must be > 0 "
                f"(got k={params['k']}).")
    elif name == 'symmetric_beta':
        if not (params['alpha'] >= 1):
            raise ValueError(
                f"for symmetric_beta {role}, a_{role} (alpha) must be >= 1 "
                f"(got alpha={params['alpha']}).")
        if not (params['b'] > 0):
            raise ValueError(
                f"for symmetric_beta {role}, b_{role} (b) must be > 0 "
                f"(got b={params['b']}).")
    elif name == 'reg_power':
        if not (params['d'] > 0):
            raise ValueError(
                f"for reg_power {role}, a_{role} (d) must be > 0 "
                f"(got d={params['d']}).")
        if not (params['e'] > 0):
            raise ValueError(
                f"for reg_power {role}, b_{role} (e) must be > 0 "
                f"(got e={params['e']}).")
    elif name == 'direct_power':
        if not (params['c'] > 0):
            raise ValueError(
                f"for direct_power {role}, a_{role} (c) must be > 0 "
                f"(got c={params['c']}).")


def eval_density(name, params, theta):
    """Evaluate a family's density at ``theta``, dispatching on family name.

    This is the WEIGHT role: the density integrated over each target's visible
    arc to set rho. ``None`` is uniform weight and returns ones.

    Parameters
    ----------
    name : str or None
        family name (key of FAMILY_INFO), or None for uniform weight.
    params : dict
        that family's canonical parameters.
    theta : float or 1D ndarray
        angle(s) to evaluate at.

    Returns
    -------
    density value(s) corresponding to input theta value(s)
    """

    if name is None:
        return np.ones_like(theta)
    elif name == 'cutoff':
        return smooth_cutoff(theta, params['a'], params['b'])
    elif name == 'lin_cutoff':
        return lin_cutoff(theta, params['a'], params['b'])
    elif name == 'vonmises':
        return vonmises(theta, params['k'])
    elif name == 'symmetric_beta':
        return symmetric_beta(theta, params['alpha'], params['b'])
    elif name == 'reg_power':
        return reg_power(theta, params['d'], params['e'])
    else:
        raise NotImplementedError(
            f"Unknown neural weight family {name!r}.")


def eval_forward_map(name, params, fwd_spline, theta):
    """Forward CDF-like angle map F(theta) for a density family, saturating
    to +-pi outside the support. Uses the supplied precomputed forward
    spline for cutoff/vonmises/reg_power; analytic scipy cdf for
    symmetric_beta. (The same evaluator serves the warp and, as an
    antiderivative for the rho arc-integral, the weight.)"""
    if name == 'cutoff':
        theta_arr = np.asarray(theta, dtype=float)
        scalar = theta_arr.ndim == 0
        b = params['b']
        clamped = np.clip(theta_arr, -b, b)
        result = fwd_spline(clamped)
        result = np.where(theta_arr >= b, np.pi, result)
        result = np.where(theta_arr <= -b, -np.pi, result)
        return float(result) if scalar else result
    elif name == 'lin_cutoff':
        return lin_cutoff_integral(
            theta, params['a'], params['b'])
    elif name == 'vonmises' or name == 'reg_power':
        theta_arr = np.asarray(theta, dtype=float)
        scalar = theta_arr.ndim == 0
        clamped = np.clip(theta_arr, -np.pi, np.pi)
        result = fwd_spline(clamped)
        result = np.where(theta_arr >= np.pi, np.pi, result)
        result = np.where(theta_arr <= -np.pi, -np.pi, result)
        return float(result) if scalar else result
    elif name == 'symmetric_beta':
        return symmetric_beta_integral(
            theta, params['alpha'], params['b'])
    else:
        raise NotImplementedError(
            f"no forward integral map for family {name!r}.")


def eval_inverse_map(name, params, inv_spline, y):
    """Inverse of eval_forward_map. Domain y in [-pi, pi]."""
    if name == 'cutoff':
        y_arr = np.asarray(y, dtype=float)
        scalar = y_arr.ndim == 0
        if np.any((y_arr < -np.pi) | (y_arr > np.pi)):
            raise ValueError("y must satisfy -pi <= y <= pi.")
        b = params['b']
        result = inv_spline(y_arr)
        result = np.where(y_arr == np.pi, b, result)
        result = np.where(y_arr == -np.pi, -b, result)
        return float(result) if scalar else result
    elif name == 'lin_cutoff':
        return lin_cutoff_int_inverse(
            y, params['a'], params['b'])
    elif name == 'vonmises' or name == 'reg_power':
        y_arr = np.asarray(y, dtype=float)
        scalar = y_arr.ndim == 0
        if np.any((y_arr < -np.pi) | (y_arr > np.pi)):
            raise ValueError("y must satisfy -pi <= y <= pi.")
        result = inv_spline(y_arr)
        result = np.where(y_arr == np.pi, np.pi, result)
        result = np.where(y_arr == -np.pi, -np.pi, result)
        return float(result) if scalar else result
    elif name == 'symmetric_beta':
        return symmetric_beta_int_inverse(
            y, params['alpha'], params['b'])
    else:
        raise NotImplementedError(
            f"no inverse integral map for family {name!r}.")


def make_integral_spline(name, params):
    """Build (forward, inverse) CubicSplines for the CDF-like integral map
    of a density family, or (None, None) for families with no spline
    (lin_cutoff and symmetric_beta are evaluated analytically;
    direct_power / None have no integral map).

    Used by both roles: the warp uses both returned splines; the weight
    uses only the forward spline (as an antiderivative for the rho
    arc-integral). The per-family node construction is preserved verbatim
    from the original single-spline builder to protect numerics: cutoff
    uses a saturated-tail monotone filter; reg_power uses a cubic node mesh
    with a monotonicity assert; vonmises uses plain equispaced nodes via
    scipy's cdf.
    """
    if name == 'cutoff':
        a = params['a']
        b = params['b']
        n_nodes = 2001
        x_nodes = np.linspace(-b, b, n_nodes)
        # Snap the center node to 0 exactly (should already hold for an
        # odd number of equispaced nodes but avoids floating-point drift).
        center = n_nodes // 2
        x_nodes[center] = 0.0
        y_nodes = np.empty(n_nodes)
        for i, x in enumerate(x_nodes):
            y_nodes[i] = smooth_cutoff_integral(x, a, b)
        # Snap endpoints and center to the exact theoretical values,
        # preserving F(-b) = -pi, F(b) = pi, and F(0) = 0 so symmetry
        # and saturation are honored at roundoff level.
        y_nodes[0] = -np.pi
        y_nodes[-1] = np.pi
        y_nodes[center] = 0.0
        # Near +/-b the cutoff is exponentially small, so quad can return
        # F values that collapse to -pi (or pi) in floating point for
        # multiple adjacent nodes. Enforce strict monotonicity by dropping
        # interior nodes whose y does not strictly increase past the
        # running maximum; always keep the two boundary nodes with their
        # exact snapped values.
        kept = [0]
        for i in range(1, n_nodes - 1):
            if y_nodes[i] > y_nodes[kept[-1]]:
                kept.append(i)
        while kept and y_nodes[kept[-1]] >= y_nodes[-1]:
            kept.pop()
        kept.append(n_nodes - 1)
        kept = np.array(kept)
        x_kept = x_nodes[kept]
        y_kept = y_nodes[kept]
        return (CubicSpline(x_kept, y_kept, bc_type='natural'),
                CubicSpline(y_kept, x_kept, bc_type='natural'))
    elif name == 'vonmises':
        k_val = params['k']
        n_nodes = 2001
        theta_nodes = np.linspace(-np.pi, np.pi, n_nodes)
        center = n_nodes // 2
        theta_nodes[center] = 0.0
        y_nodes = 2*np.pi*(vonmises_dist.cdf(theta_nodes, k_val) - 0.5)
        y_nodes[0] = -np.pi
        y_nodes[-1] = np.pi
        y_nodes[center] = 0.0
        return (CubicSpline(theta_nodes, y_nodes, bc_type='natural'),
                CubicSpline(y_nodes, theta_nodes, bc_type='natural'))
    elif name == 'reg_power':
        d = params['d']
        e = params['e']
        # Cubic mesh: theta = pi * sign(u) * |u|^3 with u = linspace(-1, 1).
        # Concentrates nodes near 0, where the integrand 1/(|x|^d + e) is
        # peaked (value 1/e there) and F has very high curvature
        # (F''(theta) ~ |theta|^(d-2) for d < 1 as theta -> 0). Cubic
        # stretching keeps cubic-spline error below ~5e-7 across
        # d in [0.3, 1.0] and e in [1e-3, 1e-1] at n_nodes = 2001;
        # quartic and higher meshes give only marginally better near-0
        # accuracy at the cost of slightly worse error elsewhere.
        n_nodes = 2001
        u = np.linspace(-1.0, 1.0, n_nodes)
        theta_nodes = np.pi * np.sign(u) * np.abs(u)**3
        center = n_nodes // 2
        theta_nodes[center] = 0.0
        theta_nodes[0] = -np.pi
        theta_nodes[-1] = np.pi
        y_nodes = np.empty(n_nodes)
        for i, x in enumerate(theta_nodes):
            y_nodes[i] = reg_power_integral(x, d, e)
        # Pin endpoints and center to exact theoretical values.
        y_nodes[0] = -np.pi
        y_nodes[-1] = np.pi
        y_nodes[center] = 0.0
        # The integrand is bounded below by 1/(pi^d + e) > 0, so the
        # numerical integral is strictly monotone up to quad noise.
        # Assert rather than filter (no flat-tail risk like cutoff).
        assert np.all(np.diff(y_nodes) > 0), (
            "reg_power integral nodes are not strictly increasing; "
            "check d, e parameters and quad tolerance.")
        return (CubicSpline(theta_nodes, y_nodes, bc_type='natural'),
                CubicSpline(y_nodes, theta_nodes, bc_type='natural'))
    else:
        # symmetric_beta (analytic), direct_power, None: no spline.
        return (None, None)


# ------------------------------------------------------------------
# Family implementations.
# ------------------------------------------------------------------

def smooth_cutoff(x, a, b):
    """
    Evaluates the smooth cutoff function at x (scalar or array).
    Returns 0.0 outside [-b, b], 1.0 on [-a, a], and a smooth bump
    in between.  -b < -a < 0 < a < b
    """
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    absx = np.abs(x)
    norm = b - a   # positive since b > a > 0

    # Compute the smooth transition value for the intermediate region
    # a < |x| < b.  Outside that region the denominators (b - absx) and
    # (absx - a) would be zero or negative, so we substitute the finite
    # fill value 1.0 to keep the division well-defined everywhere.
    # The filled elements are always masked by the outer np.where below,
    # so their values never appear in the output.

    # -norm/(b-x): negative / positive = negative
    arg1 = -norm / np.where(absx < b, b - absx, 1.0)  # fill used when |x| >= b
    # -norm/(x-a): negative / positive = negative
    arg2 = -norm / np.where(absx > a, absx - a, 1.0)  # fill used when |x| <= a
    exp1 = np.exp(arg1)
    exp2 = np.exp(arg2)
    smooth = exp1 / (exp1 + exp2)

    result = np.where(absx >= b, 0.0,
             np.where(absx <= a, 1.0,
                      smooth))

    return result.item() if scalar_input else result
    
def smooth_cutoff_integral(theta, a, b, tol=1.49e-10):
    """
    Compute F(theta; a, b) = norm * integral from 0 to theta of
    smooth_cutoff(x; a, b) dx for a single float theta, where
    norm = 2*pi/(a+b). The normalization makes F(+/-b) = +/-pi so F
    plays the role of a CDF-like transformation. Used as the
    reference implementation; hot-path callers use the precomputed
    forward spline (via eval_forward_map) instead.
    """
    if theta < 0:
        NEG = True
        theta = -theta
    elif theta == 0:
        return 0.0
    else:
        NEG = False
    if not (0 <= a < b):
        raise ValueError(f"Parameters must satisfy 0 <= a < b (a={a}, b={b}).")

    # Normalization factor for the integral of the cutoff function.
    #   The area under the curve from 0 to b is a + (b-a)/2 = 0.5*(a+b).
    norm = 2*np.pi/(a+b)

    # Check for values below a
    if theta <= a:
        # integral is just the area of the rectangle
        if NEG:
            return -theta*norm
        else:
            return theta*norm
    elif theta >= b:
        if NEG:
            return -np.pi
        else:
            return np.pi

    # All other cases: a < theta < b.
    # Calculate integral from a to theta and add area from 0 to a.

    # Nudge integration bounds inward slightly to avoid handing the
    # essential singularities directly to quad; the integrand is
    # effectively 0 in those tiny gaps anyway.
    eps = (b - a) * 1e-14
    lower = a + eps
    upper = min(theta, b - eps)

    if upper <= lower:
        if NEG:
            return -a*norm
        else:
            return a*norm

    result, _err = quad(
        smooth_cutoff,
        lower,
        upper,
        args=(a, b),
        epsabs=tol,
        epsrel=tol,
        limit=200,
    )
    if NEG:
        return -(a + result)*norm
    else:
        return (a + result)*norm

def smooth_cutoff_int_inverse(y, a, b, tol=1.0e-8):
    """
    Compute F^{-1}(y; a, b) for a single float y (the inverse of
    smooth_cutoff_integral). Used as the reference implementation;
    hot-path callers use the precomputed inverse spline (via
    eval_inverse_map) instead.
    """
    if not (0 <= a < b):
        raise ValueError(f"Parameters must satisfy 0 <= a < b (a={a}, b={b}).")

    if y < -np.pi or y > np.pi:
        raise ValueError(f"y must satisfy -pi <= y <= pi (y={y}).")

    if y == -np.pi:
        return -b
    elif y == np.pi:
        return b

    # Normalization factor for the integral of the cutoff function.
    #   The area under the curve from 0 to b is a + (b-a)/2 = 0.5*(a+b).
    norm = 2*np.pi/(a+b)

    # Check for values between -a*norm and a*norm, where the inverse is just
    #   a linear scaling of y.
    if -a*norm <= y <= a*norm:
        return y/norm

    # All other cases: a*norm < |y| < pi.
    # Calculate inverse by finding root of F(theta) - y.

    def func(theta):
        return smooth_cutoff_integral(theta, a, b, tol) - np.abs(y)

    # Bracket: F is strictly increasing from 0 to pi on (a, b).
    eps = (b - a) * 1e-12
    x_lo = a + eps
    x_hi = b - eps

    result = brentq(func, x_lo, x_hi, xtol=tol, rtol=tol, maxiter=200)
    return np.sign(y) * result

def lin_cutoff(x, a, b):
    """Trapezoidal (piecewise-linear) cutoff density: 1 on [-a, a], a
    linear ramp down to 0 on a < |x| < b, and 0 for |x| >= b. The
    piecewise-linear analog of smooth_cutoff -- same support and unit
    plateau, hence the same area (a+b)/2 (so the integral map shares the
    normalization 2*pi/(a+b)), but with a closed-form integral and
    inverse instead of an essential-singularity bump. Requires
    0 <= a < b. Vectorized.
    """
    if not (0 <= a < b):
        raise ValueError(f"Parameters must satisfy 0 <= a < b (a={a}, b={b}).")
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    absx = np.abs(x)
    result = np.where(
        absx <= a, 1.0,
        np.where(absx < b, (b - absx) / (b - a), 0.0))
    return result.item() if scalar_input else result

def lin_cutoff_integral(theta, a, b):
    """Forward CDF-like map F(theta; a, b) = norm * integral from 0 to
    theta of lin_cutoff(x; a, b) dx, with norm = 2*pi/(a+b) so that
    F(+/-b) = +/-pi (matching the smooth_cutoff_integral convention).
    Closed form and odd in theta: linear (norm*theta) on |theta| <= a,
    quadratic on a < |theta| < b, saturating to +/-pi for |theta| >= b.
    Requires 0 <= a < b. Vectorized; replaces the quad-based reference
    used for the smooth cutoff, so no spline is needed.
    """
    if not (0 <= a < b):
        raise ValueError(f"Parameters must satisfy 0 <= a < b (a={a}, b={b}).")
    theta = np.asarray(theta, dtype=float)
    scalar_input = theta.ndim == 0
    norm = 2 * np.pi / (a + b)
    s = np.sign(theta)
    at = np.abs(theta)
    # Ramp branch (a < |theta| < b). The expression is evaluated for all
    # entries but only selected on the ramp; (b - at)**2 stays finite
    # outside it, so the masked-out values are harmless.
    ramp = norm * (a + ((b - a) ** 2 - (b - at) ** 2) / (2 * (b - a)))
    result = np.where(
        at <= a, norm * at,
        np.where(at < b, ramp, np.pi))
    result = s * result
    return result.item() if scalar_input else result

def lin_cutoff_int_inverse(y, a, b):
    """Inverse of lin_cutoff_integral. Domain y in [-pi, pi]; pins
    y = +/-pi to +/-b (saturation convention). Closed form and odd in y:
    linear (y/norm) on |y| <= a*norm, a single square root on the ramp.
    Exact to machine precision (no condition limit near +/-pi, unlike the
    smooth-cutoff spline). Requires 0 <= a < b. Vectorized.
    """
    if not (0 <= a < b):
        raise ValueError(f"Parameters must satisfy 0 <= a < b (a={a}, b={b}).")
    y = np.asarray(y, dtype=float)
    scalar_input = y.ndim == 0
    if np.any(y < -np.pi) or np.any(y > np.pi):
        raise ValueError("y must satisfy -pi <= y <= pi.")
    norm = 2 * np.pi / (a + b)
    s = np.sign(y)
    ay = np.abs(y)
    # Ramp inverse: theta = b - sqrt((b-a)^2 - 2(b-a)(|y|/norm - a)).
    # Clip the discriminant to guard tiny negatives from roundoff at |y|=pi.
    disc = np.clip((b - a) ** 2 - 2 * (b - a) * (ay / norm - a), 0.0, None)
    ramp = b - np.sqrt(disc)
    result = np.where(ay <= a * norm, ay / norm, ramp)
    result = s * result
    # Force exact endpoints.
    result = np.where(y == np.pi, b, result)
    result = np.where(y == -np.pi, -b, result)
    return result.item() if scalar_input else result

def vonmises(theta, k):
    """A von Mises pdf, smooth and bell-shaped around 0.

    f(theta) = exp(k*cos(theta)) / (2*pi*I0(k))

    where I0 is the modified Bessel function of the first kind of order 0.
    The parameter k controls the width of the bell: larger k gives a
    narrower peak. Integrates to 1 over [-pi, pi].

    Implemented directly rather than via scipy.stats.vonmises.pdf to avoid
    the rv_continuous dispatch overhead.

    Parameters
    ----------
    theta : float or array_like
        Angle(s) in radians.
    k : float
        Concentration parameter; must be positive.

    Returns
    -------
    float or ndarray : The value(s) of the pdf at the given theta.
    """

    if k <= 0:
        raise ValueError(f"Parameter k must be positive (k={k}).")
    theta = np.asarray(theta, dtype=float)
    scalar_input = theta.ndim == 0
    result = np.exp(k * np.cos(theta)) / (2*np.pi*i0(k))
    return result.item() if scalar_input else result

def vonmises_integral(theta, k):
    """
    Compute G(theta; k) = (1/I0(k)) * integral from 0 to theta of
    exp(k*cos(x)) dx. Maps [-pi, pi] to [-pi, pi] (i.e. G(+/-pi) = +/-pi),
    playing the role of a CDF-like transformation for the vonmises weight.

    Equivalently (by a constant factor): G(theta; k) =
    2*pi*(vonmises_cdf(theta, k) - 0.5), which is how it is computed here
    via scipy.stats.vonmises.cdf.

    Parameters
    ----------
    theta : float or array_like
        Upper limit(s) of integration.
    k : float
        Concentration parameter; must be positive.

    Returns
    -------
    float or ndarray : The value(s) of G(theta; k).
    """
    if k <= 0:
        raise ValueError(f"Parameter k must be positive (k={k}).")
    theta = np.asarray(theta, dtype=float)
    scalar_input = theta.ndim == 0
    result = 2*np.pi*(vonmises_dist.cdf(theta, k) - 0.5)
    return result.item() if scalar_input else result

def vonmises_int_inverse(y, k):
    """
    Compute G^{-1}(y; k): the value of theta such that
    G(theta; k) = y, for y in [-pi, pi].

    Uses scipy.stats.vonmises.ppf: since G = 2*pi*(cdf - 0.5),
    theta = vonmises_dist.ppf(y/(2*pi) + 0.5, k).

    Parameters
    ----------
    y : float or array_like
        Target value(s); each must satisfy -pi <= y <= pi.
    k : float
        Concentration parameter; must be positive.

    Returns
    -------
    float or ndarray : theta value(s) satisfying G(theta; k) = y.
    """
    if k <= 0:
        raise ValueError(f"Parameter k must be positive (k={k}).")
    y = np.asarray(y, dtype=float)
    scalar_input = y.ndim == 0
    if np.any(y < -np.pi) or np.any(y > np.pi):
        raise ValueError("y must satisfy -pi <= y <= pi.")
    result = vonmises_dist.ppf(y/(2*np.pi) + 0.5, k)
    # Force exact endpoints (scipy's ppf can return nan/inf at 0 or 1).
    result = np.where(y == np.pi, np.pi, result)
    result = np.where(y == -np.pi, -np.pi, result)
    return result.item() if scalar_input else result

def symmetric_beta(theta, alpha, b):
    """A symmetric Beta(alpha, alpha) pdf rescaled to [-b, b].

    With u = (theta + b)/(2b), the pdf is
        f(theta) = (1/(2b)) * u^(alpha-1) * (1-u)^(alpha-1) / B(alpha, alpha)
    on [-b, b], and zero outside. Symmetric about 0; alpha = 1 gives the
    uniform pdf 1/(2b); larger alpha gives a narrower peak at 0.

    Parameters
    ----------
    theta : float or array_like
        Angle(s) in radians.
    alpha : float
        Beta shape parameter (alpha = beta); must satisfy alpha >= 1.
    b : float
        Half-width of the support; must be positive.

    Returns
    -------
    float or ndarray : The value(s) of the pdf at the given theta.
    """
    if alpha < 1:
        raise ValueError(f"Parameter alpha must satisfy alpha >= 1 (alpha={alpha}).")
    if b <= 0:
        raise ValueError(f"Parameter b must be positive (b={b}).")
    theta = np.asarray(theta, dtype=float)
    scalar_input = theta.ndim == 0
    result = beta_dist.pdf(theta, alpha, alpha, loc=-b, scale=2*b)
    return result.item() if scalar_input else result

def symmetric_beta_integral(theta, alpha, b):
    """
    Compute G(theta; alpha, b) = 2*pi * (cdf(theta) - 0.5), where cdf is
    the Beta(alpha, alpha) cdf rescaled to [-b, b]. Maps [-pi, pi] to
    [-pi, pi] with G(0) = 0, G(b) = pi, G(-b) = -pi, saturating to +/- pi
    outside [-b, b].

    Parameters
    ----------
    theta : float or array_like
        Upper limit(s) of integration.
    alpha : float
        Beta shape parameter (alpha = beta); must satisfy alpha >= 1.
    b : float
        Half-width of the support; must be positive.

    Returns
    -------
    float or ndarray : The value(s) of G(theta; alpha, b).
    """
    if alpha < 1:
        raise ValueError(f"Parameter alpha must satisfy alpha >= 1 (alpha={alpha}).")
    if b <= 0:
        raise ValueError(f"Parameter b must be positive (b={b}).")
    theta = np.asarray(theta, dtype=float)
    scalar_input = theta.ndim == 0
    result = 2*np.pi*(beta_dist.cdf(theta, alpha, alpha, loc=-b, scale=2*b) - 0.5)
    return result.item() if scalar_input else result

def symmetric_beta_int_inverse(y, alpha, b):
    """
    Compute G^{-1}(y; alpha, b): the value of theta such that
    G(theta; alpha, b) = y, for y in [-pi, pi]. Pins y = +-pi to +-b
    (saturation convention).

    Parameters
    ----------
    y : float or array_like
        Target value(s); each must satisfy -pi <= y <= pi.
    alpha : float
        Beta shape parameter (alpha = beta); must satisfy alpha >= 1.
    b : float
        Half-width of the support; must be positive.

    Returns
    -------
    float or ndarray : theta value(s) satisfying G(theta; alpha, b) = y.
    """
    if alpha < 1:
        raise ValueError(f"Parameter alpha must satisfy alpha >= 1 (alpha={alpha}).")
    if b <= 0:
        raise ValueError(f"Parameter b must be positive (b={b}).")
    y = np.asarray(y, dtype=float)
    scalar_input = y.ndim == 0
    if np.any(y < -np.pi) or np.any(y > np.pi):
        raise ValueError("y must satisfy -pi <= y <= pi.")
    result = beta_dist.ppf(y/(2*np.pi) + 0.5, alpha, alpha, loc=-b, scale=2*b)
    # Force exact endpoints (scipy's ppf can return nan/inf at 0 or 1).
    result = np.where(y == np.pi, b, result)
    result = np.where(y == -np.pi, -b, result)
    return result.item() if scalar_input else result

def reg_power(theta, d, e):
    """A regularized power weight, 1 / (|theta|^d + e), for d, e > 0.

    Bounded everywhere (max = 1/e at theta = 0) and symmetric about 0.
    Approximates |theta|^(-d), the (un-normalized) derivative of the
    direct_power(theta, c) angle map with c = 1 - d, with the e -> 0 singularity
    at 0 regularized away. Not a normalized pdf; the constant factor cancels
    when used as a neural weight (rho = G / G.sum()) and the normalization
    used by reg_power_integral makes the integral map [-pi, pi] -> [-pi, pi].

    Parameters
    ----------
    theta : float or array_like
        Angle(s) in radians.
    d : float
        Power exponent; must be positive.
    e : float
        Regularization parameter; must be positive.

    Returns
    -------
    float or ndarray : The value(s) of the weight at the given theta.
    """
    if d <= 0:
        raise ValueError(f"Parameter d must be positive (d={d}).")
    if e <= 0:
        raise ValueError(f"Parameter e must be positive (e={e}).")
    theta = np.asarray(theta, dtype=float)
    scalar_input = theta.ndim == 0
    result = 1.0 / (np.abs(theta)**d + e)
    return result.item() if scalar_input else result

def reg_power_integral(theta, d, e, tol=1.49e-10):
    """
    Compute F(theta; d, e) = pi * sign(theta) * I(|theta|) / I(pi), where
    I(t) = integral_0^t 1/(x^d + e) dx. Maps [-pi, pi] to [-pi, pi] with
    F(0) = 0, F(+/-pi) = +/-pi, saturating outside [-pi, pi]. As e -> 0,
    F(theta; d, e) converges to direct_power(theta, c=1-d) (analytically, since
    the antiderivative becomes |theta|^(1-d)/(1-d) for d != 1).

    Used as the reference implementation; hot-path callers use the
    precomputed forward spline (via eval_forward_map) instead.

    Parameters
    ----------
    theta : float or array_like
        Upper limit(s) of integration; saturated outside [-pi, pi].
    d : float
        Power exponent; must be positive.
    e : float
        Regularization parameter; must be positive.
    tol : float
        Absolute and relative tolerance for scipy.integrate.quad.

    Returns
    -------
    float or ndarray : The value(s) of F(theta; d, e).
    """
    if d <= 0:
        raise ValueError(f"Parameter d must be positive (d={d}).")
    if e <= 0:
        raise ValueError(f"Parameter e must be positive (e={e}).")

    def integrand(x):
        return 1.0 / (x**d + e)

    # Normalization: integral over [0, pi]. Cached per (d, e) call site
    # via a simple memo so the spline build (one quad call per node) does
    # not recompute Z each time.
    Z = reg_power_normalization(d, e, tol)

    theta_arr = np.asarray(theta, dtype=float)
    scalar_input = theta_arr.ndim == 0
    flat = np.atleast_1d(theta_arr).astype(float)
    out = np.empty_like(flat)
    for i, t in enumerate(flat):
        if t >= np.pi:
            out[i] = np.pi
        elif t <= -np.pi:
            out[i] = -np.pi
        elif t == 0.0:
            out[i] = 0.0
        else:
            I, _err = quad(integrand, 0.0, abs(t),
                           epsabs=tol, epsrel=tol, limit=200)
            out[i] = np.pi * np.sign(t) * I / Z
    if scalar_input:
        return float(out[0])
    return out

# Tiny memo for the (d, e) -> Z normalization integral. Keyed by the
# exact float bit pattern; cleared rarely. Avoids n_nodes redundant quad
# calls during a spline build.
_reg_power_norm_cache = {}

def reg_power_normalization(d, e, tol=1.49e-10):
    key = (float(d), float(e), float(tol))
    cache = _reg_power_norm_cache
    Z = cache.get(key)
    if Z is None:
        Z, _err = quad(lambda x: 1.0/(x**d + e), 0.0, np.pi,
                       epsabs=tol, epsrel=tol, limit=200)
        cache[key] = Z
    return Z

def reg_power_int_inverse(y, d, e, tol=1.0e-8):
    """
    Compute F^{-1}(y; d, e): the value of theta such that
    F(theta; d, e) = y, for y in [-pi, pi]. Pins y = +-pi to +-pi.
    Used as the reference implementation; hot-path callers use the
    precomputed inverse spline (via eval_inverse_map) instead.

    Parameters
    ----------
    y : float or array_like
        Target value(s); each must satisfy -pi <= y <= pi.
    d : float
        Power exponent; must be positive.
    e : float
        Regularization parameter; must be positive.
    tol : float
        Absolute and relative tolerance for brentq.

    Returns
    -------
    float or ndarray : theta value(s) satisfying F(theta; d, e) = y.
    """
    if d <= 0:
        raise ValueError(f"Parameter d must be positive (d={d}).")
    if e <= 0:
        raise ValueError(f"Parameter e must be positive (e={e}).")
    y_arr = np.asarray(y, dtype=float)
    scalar_input = y_arr.ndim == 0
    if np.any(y_arr < -np.pi) or np.any(y_arr > np.pi):
        raise ValueError("y must satisfy -pi <= y <= pi.")
    flat = np.atleast_1d(y_arr).astype(float)
    out = np.empty_like(flat)
    for i, yv in enumerate(flat):
        if yv == np.pi:
            out[i] = np.pi
        elif yv == -np.pi:
            out[i] = -np.pi
        elif yv == 0.0:
            out[i] = 0.0
        else:
            target = abs(yv)
            func = lambda t: reg_power_integral(
                t, d, e, tol=1.49e-10) - target
            # F(0) = 0 < target < pi = F(pi); strictly monotone.
            eps = np.pi * 1e-14
            root_pos = brentq(func, eps, np.pi - eps,
                              xtol=tol, rtol=tol, maxiter=200)
            out[i] = np.sign(yv) * root_pos
    if scalar_input:
        return float(out[0])
    return out

# Mentioned in our paper, mimics Sridhar but in the perception stage.
def direct_power(theta, c):
    """
    A power function mapping perceived angles to neural positions.

    Maps theta in [-pi, pi] to [-pi, pi] via
        f(theta) = pi * sign(theta) * (|theta| / pi)^c,
    which compresses (c > 1) or expands (c < 1) angles near the front
    relative to those near the back. Fixed points at -pi, 0, and pi.

    This mimics the neural-tuning transformation of Sridhar et al.
    (2021) -- their Eq. [2] applies the same power map to the angular
    difference inside the coupling kernel -- but applied here in the
    perception stage rather than only the decision stage.

    Parameters
    ----------
    theta : float or array_like
        Angle(s) in [-pi, pi].
    c : float
        Exponent (c > 0). c = 1 gives the identity.
    """
    return np.pi * np.sign(theta) * (np.abs(theta) / np.pi) ** c

def direct_power_inverse(y, c):
    """
    Analytical inverse of direct_power: find theta such that
    direct_power(theta, c) = y, for y in [-pi, pi].

    Because direct_power(theta, c) = pi * sign(theta) * (|theta|/pi)^c,
    the inverse is simply theta = pi * sign(y) * (|y|/pi)^(1/c).

    Accepts scalar or array y; always returns the same shape.

    Parameters
    ----------
    y : float or array_like
        Target value(s); each must satisfy -pi <= y <= pi.
    c : float
        Exponent parameter of direct_power (c > 0).

    Returns
    -------
    float or ndarray : theta value(s) satisfying direct_power(theta, c) = y.
    """
    y = np.asarray(y, dtype=float)
    scalar_input = y.ndim == 0
    result = np.pi * np.sign(y) * (np.abs(y) / np.pi) ** (1.0 / c)
    return result.item() if scalar_input else result
