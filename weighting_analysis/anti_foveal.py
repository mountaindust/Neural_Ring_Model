"""The two anti-foveal weight families, kept here rather than in the model.

WHY THIS FILE EXISTS
--------------------
`lin_dip` and `lin_ring` were added to `PerceptionModel` in 2026-08 to test
whether a neural weight with a **dip** in the middle (rather than the usual
foveal bump) could bias the observer OUTWARD -- the mechanism the locust
three-target split would need. The answer was **no**, for a structural reason
(see [outward_bias.md](outward_bias.md)), so the families were **removed from
[decision_model.py](../decision_model.py)**: the model should not carry
perception machinery it does not use.

They are preserved here in full, and re-registered onto `PerceptionModel` at
run time, for two reasons:

  1. **Reproducibility.** [outward_bias.py](outward_bias.py) needs them. Import
     this module and call `register()` and that script runs exactly as it did
     when the figures and `outward_bias.md` were produced.
  2. **Documentation.** The shapes, their closed-form integrals/inverses, and
     the numerical care they needed are the record of what was actually tried.
     Anyone revisiting "what if the front were under-weighted?" should start
     from here rather than re-deriving it.

This is a **shim, not part of the model.** Nothing in `decision_model.py` knows
about it, and nothing outside `weighting_analysis/` should import it. If these
families are ever wanted for real, the right move is to move the four blocks
below back into `PerceptionModel` (they were written to sit there) rather than
to make the model depend on this file.

HOW THE RE-REGISTRATION WORKS
-----------------------------
`PerceptionModel` dispatches families by name through four places. `register()`
patches each one to handle these two names and delegate everything else to the
original:

  * `decision_model._FAMILY_INFO`   -- a module-level dict; just updated.
  * `_validate_params`             -- per-family parameter constraints.
  * `_eval_forward_map`            -- the CDF-like angle map F(theta).
  * `_eval_inverse_map`            -- its inverse.
  * `get_neural_weight`            -- the density itself.

`_make_integral_spline` needs no patch: its final `else` already returns
`(None, None)` for any family it does not recognise, which is correct here --
both families are analytic and spline-free, like `lin_cutoff`.

THE SHAPES
----------
Both are piecewise-linear, everywhere positive, and degenerate to uniform
weight as the central floor ``m -> 1``:

  ``lin_dip(theta; m, b)``   m at 0, ramp UP to 1 at |theta| = b, flat 1 to +-pi.
                             The sign-flipped sibling of ``lin_cutoff`` and the
                             minimal perturbation of uniform weight.
  ``lin_ring(theta; m, p)``  m at 0, up to 1 at |theta| = p, back DOWN to 0 at
                             +-pi. Same frontal dip, but it also sheds weight
                             toward the rear.

Slot mapping for the generic two-slot kwargs: ``lin_dip`` a=m, b=b;
``lin_ring`` a=m, b=p. So e.g.
``PerceptionModel(..., angle_weight='lin_dip', a_weight=0.25, b_weight=pi/2)``.

Self-test: ``python weighting_analysis/anti_foveal_selftest.py``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import decision_model as model
from decision_model import PerceptionModel

FAMILIES = ('lin_dip', 'lin_ring')

_FAMILY_INFO_ADDITIONS = {
    'lin_dip':  {'slots': ('m', 'b'), 'defaults': {'m': 0.25, 'b': np.pi/2}},
    'lin_ring': {'slots': ('m', 'p'), 'defaults': {'m': 0.25, 'p': np.pi/2}},
}


# ===========================================================================
# The densities and their integral maps.
#
# These four blocks are VERBATIM the static methods that lived on
# PerceptionModel; they are plain module functions here only so that this file
# does not have to subclass. Move them back onto the class unchanged if the
# families are ever readopted.
# ===========================================================================

def lin_dip(x, m, b):
    """Anti-foveal (centre-dip) piecewise-linear density: m at x = 0, a
    linear ramp UP to 1 at |x| = b, and 1 for b <= |x| <= pi.

    The sign-flipped sibling of _lin_cutoff: instead of a frontal plateau
    that falls away to the sides, this under-weights the front and gives
    full weight to the periphery, so the observer is pushed off whatever
    sits dead ahead. It is the minimal perturbation of uniform weight --
    w == 1 everywhere except inside the frontal dip -- which is what makes
    it the controlled comparison against the model's uniform default.

    **This is not the same as an outward bias**, and the distinction is the
    whole finding it was built to establish. The weight is a function of
    EGOCENTRIC angle, so the dip suppresses whatever the observer currently
    faces -- it cannot tell a centre target from an outer one, and so
    penalizes every single-target commitment equally. In the three-target
    geometry the outer commitments are the fragile ones (marginal Ising
    saddle-nodes), so a deepening dip destroys them first: the outer
    branches are pushed outward monotonically and annihilate entirely below
    m ~ 0.5, leaving the walker MORE centre-committed, not less. See
    outward_bias.md.

    Parameters
    ----------
    m : float, 0 <= m < 1
        the central floor: the weight straight ahead relative to the
        periphery. m -> 1 recovers uniform weight; smaller m digs a deeper
        frontal hole. m = 0 is allowed but vanishes at the single point
        x = 0, so a *delta* target exactly dead ahead would carry zero
        weight (extended targets always keep positive arc mass).
    b : float, 0 < b <= pi
        half-width of the dip: the weight reaches full strength at |x| = b.
        b = pi ramps all the way to the rear (peak weight directly behind);
        b < pi gives a frontal hole with a flat full-weight periphery.

    Returned unnormalized (peak value 1), matching _lin_cutoff; the
    normalization cancels in rho = G / G.sum(). Vectorized.
    """
    if not (0 <= m < 1):
        raise ValueError(f"Parameter m must satisfy 0 <= m < 1 (m={m}).")
    if not (0 < b <= np.pi):
        raise ValueError(f"Parameter b must satisfy 0 < b <= pi (b={b}).")
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    absx = np.abs(x)
    result = np.where(absx < b, m + (1 - m) * absx / b, 1.0)
    return result.item() if scalar_input else result


def lin_dip_integral(theta, m, b):
    """Forward CDF-like map F(theta; m, b) = norm * integral from 0 to theta
    of lin_dip(x; m, b) dx, with norm = 2*pi/(2*pi - b*(1-m)) so that
    F(+/-pi) = +/-pi (the family has full support, so it saturates only at
    the branch cut).

    Closed form and odd in theta: quadratic on the ramp |theta| <= b,
    linear beyond it. Requires 0 <= m < 1 and 0 < b <= pi. Vectorized; no
    spline is needed (cf. _lin_cutoff_integral).
    """
    if not (0 <= m < 1):
        raise ValueError(f"Parameter m must satisfy 0 <= m < 1 (m={m}).")
    if not (0 < b <= np.pi):
        raise ValueError(f"Parameter b must satisfy 0 < b <= pi (b={b}).")
    theta = np.asarray(theta, dtype=float)
    scalar_input = theta.ndim == 0
    norm = 2 * np.pi / (2 * np.pi - b * (1 - m))
    s = np.sign(theta)
    at = np.clip(np.abs(theta), None, np.pi)
    # Unnormalized antiderivative U(|theta|): quadratic on the ramp, then
    # linear with slope 1. U(b) = b*(1+m)/2 makes the two pieces meet.
    U_b = b * (1 + m) / 2
    ramp = m * at + (1 - m) * at ** 2 / (2 * b)
    flat = U_b + (at - b)
    # Clip: norm * U(pi) == pi only up to roundoff, and callers (notably
    # lin_dip_int_inverse and get_neural_angle) require the image to lie
    # inside [-pi, pi].
    result = s * np.clip(norm * np.where(at <= b, ramp, flat), None, np.pi)
    return result.item() if scalar_input else result


def lin_dip_int_inverse(y, m, b):
    """Inverse of lin_dip_integral. Domain y in [-pi, pi]. Closed form and
    odd in y. Requires 0 <= m < 1 and 0 < b <= pi. Vectorized.

    The ramp branch inverts a quadratic; it is evaluated in the
    cancellation-free form theta = 2U / (m + sqrt(m^2 + 2*(1-m)*U/b))
    rather than the textbook (-m + sqrt(D)) * b / (1-m), which loses
    precision as m -> 1 (the near-uniform limit) through both a small
    difference of nearly equal roots and a small denominator.
    """
    if not (0 <= m < 1):
        raise ValueError(f"Parameter m must satisfy 0 <= m < 1 (m={m}).")
    if not (0 < b <= np.pi):
        raise ValueError(f"Parameter b must satisfy 0 < b <= pi (b={b}).")
    y = np.asarray(y, dtype=float)
    scalar_input = y.ndim == 0
    if np.any(y < -np.pi) or np.any(y > np.pi):
        raise ValueError("y must satisfy -pi <= y <= pi.")
    norm = 2 * np.pi / (2 * np.pi - b * (1 - m))
    s = np.sign(y)
    U = np.abs(y) / norm            # the unnormalized antiderivative value
    U_b = b * (1 + m) / 2
    denom = m + np.sqrt(m ** 2 + 2 * (1 - m) * U / b)
    # denom == 0 only at m = 0, U = 0, where theta = 0; avoid 0/0.
    ramp = np.where(denom > 0, 2 * U / np.where(denom > 0, denom, 1.0), 0.0)
    flat = U + b * (1 - m) / 2
    result = s * np.where(U <= U_b, ramp, flat)
    # Force exact endpoints (the family saturates only at the branch cut).
    result = np.where(y == np.pi, np.pi, result)
    result = np.where(y == -np.pi, -np.pi, result)
    return result.item() if scalar_input else result


def lin_ring(x, m, p):
    """Annular ("donut") piecewise-linear density: m at x = 0, a linear ramp
    UP to 1 at |x| = p, then a linear ramp back DOWN to 0 at |x| = pi.

    Anti-foveal like lin_dip -- both dig a hole straight ahead -- but this
    one also sheds weight toward the rear instead of holding the periphery
    at full strength. That second property is the one candidate mechanism
    for breaking the centre/outer symmetry: the frontal dip suppresses
    whichever target is dead ahead, which penalizes EVERY single-target
    commitment equally, whereas the rear falloff removes a rival that is
    present only when the observer has committed to an OUTER target.

    Measured, the rear falloff turns out to be a second-order effect next to
    the direct suppression of the faced target: lin_ring delays the loss of
    the outer branches relative to lin_dip but does not prevent it, and
    neither produces an outward bias. The pair exists because separating
    the two effects is what makes that conclusion a measurement rather than
    a guess. See outward_bias.md.

    Parameters
    ----------
    m : float, 0 <= m < 1
        the central floor: weight straight ahead relative to the ring peak.
    p : float, 0 < p < pi
        the peak angle -- where the weight reaches 1. p -> pi degenerates to
        the monotone lin_dip(., m, b=pi).

    Positive on the open interval (-pi, pi), vanishing only at the rear
    branch cut itself (a single point, unlike the _lin_cutoff families,
    which are zero on a whole rear sector). Returned unnormalized (peak
    value 1); the normalization cancels in rho = G / G.sum(). Vectorized.
    """
    if not (0 <= m < 1):
        raise ValueError(f"Parameter m must satisfy 0 <= m < 1 (m={m}).")
    if not (0 < p < np.pi):
        raise ValueError(f"Parameter p must satisfy 0 < p < pi (p={p}).")
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    absx = np.clip(np.abs(x), None, np.pi)
    result = np.where(absx <= p,
                      m + (1 - m) * absx / p,
                      (np.pi - absx) / (np.pi - p))
    return result.item() if scalar_input else result


def lin_ring_integral(theta, m, p):
    """Forward CDF-like map F(theta; m, p) = norm * integral from 0 to theta
    of lin_ring(x; m, p) dx, with norm = 2*pi/(pi + p*m) so that
    F(+/-pi) = +/-pi.

    Closed form and odd in theta: quadratic on the inner ramp |theta| <= p
    and quadratic (opening the other way) on the outer ramp. Requires
    0 <= m < 1 and 0 < p < pi. Vectorized; no spline needed.

    Note the normalization does not depend on p: the shape's mean is
    (1+m)/2 wherever the peak sits.
    """
    if not (0 <= m < 1):
        raise ValueError(f"Parameter m must satisfy 0 <= m < 1 (m={m}).")
    if not (0 < p < np.pi):
        raise ValueError(f"Parameter p must satisfy 0 < p < pi (p={p}).")
    theta = np.asarray(theta, dtype=float)
    scalar_input = theta.ndim == 0
    norm = 2 * np.pi / (np.pi + p * m)
    s = np.sign(theta)
    at = np.clip(np.abs(theta), None, np.pi)
    q = np.pi - p                       # outer-ramp width
    U_p = p * (1 + m) / 2               # inner-ramp area
    inner = m * at + (1 - m) * at ** 2 / (2 * p)
    outer = U_p + (q ** 2 - (np.pi - at) ** 2) / (2 * q)
    result = s * np.clip(norm * np.where(at <= p, inner, outer), None, np.pi)
    return result.item() if scalar_input else result


def lin_ring_int_inverse(y, m, p):
    """Inverse of lin_ring_integral. Domain y in [-pi, pi]; pins y = +/-pi
    to +/-pi. Closed form and odd in y. Requires 0 <= m < 1, 0 < p < pi.

    Note the density vanishes at +/-pi, so like the 'cutoff' families this
    inverse is condition-limited there (dF/dtheta -> 0). Only the forward
    map is used in the WEIGHT role; the limitation matters only when
    lin_ring is used as a warp.
    """
    if not (0 <= m < 1):
        raise ValueError(f"Parameter m must satisfy 0 <= m < 1 (m={m}).")
    if not (0 < p < np.pi):
        raise ValueError(f"Parameter p must satisfy 0 < p < pi (p={p}).")
    y = np.asarray(y, dtype=float)
    scalar_input = y.ndim == 0
    if np.any(y < -np.pi) or np.any(y > np.pi):
        raise ValueError("y must satisfy -pi <= y <= pi.")
    norm = 2 * np.pi / (np.pi + p * m)
    s = np.sign(y)
    U = np.abs(y) / norm
    q = np.pi - p
    U_p = p * (1 + m) / 2
    # Inner ramp: same quadratic as lin_dip, in the cancellation-free form.
    denom = m + np.sqrt(m ** 2 + 2 * (1 - m) * U / p)
    inner = np.where(denom > 0, 2 * U / np.where(denom > 0, denom, 1.0), 0.0)
    # Outer ramp: (pi - theta)^2 = q^2 - 2*q*(U - U_p); clip roundoff.
    disc = np.clip(q ** 2 - 2 * q * (U - U_p), 0.0, None)
    outer = np.pi - np.sqrt(disc)
    result = s * np.where(U <= U_p, inner, outer)
    result = np.where(y == np.pi, np.pi, result)
    result = np.where(y == -np.pi, -np.pi, result)
    return result.item() if scalar_input else result


def validate_params(name, params, role):
    """The per-family constraint checks, as they read in _validate_params."""
    if name == 'lin_dip':
        m, b = params['m'], params['b']
        if not (0 <= m < 1):
            raise ValueError(
                f"for lin_dip {role}, a_{role} (m, the central floor) must "
                f"satisfy 0 <= m < 1 (got m={m}). m = 1 is uniform weight; "
                "use angle_weight=None for that.")
        if not (0 < b <= np.pi):
            raise ValueError(
                f"for lin_dip {role}, b_{role} (b, the ramp half-width) "
                f"must satisfy 0 < b <= pi (got b={b}).")
    elif name == 'lin_ring':
        m, p = params['m'], params['p']
        if not (0 <= m < 1):
            raise ValueError(
                f"for lin_ring {role}, a_{role} (m, the central floor) must "
                f"satisfy 0 <= m < 1 (got m={m}). m = 1 would flatten the "
                "ring into the lin_cutoff-like rear ramp alone.")
        if not (0 < p < np.pi):
            raise ValueError(
                f"for lin_ring {role}, b_{role} (p, the peak angle) must "
                f"satisfy 0 < p < pi (got p={p}).")


# ===========================================================================
# Registration
# ===========================================================================

_registered = False


def register():
    """Make 'lin_dip' and 'lin_ring' usable as `PerceptionModel` warp/weight
    families for the lifetime of this process. Idempotent.

    Call this at MODULE level in any script that uses them -- not inside
    ``if __name__ == '__main__'``. On Windows, `multiprocessing` spawns
    workers by re-importing the main module, and a worker that has not
    registered will raise `KeyError: 'lin_dip'` when it builds its model.
    """
    global _registered
    if _registered:
        return
    model._FAMILY_INFO.update(_FAMILY_INFO_ADDITIONS)

    # Accessing a staticmethod on the class gives the plain function in
    # Python 3.10+; older versions hand back a wrapper with __func__.
    def _unwrap(f):
        return getattr(f, '__func__', f)

    _orig_validate = _unwrap(PerceptionModel._validate_params)
    _orig_forward = _unwrap(PerceptionModel._eval_forward_map)
    _orig_inverse = _unwrap(PerceptionModel._eval_inverse_map)
    _orig_weight = PerceptionModel.get_neural_weight

    def _validate(name, params, role):
        if name in FAMILIES:
            return validate_params(name, params, role)
        return _orig_validate(name, params, role)

    def _forward(name, params, fwd_spline, theta):
        if name == 'lin_dip':
            return lin_dip_integral(theta, params['m'], params['b'])
        if name == 'lin_ring':
            return lin_ring_integral(theta, params['m'], params['p'])
        return _orig_forward(name, params, fwd_spline, theta)

    def _inverse(name, params, inv_spline, y):
        if name == 'lin_dip':
            return lin_dip_int_inverse(y, params['m'], params['b'])
        if name == 'lin_ring':
            return lin_ring_int_inverse(y, params['m'], params['p'])
        return _orig_inverse(name, params, inv_spline, y)

    def _weight(self, theta):
        p = self._weight_params
        if self.weight_name == 'lin_dip':
            return lin_dip(theta, p['m'], p['b'])
        if self.weight_name == 'lin_ring':
            return lin_ring(theta, p['m'], p['p'])
        return _orig_weight(self, theta)

    PerceptionModel._validate_params = staticmethod(_validate)
    PerceptionModel._eval_forward_map = staticmethod(_forward)
    PerceptionModel._eval_inverse_map = staticmethod(_inverse)
    PerceptionModel.get_neural_weight = _weight
    # _make_integral_spline needs no patch: its final `else` already returns
    # (None, None) for unrecognised families, which is correct -- both of
    # these are analytic and spline-free, like lin_cutoff.
    _registered = True
