"""
Numerical integration and inverse for a generalized smooth transition function.

Computes:
    F(x; a, b) = integral from a to x of
        exp((a-b)/(b-s)) / (exp((a-b)/(b-s)) + exp((a-b)/(s-a))) ds

for a < x < b, and its inverse F^{-1}(y; a, b).

Notes:
  - The integrand has essential singularities at s=a and s=b, but is
    smooth and well-behaved on the open interval (a, b).
  - scipy.integrate.quad handles the near-singular endpoints well via
    adaptive quadrature, but we nudge the bounds slightly as a safeguard.
  - The integrand satisfies f(s) + f(a+b-s) = 1, which implies the
    total integral F(b; a, b) = (b-a)/2. This is a useful sanity check.
  - Near s=a the integrand approaches 1; near s=b it approaches 0. The
    transition is extremely steep near both endpoints (essentially all
    of the integral accumulates close to a).
  - The inverse is computed via Brent's method (bracketed root finding),
    which is reliable given that F is strictly monotone increasing.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def _integrand(s: float, a: float, b: float) -> float:
    """
    Evaluates the integrand at a single point s in (a, b).
    Returns 0.0 at the endpoints themselves (limiting value).
    """
    if s <= a or s >= b:
        return 0.0

    exponent = (a - b)  # this is negative since a < b

    arg1 = exponent / (b - s)   # (a-b)/(b-s): negative / positive = negative
    arg2 = exponent / (s - a)   # (a-b)/(s-a): negative / positive = negative

    # Both exponentials go to 0 as s->a or s->b (essential singularity),
    # and the ratio stays bounded in (0, 1) throughout (a, b).
    exp1 = np.exp(arg1)
    exp2 = np.exp(arg2)

    return exp1 / (exp1 + exp2)


def smooth_transition_integral(x: float, a: float, b: float,
                               tol: float = 1.49e-10) -> float:
    """
    Compute F(x; a, b) = integral from a to x of the smooth transition integrand.

    Parameters
    ----------
    x   : Upper limit of integration; must satisfy a < x < b.
    a   : Lower bound of the domain.
    b   : Upper bound of the domain.
    tol : Absolute and relative tolerance passed to scipy quad.

    Returns
    -------
    float : The value of the integral.

    Raises
    ------
    ValueError if x is not strictly between a and b.
    """
    if not (a < x < b):
        raise ValueError(f"x={x} must satisfy a < x < b (a={a}, b={b}).")

    # Nudge integration bounds inward slightly to avoid handing the
    # essential singularities directly to quad; the integrand is
    # effectively 0 in those tiny gaps anyway.
    eps = (b - a) * 1e-14
    lower = a + eps
    upper = min(x, b - eps)

    if upper <= lower:
        return 0.0

    result, _err = quad(
        _integrand,
        lower,
        upper,
        args=(a, b),
        epsabs=tol,
        epsrel=tol,
        limit=200,
    )
    return result


def smooth_transition_inverse(y: float, a: float, b: float,
                              tol: float = 1.49e-10) -> float:
    """
    Compute F^{-1}(y; a, b): the value of x such that F(x; a, b) = y.

    Parameters
    ----------
    y   : Target value; must lie strictly within the range of F,
          i.e. 0 < y < (b - a) / 2.
    a   : Lower bound of the domain.
    b   : Upper bound of the domain.
    tol : Tolerance for both the root finder and internal quad calls.

    Returns
    -------
    float : The x in (a, b) satisfying F(x; a, b) = y.

    Raises
    ------
    ValueError if y is outside the valid range (0, (b-a)/2).
    """
    max_y = (b - a) / 2.0  # F(b; a, b) by symmetry of the integrand
    if not (0 < y < max_y):
        raise ValueError(
            f"y={y} must satisfy 0 < y < (b-a)/2 = {max_y:.6g}."
        )

    def objective(x):
        return smooth_transition_integral(x, a, b, tol=tol) - y

    # Bracket: F is strictly increasing from 0 to (b-a)/2 on (a, b).
    eps = (b - a) * 1e-12
    x_lo = a + eps
    x_hi = b - eps

    root = brentq(objective, x_lo, x_hi, xtol=tol, rtol=tol, maxiter=200)
    return root


# ---------------------------------------------------------------------------
# Quick sanity checks when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    a, b = 0.0, 1.0

    print("=== Sanity checks (a=0, b=1) ===\n")

    # Total integral check: F(b-eps; a, b) should be close to (b-a)/2 = 0.5
    near_b = smooth_transition_integral(1.0 - 1e-6, a, b)
    print(f"F(b-eps):        {near_b:.10f}  (expected ~{(b-a)/2:.10f})")

    # A few sample values
    for x in [0.1, 0.25, 0.75, 0.9]:
        val = smooth_transition_integral(x, a, b)
        print(f"F({x}):         {val:.10f}")

    print()

    # Inverse round-trip checks
    print("=== Inverse round-trip checks ===\n")
    for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
        y = smooth_transition_integral(x, a, b)
        x_recovered = smooth_transition_inverse(y, a, b)
        print(f"x={x:.1f}  ->  F(x)={y:.10f}  ->  F_inv(F(x))={x_recovered:.10f}")

    print()

    # Example with different a, b
    a2, b2 = 2.0, 5.0
    print(f"=== Example with a={a2}, b={b2} ===\n")
    mid2 = smooth_transition_integral((a2 + b2) / 2, a2, b2)
    print(f"F((a+b)/2):      {mid2:.10f}  (expected {(b2 - a2) / 2:.10f})")
    y2 = smooth_transition_integral(3.5, a2, b2)
    x2_back = smooth_transition_inverse(y2, a2, b2)
    print(f"F(3.5)={y2:.10f}, F_inv(F(3.5))={x2_back:.10f}")
