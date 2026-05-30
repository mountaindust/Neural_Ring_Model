"""
Step 6 of basin-of-attraction vetting plan.

γ-saddle finding and ΔF_γ evaluation at fixed θ.

For each stable SC equilibrium (γ_s, θ_s) at a calibration focal_loc:

  1. Enumerate γ-equilibria at fixed θ = θ_s via gamma_equilib.
  2. Classify each γ-equilibrium by its Hessian:
     'min'    → both eigenvalues > 0 (γ-stable minimum of F̂)
     'saddle' → one negative eigenvalue (γ-basin boundary)
     'max'    → both eigenvalues < 0 (rare; would be a γ-repeller)
  3. For each γ-saddle, compute the barrier height
       ΔF_γ = F̂(γ_saddle) - F̂(γ_s)
     by two independent methods:
       (a) direct evaluation of F̂ at the two points;
       (b) line integral of ∇F̂ along the straight path γ_s → γ_saddle.
     These should agree to numerical-quadrature precision (gradient
     theorem).
  4. Verify Hessian consistency: along the unit direction
     n̂ = (γ_saddle - γ_s) / |γ_saddle - γ_s|, the analytical
     curvature n̂ᵀ H(γ_s) n̂ should match the numerical second
     derivative V''(0) of F̂(γ_s + t·n̂) along that line.

Tests:
  T1. Direct ΔF_γ matches path-integrated ΔF_γ at every (γ_s, γ_saddle)
      pair, to ~1e-8.
  T2. Analytical n̂ᵀ H(γ_s) n̂ matches numerical V''(0) along the line
      γ_s → γ_saddle, to ~1e-4 (h=1e-4 finite diff).

Calibration points:
  2stable_sym at (2.0, 0) — 2 stable SC eqs
  3stable     at (1.2, 0) — 3 stable SC eqs

Usage:
  python basin_via_gamma.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import quad
import decision_model as model

from check_free_energy import (F_hat, grad_F_hat, hess_F_hat,
                                _neural_data, nbm)


# =============================================================================
# Setup matches Steps 2-5
# =============================================================================
T_val = nbm.T
print(f"Setup: VM-k055 two-circle, T={T_val}, K={nbm.K}")


# =============================================================================
# γ-equilibria at fixed θ, classified by Hessian
# =============================================================================
def gamma_eqs_at_theta(theta, focal_loc):
    """Enumerate γ-equilibria at fixed θ. Classify each by Hessian
    eigenvalues. Returns list of dicts:
        {'gamma': complex, 'H': 2x2 ndarray, 'eigs': (lam1, lam2),
         'kind': 'min' / 'saddle' / 'max'}
    """
    gammas, _ = nbm.gamma_equilib(focal_angle=theta, focal_loc=focal_loc,
                                   stability_criterion='discrim_a')
    out = []
    for g in gammas:
        H = hess_F_hat(g.real, g.imag, theta, focal_loc)
        eigs = np.linalg.eigvalsh(H)
        if np.all(eigs > 1e-10):
            kind = 'min'
        elif eigs[0] < -1e-10 and eigs[1] > 1e-10:
            kind = 'saddle'
        elif np.all(eigs < -1e-10):
            kind = 'max'
        else:
            kind = 'degenerate'
        out.append({'gamma': g, 'H': H, 'eigs': eigs, 'kind': kind})
    return out


# =============================================================================
# ΔF_γ — direct and path-integrated
# =============================================================================
def delta_F_direct(gamma_a, gamma_b, theta, focal_loc):
    """ΔF̂ = F̂(γ_b) − F̂(γ_a) by direct evaluation."""
    Fa = F_hat(gamma_a.real, gamma_a.imag, theta, focal_loc)
    Fb = F_hat(gamma_b.real, gamma_b.imag, theta, focal_loc)
    return Fb - Fa


def delta_F_path(gamma_a, gamma_b, theta, focal_loc):
    """ΔF̂ via line integral of ∇F̂ along straight line γ_a → γ_b.

    Parametrize γ(s) = γ_a + s · (γ_b − γ_a) for s ∈ [0, 1]. Then
        dF̂/ds = ∇F̂(γ(s)) · (γ_b − γ_a)
    and integrating in s from 0 to 1 gives F̂(γ_b) − F̂(γ_a) by the
    gradient theorem.
    """
    direction = gamma_b - gamma_a  # complex; carries (dx, dy)
    dx, dy = direction.real, direction.imag

    def integrand(s):
        g = gamma_a + s * direction
        grad = grad_F_hat(g.real, g.imag, theta, focal_loc)
        return grad[0] * dx + grad[1] * dy

    val, _ = quad(integrand, 0.0, 1.0, epsabs=1e-13, epsrel=1e-12)
    return val


# =============================================================================
# Directional curvature consistency
# =============================================================================
def Vpp_along_direction(gamma_a, direction, theta, focal_loc, h=1e-4):
    """Numerical V''(0) of V(t) = F̂(γ_a + t·n̂) where n̂ = direction / |direction|.
    Centered 3-point finite difference at t=0.
    """
    n_hat = direction / np.abs(direction)

    def F_at_t(t):
        g = gamma_a + t * n_hat
        return F_hat(g.real, g.imag, theta, focal_loc)

    F0 = F_at_t(0.0)
    Fp = F_at_t(h)
    Fm = F_at_t(-h)
    return (Fp - 2.0 * F0 + Fm) / (h * h)


def directional_curvature_analytic(gamma_a, direction, theta, focal_loc):
    """n̂ᵀ H(γ_a) n̂ where n̂ = direction / |direction|."""
    n_hat = direction / np.abs(direction)
    n_vec = np.array([n_hat.real, n_hat.imag])
    H = hess_F_hat(gamma_a.real, gamma_a.imag, theta, focal_loc)
    return float(n_vec @ H @ n_vec)


# =============================================================================
# Validation
# =============================================================================
if __name__ == "__main__":
    calibration_points = [
        (np.array([2.0, 0.0]), '2stable_sym'),
        (np.array([1.2, 0.0]), '3stable'),
    ]

    t1_errors = []
    t2_errors = []
    n_pairs = 0

    for focal_loc, label in calibration_points:
        print()
        print("=" * 65)
        print(f"{label} at focal_loc = ({focal_loc[0]}, {focal_loc[1]})")
        print("=" * 65)

        sc_angles, sc_stab = nbm.sc_equilib(
            focal_loc=focal_loc, stability_criterion='coupled')
        stable_thetas = sorted([a for a, s in zip(sc_angles, sc_stab) if s])
        print(f"Stable SC θ: {[f'{a:+.4f}' for a in stable_thetas]}")

        for theta_s in stable_thetas:
            print(f"\n--- θ_s = {theta_s:+.4f} ---")
            eqs = gamma_eqs_at_theta(theta_s, focal_loc)
            print(f"  γ-equilibria at this θ:")
            for e in eqs:
                print(f"    γ = {e['gamma']:+.4f}  "
                      f"kind={e['kind']:7}  "
                      f"eigs=[{e['eigs'][0]:+.4e}, {e['eigs'][1]:+.4e}]")

            # Identify γ_s: the γ-min whose γ is close to R+0j (SC value)
            gamma_s_candidates = [
                e for e in eqs
                if e['kind'] == 'min'
                and abs(np.angle(e['gamma'])) < 0.1
                and abs(e['gamma']) > 0.05]
            if not gamma_s_candidates:
                print(f"  Could not identify SC γ_s as a γ-min; skipping.")
                continue
            gamma_s = gamma_s_candidates[0]['gamma']
            print(f"  SC γ_s = {gamma_s:+.5f}")

            saddles = [e for e in eqs if e['kind'] == 'saddle']
            if not saddles:
                print(f"  No γ-saddles found at this θ_s "
                      f"(no γ-basin boundary to evaluate ΔF).")
                continue

            for sad in saddles:
                gamma_sad = sad['gamma']
                n_pairs += 1

                # T1: ΔF̂ direct vs path-integrated
                dF_direct = delta_F_direct(
                    gamma_s, gamma_sad, theta_s, focal_loc)
                dF_path = delta_F_path(
                    gamma_s, gamma_sad, theta_s, focal_loc)
                err1 = abs(dF_direct - dF_path)
                t1_errors.append(err1)

                # T2: directional curvature at γ_s
                direction = gamma_sad - gamma_s
                curv_anal = directional_curvature_analytic(
                    gamma_s, direction, theta_s, focal_loc)
                curv_num = Vpp_along_direction(
                    gamma_s, direction, theta_s, focal_loc, h=1e-4)
                err2 = abs(curv_anal - curv_num)
                t2_errors.append(err2)

                print(f"  γ_saddle = {gamma_sad:+.5f}")
                print(f"    ΔF_γ direct  = {dF_direct:+.6e}")
                print(f"    ΔF_γ path    = {dF_path:+.6e}")
                print(f"    |Δ|          = {err1:.2e}")
                print(f"    n̂ᵀH(γ_s)n̂ anal = {curv_anal:+.6e}")
                print(f"    V''(0)   num   = {curv_num:+.6e}")
                print(f"    |Δ|            = {err2:.2e}")

    print()
    print("=" * 65)
    print(f"Summary over {n_pairs} (γ_s, γ_saddle) pairs:")
    if t1_errors:
        print(f"  T1 ΔF (direct vs path):     max err = {max(t1_errors):.2e}")
        t1_pass = max(t1_errors) < 1e-8
        print(f"    T1: {'PASS' if t1_pass else 'FAIL'}")
    else:
        t1_pass = False
        print(f"  T1: NO PAIRS")
    if t2_errors:
        print(f"  T2 n̂ᵀHn̂ (anal vs num):     max err = {max(t2_errors):.2e}")
        t2_pass = max(t2_errors) < 1e-4
        print(f"    T2: {'PASS' if t2_pass else 'FAIL'}")
    else:
        t2_pass = False
        print(f"  T2: NO PAIRS")

    print()
    results = [("T1 ΔF direct vs path", t1_pass),
                ("T2 directional curvature", t2_pass)]
    n_pass = sum(1 for _, p in results if p)
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"\n{n_pass}/{len(results)} tests passed.")
    sys.exit(0 if n_pass == len(results) else 1)
