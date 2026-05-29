"""
Step 2 of basin-of-attraction vetting plan.

Numerical validation of the free energy F̂(γ; θ, focal_loc) derived in
free_energy_derivation.md against the dgamma_dt code in
../decision_model.py.

Tests, in order:
  T1. Analytical ∇F̂ matches finite-difference ∇F̂ at random points.
  T2. -∇F̂(γ; θ, focal_loc) equals dgamma_dt at random points
      (the deterministic γ-flow is gradient flow of F̂).
  T3. ∇F̂(γ_eq) ≈ 0 at γ-equilibria from gamma_equilib.
  T4. Analytical Hessian matches finite-difference Hessian
      at random points.
  T5. Jacobian of dgamma_dt at γ_eq equals -Hessian of F̂ at γ_eq.
  T6. Hessian eigenvalue signs at SC equilibria are consistent with
      _discrim_coupled (the 3×3 coupled stability criterion that
      CLAUDE.md flags as physically correct). Specifically:
      γ-Hessian has a negative eigenvalue ⟹ _discrim_coupled=False.
      (We do NOT compare against _discrim_A here because _discrim_A
      only checks the perpendicular-to-γ direction and is known
      to miss parallel-direction γ instabilities — the documented
      "5-stable bullseye" at (1.5, 0) in VM-k055.)
  T7. At γ-equilibria, 2k·F̂(γ_eq) equals F_mean_field per spin
      evaluated at n_j*(γ_eq) — the section-5 cross-check.

If all pass, F̂ is trusted for downstream use in Steps 3, 5, 6.

Usage:
  python check_free_energy.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import decision_model as model

# -----------------------------------------------------------------------------
# Calibration setup — VM-k055 (from VM_bifurcations/VERDICT.md)
# -----------------------------------------------------------------------------
target_locs = np.array([[4.33, 2.5], [4.33, -2.5]])
targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)
percep = model.PerceptionModel(
    targets,
    focal_loc=(0, 0), focal_angle=0,
    neural_weight='vonmises', neural_angle='integral',
)
percep.k = 0.55
nbm = model.NeuralBandModel(percep)

T = nbm.T
print(f"Setup: VM-k055 two-circle, T={T}, K={nbm.K}\n")


# -----------------------------------------------------------------------------
# F̂, ∇F̂, Hessian implementations (per derivation sections 3, 6)
# -----------------------------------------------------------------------------
def _neural_data(theta, focal_loc):
    return nbm.percep_model.get_neural_signals(
        focal_angle=theta, focal_loc=focal_loc)


def F_hat(gamma_re, gamma_im, theta, focal_loc):
    """F̂(γ; θ, focal_loc), log-(1+exp) form."""
    neural_angles, rho = _neural_data(theta, focal_loc)
    if neural_angles.size == 0:
        return 0.5 * (gamma_re**2 + gamma_im**2)
    k = neural_angles.size
    v_dot_g = gamma_re * np.cos(neural_angles) + gamma_im * np.sin(neural_angles)
    u = 2.0 * k * v_dot_g / T
    return (0.5 * (gamma_re**2 + gamma_im**2)
            - (T / (2 * k)) * np.sum(rho * np.logaddexp(0, u)))


def grad_F_hat(gamma_re, gamma_im, theta, focal_loc):
    """∇F̂ analytically (derivation section 6)."""
    neural_angles, rho = _neural_data(theta, focal_loc)
    if neural_angles.size == 0:
        return np.array([gamma_re, gamma_im])
    k = neural_angles.size
    cos_t, sin_t = np.cos(neural_angles), np.sin(neural_angles)
    v_dot_g = gamma_re * cos_t + gamma_im * sin_t
    u = 2.0 * k * v_dot_g / T
    sigma = 1.0 / (1.0 + np.exp(-u))
    gx = gamma_re - np.sum(rho * sigma * cos_t)
    gy = gamma_im - np.sum(rho * sigma * sin_t)
    return np.array([gx, gy])


def hess_F_hat(gamma_re, gamma_im, theta, focal_loc):
    """Hessian of F̂ analytically (derivation section 6)."""
    neural_angles, rho = _neural_data(theta, focal_loc)
    if neural_angles.size == 0:
        return np.eye(2)
    k = neural_angles.size
    cos_t, sin_t = np.cos(neural_angles), np.sin(neural_angles)
    v_dot_g = gamma_re * cos_t + gamma_im * sin_t
    u = 2.0 * k * v_dot_g / T
    sigma = 1.0 / (1.0 + np.exp(-u))
    sp = sigma * (1 - sigma)
    coef = (2.0 * k / T) * rho * sp  # length-k
    H = np.eye(2)
    H[0, 0] -= np.sum(coef * cos_t * cos_t)
    H[1, 1] -= np.sum(coef * sin_t * sin_t)
    H[0, 1] -= np.sum(coef * cos_t * sin_t)
    H[1, 0] = H[0, 1]
    return H


def F_mf_per_spin_at_constrained(gamma_re, gamma_im, theta, focal_loc):
    """F_mf per spin evaluated at n_j = ρ_j σ(u_j(γ)) ('mean-field projection
    onto γ'). At γ-equilibria this should equal 2k·F̂(γ); at non-equilibrium γ
    it gives the value of the variational F at the fake n_j (still well-defined).
    Per derivation section 5: F_mf/N = -k·|Σ n_j e^{iθ̂_j}|² + T·Σ ρ_j [q ln q + (1-q)ln(1-q)].
    """
    neural_angles, rho = _neural_data(theta, focal_loc)
    if neural_angles.size == 0:
        return 0.0
    k = neural_angles.size
    cos_t, sin_t = np.cos(neural_angles), np.sin(neural_angles)
    v_dot_g = gamma_re * cos_t + gamma_im * sin_t
    u = 2.0 * k * v_dot_g / T
    sigma = 1.0 / (1.0 + np.exp(-u))
    n_star = rho * sigma
    # |Σ n_j e^{iθ̂_j}|² (NOT |γ|² in general — only at γ-equilibria)
    proj = np.sum(n_star * np.exp(1j * neural_angles))
    R2 = np.abs(proj)**2
    H_per_spin = -k * R2
    # Bernoulli entropy stable at boundaries
    eps = 1e-300
    q_log_q = np.where(sigma > eps, sigma * np.log(sigma + eps), 0.0)
    one_log_one = np.where(1 - sigma > eps,
                            (1 - sigma) * np.log(1 - sigma + eps), 0.0)
    S_per_spin = -np.sum(rho * (q_log_q + one_log_one))
    return H_per_spin - T * S_per_spin


# -----------------------------------------------------------------------------
# Finite-difference utilities
# -----------------------------------------------------------------------------
def fd_grad_F_hat(gamma_re, gamma_im, theta, focal_loc, h=1e-6):
    fp = lambda dx, dy: F_hat(gamma_re + dx, gamma_im + dy, theta, focal_loc)
    return np.array([(fp(h, 0) - fp(-h, 0)) / (2*h),
                     (fp(0, h) - fp(0, -h)) / (2*h)])


def fd_hess_F_hat(gamma_re, gamma_im, theta, focal_loc, h=1e-4):
    fp = lambda dx, dy: F_hat(gamma_re + dx, gamma_im + dy, theta, focal_loc)
    f0 = fp(0, 0)
    fxx = (fp(h, 0) - 2*f0 + fp(-h, 0)) / h**2
    fyy = (fp(0, h) - 2*f0 + fp(0, -h)) / h**2
    fxy = (fp(h, h) - fp(h, -h) - fp(-h, h) + fp(-h, -h)) / (4*h**2)
    return np.array([[fxx, fxy], [fxy, fyy]])


def fd_jacobian_dgamma(gamma_re, gamma_im, theta, focal_loc, h=1e-6):
    """Finite-diff Jacobian of (Re dγ/dt, Im dγ/dt) w.r.t. (γ_re, γ_im)."""
    def f(gx, gy):
        d = nbm.dgamma_dt(gamma=gx + 1j*gy,
                          focal_angle=theta, focal_loc=focal_loc)
        return np.array([d.real, d.imag])
    J = np.zeros((2, 2))
    J[:, 0] = (f(gamma_re + h, gamma_im) - f(gamma_re - h, gamma_im)) / (2*h)
    J[:, 1] = (f(gamma_re, gamma_im + h) - f(gamma_re, gamma_im - h)) / (2*h)
    return J


# -----------------------------------------------------------------------------
# Test points
# -----------------------------------------------------------------------------
rng = np.random.default_rng(42)
N_RANDOM = 200

def sample_random_point():
    return (rng.uniform(0.5, 5.5),               # focal_x
            rng.uniform(-2.5, 2.5),              # focal_y
            rng.uniform(-np.pi, np.pi),          # theta
            rng.uniform(-0.85, 0.85),            # gamma_re
            rng.uniform(-0.85, 0.85))            # gamma_im

# Calibration points where we'll enumerate γ-equilibria
eq_focal_locs = [(0.5, 0), (1.5, 0), (3.0, 0),
                  (4.0, 1.5), (2.0, -2.0), (4.33, 0)]
eq_thetas = [0.0, 0.3, 0.6, np.pi/2, 2.0, -1.0, -np.pi/3]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
results = []

# T1 — analytical vs FD gradient
max_err = 0.0
for _ in range(N_RANDOM):
    fx, fy, th, gr, gi = sample_random_point()
    floc = np.array([fx, fy])
    err = np.max(np.abs(grad_F_hat(gr, gi, th, floc)
                        - fd_grad_F_hat(gr, gi, th, floc)))
    if err > max_err:
        max_err = err
print(f"T1 analytical vs FD ∇F̂:     max err = {max_err:.2e}")
results.append(("T1", max_err < 1e-7))

# T2 — -∇F̂ equals dgamma_dt
max_err = 0.0
for _ in range(N_RANDOM):
    fx, fy, th, gr, gi = sample_random_point()
    floc = np.array([fx, fy])
    grad = grad_F_hat(gr, gi, th, floc)
    dg = nbm.dgamma_dt(gamma=gr + 1j*gi, focal_angle=th, focal_loc=floc)
    err = np.max(np.abs(-grad - np.array([dg.real, dg.imag])))
    if err > max_err:
        max_err = err
print(f"T2 -∇F̂ vs dgamma_dt:        max err = {max_err:.2e}")
results.append(("T2", max_err < 1e-13))

# T3 — ∇F̂(γ_eq) ≈ 0
max_err = 0.0
n_total = 0
for floc_tuple in eq_focal_locs:
    floc = np.array(floc_tuple)
    for th in eq_thetas:
        gammas, _ = nbm.gamma_equilib(focal_angle=th, focal_loc=floc,
                                       stability_criterion='discrim_a')
        for g in gammas:
            n_total += 1
            err = np.max(np.abs(grad_F_hat(g.real, g.imag, th, floc)))
            if err > max_err:
                max_err = err
print(f"T3 ∇F̂(γ_eq) over {n_total} eqs: max err = {max_err:.2e}")
results.append(("T3", max_err < 1e-5))

# T4 — analytical vs FD Hessian
max_err = 0.0
for _ in range(N_RANDOM):
    fx, fy, th, gr, gi = sample_random_point()
    floc = np.array([fx, fy])
    H_anal = hess_F_hat(gr, gi, th, floc)
    H_fd = fd_hess_F_hat(gr, gi, th, floc)
    err = np.max(np.abs(H_anal - H_fd))
    if err > max_err:
        max_err = err
print(f"T4 analytical vs FD Hessian: max err = {max_err:.2e}")
results.append(("T4", max_err < 1e-6))

# T5 — Jacobian of dgamma_dt at γ_eq equals -Hessian of F̂
max_err = 0.0
n_total = 0
for floc_tuple in eq_focal_locs:
    floc = np.array(floc_tuple)
    for th in eq_thetas:
        gammas, _ = nbm.gamma_equilib(focal_angle=th, focal_loc=floc,
                                       stability_criterion='discrim_a')
        for g in gammas:
            n_total += 1
            J = fd_jacobian_dgamma(g.real, g.imag, th, floc)
            H = hess_F_hat(g.real, g.imag, th, floc)
            err = np.max(np.abs(J + H))
            if err > max_err:
                max_err = err
print(f"T5 J(dgamma) + H(F̂) over {n_total} eqs: max err = {max_err:.2e}")
results.append(("T5", max_err < 1e-5))

# T6 — γ-Hessian negative eig ⟹ _discrim_coupled says unstable
# (γ-saddle in the γ subsystem ⟹ coupled 3×3 system also unstable.)
n_total = 0
n_violations = 0
violations = []
discrim_A_disagreements = []
for floc_tuple in eq_focal_locs:
    floc = np.array(floc_tuple)
    sc_angles, sc_stab_coupled = nbm.sc_equilib(
        focal_loc=floc, stability_criterion='coupled')
    _, sc_stab_discrimA = nbm.sc_equilib(
        focal_loc=floc, stability_criterion='discrim_a')
    # Align by angle
    for th_sc, coupled_stable, discrimA_stable in zip(
            sc_angles, sc_stab_coupled, sc_stab_discrimA):
        gammas, _ = nbm.gamma_equilib(focal_angle=th_sc, focal_loc=floc,
                                       stability_criterion='discrim_a')
        sc_g = None
        for g in gammas:
            if abs(np.angle(g)) < 0.05 and abs(g) > 0.01:
                sc_g = g
                break
        if sc_g is None:
            continue
        n_total += 1
        H = hess_F_hat(sc_g.real, sc_g.imag, th_sc, floc)
        eigs = np.linalg.eigvalsh(H)
        has_neg = bool(np.any(eigs < 0))
        # Required: γ-saddle ⟹ coupled-unstable
        if has_neg and coupled_stable:
            n_violations += 1
            violations.append((floc_tuple, th_sc, sc_g, eigs.tolist()))
        # Diagnostic: where _discrim_A differs from γ-Hessian sign
        if has_neg and discrimA_stable:
            discrim_A_disagreements.append(
                (floc_tuple, th_sc, sc_g, eigs.tolist()))
print(f"T6 γ-Hessian sign vs _discrim_coupled at {n_total} SC eqs:")
print(f"   violations (γ-saddle but coupled-stable): {n_violations}")
if n_violations:
    for v in violations:
        print(f"     focal_loc={v[0]}, θ_sc={v[1]:.3f}, "
              f"γ={v[2]:.3f}, H eigs={v[3]}")
print(f"   diagnostic: {len(discrim_A_disagreements)} SC eqs where "
      f"_discrim_A says stable but γ-Hessian shows a saddle")
print(f"   (these are expected — the documented _discrim_A over-counting)")
for v in discrim_A_disagreements:
    print(f"     focal_loc={v[0]}, θ_sc={v[1]:.3f}, "
          f"γ={v[2]:.3f}, H eigs={v[3]}")
results.append(("T6", n_violations == 0))

# T7 — at γ_eq, 2k·F̂ equals F_mf per spin at n_j*(γ_eq)
max_err = 0.0
n_total = 0
for floc_tuple in eq_focal_locs:
    floc = np.array(floc_tuple)
    for th in eq_thetas:
        neural_angles, _ = _neural_data(th, floc)
        if neural_angles.size == 0:
            continue
        k = neural_angles.size
        gammas, _ = nbm.gamma_equilib(focal_angle=th, focal_loc=floc,
                                       stability_criterion='discrim_a')
        for g in gammas:
            n_total += 1
            F_h = F_hat(g.real, g.imag, th, floc)
            F_mf = F_mf_per_spin_at_constrained(g.real, g.imag, th, floc)
            err = abs(2*k*F_h - F_mf)
            if err > max_err:
                max_err = err
print(f"T7 2k·F̂(γ_eq) vs F_mf at n_j*: max err over {n_total} eqs = {max_err:.2e}")
results.append(("T7", max_err < 1e-10))


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print()
print("=" * 60)
n_pass = sum(1 for _, p in results if p)
n_fail = len(results) - n_pass
for name, passed in results:
    print(f"  {name}: {'PASS' if passed else 'FAIL'}")
print(f"\n{n_pass}/{len(results)} tests passed.")
if n_fail == 0:
    print("F̂ derivation is numerically validated.")
else:
    print("F̂ derivation FAILED validation — see above.")
sys.exit(0 if n_fail == 0 else 1)
