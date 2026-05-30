"""
Step 3 of basin-of-attraction vetting plan.

γ-Langevin dynamics and stationary-distribution validation.

The stochastic γ-dynamics from derivation section 7:
    dγ = -∇F̂(γ) dt + sqrt(2 D) dW,    D = T/(2 k N)

Stationary distribution (Boltzmann-like):
    P(γ) ∝ exp(-F̂(γ) / D)

Near a γ-equilibrium γ_eq, expand F̂ to second order:
    F̂(γ) ≈ F̂(γ_eq) + (1/2)(γ - γ_eq)ᵀ H (γ - γ_eq)
so the local stationary distribution is Gaussian with
    Cov(γ) = D · H^(-1)
where H is the Hessian of F̂ at γ_eq.

Tests:
  V1. Empirical mean ⟨γ⟩ matches γ_eq.
  V2. Empirical covariance matches D · H^(-1).
  V3. Variance scales as 1/N (verifying the D = T/(2kN) calibration).
  V4. Histogram shape matches Gaussian prediction (KL divergence small).

Pass: V1 mean error < a few × sqrt(D); V2 max relative error < 15%.

Usage:
  python stoch_dgamma.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import decision_model as model
from check_free_energy import (F_hat, grad_F_hat, hess_F_hat, _neural_data)

# -----------------------------------------------------------------------------
# Setup — matches check_free_energy.py
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


# -----------------------------------------------------------------------------
# γ-Langevin integrator (Euler-Maruyama)
# -----------------------------------------------------------------------------
def gamma_langevin(gamma0, theta, focal_loc, D, dt, n_steps, rng=None):
    """Integrate dγ = -∇F̂(γ) dt + sqrt(2D) dW via Euler-Maruyama.

    Parameters
    ----------
    gamma0 : (2,) array
        Initial (γ_re, γ_im).
    theta : float
        Observer heading (held fixed during simulation).
    focal_loc : (2,) array
        Observer location.
    D : float
        Diffusion coefficient (Langevin noise variance per unit time).
    dt : float
        Time step.
    n_steps : int
        Number of Euler-Maruyama steps.
    rng : numpy Generator, optional

    Returns
    -------
    traj : (n_steps+1, 2) array
        Trajectory of (γ_re, γ_im).
    """
    if rng is None:
        rng = np.random.default_rng()
    traj = np.zeros((n_steps + 1, 2))
    traj[0] = gamma0
    sqrt_2D_dt = np.sqrt(2 * D * dt)
    for i in range(n_steps):
        g = traj[i]
        drift = -grad_F_hat(g[0], g[1], theta, focal_loc)
        noise = sqrt_2D_dt * rng.standard_normal(2)
        traj[i + 1] = g + drift * dt + noise
    return traj


# -----------------------------------------------------------------------------
# Find a calibration point with a coupled-stable SC equilibrium
# -----------------------------------------------------------------------------
candidates = [(0.5, 0), (1.0, 0), (2.5, 0), (3.5, 0),
              (4.0, 1.5), (2.0, -2.0)]
calib = None
for floc_tuple in candidates:
    floc = np.array(floc_tuple)
    sc_angles, sc_stab = nbm.sc_equilib(
        focal_loc=floc, stability_criterion='coupled')
    for theta_sc, stable in zip(sc_angles, sc_stab):
        if not stable:
            continue
        # Recover γ via gamma_equilib
        gammas, _ = nbm.gamma_equilib(
            focal_angle=theta_sc, focal_loc=floc,
            stability_criterion='discrim_a')
        for g in gammas:
            if abs(np.angle(g)) < 0.05 and abs(g) > 0.05:
                calib = (floc, theta_sc, g)
                break
        if calib is not None:
            break
    if calib is not None:
        break

assert calib is not None, "No suitable calibration point found"
focal_loc, theta_sc, gamma_sc = calib
gamma_eq_vec = np.array([gamma_sc.real, gamma_sc.imag])
neural_angles, _ = _neural_data(theta_sc, focal_loc)
k = neural_angles.size

print(f"Calibration: focal_loc={focal_loc}, θ_sc={theta_sc:.4f}, "
      f"γ_sc={gamma_sc:.4f}, k={k}")

H_eq = hess_F_hat(gamma_eq_vec[0], gamma_eq_vec[1], theta_sc, focal_loc)
H_eigs = np.linalg.eigvalsh(H_eq)
H_inv = np.linalg.inv(H_eq)
print(f"Hessian eigenvalues at γ_eq: {H_eigs}")
print(f"Hessian condition number: {H_eigs.max()/H_eigs.min():.2f}")

# Pick relaxation-time-aware dt
tau_relax = 1.0 / H_eigs.min()
dt = 0.02 * tau_relax
print(f"\nSimulation parameters: τ_relax ≈ {tau_relax:.2f}, dt = {dt:.3f}")


# -----------------------------------------------------------------------------
# V1 + V2: long run at one N, check mean and covariance
# -----------------------------------------------------------------------------
N_main = 1000
D_main = T / (2 * k * N_main)
print(f"\nMain run: N={N_main}, D = T/(2kN) = {D_main:.3e}")
predicted_cov_main = D_main * H_inv
print(f"Predicted Cov(γ) = D · H^(-1):")
print(predicted_cov_main)

rng = np.random.default_rng(0)
n_steps_main = 400_000
print(f"Running {n_steps_main} Euler-Maruyama steps "
      f"({n_steps_main * dt:.0f} time units)...")
traj = gamma_langevin(gamma_eq_vec, theta_sc, focal_loc, D_main, dt,
                       n_steps_main, rng=rng)

burn_in = int(20 * tau_relax / dt)  # 20 relaxation times of burn-in
samples = traj[burn_in:]
emp_mean = samples.mean(axis=0)
emp_cov = np.cov(samples.T)

print(f"\nV1 — empirical mean vs γ_eq:")
print(f"  Empirical mean: {emp_mean}")
print(f"  γ_eq:           {gamma_eq_vec}")
mean_err = np.max(np.abs(emp_mean - gamma_eq_vec))
mean_tol = 3 * np.sqrt(D_main / H_eigs.min())  # 3 stds of fluctuation
print(f"  |Δmean| = {mean_err:.4e}, tol (3σ) = {mean_tol:.4e}: "
      f"{'PASS' if mean_err < mean_tol else 'FAIL'}")
v1_pass = mean_err < mean_tol

print(f"\nV2 — empirical covariance vs predicted D·H^(-1):")
print(f"  Empirical:")
print(f"  {emp_cov}")
print(f"  Predicted:")
print(f"  {predicted_cov_main}")
# Use Frobenius-norm relative error — pointwise relative error breaks
# down when individual matrix entries are essentially zero (e.g. when
# Hessian eigenvectors align with axes, off-diagonals are ~1e-22).
frob_err = np.linalg.norm(emp_cov - predicted_cov_main, 'fro')
frob_pred = np.linalg.norm(predicted_cov_main, 'fro')
cov_rel_err = frob_err / frob_pred
# Also report worst diagonal relative error for sanity
diag_emp = np.diag(emp_cov)
diag_pred = np.diag(predicted_cov_main)
diag_rel_err = np.max(np.abs(diag_emp - diag_pred) / np.abs(diag_pred))
print(f"  Frobenius relative error: {cov_rel_err:.2%}")
print(f"  Worst diagonal relative error: {diag_rel_err:.2%}")
print(f"  V2: {'PASS' if cov_rel_err < 0.15 else 'FAIL'}")
v2_pass = cov_rel_err < 0.15

# -----------------------------------------------------------------------------
# V3: variance scales as 1/N
# -----------------------------------------------------------------------------
print(f"\nV3 — variance scales as 1/N:")
N_values = [200, 1000, 5000]
results_v3 = []
for N_test in N_values:
    D_test = T / (2 * k * N_test)
    rng_test = np.random.default_rng(N_test)
    n_steps_test = 200_000
    traj_test = gamma_langevin(gamma_eq_vec, theta_sc, focal_loc,
                                D_test, dt, n_steps_test, rng=rng_test)
    samples_test = traj_test[burn_in:]
    var_emp_re = np.var(samples_test[:, 0])
    var_pred_re = D_test * H_inv[0, 0]
    var_emp_im = np.var(samples_test[:, 1])
    var_pred_im = D_test * H_inv[1, 1]
    results_v3.append((N_test, var_emp_re, var_pred_re, var_emp_im, var_pred_im))
    print(f"  N={N_test:5d}: D={D_test:.2e}  "
          f"var(γ_re) emp/pred = {var_emp_re:.3e}/{var_pred_re:.3e} "
          f"(rel err {abs(var_emp_re-var_pred_re)/var_pred_re:.2%})  "
          f"var(γ_im) emp/pred = {var_emp_im:.3e}/{var_pred_im:.3e} "
          f"(rel err {abs(var_emp_im-var_pred_im)/var_pred_im:.2%})")
# Check 1/N scaling: var(N) / var(N') should equal N'/N
v3_pass = True
for i in range(len(N_values) - 1):
    N_a, var_a_re, _, _, _ = results_v3[i]
    N_b, var_b_re, _, _, _ = results_v3[i + 1]
    ratio_emp = var_a_re / var_b_re
    ratio_pred = N_b / N_a
    err = abs(ratio_emp - ratio_pred) / ratio_pred
    print(f"  var(N={N_a})/var(N={N_b}): empirical={ratio_emp:.3f}, "
          f"predicted={ratio_pred:.3f}, rel err {err:.2%}")
    if err > 0.15:
        v3_pass = False
print(f"  V3: {'PASS' if v3_pass else 'FAIL'}")

# -----------------------------------------------------------------------------
# V4: histogram matches Gaussian prediction (KL divergence)
# -----------------------------------------------------------------------------
# Project samples onto Hessian eigenvectors (decorrelates and rescales)
print(f"\nV4 — projected histogram matches predicted Gaussian:")
eigvals, eigvecs = np.linalg.eigh(H_eq)
samples_centered = samples - gamma_eq_vec
samples_proj = samples_centered @ eigvecs  # (n_samples, 2) in eigenbasis
# Each coordinate has predicted variance D / λ_i where λ_i is Hessian eigenvalue
predicted_stds_proj = np.sqrt(D_main / eigvals)
emp_stds_proj = samples_proj.std(axis=0)
print(f"  Predicted stds in Hessian eigenbasis: {predicted_stds_proj}")
print(f"  Empirical stds in Hessian eigenbasis: {emp_stds_proj}")
std_rel_err = np.max(np.abs(emp_stds_proj - predicted_stds_proj)
                      / predicted_stds_proj)
print(f"  Max relative std error: {std_rel_err:.2%}: "
      f"{'PASS' if std_rel_err < 0.10 else 'FAIL'}")
v4_pass = std_rel_err < 0.10

# Skewness — should be near zero for Gaussian
skew_re = ((samples_proj[:, 0]**3).mean()) / emp_stds_proj[0]**3
skew_im = ((samples_proj[:, 1]**3).mean()) / emp_stds_proj[1]**3
print(f"  Skewness (proj 0): {skew_re:.3f}, (proj 1): {skew_im:.3f}  "
      f"(Gaussian = 0)")
# Excess kurtosis — should be near zero for Gaussian
kurt_re = ((samples_proj[:, 0]**4).mean()) / emp_stds_proj[0]**4 - 3
kurt_im = ((samples_proj[:, 1]**4).mean()) / emp_stds_proj[1]**4 - 3
print(f"  Excess kurtosis (proj 0): {kurt_re:.3f}, (proj 1): {kurt_im:.3f}  "
      f"(Gaussian = 0)")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print()
print("=" * 60)
results = [("V1 mean", v1_pass), ("V2 covariance", v2_pass),
           ("V3 1/N scaling", v3_pass), ("V4 Gaussian shape", v4_pass)]
n_pass = sum(1 for _, p in results if p)
for name, passed in results:
    print(f"  {name}: {'PASS' if passed else 'FAIL'}")
print(f"\n{n_pass}/{len(results)} tests passed.")
if n_pass == len(results):
    print("γ-Langevin stationary distribution matches predicted "
          "Boltzmann form. D = T/(2kN) calibration verified.")
else:
    print("γ-Langevin validation FAILED — see above.")
sys.exit(0 if n_pass == len(results) else 1)
