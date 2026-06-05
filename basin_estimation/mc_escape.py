"""
Step 8 of basin-of-attraction vetting plan.

Monte Carlo escape-time validation of the Kramers predictions from
Steps 5 and 6.

Two noise modes:

  γ-Langevin only:
      dγ = -∇F̂(γ; θ) dt + sqrt(2 D) dW
      dθ = K · R(γ) · sin(arg(γ)/2) dt    [deterministic]
    Escape criterion: γ enters a small ball around a *different*
    γ-equilibrium of F̂(·; θ_s) (i.e. crosses a γ-saddle).

    Tests Step 6's ΔF_γ Kramers prediction:
      τ ~ τ_0 · exp(ΔF_γ / D)

  θ-noise only:
      γ slaved to γ_eq(θ) via a precomputed θ-scan + interpolation
      dθ = f(θ) dt + σ √dt · N(0, 1)
    Escape criterion: θ leaves a θ-window around θ_s.

    Tests Step 5's slow-manifold V barrier prediction:
      τ ~ τ_0' · exp(2 ΔV / σ²)        [D_θ = σ²/2]

Tests:
  T1: γ-Langevin τ scales as exp(ΔF_γ / D) — fitted slope of log(τ)
      vs 1/D matches ΔF_γ to within a factor of 2.
  T2: γ-Langevin τ predicts the *relative* order of escape rates
      across stable equilibria correctly (smaller ΔF_γ → faster
      escape).

We focus on γ-Langevin because the multistable y=0 setup has
γ-fold-bounded basins in θ for the side stables (Step 5 finding),
which complicates the θ-noise test. θ-noise MC is included as a
secondary diagnostic.

Uses multiprocessing.Pool for parallel ensemble runs.

Usage:
  python mc_escape.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt

import decision_model as model
from check_free_energy import (F_hat, grad_F_hat, hess_F_hat, nbm)
from basin_via_gamma import gamma_eqs_at_theta


# =============================================================================
# Module-level setup matches Steps 2-7
# =============================================================================
T_val = nbm.T
K_val = nbm.K
N_WORKERS = min(32, mp.cpu_count())


# =============================================================================
# γ-Langevin single-realization simulator (module-level for pickling)
# =============================================================================
def _gamma_langevin_one(args):
    """One γ-Langevin realization. Returns escape time, or t_max if no
    escape within max_steps.

    args = (seed, focal_loc, theta_s, gamma_s_re, gamma_s_im,
            other_gammas_re, other_gammas_im, D, dt, max_steps,
            escape_radius)
    """
    (seed, focal_loc, theta_s, gs_re, gs_im,
     other_re, other_im, D, dt, max_steps, escape_radius) = args
    rng = np.random.default_rng(seed)
    sqrt_2D_dt = np.sqrt(2.0 * D * dt)

    gr, gi = gs_re, gs_im
    theta = theta_s
    other_re = np.asarray(other_re)
    other_im = np.asarray(other_im)

    for step in range(max_steps):
        grad = grad_F_hat(gr, gi, theta, focal_loc)
        gr = gr - grad[0] * dt + sqrt_2D_dt * rng.standard_normal()
        gi = gi - grad[1] * dt + sqrt_2D_dt * rng.standard_normal()
        gamma_c = gr + 1j * gi
        R = abs(gamma_c)
        arg_g = np.angle(gamma_c)
        # Half-angle torque law in the NEURAL consensus angle arg(gamma)
        # (matches NeuralBandModel.dtheta_dt).
        theta = theta + K_val * R * np.sin(arg_g/2) * dt

        # Escape: γ enters a ball around any other γ-equilibrium
        for j in range(len(other_re)):
            if ((gr - other_re[j])**2 + (gi - other_im[j])**2
                    < escape_radius**2):
                return step * dt
    return max_steps * dt


def run_gamma_langevin_ensemble(focal_loc, theta_s, gamma_s,
                                  other_gammas, D, dt,
                                  max_steps, escape_radius,
                                  n_realizations, n_workers):
    """Run an ensemble of γ-Langevin escape simulations in parallel.
    Returns array of escape times."""
    other_re = tuple(g.real for g in other_gammas)
    other_im = tuple(g.imag for g in other_gammas)
    args_list = [(seed, focal_loc, theta_s,
                   gamma_s.real, gamma_s.imag,
                   other_re, other_im,
                   D, dt, max_steps, escape_radius)
                  for seed in range(n_realizations)]
    if n_workers <= 1:
        return np.array([_gamma_langevin_one(a) for a in args_list])
    with mp.Pool(n_workers) as pool:
        results = pool.map(_gamma_langevin_one, args_list)
    return np.array(results)


# =============================================================================
# Kramers prediction
# =============================================================================
def kramers_gamma_rate(gamma_s, gamma_saddle, theta_s, focal_loc, D):
    """2D Kramers escape rate over a single saddle.

    rate = (|λ_neg| / 2π) · sqrt(det H_min / |det H_sad|) · exp(-ΔF/D)

    where H_min, H_sad are the F̂-Hessians at γ_s and γ_saddle,
    and λ_neg is the negative eigenvalue of H_sad.
    """
    H_min = hess_F_hat(gamma_s.real, gamma_s.imag, theta_s, focal_loc)
    H_sad = hess_F_hat(gamma_saddle.real, gamma_saddle.imag, theta_s,
                        focal_loc)
    eigs_min = np.linalg.eigvalsh(H_min)
    eigs_sad = np.linalg.eigvalsh(H_sad)
    lam_neg = eigs_sad[0]  # smallest (negative)
    assert lam_neg < 0
    assert eigs_min[0] > 0
    dF = (F_hat(gamma_saddle.real, gamma_saddle.imag, theta_s, focal_loc)
          - F_hat(gamma_s.real, gamma_s.imag, theta_s, focal_loc))
    prefactor = (abs(lam_neg) / (2 * np.pi)) * np.sqrt(
        np.linalg.det(H_min) / abs(np.linalg.det(H_sad)))
    return prefactor * np.exp(-dF / D), dF, prefactor


def total_kramers_rate(gamma_s, theta_s, focal_loc, D):
    """Sum of Kramers escape rates over all γ-saddles at this θ.
    Returns total rate, list of (dF, single_rate) per saddle."""
    eqs = gamma_eqs_at_theta(theta_s, focal_loc)
    saddles = [e for e in eqs if e['kind'] == 'saddle']
    total = 0.0
    per_saddle = []
    for sad in saddles:
        rate, dF, pref = kramers_gamma_rate(
            gamma_s, sad['gamma'], theta_s, focal_loc, D)
        per_saddle.append({'gamma_sad': sad['gamma'], 'dF': dF,
                            'prefactor': pref, 'rate': rate})
        total += rate
    return total, per_saddle


# =============================================================================
# Helpers
# =============================================================================
def _find_sc_gamma(theta_sc, focal_loc):
    gammas, _ = nbm.gamma_equilib(focal_angle=theta_sc, focal_loc=focal_loc,
                                   stability_criterion='discrim_a')
    for g in gammas:
        if abs(np.angle(g)) < 0.1 and abs(g) > 0.05:
            return g
    return None


def _other_gamma_mins_at_theta(theta_s, gamma_s, focal_loc):
    """Return list of other γ-MIN equilibria at this θ (excluding γ_s).

    Only γ-mins are valid escape destinations: γ must have crossed
    the saddle and committed to a different basin. Approaching a
    γ-saddle is just a fluctuation that may turn back, not an escape.
    Earlier versions of this code included saddles in the
    destination list, which mis-detected escape ~40× too early.
    """
    eqs = gamma_eqs_at_theta(theta_s, focal_loc)
    others = []
    for e in eqs:
        if e['kind'] != 'min':
            continue
        if abs(e['gamma'] - gamma_s) > 0.05:
            others.append(e['gamma'])
    return others


# =============================================================================
# Main: run MC for one calibration point at multiple D values
# =============================================================================
def run_mc_at_calibration(focal_loc, theta_s, label,
                            D_values, dt=0.01, n_realizations=200,
                            max_t=1000.0, escape_radius=0.10):
    """Run γ-Langevin MC at multiple D values; return results."""
    gamma_s = _find_sc_gamma(theta_s, focal_loc)
    assert gamma_s is not None
    others = _other_gamma_mins_at_theta(theta_s, gamma_s, focal_loc)
    print(f"  γ_s = {gamma_s:+.5f}, other γ-mins at θ_s: {len(others)}")
    for g in others:
        print(f"    destination γ-min = {g:+.5f}")
    if not others:
        print(f"  No other γ-mins to escape to; skipping.")
        return None

    results = {'D': [], 'mean_t': [], 'std_t': [], 'fraction_escaped': [],
                'kramers_rate': [], 'kramers_tau': []}
    for D in D_values:
        rate, per_saddle = total_kramers_rate(
            gamma_s, theta_s, focal_loc, D)
        kram_tau = 1.0 / rate if rate > 0 else np.inf
        n_real = n_realizations
        # Cap simulation length tightly: cap at 5×expected τ and at
        # global max_t. Run only enough to characterize the escape
        # time distribution.
        target_max_t = min(max_t, 5.0 * kram_tau)
        target_max_t = max(target_max_t, 10.0 * dt)
        target_max_steps = int(target_max_t / dt)
        print(f"\n  D = {D:.5f}: predicted Kramers τ ≈ {kram_tau:.2f}, "
              f"using max_t = {target_max_t:.1f} ({target_max_steps} steps), "
              f"{n_real} realizations on {N_WORKERS} workers")
        for ps in per_saddle:
            print(f"    saddle at γ = {ps['gamma_sad']:+.4f}: "
                  f"ΔF = {ps['dF']:.4f}  prefactor = {ps['prefactor']:.4f}  "
                  f"rate = {ps['rate']:.4e}")
        t0 = time.time()
        times = run_gamma_langevin_ensemble(
            focal_loc, theta_s, gamma_s, others, D, dt,
            target_max_steps, escape_radius, n_real, N_WORKERS)
        elapsed = time.time() - t0
        escaped = times < (target_max_t - dt * 1.5)
        frac_esc = float(np.mean(escaped))
        if escaped.any():
            mean_t = float(np.mean(times[escaped]))
            std_t = float(np.std(times[escaped]))
        else:
            mean_t = target_max_t
            std_t = 0.0
        print(f"    Wall time: {elapsed:.1f}s. "
              f"Escaped: {frac_esc*100:.1f}%. "
              f"Mean τ (escaped only): {mean_t:.2f} ± {std_t:.2f}")
        results['D'].append(D)
        results['mean_t'].append(mean_t)
        results['std_t'].append(std_t)
        results['fraction_escaped'].append(frac_esc)
        results['kramers_rate'].append(rate)
        results['kramers_tau'].append(kram_tau)
    results['label'] = label
    results['focal_loc'] = tuple(focal_loc)
    results['theta_s'] = float(theta_s)
    results['gamma_s'] = complex(gamma_s)
    return results


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print(f"Setup: VM-k055 two-circle. T={T_val}, K={K_val}, "
          f"using {N_WORKERS} workers.")

    # =====================
    # Run MC at (1.2, 0) central stable (ΔF_γ ≈ 0.0154, 2 saddles)
    # =====================
    print()
    print("=" * 65)
    print("γ-Langevin MC: (1.2, 0) central stable")
    print("=" * 65)
    central_results = run_mc_at_calibration(
        focal_loc=np.array([1.2, 0.0]),
        theta_s=0.0,
        label='center (1.2,0)',
        # ΔF_γ = 0.0154; with 2 saddles τ_kram ~ exp(0.0154/D)/(2·prefactor)
        # D=0.005: τ ~ 60; D=0.010: τ ~ 13; D=0.020: τ ~ 4
        D_values=[0.005, 0.010, 0.020],
        dt=0.01, n_realizations=100,
        max_t=500.0, escape_radius=0.10)

    # =====================
    # Run MC at (1.2, 0) side stable (ΔF_γ ≈ 0.00426, 1 saddle)
    # =====================
    print()
    print("=" * 65)
    print("γ-Langevin MC: (1.2, 0) side stable")
    print("=" * 65)
    side_results = run_mc_at_calibration(
        focal_loc=np.array([1.2, 0.0]),
        theta_s=0.6625131041054659,
        label='side (1.2,0)',
        # ΔF_γ = 0.00426; D=0.0015: τ ~ 125; D=0.003: τ ~ 30; D=0.006: τ ~ 10
        D_values=[0.0015, 0.003, 0.006],
        dt=0.01, n_realizations=100,
        max_t=500.0, escape_radius=0.10)

    # =====================
    # T1: log(τ) vs 1/D linearity, slope ≈ ΔF_γ
    # =====================
    print()
    print("=" * 65)
    print("T1: log(τ) vs 1/D linearity for central stable")
    print("=" * 65)
    res = central_results
    D = np.array(res['D'])
    tau_emp = np.array(res['mean_t'])
    tau_pred = np.array(res['kramers_tau'])
    # Restrict to D values where escape happened
    mask = np.array(res['fraction_escaped']) > 0.5
    if mask.sum() >= 2:
        x = 1.0 / D[mask]
        y_emp = np.log(tau_emp[mask])
        y_pred = np.log(tau_pred[mask])
        slope_emp, intercept_emp = np.polyfit(x, y_emp, 1)
        slope_pred, intercept_pred = np.polyfit(x, y_pred, 1)
        print(f"  Empirical:  log τ ≈ {slope_emp:+.4f}·(1/D) + {intercept_emp:+.4f}")
        print(f"  Kramers:    log τ ≈ {slope_pred:+.4f}·(1/D) + {intercept_pred:+.4f}")
        print(f"  Slope ratio (emp/pred): {slope_emp/slope_pred:.3f}")
        t1_pass = 0.5 < slope_emp / slope_pred < 2.0
        print(f"  T1: {'PASS' if t1_pass else 'FAIL'} "
              f"(target: slope ratio in [0.5, 2])")
    else:
        t1_pass = False
        print(f"  Not enough escape events to fit slope.")

    # =====================
    # T2: side escapes faster than central at matched D (where both work)
    # =====================
    print()
    print("=" * 65)
    print("T2: relative ordering — side stable escapes faster than central")
    print("=" * 65)
    # Compare at D where both have decent escape statistics
    # Side has smaller ΔF_γ, so at matched D it should escape faster.
    # At D=0.003 (in both lists): side ΔF/D = 1.4, center ΔF/D = 5.1.
    # → center τ should be ~ exp(5.1-1.4) = 40× longer than side at this D.
    side_D = np.array(side_results['D'])
    side_tau = np.array(side_results['mean_t'])
    cent_D = np.array(central_results['D'])
    cent_tau = np.array(central_results['mean_t'])
    # Find a common D where both have escape
    side_mask = np.array(side_results['fraction_escaped']) > 0.5
    cent_mask = np.array(central_results['fraction_escaped']) > 0.5
    print(f"  Side stable D values that escaped: {side_D[side_mask]}")
    print(f"  Center stable D values that escaped: {cent_D[cent_mask]}")
    # Pick the smallest D for each
    if side_mask.any() and cent_mask.any():
        # Compare τ at the largest D each has (most reliable stats)
        side_idx = np.argmax(side_D[side_mask] / 1)  # pick anything that escaped
        cent_idx = np.argmax(cent_D[cent_mask] / 1)
        D_side = side_D[side_mask][side_idx]
        D_cent = cent_D[cent_mask][cent_idx]
        tau_side = side_tau[side_mask][side_idx]
        tau_cent = cent_tau[cent_mask][cent_idx]
        # Scale to predict: τ at given D = τ_0 · exp(ΔF/D)
        # Use Kramers full prediction (with prefactors) at the same D
        kram_side = side_results['kramers_tau'][np.where(side_mask)[0][side_idx]]
        kram_cent = central_results['kramers_tau'][np.where(cent_mask)[0][cent_idx]]
        print(f"  Side D={D_side:.4f}: τ_emp={tau_side:.2f}, τ_kram={kram_side:.2f}")
        print(f"  Cent D={D_cent:.4f}: τ_emp={tau_cent:.2f}, τ_kram={kram_cent:.2f}")
        # Just check: relative ordering at the SAME D — extrapolate
        # both to common D (D=0.003, common to both lists)
        common_D = 0.003
        # Refit side to extrapolate
        if side_mask.sum() >= 2:
            slope_side, intercept_side = np.polyfit(1.0/side_D[side_mask], np.log(side_tau[side_mask]), 1)
            tau_side_at_common = np.exp(slope_side / common_D + intercept_side)
        else:
            tau_side_at_common = tau_side
        if cent_mask.sum() >= 2:
            slope_cent, intercept_cent = np.polyfit(1.0/cent_D[cent_mask], np.log(cent_tau[cent_mask]), 1)
            tau_cent_at_common = np.exp(slope_cent / common_D + intercept_cent)
        else:
            tau_cent_at_common = tau_cent
        print(f"  Extrapolated to common D=0.003: "
              f"τ_side={tau_side_at_common:.2f}, τ_cent={tau_cent_at_common:.2f}")
        # T2 passes if center > side at common D
        t2_pass = tau_cent_at_common > tau_side_at_common
        print(f"  T2: {'PASS' if t2_pass else 'FAIL'}")
    else:
        t2_pass = False
        print(f"  Not enough escape events for comparison.")

    # =====================
    # Diagnostic plot
    # =====================
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'center (1.2,0)': 'C0', 'side (1.2,0)': 'C2'}
    for res in [central_results, side_results]:
        if res is None:
            continue
        D = np.array(res['D'])
        emp = np.array(res['mean_t'])
        pred = np.array(res['kramers_tau'])
        msk = np.array(res['fraction_escaped']) > 0.5
        c = colors[res['label']]
        ax.semilogy(1.0/D, emp, 'o-', color=c, label=f"empirical {res['label']}",
                     markersize=8)
        ax.semilogy(1.0/D, pred, 'x--', color=c,
                     label=f"Kramers {res['label']}",
                     markersize=10)
    ax.set_xlabel('1/D')
    ax.set_ylabel(r'$\tau$ (mean escape time)')
    ax.set_title('γ-Langevin MC vs Kramers prediction')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'mc_escape_kramers_comparison.png')
    plt.savefig(plot_path, dpi=110)
    print(f"\nDiagnostic plot saved to {plot_path}")

    # =====================
    # Summary
    # =====================
    print()
    print("=" * 65)
    results_summary = [
        ("T1 log(τ) vs 1/D slope matches Kramers (factor of 2)", t1_pass),
        ("T2 relative ordering: side escapes faster than center", t2_pass),
    ]
    n_pass = sum(1 for _, p in results_summary if p)
    for name, passed in results_summary:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"\n{n_pass}/{len(results_summary)} tests passed.")
    sys.exit(0 if n_pass == len(results_summary) else 1)
