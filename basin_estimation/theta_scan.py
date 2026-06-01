"""
Step 4 of basin-of-attraction vetting plan.

θ-scan with warm-start γ-continuation along the slow manifold.

Provides the function `theta_scan` (a reusable building block for Steps
5–11) and validates it on the 1stable_far calibration point.

At each θ on a circular mesh, integrate dγ/dt to steady state with γ
initialized from the previous θ's γ_eq (warm-start). This traces the
γ-branch that's continuous in θ — exactly the slow manifold the walker
follows. Returns θ, γ_eq, R, ego_angle, f(θ) = K · R · sin(ego_angle).

Tests on 1stable_far (focal_loc=(0.5, 0)):
  T1. f(θ) has exactly one stable zero and one unstable zero on S¹.
  T2. The stable zero matches sc_equilib's θ_sc to <1e-3 rad.

Also saves a diagnostic plot of R(θ), ego_angle(θ), f(θ).

Usage:
  python theta_scan.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import decision_model as model


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
target_locs = np.array([[4.33, 2.5], [4.33, -2.5]])
targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)
percep = model.PerceptionModel(
    targets,
    focal_loc=(0, 0), focal_angle=0,
    neural_angle_dist='vonmises', angle_weight='neural_angle_dist',
    a_warp=0.55,
)
nbm = model.NeuralBandModel(percep)


# -----------------------------------------------------------------------------
# Slow-manifold γ-continuation
# -----------------------------------------------------------------------------
def _relax_gamma_cached(theta, focal_loc, init_gamma, t_final=100):
    """Relax γ to steady state at fixed (θ, focal_loc) using a local RHS
    that avoids per-step perception calls. Returns complex γ_eq."""
    neur_angles, rho = nbm.percep_model.get_neural_signals(
        focal_angle=theta, focal_loc=focal_loc)
    if neur_angles.size == 0:
        # No targets visible: decay toward γ=0
        return 0.0 + 0.0j
    T_loc = nbm.T
    k = neur_angles.size
    cos_t = np.cos(neur_angles)
    sin_t = np.sin(neur_angles)

    def rhs(t, y):
        gr, gi = y[0], y[1]
        v_dot_g = gr * cos_t + gi * sin_t
        sigma = 1.0 / (1.0 + np.exp(-2 * k * v_dot_g / T_loc))
        return [np.sum(rho * sigma * cos_t) - gr,
                np.sum(rho * sigma * sin_t) - gi]

    y0 = [init_gamma.real, init_gamma.imag]
    sol = solve_ivp(rhs, [0, t_final], y0, method='LSODA',
                     rtol=1e-8, atol=1e-10)
    return sol.y[0, -1] + 1j * sol.y[1, -1]


def theta_scan(focal_loc, theta_start, gamma_start, n_mesh=200,
                t_final=100, direction='ccw'):
    """Sweep θ around S¹ from theta_start, using warm-start γ-continuation.

    Parameters
    ----------
    focal_loc : (2,) array
    theta_start : float
        Starting θ (must be in (-π, π]).
    gamma_start : complex
        γ_eq at θ_start (the slow-manifold value to continue from).
    n_mesh : int, default 200
        Number of θ samples around the circle (excluding the wrap point).
    t_final : float, default 100
        LSODA integration time per relaxation. Default matches the
        model's `run_dgamma_dt` to ensure near-saddle slow manifolds
        converge.
    direction : 'ccw' or 'cw'
        Sweep direction.

    Returns
    -------
    dict with keys:
        theta : (n_mesh,) θ values in (-π, π], in scan order
        gamma_eq : (n_mesh,) complex γ_eq(θ)
        R : (n_mesh,) |γ_eq|
        ego_angle : (n_mesh,) inverse neural mapping of arg(γ_eq)
        f : (n_mesh,) K · R · sin(ego_angle), the reduced θ-flow
    """
    sign = 1 if direction == 'ccw' else -1
    raw = theta_start + sign * np.linspace(0, 2 * np.pi, n_mesh + 1)[:-1]
    thetas = ((raw + np.pi) % (2 * np.pi)) - np.pi  # wrap to (-π, π]

    gammas = np.zeros(n_mesh, dtype=complex)
    current = gamma_start
    for i, th in enumerate(thetas):
        current = _relax_gamma_cached(th, focal_loc, current,
                                       t_final=t_final)
        gammas[i] = current

    R = np.abs(gammas)
    ego_angle = np.array([
        nbm.percep_model.get_neural_angle_inverse(np.angle(g))
        for g in gammas])
    f = nbm.K * R * np.sin(ego_angle)
    return {
        'theta': thetas,
        'gamma_eq': gammas,
        'R': R,
        'ego_angle': ego_angle,
        'f': f,
    }


# -----------------------------------------------------------------------------
# Zero-finder for f(θ) on the sorted circular mesh
# -----------------------------------------------------------------------------
def find_zeros(theta, f):
    """Find sign changes of f(θ) on a circular mesh, classify by df/dθ.

    Parameters
    ----------
    theta : (n,) array, sorted ascending in (-π, π]
    f : (n,) array, f(θ) at each θ

    Returns
    -------
    zeros : list of (θ_zero, type) where type ∈ {'stable', 'unstable'}.
        Stable = df/dθ < 0 (perturbations damped); unstable = df/dθ > 0.
    """
    n = len(theta)
    zeros = []
    for i in range(n):
        i_next = (i + 1) % n
        f_i, f_n = f[i], f[i_next]
        if f_i == 0.0 and f_n == 0.0:
            continue
        if f_i * f_n >= 0:
            continue
        # Linear interp in θ
        dtheta = theta[i_next] - theta[i]
        if dtheta > np.pi:
            dtheta -= 2 * np.pi
        elif dtheta < -np.pi:
            dtheta += 2 * np.pi
        alpha = f_i / (f_i - f_n)
        theta_zero = theta[i] + alpha * dtheta
        theta_zero = ((theta_zero + np.pi) % (2 * np.pi)) - np.pi
        df = (f_n - f_i) / dtheta
        zeros.append((theta_zero, 'stable' if df < 0 else 'unstable'))
    return zeros


def _circular_angle_diff(a, b):
    """Smallest signed angular distance from b to a (wrap-aware)."""
    return ((a - b + np.pi) % (2 * np.pi)) - np.pi


# =============================================================================
# Validation on the 1stable_far calibration point
# =============================================================================
if __name__ == "__main__":
    focal_loc = np.array([0.5, 0.0])
    print(f"Calibration: 1stable_far at focal_loc={tuple(focal_loc)}")

    # Find the (unique) stable SC equilibrium
    sc_angles, sc_stab = nbm.sc_equilib(focal_loc=focal_loc,
                                         stability_criterion='coupled')
    stable_sc = [a for a, s in zip(sc_angles, sc_stab) if s]
    print(f"  sc_equilib: {len(sc_angles)} SC eqs, {len(stable_sc)} stable")
    print(f"  stable θ_sc: {stable_sc}")
    assert len(stable_sc) == 1, (
        f"Expected exactly 1 stable SC eq; got {len(stable_sc)}")
    theta_sc = stable_sc[0]

    # Recover γ_sc via gamma_equilib at θ_sc
    gammas, _ = nbm.gamma_equilib(focal_angle=theta_sc, focal_loc=focal_loc,
                                   stability_criterion='discrim_a')
    gamma_sc = None
    for g in gammas:
        if abs(np.angle(g)) < 0.05 and abs(g) > 0.05:
            gamma_sc = g
            break
    assert gamma_sc is not None
    print(f"  γ_sc = {gamma_sc:.5f}  (R = {abs(gamma_sc):.4f})")

    # Run the scan
    n_mesh = 200
    print(f"\nRunning θ-scan ({n_mesh} mesh points, CCW from θ_sc)...")
    import time
    t0 = time.time()
    scan = theta_scan(focal_loc, theta_sc, gamma_sc,
                       n_mesh=n_mesh, direction='ccw')
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({1e3 * elapsed / n_mesh:.1f} ms/pt)")

    # Sort by theta for analysis
    order = np.argsort(scan['theta'])
    theta_sorted = scan['theta'][order]
    f_sorted = scan['f'][order]
    R_sorted = scan['R'][order]
    ego_sorted = scan['ego_angle'][order]
    gamma_sorted = scan['gamma_eq'][order]

    # ----- T1: f(θ) has exactly one stable zero and one unstable zero -----
    zeros = find_zeros(theta_sorted, f_sorted)
    print(f"\nT1: zeros of f(θ) on S¹:")
    for tz, tp in zeros:
        print(f"  θ = {tz:+.4f} rad ({np.degrees(tz):+6.2f}°)  {tp}")
    n_stable = sum(1 for _, t in zeros if t == 'stable')
    n_unstable = sum(1 for _, t in zeros if t == 'unstable')
    t1_pass = (n_stable == 1) and (n_unstable == 1)
    print(f"  Expected: 1 stable + 1 unstable")
    print(f"  Found:    {n_stable} stable + {n_unstable} unstable")
    print(f"  T1: {'PASS' if t1_pass else 'FAIL'}")

    # ----- T2: stable zero matches sc_equilib θ_sc -----
    stable_zeros = [tz for tz, tp in zeros if tp == 'stable']
    if stable_zeros:
        err = min(abs(_circular_angle_diff(tz, theta_sc))
                  for tz in stable_zeros)
        print(f"\nT2: stable zero vs sc_equilib θ_sc:")
        print(f"  scan stable θ: {stable_zeros}")
        print(f"  sc_equilib θ_sc: {theta_sc:.6f}")
        print(f"  Min |Δθ|: {err:.2e}")
        t2_pass = err < 1e-3
        print(f"  T2: {'PASS' if t2_pass else 'FAIL'}")
    else:
        print(f"\nT2: no stable zero found — cannot evaluate")
        t2_pass = False

    # ----- Diagnostic plot -----
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    deg = np.degrees(theta_sorted)
    axes[0].plot(deg, R_sorted, 'k.-', markersize=3)
    axes[0].set_ylabel('R(θ)')
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].axvline(np.degrees(theta_sc), color='C2', ls='--', alpha=0.7,
                     label=f'θ_sc = {np.degrees(theta_sc):.1f}°')
    axes[0].legend(loc='lower right')
    axes[0].set_title(f'θ-scan on slow manifold, 1stable_far '
                       f'focal_loc={tuple(focal_loc)}')

    axes[1].plot(deg, np.degrees(ego_sorted), 'k.-', markersize=3)
    axes[1].set_ylabel('ego_angle [°]')
    axes[1].axhline(0, color='gray', lw=0.5)
    axes[1].grid(alpha=0.3)

    axes[2].plot(deg, f_sorted, 'k.-', markersize=3)
    axes[2].axhline(0, color='gray', lw=0.5)
    axes[2].set_ylabel('f(θ) = K·R·sin(ego)')
    axes[2].set_xlabel('θ [°]')
    axes[2].grid(alpha=0.3)
    for tz, tp in zeros:
        c = 'C2' if tp == 'stable' else 'C3'
        axes[2].plot(np.degrees(tz), 0, 'o', color=c, markersize=8,
                      markeredgecolor='k', label=tp)
    # Dedup labels
    handles, labels = axes[2].get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels)
              if not (l in seen or seen.add(l))]
    axes[2].legend([h for h, _ in unique], [l for _, l in unique],
                    loc='lower right')

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'theta_scan_1stable_far.png')
    plt.savefig(plot_path, dpi=110)
    print(f"\nDiagnostic plot saved to {plot_path}")

    # ----- Summary -----
    print()
    print("=" * 60)
    results = [("T1 exactly 1 stable + 1 unstable", t1_pass),
                ("T2 stable zero matches θ_sc", t2_pass)]
    n_pass = sum(1 for _, p in results if p)
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"\n{n_pass}/{len(results)} tests passed.")
    sys.exit(0 if n_pass == len(results) else 1)
