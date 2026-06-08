"""
Demonstration: a θ-perturbation across a γ-fold switches the γ-basin and
hence the decision, even with NO γ-noise.

Setup: VM-k055, focal_loc = (4.0, 1.5) (the §9 asymmetric calibration
point). Two circle targets at (4.33, ±2.5); from this observer pose the
close target is at allocentric +71.7° and the far target at −85.3°.

Two stable SC equilibria:
  far-target stable   θ_s ≈ −1.489 rad  (faces the far target)
  close-target stable θ   ≈ +1.252 rad  (faces the close target)

The far-target basin is bounded on its CCW (increasing-θ) side by a
γ-FOLD ~0.1 rad away, and on its CW side by a saddle ~0.9 rad away
(findings.md §9).

Noise model used here. The real walker (`NeuralBandModel.plot_walkers`)
re-equilibrates γ to steady state — warm-started from the previous γ —
at every heading step (`dtheta_dt` -> `run_dgamma_dt(init_gamma=self.gamma)`).
That is the slaved / slow-manifold limit, and it is what the basin
estimator assumes. So the faithful "θ-noise kick" is: jump θ to θ_0 with
γ held, then run the *slaved* deterministic flow
    θ_{n+1} = θ_n + K·R·sin(arg(γ)/2)·dt,   γ ← γ_eq(θ_n)  [warm-start]
to steady state. (Integrating γ as a free, finite-speed fast variable
instead — the NON-slaved extreme — widens the basin and is *not* what the
model does; that is the adiabatic-breakdown caveat, findings §13.4/§0.2.)

We bracket the fold with two kicks of nearly equal magnitude:
  sub-fold   : θ_0 = θ_fold − δ   (still inside the far basin)
  trans-fold : θ_0 = θ_fold + δ   (just past the fold)
Claim: sub-fold returns to FAR; trans-fold flips to CLOSE — at the first
slaved step the warm-start γ-relaxation catastrophically jumps branches
(the far γ-min no longer exists past the fold), so the walker commits to
the other target. This is a real decision switch under heading noise
alone, with γ-noise identically zero.

Usage:  python fold_kick_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from theta_scan import nbm, _circular_angle_diff, _relax_gamma_cached
from basin_via_theta import find_sc_gamma, basin_features


FOCAL_LOC = np.array([4.0, 1.5])


def wrap(a):
    return ((a + np.pi) % (2 * np.pi)) - np.pi


def slaved_flow(gamma0, theta0, dt=0.1, n_steps=3000, conv_tol=1e-7):
    """Deterministic slaved θ-flow, mirroring the walker's per-step γ
    re-equilibration (warm-started). Returns (theta_path, R_path,
    gamma_final). γ is relaxed to steady state at each θ, warm-started
    from the previous γ — i.e. the slow-manifold dynamics with continuation.
    """
    gamma = gamma0
    theta = theta0
    thetas = [theta]
    Rs = [abs(gamma0)]
    for _ in range(n_steps):
        # warm-start γ relaxation at the current θ (matches run_dgamma_dt's
        # init_gamma=self.gamma used inside dtheta_dt)
        gamma = _relax_gamma_cached(theta, FOCAL_LOC, gamma, t_final=100)
        R = abs(gamma)
        dtheta = nbm.K * R * np.sin(np.angle(gamma) / 2.0)
        theta = wrap(theta + dtheta * dt)
        thetas.append(theta)
        Rs.append(R)
        if abs(dtheta) < conv_tol:
            break
    return np.array(thetas), np.array(Rs), gamma


def label_decision(theta_final, theta_far, theta_close):
    d_far = abs(_circular_angle_diff(theta_final, theta_far))
    d_close = abs(_circular_angle_diff(theta_final, theta_close))
    return ('FAR' if d_far < d_close else 'CLOSE'), d_far, d_close


if __name__ == "__main__":
    print("=" * 68)
    print("γ-fold kick demo  |  focal_loc =", tuple(FOCAL_LOC), " (VM-k055)")
    print("=" * 68)

    # --- locate the two stable SC eqs ---------------------------------
    sc_ang, sc_stab = nbm.sc_equilib(focal_loc=FOCAL_LOC,
                                     stability_criterion='coupled')
    stable = sorted([a for a, s in zip(sc_ang, sc_stab) if s])
    print("stable SC eqs:", [f"{a:+.4f}" for a in stable])
    theta_far = min(stable, key=lambda a: abs(_circular_angle_diff(a, -1.489)))
    theta_close = min(stable, key=lambda a: abs(_circular_angle_diff(a, +1.252)))
    print(f"far-target stable   θ_far   = {theta_far:+.4f} rad "
          f"({np.degrees(theta_far):+.1f}°)")
    print(f"close-target stable θ_close = {theta_close:+.4f} rad "
          f"({np.degrees(theta_close):+.1f}°)")

    gamma_far = find_sc_gamma(theta_far, FOCAL_LOC)
    print(f"γ_far = {gamma_far:.5f}  (R={abs(gamma_far):.4f})")

    # --- find the fold bounding the far basin (CCW side) --------------
    b = basin_features(FOCAL_LOC, theta_far, gamma_far)
    print(f"\nfar basin: CCW endpoint θ={b['ccw_endpoint']:+.4f} "
          f"({b['ccw_type']}), CW endpoint θ={b['cw_endpoint']:+.4f} "
          f"({b['cw_type']})")
    if b['ccw_type'] == 'fold':
        theta_fold = b['ccw_endpoint']
    elif b['cw_type'] == 'fold':
        theta_fold = b['cw_endpoint']
    else:
        raise SystemExit("No fold found bounding the far basin.")
    print(f"γ-fold at θ_fold = {theta_fold:+.4f} rad "
          f"({np.degrees(theta_fold):+.1f}°), "
          f"{abs(_circular_angle_diff(theta_fold, theta_far)):.3f} rad "
          f"from the far stable")

    # --- the scan-fold is NOT the decision boundary ------------------
    # First show that the walker's step-1 warm-start γ-relaxation does jump
    # branches just past the scan-fold, yet the decision does not flip.
    theta_trans = theta_fold + 0.03
    gtr = _relax_gamma_cached(theta_trans, FOCAL_LOC, gamma_far, t_final=200)
    print(f"\nstep-1 warm-start γ from γ_far at θ_0={theta_trans:+.4f} "
          f"(just past the scan-fold):")
    print(f"   γ={gtr:.4f}  |Δγ|={abs(gtr - gamma_far):.4f}  R: "
          f"{abs(gamma_far):.3f} -> {abs(gtr):.3f}  (γ-branch JUMPED)")
    th_chk, _, _ = slaved_flow(gamma_far, theta_trans)
    dec_chk, *_ = label_decision(th_chk[-1], theta_far, theta_close)
    print(f"   ...but slaved flow -> {dec_chk}: crossing the scan-fold is "
          f"NECESSARY but not SUFFICIENT for a decision flip (§4.7).")

    # --- find the true dynamical decision boundary by bisection ------
    def decision_of(th0):
        th, _, _ = slaved_flow(gamma_far, th0)
        d, _, _ = label_decision(th[-1], theta_far, theta_close)
        return d
    lo, hi = theta_fold, theta_close          # far at lo, close at hi
    assert decision_of(lo) == 'FAR', "expected FAR at the scan-fold"
    assert decision_of(hi) == 'CLOSE', "expected CLOSE at theta_close"
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        if decision_of(mid) == 'FAR':
            lo = mid
        else:
            hi = mid
    theta_dyn = 0.5 * (lo + hi)
    print(f"\ndynamical decision boundary (far<->close) θ_dyn = "
          f"{theta_dyn:+.4f} rad ({np.degrees(theta_dyn):+.1f}°)")
    print(f"   scan-fold was at {theta_fold:+.4f}; the true basin extends "
          f"{abs(_circular_angle_diff(theta_dyn, theta_fold)):.3f} rad past it.")
    print(f"   far-stable CCW basin: scan width "
          f"{abs(_circular_angle_diff(theta_fold, theta_far)):.3f} rad vs "
          f"dynamical width "
          f"{abs(_circular_angle_diff(theta_dyn, theta_far)):.3f} rad")

    # --- two kicks of nearly equal magnitude straddling θ_dyn --------
    delta = 0.03
    theta_sub = theta_dyn - delta     # inside far basin
    theta_trn = theta_dyn + delta     # past the decision boundary
    print(f"\nkicks straddling θ_dyn (δ={delta}):")
    results = {}
    for name, th0 in [('sub-boundary', theta_sub), ('trans-boundary', theta_trn)]:
        thetas, Rs, _ = slaved_flow(gamma_far, th0)
        decision, d_far, d_close = label_decision(thetas[-1],
                                                  theta_far, theta_close)
        results[name] = (thetas, Rs)
        print(f"  [{name}] θ_0={th0:+.4f} "
              f"(|kick|={abs(_circular_angle_diff(th0, theta_far)):.3f} rad) "
              f"-> final θ={thetas[-1]:+.4f} ({np.degrees(thetas[-1]):+.1f}°)"
              f"  DECISION: {decision}")

    # --- verdict -----------------------------------------------------
    sub_dec, *_ = label_decision(results['sub-boundary'][0][-1],
                                 theta_far, theta_close)
    tr_dec, *_ = label_decision(results['trans-boundary'][0][-1],
                                theta_far, theta_close)
    ok = (sub_dec == 'FAR' and tr_dec == 'CLOSE')
    print("\n" + "=" * 68)
    print(f"VERDICT: sub-boundary -> {sub_dec}, trans-boundary -> {tr_dec}   "
          f"{'PASS — a 2δ=0.06 rad change in heading flips the decision' if ok else 'UNEXPECTED'}")
    print("=" * 68)

    # --- plot --------------------------------------------------------
    dt = 0.1
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colors = {'sub-boundary': 'C0', 'trans-boundary': 'C3'}
    for name, (thetas, Rs) in results.items():
        t = np.arange(len(thetas)) * dt
        axes[0].plot(t, np.degrees(np.unwrap(thetas)), color=colors[name],
                     lw=2, label=f"{name}")
        axes[1].plot(t, Rs, color=colors[name], lw=2, label=f"{name}")
    axes[0].axhline(np.degrees(theta_far), color='k', ls='--', lw=0.8,
                    label=f"far stable ({np.degrees(theta_far):+.0f}°)")
    axes[0].axhline(np.degrees(theta_close), color='gray', ls=':', lw=0.8,
                    label=f"close stable ({np.degrees(theta_close):+.0f}°)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("θ(t)  [deg, unwrapped]")
    axes[0].legend(loc='best', fontsize=8)
    axes[0].set_title("Slaved θ-flow after a heading kick straddling the "
                      "decision boundary at (4.0, 1.5)\nnear-identical kicks, "
                      "opposite decisions")
    axes[1].set_ylabel("R(t) = |γ(t)|")
    axes[1].set_xlabel("t")
    axes[1].legend(loc='best', fontsize=8)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fold_kick_demo.png')
    plt.savefig(out, dpi=110)
    print(f"\nplot saved to {out}")
