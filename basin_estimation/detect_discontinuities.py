"""
Step 7 of basin-of-attraction vetting plan.

Discontinuity detection during θ-scans on the slow manifold.

Three types of basin-boundary events are recognized:

  (a) γ-fold — a saddle-node bifurcation on the γ-branch. Signature:
      |Δγ_eq| between consecutive scan samples is much larger than
      typical, both relatively (>> median) and absolutely (>~0.4).
      The walker's γ catastrophically jumps to a different γ-branch.

  (b) f-jump — f(θ) has a step-like change without a γ-jump of
      similar magnitude. This is the signature of a discrete change
      in perception (e.g. an occlusion transition for non-delta
      geometries, or a target entering/leaving the perception window
      for some cutoff weightings). Distinguished from (a) by
      |Δf| being large while |Δγ| stays at typical scale.

  (c) Perception collapse — R → 0 over an extended interval of θ.
      Occurs under narrow-cutoff weighting (e.g. cutoff b=π/2): when
      the observer's heading is such that all targets lie outside
      the perception window, every ρ_j = 0 and γ relaxes to 0.
      This is a real "blind region" in the slow manifold with f = 0
      throughout, a basin-boundary mechanism distinct from γ-folds
      and saddles.

NOTE on static vs dynamical "blind-spot trap":
  weighting_analysis/README.md documents a dynamical trap that
  emerges with cutoff a=0/b=π + integral neural mapping: a walker
  that rotates such that all targets get behind it has γ collapse
  onto the ±π branch cut where sin(±π)≈0 kills the torque. In a
  STATIC slow-manifold scan, however, this signature (R near 1,
  arg(γ)≈±π, |f|≈0) is *the same* as a perfectly normal SC saddle
  at θ=±π with γ = −R + 0j. We can't distinguish trap from saddle
  on a static scan; that's a dynamical question. So Step 7's
  perception-collapse detector targets the R→0 case (perception
  discontinuity), not the branch-cut signature.

These three events together (plus the simple saddle bounding from
Step 5) cover the basin boundary mechanisms documented in
basin_estimation_planning.md.

Tests:
  T1 — (0.5, 0) smooth scan: no spurious detections.
  T2 — (1.2, 0) and (2.0, 0): γ-folds detected at θ values matching
       Step 5's basin extraction.
  T3 — BlindSpot setup (4 delta targets, cutoff a=0/b=π, integral
       neural mapping): torque dead-zone detected behind the walker.

Usage:
  python detect_discontinuities.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import decision_model as model
import theta_scan as theta_scan_mod
from theta_scan import theta_scan, _relax_gamma_cached


# =============================================================================
# Event detection
# =============================================================================
def detect_gamma_folds(scan,
                       rel_threshold=8.0,
                       abs_threshold=0.4):
    """Detect γ-fold events along the scan.

    A fold event is flagged at step i if BOTH:
      - |Δγ_i| > rel_threshold * median(|Δγ|)  (the relative criterion;
        catches genuine outliers in the size of γ-steps), and
      - |Δγ_i| > abs_threshold                  (the absolute criterion;
        rejects events on a uniformly-rapid scan where every step is
        comparable in size — those aren't folds, just rapid rotation).

    Returns list of dicts: {'index', 'theta_before', 'theta_after',
    'delta_gamma', 'severity'} where severity is |Δγ| / median.
    """
    gam = scan['gamma_eq']
    th = scan['theta']
    n = len(gam)
    deltas = np.zeros(n - 1)
    for i in range(1, n):
        deltas[i - 1] = abs(gam[i] - gam[i - 1])
    med = float(np.median(deltas))
    events = []
    for i in range(1, n):
        dg = deltas[i - 1]
        if dg > rel_threshold * max(med, 1e-3) and dg > abs_threshold:
            events.append({
                'index': i,
                'theta_before': th[i - 1],
                'theta_after': th[i],
                'delta_gamma': dg,
                'severity': dg / max(med, 1e-3),
            })
    return events


def detect_f_jumps(scan,
                   rel_threshold=8.0,
                   abs_threshold=0.1,
                   gamma_jump_max=0.2):
    """Detect f-jump events that are NOT also γ-fold events.

    A pure f-jump is flagged at step i if:
      - |Δf_i| is large (relatively and absolutely), AND
      - |Δγ_i| is NOT large (stays below `gamma_jump_max`).

    The gamma_jump_max gate excludes γ-fold events, which also cause
    f-jumps but get attributed to detect_gamma_folds instead.

    Returns list of dicts as above.
    """
    gam = scan['gamma_eq']
    f = scan['f']
    th = scan['theta']
    n = len(f)
    df = np.zeros(n - 1)
    dg = np.zeros(n - 1)
    for i in range(1, n):
        df[i - 1] = abs(f[i] - f[i - 1])
        dg[i - 1] = abs(gam[i] - gam[i - 1])
    med_f = float(np.median(df))
    events = []
    for i in range(1, n):
        if (df[i - 1] > rel_threshold * max(med_f, 1e-3)
                and df[i - 1] > abs_threshold
                and dg[i - 1] < gamma_jump_max):
            events.append({
                'index': i,
                'theta_before': th[i - 1],
                'theta_after': th[i],
                'delta_f': df[i - 1],
                'delta_gamma': dg[i - 1],
                'severity': df[i - 1] / max(med_f, 1e-3),
            })
    return events


def detect_perception_collapse(scan,
                                 R_threshold=0.05,
                                 min_run_length=5):
    """Detect contiguous runs of small R ("perception collapse").

    Occurs under narrow-cutoff weighting where all targets fall
    outside the perception window for some θ-range — every ρ_j = 0,
    so γ_eq = 0, R = 0, f = 0. This is a genuine basin-boundary
    mechanism: a walker entering this region has no torque to
    re-orient by.

    Returns list of dicts: {'index_start', 'index_end', 'theta_start',
    'theta_end', 'run_length', 'mean_R', 'mean_f'}.
    """
    R = scan['R']
    f = scan['f']
    th = scan['theta']
    n = len(R)
    small = R < R_threshold
    events = []
    i = 0
    while i < n:
        if small[i]:
            j = i
            while j < n and small[j]:
                j += 1
            run_len = j - i
            if run_len >= min_run_length:
                events.append({
                    'index_start': i,
                    'index_end': j - 1,
                    'theta_start': float(th[i]),
                    'theta_end': float(th[j - 1]),
                    'run_length': run_len,
                    'mean_R': float(np.mean(R[i:j])),
                    'mean_f': float(np.mean(np.abs(f[i:j]))),
                })
            i = j
        else:
            i += 1
    return events


# =============================================================================
# Calibration setups
# =============================================================================
def setup_VM_k055():
    """VM-k055 two circles — the setup used throughout Steps 2-6."""
    target_locs = np.array([[4.33, 2.5], [4.33, -2.5]])
    targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)
    percep = model.PerceptionModel(
        targets, focal_loc=(0, 0), focal_angle=0,
        neural_angle_dist='vonmises', angle_weight='neural_angle_dist',
        a_warp=0.55)
    return model.NeuralBandModel(percep)


def setup_BlindSpot():
    """4 delta targets, narrow cutoff a=0/b=π/2 → genuine static
    perception-collapse region behind the walker (where ALL targets
    fall outside the cutoff window).

    This is a tighter cutoff than weighting_analysis/'s a=0/b=π. The
    a=0/b=π setup creates a *dynamical* blind-spot trap (γ collapses
    onto the ±π branch cut), but on a static slow-manifold scan its
    signature is indistinguishable from a normal saddle. The
    a=0/b=π/2 setup creates a *static* signature: R drops to 0 over
    an extended θ-range where the cutoff makes all ρ_j = 0.
    """
    target_locs = np.array([[4.33, 2.25], [4.33, -2.25],
                             [4.33, 0.75], [4.33, -0.75]])
    targets = model.Targets(locs=target_locs, geom_name=None)  # delta
    percep = model.PerceptionModel(
        targets, focal_loc=(0, 0), focal_angle=0,
        neural_angle_dist='cutoff', angle_weight='neural_angle_dist',
        a_warp=0.0, b_warp=np.pi / 2)
    return model.NeuralBandModel(percep)


# =============================================================================
# Helpers
# =============================================================================
def _find_sc_gamma(nbm, theta_sc, focal_loc):
    """Recover the γ ≈ R + 0j at SC eq."""
    gammas, _ = nbm.gamma_equilib(focal_angle=theta_sc, focal_loc=focal_loc,
                                   stability_criterion='discrim_a')
    for g in gammas:
        if abs(np.angle(g)) < 0.1 and abs(g) > 0.05:
            return g
    return None


def run_scan(nbm, focal_loc, theta_start, gamma_start,
             direction='ccw', n_mesh=200):
    """Run a θ-scan with the given model patched into theta_scan."""
    orig_nbm = theta_scan_mod.nbm
    theta_scan_mod.nbm = nbm
    try:
        scan = theta_scan(focal_loc, theta_start, gamma_start,
                           n_mesh=n_mesh, direction=direction)
    finally:
        theta_scan_mod.nbm = orig_nbm
    return scan


# =============================================================================
# Tests
# =============================================================================
if __name__ == "__main__":
    results = []

    # =====================
    # T1 — smooth, no false positives
    # =====================
    print("=" * 65)
    print("T1: smooth scan at (0.5, 0), expect no detections")
    print("=" * 65)
    nbm_vm = setup_VM_k055()
    focal_loc = np.array([0.5, 0.0])
    sc_angles, sc_stab = nbm_vm.sc_equilib(focal_loc=focal_loc,
                                            stability_criterion='coupled')
    stable = sorted([a for a, s in zip(sc_angles, sc_stab) if s])
    theta_start = stable[0]
    gamma_start = _find_sc_gamma(nbm_vm, theta_start, focal_loc)

    # Full circle from the single stable
    scan = run_scan(nbm_vm, focal_loc, theta_start, gamma_start,
                     direction='ccw', n_mesh=200)
    folds = detect_gamma_folds(scan)
    fjumps = detect_f_jumps(scan)
    pcs = detect_perception_collapse(scan)
    print(f"  γ-folds: {len(folds)}")
    print(f"  f-jumps: {len(fjumps)}")
    print(f"  perception collapse zones: {len(pcs)}")
    t1_pass = (len(folds) == 0 and len(fjumps) == 0 and len(pcs) == 0)
    print(f"  T1: {'PASS' if t1_pass else 'FAIL'}")
    results.append(("T1 no spurious detection at (0.5, 0)", t1_pass))

    # =====================
    # T2 — γ-folds at multistable VM-k055 points
    # =====================
    print()
    print("=" * 65)
    print("T2: γ-folds at (1.2, 0) and (2.0, 0)")
    print("=" * 65)
    t2_all = True
    for focal in [(1.2, 0.0), (2.0, 0.0)]:
        focal_loc = np.array(focal)
        sc_angles, sc_stab = nbm_vm.sc_equilib(
            focal_loc=focal_loc, stability_criterion='coupled')
        stable = sorted([a for a, s in zip(sc_angles, sc_stab) if s])
        # Use the rightmost stable as scan start
        theta_start = stable[-1]
        gamma_start = _find_sc_gamma(nbm_vm, theta_start, focal_loc)

        # CW scan toward the central region (where folds live)
        scan = run_scan(nbm_vm, focal_loc, theta_start, gamma_start,
                         direction='cw', n_mesh=200)
        folds = detect_gamma_folds(scan)
        print(f"\n  At focal_loc={focal}, scan from θ={theta_start:+.4f} CW:")
        if not folds:
            print(f"    NO γ-folds detected!")
            t2_all = False
        for ev in folds:
            theta_mid = 0.5 * (ev['theta_before'] + ev['theta_after'])
            print(f"    fold at θ ≈ {theta_mid:+.4f}  "
                  f"|Δγ|={ev['delta_gamma']:.4f}  "
                  f"severity={ev['severity']:.1f}×")
    print(f"\n  T2: {'PASS' if t2_all else 'FAIL'}")
    results.append(("T2 γ-folds detected at multistable points", t2_all))

    # =====================
    # T3 — Perception collapse in BlindSpot setup (b=π/2 cutoff)
    # =====================
    print()
    print("=" * 65)
    print("T3: perception collapse in BlindSpot setup (b=π/2 cutoff)")
    print("=" * 65)
    nbm_bs = setup_BlindSpot()
    focal_loc = np.array([0.0, 0.0])

    # Find a stable SC eq facing forward
    sc_angles, sc_stab = nbm_bs.sc_equilib(focal_loc=focal_loc,
                                            stability_criterion='coupled')
    stable = sorted([a for a, s in zip(sc_angles, sc_stab) if s])
    print(f"  Stable SC θ: {[f'{a:+.4f}' for a in stable]}")
    if not stable:
        print(f"  No stable SC eq found — using θ=0 with γ from relaxation")
        theta_start = 0.0
        gamma_start = _relax_gamma_cached(theta_start, focal_loc, 0.5+0j)
    else:
        # Pick the one closest to 0 (face forward)
        theta_start = min(stable, key=abs)
        gamma_start = _find_sc_gamma(nbm_bs, theta_start, focal_loc)
        if gamma_start is None:
            gamma_start = _relax_gamma_cached(theta_start, focal_loc, 0.5+0j)
    print(f"  Scan start: θ={theta_start:+.4f}, γ={gamma_start:+.4f}")

    # Scan around the full circle
    scan = run_scan(nbm_bs, focal_loc, theta_start, gamma_start,
                     direction='ccw', n_mesh=200)

    pcs = detect_perception_collapse(scan)
    folds = detect_gamma_folds(scan)
    print(f"\n  γ-folds: {len(folds)}")
    print(f"  perception collapse zones: {len(pcs)}")
    for ev in pcs:
        print(f"    θ in [{ev['theta_start']:+.4f}, "
              f"{ev['theta_end']:+.4f}]  "
              f"length = {ev['run_length']} samples  "
              f"mean R = {ev['mean_R']:.4e}  "
              f"mean |f| = {ev['mean_f']:.4e}")
    behind_zones = [ev for ev in pcs
                     if abs(ev['theta_start']) > np.pi / 2
                     or abs(ev['theta_end']) > np.pi / 2]
    t3_pass = len(behind_zones) > 0
    print(f"  T3: {'PASS' if t3_pass else 'FAIL'}")
    results.append(("T3 perception collapse in BlindSpot", t3_pass))

    # Diagnostic plot for T3
    order = np.argsort(scan['theta'])
    th_sorted = np.degrees(scan['theta'][order])
    R_sorted = scan['R'][order]
    f_sorted = scan['f'][order]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(th_sorted, R_sorted, 'k.-', markersize=3)
    axes[0].set_ylabel('R(θ)')
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].set_title('BlindSpot setup: scan around full circle')
    axes[1].plot(th_sorted, f_sorted, 'k.-', markersize=3)
    axes[1].axhline(0, color='gray', lw=0.5)
    axes[1].axhline(0.02, color='red', lw=0.5, ls=':',
                     label='|f| < 0.02 threshold')
    axes[1].axhline(-0.02, color='red', lw=0.5, ls=':')
    axes[1].set_xlabel('θ [°]')
    axes[1].set_ylabel('f(θ)')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    for ev in pcs:
        for ax in axes:
            ax.axvspan(np.degrees(ev['theta_start']),
                        np.degrees(ev['theta_end']),
                        color='red', alpha=0.2)
    if pcs:
        from matplotlib.patches import Patch
        axes[0].legend(
            handles=[Patch(facecolor='red', alpha=0.2,
                            label='perception collapse')],
            loc='lower right')
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'detect_discontinuities_blindspot.png')
    plt.savefig(plot_path, dpi=110)
    print(f"\n  Diagnostic plot saved to {plot_path}")

    # =====================
    # Summary
    # =====================
    print()
    print("=" * 65)
    n_pass = sum(1 for _, p in results if p)
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"\n{n_pass}/{len(results)} tests passed.")
    sys.exit(0 if n_pass == len(results) else 1)
