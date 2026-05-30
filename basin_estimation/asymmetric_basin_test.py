"""
Step 9 of basin-of-attraction vetting plan.

At an asymmetric 2-stable calibration point — observer significantly
closer to one target than the other — verify that the close-target
stable SC equilibrium has a *wider* basin in θ than the far-target
stable. This is the user's prior intuition from the planning notes:

  "the asymptotic dynamics that the walker follows... noise can throw
   the system into a different basin, so noise-robustness of each SC
   equilibrium matters. Expected qualitative behavior: in a bistable
   region where the observer is very close to one circular target and
   the other is far, the far target's basin should be small."

Calibration point: (4.0, 1.5) in VM-k055.

  Target 1 at (4.33, 2.5):  distance 1.05, allocentric +71.7°.
  Target 2 at (4.33, -2.5): distance 4.01, allocentric -85.3°.
  Distance ratio: 3.81×.

  sc_equilib reports 2 stable θ:
    -1.488 (faces target 2 — the FAR target)
    +1.252 (faces target 1 — the CLOSE target)

Test:
  T1: total basin width Δθ_ccw + Δθ_cw of the close-target stable
      is strictly larger than that of the far-target stable.

This test uses the truncated-scan basin extractor from Step 5.

Usage:
  python asymmetric_basin_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import decision_model as model

from basin_via_theta import basin_features
from theta_scan import nbm, _relax_gamma_cached


def _find_sc_gamma(theta_sc, focal_loc):
    gammas, _ = nbm.gamma_equilib(focal_angle=theta_sc, focal_loc=focal_loc,
                                   stability_criterion='discrim_a')
    for g in gammas:
        if abs(np.angle(g)) < 0.1 and abs(g) > 0.05:
            return g
    return None


if __name__ == "__main__":
    # ---- Setup ----
    focal_loc = np.array([4.0, 1.5])
    target_locs = np.array([[4.33, 2.5], [4.33, -2.5]])

    # Distances and allocentric angles
    d = [np.linalg.norm(t - focal_loc) for t in target_locs]
    allo = [np.degrees(np.arctan2((t - focal_loc)[1], (t - focal_loc)[0]))
            for t in target_locs]
    print(f"Calibration point: focal_loc = {tuple(focal_loc)}")
    print(f"  Target 1 at {tuple(target_locs[0])}: d = {d[0]:.3f}, "
          f"allo = {allo[0]:+.1f}°")
    print(f"  Target 2 at {tuple(target_locs[1])}: d = {d[1]:.3f}, "
          f"allo = {allo[1]:+.1f}°")
    print(f"  Distance ratio: {max(d)/min(d):.2f}")

    sc_angles, sc_stab = nbm.sc_equilib(focal_loc=focal_loc,
                                         stability_criterion='coupled')
    stable_thetas = sorted([a for a, s in zip(sc_angles, sc_stab) if s])
    print(f"\nStable SC θ values: {[f'{a:+.4f}' for a in stable_thetas]}")
    assert len(stable_thetas) == 2

    # Identify which stable corresponds to which target by angular proximity
    def closest_target_idx(theta):
        # Find target whose allocentric angle is closest to theta
        diffs = [abs(((theta - np.radians(allo[i]) + np.pi)
                       % (2 * np.pi)) - np.pi)
                  for i in range(len(target_locs))]
        return int(np.argmin(diffs))

    basin_data = []
    for theta_s in stable_thetas:
        tgt_idx = closest_target_idx(theta_s)
        tgt_distance = d[tgt_idx]
        gamma_s = _find_sc_gamma(theta_s, focal_loc)
        assert gamma_s is not None, f"Could not recover γ at θ={theta_s}"
        b = basin_features(focal_loc, theta_s, gamma_s,
                            n_max=300, gamma_jump_factor=8.0)
        total_basin_width = b['delta_theta_ccw'] + b['delta_theta_cw']
        basin_data.append({
            'theta_s': theta_s,
            'gamma_s': gamma_s,
            'tgt_idx': tgt_idx,
            'tgt_distance': tgt_distance,
            'tgt_allo_deg': allo[tgt_idx],
            'b': b,
            'total_width': total_basin_width,
        })
        label = f'target {tgt_idx + 1} (d={tgt_distance:.2f})'
        print(f"\n  Basin of θ_s = {theta_s:+.4f} [{label}]:")
        print(f"    γ_s = {gamma_s:+.5f}")
        print(f"    CCW endpoint: θ = {b['ccw_endpoint']:+.4f} "
              f"({b['ccw_type']:7})  Δθ = {b['delta_theta_ccw']:.4f}  "
              f"ΔV = {b['delta_V_ccw']:+.4e}")
        print(f"    CW  endpoint: θ = {b['cw_endpoint']:+.4f} "
              f"({b['cw_type']:7})  Δθ = {b['delta_theta_cw']:.4f}  "
              f"ΔV = {b['delta_V_cw']:+.4e}")
        print(f"    Total basin width (Δθ_ccw + Δθ_cw): "
              f"{total_basin_width:.4f} rad "
              f"= {np.degrees(total_basin_width):.1f}°")

    # ---- T1: close-target basin > far-target basin ----
    close = min(basin_data, key=lambda b: b['tgt_distance'])
    far = max(basin_data, key=lambda b: b['tgt_distance'])
    print(f"\n--- Comparison ---")
    print(f"  Close-target stable (target {close['tgt_idx']+1} at "
          f"d={close['tgt_distance']:.2f}): "
          f"basin width = {close['total_width']:.4f} rad")
    print(f"  Far-target   stable (target {far['tgt_idx']+1} at "
          f"d={far['tgt_distance']:.2f}): "
          f"basin width = {far['total_width']:.4f} rad")
    ratio = close['total_width'] / far['total_width']
    print(f"  Ratio (close / far): {ratio:.3f}")

    t1_pass = close['total_width'] > far['total_width']
    print(f"\nT1: close basin > far basin: {'PASS' if t1_pass else 'FAIL'}")

    # ---- Diagnostic plot ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    from scipy.integrate import cumulative_trapezoid
    colors = {0: 'C0', 1: 'C3'}  # target 1 blue, target 2 red

    for bd in basin_data:
        b = bd['b']
        tgt = bd['tgt_idx']
        c = colors[tgt]
        scan_ccw = b['ccw_scan']
        scan_cw = b['cw_scan']
        deg_ccw = np.degrees(scan_ccw['theta'])
        deg_cw = np.degrees(scan_cw['theta'])
        V_ccw = cumulative_trapezoid(-scan_ccw['f'], scan_ccw['theta'],
                                       initial=0)
        V_cw = cumulative_trapezoid(-scan_cw['f'], scan_cw['theta'],
                                      initial=0)
        label = (f'target {tgt+1} '
                 f'(d={bd["tgt_distance"]:.2f}, '
                 f'allo={bd["tgt_allo_deg"]:+.1f}°)')
        axes[0].plot(deg_ccw, scan_ccw['R'], 'o-', color=c,
                      markersize=3, label=label)
        axes[0].plot(deg_cw, scan_cw['R'], 'o-', color=c, markersize=3)
        axes[1].plot(deg_ccw, V_ccw, 'o-', color=c, markersize=3,
                      label=label)
        axes[1].plot(deg_cw, V_cw, 'o-', color=c, markersize=3)
        # Mark stable + endpoints
        axes[0].axvline(np.degrees(bd['theta_s']), color=c, ls='--',
                         alpha=0.5)
        axes[1].axvline(np.degrees(bd['theta_s']), color=c, ls='--',
                         alpha=0.5)
        for ep, et in [(b['ccw_endpoint'], b['ccw_type']),
                         (b['cw_endpoint'], b['cw_type'])]:
            marker = '*' if et == 'saddle' else 'X'
            axes[0].plot(np.degrees(ep), 0.05, marker, color=c,
                          markersize=12, markeredgecolor='k')

    axes[0].set_ylabel('R(θ)')
    axes[0].set_title(f'Asymmetric basin test: focal_loc = '
                       f'{tuple(focal_loc)} '
                       f'(distance ratio {max(d)/min(d):.2f}×)\n'
                       f'★ = saddle endpoint, ✗ = γ-fold endpoint')
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc='best')
    axes[1].set_ylabel('V(θ)')
    axes[1].set_xlabel('θ [°]')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'asymmetric_basin.png')
    plt.savefig(plot_path, dpi=110)
    print(f"\nDiagnostic plot saved to {plot_path}")

    # ---- Summary ----
    print()
    print("=" * 60)
    print(f"T1 close basin > far basin: {'PASS' if t1_pass else 'FAIL'}")
    print(f"\n1/{1} tests passed." if t1_pass else f"\n0/1 tests passed.")
    sys.exit(0 if t1_pass else 1)
