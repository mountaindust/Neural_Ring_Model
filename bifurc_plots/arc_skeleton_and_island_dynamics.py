"""
Publication-quality 2x2 panel combining the upper-arc bifurcation skeleton
with long-time dynamics inside the 0-stable island.

Layout (von Mises, k=0.55, two circle targets at (4.33, +/- 2.5)):

  (a) upper-left   : # stable self-consistent equilibria over the upper-arc
                     window. Adapted from the LEFT panel of
                     VM_bifurcation_old_dtheta/diagnostic_arc_skeleton.png; the legacy
                     panel showed total equilibria with a Hopf-curve overlay,
                     but at high resolution that pipeline produced speckles
                     and a noisy Hopf contour. Switching to the stable-count
                     map (computed via NeuralBandModel._count_stable_at, the
                     same maintained API used by bifurcation_compare) cleans
                     the picture and makes the 0-stable arc visible directly
                     as a dark band, removing the need for an explicit Hopf
                     overlay. Colour scheme follows the bifurcation_compare
                     plots. The (2.10, 2.45) island sample point is marked
                     with a red X. Legend lives in the middle-right.
  (b) upper-right  : heading theta(t), 0 <= t <= 2000.
                     (= upper-left of diagnostic_island_long_dynamics.png,
                      truncated.)
  (c) lower-left   : |gamma|(t), 0 <= t <= 2000.
                     (= upper-right of diagnostic_island_long_dynamics.png,
                      truncated.)
  (d) lower-right  : phase portrait (theta, ego_angle) for t > 1000, with
                     the unstable equilibrium marked.
                     (= lower-right of diagnostic_island_long_dynamics.png.)

The arc-skeleton grid and the long-time integration are both cached to
.npy/.npz files in this directory so layout iteration is cheap; delete the
cache files (or pass --regenerate) to force recomputation.
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from multiprocessing import Pool

import decision_model as model
from parallel_config import get_n_workers


# ---- model setup (matches diagnostic_arc_skeleton.py / island_dynamics) ----
TARGET_LOCS = np.array([[4.33, 2.5], [4.33, -2.5]])
targets = model.Targets(locs=TARGET_LOCS, geom_name='circle', r=0.5)
percep = model.PerceptionModel(targets, (0, 0), 0,
                               neural_angle_dist='vonmises',
                               angle_weight='neural_angle_dist',
                               a_warp=0.55)
nbm = model.NeuralBandModel(percep)
K = nbm.K


# ---- arc-skeleton grid window and resolution ----
ARC_XLIM = (1.0, 3.5)
ARC_YLIM = (1.5, 2.8)
ARC_NX = 251      # was 121 in the diagnostic
ARC_NY = 131      # was  79 in the diagnostic

# ---- island long-time dynamics ----
ISLAND_FOCAL_LOC = np.array([2.10, 2.45])
ISLAND_THETA_EQ = -0.0876     # location of the unstable equilibrium
ISLAND_T_FINAL = 4000.0
ISLAND_T_TRUNC = 2000.0       # cutoff for theta and |gamma| panels

N_WORKERS = get_n_workers()

OUT_NAME = "arc_skeleton_and_island_dynamics.png"
HERE = os.path.dirname(os.path.abspath(__file__))
# Cache filename includes "stable" so old grids (which stored grid_hopf and
# total counts) are not picked up after the schema change.
ARC_CACHE = os.path.join(HERE, "_cache_arc_skeleton_stable.npz")
ISLAND_CACHE = os.path.join(HERE, "_cache_island_traj.npz")


# Coupled (gamma_re, gamma_im, theta) RHS used by the island integration.
def coupled_rhs(y, focal_loc):
    gr, gi, th = y
    gamma = gr + 1j * gi
    dg = nbm.dgamma_dt(gamma=gamma, focal_angle=th, focal_loc=focal_loc)
    R = np.abs(gamma)
    # Half-angle torque in the neural consensus angle arg(gamma), matching
    # NeuralBandModel.dtheta_dt / _discrim_coupled (K*R*sin(arg(gamma)/2)).
    return np.array([dg.real, dg.imag, K * R * np.sin(np.angle(gamma) / 2)])


# =========================================================================
# Compute / cache the arc-skeleton grid
# =========================================================================
def compute_arc_grid(regenerate=False):
    """Evaluate # stable self-consistent equilibria on a uniform grid via
    NeuralBandModel._count_stable_at (the same routine the bifurcation
    plots use). Returns grid_n with shape (ARC_NY, ARC_NX)."""
    if (not regenerate) and os.path.exists(ARC_CACHE):
        d = np.load(ARC_CACHE)
        if (d['nx'] == ARC_NX and d['ny'] == ARC_NY
                and tuple(d['xlim']) == ARC_XLIM
                and tuple(d['ylim']) == ARC_YLIM):
            print(f"Loaded arc-skeleton grid from {ARC_CACHE}")
            return d['grid_n']
        print("Cache parameters mismatch; recomputing arc-skeleton grid.")

    xs = np.linspace(ARC_XLIM[0], ARC_XLIM[1], ARC_NX)
    ys = np.linspace(ARC_YLIM[0], ARC_YLIM[1], ARC_NY)
    args_list = [((j, i), xs[i], ys[j], 'coupled')
                 for j in range(ARC_NY) for i in range(ARC_NX)]
    print(f"Scanning arc-skeleton grid: {ARC_NX}x{ARC_NY} = "
          f"{ARC_NX * ARC_NY} cells (# stable equilibria, coupled)...")
    with Pool(N_WORKERS) as pool:
        results = pool.map(nbm._count_stable_at, args_list)

    grid_n = np.zeros((ARC_NY, ARC_NX), dtype=int)
    for (j, i), c in results:
        grid_n[j, i] = c

    np.savez(ARC_CACHE, grid_n=grid_n,
             nx=ARC_NX, ny=ARC_NY, xlim=np.array(ARC_XLIM),
             ylim=np.array(ARC_YLIM))
    print(f"Saved arc-skeleton grid to {ARC_CACHE}")
    return grid_n


# =========================================================================
# Compute / cache the long-time island trajectory
# =========================================================================
def compute_island_trajectory(regenerate=False):
    if (not regenerate) and os.path.exists(ISLAND_CACHE):
        d = np.load(ISLAND_CACHE)
        if float(d['t_final']) == ISLAND_T_FINAL and \
                np.allclose(d['focal_loc'], ISLAND_FOCAL_LOC):
            print(f"Loaded island trajectory from {ISLAND_CACHE}")
            return d['t'], d['gr'], d['gi'], d['th']
        print("Cache parameters mismatch; recomputing island trajectory.")

    print(f"Integrating coupled system at focal_loc={ISLAND_FOCAL_LOC} "
          f"to t={ISLAND_T_FINAL}...")
    y0 = [0.05, 0.0, 0.0]   # init_gamma=0.05+0j, init_theta=0

    def rhs(t, y):
        return coupled_rhs(y, ISLAND_FOCAL_LOC)

    sol = solve_ivp(rhs, [0, ISLAND_T_FINAL], y0,
                    method='LSODA', rtol=1e-10, atol=1e-12, max_step=2.0)
    np.savez(ISLAND_CACHE, t=sol.t, gr=sol.y[0], gi=sol.y[1], th=sol.y[2],
             t_final=np.array(ISLAND_T_FINAL),
             focal_loc=ISLAND_FOCAL_LOC)
    print(f"Saved island trajectory to {ISLAND_CACHE}")
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


# =========================================================================
# Plot
# =========================================================================
def main(regenerate=False):
    grid_n = compute_arc_grid(regenerate=regenerate)
    t, gr, gi, th = compute_island_trajectory(regenerate=regenerate)
    gamma = gr + 1j * gi
    R_arr = np.abs(gamma)
    ego_arr = np.array([nbm.convert_gamma(g)[0] for g in gamma])
    th_wrapped = model.convert_angles(th)

    nmax = int(max(grid_n.max(), 3))
    cmap_n = plt.get_cmap('viridis', nmax + 1)
    norm_n = BoundaryNorm(boundaries=np.arange(-0.5, nmax + 1.5),
                          ncolors=nmax + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_arc, ax_th, ax_R, ax_phase = (axes[0, 0], axes[0, 1],
                                      axes[1, 0], axes[1, 1])

    # ---- (a) arc skeleton: # stable self-consistent equilibria ----
    ax_arc.imshow(grid_n, origin='lower',
                  extent=[ARC_XLIM[0], ARC_XLIM[1],
                          ARC_YLIM[0], ARC_YLIM[1]],
                  aspect='equal', interpolation='nearest',
                  cmap=cmap_n, norm=norm_n)
    targets.plot_targets_to_axis(ax_arc)
    # Mark the island sample point used for panels (b)-(d).
    ax_arc.plot([ISLAND_FOCAL_LOC[0]], [ISLAND_FOCAL_LOC[1]],
                marker='x', color='red', markersize=12,
                markeredgewidth=2.5, linestyle='')
    ax_arc.set_xlim(ARC_XLIM)
    ax_arc.set_ylim(ARC_YLIM)
    ax_arc.set_xlabel('observer x-coordinate', fontsize=12)
    ax_arc.set_ylabel('observer y-coordinate', fontsize=12)
    ax_arc.set_title('(a) # stable self-consistent equilibria')

    # Legend: integer-count swatches + island-sample marker, mid-right.
    legend_handles = [Line2D([], [], marker='s', linestyle='',
                             markersize=11, color=cmap_n(norm_n(n)),
                             label=str(n))
                      for n in range(nmax + 1)]
    legend_handles.append(Line2D([], [], marker='x', linestyle='',
                                  color='red', markersize=10,
                                  markeredgewidth=2.5,
                                  label='island sample'))
    ax_arc.legend(handles=legend_handles,
                  title='# stable\nequilibria',
                  loc='upper right', frameon=True, framealpha=0.92,
                  fontsize=9, title_fontsize=9)

    # ---- (b) heading vs time, t <= 2000 ----
    mask_trunc = t <= ISLAND_T_TRUNC
    ax_th.plot(t[mask_trunc], th_wrapped[mask_trunc], lw=0.5)
    ax_th.set_xlabel('t', fontsize=12)
    ax_th.set_ylabel(r'$\theta$ (heading)', fontsize=12)
    ax_th.set_title(r'(b) heading $\theta(t)$, $t \leq 2000$')
    ax_th.set_xlim(0, ISLAND_T_TRUNC)

    # ---- (c) |gamma| vs time, t <= 2000 ----
    ax_R.plot(t[mask_trunc], R_arr[mask_trunc], lw=0.5)
    ax_R.set_xlabel('t', fontsize=12)
    ax_R.set_ylabel(r'$|\gamma|$', fontsize=12)
    ax_R.set_title(r'(c) $|\gamma|(t)$, $t \leq 2000$')
    ax_R.set_xlim(0, ISLAND_T_TRUNC)

    # ---- (d) phase portrait (theta, ego_angle) for t > 1000 ----
    late_mask = t > 1000
    ax_phase.plot(th_wrapped[late_mask], ego_arr[late_mask], lw=0.5)
    ax_phase.plot([ISLAND_THETA_EQ], [0], 'rx', markersize=12,
                  label=fr'unstable eq ($\theta={ISLAND_THETA_EQ:.3f}$)')
    ax_phase.set_xlabel(r'$\theta$', fontsize=12)
    ax_phase.set_ylabel('ego angle', fontsize=12)
    ax_phase.set_title(r'(d) phase portrait, $t > 1000$')
    ax_phase.legend(loc='upper left', fontsize=9)

    fig.suptitle('Upper 0-stable arc and limit-cycle dynamics inside the '
                 'island\n'
                 r'(von Mises, $k=0.55$, two circle targets at '
                 r'$(4.33, \pm 2.5)$; island sample at '
                 fr'$(x,y)=({ISLAND_FOCAL_LOC[0]:.2f},'
                 fr'{ISLAND_FOCAL_LOC[1]:.2f})$)',
                 fontsize=14, y=0.985)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.07,
                        wspace=0.22, hspace=0.28)

    out_path = os.path.join(HERE, OUT_NAME)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regenerate', action='store_true',
                        help='Force recomputation of cached grid/trajectory.')
    args = parser.parse_args()
    main(regenerate=args.regenerate)
