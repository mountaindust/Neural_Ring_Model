"""
Publication-quality 2x2 panel combining the upper-arc bifurcation skeleton
with long-time dynamics inside the 0-stable island.

Layout (von Mises, k=0.55, two circle targets at (4.33, +/- 2.5)):

  (a) upper-left   : # self-consistent equilibria over the upper-arc window,
                     with the Hopf curve overlaid in magenta. Higher-resolved
                     re-make of the LEFT panel of
                     VM_bifurcations/diagnostic_arc_skeleton.png. Colour
                     scheme follows the bifurcation_compare plots (discrete
                     viridis, BoundaryNorm). Legend lives in the middle-right
                     of the panel.
  (b) upper-right  : heading theta(t), 0 <= t <= 2000.
                     (= upper-left of diagnostic_island_long_dynamics.png,
                      truncated.)
  (c) lower-left   : |gamma|(t), 0 <= t <= 2000.
                     (= upper-right of diagnostic_island_long_dynamics.png,
                      truncated.)
  (d) lower-right  : ego_angle(t), full integration window.
                     (= lower-left of diagnostic_island_long_dynamics.png.)

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
from scipy.optimize import brentq, root
from scipy.integrate import solve_ivp
from multiprocessing import Pool

import decision_model as model


# ---- model setup (matches diagnostic_arc_skeleton.py / island_dynamics) ----
TARGET_LOCS = np.array([[4.33, 2.5], [4.33, -2.5]])
targets = model.Targets(locs=TARGET_LOCS, geom_name='circle', r=0.5)
percep = model.PerceptionModel(targets, (0, 0), 0,
                               neural_weight='vonmises',
                               neural_angle='integral')
percep.k = 0.55
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

N_WORKERS = 10

OUT_NAME = "arc_skeleton_and_island_dynamics.png"
HERE = os.path.dirname(os.path.abspath(__file__))
ARC_CACHE = os.path.join(HERE, "_cache_arc_skeleton.npz")
ISLAND_CACHE = os.path.join(HERE, "_cache_island_traj.npz")


# =========================================================================
# Arc-skeleton helpers (module-level so they pickle for multiprocessing)
# =========================================================================
def find_eqs(focal_loc, R_probe=0.5):
    theta = np.linspace(-np.pi, np.pi, 2001)
    im = np.array([nbm.dgamma_dt(gamma=R_probe + 0j, focal_angle=t,
                                 focal_loc=focal_loc).imag for t in theta])
    candidates = []
    for i in range(len(theta) - 1):
        if im[i] * im[i + 1] < 0:
            try:
                tc = brentq(lambda t: nbm.dgamma_dt(
                    gamma=R_probe + 0j, focal_angle=t,
                    focal_loc=focal_loc).imag, theta[i], theta[i + 1])
                candidates.append(tc)
            except ValueError:
                pass
    for extra in (0.0, np.pi, -np.pi):
        candidates.append(extra)
    eqs = []
    for tc in candidates:
        sol = root(nbm._self_consistent_eq, [tc, R_probe],
                   args=(focal_loc,), method='hybr', tol=1e-12)
        if not sol.success:
            continue
        teq = model.convert_angles(sol.x[0])
        Req = sol.x[1]
        if Req < 0.01 or Req > 1.0:
            continue
        residual = nbm.dgamma_dt(gamma=Req + 0j, focal_angle=teq,
                                 focal_loc=focal_loc)
        if abs(residual) > 1e-7:
            continue
        if any(abs(model.convert_angles(teq - e[0])) < 1e-3 for e in eqs):
            continue
        eqs.append((teq, Req))
    return eqs


def coupled_rhs(y, focal_loc):
    gr, gi, th = y
    gamma = gr + 1j * gi
    dg = nbm.dgamma_dt(gamma=gamma, focal_angle=th, focal_loc=focal_loc)
    ego, R = nbm.convert_gamma(gamma)
    return np.array([dg.real, dg.imag, K * R * np.sin(ego)])


def coupled_eigs(focal_loc, theta_eq, R_eq, h=1e-6):
    y0 = np.array([R_eq, 0.0, theta_eq])
    J = np.zeros((3, 3))
    for k in range(3):
        yp = y0.copy(); yp[k] += h
        ym = y0.copy(); ym[k] -= h
        J[:, k] = (coupled_rhs(yp, focal_loc) -
                   coupled_rhs(ym, focal_loc)) / (2 * h)
    return np.linalg.eigvals(J)


def cell_summary(args):
    """Return (key, n_total, hopf_indicator). hopf_indicator = max real part
    of complex eigenvalues across all equilibria at this cell, NaN if no
    complex eigenvalues exist."""
    key, x, y = args
    fl = np.array([x, y])
    eqs = find_eqs(fl)
    n_total = len(eqs)
    hopf_re = -np.inf
    for (teq, Req) in eqs:
        eigs = coupled_eigs(fl, teq, Req)
        complex_mask = np.abs(np.imag(eigs)) > 1e-6
        if complex_mask.any():
            cre = float(np.max(np.real(eigs[complex_mask])))
            if cre > hopf_re:
                hopf_re = cre
    if not np.isfinite(hopf_re):
        hopf_re = np.nan
    return key, n_total, hopf_re


# =========================================================================
# Compute / cache the arc-skeleton grid
# =========================================================================
def compute_arc_grid(regenerate=False):
    if (not regenerate) and os.path.exists(ARC_CACHE):
        d = np.load(ARC_CACHE)
        if (d['nx'] == ARC_NX and d['ny'] == ARC_NY
                and tuple(d['xlim']) == ARC_XLIM
                and tuple(d['ylim']) == ARC_YLIM):
            print(f"Loaded arc-skeleton grid from {ARC_CACHE}")
            return d['xs'], d['ys'], d['grid_n'], d['grid_hopf']
        print("Cache parameters mismatch; recomputing arc-skeleton grid.")

    xs = np.linspace(ARC_XLIM[0], ARC_XLIM[1], ARC_NX)
    ys = np.linspace(ARC_YLIM[0], ARC_YLIM[1], ARC_NY)
    args_list = [((j, i), xs[i], ys[j])
                 for j in range(ARC_NY) for i in range(ARC_NX)]
    print(f"Scanning arc-skeleton grid: {ARC_NX}x{ARC_NY} = "
          f"{ARC_NX * ARC_NY} cells...")
    with Pool(N_WORKERS) as pool:
        results = pool.map(cell_summary, args_list)

    grid_n = np.zeros((ARC_NY, ARC_NX), dtype=int)
    grid_hopf = np.full((ARC_NY, ARC_NX), np.nan)
    for (j, i), n, h in results:
        grid_n[j, i] = n
        grid_hopf[j, i] = h

    np.savez(ARC_CACHE, xs=xs, ys=ys, grid_n=grid_n, grid_hopf=grid_hopf,
             nx=ARC_NX, ny=ARC_NY, xlim=np.array(ARC_XLIM),
             ylim=np.array(ARC_YLIM))
    print(f"Saved arc-skeleton grid to {ARC_CACHE}")
    return xs, ys, grid_n, grid_hopf


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
    xs, ys, grid_n, grid_hopf = compute_arc_grid(regenerate=regenerate)
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
    ax_arc, ax_th, ax_R, ax_ego = (axes[0, 0], axes[0, 1],
                                    axes[1, 0], axes[1, 1])

    # ---- (a) arc skeleton ----
    ax_arc.imshow(grid_n, origin='lower',
                  extent=[ARC_XLIM[0], ARC_XLIM[1],
                          ARC_YLIM[0], ARC_YLIM[1]],
                  aspect='equal', interpolation='nearest',
                  cmap=cmap_n, norm=norm_n)
    cs = ax_arc.contour(xs, ys, grid_hopf, levels=[0.0],
                        colors='magenta', linewidths=2)
    targets.plot_targets_to_axis(ax_arc)
    ax_arc.set_xlim(ARC_XLIM)
    ax_arc.set_ylim(ARC_YLIM)
    ax_arc.set_xlabel('focal x')
    ax_arc.set_ylabel('focal y')
    ax_arc.set_title('(a) # self-consistent equilibria, with Hopf curve')

    # Legend: integer-count swatches + Hopf curve handle, mid-right.
    legend_handles = [Line2D([], [], marker='s', linestyle='',
                             markersize=11, color=cmap_n(norm_n(n)),
                             label=str(n))
                      for n in range(nmax + 1)]
    legend_handles.append(Line2D([], [], color='magenta', linewidth=2,
                                  label='Hopf curve'))
    ax_arc.legend(handles=legend_handles,
                  title='# equilibria',
                  loc='center right', frameon=True, framealpha=0.92,
                  fontsize=9, title_fontsize=9)

    # ---- (b) heading vs time, t <= 2000 ----
    mask_trunc = t <= ISLAND_T_TRUNC
    ax_th.plot(t[mask_trunc], th_wrapped[mask_trunc], lw=0.5)
    ax_th.set_xlabel('t')
    ax_th.set_ylabel(r'$\theta$ (heading)')
    ax_th.set_title(r'(b) heading $\theta(t)$, $t \leq 2000$')
    ax_th.set_xlim(0, ISLAND_T_TRUNC)

    # ---- (c) |gamma| vs time, t <= 2000 ----
    ax_R.plot(t[mask_trunc], R_arr[mask_trunc], lw=0.5)
    ax_R.set_xlabel('t')
    ax_R.set_ylabel(r'$|\gamma|$')
    ax_R.set_title(r'(c) $|\gamma|(t)$, $t \leq 2000$')
    ax_R.set_xlim(0, ISLAND_T_TRUNC)

    # ---- (d) ego_angle vs time, full integration ----
    ax_ego.plot(t, ego_arr, lw=0.5)
    ax_ego.set_xlabel('t')
    ax_ego.set_ylabel('ego angle')
    ax_ego.set_title(r'(d) ego angle $(t)$, full integration')
    ax_ego.set_xlim(0, ISLAND_T_FINAL)

    fig.suptitle('Upper 0-stable arc and limit-cycle dynamics inside the '
                 'island\n'
                 r'(von Mises, $k=0.55$, two circle targets at '
                 r'$(4.33, \pm 2.5)$; island sample at '
                 fr'$(x,y)=({ISLAND_FOCAL_LOC[0]:.2f},'
                 fr'{ISLAND_FOCAL_LOC[1]:.2f})$)',
                 fontsize=13, y=0.985)

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
