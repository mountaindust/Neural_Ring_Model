"""Distinguish Hopf and SN curves cleanly.

For each cell, find all self-consistent equilibria. For each equilibrium,
compute the coupled 3x3 Jacobian eigenvalues. Track:
  - The MAXIMUM real-part across only the COMPLEX-PAIR eigenvalues (this is
    the Hopf-relevant indicator: it goes through 0 at a Hopf bifurcation).
  - Whether the equilibrium is a saddle (one purely real positive eigenvalue
    with the other two having negative real parts).
  - The number of equilibria (changes only across SN curves).

Then plot:
  - n_eqs map (yellow = 3, green = 2, blue = 1, purple = 0)
  - Hopf curve = level set of max_complex_re = 0, only over cells where the
    relevant eq is NOT a saddle (so we get the "true" Hopf curve, not the SN
    curve being misread).
  - n_stable map for reference.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from scipy.optimize import brentq, root
from multiprocessing import Pool

import decision_model as model
from parallel_config import get_n_workers

target_locs = np.array([[4.33, 2.5], [4.33, -2.5]])
targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)
percep = model.PerceptionModel(targets, (0, 0), 0,
                               neural_angle_dist='vonmises',
                               angle_weight='neural_angle_dist',
                               a_warp=0.55)
nbm = model.NeuralBandModel(percep)
K = nbm.K


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


def coupled_eigs(focal_loc, theta_eq, R_eq, h=1e-6):
    def rhs(y):
        gr, gi, th = y
        dg = nbm.dgamma_dt(gamma=gr + 1j * gi, focal_angle=th,
                           focal_loc=focal_loc)
        ego, R = nbm.convert_gamma(gr + 1j * gi)
        return np.array([dg.real, dg.imag, K * R * np.sin(ego)])
    y0 = np.array([R_eq, 0.0, theta_eq])
    J = np.zeros((3, 3))
    for k in range(3):
        yp = y0.copy(); yp[k] += h
        ym = y0.copy(); ym[k] -= h
        J[:, k] = (rhs(yp) - rhs(ym)) / (2 * h)
    return np.linalg.eigvals(J)


def cell_summary(args):
    """For each cell, return:
    - n_total: total self-consistent eqs
    - n_stable: coupled-stable count
    - hopf_indicator: max-Re of complex eigvals across non-saddle eqs (or NaN)
    - saddle_present: True if any eq has 1 real+ + 2 negative-real eigenvalues
    """
    key, x, y = args
    fl = np.array([x, y])
    eqs = find_eqs(fl)
    n_total = len(eqs)
    n_stable = 0
    hopf_re = -np.inf
    saddle = False
    for (teq, Req) in eqs:
        eigs = coupled_eigs(fl, teq, Req)
        max_re = float(np.max(np.real(eigs)))
        if max_re < -1e-8:
            n_stable += 1
        # Identify saddle: real eigvals only, exactly one positive
        real_mask = np.abs(np.imag(eigs)) < 1e-6
        if real_mask.all():
            n_pos = int(np.sum(np.real(eigs) > 1e-8))
            if n_pos == 1:
                saddle = True
        # Hopf indicator: max real part among COMPLEX eigvalues
        complex_mask = np.abs(np.imag(eigs)) > 1e-6
        if complex_mask.any():
            cre = float(np.max(np.real(eigs[complex_mask])))
            if cre > hopf_re:
                hopf_re = cre
    if not np.isfinite(hopf_re):
        hopf_re = np.nan
    return key, n_total, n_stable, hopf_re, saddle


if __name__ == "__main__":
    xlim = (1.0, 3.5)
    ylim = (1.5, 2.8)
    nx, ny = 121, 79
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    args_list = [((j, i), xs[i], ys[j])
                 for j in range(ny) for i in range(nx)]
    print(f"Scanning {nx}x{ny}={nx*ny} cells (focused window)...")
    with Pool(get_n_workers()) as pool:
        results = pool.map(cell_summary, args_list)
    grid_n = np.zeros((ny, nx), dtype=int)
    grid_s = np.zeros((ny, nx), dtype=int)
    grid_hopf = np.full((ny, nx), np.nan)
    grid_sadd = np.zeros((ny, nx), dtype=bool)
    for (j, i), n, s, h, sd in results:
        grid_n[j, i] = n
        grid_s[j, i] = s
        grid_hopf[j, i] = h
        grid_sadd[j, i] = sd

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: # equilibria with overlaid Hopf curve from complex-eig contour
    nmax = max(grid_n.max(), 3)
    cmap_n = plt.get_cmap('viridis', nmax + 1)
    norm_n = BoundaryNorm(boundaries=np.arange(-0.5, nmax + 1.5),
                          ncolors=nmax + 1)
    axes[0].imshow(grid_n, origin='lower',
                   extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   aspect='equal', interpolation='nearest',
                   cmap=cmap_n, norm=norm_n)
    cs1 = axes[0].contour(xs, ys, grid_hopf, levels=[0.0],
                          colors='magenta', linewidths=2)
    try:
        axes[0].clabel(cs1, inline=True, fmt='Hopf', fontsize=9)
    except Exception:
        pass
    targets.plot_targets_to_axis(axes[0])
    axes[0].set_title("# self-consistent equilibria\n"
                      "magenta = Hopf curve (complex eig real-part = 0)\n"
                      "color boundary = saddle-node curve")
    axes[0].set_xlim(xlim); axes[0].set_ylim(ylim)

    # Panel 2: # coupled-stable + Hopf
    nmax_s = max(grid_s.max(), 3)
    cmap_s = plt.get_cmap('viridis', nmax_s + 1)
    norm_s = BoundaryNorm(boundaries=np.arange(-0.5, nmax_s + 1.5),
                          ncolors=nmax_s + 1)
    axes[1].imshow(grid_s, origin='lower',
                   extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   aspect='equal', interpolation='nearest',
                   cmap=cmap_s, norm=norm_s)
    cs2 = axes[1].contour(xs, ys, grid_hopf, levels=[0.0],
                          colors='magenta', linewidths=2)
    try:
        axes[1].clabel(cs2, inline=True, fmt='Hopf', fontsize=9)
    except Exception:
        pass
    targets.plot_targets_to_axis(axes[1])
    axes[1].set_title("# coupled-stable equilibria\n"
                      "(0-stable arc = inside Hopf curve, outside SN curve)")
    axes[1].set_xlim(xlim); axes[1].set_ylim(ylim)

    fig.suptitle("Bifurcation skeleton near the upper 0-stable arc")
    fig.tight_layout()
    fig.savefig("diagnostic_arc_skeleton.png", dpi=120)
    print("saved diagnostic_arc_skeleton.png")
