"""Re-render the island map but compute the *bifurcation count* (number of
truly stable equilibria) at every cell, not the max-Re of the most unstable
equilibrium. This is what corresponds to a "0-stable" cell in the
bifurcation diagram."""
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

target_locs = np.array([[4.33, 2.5], [4.33, -2.5]])
targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)
percep = model.PerceptionModel(targets, (0, 0), 0,
                               neural_weight='vonmises',
                               neural_angle='integral')
percep.k = 0.55
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


def coupled_jacobian_max_re(focal_loc, theta_eq, R_eq, h=1e-6):
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
    return float(np.max(np.real(np.linalg.eigvals(J))))


def cell_data(args):
    """Return key, n_total, n_coupled_stable, max_re_overall."""
    key, x, y = args
    fl = np.array([x, y])
    eqs = find_eqs(fl)
    n_total = len(eqs)
    n_stable = 0
    max_re = -np.inf
    for (teq, Req) in eqs:
        m = coupled_jacobian_max_re(fl, teq, Req)
        if m > max_re:
            max_re = m
        if m < -1e-8:
            n_stable += 1
    if not np.isfinite(max_re):
        max_re = np.nan
    return key, n_total, n_stable, max_re


if __name__ == "__main__":
    xlim = (1.0, 3.0)
    ylim = (1.5, 3.0)
    nx, ny = 81, 61
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    args_list = [((j, i), xs[i], ys[j])
                 for j in range(ny) for i in range(nx)]
    print(f"Scanning {nx}x{ny}={nx*ny} cells...")
    # HW-TEMP: 4-core laptop; restore to 10 on main workstation
    with Pool(4) as pool:
        results = pool.map(cell_data, args_list)
    grid_n = np.zeros((ny, nx), dtype=int)
    grid_s = np.zeros((ny, nx), dtype=int)
    grid_re = np.full((ny, nx), np.nan)
    for (j, i), n, s, m in results:
        grid_n[j, i] = n
        grid_s[j, i] = s
        grid_re[j, i] = m

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    nmax_n = max(grid_n.max(), 3)
    cmap = plt.get_cmap('viridis', nmax_n + 1)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, nmax_n + 1.5),
                        ncolors=nmax_n + 1)
    im0 = axes[0].imshow(grid_n, origin='lower',
                         extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                         aspect='equal', interpolation='nearest',
                         cmap=cmap, norm=norm)
    targets.plot_targets_to_axis(axes[0])
    axes[0].set_title("# self-consistent equilibria (total)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(grid_s, origin='lower',
                         extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                         aspect='equal', interpolation='nearest',
                         cmap=cmap, norm=norm)
    targets.plot_targets_to_axis(axes[1])
    axes[1].set_title("# coupled-stable equilibria")
    plt.colorbar(im1, ax=axes[1])

    # Highlight cells with 0 stable
    zero_stable = (grid_s == 0)
    print(f"\nTrue 0-stable cells: {zero_stable.sum()} of {nx*ny}")
    if zero_stable.sum():
        ys_z = ys[np.where(zero_stable)[0]]
        xs_z = xs[np.where(zero_stable)[1]]
        print(f"  span: x in [{xs_z.min():.3f}, {xs_z.max():.3f}], "
              f"y in [{ys_z.min():.3f}, {ys_z.max():.3f}]")

    vmax = max(0.15, abs(np.nanmax(grid_re)))
    im2 = axes[2].imshow(grid_re, origin='lower',
                         extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                         aspect='equal', interpolation='nearest',
                         cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    targets.plot_targets_to_axis(axes[2])
    # Overlay zero-stable cells with hatch
    axes[2].contour(xs, ys, zero_stable.astype(float),
                    levels=[0.5], colors='black', linewidths=1.5)
    axes[2].set_title("max Re(eigenvalue), most unstable eq\n"
                      "black contour = 0-stable region")
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle("Upper 0-stable island (vonmises k=0.55)")
    fig.tight_layout()
    fig.savefig("diagnostic_island_final.png", dpi=120)
    print("saved diagnostic_island_final.png")
