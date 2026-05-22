"""
Recompute the bifurcation diagram on the user's setup, but classify each
self-consistent equilibrium by BOTH:

  (a) the existing _discrim_A criterion (gamma-stability at fixed heading)
  (b) the full 3x3 coupled (gamma_real, gamma_imag, theta) Jacobian

so we can visualize where the two disagree.

Also sweeps along the y=0 axis at fine resolution to locate the pitchfork-like
bifurcations in x where intermediate saddles appear/disappear.
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

# ---- setup ----
target_locs = np.array([[4.33, 2.5], [4.33, -2.5]])
targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)
percep = model.PerceptionModel(targets, (0, 0), 0,
                               neural_weight='vonmises',
                               neural_angle='integral')
percep.k = 0.55
nbm = model.NeuralBandModel(percep)
K = nbm.K


def find_eqs(focal_loc):
    """Return list of (theta_eq, R_eq) for self-consistent equilibria."""
    R_probe = 0.5
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


def coupled_jacobian_eigs(focal_loc, theta_eq, R_eq, h=1e-6):
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


def count_at(args):
    key, x, y = args
    focal_loc = np.array([x, y])
    eqs = find_eqs(focal_loc)
    n_disc = 0
    n_coupled = 0
    for (teq, Req) in eqs:
        if nbm._discrim_A(Req + 0j, teq, focal_loc):
            n_disc += 1
        eigs = coupled_jacobian_eigs(focal_loc, teq, Req)
        if np.all(np.real(eigs) < -1e-8):
            n_coupled += 1
    return key, n_disc, n_coupled, len(eqs)


if __name__ == "__main__":
    # ---- Grid scan ----
    xlim = (0.0, 6.0)
    ylim = (-3.5, 3.5)
    nx, ny = 61, 61
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)

    args_list = [((j, i), xs[i], ys[j])
                 for j in range(ny) for i in range(nx)]

    print(f"Scanning {nx}x{ny} grid with coupled stability check...")
    with Pool(get_n_workers()) as pool:
        results = pool.map(count_at, args_list)

    grid_disc = np.zeros((ny, nx), dtype=int)
    grid_coupled = np.zeros((ny, nx), dtype=int)
    grid_total = np.zeros((ny, nx), dtype=int)
    for (j, i), n_disc, n_coupled, n_total in results:
        grid_disc[j, i] = n_disc
        grid_coupled[j, i] = n_coupled
        grid_total[j, i] = n_total

    np.save('recount_grid_disc.npy', grid_disc)
    np.save('recount_grid_coupled.npy', grid_coupled)
    np.save('recount_grid_total.npy', grid_total)
    np.save('recount_xs.npy', xs)
    np.save('recount_ys.npy', ys)

    print("\nDistribution of grid_disc (= existing bifurc plot):")
    u, c = np.unique(grid_disc, return_counts=True)
    for uu, cc in zip(u, c):
        print(f"  {uu}: {cc}")
    print("\nDistribution of grid_coupled (= true coupled-stable count):")
    u, c = np.unique(grid_coupled, return_counts=True)
    for uu, cc in zip(u, c):
        print(f"  {uu}: {cc}")

    # ---- Plot ----
    nmax = max(grid_disc.max(), grid_coupled.max())
    cmap = plt.get_cmap('viridis', nmax + 1)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, nmax + 1.5),
                        ncolors=nmax + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, data, title in zip(
            axes,
            [grid_disc, grid_coupled, grid_disc - grid_coupled],
            ["_discrim_A count (current bifurc plot)",
             "Coupled 3D Jacobian count (true)",
             "Difference (current minus true)"]):
        if "Difference" in title:
            cmax = max(1, abs(data).max())
            im = ax.imshow(data, origin='lower',
                           extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                           aspect='equal', interpolation='nearest',
                           cmap='RdBu_r', vmin=-cmax, vmax=cmax)
        else:
            im = ax.imshow(data, origin='lower',
                           extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                           aspect='equal', interpolation='nearest',
                           cmap=cmap, norm=norm)
        targets.plot_targets_to_axis(ax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    fig.suptitle("Bifurcation diagram: _discrim_A vs coupled 3D Jacobian "
                 "(VM k=0.55, two circle targets)")
    fig.tight_layout()
    fig.savefig("diagnostic_recount_compare.png", dpi=120)
    print("\nWrote diagnostic_recount_compare.png")

    # ---- 1D slice along y=0 to find pitchforks ----
    print("\n1D scan along y=0 (fine):")
    xs_fine = np.linspace(0.0, 3.0, 121)
    n_disc_slice = []
    n_coupled_slice = []
    for x in xs_fine:
        eqs = find_eqs(np.array([x, 0.0]))
        nd = sum(1 for (t, r) in eqs
                 if nbm._discrim_A(r + 0j, t, np.array([x, 0.0])))
        nc = 0
        for (t, r) in eqs:
            eigs = coupled_jacobian_eigs(np.array([x, 0.0]), t, r)
            if np.all(np.real(eigs) < -1e-8):
                nc += 1
        n_disc_slice.append(nd)
        n_coupled_slice.append(nc)

    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs_fine, n_disc_slice, 'o-', label='_discrim_A count', alpha=0.7)
    ax.plot(xs_fine, n_coupled_slice, 's-',
            label='coupled-stable count', alpha=0.7)
    ax.set_xlabel("x (focal_loc x at y=0)")
    ax.set_ylabel("# stable equilibria")
    ax.set_title("Stability count along y=0 slice (VM k=0.55)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("diagnostic_y0_slice.png", dpi=120)
    print("Wrote diagnostic_y0_slice.png")
