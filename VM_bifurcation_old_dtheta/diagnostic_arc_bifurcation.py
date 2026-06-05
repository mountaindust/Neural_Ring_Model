"""
Map the bifurcation structure around the upper 0-stable arc.

The middle panel of diagnostic_island_final.png shows three regions:
  (A) upper-left:  1 eq, coupled-stable
  (B) the arc:     1 eq, Hopf-unstable (limit cycle attractor)
  (C) lower-right: 3 eqs, 2 stable + 1 saddle

Question: how do (B) and (C) connect? In particular, by what bifurcation do
the two stable equilibria of (C) disappear as we approach (B) and ultimately
the end of the arc?

Strategy:
  1. Take horizontal slices y = const through both (B) and (C). For each x,
     enumerate all self-consistent eqs and tag them with their max-Re
     coupled-Jacobian eigenvalue. Plot theta_eq(x) and max_re(x) per slice.
  2. Take a vertical slice x = const that passes through (B) end-on, to see
     the SN birth of the new pair as y decreases.
  3. Detect saddle-node and Hopf transitions by tracking when (a) the count
     of equilibria changes (SN), (b) max_re crosses zero with the count
     constant (Hopf).
  4. Sweep many y values to assemble the SN curve(s) and Hopf curve(s) in
     (x, y) space and overlay them on the n_stable map.
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


# -------------------------------------------------------------------
# Part 1: horizontal slices at several y to dissect the transition
# -------------------------------------------------------------------
def slice_data(y_val, x_arr):
    """Return list of (x, list of (theta_eq, R_eq, max_re_eig, all_eigs))."""
    out = []
    for x in x_arr:
        fl = np.array([x, y_val])
        eqs = find_eqs(fl)
        eq_info = []
        for (teq, Req) in eqs:
            eigs = coupled_eigs(fl, teq, Req)
            eq_info.append((teq, Req, float(np.max(np.real(eigs))), eigs))
        out.append((x, eq_info))
    return out


x_arr = np.linspace(1.0, 3.5, 301)
y_levels = [2.55, 2.45, 2.35, 2.25, 2.15, 2.05]

print("Computing slices (this is fast with no pool)...")
slices = {}
for y_val in y_levels:
    print(f"  y={y_val}")
    slices[y_val] = slice_data(y_val, x_arr)

# ---- Plot slices: theta_eq(x) and max_re(x) per slice ----
fig, axes = plt.subplots(len(y_levels), 2, figsize=(13, 2.6 * len(y_levels)),
                         sharex=True)
for row, y_val in enumerate(y_levels):
    ax_th = axes[row, 0]
    ax_re = axes[row, 1]
    for (x, eq_info) in slices[y_val]:
        for (teq, Req, max_re, _) in eq_info:
            color = 'g' if max_re < -1e-8 else 'r'
            ax_th.plot(x, teq, '.', color=color, markersize=3)
            ax_re.plot(x, max_re, '.', color=color, markersize=3)
    ax_th.set_ylabel(f"y={y_val}\ntheta_eq")
    ax_re.set_ylabel("max Re(eig)")
    ax_re.axhline(0, color='k', lw=0.4)
    if row == 0:
        ax_th.set_title("equilibrium positions  (green=stable, red=unstable)")
        ax_re.set_title("max Re of coupled eigenvalues")
    if row == len(y_levels) - 1:
        ax_th.set_xlabel("x")
        ax_re.set_xlabel("x")
fig.tight_layout()
fig.savefig("diagnostic_arc_slices.png", dpi=120)
print("saved diagnostic_arc_slices.png")

# -------------------------------------------------------------------
# Part 2: identify SN and Hopf bifurcation events in each slice
# -------------------------------------------------------------------
print("\nDetected bifurcation events per slice:")
events = []  # list of (y, x, type, detail)
for y_val in y_levels:
    sl = slices[y_val]
    print(f"\n  y = {y_val}:")
    prev_n = None
    prev_status = None
    for (x, eq_info) in sl:
        n = len(eq_info)
        if prev_n is not None and n != prev_n:
            kind = "SN-birth" if n > prev_n else "SN-death"
            print(f"    x~{x:.4f}: {kind}  ({prev_n} -> {n} equilibria)")
            events.append((y_val, x, kind, n - prev_n))
        # Detect Hopf: eq count constant but stability flips for some eq.
        # We rely on theta-continuity to pair up the eqs across consecutive x.
        if prev_n is not None and n == prev_n:
            # Pair by closest theta
            prev_th = [e[0] for e in prev_info]
            cur_th = [e[0] for e in eq_info]
            for i, (teq, Req, max_re, _) in enumerate(eq_info):
                # closest previous
                j = int(np.argmin([abs(model.convert_angles(teq - pt))
                                    for pt in prev_th]))
                prev_max_re = prev_info[j][2]
                if prev_max_re * max_re < 0:
                    kind = ("Hopf-up" if max_re > 0 else "Hopf-down")
                    print(f"    x~{x:.4f}: {kind} on eq[theta~{teq:+.3f}]"
                          f"  (max_re {prev_max_re:+.4f} -> {max_re:+.4f})")
                    events.append((y_val, x, kind, teq))
        prev_n = n
        prev_info = eq_info

# -------------------------------------------------------------------
# Part 3: refined SN and Hopf curves in 2D
# -------------------------------------------------------------------
print("\nMapping SN and Hopf curves in 2D...")

def cell_n_and_max_re(args):
    key, x, y = args
    fl = np.array([x, y])
    eqs = find_eqs(fl)
    if not eqs:
        return key, 0, np.nan
    max_re = -np.inf
    for (teq, Req) in eqs:
        eigs = coupled_eigs(fl, teq, Req)
        m = float(np.max(np.real(eigs)))
        if m > max_re:
            max_re = m
    return key, len(eqs), max_re


if __name__ == "__main__":
    xlim = (1.0, 3.5)
    ylim = (1.5, 3.0)
    nx, ny = 121, 91
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    args_list = [((j, i), xs[i], ys[j])
                 for j in range(ny) for i in range(nx)]
    print(f"Scanning {nx}x{ny}={nx*ny} cells...")
    with Pool(get_n_workers()) as pool:
        results = pool.map(cell_n_and_max_re, args_list)
    grid_n = np.zeros((ny, nx), dtype=int)
    grid_re = np.full((ny, nx), np.nan)
    for (j, i), n, m in results:
        grid_n[j, i] = n
        grid_re[j, i] = m

    np.save('arc_bif_grid_n.npy', grid_n)
    np.save('arc_bif_grid_re.npy', grid_re)

    # Plot: n_eqs colormap with overlaid Hopf contour (max_re=0) and SN
    # boundary (count change).
    fig2, ax = plt.subplots(figsize=(10, 7))
    nmax = max(grid_n.max(), 3)
    cmap = plt.get_cmap('viridis', nmax + 1)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, nmax + 1.5),
                        ncolors=nmax + 1)
    ax.imshow(grid_n, origin='lower',
              extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
              aspect='equal', interpolation='nearest', cmap=cmap, norm=norm)
    # Hopf curve = level set max_re = 0
    cs = ax.contour(xs, ys, grid_re, levels=[0.0],
                    colors='magenta', linewidths=2)
    ax.clabel(cs, inline=True, fmt='Hopf', fontsize=9)
    targets.plot_targets_to_axis(ax)
    ax.set_title("# self-consistent equilibria, with Hopf curve "
                 "(magenta) overlaid\n"
                 "SN curve = boundary between count regions")
    fig2.tight_layout()
    fig2.savefig("diagnostic_arc_curves.png", dpi=120)
    print("saved diagnostic_arc_curves.png")
