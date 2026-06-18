"""
Panel-B prototype (standalone — does NOT touch decision_model.py): a sparse
(x,y) mesh of allocentric "robustness arrows" over the stable-count colormap.
Parallelized (multiprocessing) with adaptive arrow placement.

Two-panel layout:
  - LEFT (panel A): dense colormap of the # of stable SC equilibria.
  - RIGHT (panel B): the same field faint, overlaid with an ADAPTIVE mesh
    of arrows — one per stable SC direction at each chosen grid point,
    pointing in the allocentric heading, with length ∝ that direction's
    basin arc width (the robustness scalar from the neutral-seed
    slaved-flow protocol in basin_arcs.py).

Adaptive placement: a sparse uniform base grid, AUGMENTED so that every
connected multistable (≥2) and ≥3-stable region is guaranteed at least one
arrow — this catches thin "finger" features and small islands a uniform
grid would miss. 1-stable cells get one full-length arrow (no sweep);
0-stable cells get a dot.

Parallelism: the count grid and the per-cell slaved-flow sweeps are both
embarrassingly parallel. Workers inherit the module-global `nbm` via fork.
Worker count from parallel_config.get_n_workers(). For CPU-bound numpy,
run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 to avoid oversubscription.

Usage:  python basin_mesh.py
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Wedge, Patch
from scipy.ndimage import label
from multiprocessing import Pool

from parallel_config import get_n_workers
from basin_arcs import compute_point, R_SEED
from theta_scan import nbm, target_locs

MESH_N_COARSE = 72
MESH_N_BISECT = 12
VALIDATION_POINT = (4.0, 1.5)   # §15 calibration point


def stable_dirs(focal_loc):
    """Allocentric headings of the stable SC equilibria (reduced criterion,
    matching the bifurcation panel)."""
    ang, stab = nbm.sc_equilib(focal_loc=focal_loc,
                               stability_criterion='reduced')
    return [a for a, s in zip(ang, stab) if s]


# -----------------------------------------------------------------------------
# Parallel workers (module-level so they pickle; `nbm` inherited via fork)
# -----------------------------------------------------------------------------
def _count_cell(xy):
    return len(stable_dirs(xy))


def _cell(xy):
    """(xy, count, stable_dirs, arcs, widths) for one wheel cell. Cheap for
    count<=1; runs the slaved-flow sweep for multistable cells. `arcs`
    partitions S¹ into (theta_start, theta_end, label) heading-basins
    (label = stable-eq index, or -1 for no-basin)."""
    s = stable_dirs(xy)
    cnt = len(s)
    if cnt == 0:
        return (xy, 0, s, [(-np.pi, np.pi, -1)], {-1: 2 * np.pi})
    if cnt == 1:
        return (xy, 1, s, [(-np.pi, np.pi, 0)], {0: 2 * np.pi})
    pt = compute_point(xy, n_coarse=MESH_N_COARSE, n_bisect=MESH_N_BISECT)
    return (xy, cnt, pt['stable'], pt['arcs'], pt['widths'])


def count_grid_parallel(xb, yb, pool):
    pts = [(float(x), float(y)) for y in yb for x in xb]
    res = pool.map(_count_cell, pts, chunksize=16)
    return np.array(res, dtype=int).reshape(len(yb), len(xb))


# -----------------------------------------------------------------------------
# Adaptive arrow placement
# -----------------------------------------------------------------------------
def adaptive_cells(xb, yb, C, base_x, base_y, min_sep=0.45):
    """Region-aware placement: one representative wheel at the CENTROID of
    every >=3 component and every base-uncovered >=2 component (so thin
    features and the tri-stable island are sampled at their *centres*, where
    all basins are nonzero), then a uniform base grid for context. Cells
    closer than `min_sep` are merged, region representatives taking priority
    over base cells. Returns (cells, n_reps)."""
    base = [(float(x), float(y)) for y in base_y for x in base_x]

    def nearest_idx(x, y):
        return int(np.argmin(np.abs(xb - x))), int(np.argmin(np.abs(yb - y)))

    base_idx = [nearest_idx(x, y) for (x, y) in base]

    def centroid_cell(js, is_):
        ci, cj = is_.mean(), js.mean()
        kb = int(np.argmin((is_ - ci) ** 2 + (js - cj) ** 2))
        return (float(xb[is_[kb]]), float(yb[js[kb]]))

    reps = []
    # every >=3 component: always sample its centre (3-basin cells live there,
    # not at the island edge where the weak targets capture no headings)
    lab3, n3 = label(C >= 3)
    for k in range(1, n3 + 1):
        js, is_ = np.where(lab3 == k)
        reps.append(centroid_cell(js, is_))
    # >=2 components a base cell doesn't already fall inside
    lab2, n2 = label(C >= 2)
    for k in range(1, n2 + 1):
        js, is_ = np.where(lab2 == k)
        comp = set(zip(is_.tolist(), js.tolist()))
        if not any((bi, bj) in comp for (bi, bj) in base_idx):
            reps.append(centroid_cell(js, is_))

    # merge: representatives first (priority), then base; drop within min_sep
    cells = []
    for (x, y) in reps + base:
        if all((x - kx) ** 2 + (y - ky) ** 2 >= min_sep ** 2
               for (kx, ky) in cells):
            cells.append((x, y))
    return cells, len(reps)


# -----------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    n_workers = get_n_workers()
    print(f"workers: {n_workers}")

    nx_bg, ny_bg = 64, 58
    xb = np.linspace(0.4, 4.6, nx_bg)
    yb = np.linspace(-2.5, 2.5, ny_bg)
    base_x = np.linspace(1.0, 4.4, 6)
    base_y = np.linspace(-2.4, 2.4, 7)

    with Pool(n_workers) as pool:
        # --- background count field (panel A) ---
        print(f"count grid {nx_bg}x{ny_bg} ...")
        t0 = time.time()
        C = count_grid_parallel(xb, yb, pool)
        maxc = int(C.max())
        print(f"  done in {time.time()-t0:.0f}s   (max stable count = {maxc})")

        # --- adaptive arrow cells ---
        cells, n_reps = adaptive_cells(xb, yb, C, base_x, base_y)
        print(f"wheel cells: {len(cells)} ({n_reps} region representatives "
              f"+ base grid, merged)")

        # --- per-cell sweeps (parallel, imbalanced -> chunksize=1) ---
        t1 = time.time()
        results = []
        done = 0
        for r in pool.imap_unordered(_cell, cells, chunksize=1):
            results.append(r)
            done += 1
            if r[1] >= 2:
                print(f"  [{done}/{len(cells)}] multistable "
                      f"({r[0][0]:.2f},{r[0][1]:+.2f}) [{r[1]} stable]")
        print(f"arrow mesh computed in {time.time()-t1:.0f}s")

    # --- plot ------------------------------------------------------------
    extent = [xb[0], xb[-1], yb[0], yb[-1]]
    Xc, Yc = np.meshgrid(xb, yb)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(17, 7.5),
                                   sharex=True, sharey=True)

    # Panel A: viridis stable-count colormap
    cmap = plt.cm.viridis
    norm = mcolors.BoundaryNorm(np.arange(-0.5, maxc + 1.5, 1.0), cmap.N)
    im = axL.imshow(C, origin='lower', extent=extent, cmap=cmap, norm=norm,
                    aspect='equal')
    axL.set_title('Panel A — number of stable SC equilibria')
    axL.set_ylabel('y')
    cb = fig.colorbar(im, ax=axL, ticks=range(maxc + 1), fraction=0.046,
                      pad=0.04)
    cb.set_label('stable count')

    # Panel B: muted gray count regions + crisp boundary contours, then
    # arrows colored AND sized by basin width (robustness).
    # faint copy of the Panel A count field as an unobtrusive backdrop
    axR.imshow(C, origin='lower', extent=extent, cmap=cmap, norm=norm,
               aspect='equal', alpha=0.30)
    axR.set_title('Panel B — basin wheels (arrow = stable direction; ring '
                  'sector = its θ-basin; color = basin rank, size ∝ robustness)')

    # categorical palette by basin RANK (largest basin -> smallest), chosen
    # for max contrast between consecutive ranks (colorblind-safe):
    #   1st=gold, 2nd=blue, 3rd=vermilion, 4th=green, 5th=purple
    PALETTE = ['#F0C300', '#1463C8', '#D55E00', '#2CA02C', '#9467BD']

    def rank_color(rank):
        return PALETTE[rank % len(PALETTE)]

    r_out = 0.34 * min(base_x[1] - base_x[0], base_y[1] - base_y[0])
    r_in = 0.83 * r_out              # thin annulus; arrows live inside it
    ring_w = r_out - r_in
    stroke = [pe.withStroke(linewidth=1.8, foreground='black')]
    max_rank = 1
    for (xy, cnt, s, arcs, widths) in results:
        cx, cy = xy
        # reachable directions, ranked largest -> smallest basin
        nz = [lab for lab in range(len(s)) if widths.get(lab, 0.0) > 1e-9]
        if not nz:
            continue                 # nothing reachable; faint backdrop shows it
        order = sorted(nz, key=lambda l: -widths[l])
        rank_of = {lab: r for r, lab in enumerate(order)}
        max_rank = max(max_rank, len(order))
        multi = len(nz) >= 2         # >=2 reachable basins -> draw the annulus
        if multi:
            # annulus of θ-basins, each sector colored by its basin robustness
            for (s_start, s_end, lab) in arcs:
                span = (s_end - s_start) % (2 * np.pi)
                if len(arcs) == 1:
                    span = 2 * np.pi          # lone full-circle arc
                if span <= 1e-9:
                    continue
                th1 = np.degrees(s_start)
                col = ('lightgray' if lab < 0
                       else rank_color(rank_of.get(lab, 0)))
                axR.add_patch(Wedge((cx, cy), r_out, th1,
                                    th1 + np.degrees(span), width=ring_w,
                                    facecolor=col, edgecolor='0.3', lw=0.3,
                                    zorder=5))
            cap = r_in
        else:
            cap = r_out              # one reachable direction: lone arrow, no ring

        # arrows: one per reachable direction, length ∝ basin width (floored
        # so minor basins still show), colored to match its sector
        for lab in nz:
            ang = s[lab]
            frac = widths[lab] / (2 * np.pi)
            L = cap * (0.30 + 0.70 * frac) if multi else cap
            axR.annotate('', xy=(cx + L * np.cos(ang), cy + L * np.sin(ang)),
                         xytext=(cx, cy), zorder=6,
                         arrowprops=dict(arrowstyle='-|>',
                                         color=rank_color(rank_of[lab]),
                                         lw=1.3, shrinkA=0, shrinkB=0,
                                         mutation_scale=5, path_effects=stroke))

    # categorical legend: color = basin rank by size
    rank_labels = ['largest basin', '2nd largest', '3rd largest',
                   '4th largest', '5th largest']
    handles = [Patch(facecolor=rank_color(i), edgecolor='0.3',
                     label=rank_labels[i]) for i in range(max_rank)]
    axR.legend(handles=handles, loc='lower left', fontsize=7, framealpha=0.9,
               title='arrow / sector color', title_fontsize=7)

    # validation point + targets on both panels
    txt_stroke = [pe.withStroke(linewidth=2.0, foreground='white')]
    for ax in (axL, axR):
        ax.plot(*VALIDATION_POINT, marker='D', mfc='magenta', mec='k',
                ms=9, zorder=7)
        ax.annotate('validated', VALIDATION_POINT, textcoords='offset points',
                    xytext=(8, 4), fontsize=7.5, color='magenta', zorder=7,
                    path_effects=txt_stroke)
        ax.scatter(target_locs[:, 0], target_locs[:, 1], marker='*', s=260,
                   color='red', edgecolor='k', zorder=6)
        ax.set_xlabel('x')
    axL.annotate('target', target_locs[0], textcoords='offset points',
                 xytext=(6, 4), fontsize=7.5, color='red', zorder=7,
                 path_effects=txt_stroke)

    fig.suptitle('Bifurcation + basin-of-attraction (VM-k055): stable count '
                 'and per-direction robustness', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(here, 'basin_mesh.png')
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
