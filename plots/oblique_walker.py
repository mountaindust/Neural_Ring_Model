# Neural Ring Model: Ising-type dynamics of spatial decision-making.
# Copyright (C) 2026 Christopher Strickland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Oblique-walker figure (DRAFT -- only the first row is built so far).

Three-row layout via GridSpec:

  Row 1 -- three bifurcation panels (no basin wheels), side by side, for the
           two-target fruit-fly geometry (GODM 'fly2': dist 5, +-30 deg ->
           (4.33, +-2.5)) under a ``lin_cutoff`` neural density (a=pi/8, b=pi):
             col 1 -- delta (point) targets, uniform weight
             col 2 -- circle targets (r=0.5), uniform weight
             col 3 -- circle targets (r=0.5), weight tied to the neural density
  Row 2 -- single panel spanning all three columns (TODO -- left blank).
  Row 3 -- single panel spanning all three columns (TODO -- left blank).

The bifurcation panels are rendered exactly like the sweep figures
(``plots/neural_weight_sweep.py``): viridis keyed on the stable-equilibrium
count alone, pinned to ``MAX_COUNT`` for cross-panel color comparability, no
two-axis coding, no basin wheels. A single shared stable-count legend serves the
whole row.

Rendered at exploration quality while we iterate on the layout; bump QUALITY (or
the grid/refinement/DPI constants) once the design settles.

Run (from anywhere):  python plots/oblique_walker.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import decision_model as model            # noqa: E402
from parallel_config import get_n_workers  # noqa: E402

pi = np.pi

# ----------------------------- config -----------------------------

# Fixed scene: two-target GODM 'fly2' geometry (dist 5, +-30 deg).
TARGET_LOCS = np.array([[4.33, 2.5], [4.33, -2.5]])
TARGET_RADIUS = 0.5
FOCAL_LOC = (0, 0)
FOCAL_ANGLE = 0

# Shared neural density (warp) for every column.
WARP_FAMILY = 'lin_cutoff'
A_WARP = pi / 8
B_WARP = pi

# Bifurcation-diagram frame, shared across the three panels.
XLIM = (0.0, 6.0)
YLIM = (-3.5, 3.5)
MAX_COUNT = 3                 # pin the color scale across the three panels
STABILITY_CRITERION = 'reduced'

# Quality knob -- LOW for fast layout iteration; HIGH for production output.
# HIGH refines past the sweep-figure resolution (49x49 base mesh, 3 adaptive
# refinement passes; boundary_dilation defaults to 1 in render_bifurcation, as
# in the sweep).
QUALITY = 'high'
if QUALITY == 'low':
    NUM_X, NUM_Y, REFINEMENT_LEVELS, DPI = 21, 21, 1, 110
else:
    NUM_X, NUM_Y, REFINEMENT_LEVELS, DPI = 49, 49, 3, 300

# Production output: a single high-quality png at 300 dpi, the standard used by
# the other publication figures in this directory (the old jpg + LZW-tif pair
# was retired -- one lossless raster instead of a lossy copy plus a bulky one).
OUT_BASE = os.path.join(HERE, 'oblique_walker')
OUTPUT_FORMATS = ('png',)

# ---- walker ensemble (rows 2 & 3) ----
# Same perception parameterization as column 3 (lin_cutoff warp a=pi/8,b=pi,
# weight tied to the neural density) on the re-oriented geometry: a near target
# slightly off the +x axis and a far target on the +x line of sight. Walkers
# start at the origin facing +x. The two rows are IDENTICAL except for the
# target geometry -- row 2 uses delta (point) targets, row 3 r=0.25 circles.
WALK_LOCS = np.array([[2.5, 0.8],     # near ("first")
                      [7.5, 0.0]])    # far ("second"), on the +x line of sight
WALK_TARGET_R = 0.25                  # circle radius (row 3)
WALK_WEIGHT = 'neural_angle_dist'     # match column 3
WALK_REPS = 100
WALK_K = 2.0
WALK_STD = 0.5
WALK_NOISE_EXP = 0                    # 0 = constant noise sigma*dW
WALK_WALK_STD = 0.5 * pi              # blind-search intensity (plot_walkers default)
WALK_R_EXP = 1                        # drift exponent (plot_walkers default)
WALK_V = 0.2
WALK_DT = 0.05
WALK_TARGET_TOL = 0.1
WALK_MAX_STEPS = 4000
WALK_SEED = 0                         # fixed seed -> reproducible ensemble
WALK_ALPHA = 0.35
WALK_XLIM = (-0.5, 8.2)
WALK_YLIM = (-2.2, 2.6)

# Walker-track coloring by which target the walk reaches (near=0, far=1, or -1
# if it ran out of steps without reaching either).
TRACK_COLORS = {0: 'tab:blue', 1: 'tab:orange', -1: '0.6'}
TRACK_LABELS = {0: 'reached near target', 1: 'reached far target',
                -1: 'no target (max steps)'}

# Per-column definitions for row 1. Point (delta) targets are geom_name=None.
COLUMNS = [
    dict(geom=None, radius=None, weight=None,
         title='Delta targets, uniform weight'),
    dict(geom='circle', radius=TARGET_RADIUS, weight=None,
         title='Circle targets, uniform weight'),
    dict(geom='circle', radius=TARGET_RADIUS, weight='neural_angle_dist',
         title='Circle targets, neural density weight'),
]

# font sizes (matched to stability_comparison_figure.py)
TITLE_FS = 16
LABEL_FS = 14
TICK_FS = 12
SUPTITLE_FS = 18
LEGEND_FS = 14
PANEL_FS = 18         # interior panel-letter labels (A-E)
TRACK_LEGEND_FS = 10  # walker-track destination legend


def build_model(geom, radius, weight):
    """NBM for one column: shared lin_cutoff warp, per-column geometry/weight."""
    tkw = {} if radius is None else dict(r=radius)
    targets = model.Targets(locs=TARGET_LOCS, geom_name=geom, **tkw)
    percep = model.PerceptionModel(
        targets, FOCAL_LOC, FOCAL_ANGLE,
        neural_angle_dist=WARP_FAMILY, angle_weight=weight,
        a_warp=A_WARP, b_warp=B_WARP)
    return model.NeuralBandModel(percep)


def render_bifurcation(ax, nbm, title, pool, show_ylabel=True,
                       point_target_color=None):
    nbm.plot_bifurcation_diagram(
        xlim=XLIM, ylim=YLIM, num_x=NUM_X, num_y=NUM_Y,
        refinement_levels=REFINEMENT_LEVELS, max_count=MAX_COUNT,
        stability_criterion=STABILITY_CRITERION, pool=pool, ax=ax, title=title)
    # Recolor the point (delta) targets: plot_targets_to_axis hardcodes grey,
    # so overplot red markers to set the point geometry apart from the
    # finite-extent (circle) targets in the other panels.
    if point_target_color is not None:
        ax.plot(TARGET_LOCS[:, 0], TARGET_LOCS[:, 1], 'o',
                color=point_target_color, markersize=7, zorder=6)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel('observer x', fontsize=LABEL_FS)
    if show_ylabel:
        ax.set_ylabel('observer y', fontsize=LABEL_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)


def build_walker_model(geom, radius, weight=WALK_WEIGHT):
    """NBM for a walker panel: column-3 perception parameterization on the
    re-oriented geometry, with the walker turning gain K. ``geom``/``radius``
    select the target geometry (None -> delta points; 'circle' -> r=radius);
    ``weight`` is the angle_weight role (defaults to the neural-density weight,
    pass None for uniform)."""
    tkw = {} if radius is None else dict(r=radius)
    targets = model.Targets(locs=WALK_LOCS, geom_name=geom, **tkw)
    percep = model.PerceptionModel(
        targets, FOCAL_LOC, FOCAL_ANGLE,
        neural_angle_dist=WARP_FAMILY, angle_weight=weight,
        a_warp=A_WARP, b_warp=B_WARP)
    return model.NeuralBandModel(percep, K=WALK_K)


def run_walkers(nbm, pool):
    """Reproduce ``plot_walkers``' fixed-seed ensemble via ``_simulate_one_walk``
    so the individual trajectories can be recovered and classified. Mirrors the
    seeding plot_walkers does internally (child seeds spawned from a freshly
    seeded rng; the model's initial gamma as the start coherence). Returns
    (walks, warns) where warn is None iff that walk reached a target."""
    nbm.rng = np.random.default_rng(WALK_SEED)
    base = int(nbm.rng.integers(0, 2**63 - 1))
    child_seeds = np.random.SeedSequence(base).spawn(WALK_REPS)
    args = [(n, child_seeds[n], (0.0, 0.0), 0.0, nbm.gamma, WALK_DT, WALK_V,
             WALK_STD, WALK_WALK_STD, WALK_NOISE_EXP, WALK_R_EXP,
             WALK_MAX_STEPS, WALK_TARGET_TOL) for n in range(WALK_REPS)]
    results = pool.map(nbm._simulate_one_walk, args)
    return [w for w, _ in results], [warn for _, warn in results]


def classify_walks(walks, warns):
    """Index of the target each walk ends at (0=near, 1=far), or -1 if it ran
    out of steps (warn set) without reaching a target."""
    cats = []
    for walk, warn in zip(walks, warns):
        if warn is not None:
            cats.append(-1)
        else:
            end = walk[:, -1]
            cats.append(int(np.argmin(np.linalg.norm(WALK_LOCS - end, axis=1))))
    return cats


def render_walkers(ax, pool, geom, radius, title, weight=WALK_WEIGHT,
                   show_ylabel=True):
    nbm = build_walker_model(geom, radius, weight)
    walks, warns = run_walkers(nbm, pool)
    cats = classify_walks(walks, warns)

    counts = {k: cats.count(k) for k in (0, 1, -1)}
    print('  %-40s near=%d far=%d timeout=%d'
          % (title, counts[0], counts[1], counts[-1]))

    ax.set_aspect('equal')
    nbm.percep_model.targets.plot_targets_to_axis(ax)
    # delta (point) targets render as faint grey dots; overplot crisp red
    # markers (matching the delta bifurcation panel, set apart from circles)
    if geom is None:
        ax.plot(WALK_LOCS[:, 0], WALK_LOCS[:, 1], 'o', color='red', mec='k',
                mew=0.5, ms=9, zorder=5)

    # tracks colored by which target the walk reaches
    for walk, cat in zip(walks, cats):
        ax.plot(walk[0], walk[1], color=TRACK_COLORS[cat], alpha=WALK_ALPHA,
                lw=0.9, zorder=2)

    # mark the starting heading (origin, facing +x)
    ax.annotate('', xy=(0.8, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle='-|>', color='red', lw=2.0,
                                mutation_scale=16), zorder=6)
    ax.plot(0, 0, marker='*', ms=14, color='red', mec='k', mew=0.5, zorder=7)

    ax.set_xlim(*WALK_XLIM)
    ax.set_ylim(*WALK_YLIM)
    ax.set_xlabel('x', fontsize=LABEL_FS)
    if show_ylabel:
        ax.set_ylabel('y', fontsize=LABEL_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.set_title(title, fontsize=TITLE_FS)

    # legend: only the destination categories that actually occur (solid swatch
    # at full opacity, since the tracks themselves are alpha-blended)
    present = [k for k in (0, 1, -1) if counts[k] > 0]
    handles = [Line2D([], [], color=TRACK_COLORS[k], lw=2.5,
                      label='%s (%d)' % (TRACK_LABELS[k], counts[k]))
               for k in present]
    ax.legend(handles=handles, loc='upper right', fontsize=TRACK_LEGEND_FS,
              framealpha=0.9)


def panel_letter(ax, letter):
    """Bold interior panel-letter label at the upper-left of an axes."""
    ax.text(0.025, 0.96, letter, transform=ax.transAxes, fontsize=PANEL_FS,
            fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none',
                      alpha=0.8), zorder=10)


def main():
    fig = plt.figure(figsize=(15, 10))
    # 6 base columns: row 1 = three bifurcation panels (2 cols each); row 2 =
    # two walker panels side by side (3 cols each).
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.25, 1.0],
                  hspace=0.30, wspace=0.55)

    row1_axes = [fig.add_subplot(gs[0, 2 * c:2 * c + 2]) for c in range(3)]
    walk_left = fig.add_subplot(gs[1, 0:3])
    walk_right = fig.add_subplot(gs[1, 3:6])

    with Pool(get_n_workers()) as pool:
        for c, (ax, col) in enumerate(zip(row1_axes, COLUMNS)):
            nbm = build_model(col['geom'], col['radius'], col['weight'])
            pt_color = 'red' if col['geom'] is None else None
            render_bifurcation(ax, nbm, col['title'], pool, show_ylabel=(c == 0),
                               point_target_color=pt_color)
            panel_letter(ax, 'ABC'[c])
        # row 2 -- delta (point) targets | r=0.25 circle targets, side by side
        render_walkers(walk_left, pool, geom=None, radius=None,
                       title='Random walkers with delta targets')
        render_walkers(walk_right, pool, geom='circle', radius=WALK_TARGET_R,
                       title='Circle targets, neural density weight',
                       show_ylabel=False)
        panel_letter(walk_left, 'D')
        panel_letter(walk_right, 'E')

    # one shared stable-count legend on the right of row 1
    cmap = plt.get_cmap('viridis', MAX_COUNT + 1)
    handles = [plt.Line2D([], [], marker='s', markersize=16, linestyle='',
                          color=cmap(i / MAX_COUNT), label=f'{i}')
               for i in range(MAX_COUNT + 1)]
    fig.legend(handles=handles, title='# stable\nequilibria', frameon=False,
               loc='center left', bbox_to_anchor=(0.905, 0.70),
               fontsize=LEGEND_FS, title_fontsize=LEGEND_FS)

    fig.suptitle('Delta functions vs. finite targets: bifurcation geometry '
                 'and walker dynamics', fontsize=SUPTITLE_FS)
    fig.subplots_adjust(left=0.06, right=0.90, top=0.91, bottom=0.07)
    for fmt in OUTPUT_FORMATS:
        out = f'{OUT_BASE}.{fmt}'
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        print('wrote', out)
    plt.close(fig)


if __name__ == '__main__':
    main()
