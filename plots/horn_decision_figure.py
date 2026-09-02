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

"""Horn-decision figure: how a deterministic (noise-free) walker traverses the
two-target 'horn' and commits to the near target.

Single row, two panels:

  Panel A (left) -- 1-D bifurcation diagram of the self-consistent equilibrium
      heading vs observer x, taken along the horizontal slice y=1.6 of the
      fly two-target scene. Stable branches are green dots, unstable (the
      separatrix) red x's. The story reads left-to-right:
        * single stable branch = the consensus/compromise heading;
        * at x~0.77 a saddle-node opens the HORN -- a second stable branch
          (toward the NEAR/upper target) plus an unstable separatrix appear;
        * at x~1.0 the consensus branch annihilates with the separatrix, so
          the lone survivor points at the near target (the decision is made);
        * at x~1.45 a third saddle-node births the FAR/lower-target branch
          (the main near-vs-far bistability), which never recaptures the
          already-committed walker.
      Dashed lines are the live compass bearings from the moving observer to
      each target: where a green branch lies on a dashed line, that
      equilibrium points exactly at that target (the consensus branch sits
      between them, a compromise).

  Panel B (right) -- the same scene's NBM bifurcation diagram (stable-count
      raster) zoomed on the upper target, rendered exactly like the
      oblique_walker.py panels (adaptive refinement, viridis keyed on stable
      count, max_count pinned, model target plotting). Four deterministic
      walkers launched from off-axis-left starts (facing +x) are overlaid;
      all curve up and reach the near (upper) target. The colored raster is
      computed across the full x-extent of the panel (no white margin).

Run:  python plots/horn_decision_figure.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import decision_model as model            # noqa: E402
from parallel_config import get_n_workers  # noqa: E402

pi = np.pi

# ----------------------------- scene -----------------------------
TARGET_LOCS = np.array([[4.33, 2.5], [4.33, -2.5]])   # fly two-target scene
TARGET_RADIUS = 0.5
WARP_FAMILY = 'lin_cutoff'
A_WARP = pi / 8
B_WARP = pi
ANGLE_WEIGHT = None        # uniform = the default/'standard' parameterization
K = 2.0

# ------------------------ panel A (slice) ------------------------
SLICE_Y = 1.6
SLICE_XLIM = (0.0, 3.0)
SLICE_DX = 0.02

# ------------------------ panel B (raster) -----------------------
# Extended to cover the walker x-extent and the near target (no white margin).
RASTER_XLIM = (0.0, 5.0)
RASTER_YLIM = (0.6, 3.3)
MAX_COUNT = 3
STABILITY_CRITERION = 'reduced'

# Quality knob -- LOW for fast layout iteration, HIGH for production, matching
# the oblique_walker.py sweep resolution (adaptive refinement).
QUALITY = 'high'
if QUALITY == 'low':
    NUM_X, NUM_Y, REFINE, DPI = 25, 17, 1, 110
else:
    NUM_X, NUM_Y, REFINE, DPI = 49, 33, 3, 300

# --------------------------- walkers -----------------------------
WALK_STARTS = [((0.15, 1.6), 0.0), ((0.15, 2.0), 0.0),
               ((0.15, 2.4), 0.0), ((0.15, 2.7), 0.0)]
WALK_V = 0.2
WALK_DT = 0.03
WALK_MAX_T = 70.0
WALK_TOL = 0.12
WALK_COLOR = 'crimson'

OUT_BASE = os.path.join(HERE, 'horn_decision_figure')
OUTPUT_FORMATS = ('png',)

# font sizes (matched to the other plots/ figures)
TITLE_FS = 15
LABEL_FS = 13
TICK_FS = 11
SUPTITLE_FS = 17
PANEL_FS = 17

# module-global model so pool workers can evaluate the slice cheaply
_M = None


def _init_worker():
    """Build the module-global model inside each pool worker.

    Under the 'fork' start method a worker inherits the parent's _M for free,
    but under 'spawn' (the Windows/macOS default) the module is re-imported and
    _M would stay None, so _slice_eqs raised AttributeError. An explicit
    initializer is correct under both start methods.
    """
    global _M
    _M = build_model()


def build_model(weight=ANGLE_WEIGHT):
    targets = model.Targets(locs=TARGET_LOCS, geom_name='circle',
                            r=TARGET_RADIUS)
    percep = model.PerceptionModel(
        targets, (0, 0), 0.0,
        neural_angle_dist=WARP_FAMILY, angle_weight=weight,
        a_warp=A_WARP, b_warp=B_WARP)
    return model.NeuralBandModel(percep, K=K)


def _slice_eqs(x):
    """All SC equilibria at (x, SLICE_Y): list of (x, heading_deg, stable)."""
    angs, stab = _M.sc_equilib(focal_loc=(x, SLICE_Y))
    return [(x, float(np.degrees(a)), bool(s)) for a, s in zip(angs, stab)]


def walk(nbm, start, theta0):
    """Deterministic (noise-free) walk: gamma slaved each step, half-angle
    torque, constant speed. Returns (xs, ys, dest) with dest 0=near/upper,
    1=far/lower, None=neither."""
    x, y = float(start[0]), float(start[1])
    th = float(theta0)
    nbm.gamma = 0.15 + 0j
    xs, ys = [x], [y]
    dest = None
    for _ in range(int(WALK_MAX_T / WALK_DT)):
        g = nbm.run_dgamma_dt(focal_angle=th, focal_loc=(x, y),
                              init_gamma=nbm.gamma, warn=False)
        nbm.gamma = g
        R = abs(g)
        Theta = np.angle(g)
        th = th + K * R * np.sin(Theta / 2) * WALK_DT
        x = x + WALK_V * np.cos(th) * WALK_DT
        y = y + WALK_V * np.sin(th) * WALK_DT
        xs.append(x)
        ys.append(y)
        d = nbm.percep_model.targets.get_dist_to_targets((x, y))
        if np.min(d) < WALK_TOL:
            dest = int(np.argmin(d))
            break
        if x > RASTER_XLIM[1] + 0.5 or x < -1 or abs(y) > 4:
            break
    return np.array(xs), np.array(ys), dest


def render_slice(ax, pool):
    xs = np.round(np.arange(SLICE_XLIM[0], SLICE_XLIM[1] + 1e-9, SLICE_DX), 4)
    res = pool.map(_slice_eqs, xs)
    pts = [p for r in res for p in r]
    Sx = [p[0] for p in pts if p[2]]
    Sy = [p[1] for p in pts if p[2]]
    Ux = [p[0] for p in pts if not p[2]]
    Uy = [p[1] for p in pts if not p[2]]

    # live compass bearings from the moving observer to each target
    bu = np.degrees(np.arctan2(TARGET_LOCS[0, 1] - SLICE_Y,
                               TARGET_LOCS[0, 0] - xs))
    bl = np.degrees(np.arctan2(TARGET_LOCS[1, 1] - SLICE_Y,
                               TARGET_LOCS[1, 0] - xs))
    ax.plot(xs, bu, '--', color='tab:blue', lw=1.1, alpha=0.7,
            label='bearing to near (upper) target')
    ax.plot(xs, bl, '--', color='tab:orange', lw=1.1, alpha=0.7,
            label='bearing to far (lower) target')

    ax.axvspan(0.77, 1.02, color='gold', alpha=0.16, zorder=0)
    ax.scatter(Sx, Sy, s=13, color='green', label='stable SC equilibrium',
               zorder=5)
    ax.scatter(Ux, Uy, s=12, color='red', marker='x',
               label='unstable (separatrix)', zorder=5)

    # HORN label lifted to mid-plot, inside the gold band, clear of branches
    ax.text(0.895, -25, 'HORN\n(consensus + near)', ha='center', va='center',
            fontsize=10, color='#806000', fontweight='bold')
    ax.annotate('consensus branch dies\n(saddle-node ~x=1.0)',
                xy=(1.0, -3), xytext=(1.32, -38), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='k'))
    ax.annotate('far-target born\n(main bistability, ~x=1.45)',
                xy=(1.46, -49), xytext=(1.78, -72), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='k'))

    ax.set_xlim(*SLICE_XLIM)
    ax.set_ylim(-95, 45)
    ax.set_xlabel('observer x  (moving right along y = 1.6)', fontsize=LABEL_FS)
    ax.set_ylabel('SC-equilibrium heading (deg, allocentric)', fontsize=LABEL_FS)
    ax.set_title('Equilibrium branches along y = 1.6', fontsize=TITLE_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.grid(alpha=0.25)
    ax.legend(loc='lower left', fontsize=8.5, framealpha=0.92)


def render_raster(ax, nbm, pool):
    nbm.plot_bifurcation_diagram(
        xlim=RASTER_XLIM, ylim=RASTER_YLIM, num_x=NUM_X, num_y=NUM_Y,
        refinement_levels=REFINE, max_count=MAX_COUNT,
        stability_criterion=STABILITY_CRITERION, pool=pool, ax=ax,
        title='Stable-count raster + deterministic walkers')

    # slice indicator connecting to panel A
    ax.axhline(SLICE_Y, color='w', ls='--', lw=1.0, alpha=0.8, zorder=3)
    ax.text(RASTER_XLIM[1] - 0.05, SLICE_Y + 0.07, 'y = 1.6 (panel A)',
            color='w', fontsize=8, ha='right', va='bottom', zorder=4)

    # deterministic walkers
    dests = []
    for st, th0 in WALK_STARTS:
        wx, wy, dest = walk(nbm, st, th0)
        dests.append(dest)
        ax.plot(wx, wy, '-', color=WALK_COLOR, lw=1.7, alpha=0.95, zorder=4)
        ax.plot(st[0], st[1], marker='*', ms=11, color='white',
                mec='k', mew=0.6, zorder=6)

    ax.set_xlim(*RASTER_XLIM)
    ax.set_ylim(*RASTER_YLIM)
    ax.set_xlabel('observer x', fontsize=LABEL_FS)
    ax.set_ylabel('observer y', fontsize=LABEL_FS)
    ax.set_title('Stable-count raster + deterministic walkers', fontsize=TITLE_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)

    # two legends: viridis stable-count swatches + the walker overlay
    cmap = plt.get_cmap('viridis', MAX_COUNT + 1)
    count_handles = [Line2D([], [], marker='s', markersize=11, linestyle='',
                            color=cmap(i / MAX_COUNT), label=f'{i}')
                     for i in range(MAX_COUNT + 1)]
    leg1 = ax.legend(handles=count_handles, title='# stable\nequilibria',
                     loc='lower right', fontsize=9, title_fontsize=9,
                     framealpha=0.92)
    ax.add_artist(leg1)
    walk_handles = [
        Line2D([], [], color=WALK_COLOR, lw=2.2,
               label='walker (all reach near target)'),
        Line2D([], [], marker='*', color='white', mec='k', mew=0.6,
               linestyle='', markersize=11, label='start (facing +x)')]
    ax.legend(handles=walk_handles, loc='upper left',
              bbox_to_anchor=(0.005, 0.905), fontsize=8.5, framealpha=0.92)
    print('  walker destinations (0=near/upper,1=far/lower,None=neither):',
          dests)


def panel_letter(ax, letter):
    ax.text(0.02, 0.975, letter, transform=ax.transAxes, fontsize=PANEL_FS,
            fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none',
                      alpha=0.85), zorder=12)


def main():
    global _M
    _M = build_model()      # the parent's copy, used directly by render_raster

    fig = plt.figure(figsize=(17.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.32], wspace=0.12)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    # Panel B has equal aspect; anchor its box to the west so it hugs panel A
    # rather than floating centered in its (wider) cell.
    ax_b.set_anchor('W')

    with Pool(get_n_workers(), initializer=_init_worker) as pool:
        render_slice(ax_a, pool)
        render_raster(ax_b, _M, pool)

    panel_letter(ax_a, 'A')
    panel_letter(ax_b, 'B')
    fig.suptitle('How a deterministic walker traverses the horn and commits '
                 'to the near target', fontsize=SUPTITLE_FS)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.11)

    for fmt in OUTPUT_FORMATS:
        out = f'{OUT_BASE}.{fmt}'
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        print('wrote', out)
    plt.close(fig)


if __name__ == '__main__':
    main()
