'''
Fly-geometry publication figures. Two outputs, both 300-dpi png:

``fly_geom_3target_branch.png``
    The 3-target fly. LEFT: the stable-equilibrium-count map over (x, y) with
    walker tracks overlaid. RIGHT: the self-consistent-equilibrium *branch*
    diagram along the y=0 cut -- every SC equilibrium heading vs observer x,
    stable filled / unstable open, with the bifurcation x-locations marked.
    Both panels are built from the SAME model, so the right panel is literally
    a cut through the left one: its bifurcations are the colour transitions
    along y=0. (The branch panel replaces the fly y=0 panel of
    ``branch_diagram_combined.png``, which is drawn with the locked empirical
    fit -- a different fly -- and is left alone.)

``fly_geom_9target.png``
    The 9-target ring case on its own (formerly the right panel of the old
    two-panel ``fly_geom.png``).

Run:  python fly_geom.py                   # both figures
      python fly_geom.py 9target           # just one (the 9-target solve is slow)
      FLYGEOM_FAST=1 python fly_geom.py    # coarse/fast layout check
'''

import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, 'walker_analysis'), HERE):
    sys.path.insert(0, p)

from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker
import decision_model as model

from parallel_config import get_n_workers
# The branch scan (and its bifurcation-location clustering) is shared with the
# combined fly+locust branch diagram -- import it rather than re-deriving it.
from decision_skeleton import _branch_scan, _cluster

# Render mathtext in Computer Modern, matching the default math font of a LaTeX
# document (and stability_comparison_figure.py). Needs no LaTeX installation.
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12

SUPTITLE_FS = 16
NOTE_FS = 11                # the smaller parenthetical side-note in the title
PANEL_FS = 20               # interior panel-letter labels (A, B)
BIF_LABEL_FS = 9            # bifurcation-location labels on the dashed lines

OUT_3TARGET = os.path.join(HERE, 'fly_geom_3target_branch.png')
OUT_9TARGET = os.path.join(HERE, 'fly_geom_9target.png')
DPI = 300

# Warp/weight/T shared by both cases (K only scales the turning rate, not the
# SC structure). Foveal lin_cutoff warp with the weight tied to it.
A_WARP, B_WARP = 0.25*np.pi, 0.9*np.pi
T, K = 0.2, 4.5

# The y=0 branch scan (scanned over exactly the plotted range).
BRANCH_XLIM = (-1.0, 4.0)
BRANCH_Y0 = 0.0

FAST = bool(os.environ.get('FLYGEOM_FAST', ''))
if FAST:                    # layout/label check only -- NOT publication quality
    NUM_X = NUM_Y = 15
    RLEV = 1
    BRANCH_NUM_X = 60
    REPS = 10
else:                       # publication resolution
    NUM_X = NUM_Y = 57
    RLEV = 3
    BRANCH_NUM_X = 400
    REPS = 50


def build_model(target_locs):
    targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)
    percep_model = model.PerceptionModel(targets, (0, 0), 0,
                                         neural_angle_dist='lin_cutoff',
                                         angle_weight='neural_angle_dist',
                                         a_warp=A_WARP, b_warp=B_WARP)
    return model.NeuralBandModel(percep_model, T=T, K=K)


def mixed_suptitle(fig, main, note, *, main_fs=SUPTITLE_FS, note_fs=NOTE_FS,
                   sep=8, top=0.93):
    '''Figure title whose trailing parenthetical is set smaller, as a side-note.

    A single matplotlib ``Text`` (what ``fig.suptitle`` makes) cannot mix font
    sizes, so pack two ``TextArea``s side by side on a shared baseline and
    anchor the pair to the top centre of the figure. ``top`` is the fraction of
    the figure left to the axes by the constrained-layout engine -- the packed
    title is drawn in the strip above it.
    '''
    box = HPacker(children=[TextArea(main, textprops=dict(fontsize=main_fs)),
                            TextArea(note, textprops=dict(fontsize=note_fs))],
                  align='baseline', pad=0, sep=sep)
    fig.add_artist(AnchoredOffsetbox(loc='upper center', child=box, pad=0.0,
                                     borderpad=0.0, frameon=False,
                                     bbox_to_anchor=(0.5, 1.0),
                                     bbox_transform=fig.transFigure))
    fig.get_layout_engine().set(rect=(0, 0, 1, top))


def panel_letter(ax, letter, *, color='k', y=0.965):
    '''Interior panel letter in the upper-right corner. ``color`` has to suit
    the panel background (white over the dark count map, black over the white
    branch panel); ``y`` (axes fraction, top of the text) drops the label clear
    of anything drawn in that corner.'''
    ax.text(0.975, y, letter, transform=ax.transAxes, fontsize=PANEL_FS,
            fontweight='bold', color=color, va='top', ha='right', zorder=8)


def draw_branch_panel(ax, neur_model, *, y0=BRANCH_Y0, xlim=BRANCH_XLIM,
                      num_x=BRANCH_NUM_X, criterion='reduced'):
    '''SC-equilibrium branch diagram along the horizontal cut y=y0.

    Sweeps the observer x across ``xlim`` and plots every self-consistent
    equilibrium heading (stable filled blue / unstable open red) vs x; vertical
    dashed lines mark the x where the stable count changes, i.e. the
    bifurcations. Rendering mirrors ``decision_skeleton.plot_diagram_both``.
    '''
    xs = np.linspace(xlim[0], xlim[1], num_x)
    stable, unstable, n_stable = _branch_scan(neur_model, y0, xs, criterion)

    # bifurcation x-locations: where the stable count changes (clustered, since
    # a saddle-node cluster + near-SN solver jitter would otherwise flood it)
    bif_x = _cluster(xs[1:][np.diff(n_stable) != 0], gap=0.12)
    for bx in bif_x:
        ax.axvline(bx, color='0.8', lw=0.8, ls='--', zorder=0)
        # the line is only useful if you can read off where it is: label each
        # with its x, rotated up the line's left side in the empty band below
        # the top of the panel
        ax.text(bx, 178, f'{bx:.2f}', rotation=90, ha='right', va='top',
                fontsize=BIF_LABEL_FS, color='0.35', zorder=1)

    ax.scatter(unstable['x'], np.degrees(unstable['th']), s=4,
               facecolors='none', edgecolors='tab:red', lw=0.5,
               label='unstable', zorder=2)
    ax.scatter(stable['x'], np.degrees(stable['th']), s=5, color='tab:blue',
               label='stable', zorder=3)

    ax.set_xlim(xlim)
    ax.set_ylim(-185, 185)
    ax.set_yticks(range(-180, 181, 90))
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('observer x')
    ax.set_ylabel(r'Equilib. heading $\varphi$ [deg]')
    ax.set_title(f'Self-consistent equilibria along y={y0:.2f}')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    return bif_x


def figure_3target():
    '''Count map + walker tracks (left) beside the y=0 branch diagram (right).'''
    # THREE TARGET FLY CASE FROM PAPER
    target_locs = np.array([[5.0000,  0.0000],
                            [3.8302,  3.2139],
                            [3.8302, -3.2139]])
    neur_model = build_model(target_locs)
    neur_model.rng = np.random.default_rng(seed=3)

    # The map window is a SQUARE (10 x 10) on purpose: plot_bifurcation_diagram
    # renders with aspect='equal', so its axes box is sized to its data-range
    # aspect. The column widths below are then set so the square map and the
    # (wider than tall) branch panel come out the same height with no gap.
    fig, (ax, ax_br) = plt.subplots(
        1, 2, figsize=(12.5, 5.6), constrained_layout=True,
        gridspec_kw={'width_ratios': [1.0, 1.15]})

    with Pool(get_n_workers()) as pool:
        neur_model.plot_bifurcation_diagram(xlim=(-4, 6), num_x=NUM_X,
                                            ylim=(-5, 5), num_y=NUM_Y,
                                            refinement_levels=RLEV,
                                            max_count=None, pool=pool, ax=ax,
                                            title=None, wb_plot=False,
                                            stability_criterion='reduced')
    # Walkers launch from (-1, 0) -- upstream of the first bifurcation on the
    # y=0 cut, so every track starts in the single-stable-heading region.
    neur_model.plot_walkers(dt=0.1, v=0.3, std=0.4, noise_exp=0,
                            repetitions=REPS,
                            start_loc=(-1, 0), start_angle=None,
                            alpha=0.35, ax=ax, wb_plot=False,
                            title='Constant noise random walks: '
                                  'K=4.5, $\\sigma=0.4$',)
    ax.set_xlim(-4, 6)      # lock the square view (walkers can autoscale it)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper left')
    # white over the dark (1-stable) upper-right corner of the count map
    panel_letter(ax, 'A', color='white')

    bif_x = draw_branch_panel(ax_br, neur_model)
    # same corner placement as A, in black over the white panel
    panel_letter(ax_br, 'B', color='k')
    print('  y=0 bifurcations at x ~ '
          + ', '.join(f'{b:.2f}' for b in bif_x))

    mixed_suptitle(fig, 'Fly Target Geometry',
                   '(for illustration: not fit to data)')

    fig.savefig(OUT_3TARGET, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT_3TARGET)


def figure_9target():
    '''The 9-target ring case, on its own.'''
    # EXPANDED TARGET FLY CASE
    target_locs = np.array([[5.0000,  0.0000],
                            [3.8302,  3.2139],
                            [0.8682,  4.9240],
                            [-2.5000, 4.3301],
                            [-4.6985, 1.7101],
                            [-4.6985, -1.7101],
                            [-2.5000, -4.3301],
                            [0.8682,  -4.9240],
                            [3.8302,  -3.2139]])
    neur_model = build_model(target_locs)

    # Already a square (12 x 12) window. The 9-target sc_equilib solve is by far
    # the expensive part of these figures, so num_x/num_y and refinement_levels
    # are the main speed lever -- FLYGEOM_FAST=1 drops them for layout checks.
    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    with Pool(get_n_workers()) as pool:
        neur_model.plot_bifurcation_diagram(xlim=(-6, 6), num_x=NUM_X,
                                            ylim=(-6, 6), num_y=NUM_Y,
                                            refinement_levels=RLEV,
                                            max_count=None, pool=pool, ax=ax,
                                            title=None, wb_plot=False,
                                            stability_criterion='reduced')

    ax.set_title('Fly geometry: 9 targets')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # The stable-count legend is long (many counts for 9 targets), so place it
    # OUTSIDE the axes, to the right.
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              title='# stable\nequilibria', frameon=False)

    fig.savefig(OUT_9TARGET, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT_9TARGET)


def main(argv=()):
    # Optional selector: the 9-target sc_equilib solve takes far longer than the
    # 3-target pair, so allow rendering either figure on its own.
    which = argv[0].lower() if argv else 'all'
    if which not in ('all', '3', '3target', '9', '9target'):
        raise SystemExit(f'usage: python fly_geom.py [all|3target|9target] '
                         f'(got {which!r})')
    if which in ('all', '3', '3target'):
        figure_3target()
    if which in ('all', '9', '9target'):
        figure_9target()


if __name__ == '__main__':
    main(sys.argv[1:])
