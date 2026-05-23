"""
Exploratory version of VM_bifurcations/diagnostic_recount_compare.png.
Companion check that quantifies how much the legacy gamma-only `_discrim_A`
stability test overcounts vs. the full 3x3 coupled Jacobian.

Three-panel bifurcation diagram for the neural-band model with two
circle targets at TARGET_LOCS:

  (1) # stable equilibria using the legacy gamma-only `_discrim_A` test
  (2) # stable equilibria using the full 3x3 coupled Jacobian (correct)
  (3) Difference (1) - (2): cells the legacy test overcounts

Panels (1) and (2) use NeuralBandModel.plot_bifurcation_diagram, which
adaptively refines cell boundaries. Panel (3) is computed on a uniform
fine grid using the same _count_stable_at helper, since the diff requires
both criteria evaluated at the same points.

These plots are intentionally NOT publication-quality -- the grid is sized
for fast iteration on a many-core machine. Bump NUM_X / NUM_Y / DIFF_NX /
DIFF_NY / REFINEMENT_LEVELS for finer resolution at the cost of runtime.

Set NEURAL_WEIGHT to 'vonmises' or 'cutoff' to choose the front-bias
weighting. The output filename is suffixed with _VM or _SC accordingly.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool

import decision_model as model
from parallel_config import get_n_workers


# ---- weighting choice ----
# 'vonmises' -> von Mises pdf with parameter K_VONMISES (suffix _VM)
# 'cutoff'   -> smooth cutoff with parameters A_CUTOFF, B_CUTOFF (suffix _SC)
NEURAL_WEIGHT = 'cutoff'

K_VONMISES = 0.55
A_CUTOFF = np.pi / 3
B_CUTOFF = 4 * np.pi / 5

# ---- model setup (matches diagnostic_recount_grid.py) ----
TARGET_LOCS = np.array([[4.33, 2.5], [4.33, -2.5]])
TARGET_RADIUS = 0.5
XLIM = (0.0, 6.0)
YLIM = (-3.5, 3.5)

# Plot-bifurcation-diagram settings for panels 1 and 2 (adaptive).
# Exploration-quality; bump up for finer Hopf-region resolution at the cost
# of runtime.
NUM_X = 29
NUM_Y = 29
REFINEMENT_LEVELS = 2
BOUNDARY_DILATION = 1
MAX_COUNT = 3   # pin colour scale; >=3 stable equilibria not expected here

# Uniform grid for the difference panel. Coarser than HQ so each pair of
# stability evaluations stays fast.
DIFF_NX = 121
DIFF_NY = 121

N_WORKERS = get_n_workers()

_WEIGHT_SUFFIX = {'vonmises': '_VM', 'cutoff': '_SC'}
if NEURAL_WEIGHT not in _WEIGHT_SUFFIX:
    raise ValueError(
        f"NEURAL_WEIGHT must be 'vonmises' or 'cutoff', got {NEURAL_WEIGHT!r}")
OUTPUT_NAME = (f"bifurcation_compare_discrim_vs_coupled"
               f"{_WEIGHT_SUFFIX[NEURAL_WEIGHT]}.png")


def weight_label():
    """Human-readable name and parameter string for the active weighting."""
    if NEURAL_WEIGHT == 'vonmises':
        return 'von Mises', rf'$k={K_VONMISES:g}$'
    return ('smooth cutoff',
            rf'$a={A_CUTOFF/np.pi:.2g}\pi,\ b={B_CUTOFF/np.pi:.2g}\pi$')


def build_model():
    targets = model.Targets(locs=TARGET_LOCS, geom_name='circle',
                            r=TARGET_RADIUS)
    percep = model.PerceptionModel(targets, (0, 0), 0,
                                   neural_weight=NEURAL_WEIGHT,
                                   neural_angle='integral')
    if NEURAL_WEIGHT == 'vonmises':
        percep.k = K_VONMISES
    else:
        percep.a = A_CUTOFF
        percep.b = B_CUTOFF
    return model.NeuralBandModel(percep)


def compute_diff_grid(nbm, pool):
    """Evaluate both stability criteria on a uniform DIFF_NX x DIFF_NY grid
    and return (xs, ys, grid_disc, grid_coupled)."""
    xs = np.linspace(XLIM[0], XLIM[1], DIFF_NX)
    ys = np.linspace(YLIM[0], YLIM[1], DIFF_NY)

    disc_args = [((j, i), xs[i], ys[j], 'discrim_a')
                 for j in range(DIFF_NY) for i in range(DIFF_NX)]
    coupled_args = [((j, i), xs[i], ys[j], 'coupled')
                    for j in range(DIFF_NY) for i in range(DIFF_NX)]

    print(f"Diff panel: evaluating {DIFF_NX}x{DIFF_NY} grid "
          f"({2 * DIFF_NX * DIFF_NY} stability counts)...")
    disc_results = pool.map(nbm._count_stable_at, disc_args)
    coupled_results = pool.map(nbm._count_stable_at, coupled_args)

    grid_disc = np.zeros((DIFF_NY, DIFF_NX), dtype=int)
    grid_coupled = np.zeros((DIFF_NY, DIFF_NX), dtype=int)
    for (j, i), c in disc_results:
        grid_disc[j, i] = c
    for (j, i), c in coupled_results:
        grid_coupled[j, i] = c
    return xs, ys, grid_disc, grid_coupled


def main():
    nbm = build_model()

    fig, axes = plt.subplots(1, 3, figsize=(12, 6.5))
    ax_disc, ax_coup, ax_diff = axes

    with Pool(N_WORKERS) as pool:
        # Panels 1 and 2: adaptive bifurcation diagrams.
        print("Panel 1: discrim_A bifurcation diagram (adaptive)...")
        nbm.plot_bifurcation_diagram(
            xlim=XLIM, ylim=YLIM,
            num_x=NUM_X, num_y=NUM_Y,
            refinement_levels=REFINEMENT_LEVELS,
            boundary_dilation=BOUNDARY_DILATION,
            max_count=MAX_COUNT,
            pool=pool,
            ax=ax_disc,
            stability_criterion='discrim_a',
            title=r'(a) gamma-only discriminant')

        print("Panel 2: coupled Jacobian bifurcation diagram (adaptive)...")
        nbm.plot_bifurcation_diagram(
            xlim=XLIM, ylim=YLIM,
            num_x=NUM_X, num_y=NUM_Y,
            refinement_levels=REFINEMENT_LEVELS,
            boundary_dilation=BOUNDARY_DILATION,
            max_count=MAX_COUNT,
            pool=pool,
            ax=ax_coup,
            stability_criterion='coupled',
            title=r'(b) coupled 3D Jacobian (correct)')

        # Panel 3: uniform-grid difference.
        xs, ys, grid_disc, grid_coupled = compute_diff_grid(nbm, pool)

    diff = grid_disc - grid_coupled
    print(f"Diff distribution: min={diff.min()}, max={diff.max()}, "
          f"# nonzero pixels = {(diff != 0).sum()}/{diff.size}")

    # Style for the count panels: a discrete colorbar, single source of truth.
    cmap_count = plt.get_cmap('viridis', MAX_COUNT + 1)
    norm_count = BoundaryNorm(boundaries=np.arange(-0.5, MAX_COUNT + 1.5),
                              ncolors=MAX_COUNT + 1)
    sm_count = plt.cm.ScalarMappable(cmap=cmap_count, norm=norm_count)
    sm_count.set_array([])

    cax = fig.add_axes([0.35, 0.08, 0.28, 0.025])  # below panels (a),(b)
    cb = fig.colorbar(sm_count, cax=cax, orientation='horizontal',
                      ticks=np.arange(0, MAX_COUNT + 1))
    cb.set_label('# stable self-consistent equilibria', fontsize=12)

    # Difference panel.
    cmax = max(1, int(np.abs(diff).max()))
    im_diff = ax_diff.imshow(
        diff, origin='lower',
        extent=[XLIM[0], XLIM[1], YLIM[0], YLIM[1]],
        aspect='equal', interpolation='nearest',
        cmap='RdBu_r', vmin=-cmax, vmax=cmax)
    nbm.percep_model.targets.plot_targets_to_axis(ax_diff)
    ax_diff.set_title('(c) a - b overcount')

    # Allocate matching right-side cax slots on all three panels so that
    # aspect='equal' resolves to the same axes height. Panels (a) and (b)
    # get hidden spacer axes; panel (c) gets the visible diff colorbar.
    cax_size = "4%"
    cax_pad = 0.08
    for ax in (ax_disc, ax_coup):
        spacer = make_axes_locatable(ax).append_axes(
            "right", size=cax_size, pad=cax_pad)
        spacer.axis('off')
    cax_diff = make_axes_locatable(ax_diff).append_axes(
        "right", size=cax_size, pad=cax_pad)
    cb_diff = fig.colorbar(im_diff, cax=cax_diff,
                           ticks=np.arange(-cmax, cmax + 1))
    cb_diff.set_label('count difference', fontsize=12)

    # Common axis cosmetics.
    for ax, label in zip(axes, ['(a)', '(b)', '(c)']):
        if label == '(b)':
            ax.set_xlabel('observer x-coordinate', fontsize=12)
        if label == '(a)':
            ax.set_ylabel('observer y-coordinate', fontsize=12)
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)

    weight_name, weight_params = weight_label()
    target_str = ', '.join(
        rf'$({x:g}, {y:+g})$' for x, y in TARGET_LOCS)
    fig.suptitle('Self-consistent equilibria: gamma-only stability vs. '
                 'coupled 3D Jacobian\n'
                 rf'({weight_name}, {weight_params}, '
                 rf'circle targets (r={TARGET_RADIUS:g}) at '
                 + target_str + ')',
                 fontsize=14, y=0.88)

    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    fig.subplots_adjust(wspace=0.12)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            OUTPUT_NAME)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
