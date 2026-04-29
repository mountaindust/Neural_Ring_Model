"""
Publication-quality version of VM_bifurcations/diagnostic_recount_compare.png.

Three-panel bifurcation diagram for the von Mises neural-band model with two
circle targets at (4.33, +/-2.5), k=0.55:

  (1) # stable equilibria using the legacy gamma-only `_discrim_A` test
  (2) # stable equilibria using the full 3x3 coupled Jacobian (correct)
  (3) Difference (1) - (2): cells the legacy test overcounts

Panels (1) and (2) use NeuralBandModel.plot_bifurcation_diagram, which adaptively
refines cell boundaries up to a high virtual resolution. Panel (3) is computed
on a uniform fine grid using the same _count_stable_at helper, since the diff
requires both criteria evaluated at the same points.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from multiprocessing import Pool

import decision_model as model


# ---- model setup (matches diagnostic_recount_grid.py) ----
TARGET_LOCS = np.array([[4.33, 2.5], [4.33, -2.5]])
XLIM = (0.0, 6.0)
YLIM = (-3.5, 3.5)

# Plot-bifurcation-diagram settings for panels 1 and 2 (adaptive).
NUM_X = 31 #61
NUM_Y = 31 #61
REFINEMENT_LEVELS = 2 #4
BOUNDARY_DILATION = 1
MAX_COUNT = 3   # pin colour scale; >=3 stable equilibria not expected here

# Uniform fine grid for the difference panel.
DIFF_NX = 61 #241
DIFF_NY = 61 #241

N_WORKERS = 10
OUTPUT_NAME = "bifurcation_compare_discrim_vs_coupled.png"


def build_model():
    targets = model.Targets(locs=TARGET_LOCS, geom_name='circle', r=0.5)
    percep = model.PerceptionModel(targets, (0, 0), 0,
                                   neural_weight='vonmises',
                                   neural_angle='integral')
    percep.k = 0.55
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
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

    cax = fig.add_axes([0.355, 0.08, 0.30, 0.025])  # below panels (a),(b)
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
    cb_diff = fig.colorbar(im_diff, ax=ax_diff,
                           ticks=np.arange(-cmax, cmax + 1),
                           fraction=0.046, pad=0.04)
    cb_diff.set_label('count difference', fontsize=12)

    # Common axis cosmetics.
    for ax, label in zip(axes, ['(a)', '(b)', '(c)']):
        if label == '(b)':
            ax.set_xlabel('x obs. coordinate', fontsize=12)
        if label == '(a)':
            ax.set_ylabel('y obs. coordinate', fontsize=12)
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)

    fig.suptitle('Self-consistent equilibria: legacy gamma-only stability vs. '
                 'coupled 3D Jacobian\n'
                 r'(von Mises, $k=0.55$, two circle targets at '
                 r'$(4.33, \pm 2.5)$)',
                 fontsize=13, y=0.92)

    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    fig.subplots_adjust(wspace=0.12)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            OUTPUT_NAME)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
