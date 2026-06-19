"""Publication panel: fly bifurcation diagram with basin-of-attraction wheels.

Renders the three-target fly (GODM-refit) stable-equilibrium-count map over
(x, y) with the basin-wheel overlay -- each wheel a heading-basin annulus plus
direction arrows, the lattice of wheels placed symmetrically about y=0 and
seeded on the multistable regions (NBM.plot_bifurcation_diagram(
overlay_basins=True)). Writes ``fly_bifurcation.png`` next to this script.

The fly model (geometry + GODM-refit warp/weight/K) is imported from
``walker_analysis/three_target_fly.py`` so the params stay single-sourced.

Run (from the plots/ directory):  python fly_bifurcation_plot.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'walker_analysis'))

from parallel_config import get_n_workers          # noqa: E402
from three_target_fly import build_model            # noqa: E402

OUT = os.path.join(HERE, 'fly_bifurcation.png')


def main():
    nbm = build_model()
    with Pool(get_n_workers()) as pool:
        fig, ax = plt.subplots(figsize=(8.5, 9))
        nbm.plot_bifurcation_diagram(
            xlim=(0.0, 5.4), ylim=(-3.7, 3.7), num_x=26, num_y=31,
            refinement_levels=3, boundary_dilation=1, pool=pool, ax=ax,
            overlay_basins=True, basin_nx=6, basin_ny=4)
        ax.legend(title='# stable\nequilibria', loc='center left',
                  bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout()
        fig.savefig(OUT, dpi=150)
        plt.close(fig)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
