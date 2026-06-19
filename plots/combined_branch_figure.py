"""Publication figure: combined fly+locust SC-equilibrium HEADING branch diagram.

Takes the *heading* (x, theta) row of each per-case branch diagram -- fly on top,
locust on bottom, one column per y-cut -- and stacks them into one figure (the
(x, R) coherence rows of the per-case diagrams are dropped). This is exactly
``decision_skeleton.plot_diagram_both`` (the renderer behind the analysis-only
``branch_diagram_both.png``), reused here -- no duplicated plotting code -- and
saved as a 300-dpi publication pair ``branch_diagram_combined.{jpg,tif}``.
The stable/unstable legend sits at lower left (matching the per-case diagrams).

Run (from the plots/ directory):  python combined_branch_figure.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    sys.path.insert(0, p)

from decision_skeleton import plot_diagram_both        # noqa: E402

OUT_JPG = os.path.join(HERE, 'branch_diagram_combined.jpg')
OUT_TIF = os.path.join(HERE, 'branch_diagram_combined.tif')


def main():
    fig = plot_diagram_both()          # save=None -> just returns the figure
    fig.savefig(OUT_JPG, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_TIF, dpi=300, bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print('wrote', OUT_JPG)
    print('wrote', OUT_TIF)


if __name__ == '__main__':
    main()
