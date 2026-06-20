"""Publication figure: two-panel fly bifurcation diagram with basin wheels.

Left  -- two-target fly (GODM 'fly2', 60 deg separation, targets at (4.33, +-2.5)).
Right -- three-target fly (GODM 'fly3', the case worked up in the basin-overlay
         iteration).

Both panels are the SAME fly (identical warp/weight/K/T -- the GODM refit set);
only the target geometry differs, so they share axis limits and a single,
enlarged legend (stable-count colormap + basin-wheel rank). Each panel is
``NBM.plot_bifurcation_diagram(overlay_basins=True)``: the stable-equilibrium
count map with a symmetric lattice of basin wheels (heading-basin annulus +
direction arrows). The two model builders are imported (single source of truth):
``walker_analysis/three_target_fly.py`` and ``plots/two_target_fly_refine.py``.

Writes ``fly_bifurcation.jpg`` and ``fly_bifurcation.tif`` (both 300 dpi) next
to this script.

Run (from the plots/ directory):  python fly_bifurcation_plot.py
  FLYBIF_FAST=1 python fly_bifurcation_plot.py   # quick low-res layout check
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, 'walker_analysis'), HERE):
    sys.path.insert(0, p)

from parallel_config import get_n_workers              # noqa: E402
from three_target_fly import build_model as build_three  # noqa: E402
from two_target_fly_refine import build_model as build_two  # noqa: E402

OUT_JPG = os.path.join(HERE, 'fly_bifurcation.jpg')
OUT_TIF = os.path.join(HERE, 'fly_bifurcation.tif')

# Shared frame so the panels align and one legend serves both.
XLIM, YLIM = (0.0, 5.4), (-3.7, 3.7)
MAX_COUNT = 5                       # pin the colour scale across both panels

FAST = bool(os.environ.get('FLYBIF_FAST', ''))
if FAST:                            # coarse -- layout/legend/title check only
    NUM_X, NUM_Y, RLEV = 12, 14, 1
    NC, NB, NXW, NYW = 24, 5, 3, 2
else:                              # publication: ~2x finer base grid, 3 refines
    NUM_X, NUM_Y, RLEV = 52, 62, 3
    NC, NB, NXW, NYW = 64, 12, 6, 4


def render_panel(ax, nbm, title, pool):
    nbm.plot_bifurcation_diagram(
        xlim=XLIM, ylim=YLIM, num_x=NUM_X, num_y=NUM_Y, refinement_levels=RLEV,
        boundary_dilation=1, pool=pool, ax=ax, title=title, max_count=MAX_COUNT,
        overlay_basins=True, basin_nx=NXW, basin_ny=NYW,
        basin_n_coarse=NC, basin_n_bisect=NB)


def main():
    two, three = build_two(), build_three()
    with Pool(get_n_workers()) as pool:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.0, 6.0),
                                       layout='constrained')
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02,
                                    hspace=0.0)
        render_panel(axL, two, 'Fly two-target bifurcation diagram', pool)
        render_panel(axR, three, 'Fly three-target bifurcation diagram', pool)
        # the 3-target target circles (y=+-3.21, r=0.5) reach +-3.71 and would
        # autoscale that panel past the 2-target's limits; clamp both identical
        for a in (axL, axR):
            a.set_xlim(*XLIM)
            a.set_ylim(*YLIM)

    # --- collect handles for ONE shared, enlarged legend (both panels match) ---
    # stable-count colormap proxies (squares labelled '0'..'5')
    ch, cl = axR.get_legend_handles_labels()
    count_h = [h for h, lab in zip(ch, cl) if lab.isdigit()]
    count_l = [lab for lab in cl if lab.isdigit()]
    # basin-wheel rank patches, lifted from whichever panel exposes the most
    # ranks (each panel's auto rank legend is added inside _render_basin_wheels)
    rank_h, rank_l = [], []
    for a in (axR, axL):
        leg = a.get_legend()
        if leg is None:
            continue
        hs = getattr(leg, 'legend_handles', None) or getattr(
            leg, 'legendHandles', [])
        if len(hs) > len(rank_h):
            rank_h = [Patch(facecolor=h.get_facecolor(), edgecolor='0.3')
                      for h in hs]
            rank_l = [t.get_text() for t in leg.get_texts()]
    # drop every per-panel legend so only the shared ones remain (the overlay
    # registers its rank legend twice -- ax.legend() + add_artist -- so dedup
    # by id and guard the remove)
    seen = set()
    for a in (axL, axR):
        for child in list(a.get_children()):
            if isinstance(child, Legend) and id(child) not in seen:
                seen.add(id(child))
                try:
                    child.remove()
                except ValueError:
                    pass
        a.legend_ = None

    # one compact, vertically-centred legend covering both panels: the count
    # colormap, then an inline sub-header and the basin-wheel rank patches
    spacer = Line2D([], [], color='none')
    handles = count_h + [spacer, spacer] + rank_h
    labels = count_l + ['', 'basin wheel rank:'] + rank_l
    fig.legend(handles, labels, title='# stable equilibria',
               loc='outside right', fontsize=13, title_fontsize=14,
               markerscale=1.8, labelspacing=0.5, handletextpad=0.6)

    fig.savefig(OUT_JPG, dpi=300)
    fig.savefig(OUT_TIF, dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print('wrote', OUT_JPG)
    print('wrote', OUT_TIF)


if __name__ == '__main__':
    main()
