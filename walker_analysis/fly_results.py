"""Publication panel: fly three-target walker tracks over the GODM fly3 heatmap.

Loads the self-contained ``three_target_fly_refine.npz`` (walker tracks + the
empirical heatmap + parameters + Pearson corr) written by
``three_target_fly_refine.py`` and renders a single, undistorted panel suitable
for a journal subplot: the empirical occupancy heatmap with a random selection of
walker trajectories overlaid at alpha=0.4.

No pandas / GODM-data dependency -- everything needed is in the npz.

Run:  python walker_analysis/fly_results.py
      NR_MAX_TRACKS=200 python walker_analysis/fly_results.py    # tune overlay density
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, 'three_target_fly_refine.npz')
OUT = os.path.join(HERE, 'fly_results_3target.png')

# Overlaying all tracks at alpha=0.4 saturates the bright ridge and hides the
# heatmap underneath, so a random subset is drawn. Override with NR_MAX_TRACKS.
MAX_TRACKS = int(os.environ.get('NR_MAX_TRACKS', 100))
TRACK_ALPHA = 0.4
SUBSET_SEED = 0


def main():
    if not os.path.exists(NPZ):
        sys.exit('missing %s -- run three_target_fly_refine.py first '
                 '(NR_REPS=2500 for the std=4.0 result)' % NPZ)
    d = np.load(NPZ, allow_pickle=True)
    walks = d['walks']                      # object array of (2, n) tracks
    ref_img = d['ref_img']
    extent = tuple(d['extent'].tolist())    # (xmin, xmax, ymin, ymax)
    f = lambda k: float(d[k])
    pi = np.pi

    n_total = len(walks)
    if n_total > MAX_TRACKS:
        idx = np.random.default_rng(SUBSET_SEED).choice(n_total, MAX_TRACKS, replace=False)
        n_plotted = MAX_TRACKS
    else:
        idx = np.arange(n_total)
        n_plotted = n_total
    print('plotting %d of %d tracks (alpha=%.2f)' % (n_plotted, n_total, TRACK_ALPHA))

    # --- figure: equal aspect so 1 x-unit == 1 y-unit (no distortion) ---
    x0, x1, y0, y1 = extent
    width_in = 5.0                                  # >= 4 in as required
    height_in = width_in * (y1 - y0) / (x1 - x0)    # match the data aspect
    fig, ax = plt.subplots(figsize=(width_in, height_in + 1.1))  # +title space

    ax.imshow(ref_img, extent=extent, origin='upper', aspect='equal',
              interpolation='bilinear')
    for i in idx:
        w = walks[i]
        ax.plot(w[0], w[1], color='k', alpha=TRACK_ALPHA, lw=0.5,
                solid_capstyle='round')
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect('equal')                          # true aspect, no distortion
    ax.set_xlabel('x'); ax.set_ylabel('y')

    params = (r'$K$=%.1f  $T$=%.2f  $\sigma$=%.1f  $v$=%.2f  $dt$=%.2f' '\n'
              r'warp $(a,b)$=(%.2f, %.2f)$\pi$   weight $(a,b)$=(%.2f, %.2f)$\pi$'
              '\n'
              r'$q$(noise_exp)=%g  $p$(R_exp)=%g   start jitter '
              r'$\sigma_{pos}$=%.3f  $\sigma_{head}$=%.0f$\degree$'
              % (f('K'), f('T'), f('std'), f('v'), f('dt'),
                 f('a_warp')/pi, f('b_warp')/pi, f('a_weight')/pi, f('b_weight')/pi,
                 f('noise_exp'), f('R_exp'),
                 f('start_pos_std'), np.degrees(f('start_head_std'))))
    title = ('Fly 3-target walker tracks on GODM heatmap '
             '(showing %d/%d tracks)\n%s\ncorr(all)=%.3f'
             % (n_plotted, n_total, params, f('corr_all')))
    ax.set_title(title, fontsize=8)

    fig.savefig(OUT, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    # Report final raster size so the >=4 in / 300 dpi requirement is verifiable.
    try:
        from PIL import Image
        w_px, h_px = Image.open(OUT).size
        print('wrote %s  (%d x %d px = %.2f x %.2f in @ 300 dpi)'
              % (OUT, w_px, h_px, w_px / 300.0, h_px / 300.0))
    except Exception:
        print('wrote', OUT)


if __name__ == '__main__':
    main()
