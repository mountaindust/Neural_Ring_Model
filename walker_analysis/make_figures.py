"""Representative walker figures for the half-angle (sin(ego/2)) model.

Produces 13 heatmap-with-tracks figures documenting walker behavior under the
half-angle heading torque dtheta/dt = K*R*sin(ego/2). The figures sweep the
turning gain K to show how it shapes the approach; with the corrected turning
step (the rate is no longer wrapped before the Euler update) every
configuration homes all 30/30 walkers, so the differences are in trajectory
shape, not capture success:

  - K=2  (default):  4 delta, 4 circle, 2 delta, 2 circle   
  - K=4:             4 delta, 4 circle, 2 delta, 2 circle   
  - K=6:             4 delta, 4 circle, 2 delta, 2 circle   
  - K=10:            4 delta                                

All: observer starts at (0,0) facing +x (east); cutoff warp a=0,b=pi with
uniform weighting (angle_weight=None); std=0.1 angular noise; v=1, dt=0.1;
30 walkers; seed 0. Titles are labelled with the K value. The K=4 and K=6
figures use a clamped view so a single wide track does not blow up the axes.

See README.md for the shared setup and K-series summary.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decision_model as dm

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0

FOUR = np.array([(4.33, 2.25), (4.33, -2.25), (4.33, 0.75), (4.33, -0.75)])
TWO = np.array([(4.33, 2.5), (4.33, -2.5)])


def make_model(locs, geom, K, r=None):
    targets = dm.Targets(locs=locs, geom_name=geom, r=r)
    pm = dm.PerceptionModel(targets=targets, focal_loc=(0.0, 0.0),
                            focal_angle=0.0,
                            neural_angle_dist='cutoff', a_warp=0.0, b_warp=np.pi,
                            angle_weight=None)
    return dm.NeuralBandModel(percep_model=pm, T=0.2, K=K)


def figure(locs, geom, K, fname, title, *, r=None, max_steps=1500,
           xlim=None, ylim=None):
    nbm = make_model(locs, geom, K, r=r)
    nbm.rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    nbm.plot_walkers(dt=0.1, v=1, std=0.1, repetitions=30, max_steps=max_steps,
                     start_loc=(0.0, 0.0), start_angle=0.0,
                     plot_tracks=True, ax=ax, title=title)
    # mark target positions for clarity
    ax.plot(locs[:, 0], locs[:, 1], 'r*', markersize=12, zorder=5)
    # optionally clamp the view so a single escaping walker does not blow up
    # the auto-scaled axes (the heatmap bins are still computed over the full
    # range, so resolution in-window is unchanged)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    out = os.path.join(HERE, fname)
    fig.savefig(out, dpi=90, bbox_inches='tight')
    plt.close(fig)
    kb = os.path.getsize(out) / 1024
    print(f"  {fname:28s} ({kb:5.1f} KB)  {title}")
    return out


def main():
    print("Generating walker figures:")
    figure(FOUR, None, 2, 'walkers_4delta_K2.png',
           '4 delta targets, K=2')
    figure(FOUR, 'circle', 2, 'walkers_4circle_K2.png',
           '4 circle targets (r=0.5), K=2', r=0.5)
    figure(FOUR, None, 10, 'walkers_4delta_K10.png',
           '4 delta targets, K=10') 
    figure(TWO, None, 2, 'walkers_2delta_K2.png',
           '2 delta targets, K=2')
    figure(TWO, 'circle', 2, 'walkers_2circle_K2.png',
           '2 circle targets (r=0.5), K=2', r=0.5)

    # K=4: delta cases can produce an escaping walker, so clamp the view to
    # the target region for readable, comparable figures.
    figure(FOUR, None, 4, 'walkers_4delta_K4.png',
           '4 delta targets, K=4', xlim=(-1, 7), ylim=(-5, 5))
    figure(FOUR, 'circle', 4, 'walkers_4circle_K4.png',
           '4 circle targets (r=0.5), K=4', r=0.5, xlim=(-1, 7), ylim=(-5, 5))
    figure(TWO, None, 4, 'walkers_2delta_K4.png',
           '2 delta targets, K=4', xlim=(-1, 7), ylim=(-5, 5))
    figure(TWO, 'circle', 4, 'walkers_2circle_K4.png',
           '2 circle targets (r=0.5), K=4', r=0.5, xlim=(-1, 7), ylim=(-5, 5))

    # K=6 (above the default): sharper turning. Same clamped view for
    # comparability across the K series.
    figure(FOUR, None, 6, 'walkers_4delta_K6.png',
           '4 delta targets, K=6', xlim=(-1, 7), ylim=(-5, 5))
    figure(FOUR, 'circle', 6, 'walkers_4circle_K6.png',
           '4 circle targets (r=0.5), K=6', r=0.5, xlim=(-1, 7), ylim=(-5, 5))
    figure(TWO, None, 6, 'walkers_2delta_K6.png',
           '2 delta targets, K=6', xlim=(-1, 7), ylim=(-5, 5))
    figure(TWO, 'circle', 6, 'walkers_2circle_K6.png',
           '2 circle targets (r=0.5), K=6', r=0.5, xlim=(-1, 7), ylim=(-5, 5))
    print("done.")


if __name__ == '__main__':
    main()
