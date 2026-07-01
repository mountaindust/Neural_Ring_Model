'''
Generate a figure showing some details behind the 3+ target case.
'''

import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, 'walker_analysis'), HERE):
    sys.path.insert(0, p)

from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import decision_model as model

from parallel_config import get_n_workers

def main():

    # THREE TARGET FLY CASE FROM PAPER
    target_locs = np.array([[5.0000,  0.0000],
    [3.8302,  3.2139],
    [3.8302,  -3.2139]])

    targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)

    focal_loc = (0,0)
    focal_angle = 0
    percep_model = model.PerceptionModel(targets, focal_loc, focal_angle,
                                        neural_angle_dist='lin_cutoff', 
                                        angle_weight='neural_angle_dist',
                                        a_warp=0.25*np.pi, b_warp=0.9*np.pi)

    neur_model = model.NeuralBandModel(percep_model, T=0.2, K=4.5)
    neur_model.rng = np.random.default_rng(seed = 3)

    fig = plt.figure(figsize=(12, 6))

    # LEFT panel: 3-target case, bifurcation diagram + walker tracks.
    # The window is a SQUARE (10 x 10) on purpose: plot_bifurcation_diagram
    # renders with aspect='equal', so each axes box is sized to its data-range
    # aspect. Making both panels' windows square is what forces the two
    # subplots to come out the same physical size.
    ax = fig.add_subplot(1, 2, 1)
    with Pool(get_n_workers()) as pool:
        neur_model.plot_bifurcation_diagram(xlim=(-4, 6), num_x=57, ylim=(-5, 5),
                                    num_y=57, refinement_levels=3, max_count=None,
                                    pool=pool, ax=ax, title=None, wb_plot=False,
                                    stability_criterion='reduced')
    neur_model.plot_walkers(dt=0.1, v=0.3, std=0.4, noise_exp=0,
                            repetitions=50,
                            start_loc=None, start_angle=None,
                            alpha=0.35, ax=ax, wb_plot=False,
                            title='Fly geometry: 3 targets, K=4.5, constant noise $\\sigma=0.4$',)
    ax.set_xlim(-4, 6)      # lock the square view (walkers can autoscale it)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper left')

    # --------------------------------------------------------------------------- #

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

    targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)

    percep_model = model.PerceptionModel(targets, focal_loc, focal_angle,
                                        neural_angle_dist='lin_cutoff', 
                                        angle_weight='neural_angle_dist',
                                        a_warp=0.25*np.pi, b_warp=0.9*np.pi)

    neur_model = model.NeuralBandModel(percep_model, T=0.2, K=4.5)

    # RIGHT panel: 9-target case. Already a square (12 x 12) window, so it
    # matches the left panel's size. num_x/num_y and refinement_levels are
    # reduced from 19/19/2 -> 15/15/1: the 9-target sc_equilib solve is the
    # expensive part, so this is the main speed lever. Bump these back up
    # (e.g. 19/refinement_levels=2) for a finer boundary once the layout looks
    # right.
    ax = fig.add_subplot(1, 2, 2)
    with Pool(get_n_workers()) as pool:
        neur_model.plot_bifurcation_diagram(xlim=(-6, 6), num_x=57, ylim=(-6, 6),
                                    num_y=57, refinement_levels=3, max_count=None,
                                    pool=pool, ax=ax, title=None, wb_plot=False,
                                    stability_criterion='reduced')

    ax.set_title('Fly geometry: 9 targets')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # The stable-count legend is long (many counts for 9 targets), so place it
    # OUTSIDE the axes, to the right.
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              title='# stable\nequilibria', frameon=False)

    # Equal-width columns, with room reserved on the right for the external
    # legend. Both axes are square, so they render at the same size.
    fig.subplots_adjust(left=0.06, right=0.86, top=0.92, bottom=0.1, wspace=0.2)

    fig.savefig(os.path.join(HERE, 'fly_geom_refined.png'), dpi=300, bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':
    main()