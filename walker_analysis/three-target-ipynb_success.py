'''
For this to work, in decision_model.py you need gamma noise:
init_gamma = self.gamma + self.rng.normal()*1e-2 + 1j*self.rng.normal()*1e-2
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import decision_model as model

# TWO TARGET CASE FROM PAPER
# target_locs = np.array([[4.33,2.5],[4.33,-2.5]]) # (x,y) coordinates

# THREE TARGET CASE FROM PAPER
target_locs = np.array([[5.0000,  0.0000],
[3.8302,  3.2139],
# [0.8682,  4.9240],
# [-2.5000, 4.3301],
# [-4.6985, 1.7101],
# [-4.6985, -1.7101],
# [-2.5000, -4.3301],
# [0.8682,  -4.9240],
[3.8302,  -3.2139]])

# --------------- FOUR TARGET CASE ---------------
# target_locs = np.array([[4.33,2.25],[4.33,-2.25],
#                         [4.33,0.75],[4.33,-0.75]])

# targets = model.Targets(locs=target_locs, geom_name=None)
targets = model.Targets(locs=target_locs, geom_name='circle', r=0.1)

focal_loc = (-2,0)
focal_angle = 0
percep_model = model.PerceptionModel(targets, focal_loc, focal_angle,
                                     neural_angle_dist='lin_cutoff', 
                                     # angle_weight = None,
                                     angle_weight='neural_angle_dist',
                                     a_warp=0.2, b_warp=0.8*np.pi)

# beta is the neural Boltzmann factor. This scene has 3 targets and the notebook
# result was produced under the earlier per-target temperature T=0.2, whose
# effective coupling was N_targets/T, so beta = 3/0.2 = 15 reproduces it (the
# model default of 10 corresponds to two targets).
neur_model = model.NeuralBandModel(percep_model, beta=15.0)
neur_model.rng = np.random.default_rng(seed = 3)

neur_model.K = 4.5 # Coupling strength for physical turning

fig = plt.figure(figsize=(10,6))
ax = plt.subplot()
# plt.savefig('random_walkers_4circ_low_K.png', dpi=300)
# from multiprocessing import Pool
# with Pool(10) as pool:
#     neur_model.plot_bifurcation_diagram(xlim=(-6,6), num_x=19, ylim=(-6,6),
#                                  num_y=19, refinement_levels=3, max_count=None,
#                                  pool=pool, ax=ax, title=None, wb_plot=True,
#                                  stability_criterion='reduced')
neur_model.plot_walkers(dt=0.1, v=1, std=0.025, repetitions=30, max_steps=200,
                        start_loc=None, start_angle=None, #if None, uses what's set cell above
                        alpha=1, ax=ax, wb_plot=True,
                        title='Three Targets')
plt.show()