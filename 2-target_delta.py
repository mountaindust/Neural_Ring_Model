import numpy as np
import matplotlib.pyplot as plt
import decision_model as model

target_locs = np.array([[4.33,2.5],[4.33,-2.5]]) # (x,y) coordinates

# targets = model.Targets(locs=target_locs, geom_name=None)

targets = model.Targets(locs=target_locs, geom_name='circle', r=0.25)

focal_loc = (0,0)
focal_angle = 0
# percep_model = model.PerceptionModel(targets, focal_loc, focal_angle,
#                                      neural_angle_dist=None, 
#                                      angle_weight=None)

percep_model = model.PerceptionModel(targets, focal_loc, focal_angle,
                                     neural_angle_dist='lin_cutoff', 
                                     angle_weight='neural_angle_dist',
                                     a_warp=0.25*np.pi, b_warp=0.9*np.pi)

neur_model = model.NeuralBandModel(percep_model, beta=10.0)
neur_model.rng = np.random.default_rng(seed = 3)

neur_model.K = 4.5 # Coupling strength for physical turning

neur_model.plot_walkers(dt=0.1, v=0.3, std=0.1, noise_exp=0,  
                        repetitions=50, target_tol=0.25,
                        start_loc=None, start_angle=None, #if None, uses what's set cell above
                        target_color='0.5', alpha=0.6,
                        title='2 circle targets, K=4.5, constant noise $\\sigma=0.1$')

plt.show()