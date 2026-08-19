'''
Script to plot targets and perception from a given location
'''

import numpy as np
import matplotlib.pyplot as plt

import decision_model as model

# focal_loc = np.array([1,10])
# focal_angle = 0
# #focal_angle = -1.1 - np.pi/2

# targets = model.Targets(locs=np.array([[9,12],[15,14],[13,7]]), geom_name='circle', 
#                         r=np.array([0.5, 1.25, 0.75]))
# # targets = model.Targets(geom_name='segment', l=1, theta=np.array([0.2, 2.5]))

# percep_model = model.PerceptionModel(targets, focal_loc, focal_angle)

# percep_model.plot()

# dir_model = model.DirectionModel(percep_model)

# # dir_model.plot_weighting()

# dir_model.plot_hamiltonian(with_signal=True)

# # num_stps = 1000
# # focal_loc_mesh = np.column_stack((np.linspace(1,14,num_stps), 10*np.ones(num_stps)))
# # dir_model.plot_hamiltonian(focal_loc_mesh, with_signal=True)




target_locs = np.array([[20,5],[20,15]]) # (x,y) coordinates

targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)

focal_loc = (0,10)
focal_angle = 0

percep_model = model.PerceptionModel(targets, focal_loc, focal_angle)

# You can plot the perception model to see the current geometry and perception signal
# percep_model.plot(wb_plot=True)

weighting_name='truncnorm'
mu = 0
sigma = np.pi/8
left = -np.pi
right = np.pi

consensus_type='additive'

dir_model = model.DirectionModel(percep_model, consensus_type, weighting_name, mu, sigma, left, right)

dir_model.plot_walker()