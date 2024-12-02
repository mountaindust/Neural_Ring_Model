'''
Script to plot targets and perception from a given location
'''

import numpy as np
import matplotlib.pyplot as plt

import decision_model as model

focal_loc = np.array([5,10])
focal_angle = -1.1 - np.pi/2

# targets = model.Targets(pos=np.array([[9,12],[15,14],[13,7]]), geom_name='circle', 
#                         r=np.array([0.5, 1.25, 0.75]))
targets = model.Targets(geom_name='segment', l=1, theta=np.array([0.2, 2.5]))

percep_model = model.PerceptionModel(focal_loc, focal_angle, targets)

percep_model.plot()

