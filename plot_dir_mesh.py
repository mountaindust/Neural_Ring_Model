'''
A script that uses the Ising model to plot a direction mesh based on running the 
torque model dynamical system for a long time and arriving at steady states.

It is parameterized using the same geometry as in the the PRXLife and PNAS papers, 
but with non-delta function target geometries, signal blocking, and a signal strength.

The main purpose of this is to parallelize the code so that the plots can be generated
faster.
'''

from multiprocessing import Pool
import pickle
import numpy as np
import matplotlib.pyplot as plt
import decision_model as model


def run_direction_mesh(pool=None):
    # First, define target locations and geometries as a Targets object.
    target_locs = np.array([[4.33,2.5],[4.33,-2.5]]) # (x,y) coordinates
    # targets = model.Targets(locs=target_locs, geom_name=None)
    targets = model.Targets(locs=target_locs, geom_name='circle', r=0.15)


    # Next, define an observer location and angle of observation.
    focal_loc = (0,0)
    focal_angle = 0
    percep_model = model.PerceptionModel(targets, focal_loc, focal_angle)

    # Plot the perception model to see the current geometry and perception signal
    # percep_model.plot_blocked_signals()

    # Now, define an Ising model based on the perception model.
    dir_model = model.IsingExtModel(percep_model)

    # Plot the truncated cosine interaction function
    # dir_model.plot_trunccosine()

    # Plot the direction mesh by running the dynamical system to steady state
    data = dir_model.plot_direction_mesh(pool=pool)

    pickle.dump(data, open("direction_mesh_data.pkl", "wb"))    

    # TODO: This is acting very stiff. May need to use a stiff solver.
    #     Also, output data from this function call so we don't have to recompute
    #     while adjusting plotting parameters.


if __name__ == "__main__":
    # Create a multiprocessing pool to parallelize the computation
    with Pool(10) as pool:
        run_direction_mesh(pool)

# # For testing without multiprocessing
# if __name__ == "__main__":
#     run_direction_mesh()