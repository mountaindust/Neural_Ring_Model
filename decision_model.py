'''
Sets up a scenario in which a single locust makes decisions about the direction
it wants to go based on static targets with certain geometry
'''

import numpy as np

class targets:

    def __init__(self, Lxy=(12,12), pos=np.array([[8,4],[8,8]]), theta=None, geom=None):
        '''Set up a domain using dimensions Lx, Ly, with LLC at the origin.
        Create and store information about targets located at (pos_x,pos_y) with 
        orientation given by theta and geometry given by geom.

        Parameters
        ----------
        Lxy : (x,y) of floats
            length of domain in the x and y direction
        pos : Nx2 ndarray
            x,y coordinates of targets
        theta : length N ndarray TODO
            orientation of targets
        geom : {TODO}, (optional)
            geometry of targets
        '''

        self.Lxy = Lxy
        self.target_locs = pos
        self.target_angles = theta
        self.geom = geom

    

    