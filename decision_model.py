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
        self.locs = pos
        self.angles = theta
        self.geom = geom

    
    def get_percep_angles(self,loc,angle=0):
        '''Given the (x,y) coordinate of an observer, loc, return an array of
        angles corresponding to how the targets are percieved from the position 
        of the observer when the observer is facing a direction given by angle.

        Parameters
        ----------
        loc : (x,y) of floats
        angle : float

        Returns
        -------
        Nx2 ndarray of angles theta_1 < theta_2, unless geom is None, then a 
        length N ndarray of single theta values instead.
        '''

        # Get a vector toward each target
        vecs = self.locs - loc

        if self.geom is None:
            ##### Point targets #####
            target_angles = np.arctan2(vecs[:,1],vecs[:,0])
            return target_angles - angle
            
