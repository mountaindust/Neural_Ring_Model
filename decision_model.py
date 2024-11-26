'''
Sets up a scenario in which a single locust makes decisions about the direction
it wants to go based on static targets with certain geometry
'''

import numpy as np

class targets:

    def __init__(self, pos=None, theta=None, geom=None):
        '''Set up targets for attraction model.
        The only thing taken care of here is storage of target locations and 
        calculation of unbiased, unwarped perception of the targets (angluar 
        interval) depending on the geometry of the targets.

        Default is two targets located at (15,5) and (15,15) so that an organism 
        starting at (0,10) is right inbetween them as it moves along the 
        x-direction.

        Parameters
        ----------
        pos : Nx2 ndarray (default=np.array([[15,5],[15,15]]))
            x,y coordinates of targets
        theta : length N ndarray TODO
            orientation of targets
        geom : {TODO}, (optional)
            geometry of targets
        '''

        if pos is None:
            self.locs = np.array([[15,5],[15,15]])
        else:
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
            
