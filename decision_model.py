'''
Sets up a scenario in which a single locust makes decisions about the direction
it wants to go based on static targets with certain geometry
'''

import numpy as np

class Targets:

    def __init__(self, pos=None, geom_name=None, r=None, l=None, theta=0):
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
        geom : {'circle'}, (optional)
            geometry of targets. Depending on the choice, additional parameters 
            must be set to quantify the geometry. Options are:
            - 'circle' : must specify a radius r to be used for all targets or 
            an array of radii r, one for each target. The position of the target 
            is the midpoint of the circle.
            - 'segment' : must specify a length l to be used for all targets or 
            an array of lengths l, one for each target. Similarly, must specify 
            a theta (for all targets or each), specifying the angle each target 
            segment is at in the plane. The position of the target is the 
            midpoint of the segment.
        r : float or ndarray
            radius of circles in the geometry; see geom for requirements
        l : float or ndarray
            line segment lengths in the geometry; see geom for requirements
        theta : float or length N ndarray, default=0
            orientation of targets; see geom for requirements
        '''

        if pos is None:
            self.locs = np.array([[15,5],[15,15]])
        else:
            self.locs = pos
        self.geom_name = geom_name
        self.r = r
        self.l = l
        self.theta = self.convert_angles(theta)

    
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

        if self.geom_name is None:
            ##### Point targets #####
            # Get a vector toward each target
            vecs = self.locs - loc
            target_angles = np.arctan2(vecs[:,1],vecs[:,0])
            return self.convert_angles(target_angles - angle)
        elif self.geom_name == 'circle':
            vecs = self.locs - loc
            target_angles = np.arctan2(vecs[:,1],vecs[:,0])
            if vecs.ndim > 1:
                vecs_length = np.linalg.norm(vecs, axis=1)
            else:
                vecs_length = np.linalg.norm(vecs)
            pm_theta = np.arcsin(self.r/vecs_length)
            return self.convert_angles(np.column_stack([target_angles-pm_theta-angle,
                                                        target_angles+pm_theta-angle]))
        elif self.geom_name == 'segment':
            # find location of segment endpoints
            diff = np.column_stack([self.l/2*np.cos(self.theta),
                                    self.l/2*np.sin(self.theta)])
            endpt1 = self.locs + diff
            endpt2 = self.locs - diff # difference in heading angle is pi between seg endpoints
            # get a vector to each
            vecs1 = endpt1 - loc; vecs2 = endpt2 - loc
            # get angles to each
            angles1 = np.arctan2(vecs1[:,1],vecs1[:,0])
            angles2 = np.arctan2(vecs2[:,1],vecs2[:,0])
            # store sorted and return
            target_angles = np.zeros((len(angles1),2))
            one_two = angles1 <= angles2
            target_angles[one_two,:] = self.convert_angles(
                                        np.column_stack([angles1[one_two]-angle,
                                                         angles2[one_two]-angle]))
            target_angles[~one_two,:] = self.convert_angles(
                                        np.column_stack([angles2[~one_two]-angle,
                                                         angles1[~one_two]-angle]))
            return target_angles
        

    @staticmethod
    def convert_angles(theta):
        '''Given a scalar or array of angles, convert to angles in 
        [-np.pi,np.pi]
        '''
        return theta - (theta+np.pi)//(2*np.pi)*2*np.pi
