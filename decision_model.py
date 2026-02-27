'''
Sets up a scenario in which a single locust makes decisions about the direction
it wants to go based on static targets with certain geometry
'''

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root, brentq
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage

def convert_angles(theta):
    '''Given a scalar or array of angles, convert to angles in 
    [-np.pi,np.pi]
    '''
    return theta - (theta+np.pi)//(2*np.pi)*2*np.pi


class Targets:

    def __init__(self, locs=None, geom_name=None, r=None, l=None, theta=0):
        '''Set up targets for attraction model.
        The only thing taken care of here is storage of target locations and 
        calculation of unbiased, unwarped perception of the targets (angluar 
        interval) depending on the geometry of the targets.

        Default is two targets located at (15,5) and (15,15) so that an organism 
        starting at (0,10) is right inbetween them as it moves along the 
        x-direction.

        Parameters
        ----------
        locs : Nx2 ndarray (default=np.array([[15,5],[15,15]]))
            x,y coordinates of targets
        geom : {'circle','segment'}, (optional)
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
        r : float or sequence of length N
            radius of circles in the geometry; see geom for requirements
        l : float or sequence of length N
            line segment lengths in the geometry; see geom for requirements
        theta : float or length N ndarray, default=0
            orientation of targets; see geom for requirements
        '''

        if locs is None:
            self.locs = np.array([[15,5],[15,15]])
        else:
            try:
                if locs.ndim == 1:
                    self.locs = np.array([locs])
                else:
                    self.locs = locs
            except AttributeError:
                print('locs must be an ndarray')
                raise
        self.geom_name = geom_name
        if hasattr(r,'__iter__'):
            assert len(r) == self.locs.shape[0], "length of r must match first dim of locs"
            self.r = np.array(r)
        else:
            self.r = r
        if hasattr(l,'__iter__'):
            assert len(l) == self.locs.shape[0], "length of l must match first dim of locs"
            self.l = np.array(l)
        else:
            self.l = l
        if hasattr(theta,'__iter__'):
            assert len(theta) == self.locs.shape[0], "length of theta must match first dim of locs"
            self.theta = convert_angles(np.array(theta))
        else:
            self.theta = convert_angles(theta)

    
    def get_percep_angles(self,loc,angle=0):
        '''Given the (x,y) coordinate of an observer, loc, return an array of
        angles corresponding to how the targets are percieved (angular extents)
        from the position of the observer when the observer is facing a 
        direction given by angle.

        Returns -pi,pi for any target that loc overlaps, except in the case of 
        delta function targets, which returns angle instead.

        Parameters
        ----------
        loc : (x,y) of floats
        angle : float

        Returns
        -------
        Nx2 ndarray of angles in counter-clockwise order between -pi and pi, 
        unless geom is None, then a length N ndarray of single theta values 
        instead.
        '''
        loc = np.array(loc)

        if self.geom_name is None:
            ##### Point targets #####
            on_target_bool = self.check_target_overlap(loc)
            if np.any(on_target_bool):
                angle_to_targets = self.get_angles_to_targets(loc,angle)
                angle_to_targets[on_target_bool] = angle
                return convert_angles(angle_to_targets)
            else:
                return convert_angles(self.get_angles_to_targets(loc,angle))
        
        elif self.geom_name == 'circle':
            on_target_bool = self.check_target_overlap(loc)
            # Make sure locs is 2D
            if self.locs.ndim == 1:
                self.locs = np.array([self.locs])
            ##### Circle targets #####
            vecs = self.locs[~on_target_bool,:] - loc
            target_angles = np.arctan2(vecs[:,1],vecs[:,0])
            if vecs.ndim > 1:
                vecs_length = np.linalg.norm(vecs, axis=1)
            else:
                vecs_length = np.linalg.norm(vecs)
            if not np.any(on_target_bool):
                pm_theta = np.arcsin(self.r/vecs_length)
                return convert_angles(np.column_stack([target_angles-pm_theta-angle,
                                                            target_angles+pm_theta-angle]))
            else:
                if isinstance(self.r,np.ndarray):
                    pm_theta = np.arcsin(self.r[~on_target_bool]/vecs_length)
                else:
                    pm_theta = np.arcsin(self.r/vecs_length)
                angle_to_targets = np.zeros(self.locs.shape)
                angle_to_targets[on_target_bool,:] = np.array([-np.pi,np.pi])
                angle_to_targets[~on_target_bool,:] = \
                    convert_angles(np.column_stack([target_angles-pm_theta-angle,
                                                         target_angles+pm_theta-angle]))
                return angle_to_targets

        elif self.geom_name == 'segment':
            ##### Segment targets #####
            # find location of segment endpoints
            diff = np.column_stack([self.l/2*np.cos(self.theta),
                                    self.l/2*np.sin(self.theta)])
            endpt1 = self.locs + diff
            endpt2 = self.locs - diff # difference in heading angle is pi between seg endpoints
             # check for loc on segment
            on_target_bool = self.check_target_overlap(loc)
            # get a vector to each
            vecs1 = endpt1[~on_target_bool] - loc
            vecs2 = endpt2[~on_target_bool] - loc
            # get angles to each
            angles1 = convert_angles(np.arctan2(vecs1[:,1],vecs1[:,0])-angle)
            angles2 = convert_angles(np.arctan2(vecs2[:,1],vecs2[:,0])-angle)
            # store sorted
            target_angles = np.zeros((len(angles1),2))
            one_two = np.logical_and(angles1 <= angles2, angles2-angles1 < np.pi)
            one_two = np.logical_or(one_two, np.logical_and(angles1 > angles2,
                                                            angles2+2*np.pi-angles1 < np.pi))
            target_angles[one_two,:] = np.column_stack([angles1[one_two],
                                                         angles2[one_two]])
            target_angles[~one_two,:] = np.column_stack([angles2[~one_two],
                                                         angles1[~one_two]])
            if not np.any(on_target_bool):
                return target_angles
            else:
                angle_to_targets = np.zeros(self.locs.shape)
                angle_to_targets[on_target_bool] = np.array([-np.pi,np.pi])
                angle_to_targets[~on_target_bool] = target_angles
                return angle_to_targets
        

    def get_angles_to_targets(self,loc,angle=0):
        '''Given the (x,y) coordinate of an observer, loc, return an array of
        angles corresponding to where the center of the targets are as percieved 
        from the position of the observer when the observer is facing a 
        direction given by angle.

        This is the same as get_percep_angles when geom_name is None 
        (point targets).

        Parameters
        ----------
        loc : (x,y) of floats
        angle : float

        Returns
        -------
        length N ndarray of theta values corresponding to target centers
        '''
        loc = np.array(loc)
        # Get a vector toward each target
        vecs = self.locs - loc
        target_angles = np.arctan2(vecs[:,1],vecs[:,0])
        return convert_angles(target_angles - angle)
    

    def get_dist_to_targets(self,loc):
        '''Given the (x,y) coordinate of an observer, loc, return an array of 
        distances to the targets.
        
        Parameters
        ----------
        loc : (x,y) of floats

        Returns
        -------
        length N ndarray of distance values
        '''
        if self.geom_name is None:
            return np.linalg.norm(loc-self.locs, axis=1)
        elif self.geom_name == 'circle':
            return np.linalg.norm(loc-self.locs, axis=1) - self.r
        elif self.geom_name == 'segment':
            seg_vec = 0.5*self.l*np.array([np.cos(self.theta),np.sin(self.theta)]).T
            return self.closest_dist_btwn_lines_and_pt(self.locs-seg_vec,
                                                       self.locs+seg_vec, loc)


    def check_target_overlap(self,loc):
        '''Check to see if loc overlaps with any of the targets. Return a bool
        array of length N (where N is the number of targets) indicating overlap.
        '''
        eps = np.finfo(np.float32).eps

        if self.geom_name is None:
            return (self.locs == loc).all(axis=1)
        elif self.geom_name == 'circle':
            return np.linalg.norm(self.locs - loc,axis=1) <= self.r
        elif self.geom_name == 'segment':
            # Make sure locs is 2D
            if self.locs.ndim == 1:
                self.locs = np.array([self.locs])
            diff = np.column_stack([self.l/2*np.cos(self.theta),
                                    self.l/2*np.sin(self.theta)])
            pt1 = self.locs + diff
            pt2 = self.locs - diff
            cross = (loc[1]-pt1[:,1])*(pt2[:,0]-pt1[:,0])-\
                    (loc[0]-pt1[:,0])*(pt2[:,1]-pt1[:,1])
            dot = (loc[0]-pt1[:,0])*(pt2[:,0]-pt1[:,0])+\
                  (loc[1]-pt1[:,1])*(pt2[:,1]-pt1[:,1])
            sqdist = (pt2[:,0]-pt1[:,0])**2 + (pt2[:,1]-pt1[:,1])**2
            return np.abs(cross)<eps & dot>=0 & dot<=sqdist
    

    @staticmethod
    def closest_dist_btwn_lines_and_pt(Q0_list, Q1_list, pt):
        '''
        Given line segments that begin at Q0 and end at Q1, and a point in space, 
        return the minimum distance between each of the line segments and the point.

        Parameters
        ----------
        Q0_list : Nx2 or Nx3 ndarray 
            start points of line segments
        Q1_list : Nx2 or Nx3 ndarray
            end points of line segments
        pt : 1D ndarray
            point in 2D or 3D space
        '''

        seg_lengths_2 = np.linalg.norm(Q1_list - Q0_list, axis=1)**2

        dist_list = np.empty(Q0_list.shape[0])

        # Wherever the segment lengths are close to zero, calculate pt distance
        z_check = seg_lengths_2 < np.finfo(float).eps * 100
        dist_list[z_check] = np.linalg.norm(Q0_list[z_check] - pt, axis=1)

        Q0 = Q0_list[~z_check]
        Q1 = Q1_list[~z_check]
        seg_lengths_2[~z_check] = seg_lengths_2

        # For the rest, follow the same math as in closest_dist_btwn_line_and_pts
        # First, find the projection of the point onto the line and clamp to segments
        dot = ((pt-Q0)*(Q1-Q0)).sum(1)/seg_lengths_2 # dot prod of each row vec
        t_list = np.maximum(0,np.minimum(1,dot))
        # Find the point on the segments
        proj_pt_list = Q0 + np.tile(t_list,(pt.shape[0],1)).T*(Q1-Q0)
        dist_list[~z_check] = np.linalg.norm(pt-proj_pt_list,axis=1)

        return dist_list


    def plot_targets_to_axis(self, ax):
        '''Plots the targets on a given axis object.
        '''

        if self.geom_name is None:
            # delta functions
            ax.plot(self.locs[:,0],self.locs[:,1],'.')
        elif self.geom_name == 'circle':
            for n,pos in enumerate(self.locs):
                try:
                    circle = plt.Circle(pos, self.r[n], color='b')
                except TypeError:
                    circle = plt.Circle(pos, self.r, color='b')
                ax.add_patch(circle)
        elif self.geom_name == 'segment':
            # plot segment targets
            for n,pos in enumerate(self.locs):
                try:
                    l = self.l[n]
                except TypeError:
                    l = self.l
                try:
                    theta = self.theta[n]
                except TypeError:
                    theta = self.theta
                x = (pos[0] - l/2*np.cos(theta), pos[0] + l/2*np.cos(theta))
                y = (pos[1] - l/2*np.sin(theta), pos[1] + l/2*np.sin(theta))
                ax.plot(x,y,'b')
        else:
            raise NotImplementedError("This geometry still TBD in Targets.")



class PerceptionModel:
    '''This class takes in a Targets object and a focal location and angle for an
    observer, and then translates that into a neural angular position and a neural 
    spin group size for each target based on the perceived angular extents of the 
    target, its signal strength, and a weighting function that describes the 
    density of neurons in the ring as a function of angle.'''

    def __init__(self, targets=None, focal_loc=(5,10), focal_angle=0, 
                 neural_weight='cutoff', theta_mesh=2000):
        '''Establishes an observer at location focal_loc, looking in a direction 
        given by focal_angle, at targets given by the targets object. All three 
        of these can be changed at any time as attributes.

        Parameters
        ----------
        targets : Targets object, optional.
            the targets around the observer as a Targets object. If no targets 
            object is given, a default target object will be set.
        focal_loc : array-like of length 2
            (x,y) location of observer in Euclidean space. Will be stored as an 
            ndarray.
        focal_angle : float
            direction observer is facing in Euclidean space from [-pi,pi).
        neural_weight : string (default = 'cutoff')
            Neural weighting function. Can be 'cutoff', 'tanh_plus', or None.
        theta_mesh : float or 1D ndarray
            the number of equally spaced mesh points on [-pi,pi) to evaluate at 
            or a mesh of theta values to evaluate at
        '''

        self.focal_loc = np.array(focal_loc, dtype=float)
        self.focal_angle = focal_angle
        self.neural_weight = neural_weight
        if neural_weight == 'tanh_plus':
            self.c = 2
            self.d = 2*np.pi/3
        elif neural_weight == 'cutoff':
            # = 1 when |theta|<self.c, = 0 when |theta|>self.d, smooth in between
            self.c = np.pi/2 
            self.d = 4*np.pi/5
        else:
            self.c = None
            self.d = None
        if targets is None:
            self.targets = Targets()
        else:
            assert isinstance(targets,Targets), "targets must be a Targets object."
            self.targets = targets
        if isinstance(theta_mesh, int):
            self.theta_mesh = np.linspace(-np.pi, np.pi, theta_mesh+1)[:-1]
        else:
            self.theta_mesh = theta_mesh


    @staticmethod
    def _bump(t):
        result = np.zeros_like(t)
        result[t>0] = np.exp(-1/t[t>0])
        return result
    
    @staticmethod
    def _smoothstep(x):
        return PerceptionModel._bump(x)/(PerceptionModel._bump(x)+PerceptionModel._bump(1-x))
    
    @staticmethod
    def _cutoff(x, left_off, left_on, right_on, right_off):
        return PerceptionModel._smoothstep(
            (x - left_off)/(left_on - left_off) ) * PerceptionModel._smoothstep(
            (right_off - x)/(right_off - right_on) )
    
    @staticmethod
    def _tanh_plus(theta, c, d):
        return (np.tanh(c*(1-(theta/d)**2) ) + 1.0001)/(1.0001+np.tanh(c))

    def neural_weight(self, theta):
        '''Returns the neural weight for a given angle theta based on the 
        weighting function. This is a proxy for the density of neurons in the 
        ring as a function of angle, and weights things in front more highly than 
        in back. Uses a standard cuttoff function or tanh_plus or returns ones.

        Parameters
        ----------
        theta : float or 1D ndarray
            angle(s) to evaluate the neural weight at

        Returns
        -------
        neural weight(s) corresponding to input theta value(s)
        '''

        if self.neural_weight is None:
            return np.ones_like(theta)
        elif self.neural_weight == 'cutoff':
            return self._cutoff(theta, -self.d, -self.c, self.c, self.d)
        elif self.neural_weight == 'tanh_plus':
            return self._tanh_plus(theta, self.c, self.d)
        else:
            raise NotImplementedError("Unknown neural weight function name.")



    def get_target_signals(self, focal_angle=None, focal_loc=None, 
                           norm=np.pi/8, full_signal=False):
        '''Returns the egocentric angular location of the center of each VISIBLE 
        target (closer targets that are not delta functions block ones behind) as 
        a length N array, and a visual signal for each that is either a scalar 
        for each target or a binary array for each target with support on the 
        visible angular extents of the target.
        
        The scalar visual signal is computed by integrating the array visual 
        signal against a kernel (neural_weight), divided by norm.
        
        Uses a mesh of theta values to determine blocking, resulting in an 
        approximation of extents. This adds noise, but maybe the right kind of 
        noise (i.e., if less than 2pi/len(self.theta_mesh) is visible to the right 
        or left of a blocking locust, then the blocked locust is treated as not 
        visible). Larger meshes result in a finer mesh and a better approximation 
        of exact blocking.

        Parameters
        ----------
        focal_angle : float, optional
            the focal angle for egocentric perception. If None, uses the object's 
            focal_angle attribute.
        focal_loc : array-like, optional
            the (x,y) focal location for egocentric perception. If None, uses the 
            object's focal_loc attribute.
        norm : float, default=np.pi/8
            the normalization factor for the scalar visual signal. Default is 
            chosen as some sort of approximation for how much visual space 
            might be occupied by a target at reasonable decision-making distances.
        full_signal : bool
            if True, return the full signal for each target as a theta mesh, 
            otherwise return only the value of the signal for each target

        Returns
        -------
        angles : length N ndarray
            angles to the centers of visible targets
        signals : length N or Nxlen(theta_mesh) ndarray
            perception signals for each visible target, with amplitude equal 
            to 1/distance to target
        '''

        if focal_angle is None:
            focal_angle = self.focal_angle
        if focal_loc is None:
            focal_loc = self.focal_loc
        else:
            focal_loc = np.array(focal_loc, dtype=float)

        dists = self.targets.get_dist_to_targets(focal_loc)
        c_angles = self.targets.get_angles_to_targets(focal_loc, focal_angle)
        angles = self.targets.get_percep_angles(focal_loc, focal_angle)
        signals = np.zeros((angles.shape[0], self.theta_mesh.size))

        if self.targets.geom_name is None:
            # TODO: come back to this
            for n, theta in enumerate(angles):
                idx = np.searchsorted(self.theta_mesh,theta)
                if idx == len(self.theta_mesh):
                    idx = 0
                # step function perception
                if idx != 0 and \
                theta-self.theta_mesh[idx-1] < self.theta_mesh[idx]-theta:
                    signals[n,idx-1] = 1
                elif idx == 0 and \
                theta-self.theta_mesh[-1] < -self.theta_mesh[0]-theta:
                    signals[n,-1] = 1
                else:
                    signals[n,idx] = 1

        elif self.targets.geom_name == 'circle' or self.targets.geom_name == 'segment':
            # sort by distance
            # TODO: Andy points out that you can have two line segments where 
            # the one with the farther center occludes the one with the closer center.
            arg_srt = dists.argsort()
            angles = angles[arg_srt]
            c_angles = c_angles[arg_srt]
            # determine blocking by creating binary signals for each angle extent
            for n, thetas in enumerate(angles):
                if thetas[1] > thetas[0]:
                    theta_bool = np.logical_and(thetas[0]<=self.theta_mesh,
                                                self.theta_mesh<=thetas[1])
                else:
                    theta_bool = np.logical_or(thetas[0]<=self.theta_mesh,
                                               self.theta_mesh<=thetas[1])
                signals[n,theta_bool] = 1
            # determine blocking based on sorted order
            # closest targets are earlier in the signals array
            blocked = signals[0,:] != 0
            for n in range(1,signals.shape[0]):
                signals[n,blocked] = 0
                blocked = np.logical_or(blocked, signals[n,:] != 0)
            # undo sorting to main target consistency across methods
            inv_arg_srt = np.empty_like(arg_srt)
            inv_arg_srt[arg_srt] = np.arange(len(arg_srt))
            signals = signals[inv_arg_srt,:]
            c_angles = c_angles[inv_arg_srt]
            # remove all completely blocked targets
            vis = signals.max(axis=1) > 0
            signals = signals[vis,:]
            c_angles = c_angles[vis]
        else:
            raise NotImplementedError("Unknown target geometry name.")
        
        # Apply visual weighting, normalize signals, and return
        domain_length = 2*np.pi
        weighted_signals = signals*self.neural_weight(self.theta_mesh)
        
        # calculate signals
        signals_final = domain_length/self.theta_mesh.size*weighted_signals.sum(axis=1)/norm

        if full_signal:
            return c_angles, weighted_signals
        else:
            return c_angles, signals_final



    def plot_blocked_signals(self, wb_plot=False):
        '''Plots visible targets and their angular direction from the observer, 
        and also the signal distribution from the point of view of the observer.

        Use as a test for get_target_signals.
        
        Set wb_plot to True if plotting in a Jupyber notebook
        '''

        angles, signals = self.get_target_signals(full_signal=True)

        if wb_plot:
            plt.figure(figsize=(6.5,3.25))
        else:
            plt.figure(figsize=(12,6))

        ###### Target Geometry Plot ######
        ax1 = plt.subplot(121)

        # First, plot the targets themselves
        self.targets.plot_targets_to_axis(ax1)

        # Now plot perception lines. Requires adding back angle of focal locust
        #   to get allocentric angles.
        for n, theta in enumerate(angles+self.focal_angle):
            r = self.targets.get_dist_to_targets(self.focal_loc)[n]
            x = (self.focal_loc[0],self.focal_loc[0] + r*np.cos(theta))
            y = (self.focal_loc[1],self.focal_loc[1] + r*np.sin(theta))
            ax1.plot(x,y,'k')
        ax1.arrow(self.focal_loc[0],self.focal_loc[1],
                  0.5*np.cos(self.focal_angle),0.5*np.sin(self.focal_angle), 
                width=0.1, head_length=0.25)
        ax1.set_aspect('equal')
        ax1.set_title('Target Geometry')

        ###### Perception Signal Plot ######
        ax2 = plt.subplot(122, projection='polar')

        p_func = signals.sum(axis=0)

        ax2.plot(self.theta_mesh,p_func)
        ax2.arrow(0,-0.5,0,0.25, width=0.2, head_length=0.15)
        ax2.set_rmin(-0.5)
        ax2.set_rlabel_position(0)
        ax2.set_title('Perception Signal')

        plt.show()



class IsingExtModel:
    '''
    This only extends Ising slightly. The primary novelty is the underlying 
    perception model that allows for non-delta function targets that can 
    occlude each other, and the addition of a weighting function for the Ising
    interactions based on distance to target.

    The underlying assumpion is that there is a two-step process:
    1) Perception of targets based on location (angle and dist), and geometry
    2) Decision making based on discrete perceived targets via Ising model

    One could try to combine these into a single-step, continuous Ising model, with 
    the distance weighting doing the work of distinguishing between one close locust
    and two adjacent locusts that are farther away (angular size alone is 
    insufficient for this task). However, it would mean that a partially occluded 
    locust would be seen as being located at its visible extent rather than its 
    center. Hard to know if this is better or worse.
    '''

    def __init__(self, percep_model=None, T=0.2, K=1, nu=1):
        '''From a PerceptionModel with its Targets object, establishes a model 
        for chosing direction based on discrete Ising. Relies on get_target_signals 
        from the PerceptionModel to obtain perceived target angles and signal strength.
        
        Parameters
        ----------
        percep_model : PerceptionModel
            A PerceptionModel object (with its Targets object) that establishes 
            the geometry of the scenario. If none is provided, a default one 
            will be created. The PerceptionModel can be updated to obtain 
            consensus directions for different layouts or focal locations/angles.
        T : float
            Temperature for Ising model
        K : float
            Coupling strength for Kuramoto turning speed.
        nu : float
            Exponent for cosine weighting kernel. Higher values lead to sharper peaks.
        '''

        self.T = T
        self.K = K
        self.nu = nu
        self.gamma = None  # last coherence value found. used by dgamma_dt if none provided

        if percep_model is None:
            self.percep_model = PerceptionModel()
        else:
            assert isinstance(percep_model,PerceptionModel),\
            "percep_model must be a PerceptionModel object."
            self.percep_model = percep_model

        # Random number generator for certain processes within the class;
        #   Can seed here for reproducability.
        #seed = 3
        self.rng = np.random.default_rng()


    def cosine(self, x):
        '''Function that returns cos(pi*(x/pi)^nu).'''

        return np.cos(np.pi*(x/np.pi)**self.nu)


    def plot_cosine(self, wb_plot=False):
        '''Plot the cosine weighting function over [-pi,pi].
        
        Set wb_plot to True if plotting in a Jupyter notebook
        '''

        xmesh = np.linspace(-np.pi, np.pi, 1000)
        ymesh = self.cosine(xmesh)

        if wb_plot:
            plt.figure(figsize=(6.5,3.25))
        else:
            plt.figure(figsize=(8,4))
        plt.plot(xmesh, ymesh)
        plt.title('Cosine Weighting Function, $\\nu={}$'.format(self.nu))
        plt.xlabel('Angle (radians)')
        plt.ylabel('Weighting')
        plt.ylim(-1.1,1.1)
        plt.grid()
        plt.show()


    def dgamma_dt(self, t=None, gamma=None, focal_theta=None, focal_loc=None):
        '''Get the complex time derivative of gamma according to the 
        Ising model. For use directly in ODE solvers.

        gamma is the coherence value of the neural band, which is a complex 
        number whose angle represents the current consensus direction and whose 
        magnitude (a value between 0 and 1) represents the coherence strength.
        

        Parameters
        ----------
        t : float, optional
            time variable for ODE solver compatibility. Not used.
        gamma : complex, optional
            current coherence value of the neural band. 
            If None, use self.percep_model.focal_angle with a coherence 
            strength of 0.1.
        focal_theta : float, optional
            the current heading (angle) of the observer. If None, use 
            self.percep_model.focal_angle.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use the 
            self.percep_model.focal_loc.

        Returns
        -------
        dgamma_dt : complex
            time derivative of gamma according to the Ising model
        '''

        if gamma is None:
            Theta = self.percep_model.focal_angle
            R = 0.1
            gamma = R*np.exp(1j*Theta)
        else:
            Theta = np.angle(gamma)
            R = np.abs(gamma)
        if focal_theta is None:
            focal_theta = self.percep_model.focal_angle

        angles_rel, signals = self.percep_model.get_target_signals(focal_theta, focal_loc)
        if angles_rel.size == 0:
            return -gamma
        # The angles recieved above are relative to focal_theta, i.e., the 
        #   angle between the polar location of each target and focal_theta.
        # Convert to allocentric polar angles.
        angles = convert_angles(angles_rel+focal_theta)
        # Convert to angles relative to current consensus direction (gamma angle).
        angles_rel = convert_angles(angles-Theta)
        
        # Compute the sum over all target locusts.
        # suppress overflow warnings: they indicate R is too large and are 
        #   likely the result of whatever numerical algorithm is being used
        #   trying out a bad R value.
        with np.errstate(over='ignore'):
            summands = signals*np.exp(1j*angles)/(1+np.exp(
                -2*angles.size*R*self.cosine(angles_rel)/self.T))
        
        return np.sum(summands)/signals.sum() - gamma
    

    def dgamma_dt_vec(self, gamma_vec, focal_theta=None, focal_loc=None):
        '''Wrapper around dgamma_dt for use in root finding for equilibria.
        
        Here, gamma_vec is a length 2 ndarray of real and imaginary parts 
        representing the complex coherence value of the neural band. The 
        magnitude of the coherence value is the coherence strength and the 
        argument is the current consensus direction. This function describes 
        how the coherence value should change based on its current value 
        plus the neural band reacting to the surrounding target geometry.

        Parameters
        ----------
        gamma_vec : length 2 ndarray of float
            current (complex) coherence value of the neural band.
        focal_theta : float or bool, optional
            the current heading (angle) of the observer. If None, use 
            self.percep_model.focal_angle. If True, assume theta is the angle of 
            the current gamma value. Use for exploring the geometry of equilibria 
            as a function of gamma angle rather than observer angle.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use self.percep_model.focal_loc.

        Returns
        -------
        dgamma_dt_vec : length 2 ndarray of float
            complex time derivative of gamma according to the Ising model
        '''
        if focal_theta is False:
            focal_theta = None
        elif focal_theta is True:
            focal_theta = np.angle(gamma_vec[0] + 1j*gamma_vec[1])

        dgamma = self.dgamma_dt(None, gamma_vec[0] + 1j*gamma_vec[1], 
                                focal_theta, focal_loc)
        return np.array([dgamma.real, dgamma.imag])
    

    def run_dgamma_dt(self, focal_theta=None, focal_loc=None, init_gamma=None, 
                      t_Final=30):
        '''Integrate the dgamma/dt equation in order to approach a stable 
        equilibrium for the neural ring model. Uses RK45 solver from scipy. 
        
        init_gamma is the initial coherence value to start from. If None, 
        it will see if this object has a current gamma value and use that, 
        otherwise it will use a 0.1 magnitude vector in the direction of the 
        percep_model's focal_angle.

        Parameters
        ----------
        focal_theta : float, optional
            the current heading (angle) of the observer. If None, use 
            self.percep_model.focal_angle.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use the percep_model's 
            focal_loc.
        init_gamma : complex float, optional
            initial coherence value. If None, use the model's
            current gamma value if set, otherwise use a vector of magnitude 
            0.1 based on focal_theta.
        t_Final : float, optional
            final time for integration.

        Returns
        -------
        gamma_equilib : complex float
            Equilibrium gamma value reached by integration.
        '''

        if init_gamma is None:
            if hasattr(self, 'gamma') and self.gamma is not None:
                init_gamma = self.gamma
            else:
                if focal_theta is None:
                    focal_theta = self.percep_model.focal_angle
                init_gamma = 0.1*np.exp(1j*focal_theta)
        if np.isscalar(init_gamma):
            init_gamma = np.array([init_gamma])

        # to stop solving for gamma when sufficiently close to equilibrium we:
        # 1) initially solve for 5 unit of time
        #    (Chose 5 by trial and error, seems to balance overshooting vs. looping, at which python is slow)
        # 2) check whether the derivative is bigger than tolerance tol
        # 3) repeat for 1 unit of time, checking the size of the derivative each time
        sol = solve_ivp(self.dgamma_dt, [0, 5], init_gamma, 
                        args=(focal_theta, focal_loc))
        tol = 1e-4
        T = 1
        while np.abs(sol.y[0,-1]-sol.y[0,-2])/(sol.t[-1]-sol.t[-2]) > tol:
            sol = solve_ivp(self.dgamma_dt, [0, 1], sol.y[:,-1], 
                        args=(focal_theta, focal_loc))
            T += 1
            if T > t_Final:
                break
        #print(T) # in testing this was 10 for the first, then 5\pm 1 (repeated)
        if np.abs(sol.y[0,-1]-sol.y[0,-2])/(sol.t[-1]-sol.t[-2]) > tol:
            print("Warning: Integration may not have reached equilibrium.")
        return sol.y[0,-1]
    

    def dtheta_dt(self, t=None, theta=None, gamma=None, focal_loc=None,
                  t_Final=30):
        '''Torque model based on the Ising model coherence value.
        
        Parameters
        ----------
        t : float, optional
            time variable for ODE solver compatibility. Not used.
        theta : float, optional
            current angle of the focal locust in allocentric coordinates. 
            If None, use self.percep_model.focal_angle.
        gamma : complex float or bool, optional
            Equilibrium coherence value from neural ring. If None, it will be 
            computed by solving dgamma_dt to t_Final (using run_dgamma_dt) and 
            taking the final solution as an equilibrium value. The IC for 
            run_dgamma_dt will be either self.gamma (the last gamma value found 
            in this function) if it exists, or a 0.1 magnitude vector based on 
            theta if not. self.gamma will be updated to the final gamma value 
            found. If False, do not use self.gamma as the IC and do not update 
            self.gamma with the final value found. This is useful for basing 
            the result on a specific theta value rather than the current state 
            of the neural ring.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use the percep_model's 
            focal_loc.
        t_Final : float, optional (default=30)
            final time for integration of dgamma_dt if gamma is None or False.'''
        if gamma is True:
            gamma = None
        if theta is None:
            theta = self.percep_model.focal_angle
        if gamma is None:
            self.gamma = self.run_dgamma_dt(focal_theta=theta, focal_loc=focal_loc, 
                                            init_gamma=gamma, t_Final=t_Final)
            gamma = self.gamma
        elif gamma is False:
            gamma = self.run_dgamma_dt(focal_theta=theta, focal_loc=focal_loc, 
                                       init_gamma=0.1*np.exp(1j*theta), 
                                       t_Final=t_Final)
        return self.K*np.abs(gamma)*np.sin(np.angle(gamma)-theta)
    

    def plot_dtheta_dt(self, gamma=None, focal_loc=None, wb_plot=False):
        '''Plot dtheta/dt as a function of theta for given focal location.
        
        Set wb_plot to True if plotting in a Jupyter notebook

        Parameters
        ----------
        gamma : complex float or bool, optional
            Equilibrium coherence value from neural ring. If None, it will be 
            computed by solving dgamma_dt to t_Final (using run_dgamma_dt) and 
            taking the final solution as an equilibrium value. The IC for 
            run_dgamma_dt will be either self.gamma (the last gamma value found 
            in this function) if it exists, or a 0.1 magnitude vector based on 
            theta if not. self.gamma will be updated to the final gamma value 
            found. If False, do not use self.gamma as the IC and do not update 
            self.gamma with the final value found. This is useful for basing 
            the result on a specific theta value rather than the current state 
            of the neural ring.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use the percep_model's 
            focal_loc.
        '''

        thetas = np.linspace(-np.pi, np.pi, 1000)
        dthetas = np.array([self.dtheta_dt(theta=theta, gamma=gamma, focal_loc=focal_loc) 
                            for theta in thetas])

        if wb_plot:
            plt.figure(figsize=(6.5,4.5))
        else:
            plt.figure(figsize=(8,4))
        plt.plot(thetas, dthetas)
        plt.axhline(0, color='k', linestyle='--')
        plt.title('$d\\theta/dt$ vs $\\theta$')
        plt.xlabel('$\\theta$ (locust heading, radians)')
        plt.ylabel('$d\\theta/dt$')
        # plt.ylim(-1.1,1.1)
        plt.grid()
        plt.show()


    def get_direction(self, dt):
        '''Integrate the Ising torque model for one time step dt to get new direction.
        
        Parameters
        ----------
        dt : float
            time step for integration of torque model.
        '''
        
        return convert_angles(solve_ivp(self.dtheta_dt, [0, dt], 
                              [self.percep_model.focal_angle]).y[0,-1])
    

    def gamma_equilib(self, focal_theta=None, focal_loc=None):
        '''Find zeros (equilibria) of dgamma/dt for a given focal location. 
        
        Uses a multistart root finding starting from a mesh of points on the 
        circle of radius 0.2. Returns a list of unique equilibrium gamma values 
        found.

        Parameters
        ----------
        focal_theta : float or bool, optional
            the current heading (angle) of the observer. If None, use 
            self.percep_model.focal_angle. If True, assume theta is the angle of 
            the current gamma value. Use for exploring the geometry of equilibria 
            as a function of gamma angle rather than observer angle.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use self.percep_model.focal_loc.

        Returns
        -------
        gamma_eqs : list of complex
            list of equilibrium gamma values
        '''

        init_angles = np.linspace(-np.pi, np.pi-0.01)
        init_vals = np.zeros((init_angles.size, 2), dtype=np.double)
        init_vals[:,0] = 0.2*np.cos(init_angles)
        init_vals[:,1] = 0.2*np.sin(init_angles)
        final_gammas = []
        for init_val in init_vals:
            sol = root(self.dgamma_dt_vec, init_val, args=(focal_theta, focal_loc),
                       method='hybr', tol=1e-7)
            # Only store unique solutions
            if sol.success:
                gamma_eq = sol.x[0] + 1j*sol.x[1]
                # Check if close to any existing solution
                close_check = False
                for existing_gamma in final_gammas:
                    if np.abs(gamma_eq - existing_gamma) < 0.1:
                        close_check = True
                        break
                if not close_check:
                    final_gammas.append(gamma_eq)
        return final_gammas
    

    def gamma_equilib2(self, focal_theta=None, focal_loc=None):
        '''Find zeros (equilibria) of dgamma/dt for a given focal location with 
        new algorithm.'''

        if focal_theta is None:
            focal_theta = self.percep_model.focal_angle

        angles_rel, signals = self.percep_model.get_target_signals(focal_theta, focal_loc)
        if angles_rel.size == 0:
            return None
        # The angles recieved above are relative to focal_theta, i.e., the 
        #   angle between the polar location of each target and focal_theta.
        # Convert to allocentric polar angles.
        angles = convert_angles(angles_rel+focal_theta)

        # Create p function
        def p_func(R, Theta, angles):
            angles_rel = convert_angles(angles-Theta)
            with np.errstate(over='ignore'):
                return R - np.sum(signals/signals.sum()*np.cos(angles_rel)/
                    (1+np.exp(-2*angles.size*R*self.cosine(angles_rel)/self.T)))
            
        def Theta_out(R, Theta, angles):
            angles_rel = convert_angles(angles-Theta)
            with np.errstate(over='ignore'):
                return convert_angles(np.angle(np.sum(signals/signals.sum()*np.exp(1j*angles)/
                    (1+np.exp(-2*angles.size*R*self.cosine(angles_rel)/self.T)))))
        
        # Build Theta grid
        Theta_mesh = np.linspace(-np.pi, np.pi-0.01, 200)
        Theta_vals = []
        Theta_sols = []
        R_sols = []
        # We want to break the mesh into segments where p_func changes sign
        new_list_start = True # flag to indicate when to start a new list of solutions
        for Theta in Theta_mesh:
            if p_func(0, Theta, angles)*p_func(1, Theta, angles) > 0:
                new_list_start = True
                continue
            if new_list_start:
                Theta_vals.append([])
                Theta_sols.append([])
                R_sols.append([])
                new_list_start = False
            R_sol = brentq(p_func, 0, 1, args=(Theta, angles), xtol=1e-7)
            Theta_sol = Theta_out(R_sol, Theta, angles)
            Theta_vals[-1].append(Theta)
            Theta_sols[-1].append(Theta_sol)
            R_sols[-1].append(R_sol)
        Theta_sols = [np.array(item) for item in Theta_sols]
        R_sols = [np.array(item) for item in R_sols]
        Theta_vals = [np.array(item) for item in Theta_vals]

        # Find places where the difference between Theta_sols and Theta_vals 
        #   changes sign, indicating a crossing (equilibrium).
        Theta_equilibs = []
        R_equilibs = []
        angle_diffs = []
        for i in range(len(Theta_sols)):
            angle_diff = convert_angles(Theta_sols[i] - Theta_vals[i])
            idx = np.where(np.diff(np.sign(angle_diff)))[0]
            if len(idx) > 0:
                # Interpolate the crossings to get better estimates of equilibrium locations.
                Theta_equilibs.extend(Theta_vals[i][idx] - (Theta_sols[i][idx]-Theta_vals[i][idx]) * (
                    Theta_vals[i][idx+1]-Theta_vals[i][idx]) / (Theta_sols[i][idx+1]-Theta_sols[i][idx] - 
                    (Theta_sols[i][idx]-Theta_vals[i][idx])))
                R_equilibs.extend(R_sols[i][idx] - (Theta_sols[i][idx]-Theta_vals[i][idx]) * (
                    R_sols[i][idx+1]-R_sols[i][idx]) / (Theta_sols[i][idx+1]-Theta_sols[i][idx] - 
                    (Theta_sols[i][idx]-Theta_vals[i][idx])))
            angle_diffs.extend(angle_diff)
        angle_diffs = np.array(angle_diffs)
        R_equilibs = np.array(R_equilibs)
        Theta_equilibs = convert_angles(np.array(Theta_equilibs))
        # flatten R_sols and Theta_vals
        R_sols = np.concatenate(R_sols)
        Theta_vals = np.concatenate(Theta_vals)

        # Now return as complex numbers
        return R_equilibs * np.exp(1j*Theta_equilibs), Theta_vals, angle_diffs, R_sols


    def _process_point(self, args):
        '''Helper function for processing mesh points in plot_direction_mesh.
        
        Parameters
        ----------
        args : tuple
            (ii, jj, X, Y) where ii and jj are the mesh indices and X and Y 
            are the mesh coordinate arrays.

        Returns
        -------
        ii : int
            mesh x index
        jj : int
            mesh y index
        final_gammas : list of complex
            list of stable equilibrium angles found at this mesh point
        '''

        ii, jj, X, Y = args
        focal_loc = np.array([X[jj,ii], Y[jj,ii]])

        final_gammas = self.gamma_equilib(focal_theta=True, focal_loc=focal_loc)

        return ii, jj, final_gammas
    
    
    def _process_point2(self, args):
        '''Helper function for processing mesh points in plot_direction_mesh.
        
        Parameters
        ----------
        args : tuple
            (ii, jj, X, Y) where ii and jj are the mesh indices and X and Y 
            are the mesh coordinate arrays.

        Returns
        -------
        ii : int
            mesh x index
        jj : int
            mesh y index
        final_gammas : list of complex
            list of stable equilibrium angles found at this mesh point
        '''

        ii, jj, X, Y = args
        focal_loc = np.array([X[jj,ii], Y[jj,ii]])

        final_gammas, _, _, _ = self.gamma_equilib2(focal_theta=True, focal_loc=focal_loc)

        return ii, jj, final_gammas
    

    def _discrim_A(self, gamma_star, focal_loc):
        '''Determines stability of equilibria based on perturbation analysis.
        Assumes that the focal_theta is the angle of gamma_star and calculates 
        the A value of the linear coefficient. If A < 1, the equilibrium is stable, 
        if A > 1, the equilibrium is unstable.

        This is based on a cosine kernel with nu exponent.

        Parameters
        ----------
        gamma_star : complex
            equilibrium gamma value at which to compute the Jacobian
        focal_loc : array-like of length 2
            (x,y) location of the observer

        Returns
        -------
        True if stable, False if unstable
        '''

        Theta = np.angle(gamma_star)
        R = np.abs(gamma_star)
        angles_rel, signals = self.percep_model.get_target_signals(Theta, focal_loc)
        k = signals.size
        # angles_rel = theta_j - Theta
        with np.errstate(over='ignore'):
            summands = ((signals/np.cosh(k*R*self.cosine(angles_rel)/self.T)**2)
                        *np.sin(angles_rel)*self.nu*
                        np.sin(np.pi*(angles_rel/np.pi)**self.nu)*
                        (angles_rel/np.pi)**(self.nu-1))
            A = k*summands.sum()/(2*self.T*signals.sum())
        return A < 1
    

    def plot_direction_mesh(self, xlim=(0,6), num_x=19, ylim=(-3.5,3.5), num_y=19, 
                            wb_plot=False, pool=None):
        '''Create a mesh of starting locations and, for each point in the mesh, 
        find the (eventually) stable equilibria of the direction model dgamma/dt 
        and plot the consensus directions.

        For now, just plot all the equilibria found at each mesh point.

        Set wb_plot to True if plotting in a Jupyter notebook

        Parameters
        ----------
        xlim : (xmin,xmax) tuple of floats
            x limits for mesh, inclusive
        num_x : number of steps in x direction
        ylim : (ymin,ymax) tuple of floats
            y limits for mesh, inclusive
        num_y : number of steps in y direction
        wb_plot : bool
            whether or not plotting in a Jupyter notebook
        pool : multiprocessing.Pool, optional
            If provided, use this pool to parallelize the solving of the ODEs.

        Returns
        -------
        multi_thetas : list of length N arrays
            Each entry in the list corresponds to one of the multiple solutions
            found. Each array is of shape (num_y,num_x) and contains the
            stable equilibrium angle at each mesh point for that solution.
        X : (num_y,num_x) ndarray
            x coordinates of the mesh
        Y : (num_y,num_x) ndarray
            y coordinates of the mesh
        U_list : list of (num_y,num_x) ndarrays
            Each entry in the list corresponds to one of the multiple solutions
            found. Each array is of shape (num_y,num_x) and contains the
            x-component of the unit vector at each mesh point for that solution.
        V_list : list of (num_y,num_x) ndarrays
            Each entry in the list corresponds to one of the multiple solutions
            found. Each array is of shape (num_y,num_x) and contains the
            y-component of the unit vector at each mesh point for that solution.
        '''

        # create mesh of focal locations
        xmesh = np.linspace(xlim[0], xlim[1], num_x)
        ymesh = np.linspace(ylim[0], ylim[1], num_y)

        X, Y = np.meshgrid(xmesh, ymesh)
        U_list = []
        V_list = []
        stability_list = []
        # boolean mesh for multiple solutions
        multi_sol = np.full(X.shape, False, dtype=bool)

        # create mesh of initial angles, perturbed slightly to avoid
        #   exact angles. This is to try and avoid hitting an unstable
        #   equilibrium exactly.
        multi_thetas = []
        
        if pool is None:
            results = []
            for ii in range(num_x):
                for jj in range(num_y):
                    print("Processing point ({},{})".format(ii,jj))
                    results.append(self._process_point((ii,jj,X,Y)))
        else:
            args_list = [(ii, jj, X, Y) for ii in range(num_x) for jj in range(num_y)]
            print("Processing points in parallel using pool...")
            results = pool.map(self._process_point, args_list)
            results2 = pool.map(self._process_point2, args_list)

        # plot the vector field
        if wb_plot:
            fig = plt.figure(figsize=(6.5,4))
        else:
            fig = plt.figure(figsize=(5.5,5))

        for fignum, result_ver in enumerate([results, results2]):
            for result in result_ver:
                ii, jj, final_gammas = result
                for n, gamma in enumerate(final_gammas):
                    theta = convert_angles(np.angle(gamma))
                    if len(multi_thetas) < n+1:
                        multi_thetas.append(np.zeros(X.shape))
                        U_list.append(np.zeros(X.shape))
                        V_list.append(np.zeros(X.shape))
                        stability_list.append(np.full(X.shape, False, dtype=bool))
                    multi_thetas[n][jj,ii] = theta
                    U_list[n][jj,ii] = np.cos(theta)
                    V_list[n][jj,ii] = np.sin(theta)
                    focal_loc = np.array([X[jj,ii], Y[jj,ii]])
                    stability_list[n][jj,ii] = self._discrim_A(gamma, focal_loc)
                if len(final_gammas) > 1:
                    multi_sol[jj,ii] = True

            ax = plt.subplot(1,2,fignum+1)
            # Plot targets
            self.percep_model.targets.plot_targets_to_axis(ax)
            ##### Plot arrows, coloring multi-solution points differently #####
            # Single solution points (blue if stable, cyan if unstable)
            X_single_stable = X[(multi_sol==False) & stability_list[0]]
            Y_single_stable = Y[(multi_sol==False) & stability_list[0]]
            U_single_stable = U_list[0][(multi_sol==False) & stability_list[0]]
            V_single_stable = V_list[0][(multi_sol==False) & stability_list[0]]
            ax.quiver(X_single_stable, Y_single_stable, U_single_stable, V_single_stable,
                    angles='xy', color='blue', scale=30, width=0.004,
                    label='Single Solution (Stable)')
            X_single_unstable = X[(multi_sol==False) & ~stability_list[0]]
            Y_single_unstable = Y[(multi_sol==False) & ~stability_list[0]]
            U_single_unstable = U_list[0][(multi_sol==False) & ~stability_list[0]]
            V_single_unstable = V_list[0][(multi_sol==False) & ~stability_list[0]]
            ax.quiver(X_single_unstable, Y_single_unstable, U_single_unstable, V_single_unstable,
                    angles='xy', color='cyan', scale=30, width=0.004,
                    label='Single Solution (Unstable)')
            # ax.quiver(X[multi_sol==False], Y[multi_sol==False], 
            #             U_list[0][multi_sol==False], V_list[0][multi_sol==False], 
            #             angles='xy', color='blue', scale=30, width=0.004, 
            #             label='Single Solution')
            # Multi solution points (red if stable, black if unstable)
            X_multi_stable = X[multi_sol & stability_list[0]]
            Y_multi_stable = Y[multi_sol & stability_list[0]]
            U_multi_stable = U_list[0][multi_sol & stability_list[0]]
            V_multi_stable = V_list[0][multi_sol & stability_list[0]]
            ax.quiver(X_multi_stable, Y_multi_stable, U_multi_stable, V_multi_stable,
                    angles='xy', color='black', scale=30, width=0.004,
                    label='Multiple Solutions (Stable)')
            X_multi_unstable = X[multi_sol & ~stability_list[0]]
            Y_multi_unstable = Y[multi_sol & ~stability_list[0]]
            U_multi_unstable = U_list[0][multi_sol & ~stability_list[0]]
            V_multi_unstable = V_list[0][multi_sol & ~stability_list[0]]
            ax.quiver(X_multi_unstable, Y_multi_unstable, U_multi_unstable, V_multi_unstable,
                    angles='xy', color='red', scale=30, width=0.004,
                    label='Multiple Solutions (Unstable)')
            # ax.quiver(X[multi_sol], Y[multi_sol], 
            #             U_list[0][multi_sol], V_list[0][multi_sol], 
            #             angles='xy', color='red', scale=30, width=0.004,
            #             label='Multiple Solutions')
            for n in range(1, len(U_list)):
                # Continue with multi-solution points, coloring them red if stable and black if unstable
                nonzero_mask = (U_list[n]!=0) | (V_list[n]!=0)
                X_multi_stable = X[nonzero_mask & stability_list[n]]
                Y_multi_stable = Y[nonzero_mask & stability_list[n]]
                U_multi_stable = U_list[n][nonzero_mask & stability_list[n]]
                V_multi_stable = V_list[n][nonzero_mask & stability_list[n]]
                ax.quiver(X_multi_stable, Y_multi_stable, U_multi_stable, V_multi_stable,
                        angles='xy', color='black', scale=30, width=0.004)
                X_multi_unstable = X[nonzero_mask & ~stability_list[n]]
                Y_multi_unstable = Y[nonzero_mask & ~stability_list[n]]
                U_multi_unstable = U_list[n][nonzero_mask & ~stability_list[n]]
                V_multi_unstable = V_list[n][nonzero_mask & ~stability_list[n]]
                ax.quiver(X_multi_unstable, Y_multi_unstable, U_multi_unstable, V_multi_unstable,
                        angles='xy', color='red', scale=30, width=0.004)
                # ax.quiver(X[nonzero_mask], Y[nonzero_mask], 
                #           U_list[n][nonzero_mask], V_list[n][nonzero_mask], 
                #           angles='xy', color='red', scale=30, width=0.004)
                ax.set_aspect('equal')
        # fig.legend(loc='outside center right')
        plt.title("Direction Model")
        plt.show()
        return multi_thetas, X, Y, U_list, V_list


    def plot_walkers(self, dt=0.1, v=1, std=0, repetitions=20, max_steps=3000,
                     start_loc=None, start_angle=None, plot_tracks=False, 
                     wb_plot=False):
        '''Plot a walker that starts at a specified location looking in a 
        specified angle (defaults to the focal_loc and focal_angle in attached 
        PerceptionModel) and moves according to the Ising torque model on a dt 
        step size with zero-mean angular Gaussian noise with standard deviation 
        as specified. Repeat for a number of repetitions and plot a heat map of 
        these walks in 2D space.

        The walker stops whenever it is detected to be overlapping a target or 
        after max_steps.

        Set wb_plot to True if plotting in a Jupyter notebook

        Parameters
        ----------
        dt : float
            Time step for the walk
        v : float
            Speed of the walker, assumed constant
        std : float
            Standard deviation of angular Gaussian noise with mean zero.
            If zero (default), run without any angular noise.
        repetitions : int
            Number of walks to perform and aggregate
        max_steps : int
            Maximum number of steps for each walker
        start_loc : (x,y) coordinates, optional
            Starting location of the walk, defaults to focal_loc in the attached 
            PerceptionModel
        start_angle : float
            Starting direction that the walker is facing. Defaults to 
            focal_angle in the attached PerceptionModel
        plot_tracks : bool
            Whether or not to overlay the walker trajectories
        wb_plot : bool
            Whether or not plotting in a Jupyter notebook (adjusts size of figure)
        '''

        if start_loc is None:
            start_loc = self.percep_model.focal_loc.copy()
        else:
            start_loc = np.array(start_loc, dtype=float)
        if start_angle is None:
            start_angle = self.percep_model.focal_angle
        orig_loc = self.percep_model.focal_loc.copy()
        orig_angle = self.percep_model.focal_angle

        all_walks = []

        for n in range(repetitions):
            self.percep_model.focal_loc = start_loc.copy()
            self.percep_model.focal_angle = start_angle
            walk = [start_loc.copy()]
            for step in range(max_steps):
                # check for target overlap
                if np.any(self.percep_model.targets.check_target_overlap(
                          self.percep_model.focal_loc)):
                    break
                elif self.percep_model.targets.geom_name is None and \
                np.any(np.linalg.norm(
                       self.percep_model.focal_loc-self.percep_model.targets.locs,
                       axis=1)<v*dt):
                    break
                # determine allocentric direction and take a step
                #   assume turning speed is infinite
                if std > 0:
                    noise = self.rng.normal(scale=std)
                else:
                    noise = 0
                # NOTE:
                # This walk is modeled on a two-step process:
                # 1) Solve ODE over dt time to get new direction, then add noise.
                # 2) move forward in that direction a distance of v*dt.
                # TODO/to try:
                # simplify this to be an Euler (Honeycutt) step instead of solving the theta equation
                #theta = self.get_direction(dt) + noise*dt
                theta = self.percep_model.focal_angle + convert_angles(self.dtheta_dt())*dt + noise*dt
                mv_vec = v*dt*np.array([np.cos(theta),np.sin(theta)])
                self.percep_model.focal_loc += mv_vec
                self.percep_model.focal_angle = convert_angles(theta)
                # append location to walk list
                walk.append(self.percep_model.focal_loc.copy())
            # done. save to all_walks
            all_walks.append(list(walk))

        # Restore focal location and angle
        self.percep_model.focal_loc = orig_loc
        self.percep_model.focal_angle = orig_angle

        # concatenate walks
        walks = sum(all_walks, [])

        # Convert list to 2xN array: row of x-vals then row of y-vals
        walks = np.column_stack(walks)
        # Detect good bin edges
        dim_min = np.floor(walks.min(axis=1))
        dim_max = np.ceil(walks.max(axis=1))
        n_xedges = max(2, round((dim_max[0]-dim_min[0])/0.25))
        n_yedges = max(2, round((dim_max[1]-dim_min[1])/0.25))
        xedges = np.linspace(dim_min[0], dim_max[0], n_xedges)
        yedges = np.linspace(dim_min[1], dim_max[1], n_yedges)

        H, xedges, yedges = np.histogram2d(walks[0,:],walks[1,:], 
                                           bins=(xedges,yedges))
        # Keep original H for interpolation (shape: (len(xcenters), len(ycenters)))
        H_for_spline = H.copy()
        # For plotting with imshow, use transposed version
        H_plot = H_for_spline.T

        if wb_plot:
            fig = plt.figure(figsize=(6.5,4))
        else:
            fig = plt.figure(figsize=(5.5,5))

        # Get bin centers
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        # Interpolate to finer grid for smoother plotting
        # Guard against degenerate bin centers
        if xcenters.size < 2 or ycenters.size < 2:
            # fall back to simple imshow without interpolation
            ax = fig.add_subplot(title='Random walker path histogram, interpolated',
                                 aspect='equal')
            im = ax.imshow(H_plot, extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                           origin='lower', interpolation='nearest', aspect='equal')
            fig.colorbar(im, ax=ax)
        else:
            x_fine = np.linspace(xcenters[0], xcenters[-1], 1000)
            y_fine = np.linspace(ycenters[0], ycenters[-1], 1000)
            spline_interp = RectBivariateSpline(xcenters, ycenters, H_for_spline)
            H_fine = spline_interp(x_fine, y_fine).T  # transpose to shape (len(y_fine), len(x_fine))
            # Plot interpolated histogram
            ax = fig.add_subplot(title='Random walker path histogram, interpolated',
                                 aspect='equal')
            im = ax.imshow(H_fine, extent=(x_fine[0], x_fine[-1], y_fine[0], y_fine[-1]),
                           origin='lower', interpolation='bilinear', aspect='equal')
            fig.colorbar(im, ax=ax)

        # Display actual historgram
        # ax = fig.add_subplot(title='Random walker path histogram',
        #                      aspect='equal')
        # X, Y = np.meshgrid(xedges, yedges)
        # ax.pcolormesh(X, Y, H)

        # Old display with interpolation
        # ax = fig.add_subplot(title='Random walker path histogram, interpolated',
        #                      aspect='equal')
        # im = NonUniformImage(ax, interpolation='bilinear')
        # xcenters = (xedges[:-1] + xedges[1:]) / 2
        # ycenters = (yedges[:-1] + yedges[1:]) / 2
        # im.set_data(xcenters, ycenters, H)
        # ax.add_image(im)
        # self.percep_model.targets.plot_targets_to_axis(ax)
        # ax.set_xlim(dim_min[0],dim_max[0])
        # ax.set_ylim(dim_min[1],dim_max[1])

        self.percep_model.targets.plot_targets_to_axis(ax)

        # Plot individual walks
        if plot_tracks:
            for walk in all_walks:
                walk = np.column_stack(walk)
                ax.plot(walk[0,:], walk[1,:], 'k')
        plt.show()
