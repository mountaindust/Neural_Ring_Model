'''
Sets up a scenario in which a single locust makes decisions about the direction
it wants to go based on static targets with certain geometry
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from basic_units import radians

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
            self.theta = self.convert_angles(np.array(theta))
        else:
            self.theta = self.convert_angles(theta)

    
    def get_percep_angles(self,loc,angle=0):
        '''Given the (x,y) coordinate of an observer, loc, return an array of
        angles corresponding to how the targets are percieved from the position 
        of the observer when the observer is facing a direction given by angle.

        Returns -pi,pi for any target that loc overlaps, except in the case of 
        delta function targets, which returns angle instead.

        Parameters
        ----------
        loc : (x,y) of floats
        angle : float

        Returns
        -------
        Nx2 ndarray of angles theta_1 < theta_2, unless geom is None, then a 
        length N ndarray of single theta values instead.
        '''
        loc = np.array(loc)

        if self.geom_name is None:
            ##### Point targets #####
            on_target_bool = self.check_target_overlap(loc)
            if np.any(on_target_bool):
                angle_to_targets = self.get_angles_to_targets(loc,angle)
                angle_to_targets[on_target_bool] = angle
                return angle_to_targets
            else:
                return self.get_angles_to_targets(loc,angle)
        
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
                return self.convert_angles(np.column_stack([target_angles-pm_theta-angle,
                                                            target_angles+pm_theta-angle]))
            else:
                if isinstance(self.r,np.ndarray):
                    pm_theta = np.arcsin(self.r[~on_target_bool]/vecs_length)
                else:
                    pm_theta = np.arcsin(self.r/vecs_length)
                angle_to_targets = np.zeros(self.locs.shape)
                angle_to_targets[on_target_bool,:] = np.array([-np.pi,np.pi])
                angle_to_targets[~on_target_bool,:] = \
                    self.convert_angles(np.column_stack([target_angles-pm_theta-angle,
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
            angles1 = self.convert_angles(np.arctan2(vecs1[:,1],vecs1[:,0])-angle)
            angles2 = self.convert_angles(np.arctan2(vecs2[:,1],vecs2[:,0])-angle)
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
        return self.convert_angles(target_angles - angle)


    def check_target_overlap(self,loc):
        '''Check to see if loc overlaps with any of the targets. Return a bool
        array of length N (where N is the number of targets) indicating overlap.
        '''
        eps = np.finfo(np.float32).eps

        if self.geom_name is None:
            return self.locs == loc
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
    def convert_angles(theta):
        '''Given a scalar or array of angles, convert to angles in 
        [-np.pi,np.pi]
        '''
        return theta - (theta+np.pi)//(2*np.pi)*2*np.pi
    

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

    def __init__(self, targets=None, focal_loc=(5,10), focal_angle=0, type=None):
        '''Establishes an observer at location focal_loc, looking in a direction 
        given by focal_angle, at targets given by targets. All three of these 
        can be changed at any time as attributes.

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
        type : TODO
            how the angular extent of targets should be translated into 
            observation signal. None corresponds to an indicator function, but 
            this could be other things (blurred vision, higher weighting toward 
            the front of the observed object, etc.)
        '''

        self.focal_loc = np.array(focal_loc)
        self.focal_angle = focal_angle
        if targets is None:
            self.targets = Targets()
        else:
            assert isinstance(targets,Targets), "targets must be a Targets object."
            self.targets = targets
        self.type = type


    def get_signal(self,theta_mesh=2000):
        '''Translates the focal position/angle and information about targets 
        into a perception signal (1D mesh) with length res_num.

        Parameters
        ----------
        theta_mesh : float or 1D ndarray
            the number of equally spaced mesh points on [-pi,pi) to evaluate at 
            or a mesh of theta values to evaluate at

        Returns
        -------
        ndarray of length res_num representing the perception signal. Values in 
        the signal are normalized between 0 and 1.
        '''

        angles = self.targets.get_percep_angles(self.focal_loc, self.focal_angle)
        if isinstance(theta_mesh, int):
            signal = np.zeros(theta_mesh)
            theta_mesh = np.linspace(-np.pi, np.pi, theta_mesh+1)[:-1]
        else:
            signal = np.zeros(theta_mesh.shape)

        if self.targets.geom_name is None:
            for theta in angles:
                idx = np.searchsorted(theta_mesh,theta)
                if self.type is None:
                    # step function perception
                    if (theta-theta_mesh[idx-1]) < (theta_mesh[idx]-theta):
                        signal[idx-1] = 1
                    else:
                        signal[idx] = 1
                else:
                    raise NotImplementedError("Unknown perception type.")
        elif self.targets.geom_name == 'circle' or self.targets.geom_name == 'segment':
            for thetas in angles:
                if self.type is None:
                    # step function perception
                    if thetas[1] > thetas[0]:
                        theta_bool = np.logical_and(thetas[0]<=theta_mesh,
                                                    theta_mesh<=thetas[1])
                    else:
                        theta_bool = np.logical_or(thetas[0]<=theta_mesh,
                                                   theta_mesh<=thetas[1])
                    signal[theta_bool] = 1
                else:
                    raise NotImplementedError("Unknown perception type.")
        else:
            raise NotImplementedError("Unknown target geometry name.")
        
        return signal


    def plot(self, wb_plot=False):
        '''Plots the targets and their angular extents from the observer, and 
        also the signal distribution from the point of view of the observer.
        
        Set wb_plot to True if plotting in a Jupyber notebook
        '''

        angles = self.targets.get_percep_angles(self.focal_loc, self.focal_angle)

        if wb_plot:
            plt.figure(figsize=(6.5,3.25))
        else:
            plt.figure(figsize=(12,6))

        ###### Target Geometry Plot ######
        ax1 = plt.subplot(121)

        # First, plot the targets themselves
        self.targets.plot_targets_to_axis(ax1)

        # Now plot perception lines
        if self.targets.geom_name is None:
            # plot geometric angles. Requires adding back angle of focal locust
            for n, theta in enumerate(angles+self.focal_angle):
                r = np.linalg.norm(self.targets.locs[n,:] - self.focal_loc)
                x = (self.focal_loc[0],self.focal_loc[0] + r*np.cos(theta))
                y = (self.focal_loc[1],self.focal_loc[1] + r*np.sin(theta))
                ax1.plot(x,y,'k')
        elif self.targets.geom_name == 'circle':
            # plot perception angles. Requires adding back angle of focal locust
            for n, thetas in enumerate(angles+self.focal_angle):
                r = np.linalg.norm(self.targets.locs[n,:] - self.focal_loc)
                for ii in range(2):
                    x = (self.focal_loc[0],self.focal_loc[0] + r*np.cos(thetas[ii]))
                    y = (self.focal_loc[1],self.focal_loc[1] + r*np.sin(thetas[ii]))
                    ax1.plot(x,y,'k')
        elif self.targets.geom_name == 'segment':
            # plot perception angles. Requires adding back angle of focal locust
            for n, thetas in enumerate(angles+self.focal_angle):
                r = np.linalg.norm(self.targets.locs[n,:] - self.focal_loc)
                try:
                    l = self.targets.l[n]
                    r += l[n]
                except TypeError:
                    l = self.targets.l
                    r += l
                for ii in range(2):
                    x = (self.focal_loc[0],self.focal_loc[0] + r*np.cos(thetas[ii]))
                    y = (self.focal_loc[1],self.focal_loc[1] + r*np.sin(thetas[ii]))
                    ax1.plot(x,y,'k')
        else:
            raise NotImplementedError("This geometry still TBD in PerceptionModel.")

        ax1.arrow(self.focal_loc[0],self.focal_loc[1],
                  0.5*np.cos(self.focal_angle),0.5*np.sin(self.focal_angle), 
                width=0.1, head_length=0.25)
        ax1.set_aspect('equal')
        ax1.set_title('Target Geometry')

        ###### Perception Signal Plot ######
        ax2 = plt.subplot(122, projection='polar')

        p_func = self.get_signal(2000)
        theta_mesh = np.linspace(-np.pi, np.pi, 2000+1)[:-1]

        ax2.plot(theta_mesh,p_func)
        ax2.arrow(0,-0.5,0,0.25, width=0.2, head_length=0.15)
        ax2.set_rmin(-0.5)
        ax2.set_rmax(1)
        ax2.set_rticks([0, 0.5, 1])
        ax2.set_rlabel_position(0)
        ax2.set_title('Perception Signal')
        plt.show()



class DirectionModel:

    def __init__(self, percep_model=None, consensus_type='additive', 
                 weighting_name='truncnorm', *args, **kwargs):
        '''From a PerceptionModel with its Targets object, establishes a model 
        for chosing direction based on a weighting of the signal via convolution 
        and a method for finding a consensus direction from the result.
        
        Parameters
        ----------
        percep_model : PerceptionModel
            A PerceptionModel object (with its Targets object) that establishes 
            the geometry of the scenario. If none is provided, a default one 
            will be created. The PerceptionModel can be updated to obtain 
            consensus directions for different layouts or focal locations/angles.
        consensus_type : {'additive', 'argmax'}
            Method that will be used to find a consensus direction after 
            convoluting the perception signal with the weighting.
        weighting_name : {'truncnorm'}
            Weighting for the signal convolution. The necessary parameters for 
            this weighting should be provided after this key word argument. The 
            following is a list of parameterizations.
            - 'truncnorm'
                - mu=0 : mean
                - sigma=np.pi/8 : standard deviation
                - left=-np.pi : left cutoff
                - right=np.pi : right cutoff
        '''
        if percep_model is None:
            self.percep_model = PerceptionModel()
        else:
            assert isinstance(percep_model,PerceptionModel),\
            "percep_model must be a PerceptionModel object."
            self.percep_model = percep_model
        self.consensus_type = consensus_type
        if weighting_name == 'truncnorm':
            self.weighting = self.truncnorm(*args, **kwargs)
            self.weighting_name = weighting_name
        else:
            raise NotImplementedError("Unknown weighting kernel.")


    def truncnorm(self, mu=0, sigma=np.pi/8, left=-np.pi, right=np.pi):
        '''Function generator that returns a truncated normal pdf with mean mu, 
        std sigma, and truncated at left and right.
        '''

        a, b = (left - mu) / sigma, (right - mu) / sigma
        rv = stats.truncnorm(a,b,mu,sigma)
        # x will be theta - phi
        return lambda x: rv.pdf(x)
    

    def hamiltonian(self, theta_mesh):
        '''Convolution of the weighting kernel with the signal over [-pi,pi]. 

        Parameters
        ----------
        theta_mesh : 1D ndarray
            a mesh of points on [-pi,pi) to evaluate the hamiltonian at

        Returns
        -------
        ndarray of convoluted signal and weighting on [-pi,pi). The length of 
        the returned array will match the input array.
        '''
        if self.weighting_name == 'truncnorm':
            kernel = self.weighting(theta_mesh)
            signal = self.percep_model.get_signal(theta_mesh)
            # Periodic convolution via convolution theorem.
            # Results must be shifted because numpy fft puts the zero frequency
            #   at the left-most position of the array.
            return np.fft.fftshift(np.fft.irfft( 
                                   np.fft.rfft(kernel)*np.fft.rfft(signal),
                                   len(signal)))
        else:
            raise NotImplementedError("Unknown weighting kernel.")
        

    def get_direction(self, res_num=2000, return_H=False):
        '''Get consensus direction for current parameterization of the 
        DirectionModel.

        If the difference between the min and the max of the Hamiltonian is 
        essentially zero (within machine precision), returns the focal angle of 
        the PerceptionModel. This also happens in the additive model if the 
        resulting directional angle is essentially the zero vector.

        Parameters
        ----------
        res_num : float
            the number of equally spaced mesh points on [-pi,pi) to evaluate the
            hamiltonian at. More increases accuracy at the expense of speed.
        return_H : bool
            whether or not to return the full Hamiltonian too

        Returns
        -------
        theta : the Euclidean (allocentric) consensus direction as a float 
        within [-pi,pi)
        '''
        eps = np.finfo(np.float32).eps

        theta_mesh = np.linspace(-np.pi, np.pi, res_num+1)[:-1]
        H = self.hamiltonian(theta_mesh)
        
        if H.max()-H.min() < eps:
            if return_H:
                return self.percep_model.focal_angle, H
            else:
                return self.percep_model.focal_angle

        if self.consensus_type == 'additive':
            x = np.sum(np.cos(theta_mesh)*H)
            y = np.sum(np.sin(theta_mesh)*H)
            if np.abs(x)<eps and np.abs(y)<eps:
                if return_H:
                    return self.percep_model.focal_angle, H
                else:
                    return self.percep_model.focal_angle
            else:
                if return_H:
                    return np.arctan2(y,x), H
                else:
                    return np.arctan2(y,x)
        elif self.consensus_type == 'argmax':
            idx_array = H.argmax()
            if len(idx_array) > 1:
                idx = np.random.choice(idx_array)
            else:
                idx = idx_array[0]
            if return_H:
                return theta_mesh[idx], H
            else:
                return theta_mesh[idx]
        

    def plot_weighting(self, wb_plot=False):
        '''Plot the currently selected weighting function.
        
        Set wb_plot to True if plotting in a Jupyber notebook
        '''

        if wb_plot:
            plt.figure(figsize=(6.5,2))
        else:
            plt.figure(figsize=(8,5))
        theta_mesh = np.linspace(-np.pi, np.pi, 2001)
        plt.plot(theta_mesh,self.weighting(theta_mesh))
        plt.title(self.weighting_name)
        plt.show()


    def plot_hamiltonian(self, focal_loc_mesh=None, with_signal=False,
                         wb_plot=True):
        '''Plot the hamiltonian with or without the signal alongside it.

        Set wb_plot to True if plotting in a Jupyber notebook
        
        Parameters
        ----------
        focal_loc_mesh : Nx2 ndarray, optional
            If provided, will plot the hamiltonian as a 3D surface plot 
            parameterized by the focal (x,y) locations in Euclidean space given 
            in this mesh. Otherwise, will plot the current focal location given 
            by the PerceptionModel object.
        with_signal : bool, default=False
            Include a plot of the signal alongside the Hamiltonian for comparison.
        '''
        res_num = 3000
        theta_mesh = np.linspace(-np.pi, np.pi, res_num+1)[:-1]

        if focal_loc_mesh is None:
            if with_signal:
                if wb_plot:
                    fig, axs = plt.subplots(2, figsize=(6.5,4))
                else:
                    fig, axs = plt.subplots(2, figsize=(8,5))
            else:
                if wb_plot:
                    fig, axs = plt.subplots(figsize=(6.5,2))
                else:
                    fig, axs = plt.subplots(figsize=(8,2.5))
                axs = np.array([axs])
            # Get direction angle and hamiltonian
            dir_angle, H_array = self.get_direction(res_num, return_H=True)
            axs[0].plot(theta_mesh*radians, H_array, xunits=radians)
            axs[0].set_title('Hamiltonian')
            # Indicate chosen angle
            idx = np.searchsorted(theta_mesh, dir_angle)
            axs[0].plot(dir_angle*radians, H_array[idx], 'ok')
            if with_signal:
                signal = self.percep_model.get_signal(theta_mesh)
                axs[1].plot(theta_mesh*radians, signal, xunits=radians)
                axs[1].plot(dir_angle*radians, signal[idx], 'ok')
                axs[1].set_title('Perceived Signal')
        else:
            if with_signal:
                if wb_plot:
                    fig, axs = plt.subplots(1, 2, figsize=(6.5,4), 
                                            subplot_kw=dict(projection='3d'))
                else:
                    fig, axs = plt.subplots(1, 2, figsize=(10,5), 
                                            subplot_kw=dict(projection='3d'))
                # create array to store signal
                signal_array = np.zeros((focal_loc_mesh.shape[0],res_num))
            else:
                if wb_plot:
                    fig, axs = plt.subplots(figsize=(6.5,4), 
                                            subplot_kw=dict(projection='3d'))
                else:
                    fig, axs = plt.subplots(figsize=(5.5,5), 
                                            subplot_kw=dict(projection='3d'))
                axs = np.array([axs])
            # save current loc
            current_loc = self.percep_model.focal_loc.copy()
            # create array to store hamiltonian
            H_array = np.zeros((focal_loc_mesh.shape[0],res_num))
            # also get an array of angles to create axis ticks
            angle_array = np.zeros(focal_loc_mesh.shape[0])
            # get array to store consensus angle
            dir_angle = np.zeros(angle_array.shape)
            idx_array = np.zeros(angle_array.shape, dtype=int)
            for n,loc in enumerate(focal_loc_mesh):
                self.percep_model.focal_loc = loc
                angle_array[n] = self.percep_model.targets.get_angles_to_targets(loc,
                                                    self.percep_model.focal_angle)[-1]
                if with_signal:
                    signal_array[n,:] = self.percep_model.get_signal(theta_mesh)
                # Get direction angle and hamiltonian
                dir_angle[n], H_array[n,:] = self.get_direction(res_num, return_H=True)
                idx_array[n] = np.searchsorted(theta_mesh, dir_angle[n])
            
            # Create 3D plot(s)
            theta_mgrid, angle_mgrid = np.meshgrid(theta_mesh, angle_array)
            axs[0].plot_surface(theta_mgrid, angle_mgrid, H_array, 
                                cmap=cm.viridis)
            axs[0].set_title('Hamiltonian')
            axs[0].set_xlabel(r'$\theta$')
            axs[0].set_ylabel('angle to last target')
            # Indicate chosen angles
            H_vals = [H_array[n,idx_array[n]] for n in range(len(angle_array))]
            axs[0].scatter(dir_angle, angle_array, H_vals, c='k')
            if with_signal:
                axs[1].plot_surface(theta_mgrid, angle_mgrid, signal_array,
                                    cmap=cm.viridis)
                axs[1].set_title('Perception Signal')
                axs[1].set_xlabel(r'$\theta$')
                axs[1].set_ylabel('angle to last target')
                sig_vals = [signal_array[n,idx_array[n]] for n in range(len(angle_array))]
                axs[1].scatter(dir_angle, angle_array, sig_vals, c='k')

        if focal_loc_mesh is not None:
            # replace current loc
            self.percep_model.focal_loc = current_loc
        fig.tight_layout()
        plt.show()


    def plot_direction_mesh(self, xlim=(0,24), num_x=25, ylim=(0,24), num_y=25, 
                            return_theta=False, wb_plot=False):
        '''Create a mesh of starting locations and, for each point in the mesh, 
        get the direction of travel as a scalar theta. Plots the result as a 
        vector field of unit vectors and optionally returns a scalar field of 
        the values theta within [-pi,pi].

        Set wb_plot to True if plotting in a Jupyber notebook

        Parameters
        ----------
        xlim : (xmin,xmax) tuple of floats
            x limits for mesh, inclusive
        num_x : number of steps in x direction
        ylim : (ymin,ymax) tuple of floats
            y limits for mesh, inclusive
        num_y : number of steps in y direction
        return_theta : bool
            whether or not to return a theta scalar field
        '''

        current_focal_loc = self.percep_model.focal_loc.copy()

        xmesh = np.linspace(xlim[0], xlim[1], num_x)
        ymesh = np.linspace(ylim[0], ylim[1], num_y)

        X, Y = np.meshgrid(xmesh, ymesh)
        theta_mesh = np.zeros(X.shape)
        U = np.zeros_like(theta_mesh)
        V = np.zeros_like(theta_mesh)

        for ii in range(num_x):
            for jj in range(num_y):
                this_x = X[jj,ii]
                this_y = Y[jj,ii]
                self.percep_model.focal_loc = np.array([this_x,this_y])
                theta_mesh[jj,ii] = self.get_direction()
                U[jj,ii] = np.cos(theta_mesh[jj,ii])
                V[jj,ii] = np.sin(theta_mesh[jj,ii])

        self.percep_model.focal_loc = current_focal_loc
        if wb_plot:
            plt.figure(figsize=(6.5,4))
        else:
            plt.figure(figsize=(5.5,5))
        ax = plt.subplot()

        # Plot targets
        self.percep_model.targets.plot_targets_to_axis(ax)
        # Plot arrows
        ax.quiver(X, Y, U, V, angles='xy')
        ax.set_title("Direction Model")
        plt.show()
        if return_theta:
            return theta_mesh
        

    def plot_walker(self, step=0.1, std=np.pi/10, trials=10, max_steps=3000,
                    start_loc=None, start_angle=None):
        '''Plot a walker that starts at a specified location looking in a 
        specified angle (defaults to the focal_loc and focal_angle in attached 
        PerceptionModel) and moves in the direction given by the current 
        direction model with a specified step and angular Gaussian noise with 
        standard deviation as specified. Repeat for a number of trials and plot 
        a heat map of these walks in 2D space.

        The walker stops whenever it is detected to be overlapping a target or 
        after max_steps.

        Parameters
        ----------
        step : float
            How far to move in the determined direction
        std : float
            Standard deviation of angular Gaussian noise with mean zero
        trials : int
            Number of walks to perform and aggregate
        max_steps : int
            Maximum number of steps for each walker
        start_loc : (x,y) coordinates, optional
            Starting location of the walk, defaults to focal_loc in the attached 
            PerceptionModel
        start_angle : float
            Starting direction that the walker is facing. Defaults to 
            focal_angle in the attached PerceptionModel
        '''
        pass
