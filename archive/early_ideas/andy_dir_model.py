'''
Model of decision making from conversations with Andy.

Requires a Targets object and a PerceptualModel object. May or may not work 
with the new versions of these. The code has been put here in case it needs to 
be resurrected.
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.image import NonUniformImage
from basic_units import radians

class AndyDirectionModel:

    def __init__(self, percep_model=None, consensus_type='additive', 
                 weighting_name='truncnorm', *args, **kwargs):
        '''From a PerceptionModel with its Targets object, establishes a model 
        for chosing direction based on a weighting of a binary signal via convolution 
        and a method for finding a consensus direction from the result.

        This model relies on get_binary_signal from the PerceptionModel, which
        creates an egocentric binary signal based on target locations and
        focal angle.
        
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
        res_num : int
            the number of equally spaced mesh points on [-pi,pi) to evaluate the
            hamiltonian at. More increases accuracy at the expense of speed.
        weighting_name : {'truncnorm'}
            Weighting for the signal convolution. The necessary parameters for 
            this weighting should be provided after this key word argument. The 
            following is a list of parameterizations.
            - 'truncnorm'
                - mu=0 : mean
                - sigma=np.pi/8 : standard deviation
                - left=-np.pi : left cutoff
                - right=np.pi : right cutoff
            - 'trunccosine'
                - beta=2 : stretch
                - phi=0 : phase
                - left=-np.pi/2 : left cutoff
                - right=np.pi/2 : right cutoff
        '''

        self.consensus_type = consensus_type
        if weighting_name == 'truncnorm':
            self.weighting = self.truncnorm(*args, **kwargs)
            self.weighting_name = weighting_name
        elif weighting_name == 'trunccosine':
            self.weighting = self.trunccosine(*args, **kwargs)
            self.weighting_name = weighting_name
        else:
            raise NotImplementedError("Unknown weighting kernel.")
        
        if percep_model is None:
            self.percep_model = PerceptionModel()
        else:
            assert isinstance(percep_model,PerceptionModel),\
            "percep_model must be a PerceptionModel object."
            self.percep_model = percep_model

        # Random number generator for certain processes within the class;
        #   Can seed here for reproducability.
        self.rng = np.random.default_rng()


    ############################################################################
    #### Weighting function generators. All return a function on an ndarray ####
    ############################################################################

    def truncnorm(self, mu=0, sigma=np.pi/8, left=-np.pi, right=np.pi):
        '''Function generator that returns a truncated normal pdf with mean mu, 
        std sigma, and truncated at left and right.
        '''

        a, b = (left - mu) / sigma, (right - mu) / sigma
        rv = stats.truncnorm(a,b,mu,sigma)
        # x will be theta - phi
        return lambda x: rv.pdf(x)
    

    def trunccosine(self, left=-np.pi, right=np.pi, nu=1):
        '''Function generator that returns a cos(pi*(x/pi)^nu) with support on the
        interval [left,right]. Outside this interval, the function returns zero.
        
        The idea here is to rescale theta to be between 0 and 1, then raise to the
        power nu to control the steepness of the function, then scale back to
        between -pi and pi and take the cosine. Truncation allows for a blindspot.'''

        def trunccos(x):
            result = np.zeros_like(x)
            result[(x>=left) & (x<=right)] = np.cos(
                np.pi*(x[(x>=left) & (x<=right)]/np.pi)**nu)
            return result

        return trunccos
    
    ############################################################################


    def hamiltonian(self):
        '''Convolution of the weighting kernel with the signal over [-pi,pi]. 
        Since the signal is egocentric, the hamiltonian will be as well.

        Returns
        -------
        ndarray of convoluted signal and weighting on [-pi,pi). The length of 
        the returned array will match the input array.
        '''

        kernel = self.weighting(self.percep_model.theta_mesh)
        signal = self.percep_model.get_binary_signal()
        # Periodic convolution via convolution theorem.
        # Results must be shifted because numpy fft puts the zero frequency
        #   at the left-most position of the array.
        return np.fft.fftshift(np.fft.irfft( 
                                np.fft.rfft(kernel)*np.fft.rfft(signal),
                                len(signal)))


    def get_direction(self, dt=None, return_H=False):
        '''Get consensus ALLOCENTRIC direction for current parameterization of 
        the DirectionModel.

        If the difference between the min and the max of the Hamiltonian is 
        essentially zero (within machine precision), returns zero (corresponding 
        to the current allocentric heading). This also happens in the additive 
        model if the resulting directional angle is essentially the zero vector.

        Parameters
        ----------
        dt : float, optional
            time step for integration of torque model. Not used in this model.
        return_H : bool
            whether or not to return the full Hamiltonian too

        Returns
        -------
        theta : the egocentric consensus direction as a float within [-pi,pi)
        '''
        eps = np.finfo(np.float32).eps

        theta_mesh = self.percep_model.theta_mesh
        H = self.hamiltonian()
        
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
                    return np.arctan2(y,x)+self.percep_model.focal_angle, H
                else:
                    return np.arctan2(y,x)+self.percep_model.focal_angle
        elif self.consensus_type == 'argmax':
            # a straight argmax is VERY numerically unstable, especially in 
            #   cases (which are of interest) where H is multimodal with the 
            #   different peaks at essentially the exact same height. In this 
            #   scenario, it will return the one which happens to be highest due 
            #   to numerical error, which is in turn tends to depend upon things 
            #   like the size of the theta mesh. This ends up yielding consistent
            #   choices rather than random ones. To fix this, get all locations 
            #   within numerical error and explicitly pick randomly from among 
            #   them.
            
            max_val = H.max()
            # get all indices which are within numerical error of this value
            idx_array = np.argwhere(max_val-H<eps).flatten()
            if len(idx_array) > 1:
                idx = self.rng.choice(idx_array)
            else:
                idx = idx_array[0]
            if return_H:
                return theta_mesh[idx]+self.percep_model.focal_angle, H
            else:
                return theta_mesh[idx]+self.percep_model.focal_angle
        

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
        theta_mesh = self.percep_model.theta_mesh
        res_num = theta_mesh.size

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
            dir_angle, H_array = self.get_direction(return_H=True)
            axs[0].plot(theta_mesh*radians, H_array, xunits=radians)
            axs[0].set_title('Hamiltonian')
            # Indicate chosen angle
            idx = np.searchsorted(theta_mesh, dir_angle)
            axs[0].plot(dir_angle*radians, H_array[idx], 'ok')
            if with_signal:
                signal = self.percep_model.get_binary_signal()
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
            # also get an array of positions for axis ticks
            if focal_loc_mesh[:,0].max() - focal_loc_mesh[:,0].min() >= \
               focal_loc_mesh[:,1].max() - focal_loc_mesh[:,1].min():
                axis_mesh = focal_loc_mesh[:,0]
                axis_x = True
            else:
                axis_mesh = focal_loc_mesh[:,1]
                axis_x = False
            # get array to store consensus angle
            dir_angle = np.zeros(focal_loc_mesh.shape[0])
            idx_array = np.zeros(focal_loc_mesh.shape[0], dtype=int)
            for n,loc in enumerate(focal_loc_mesh):
                self.percep_model.focal_loc = loc
                if with_signal:
                    signal_array[n,:] = self.percep_model.get_binary_signal()
                # Get direction angle and hamiltonian
                dir_angle[n], H_array[n,:] = self.get_direction(return_H=True)
                idx_array[n] = np.searchsorted(theta_mesh, dir_angle[n])
            
            # Create 3D plot(s)
            theta_mgrid, pos_mgrid = np.meshgrid(theta_mesh, axis_mesh)
            axs[0].plot_surface(theta_mgrid, pos_mgrid, H_array, 
                                cmap=cm.viridis)
            axs[0].set_title('Hamiltonian')
            axs[0].set_xlabel(r'$\theta$')
            if axis_x:
                axs[0].set_ylabel('x position')
            else:
                axs[0].set_ylabel('y position')
            # Indicate chosen angles
            H_vals = [H_array[n,idx_array[n]] for n in range(len(axis_mesh))]
            axs[0].scatter(dir_angle, axis_mesh, H_vals, c='k')
            if with_signal:
                axs[1].plot_surface(theta_mgrid, pos_mgrid, signal_array,
                                    cmap=cm.viridis)
                axs[1].set_title('Perception Signal')
                axs[1].set_xlabel(r'$\theta$')
                if axis_x:
                    axs[1].set_ylabel('x position')
                else:
                    axs[1].set_ylabel('y position')
                sig_vals = [signal_array[n,idx_array[n]] for n in range(len(axis_mesh))]
                axs[1].scatter(dir_angle, axis_mesh, sig_vals, c='k')

        if focal_loc_mesh is not None:
            # replace current loc
            self.percep_model.focal_loc = current_loc
        fig.tight_layout()
        plt.show()


    def plot_weighting(self, wb_plot=False):
        '''Plot the currently selected weighting function.
        
        Set wb_plot to True if plotting in a Jupyter notebook
        '''

        if wb_plot:
            plt.figure(figsize=(6.5,2))
        else:
            plt.figure(figsize=(8,5))
        theta_mesh = self.percep_model.theta_mesh
        plt.plot(theta_mesh,self.weighting(theta_mesh))
        plt.title(self.weighting_name)
        plt.show()


    def plot_direction_mesh(self, xlim=(0,24), num_x=25, ylim=(0,20), num_y=21, 
                            return_theta=False, wb_plot=False):
        '''Create a mesh of starting locations and, for each point in the mesh, 
        get the allocentric direction of travel as a scalar theta. Plots the 
        result as a vector field of unit vectors and optionally returns a scalar 
        field of the values theta within [-pi,pi].

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
                # TODO:
                # Better would probably be to calculaute the steady-state direction.
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
        ax.set_aspect('equal')
        plt.show()
        if return_theta:
            return theta_mesh


    def plot_walker(self, dt=0.1, v=1, std=0, repetitions=20, max_steps=3000,
                    start_loc=None, start_angle=None, plot_tracks=False):
        '''Plot a walker that starts at a specified location looking in a 
        specified angle (defaults to the focal_loc and focal_angle in attached 
        PerceptionModel) and moves in the direction given by the current 
        direction model with a specified step size and angular Gaussian noise 
        with standard deviation as specified. Repeat for a number of 
        repetitions and plot a heat map of these walks in 2D space.

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
                # if step > 75:
                #     import pdb; pdb.set_trace()
                # determine allocentric direction and take a step
                #   assume turning speed is infinite
                if std > 0:
                    noise = self.rng.normal(scale=std)
                else:
                    noise = 0
                # NOTE:
                # This walk is modeled on a two-step process:
                # 1) decide on a direction to face based on an amount of time dt
                # 2) move forward in that direction instantaneously
                theta = self.get_direction(dt) + noise
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
        n_xedges = round((dim_max[0]-dim_min[0])/0.25)
        n_yedges = round((dim_max[1]-dim_min[1])/0.25)
        xedges = np.linspace(dim_min[0], dim_max[0], n_xedges)
        yedges = np.linspace(dim_min[1], dim_max[1], n_yedges)

        H, xedges, yedges = np.histogram2d(walks[0,:],walks[1,:], 
                                           bins=(xedges,yedges))
        H = H.T # for plotting

        fig = plt.figure(figsize=(5,5))

        # Display actual historgram
        # ax = fig.add_subplot(title='Random walker path histogram',
        #                      aspect='equal')
        # X, Y = np.meshgrid(xedges, yedges)
        # ax.pcolormesh(X, Y, H)
        # self.percep_model.targets.plot_targets_to_axis(ax)

        # Display with interpolation
        ax = fig.add_subplot(title='Random walker path histogram, interpolated',
                             aspect='equal')
        im = NonUniformImage(ax, interpolation='bilinear')
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        im.set_data(xcenters, ycenters, H)
        ax.add_image(im)
        self.percep_model.targets.plot_targets_to_axis(ax)
        ax.set_xlim(dim_min[0],dim_max[0])
        ax.set_ylim(dim_min[1],dim_max[1])

        # Plot individual walks
        if plot_tracks:
            for walk in all_walks:
                walk = np.column_stack(walk)
                ax.plot(walk[0,:], walk[1,:], 'k')
        plt.show()
