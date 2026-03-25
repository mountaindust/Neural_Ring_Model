'''
Sets up a scenario in which a single locust makes decisions about the direction
it wants to go based on static targets with certain geometry
'''

import numpy as np
from scipy.integrate import solve_ivp, quad
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

    def __init__(self, locs=None, geom_name=None, r=None, l=None, theta=0, 
                 values=1):
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
        value : float or length N ndarray, default=1
            attractiveness of each target as a scalar
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
        if np.ndim(values) == 0:
            self.values = np.full(self.locs.shape[0], values)
        else:
            self.values = np.array(values)

    
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
    density of neurons in the ring as a function of angle.
    
    Following mathematics conventions, egocentric angles increase counterclockwise;
    e.g., positive egocentric angles are to the left of the observer and 
    negative egocentric angles are to the right.'''

    def __init__(self, targets=None, focal_loc=(5,10), focal_angle=0, 
                 neural_weight='cutoff', neural_angle='integral', theta_mesh=2000):
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
        neural_weight : {'cutoff', 'tanh_plus', None} (default = 'cutoff')
            Weighting function for the neural band.
                - 'cutoff' : a smooth cutoff function that is 1 in front and 
                             0 in back, with a smooth transition in between. 
                             It is parameterized by a and b, which control the 
                             angles at which the cutoff starts and ends, respectively.
                             See _smooth_cutoff for details.
                - None : no weighting, i.e. flat. All angles are weighted equally.
        neural_angle : {'integral', 'power', None} (default = 'integral')
            This defines a mapping between perceived center of each target and
            the neural position of the corresponding spin group.
            Options are:
                - 'integral' : the neural position is given by integrating the
                               neural weight function like a CDF to the perceived
                               center of the target.
                - 'power' : the neural position is given by applying a power
                            function to the perceived center of the target:
                            f(theta) = pi * sign(theta) * (|theta|/pi)^c.
                            The parameter c controls the exponent; c = 1 gives
                            the identity, c < 1 expands front angles, c > 1
                            compresses them.
                - None : no transformation, i.e. identity. The neural position is
                         the same as the perceived center of the target.
        theta_mesh : float or 1D ndarray
            the number of equally spaced mesh points on [-pi,pi) to evaluate at 
            or a mesh of theta values to evaluate at
        '''

        self.focal_loc = np.array(focal_loc, dtype=float)
        self.focal_angle = focal_angle
        self.neural_weight = neural_weight
        self.neural_angle = neural_angle

        # Set default parameters for the weighting function.
        if neural_weight == 'cutoff':
            # = 1 when |theta|<self.a, = 0 when |theta|>self.b, smooth in between
            self.a = np.pi/3 
            self.b = 4*np.pi/5
        # elif neural_weight == 'tanh_plus':
        #     self.a = 2
        #     self.b = 2*np.pi/3
        else:
            self.a = None
            self.b = None

        # Set default parameters for the neural position transformation function.
        if neural_angle is None or neural_angle == 'integral':
            self.c = None
        else:
            self.c = 0.5

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
    def _smooth_cutoff(x, a, b):
        """
        Evaluates the smooth cutoff function at x (scalar or array).
        Returns 0.0 outside [-b, b], 1.0 on [-a, a], and a smooth bump
        in between.  -b < -a < 0 < a < b
        """
        x = np.asarray(x, dtype=float)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)

        absx = np.abs(x)
        norm = b - a   # positive since b > a > 0

        # Compute the smooth transition value for the intermediate region
        # a < |x| < b.  Outside that region the denominators (b - absx) and
        # (absx - a) would be zero or negative, so we substitute the finite
        # fill value 1.0 to keep the division well-defined everywhere.
        # The filled elements are always masked by the outer np.where below,
        # so their values never appear in the output.

        # -norm/(b-x): negative / positive = negative
        arg1 = -norm / np.where(absx < b, b - absx, 1.0)  # fill used when |x| >= b
        # -norm/(x-a): negative / positive = negative
        arg2 = -norm / np.where(absx > a, absx - a, 1.0)  # fill used when |x| <= a
        exp1 = np.exp(arg1)
        exp2 = np.exp(arg2)
        smooth = exp1 / (exp1 + exp2)

        result = np.where(absx >= b, 0.0,
                 np.where(absx <= a, 1.0,
                          smooth))

        return result.item() if scalar_input else result
        
    @staticmethod
    def _smooth_cutoff_integral_scalar(theta, a, b, tol=1.49e-10):
        """
        Scalar kernel for _smooth_cutoff_integral. Computes F(theta; a, b)
        for a single float theta.
        """
        if theta < 0:
            NEG = True
            theta = -theta
        elif theta == 0:
            return 0.0
        else:
            NEG = False
        if not (0 <= a < b):
            raise ValueError(f"Parameters must satisfy 0 <= a < b (a={a}, b={b}).")

        # Normalization factor for the integral of the cutoff function.
        #   The area under the curve from 0 to b is a + (b-a)/2 = 0.5*(a+b).
        norm = 2*np.pi/(a+b)

        # Check for values below a
        if theta <= a:
            # integral is just the area of the rectangle
            if NEG:
                return -theta*norm
            else:
                return theta*norm
        elif theta >= b:
            if NEG:
                return -np.pi
            else:
                return np.pi

        # All other cases: a < theta < b.
        # Calculate integral from a to theta and add area from 0 to a.

        # Nudge integration bounds inward slightly to avoid handing the
        # essential singularities directly to quad; the integrand is
        # effectively 0 in those tiny gaps anyway.
        eps = (b - a) * 1e-14
        lower = a + eps
        upper = min(theta, b - eps)

        if upper <= lower:
            if NEG:
                return -a*norm
            else:
                return a*norm

        result, _err = quad(
            PerceptionModel._smooth_cutoff,
            lower,
            upper,
            args=(a, b),
            epsabs=tol,
            epsrel=tol,
            limit=200,
        )
        if NEG:
            return -(a + result)*norm
        else:
            return (a + result)*norm

    @staticmethod
    def _smooth_cutoff_integral(theta, a, b, tol=1.49e-10):
        """
        Compute F(x; a, b) = integral from 0 to x of the smooth cutoff function.
        Accepts scalar or array theta; always returns the same shape.

        Parameters
        ----------
        theta : float or array_like
            Upper limit(s) of integration.
        a, b : Parameters of the smooth cutoff function; must satisfy 0 <= a < b.
        tol  : Absolute and relative tolerance passed to scipy quad.

        Returns
        -------
        float or ndarray : The value(s) of the integral.
        """
        theta = np.asarray(theta, dtype=float)
        scalar_input = theta.ndim == 0
        vfunc = np.vectorize(
            PerceptionModel._smooth_cutoff_integral_scalar,
            excluded=['a', 'b', 'tol'],
        )
        result = vfunc(theta, a=a, b=b, tol=tol)
        return result.item() if scalar_input else result

    @staticmethod
    def _smooth_cutoff_int_inverse_scalar(y, a, b, tol=1.0e-8):
        """
        Scalar kernel for _smooth_cutoff_int_inverse. Computes F^{-1}(y; a, b)
        for a single float y.
        """
        if not (0 <= a < b):
            raise ValueError(f"Parameters must satisfy 0 <= a < b (a={a}, b={b}).")

        if y < -np.pi or y > np.pi:
            raise ValueError(f"y must satisfy -pi <= y <= pi (y={y}).")

        if y == -np.pi:
            return -b
        elif y == np.pi:
            return b

        # Normalization factor for the integral of the cutoff function.
        #   The area under the curve from 0 to b is a + (b-a)/2 = 0.5*(a+b).
        norm = 2*np.pi/(a+b)

        # Check for values between -a*norm and a*norm, where the inverse is just
        #   a linear scaling of y.
        if -a*norm <= y <= a*norm:
            return y/norm

        # All other cases: a*norm < |y| < pi.
        # Calculate inverse by finding root of F(theta) - y.

        def func(theta):
            return PerceptionModel._smooth_cutoff_integral_scalar(theta, a, b, tol) - np.abs(y)

        # Bracket: F is strictly increasing from 0 to pi on (a, b).
        eps = (b - a) * 1e-12
        x_lo = a + eps
        x_hi = b - eps

        result = brentq(func, x_lo, x_hi, xtol=tol, rtol=tol, maxiter=200)
        return np.sign(y) * result

    @staticmethod
    def _smooth_cutoff_int_inverse(y, a, b, tol=1.0e-8):
        """
        Compute F^{-1}(y; a, b): the value of x such that F(x; a, b) = y.
        Accepts scalar or array y; always returns the same shape.

        Parameters
        ----------
        y    : float or array_like
            Target value(s); each must satisfy -pi <= y <= pi.
        a, b : Parameters of the smooth cutoff function; must satisfy 0 <= a < b.
        tol  : Absolute and relative tolerance passed to scipy brentq.

        Returns
        -------
        float or ndarray : The value(s) of x such that F(x; a, b) = y.
        """
        y = np.asarray(y, dtype=float)
        scalar_input = y.ndim == 0
        vfunc = np.vectorize(
            PerceptionModel._smooth_cutoff_int_inverse_scalar,
            excluded=['a', 'b', 'tol'],
        )
        result = vfunc(y, a=a, b=b, tol=tol)
        return result.item() if scalar_input else result
    
    @staticmethod # An alternative idea to the cutoff function? Currently unused.
    def _tanh_plus(theta, a, b):
        return (np.tanh(a*(1-(theta/b)**2) ) + 1.0001)/(1.0001+np.tanh(a))

    @staticmethod # Mentioned in our paper, mimics Sridhar but in the perception stage.
    def _power(theta, c):
        """
        A power function mapping perceived angles to neural positions.

        Maps theta in [-pi, pi] to [-pi, pi] via
            f(theta) = pi * sign(theta) * (|theta| / pi)^c,
        which compresses (c > 1) or expands (c < 1) angles near the front
        relative to those near the back. Fixed points at -pi, 0, and pi.

        This mimics the transformation used in Sridhar et al. (2018) but
        applied in the perception stage rather than only the decision stage.

        Parameters
        ----------
        theta : float or array_like
            Angle(s) in [-pi, pi].
        c : float
            Exponent (c > 0). c = 1 gives the identity.
        """
        return np.pi * np.sign(theta) * (np.abs(theta) / np.pi) ** c

    @staticmethod
    def _power_inverse(y, c):
        """
        Analytical inverse of _power: find theta such that
        _power(theta, c) = y, for y in [-pi, pi].

        Because _power(theta, c) = pi * sign(theta) * (|theta|/pi)^c,
        the inverse is simply theta = pi * sign(y) * (|y|/pi)^(1/c).

        Accepts scalar or array y; always returns the same shape.

        Parameters
        ----------
        y : float or array_like
            Target value(s); each must satisfy -pi <= y <= pi.
        c : float
            Exponent parameter of _power (c > 0).

        Returns
        -------
        float or ndarray : theta value(s) satisfying _power(theta, c) = y.
        """
        y = np.asarray(y, dtype=float)
        scalar_input = y.ndim == 0
        result = np.pi * np.sign(y) * (np.abs(y) / np.pi) ** (1.0 / c)
        return result.item() if scalar_input else result


    @staticmethod
    def _subtract_interval_pair(interval, hole):
        """Subtract a single non-wrapping hole from a single non-wrapping interval.

        Both interval and hole must satisfy lo <= hi (non-wrapping).

        Parameters
        ----------
        interval : (float, float)
            Non-wrapping interval [lo, hi] with lo <= hi.
        hole : (float, float)
            Non-wrapping hole [lo, hi] with lo <= hi.

        Returns
        -------
        list of (float, float)
            Remaining non-wrapping intervals after subtraction.
        """
        a, b = interval
        h_lo, h_hi = hole
        eps = 1e-14

        # Degenerate hole (zero width)
        if h_hi - h_lo <= eps:
            return [(a, b)]

        # No overlap
        if h_hi <= a or h_lo >= b:
            return [(a, b)]
        # Full overlap
        if h_lo <= a and h_hi >= b:
            return []
        # Left bite
        if h_lo <= a and h_hi < b:
            if b - h_hi > eps:
                return [(h_hi, b)]
            return []
        # Right bite
        if h_lo > a and h_hi >= b:
            if h_lo - a > eps:
                return [(a, h_lo)]
            return []
        # Middle bite: h_lo > a and h_hi < b
        result = []
        if h_lo - a > eps:
            result.append((a, h_lo))
        if b - h_hi > eps:
            result.append((h_hi, b))
        return result

    @staticmethod
    def _unwrap_interval(interval):
        """Decompose an angular interval into non-wrapping pieces.

        If lo <= hi, the interval is already non-wrapping and returned as-is.
        If lo > hi, the interval wraps around ±pi and is split into
        (lo, pi) and (-pi, hi).

        Parameters
        ----------
        interval : (float, float)
            Angular interval (lo, hi) on [-pi, pi].

        Returns
        -------
        list of (float, float)
            One or two non-wrapping intervals with lo <= hi.
        """
        lo, hi = interval
        if lo <= hi:
            return [(lo, hi)]
        else:
            pieces = []
            if np.pi - lo > 1e-14:
                pieces.append((lo, np.pi))
            if hi - (-np.pi) > 1e-14:
                pieces.append((-np.pi, hi))
            return pieces

    @staticmethod
    def _subtract_intervals_circle(intervals, hole):
        """Subtract an angular interval (hole) from a list of angular intervals
        on the circle [-pi, pi].

        Both the input intervals and the hole may wrap around ±pi (lo > hi).
        All returned intervals are non-wrapping (lo <= hi).

        Parameters
        ----------
        intervals : list of (float, float)
            Angular intervals on [-pi, pi]. Each (lo, hi) with lo <= hi is
            the arc [lo, hi]. If lo > hi, the interval wraps around ±pi.
        hole : (float, float)
            The angular interval to subtract. May wrap around ±pi.

        Returns
        -------
        list of (float, float)
            Remaining non-wrapping intervals after subtraction.
        """
        # Decompose all input intervals into non-wrapping pieces
        unwrapped = []
        for iv in intervals:
            unwrapped.extend(PerceptionModel._unwrap_interval(iv))

        # Decompose the hole into non-wrapping pieces
        hole_pieces = PerceptionModel._unwrap_interval(hole)

        # Subtract each hole piece from each interval piece
        current = unwrapped
        for hp in hole_pieces:
            next_intervals = []
            for iv in current:
                next_intervals.extend(
                    PerceptionModel._subtract_interval_pair(iv, hp))
            current = next_intervals

        return current


    def _integrate_neural_weight(self, intervals):
        """Integrate the neural weight function over a union of angular intervals.

        Computes the exact integral of get_neural_weight(theta) over the
        given intervals. For the 'cutoff' weight, this uses the existing
        _smooth_cutoff_integral antiderivative. For uniform weight (None),
        the integral is simply the total arc length.

        Parameters
        ----------
        intervals : list of (float, float)
            Non-wrapping intervals [lo, hi] with lo <= hi, all in [-pi, pi].

        Returns
        -------
        float
            The integral value. A shared constant factor (from the
            _smooth_cutoff_integral normalization) is present but cancels
            in rho = G / G.sum(), so the result is suitable for relative
            group size computation.
        """
        if not intervals:
            return 0.0

        if self.neural_weight is None:
            # Uniform weight: integral is arc length
            return sum(hi - lo for lo, hi in intervals)
        elif self.neural_weight == 'cutoff':
            # Use _smooth_cutoff_integral_scalar as antiderivative F(theta).
            # F(theta) = norm * integral_0^theta cutoff(x) dx
            # where norm = 2*pi/(a+b). The constant cancels in
            # rho = G/G.sum(), so we use F(hi) - F(lo) directly.
            # Call the scalar kernel directly to avoid np.vectorize overhead.
            F = self._smooth_cutoff_integral_scalar
            a, b = self.a, self.b
            return sum(F(hi, a, b) - F(lo, a, b) for lo, hi in intervals)
        else:
            raise NotImplementedError("Unknown neural weight function name.")


    def get_neural_weight(self, theta):
        '''Returns the neural weight for given angles theta based on the
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
            return self._smooth_cutoff(theta, self.a, self.b)
        # elif self.neural_weight == 'tanh_plus':
        #     return self._tanh_plus(theta, self.a, self.b)
        else:
            raise NotImplementedError("Unknown neural weight function name.")
        

    def get_neural_angle(self, theta):
        '''Returns the neural position for a given angle theta based on the 
        neural position transformation function. This is a mapping between the 
        perceived center of each target and the neural position of the 
        corresponding spin group. Uses an integral of the neural weight or a 
        smooth power function or returns identity.

        Parameters
        ----------
        theta : float or 1D ndarray
            angle(s) to evaluate the neural position transformation at

        Returns
        -------
        neural position(s) corresponding to input theta value(s)
        '''

        if self.neural_angle is None:
            return theta
        elif self.neural_angle == 'integral' and self.neural_weight is None:
            return theta
        elif self.neural_angle == 'integral' and self.neural_weight == 'cutoff':
            return self._smooth_cutoff_integral(theta, self.a, self.b)
        elif self.neural_angle == 'power':
            return self._power(theta, self.c)
        else:
            raise NotImplementedError("Unknown neural position function name.")
        

    def get_neural_angle_inverse(self, theta):
        '''Returns the angle corresponding to a given neural position theta based on the 
        inverse of the neural position transformation function. This is a mapping 
        from neural position back to perceived center of each target. Uses an 
        integral of the neural weight or a smooth power function or returns identity.

        Parameters
        ----------
        theta : float or 1D ndarray
            neural position(s) to evaluate the inverse transformation at

        Returns
        -------
        angle(s) corresponding to input neural position value(s)
        '''

        if self.neural_angle is None:
            return theta
        elif self.neural_angle == 'integral' and self.neural_weight is None:
            return theta
        elif self.neural_angle == 'integral' and self.neural_weight == 'cutoff':
            return self._smooth_cutoff_int_inverse(theta, self.a, self.b)
        elif self.neural_angle == 'power':
            return self._power_inverse(theta, self.c)
        else:
            raise NotImplementedError("Unknown neural position function name.")


    def _get_target_signals(self, focal_angle=None, focal_loc=None, mesh_signal=False):
        '''Returns the egocentric angular location of the center of each VISIBLE
        target (closer targets that are not delta functions block ones behind) as
        a length N array, and a normalized neural group size (rho) for each.

        For circle (and segment) targets, blocking and neural group sizes are
        computed using exact interval arithmetic rather than a discrete mesh.

        If mesh_signal is True, instead returns the neural weighting
        (including attractiveness) for each target evaluated on the theta mesh,
        as an Nxlen(theta_mesh) array. This is used for plotting the perception
        signal (see plot_blocked_signals).

        If all targets are fully blocked or have zero neural weight, returns
        empty arrays (length-0 c_angles and rho).

        Parameters
        ----------
        focal_angle : float, optional
            the focal angle for egocentric perception. If None, uses the object's
            focal_angle attribute.
        focal_loc : array-like, optional
            the (x,y) focal location for egocentric perception. If None, uses the
            object's focal_loc attribute.
        mesh_signal : bool
            if True, return the neural weighting for each target on the theta
            mesh (for plotting). Otherwise return integrated group sizes (rho).

        Returns
        -------
        c_angles : length N ndarray
            angles to the visual centers of visible targets
        rho : length N ndarray (mesh_signal=False)
            normalized neural group size for each visible target, sums to 1.
        signals : Nxlen(theta_mesh) ndarray (mesh_signal=True)
            neural weighting times attractiveness on the theta mesh.
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

        if self.targets.geom_name is None:
            ##### Delta function targets (no blocking) #####
            # Each target is a single angle; evaluate neural weight there directly.
            s_values = self.targets.values
            G = self.get_neural_weight(angles) * s_values

            if mesh_signal:
                theta_supp = np.zeros((angles.shape[0], self.theta_mesh.size))
                for n, theta in enumerate(angles):
                    idx = np.searchsorted(self.theta_mesh, theta)
                    if idx == len(self.theta_mesh):
                        idx = 0
                    elif idx != 0 and \
                    theta-self.theta_mesh[idx-1] < self.theta_mesh[idx]-theta:
                        idx = idx - 1
                    theta_supp[n, idx] = 1
                weighted_signals = theta_supp*self.get_neural_weight(self.theta_mesh)
                return c_angles, weighted_signals*s_values[:, np.newaxis]
            else:
                G_total = G.sum()
                if G_total == 0:
                    return np.array([]), np.array([])
                return c_angles, G/G_total

        elif self.targets.geom_name == 'circle' or self.targets.geom_name == 'segment':
            ##### Extended targets: exact interval arithmetic #####
            # Sort by distance (closest first for blocking priority).
            # TODO: Andy points out that you can have two line segments where
            # the one with the farther center occludes the one with the closer
            # center. This sort-by-center-distance is correct for circles but
            # not for segments in general.
            arg_srt = dists.argsort()
            angles_sorted = angles[arg_srt]
            c_angles_sorted = c_angles[arg_srt]

            # Build visible intervals for each target using interval subtraction.
            # Each target starts with its full angular extent, then all closer
            # targets' original extents are subtracted (they block it).
            num_targets = len(arg_srt)
            # original_extents[n] = (lo, hi) as returned by get_percep_angles
            original_extents = [(float(angles_sorted[n, 0]),
                                 float(angles_sorted[n, 1]))
                                for n in range(num_targets)]
            visible_intervals = []  # list of lists of (lo, hi) tuples
            for n in range(num_targets):
                intervals = [original_extents[n]]
                for closer in range(n):
                    intervals = self._subtract_intervals_circle(
                        intervals, original_extents[closer])
                    if not intervals:
                        break
                visible_intervals.append(intervals)

            # Compute neural group sizes by integrating neural weight over
            # visible intervals, weighted by target attractiveness.
            G_sorted = np.empty(num_targets)
            for n in range(num_targets):
                G_sorted[n] = self._integrate_neural_weight(
                    visible_intervals[n]) * self.targets.values[arg_srt[n]]

            # Undo sorting to restore original target order
            inv_arg_srt = np.empty_like(arg_srt)
            inv_arg_srt[arg_srt] = np.arange(num_targets)
            G = G_sorted[inv_arg_srt]
            c_angles = c_angles_sorted[inv_arg_srt]
            visible_intervals_orig = [visible_intervals[inv_arg_srt[n]]
                                      for n in range(num_targets)]

            # Remove completely blocked targets (G == 0)
            vis = G > 0
            c_angles = c_angles[vis]
            s_values = self.targets.values[vis]
            G = G[vis]
            visible_intervals_vis = [visible_intervals_orig[n]
                                     for n in range(num_targets) if vis[n]]

            if mesh_signal:
                # Build mesh representation from exact intervals for plotting
                theta_supp = np.zeros((len(c_angles), self.theta_mesh.size))
                for n, ivs in enumerate(visible_intervals_vis):
                    for lo, hi in ivs:
                        mask = (self.theta_mesh >= lo) & (self.theta_mesh <= hi)
                        theta_supp[n, mask] = 1
                weighted_signals = theta_supp*self.get_neural_weight(self.theta_mesh)
                return c_angles, weighted_signals*s_values[:, np.newaxis]
            else:
                G_total = G.sum()
                if G_total == 0:
                    return np.array([]), np.array([])
                return c_angles, G/G_total

        else:
            raise NotImplementedError("Unknown target geometry name.")


    def plot_blocked_signals(self, wb_plot=False):
        '''Plots visible targets, their angular direction from the observer, 
        their associated neural angles, and also the signal distribution from 
        the point of view of the observer.

        Use as a test for _get_target_signals and get_neural_angle.
        
        Set wb_plot to True if plotting in a Jupyber notebook
        '''

        vis_angles, signals = self._get_target_signals(mesh_signal=True)
        neur_angles = self.get_neural_angle(vis_angles)

        if wb_plot:
            plt.figure(figsize=(8,5))
        else:
            plt.figure(figsize=(12,6))

        ###### Target Geometry Plot ######
        ax1 = plt.subplot(121)

        # First, plot the targets themselves
        self.targets.plot_targets_to_axis(ax1)

        # Now plot perception lines. Requires adding back angle of focal locust
        #   to get allocentric angles.
        neur_angles_allo = neur_angles + self.focal_angle
        for n, theta in enumerate(vis_angles+self.focal_angle):
            r = self.targets.get_dist_to_targets(self.focal_loc)[n]
            x = (self.focal_loc[0],self.focal_loc[0] + r*np.cos(theta))
            y = (self.focal_loc[1],self.focal_loc[1] + r*np.sin(theta))
            ax1.plot(x,y,'k')
            x_neur = (self.focal_loc[0],self.focal_loc[0] + r*np.cos(neur_angles_allo[n]))
            y_neur = (self.focal_loc[1],self.focal_loc[1] + r*np.sin(neur_angles_allo[n]))
            ax1.plot(x_neur, y_neur, 'r--', alpha=0.5)
        ax1.arrow(self.focal_loc[0],self.focal_loc[1],
                  0.5*np.cos(self.focal_angle),0.5*np.sin(self.focal_angle), 
                width=0.1, head_length=0.25)
        ax1.set_aspect('equal')
        ax1.set_title('Target Geometry and\n Neural Directions')

        ###### Perception Signal Plot ######
        ax2 = plt.subplot(122, projection='polar')

        p_func = signals.sum(axis=0)

        ax2.plot(self.theta_mesh,p_func)
        ax2.arrow(0,-0.5,0,0.25, width=0.2, head_length=0.15)
        ax2.set_rmin(-0.5)
        ax2.set_rmax(1)
        ax2.set_rlabel_position(10)
        ax2.set_rticks([0, 0.5, 1])
        # Define positions in degrees (0 to 360)
        angles_deg = np.linspace(0, 360, 8, endpoint=False)
        # Define corresponding labels in radians (0 to 2π)
        labels = [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
                r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']

        ax2.set_thetagrids(angles_deg, labels)
        ax2.set_title('Perception Signal')

        plt.show()


    def get_neural_signals(self, focal_angle=None, focal_loc=None):
        '''Returns the neural angles and neural group sizes for each target based 
        on the perceived angles and the neural position transformation function. 
        This is a mapping from perceived center of each target to the neural 
        position of the corresponding spin group.

        Parameters
        ----------
        focal_angle : float, optional
            the focal angle for egocentric perception. If None, uses the object's 
            focal_angle attribute.
        focal_loc : array-like, optional
            the (x,y) focal location for egocentric perception. If None, uses the 
            object's focal_loc attribute.

        Returns
        -------
        neural_angles : length N ndarray
            neural angles corresponding to visible targets
        rho : length N ndarray
            normalized neural group size for each visible target
        '''

        if focal_angle is None:
            focal_angle = self.focal_angle
        if focal_loc is None:
            focal_loc = self.focal_loc

        angles, rho = self._get_target_signals(focal_angle=focal_angle, 
                                              focal_loc=focal_loc)
        neural_angles = self.get_neural_angle(angles)

        return neural_angles, rho



class NeuralBandModel:
    '''This class takes in a PerceptionModel and uses the neural angles and group 
    sizes to chose a consensus neural direction based on a discrete Ising 
    Hamiltonian. It then translates this into a consensus physical direction. 
    Relies on get_neural_signals from the PerceptionModel to obtain 
    neural angles and relative neural group size.'''

    def __init__(self, percep_model=None, T=0.2, K=1):
        '''From a PerceptionModel with its Targets object, establishes a model 
        for chosing direction based on discrete Ising. Relies on get_neural_signals 
        from the PerceptionModel to obtain neural angles and relative neural 
        group size.
        
        Parameters
        ----------
        percep_model : PerceptionModel
            A PerceptionModel object (with its Targets object) that establishes 
            the geometry of the scenario. If none is provided, a default one 
            will be created. The PerceptionModel can be updated to obtain 
            consensus directions for different layouts or focal locations/angles.
        T : float
            Temperature for the neural ring representing amount of noise in the system.
        K : float
            Coupling strength for Kuramoto turning speed. Used in models of 
            walkers only.
        '''

        self.T = T
        self.K = K

        if percep_model is None:
            self.percep_model = PerceptionModel()
        else:
            assert isinstance(percep_model,PerceptionModel),\
            "percep_model must be a PerceptionModel object."
            self.percep_model = percep_model

        # Initial gamma value based on the focal angle of the perception model, 
        #   with a small coherence strength of 1e-5.
        self.gamma = 1e-5*np.exp(1j*self.percep_model.focal_angle)

        # Random number generator for certain processes within the class;
        #   Can seed here for reproducability.
        #seed = 3
        self.rng = np.random.default_rng()


    def dgamma_dt(self, t=None, gamma=None, focal_angle=None, focal_loc=None):
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
        focal_angle : float, optional
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
            gamma = self.gamma
        Theta = np.angle(gamma)
        R = np.abs(gamma)

        neur_angles, rho = self.percep_model.get_neural_signals(focal_angle, focal_loc)
        if neur_angles.size == 0:
            return -gamma
        
        # Compute the sum over all target angles.
        # Suppress overflow warnings: they indicate R is too large and are 
        #   likely the result of whatever numerical algorithm is being used
        #   trying out a bad R value.
        with np.errstate(over='ignore'):
            summands = rho*np.exp(1j*neur_angles)/(1+np.exp(
                -2*neur_angles.size*R*np.cos(neur_angles-Theta)/self.T))
        
        return np.sum(summands) - gamma


    def dgamma_dt_vec(self, gamma_vec, focal_angle=None, focal_loc=None):
        '''Wrapper around dgamma_dt for use in root finding for equilibria.
        
        Here, gamma_vec is a length 2 ndarray of real and imaginary parts 
        representing the complex coherence value of the neural band.

        Parameters
        ----------
        gamma_vec : length 2 ndarray of float
            current (complex) coherence value of the neural band.
        focal_angle : float, optional
            the current heading (angle) of the observer. If None, use 
            self.percep_model.focal_angle.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use self.percep_model.focal_loc.

        Returns
        -------
        dgamma_dt_vec : length 2 ndarray of float
            complex time derivative of gamma according to the Ising model
        '''
        dgamma = self.dgamma_dt(None, gamma_vec[0] + 1j*gamma_vec[1], 
                                focal_angle, focal_loc)
        return np.array([dgamma.real, dgamma.imag])
    

    def _self_consistent_eq(self, x, focal_loc):
        '''System of equations for finding self-consistent equilibria.

        At a self-consistent equilibrium, the observer is facing the consensus
        direction (heading = allocentric consensus), which means the egocentric
        consensus angle is zero, and therefore Theta_neural = 0 (gamma is real
        positive). We solve dgamma_dt(R + 0j, theta, focal_loc) = 0 for both
        theta (the allocentric heading/consensus direction) and R (the coherence
        strength).

        Parameters
        ----------
        x : length 2 ndarray
            [theta, R] where theta is the allocentric heading and R is the
            coherence strength.
        focal_loc : array-like of length 2
            (x,y) location of the observer.

        Returns
        -------
        length 2 ndarray : [Re(dgamma_dt), Im(dgamma_dt)]
        '''
        theta, R = x
        gamma = R + 0j
        dg = self.dgamma_dt(gamma=gamma, focal_angle=theta, focal_loc=focal_loc)
        return np.array([dg.real, dg.imag])

    def gamma_equilib(self, focal_angle=None, focal_loc=None):
        '''Attempt to find all zeros (equilibria) of dgamma/dt for a given focal
        location by using a multistart root finding algorithm on a mesh of
        initial angle values.

        If focal_angle is supplied or None, the initial angle values will
        correspond to the argument of the initial gamma value with coherence
        strength of 0.5. If focal_angle is True, find self-consistent equilibria
        where the observer's heading equals the allocentric consensus direction.
        This is what you want to use for creating a bifurcation diagram in
        x,y-space because it finds the physically meaningful equilibria of the
        coupled heading-consensus system.

        Self-consistent equilibria have gamma = R + 0j (real positive), since
        the egocentric consensus angle is zero when heading equals consensus.
        The method sweeps over candidate allocentric headings and solves
        dgamma_dt(R + 0j, theta, focal_loc) = 0 for theta and R simultaneously.

        Returns a list of unique equilibrium gamma values found if focal_angle
        is supplied or None, otherwise returns a list of allocentric equilibrium
        angles found. Also returns a list of booleans indicating whether each
        equilibrium is stable (True) or unstable (False) based on an analytical
        perturbation analysis.

        Parameters
        ----------
        focal_angle : float or bool, optional
            The current heading (angle) of the observer.
            - If None, use self.percep_model.focal_angle.
            - If True, find self-consistent equilibria across all headings.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use self.percep_model.focal_loc.

        Returns
        -------
        gamma_eqs : list of complex OR angle_eqs : list of float
            list of equilibrium gamma values or allocentric equilibrium angles
        stability : list of bool
            for each equilibrium gamma value, whether it is stable (True) or
            unstable (False)
        '''
        if focal_angle is False:
            focal_angle = None
        if focal_angle is True:
            # Find self-consistent equilibria: heading = consensus direction.
            # gamma = R + 0j, solve dgamma_dt(R+0j, theta, focal_loc) = 0.
            #
            # Strategy: scan Im(dgamma_dt) across theta at a probe R to find
            # sign changes, use brentq to pin down candidate theta values,
            # then polish each candidate with the 2D hybr root finder.
            # The Jacobian is near-block-diagonal (Im depends mostly on theta,
            # Re depends mostly on R), so the Im scan seeds theta well and
            # hybr converges reliably from there.

            theta_mesh = np.linspace(-np.pi, np.pi, 100)
            final_angles = []
            stability = []
            R_probe = 0.5

            # Collect candidate theta values from sign changes in Im
            candidates = []
            imag_vals = np.array([self.dgamma_dt(
                gamma=R_probe+0j, focal_angle=t,
                focal_loc=focal_loc).imag for t in theta_mesh])
            for i in range(len(imag_vals)-1):
                if imag_vals[i]*imag_vals[i+1] < 0:
                    try:
                        theta_c = brentq(
                            lambda t: self.dgamma_dt(
                                gamma=R_probe+0j, focal_angle=t,
                                focal_loc=focal_loc).imag,
                            theta_mesh[i], theta_mesh[i+1])
                        candidates.append(theta_c)
                    except ValueError:
                        pass
            # Always include theta=0 and theta=+-pi as candidates since
            # Im(dgamma_dt) is often zero there by symmetry but the sign
            # change can be narrower than the mesh spacing.
            for theta_extra in [0.0, np.pi, -np.pi]:
                candidates.append(theta_extra)

            # Polish each candidate with the 2D root finder
            for theta_c in candidates:
                sol = root(self._self_consistent_eq, [theta_c, R_probe],
                           args=(focal_loc,), method='hybr', tol=1e-10)
                if not sol.success:
                    continue
                theta_eq = convert_angles(sol.x[0])
                R_eq = sol.x[1]

                if R_eq < 0.01 or R_eq > 1.0:
                    continue

                # Verify residual is small
                residual = self.dgamma_dt(gamma=R_eq+0j,
                                          focal_angle=theta_eq,
                                          focal_loc=focal_loc)
                if np.abs(residual) > 1e-6:
                    continue

                # Check if close to any existing solution
                close_check = False
                for existing_angle in final_angles:
                    angle_diff = np.abs(convert_angles(
                        theta_eq - existing_angle))
                    if angle_diff < 0.02:
                        close_check = True
                        break
                if not close_check:
                    final_angles.append(theta_eq)
                    gamma_eq = R_eq + 0j
                    stability.append(self._discrim_A(
                        gamma_eq, theta_eq, focal_loc))

            return final_angles, stability
        else:
            # Fixed focal_angle: find gamma equilibria in the neural plane.
            if focal_angle is None:
                focal_angle = self.percep_model.focal_angle
            init_angles = np.linspace(-np.pi, np.pi-0.01)
            final_gammas = []
            stability = []
            for angle in init_angles:
                init_gamma = np.array([0.5*np.cos(angle), 0.5*np.sin(angle)])
                sol = root(self.dgamma_dt_vec, init_gamma,
                           args=(focal_angle, focal_loc),
                           method='hybr', tol=1e-7)
                if sol.success:
                    gamma_eq = sol.x[0] + 1j*sol.x[1]
                    close_check = False
                    for existing_gamma in final_gammas:
                        if np.abs(gamma_eq - existing_gamma) < 0.01:
                            close_check = True
                            break
                    if not close_check:
                        final_gammas.append(gamma_eq)
                        stability.append(self._discrim_A(
                            gamma_eq, focal_angle, focal_loc))
            return final_gammas, stability
    

    def convert_gamma(self, gamma):
        '''Convert a given neural gamma value to an egocentric physical coherence 
        angle and coherence strength.

        Parameters
        ----------
        gamma : complex or array of complex
            coherence value of the neural band

        Returns
        -------
        angle : float or array of float
            physical angle corresponding to the input gamma value
        R : float or array of float
            coherence strength corresponding to the input gamma value
        '''
        Theta = np.angle(gamma)
        R = np.abs(gamma)
        return self.percep_model.get_neural_angle_inverse(Theta), R
        

    def _discrim_A(self, gamma_star, focal_angle, focal_loc):
        '''Determines stability of equilibria based on perturbation analysis.
        Assumes that the focal_angle is the angle of gamma_star and calculates 
        the A value of the linear coefficient. If A < 1, the equilibrium is stable, 
        if A > 1, the equilibrium is unstable.

        This is based on a cosine kernel with nu exponent.

        Parameters
        ----------
        gamma_star : complex
            equilibrium gamma value at which to compute the Jacobian
        focal_angle : float
            the current heading (angle) of the observer. If None, use 
            self.percep_model.focal_angle.
        focal_loc : array-like of length 2
            (x,y) location of the observer

        Returns
        -------
        True if stable, False if unstable
        '''

        Theta = np.angle(gamma_star)
        R = np.abs(gamma_star)
        neur_angles, rho = self.percep_model.get_neural_signals(focal_angle, focal_loc)
        k = rho.size
        with np.errstate(over='ignore'):
            summands = ((rho/np.cosh(k*R*np.cos(Theta-neur_angles)/self.T)**2)
                        *np.sin(Theta-neur_angles)**2)
            A = k*summands.sum()/(2*self.T)
        return A < 1
    

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
        thetas : array of float
            array of equilibrium angles found at this mesh point
        Rs : array of float
            array of coherence strengths corresponding to the equilibrium angles
        stability : list of bool
            list indicating whether each equilibrium is stable (True) or unstable (False)
        '''

        ii, jj, X, Y = args
        focal_loc = np.array([X[jj,ii], Y[jj,ii]])

        final_angles, stability = self.gamma_equilib(focal_angle=True, focal_loc=focal_loc)

        return ii, jj, final_angles, stability
    

    def plot_direction_mesh(self, xlim=(0,6), num_x=19, ylim=(-3.5,3.5), num_y=19, 
                            pool=None, ax=None, wb_plot=False):
        '''Create a mesh of starting locations and, for each point in the mesh, 
        find the equilibria of dgamma/dt and plot the corresponding consensus 
        directions.

        Set wb_plot to True if plotting in a Jupyter notebook

        Parameters
        ----------
        xlim : (xmin,xmax) tuple of floats
            x limits for mesh, inclusive
        num_x : number of steps in x direction
        ylim : (ymin,ymax) tuple of floats
            y limits for mesh, inclusive
        num_y : number of steps in y direction
        pool : multiprocessing.Pool, optional
            If provided, use this pool to parallelize the solving of the ODEs.
        axis : matplotlib axis, optional
            If provided, plot on this axis instead of creating a new figure and axis.
        wb_plot : bool
            whether or not plotting in a Jupyter notebook

        Returns
        -------
        ax : matplotlib axis, if axis was provided as an argument. Otherwise, None.
        '''

        # create mesh of focal locations
        xmesh = np.linspace(xlim[0], xlim[1], num_x)
        ymesh = np.linspace(ylim[0], ylim[1], num_y)

        X, Y = np.meshgrid(xmesh, ymesh)
        
        
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

        # plot the vector field
        if ax is None:
            local_plot = True
            if wb_plot:
                fig = plt.figure(figsize=(12,6))
            else:
                fig = plt.figure(figsize=(5.5,5))
        else:
            local_plot = False

        # List of x- and y-components of unit vectors for a list of solution arrays 
        #   corresponding to solutions found at each mesh point
        U_list = []; V_list = []
        # Record stability of each equilibrium solution
        stability_list = []
        # boolean mesh for multiple solutions
        multi_sol = np.full(X.shape, False, dtype=bool)
        # List of arrays of equilibrium angles for each solution found at each mesh point
        multi_thetas = []
        for result in results:
            ii, jj, thetas, stabilities = result
            for n, (theta, stable) in enumerate(zip(thetas, stabilities)):
                if len(multi_thetas) < n+1:
                    multi_thetas.append(np.zeros(X.shape))
                    U_list.append(np.zeros(X.shape))
                    V_list.append(np.zeros(X.shape))
                    stability_list.append(np.full(X.shape, False, dtype=bool))
                multi_thetas[n][jj,ii] = theta
                U_list[n][jj,ii] = np.cos(theta)
                V_list[n][jj,ii] = np.sin(theta)
                stability_list[n][jj,ii] = stable
            if len(thetas) > 1:
                multi_sol[jj,ii] = True

        if local_plot:
            ax = plt.subplot(1,1,1)

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
        ax.set_title('New model equilibrium plot')
        if local_plot:
            fig.legend(loc='outside center right')
            plt.show()
        else:
            return ax



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
        for chosing direction based on discrete Ising. Relies on get_neural_signals 
        from the PerceptionModel to obtain neural angles and relative neural 
        group size.
        
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
            Exponent for cosine weighting kernel in Sridhar et al. (2018). 
            Higher values lead to sharper peaks. This should be 1 unless you are 
            running a simulation to recover their results, in which case set it 
            to 0.5 and use a flat neural weight in the PerceptionModel.
        '''

        self.T = T
        self.K = K
        self.nu = nu

        if percep_model is None:
            self.percep_model = PerceptionModel()
        else:
            assert isinstance(percep_model,PerceptionModel),\
            "percep_model must be a PerceptionModel object."
            self.percep_model = percep_model

        # Initial gamma value based on the focal angle of the perception model, 
        #   with a small coherence strength of 1e-5.
        self.gamma = 1e-5*np.exp(1j*self.percep_model.focal_angle)

        # Random number generator for certain processes within the class;
        #   Can seed here for reproducability.
        #seed = 3
        self.rng = np.random.default_rng()


    def cosine(self, x):
        '''Function that returns cos(pi*(x/pi)^nu).'''
        
        if self.nu == 1:
            return np.cos(x)
        else:
            return np.cos(np.pi*(np.abs(x)/np.pi)**self.nu)


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


    def dgamma_dt(self, t=None, gamma=None, focal_angle=None, focal_loc=None):
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
            current coherence value of the neural band. If None, use self.gamma.
        focal_angle : float, optional
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
            gamma = self.gamma
        Theta = np.angle(gamma)
        R = np.abs(gamma)
        if focal_angle is None:
            focal_angle = self.percep_model.focal_angle

        angles_rel, signals = self.percep_model.get_neural_signals(focal_angle, focal_loc)
        if angles_rel.size == 0:
            return -gamma
        # The angles recieved above are relative to focal_angle, i.e., the 
        #   angle between the polar location of each target and focal_angle.
        # Convert to allocentric polar angles.
        angles = convert_angles(angles_rel+focal_angle)
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
    

    def dgamma_dt_vec(self, gamma_vec, focal_angle=None, focal_loc=None):
        '''Wrapper around dgamma_dt for use in root finding for equilibria.
        
        Here, gamma_vec is a length 2 ndarray of real and imaginary parts 
        representing the complex coherence value of the neural band.

        Parameters
        ----------
        gamma_vec : length 2 ndarray of float
            current (complex) coherence value of the neural band.
        focal_angle : float, optional
            the current heading (angle) of the observer. If None, use 
            self.percep_model.focal_angle.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use self.percep_model.focal_loc.

        Returns
        -------
        dgamma_dt_vec : length 2 ndarray of float
            complex time derivative of gamma according to the Ising model
        '''
        dgamma = self.dgamma_dt(None, gamma_vec[0] + 1j*gamma_vec[1], 
                                focal_angle, focal_loc)
        return np.array([dgamma.real, dgamma.imag])
    

    def run_dgamma_dt(self, focal_angle=None, focal_loc=None, init_gamma=None, 
                      t_Final=30):
        '''Solve the dgamma/dt equation and look for solutions to approach a 
        stable equilibrium for the neural ring model. Uses RK45 solver from scipy. 
        
        init_gamma is the initial coherence value to start from. If None, 
        it will start from self.gamma.

        Parameters
        ----------
        focal_angle : float, optional
            the current heading (angle) of the observer. If None, use 
            self.percep_model.focal_angle.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use the percep_model's 
            focal_loc.
        init_gamma : complex float, optional
            initial coherence value. If None, use the model's current gamma value.
        t_Final : float, optional
            final time for integration.

        Returns
        -------
        gamma_equilib : complex float
            Equilibrium gamma value reached by integration.
        '''

        if init_gamma is None:
            init_gamma = self.gamma

        if np.isscalar(init_gamma):
            init_gamma = np.array([init_gamma])
        # to stop solving for gamma when sufficiently close to equilibrium we:
        # 1) initially solve for 5 unit of time
        #    (Chose 5 by trial and error, seems to balance overshooting vs. looping, at which python is slow)
        # 2) check whether the derivative is bigger than tolerance tol
        # 3) repeat for 1 unit of time, checking the size of the derivative each time
        sol = solve_ivp(self.dgamma_dt, [0, 5], init_gamma, 
                        args=(focal_angle, focal_loc))
        tol = 1e-4
        T = 1
        while np.abs(sol.y[0,-1]-sol.y[0,-2])/(sol.t[-1]-sol.t[-2]) > tol:
            sol = solve_ivp(self.dgamma_dt, [0, 1], sol.y[:,-1], 
                        args=(focal_angle, focal_loc))
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
            run_dgamma_dt will be self.gamma (the last gamma value found 
            in this function) in this case. self.gamma will be updated to the 
            final gamma value found. If False, do not use self.gamma as the IC 
            and do not update self.gamma with the final value found. Instead, 
            the IC will be 0.1*exp(-1j*theta). This is useful for basing the 
            result on a specific theta value rather than the current state of 
            the neural ring.
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
            self.gamma = self.run_dgamma_dt(focal_angle=theta, focal_loc=focal_loc, 
                                            init_gamma=gamma, t_Final=t_Final)
            gamma = self.gamma
        elif gamma is False:
            gamma = self.run_dgamma_dt(focal_angle=theta, focal_loc=focal_loc, 
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
    

    def gamma_equilib(self, focal_angle=None, focal_loc=None):
        '''Attempt to find all zeros (equilibria) of dgamma/dt for a given focal 
        location as focal angle is varied.
        
        Uses a multistart root finding starting from a mesh of points on the 
        circle of radius 0.5. Returns a list of unique equilibrium gamma values 
        found.

        Parameters
        ----------
        focal_angle : float or bool, optional
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
        if focal_angle is False:
            focal_angle = None
        if focal_angle is True:
            # set a persistant flag
            get_focal_angle = True
        else:
            get_focal_angle = False

        init_angles = np.linspace(-np.pi, np.pi-0.01)
        init_gamma_vecs = np.zeros((init_angles.size, 2), dtype=np.double)
        init_gamma_vecs[:,0] = 0.5*np.cos(init_angles)
        init_gamma_vecs[:,1] = 0.5*np.sin(init_angles)
        final_gammas = []
        for init_val in init_gamma_vecs:
            if get_focal_angle is True:
                focal_angle = np.angle(init_val[0] + 1j*init_val[1])
            sol = root(self.dgamma_dt_vec, init_val, args=(focal_angle, focal_loc),
                       method='hybr', tol=1e-7)
            if sol.success:
                gamma_eq = sol.x[0] + 1j*sol.x[1]
                # When sweeping focal_angle (self-consistent mode), verify
                # that the found gamma is actually a self-consistent
                # equilibrium: dgamma_dt must be zero when focal_angle
                # equals the allocentric consensus direction angle(gamma).
                if get_focal_angle:
                    residual = self.dgamma_dt(
                        gamma=gamma_eq,
                        focal_angle=np.angle(gamma_eq),
                        focal_loc=focal_loc)
                    if np.abs(residual) > 1e-6:
                        continue
                # Check if close to any existing solution
                close_check = False
                for existing_gamma in final_gammas:
                    if np.abs(gamma_eq - existing_gamma) < 0.01:
                        close_check = True
                        break
                if not close_check:
                    final_gammas.append(gamma_eq)
        return final_gammas
    

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

        final_gammas = self.gamma_equilib(focal_angle=True, focal_loc=focal_loc)

        return ii, jj, final_gammas
    

    def _discrim_A_nu(self, gamma_star, focal_loc):
        '''Determines stability of equilibria based on perturbation analysis.
        Assumes that the focal_angle is the angle of gamma_star and calculates 
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
        neur_angles, rho = self.percep_model.get_neural_signals(Theta, focal_loc)
        k = rho.size
        with np.errstate(over='ignore'):
            summands = ((rho/np.cosh(k*R*self.cosine(neur_angles)/self.T)**2)
                        *np.sin(neur_angles)*self.nu*
                        np.sin(np.pi*np.sign(neur_angles)*np.abs(neur_angles/np.pi)**self.nu)*
                        np.abs(neur_angles/np.pi)**(self.nu-1))
            A = k*summands.sum()/(2*self.T)
        return A < 1
    

    def plot_direction_mesh(self, xlim=(0,6), num_x=19, ylim=(-3.5,3.5), num_y=19, 
                            pool=None, ax=None, wb_plot=False):
        '''Create a mesh of starting locations and, for each point in the mesh, 
        find the equilibria of dgamma/dt and plot the corresponding consensus 
        directions.

        Set wb_plot to True if plotting in a Jupyter notebook

        Parameters
        ----------
        xlim : (xmin,xmax) tuple of floats
            x limits for mesh, inclusive
        num_x : number of steps in x direction
        ylim : (ymin,ymax) tuple of floats
            y limits for mesh, inclusive
        num_y : number of steps in y direction
        pool : multiprocessing.Pool, optional
            If provided, use this pool to parallelize the solving of the ODEs.
        axis : matplotlib axis, optional
            If provided, plot on this axis instead of creating a new figure and axis.
        wb_plot : bool
            whether or not plotting in a Jupyter notebook

        Returns
        -------
        ax : matplotlib axis, if axis was provided as an argument. Otherwise, None.
        '''

        # create mesh of focal locations
        xmesh = np.linspace(xlim[0], xlim[1], num_x)
        ymesh = np.linspace(ylim[0], ylim[1], num_y)

        X, Y = np.meshgrid(xmesh, ymesh)
        
        
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

        # plot the vector field
        if ax is None:
            local_plot = True
            if wb_plot:
                fig = plt.figure(figsize=(12,6))
            else:
                fig = plt.figure(figsize=(5.5,5))
        else:
            local_plot = False

        # List of x- and y-components of unit vectors for a list of solution arrays 
        #   corresponding to solutions found at each mesh point
        U_list = []; V_list = []
        # Record stability of each equilibrium solution
        stability_list = []
        # boolean mesh for multiple solutions
        multi_sol = np.full(X.shape, False, dtype=bool)
        # List of arrays of equilibrium angles for each solution found at each mesh point
        multi_thetas = []
        for result in results:
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
                stability_list[n][jj,ii] = self._discrim_A_nu(gamma, focal_loc)
            if len(final_gammas) > 1:
                multi_sol[jj,ii] = True

        if local_plot:
            ax = plt.subplot(1,1,1)

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
        ax.set_title('Equilibrium plot')
        if local_plot:
            fig.legend(loc='outside center right')
            plt.show()
        else:
            return ax
        

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
                # Use an Euler (TODO: Honeycutt) step instead of solving the theta equation
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
