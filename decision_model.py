'''
Sets up a scenario in which a single locust makes decisions about the direction
it wants to go based on static targets with certain geometry
'''

import warnings
from contextlib import contextmanager

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import root, brentq
from scipy.interpolate import CubicSpline
from scipy.special import i0
from scipy.stats import vonmises
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def convert_angles(theta):
    '''Given a scalar or array of angles, convert to angles in the closed
    interval [-np.pi, np.pi].

    The map is odd: convert_angles(-x) == -convert_angles(x) for every x. At
    the endpoints this means +pi and -pi are both fixed points, so the interval
    is closed rather than half-open: they are two representations of the same
    (facing-away) direction, and which one is returned is inherited from the
    sign of the argument.

    That sign is meaningful. Facing-away is the branch cut of the heading
    torque, where dtheta/dt jumps between +K*R and -K*R; carrying the argument's
    sign through lets the caller's approach direction select the branch, and
    makes the map commute with the mirror theta -> -theta. It is the same
    convention as np.angle, which selects the branch by the sign of the
    imaginary part (including IEEE signed zero).
    '''
    wrapped = theta - (theta+np.pi)//(2*np.pi)*2*np.pi
    # The floor division lands the +pi endpoint on -pi; take the sign from theta.
    flip = (wrapped == -np.pi) & (theta > 0)
    if np.ndim(wrapped) == 0:
        return np.pi if flip else wrapped
    wrapped = np.array(wrapped, copy=True)
    wrapped[flip] = np.pi
    return wrapped


def _smallest_enclosing_arc(angles):
    """Return (lo, hi) for the shortest arc on [-pi, pi] containing all angles.

    The arc goes counter-clockwise from lo to hi. If lo > hi, the arc wraps
    around ±pi. Input angles must be in [-pi, pi].

    Parameters
    ----------
    angles : 1D array
        Angles in [-pi, pi].

    Returns
    -------
    (lo, hi) : tuple of float
    """
    s = np.sort(angles % (2*np.pi))  # sort on [0, 2pi)
    n = len(s)
    # Compute the gap between consecutive sorted angles (including wrap-around)
    gaps = np.empty(n)
    gaps[:-1] = s[1:] - s[:-1]
    gaps[-1] = s[0] + 2*np.pi - s[-1]
    # The largest gap is the one NOT covered by the arc.
    # The arc starts just after the largest gap and ends just before it.
    k = np.argmax(gaps)
    # lo is the angle just after the largest gap, hi is the angle just before it
    lo = s[(k + 1) % n]
    hi = s[k]
    return (convert_angles(lo), convert_angles(hi))


class Targets:

    def __init__(self, locs=None, geom_name=None, r=None, l=None, w=0,
                 theta=0, values=1):
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
        geom_name : {'circle','capsule'}, (optional)
            geometry of targets. Depending on the choice, additional parameters
            must be set to quantify the geometry. Options are:
            - 'circle' : must specify a radius r to be used for all targets or
            an array of radii r, one for each target. The position of the target
            is the midpoint of the circle.
            - 'capsule' : a line segment of length l with semicircular endcaps
            of radius w/2. Must specify l (scalar or array) and theta (scalar
            or array) for the orientation of each target. The position of the
            target is the midpoint of the spine. w defaults to 0 (zero-width
            line segment).
        r : float or sequence of length N
            radius of circles in the geometry; see geom_name for requirements
        l : float or sequence of length N
            spine length of capsules; see geom_name for requirements
        w : float or sequence of length N, default=0
            width (diameter of endcaps) of capsules
        theta : float or length N ndarray, default=0
            orientation of targets; see geom_name for requirements
        values : float or length N ndarray, default=1
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
        if hasattr(w,'__iter__'):
            assert len(w) == self.locs.shape[0], "length of w must match first dim of locs"
            self.w = np.array(w)
        else:
            self.w = w
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

        elif self.geom_name == 'capsule':
            ##### Capsule targets (line segment spine + semicircular endcaps) #####
            on_target_bool = self.check_target_overlap(loc)
            # Spine endpoint locations
            diff = np.column_stack([self.l/2*np.cos(self.theta),
                                    self.l/2*np.sin(self.theta)])
            endpt1 = self.locs + diff
            endpt2 = self.locs - diff
            # Work only with non-overlapping targets
            ep1 = endpt1[~on_target_bool]
            ep2 = endpt2[~on_target_bool]
            if isinstance(self.w, np.ndarray):
                hw = self.w[~on_target_bool] / 2
            else:
                hw = self.w / 2  # half-width (endcap radius)
            vecs1 = ep1 - loc
            vecs2 = ep2 - loc
            d1 = np.linalg.norm(vecs1, axis=1)
            d2 = np.linalg.norm(vecs2, axis=1)
            center_angles1 = np.arctan2(vecs1[:,1], vecs1[:,0]) - angle
            center_angles2 = np.arctan2(vecs2[:,1], vecs2[:,0]) - angle
            # Angular half-width of each endcap circle from the observer.
            # Clamp hw/d to 1 to handle observer very close to an endpoint.
            half1 = np.arcsin(np.minimum(hw / np.maximum(d1, 1e-15), 1.0))
            half2 = np.arcsin(np.minimum(hw / np.maximum(d2, 1e-15), 1.0))
            # Four candidate tangent angles (egocentric)
            t1lo = convert_angles(center_angles1 - half1)
            t1hi = convert_angles(center_angles1 + half1)
            t2lo = convert_angles(center_angles2 - half2)
            t2hi = convert_angles(center_angles2 + half2)
            # The capsule extent is the shortest arc containing all 4 tangent
            # angles. For each target, find (lo, hi) giving the smallest arc.
            num = len(d1)
            target_angles = np.zeros((num, 2))
            for n in range(num):
                pts = np.array([t1lo[n], t1hi[n], t2lo[n], t2hi[n]])
                target_angles[n] = _smallest_enclosing_arc(pts)
            if not np.any(on_target_bool):
                return target_angles
            else:
                angle_to_targets = np.zeros(self.locs.shape)
                angle_to_targets[on_target_bool] = np.array([-np.pi, np.pi])
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
        elif self.geom_name == 'capsule':
            seg_vec = np.column_stack([self.l/2*np.cos(self.theta),
                                       self.l/2*np.sin(self.theta)])
            spine_dist = self.closest_dist_btwn_lines_and_pt(
                self.locs - seg_vec, self.locs + seg_vec, loc)
            return np.maximum(spine_dist - self.w / 2, 0.0)


    def check_target_overlap(self,loc):
        '''Check to see if loc overlaps with any of the targets. Return a bool
        array of length N (where N is the number of targets) indicating overlap.
        '''
        eps = np.finfo(np.float32).eps

        if self.geom_name is None:
            return (self.locs == loc).all(axis=1)
        elif self.geom_name == 'circle':
            return np.linalg.norm(self.locs - loc,axis=1) <= self.r
        elif self.geom_name == 'capsule':
            # Observer is inside the capsule if distance to spine <= w/2
            seg_vec = np.column_stack([self.l/2*np.cos(self.theta),
                                       self.l/2*np.sin(self.theta)])
            spine_dist = self.closest_dist_btwn_lines_and_pt(
                self.locs - seg_vec, self.locs + seg_vec, loc)
            return spine_dist <= self.w / 2
    

    def check_trajectory_intersection(self, old_loc, new_loc):
        '''Check if the line segment from old_loc to new_loc passes through
        any target. Returns a bool array of length N.

        Parameters
        ----------
        old_loc : 1D ndarray of length 2
            Start of the trajectory segment
        new_loc : 1D ndarray of length 2
            End of the trajectory segment

        Returns
        -------
        length N ndarray of bool
        '''
        n_targets = len(self.locs)

        if self.geom_name is None:
            return np.zeros(n_targets, dtype=bool)

        elif self.geom_name == 'circle':
            d = new_loc - old_loc
            seg_len2 = np.dot(d, d)
            if seg_len2 < 1e-24:
                return np.zeros(n_targets, dtype=bool)
            f = old_loc - self.locs  # (N, 2)
            t = np.clip(-np.sum(f * d, axis=1) / seg_len2, 0, 1)
            proj = old_loc + np.outer(t, d)  # (N, 2)
            dist = np.linalg.norm(self.locs - proj, axis=1)
            return dist <= self.r

        elif self.geom_name == 'capsule':
            seg_vec = np.column_stack([self.l/2*np.cos(self.theta),
                                       self.l/2*np.sin(self.theta)])
            spine_start = self.locs - seg_vec
            spine_end = self.locs + seg_vec
            if isinstance(self.w, np.ndarray):
                half_w = self.w / 2
            else:
                half_w = np.full(n_targets, self.w / 2)
            result = np.zeros(n_targets, dtype=bool)
            for i in range(n_targets):
                dist = self._min_dist_segments(old_loc, new_loc,
                                               spine_start[i], spine_end[i])
                result[i] = dist <= half_w[i]
            return result


    @staticmethod
    def _min_dist_segments(P0, P1, Q0, Q1):
        '''Minimum distance between line segment P0-P1 and line segment Q0-Q1.

        Parameters
        ----------
        P0, P1 : 1D ndarray
            Endpoints of first segment
        Q0, Q1 : 1D ndarray
            Endpoints of second segment

        Returns
        -------
        float
        '''
        d1 = P1 - P0
        d2 = Q1 - Q0
        r = P0 - Q0

        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)

        if a < 1e-12 and e < 1e-12:
            return np.linalg.norm(r)

        if a < 1e-12:
            t = np.clip(f / e, 0, 1)
            return np.linalg.norm(P0 - (Q0 + t * d2))

        c = np.dot(d1, r)

        if e < 1e-12:
            s = np.clip(-c / a, 0, 1)
            return np.linalg.norm((P0 + s * d1) - Q0)

        b = np.dot(d1, d2)
        denom = a * e - b * b

        if abs(denom) > 1e-12:
            s = np.clip((b * f - c * e) / denom, 0, 1)
        else:
            s = 0.0

        t = (b * s + f) / e

        if t < 0:
            t = 0.0
            s = np.clip(-c / a, 0, 1)
        elif t > 1:
            t = 1.0
            s = np.clip((b - c) / a, 0, 1)

        return np.linalg.norm((P0 + s * d1) - (Q0 + t * d2))


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
        seg_len2 = seg_lengths_2[~z_check]

        # For the rest, follow the same math as in closest_dist_btwn_line_and_pts
        # First, find the projection of the point onto the line and clamp to segments
        dot = ((pt-Q0)*(Q1-Q0)).sum(1)/seg_len2 # dot prod of each row vec
        t_list = np.maximum(0,np.minimum(1,dot))
        # Find the point on the segments
        proj_pt_list = Q0 + np.tile(t_list,(pt.shape[0],1)).T*(Q1-Q0)
        dist_list[~z_check] = np.linalg.norm(pt-proj_pt_list,axis=1)

        return dist_list


    def plot_targets_to_axis(self, ax, zorder=5):
        '''Plots the targets on a given axis object.

        zorder sets the draw layer (default 5, above direction-mesh quivers);
        pass a higher value to draw the targets on top of e.g. basin wheels.
        '''

        target_color = '0.5'  # medium grey
        # Render targets above other layers (e.g., direction-mesh quivers)
        # and fully opaque so they hide arrows passing through them.
        zo = zorder

        if self.geom_name is None:
            # delta functions
            ax.plot(self.locs[:,0],self.locs[:,1],'.', color=target_color,
                    zorder=zo)
        elif self.geom_name == 'circle':
            for n,pos in enumerate(self.locs):
                try:
                    circle = plt.Circle(pos, self.r[n], color=target_color,
                                        zorder=zo)
                except TypeError:
                    circle = plt.Circle(pos, self.r, color=target_color,
                                        zorder=zo)
                ax.add_patch(circle)
        elif self.geom_name == 'capsule':
            from matplotlib.patches import FancyBboxPatch
            for n, pos in enumerate(self.locs):
                l_n = self.l[n] if isinstance(self.l, np.ndarray) else self.l
                w_n = self.w[n] if isinstance(self.w, np.ndarray) else self.w
                th_n = self.theta[n] if isinstance(self.theta, np.ndarray) else float(self.theta)
                hw = w_n / 2
                if hw < 1e-15:
                    # Zero-width: draw as a line segment
                    dx, dy = l_n/2*np.cos(th_n), l_n/2*np.sin(th_n)
                    ax.plot([pos[0]-dx, pos[0]+dx],
                            [pos[1]-dy, pos[1]+dy], color=target_color,
                            zorder=zo)
                else:
                    # Draw two endpoint circles + connecting rectangle
                    ep1 = pos + np.array([l_n/2*np.cos(th_n), l_n/2*np.sin(th_n)])
                    ep2 = pos - np.array([l_n/2*np.cos(th_n), l_n/2*np.sin(th_n)])
                    # Perpendicular direction for rectangle edges
                    perp = np.array([-np.sin(th_n), np.cos(th_n)]) * hw
                    rect_x = [ep1[0]+perp[0], ep2[0]+perp[0],
                              ep2[0]-perp[0], ep1[0]-perp[0], ep1[0]+perp[0]]
                    rect_y = [ep1[1]+perp[1], ep2[1]+perp[1],
                              ep2[1]-perp[1], ep1[1]-perp[1], ep1[1]+perp[1]]
                    ax.fill(rect_x, rect_y, color=target_color, zorder=zo)
                    ax.plot(rect_x, rect_y, color=target_color, zorder=zo)
                    c1 = plt.Circle(ep1, hw, color=target_color, zorder=zo)
                    c2 = plt.Circle(ep2, hw, color=target_color, zorder=zo)
                    ax.add_patch(c1)
                    ax.add_patch(c2)
        else:
            raise NotImplementedError("This geometry still TBD in Targets.")



# Per-family metadata for PerceptionModel's two roles (warp + weight). Maps the
# generic two-slot constructor params (a_warp/b_warp, a_weight/b_weight) onto
# each distribution family's real parameter names, with defaults. 'slots' gives
# the real name for the (a_*, b_*) slots; None means that slot is unused by the
# family. The same families serve as warp (CDF-integrated angle map) and as
# weight (rho attractiveness), except 'direct_power' which is warp-only.
_FAMILY_INFO = {
    'cutoff':         {'slots': ('a', 'b'),     'defaults': {'a': np.pi/3, 'b': 4*np.pi/5}},
    'lin_cutoff':     {'slots': ('a', 'b'),     'defaults': {'a': np.pi/3, 'b': 4*np.pi/5}},
    'vonmises':       {'slots': ('k', None),    'defaults': {'k': 1.0}},
    'symmetric_beta': {'slots': ('alpha', 'b'), 'defaults': {'alpha': 5.0, 'b': np.pi}},
    'reg_power':      {'slots': ('d', 'e'),     'defaults': {'d': 0.5, 'e': 1e-3}},
    'direct_power':   {'slots': ('c', None),    'defaults': {'c': 0.5}},
}


class _ReadOnlyParams(dict):
    """A read-only dict view of a PerceptionModel role's parameters.

    Reads behave exactly like a dict (indexing, iteration, ``==`` against a
    plain dict, ``dict(...)``); any attempt to mutate raises with a message
    pointing the user at the ``a_warp``/``b_warp`` / ``a_weight``/``b_weight``
    properties, which are the supported way to change parameters (they rebuild
    the affected splines). Subclasses ``dict`` so equality and read access are
    free; only the mutators are overridden.
    """
    _MSG = ("PerceptionModel parameter dicts are read-only; set parameters via "
            "the a_warp/b_warp (or a_weight/b_weight) properties, which rebuild "
            "the splines.")

    def __setitem__(self, *args):
        raise TypeError(self._MSG)

    def __delitem__(self, *args):
        raise TypeError(self._MSG)

    def update(self, *args, **kwargs):
        raise TypeError(self._MSG)

    def setdefault(self, *args):
        raise TypeError(self._MSG)

    def pop(self, *args):
        raise TypeError(self._MSG)

    def popitem(self):
        raise TypeError(self._MSG)

    def clear(self):
        raise TypeError(self._MSG)


class PerceptionModel:
    '''This class takes in a Targets object and a focal location and angle for an
    observer, and then translates that into a neural angular position and a neural 
    spin group size for each target based on the perceived angular extents of the 
    target, its signal strength, and a weighting function that describes the 
    density of neurons in the ring as a function of angle.
    
    Following mathematics conventions, egocentric angles increase counterclockwise;
    e.g., positive egocentric angles are to the left of the observer and 
    negative egocentric angles are to the right.'''

    # Memo installed by the signal_cache() context manager; None means
    # "no caching" (the normal state). A class attribute rather than an
    # instance one so objects unpickled from before this existed still read
    # None instead of raising.
    _signal_cache = None

    def __init__(self, targets=None, focal_loc=(5,10), focal_angle=0,
                 neural_angle_dist='lin_cutoff', angle_weight=None,
                 a_warp=None, b_warp=None, a_weight=None, b_weight=None,
                 theta_mesh=2000):
        '''Establishes an observer at location focal_loc, looking in a direction
        given by focal_angle, at targets given by the targets object.

        The perception model has two independent roles:

          1. WARPING (neural_angle_dist): a distribution that is integrated in a
             CDF-like fashion to produce the egocentric -> neural angle map.
          2. WEIGHTING (angle_weight): a distribution integrated over each
             target's visible angular extent to set rho (the relative neural
             group size / attractiveness driving gamma).

        The two roles were a single coupled function in earlier versions; they
        are now decoupled so warp shape and weight shape/parameters can be tuned
        independently. The default (angle_weight=None) gives uniform weighting.

        Mutable post-init: focal_loc, focal_angle, targets. Warp and weight
        parameters are changed by assigning the a_warp/b_warp/a_weight/b_weight
        properties (same slot names as the constructor args below), which
        rebuild only the affected role's splines automatically. The
        warp_params/weight_params attributes are read-only views of the current
        parameters (keyed by canonical name); assign the a_*/b_* properties to
        change them. The roles (neural_angle_dist, angle_weight) themselves are
        not mutable post-init.

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
        neural_angle_dist : {'cutoff', 'lin_cutoff', 'vonmises', 'symmetric_beta', 'reg_power', 'direct_power', None} (default = 'lin_cutoff')
            WARP role. A named distribution is integrated CDF-like to map
            egocentric angles to neural angles (denser neural representation
            where the distribution is concentrated). The families:
                - 'cutoff' : smooth cutoff, 1 in front and 0 in back with a
                             smooth transition. Params a (inner), b (outer);
                             0 <= a < b. See _smooth_cutoff.
                - 'lin_cutoff' : trapezoidal cutoff -- the piecewise-linear
                             analog of 'cutoff' (1 on [-a, a], linear ramp to 0
                             on a < |theta| < b). Same params/support; closed-form
                             integral and inverse (no spline). See _lin_cutoff.
                - 'vonmises' : von Mises pdf exp(k*cos(theta))/(2*pi*I0(k)).
                             Param k > 0 (larger k = narrower peak). See _vonmises.
                - 'symmetric_beta' : Beta(alpha, alpha) rescaled to [-b, b].
                             Params alpha >= 1, b > 0 (alpha = 1 is uniform on
                             [-b, b]). See _symmetric_beta.
                - 'reg_power' : regularized power weight 1/(|theta|^d + e),
                             d, e > 0. The normalized integral converges to
                             _power(theta, c=1-d) as e -> 0. See _reg_power.
                - 'direct_power' : the power angle map directly (NOT a
                             CDF-integral of a density):
                             f(theta) = pi * sign(theta) * (|theta|/pi)^c, c > 0.
                             c = 1 is identity, c < 1 expands front angles, c > 1
                             compresses them. WARP ONLY -- not valid as a weight
                             (it is a signed angle map, not a density).
                - None : identity warp (neural angle = egocentric angle).
        angle_weight : {'cutoff', 'lin_cutoff', 'vonmises', 'symmetric_beta', 'reg_power', 'neural_angle_dist', None} (default = None)
            WEIGHT role. A named density family (same families as the warp,
            EXCEPT 'direct_power', which is disallowed) integrated over each
            target's visible arc to set rho. Plus:
                - 'neural_angle_dist' : tie the weight to the warp -- use the
                             same family and parameters as neural_angle_dist.
                             Reproduces the old full-weighting behavior.
                - None : uniform weight (rho from visible arc length / visible
                             count only). Reproduces the old weight_angle_only
                             behavior. This is the default.
        a_warp, b_warp : float, optional
            Parameters for the warp family, by slot. The real meaning of each
            slot depends on the family (see _FAMILY_INFO):
                cutoff/lin_cutoff: a_warp=a, b_warp=b;  vonmises: a_warp=k;
                symmetric_beta: a_warp=alpha, b_warp=b;
                reg_power: a_warp=d, b_warp=e;  direct_power: a_warp=c.
            If left None, the family default is used. Unused slots (e.g. b_warp
            for vonmises/direct_power) must be left None.
        a_weight, b_weight : float, optional
            Parameters for the weight family, same slot convention as the warp
            params. Must be None when angle_weight is None or
            'neural_angle_dist' (no independent weight family to parameterize).
        theta_mesh : float or 1D ndarray
            the number of equally spaced mesh points on [-pi,pi) to evaluate at
            or a mesh of theta values to evaluate at
        '''

        self.focal_loc = np.array(focal_loc, dtype=float)
        self.focal_angle = focal_angle

        # --- Resolve and validate the WARP role. ---
        if neural_angle_dist is not None and neural_angle_dist not in _FAMILY_INFO:
            raise ValueError(
                f"neural_angle_dist must be one of "
                f"{sorted(_FAMILY_INFO)} or None, got {neural_angle_dist!r}.")
        self.warp_name = neural_angle_dist
        self._warp_params = self._resolve_params(
            neural_angle_dist, a_warp, b_warp, role='warp')

        # --- Resolve and validate the WEIGHT role. ---
        # 'direct_power' is a signed angle map, not a density: disallow.
        if angle_weight == 'direct_power':
            raise ValueError(
                "angle_weight='direct_power' is not allowed: the power map is a "
                "signed angle map, not a non-negative density, so it cannot be "
                "integrated over visible arcs as a weight. Use 'reg_power' "
                "(whose density is the power-map derivative) instead.")
        self._weight_tied_to_warp = (angle_weight == 'neural_angle_dist')
        if self._weight_tied_to_warp:
            if a_weight is not None or b_weight is not None:
                raise ValueError(
                    "a_weight/b_weight must be None when "
                    "angle_weight='neural_angle_dist' (the weight reuses the "
                    "warp family and parameters). Set the parameters via "
                    "a_warp/b_warp instead.")
            if self.warp_name == 'direct_power' or self.warp_name is None:
                raise ValueError(
                    "angle_weight='neural_angle_dist' requires neural_angle_dist "
                    f"to be a density family, got {self.warp_name!r}. "
                    "'direct_power' and None have no associated density to "
                    "weight with.")
            # Weight uses the warp family + params.
            self.weight_name = self.warp_name
            self._weight_params = dict(self._warp_params)
        elif angle_weight is None:
            if a_weight is not None or b_weight is not None:
                raise ValueError(
                    "a_weight/b_weight must be None when angle_weight is None "
                    "(uniform weighting has no parameters).")
            self.weight_name = None
            self._weight_params = {}
        else:
            if angle_weight not in _FAMILY_INFO:
                raise ValueError(
                    f"angle_weight must be one of "
                    f"{sorted(k for k in _FAMILY_INFO if k != 'direct_power')}, "
                    f"'neural_angle_dist', or None, got {angle_weight!r}.")
            self.weight_name = angle_weight
            self._weight_params = self._resolve_params(
                angle_weight, a_weight, b_weight, role='weight')

        if targets is None:
            self.targets = Targets()
        else:
            assert isinstance(targets,Targets), "targets must be a Targets object."
            self.targets = targets
        if isinstance(theta_mesh, int):
            self.theta_mesh = np.linspace(-np.pi, np.pi, theta_mesh+1)[:-1]
        else:
            self.theta_mesh = theta_mesh

        # Build spline lookups for each role exactly once. The weight splines
        # are skipped when the weight is uniform or tied to the warp (the warp
        # antiderivative is reused in that case).
        self._build_warp_splines()
        self._build_weight_splines()

    # --- per-role parameter handling ---

    @staticmethod
    def _resolve_params(name, a_slot, b_slot, role):
        """Map the generic (a_slot, b_slot) constructor params onto a family's
        real parameter dict, filling family defaults for any slot left None and
        validating per-family constraints.

        Parameters
        ----------
        name : str or None
            family name (key of _FAMILY_INFO) or None (identity/uniform).
        a_slot, b_slot : float or None
            the generic slot values (a_warp/b_warp or a_weight/b_weight).
        role : {'warp', 'weight'}
            used only for clearer error messages.

        Returns
        -------
        dict : real parameter names -> values (empty dict for name=None).
        """
        if name is None:
            if a_slot is not None or b_slot is not None:
                raise ValueError(
                    f"a_{role}/b_{role} must be None when the {role} family is "
                    "None (identity warp / uniform weight has no parameters).")
            return {}

        info = _FAMILY_INFO[name]
        a_key, b_key = info['slots']
        params = dict(info['defaults'])

        if a_slot is not None:
            if a_key is None:
                raise ValueError(
                    f"a_{role} is not used by {name!r}; leave it None.")
            params[a_key] = a_slot
        if b_slot is not None:
            if b_key is None:
                raise ValueError(
                    f"b_{role} is not used by {name!r} (it has a single "
                    f"parameter {a_key!r}); leave b_{role} None.")
            params[b_key] = b_slot

        PerceptionModel._validate_params(name, params, role)
        return params

    @staticmethod
    def _validate_params(name, params, role):
        """Validate a family's resolved parameter dict, naming the real
        parameter (and the generic slot) in any error message."""
        if name in ('cutoff', 'lin_cutoff'):
            a, b = params['a'], params['b']
            if not (0 <= a < b):
                raise ValueError(
                    f"for {name} {role}, a_{role} (a) and b_{role} (b) must "
                    f"satisfy 0 <= a < b (got a={a}, b={b}).")
        elif name == 'vonmises':
            if not (params['k'] > 0):
                raise ValueError(
                    f"for vonmises {role}, a_{role} (k) must be > 0 "
                    f"(got k={params['k']}).")
        elif name == 'symmetric_beta':
            if not (params['alpha'] >= 1):
                raise ValueError(
                    f"for symmetric_beta {role}, a_{role} (alpha) must be >= 1 "
                    f"(got alpha={params['alpha']}).")
            if not (params['b'] > 0):
                raise ValueError(
                    f"for symmetric_beta {role}, b_{role} (b) must be > 0 "
                    f"(got b={params['b']}).")
        elif name == 'reg_power':
            if not (params['d'] > 0):
                raise ValueError(
                    f"for reg_power {role}, a_{role} (d) must be > 0 "
                    f"(got d={params['d']}).")
            if not (params['e'] > 0):
                raise ValueError(
                    f"for reg_power {role}, b_{role} (e) must be > 0 "
                    f"(got e={params['e']}).")
        elif name == 'direct_power':
            if not (params['c'] > 0):
                raise ValueError(
                    f"for direct_power {role}, a_{role} (c) must be > 0 "
                    f"(got c={params['c']}).")

    # --- read-only views of the live parameter dicts ---

    @property
    def warp_params(self):
        """Read-only view of the warp family's current parameters (keyed by
        canonical name, e.g. {'k': 0.55}). To change a parameter, assign the
        a_warp/b_warp properties (they rebuild the splines)."""
        return _ReadOnlyParams(dict(self._warp_params))

    @warp_params.setter
    def warp_params(self, value):
        raise AttributeError(_ReadOnlyParams._MSG)

    @property
    def weight_params(self):
        """Read-only view of the weight family's current parameters. To change a
        parameter, assign the a_weight/b_weight properties (they rebuild the
        splines)."""
        return _ReadOnlyParams(dict(self._weight_params))

    @weight_params.setter
    def weight_params(self, value):
        raise AttributeError(_ReadOnlyParams._MSG)

    # --- assignable two-slot parameter properties (auto-respline on write) ---
    #
    # a_warp/b_warp/a_weight/b_weight map a role's generic slots onto its
    # family's canonical parameter (see _FAMILY_INFO and the constructor
    # docstring for the slot->name table). The getter is permissive (returns
    # None for an unused slot / identity warp / uniform weight); the setter is
    # strict and rebuilds the affected role's splines.

    @property
    def a_warp(self):
        return self._get_slot('warp', 0)

    @a_warp.setter
    def a_warp(self, value):
        self._set_slot('warp', 0, value)

    @property
    def b_warp(self):
        return self._get_slot('warp', 1)

    @b_warp.setter
    def b_warp(self, value):
        self._set_slot('warp', 1, value)

    @property
    def a_weight(self):
        return self._get_slot('weight', 0)

    @a_weight.setter
    def a_weight(self, value):
        self._set_slot('weight', 0, value)

    @property
    def b_weight(self):
        return self._get_slot('weight', 1)

    @b_weight.setter
    def b_weight(self, value):
        self._set_slot('weight', 1, value)

    def _role_state(self, role):
        """Return (name, params_dict) for 'warp' or 'weight'. For a tied weight
        the warp's name + live params are returned (the weight mirrors them)."""
        if role == 'warp':
            return self.warp_name, self._warp_params
        if self._weight_tied_to_warp:
            return self.warp_name, self._warp_params
        return self.weight_name, self._weight_params

    def _get_slot(self, role, slot_idx):
        """Permissive read of a role's slot value. Returns None when the role is
        identity/uniform (no family) or the slot is unused by the family."""
        name, params = self._role_state(role)
        if name is None:
            return None
        key = _FAMILY_INFO[name]['slots'][slot_idx]
        if key is None:
            return None
        return params.get(key)

    def _set_slot(self, role, slot_idx, value):
        """Strict write of a role's slot value, then rebuild that role's splines
        (mirroring + rebuilding the weight too when a tied warp is changed).
        Raises when there is no parameter to set for this slot."""
        slot_name = f'{"a" if slot_idx == 0 else "b"}_{role}'
        if role == 'weight' and self._weight_tied_to_warp:
            raise ValueError(
                f"cannot set {slot_name}: the weight is tied to the warp "
                "(angle_weight='neural_angle_dist'); set a_warp/b_warp instead.")
        name, params = self._role_state(role)
        if name is None:
            if role == 'warp':
                raise ValueError(
                    f"cannot set {slot_name}: neural_angle_dist is None "
                    "(identity warp has no parameters).")
            raise ValueError(
                f"cannot set {slot_name}: angle_weight is None "
                "(uniform weight has no parameters).")
        key = _FAMILY_INFO[name]['slots'][slot_idx]
        if key is None:
            raise ValueError(
                f"{slot_name} is not used by {name!r} "
                f"(it has a single parameter {_FAMILY_INFO[name]['slots'][0]!r}).")
        # Validate against a trial copy so a bad value leaves state unchanged.
        trial = dict(params)
        trial[key] = value
        self._validate_params(name, trial, role)
        params[key] = value
        if role == 'warp':
            self._build_warp_splines()
            if self._weight_tied_to_warp:
                self._weight_params = dict(self._warp_params)
                self._build_weight_splines()
        else:
            self._build_weight_splines()

    @staticmethod
    def _make_integral_spline(name, params):
        """Build (forward, inverse) CubicSplines for the CDF-like integral map
        of a density family, or (None, None) for families with no spline
        (lin_cutoff and symmetric_beta are evaluated analytically;
        direct_power / None have no integral map).

        Used by both roles: the warp uses both returned splines; the weight
        uses only the forward spline (as an antiderivative for the rho
        arc-integral). The per-family node construction is preserved verbatim
        from the original single-spline builder to protect numerics: cutoff
        uses a saturated-tail monotone filter; reg_power uses a cubic node mesh
        with a monotonicity assert; vonmises uses plain equispaced nodes via
        scipy's cdf.
        """
        if name == 'cutoff':
            a = params['a']
            b = params['b']
            n_nodes = 2001
            x_nodes = np.linspace(-b, b, n_nodes)
            # Snap the center node to 0 exactly (should already hold for an
            # odd number of equispaced nodes but avoids floating-point drift).
            center = n_nodes // 2
            x_nodes[center] = 0.0
            y_nodes = np.empty(n_nodes)
            for i, x in enumerate(x_nodes):
                y_nodes[i] = PerceptionModel._smooth_cutoff_integral(x, a, b)
            # Snap endpoints and center to the exact theoretical values,
            # preserving F(-b) = -pi, F(b) = pi, and F(0) = 0 so symmetry
            # and saturation are honored at roundoff level.
            y_nodes[0] = -np.pi
            y_nodes[-1] = np.pi
            y_nodes[center] = 0.0
            # Near +/-b the cutoff is exponentially small, so quad can return
            # F values that collapse to -pi (or pi) in floating point for
            # multiple adjacent nodes. Enforce strict monotonicity by dropping
            # interior nodes whose y does not strictly increase past the
            # running maximum; always keep the two boundary nodes with their
            # exact snapped values.
            kept = [0]
            for i in range(1, n_nodes - 1):
                if y_nodes[i] > y_nodes[kept[-1]]:
                    kept.append(i)
            while kept and y_nodes[kept[-1]] >= y_nodes[-1]:
                kept.pop()
            kept.append(n_nodes - 1)
            kept = np.array(kept)
            x_kept = x_nodes[kept]
            y_kept = y_nodes[kept]
            return (CubicSpline(x_kept, y_kept, bc_type='natural'),
                    CubicSpline(y_kept, x_kept, bc_type='natural'))
        elif name == 'vonmises':
            k_val = params['k']
            n_nodes = 2001
            theta_nodes = np.linspace(-np.pi, np.pi, n_nodes)
            center = n_nodes // 2
            theta_nodes[center] = 0.0
            y_nodes = 2*np.pi*(vonmises.cdf(theta_nodes, k_val) - 0.5)
            y_nodes[0] = -np.pi
            y_nodes[-1] = np.pi
            y_nodes[center] = 0.0
            return (CubicSpline(theta_nodes, y_nodes, bc_type='natural'),
                    CubicSpline(y_nodes, theta_nodes, bc_type='natural'))
        elif name == 'reg_power':
            d = params['d']
            e = params['e']
            # Cubic mesh: theta = pi * sign(u) * |u|^3 with u = linspace(-1, 1).
            # Concentrates nodes near 0, where the integrand 1/(|x|^d + e) is
            # peaked (value 1/e there) and F has very high curvature
            # (F''(theta) ~ |theta|^(d-2) for d < 1 as theta -> 0). Cubic
            # stretching keeps cubic-spline error below ~5e-7 across
            # d in [0.3, 1.0] and e in [1e-3, 1e-1] at n_nodes = 2001;
            # quartic and higher meshes give only marginally better near-0
            # accuracy at the cost of slightly worse error elsewhere.
            n_nodes = 2001
            u = np.linspace(-1.0, 1.0, n_nodes)
            theta_nodes = np.pi * np.sign(u) * np.abs(u)**3
            center = n_nodes // 2
            theta_nodes[center] = 0.0
            theta_nodes[0] = -np.pi
            theta_nodes[-1] = np.pi
            y_nodes = np.empty(n_nodes)
            for i, x in enumerate(theta_nodes):
                y_nodes[i] = PerceptionModel._reg_power_integral(x, d, e)
            # Pin endpoints and center to exact theoretical values.
            y_nodes[0] = -np.pi
            y_nodes[-1] = np.pi
            y_nodes[center] = 0.0
            # The integrand is bounded below by 1/(pi^d + e) > 0, so the
            # numerical integral is strictly monotone up to quad noise.
            # Assert rather than filter (no flat-tail risk like cutoff).
            assert np.all(np.diff(y_nodes) > 0), (
                "reg_power integral nodes are not strictly increasing; "
                "check d, e parameters and quad tolerance.")
            return (CubicSpline(theta_nodes, y_nodes, bc_type='natural'),
                    CubicSpline(y_nodes, theta_nodes, bc_type='natural'))
        else:
            # symmetric_beta (analytic), direct_power, None: no spline.
            return (None, None)

    def _build_warp_splines(self):
        """Build the warp role's forward+inverse integral splines (left None for
        analytic symmetric_beta, identity None, or the direct_power map)."""
        self._warp_forward_spline, self._warp_inverse_spline = \
            self._make_integral_spline(self.warp_name, self._warp_params)

    def _build_weight_splines(self):
        """Build the weight role's forward integral spline (the antiderivative
        used by the rho arc-integral). Left None when the weight is uniform
        (weight_name is None), tied to the warp (the warp antiderivative is
        reused), or analytic (symmetric_beta)."""
        if self.weight_name is None or self._weight_tied_to_warp:
            self._weight_forward_spline = None
            return
        self._weight_forward_spline, _ = \
            self._make_integral_spline(self.weight_name, self._weight_params)

    @staticmethod
    def _eval_forward_map(name, params, fwd_spline, theta):
        """Forward CDF-like angle map F(theta) for a density family, saturating
        to +-pi outside the support. Uses the supplied precomputed forward
        spline for cutoff/vonmises/reg_power; analytic scipy cdf for
        symmetric_beta. (The same evaluator serves the warp and, as an
        antiderivative for the rho arc-integral, the weight.)"""
        if name == 'cutoff':
            theta_arr = np.asarray(theta, dtype=float)
            scalar = theta_arr.ndim == 0
            b = params['b']
            clamped = np.clip(theta_arr, -b, b)
            result = fwd_spline(clamped)
            result = np.where(theta_arr >= b, np.pi, result)
            result = np.where(theta_arr <= -b, -np.pi, result)
            return float(result) if scalar else result
        elif name == 'lin_cutoff':
            return PerceptionModel._lin_cutoff_integral(
                theta, params['a'], params['b'])
        elif name == 'vonmises' or name == 'reg_power':
            theta_arr = np.asarray(theta, dtype=float)
            scalar = theta_arr.ndim == 0
            clamped = np.clip(theta_arr, -np.pi, np.pi)
            result = fwd_spline(clamped)
            result = np.where(theta_arr >= np.pi, np.pi, result)
            result = np.where(theta_arr <= -np.pi, -np.pi, result)
            return float(result) if scalar else result
        elif name == 'symmetric_beta':
            return PerceptionModel._symmetric_beta_integral(
                theta, params['alpha'], params['b'])
        else:
            raise NotImplementedError(
                f"no forward integral map for family {name!r}.")

    @staticmethod
    def _eval_inverse_map(name, params, inv_spline, y):
        """Inverse of _eval_forward_map. Domain y in [-pi, pi]."""
        if name == 'cutoff':
            y_arr = np.asarray(y, dtype=float)
            scalar = y_arr.ndim == 0
            if np.any((y_arr < -np.pi) | (y_arr > np.pi)):
                raise ValueError("y must satisfy -pi <= y <= pi.")
            b = params['b']
            result = inv_spline(y_arr)
            result = np.where(y_arr == np.pi, b, result)
            result = np.where(y_arr == -np.pi, -b, result)
            return float(result) if scalar else result
        elif name == 'lin_cutoff':
            return PerceptionModel._lin_cutoff_int_inverse(
                y, params['a'], params['b'])
        elif name == 'vonmises' or name == 'reg_power':
            y_arr = np.asarray(y, dtype=float)
            scalar = y_arr.ndim == 0
            if np.any((y_arr < -np.pi) | (y_arr > np.pi)):
                raise ValueError("y must satisfy -pi <= y <= pi.")
            result = inv_spline(y_arr)
            result = np.where(y_arr == np.pi, np.pi, result)
            result = np.where(y_arr == -np.pi, -np.pi, result)
            return float(result) if scalar else result
        elif name == 'symmetric_beta':
            return PerceptionModel._symmetric_beta_int_inverse(
                y, params['alpha'], params['b'])
        else:
            raise NotImplementedError(
                f"no inverse integral map for family {name!r}.")

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
    def _smooth_cutoff_integral(theta, a, b, tol=1.49e-10):
        """
        Compute F(theta; a, b) = norm * integral from 0 to theta of
        _smooth_cutoff(x; a, b) dx for a single float theta, where
        norm = 2*pi/(a+b). The normalization makes F(+/-b) = +/-pi so F
        plays the role of a CDF-like transformation. Used as the
        reference implementation; hot-path callers use the precomputed
        forward spline (via _eval_forward_map) instead.
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
    def _smooth_cutoff_int_inverse(y, a, b, tol=1.0e-8):
        """
        Compute F^{-1}(y; a, b) for a single float y (the inverse of
        _smooth_cutoff_integral). Used as the reference implementation;
        hot-path callers use the precomputed inverse spline (via
        _eval_inverse_map) instead.
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
            return PerceptionModel._smooth_cutoff_integral(theta, a, b, tol) - np.abs(y)

        # Bracket: F is strictly increasing from 0 to pi on (a, b).
        eps = (b - a) * 1e-12
        x_lo = a + eps
        x_hi = b - eps

        result = brentq(func, x_lo, x_hi, xtol=tol, rtol=tol, maxiter=200)
        return np.sign(y) * result

    @staticmethod
    def _lin_cutoff(x, a, b):
        """Trapezoidal (piecewise-linear) cutoff density: 1 on [-a, a], a
        linear ramp down to 0 on a < |x| < b, and 0 for |x| >= b. The
        piecewise-linear analog of _smooth_cutoff -- same support and unit
        plateau, hence the same area (a+b)/2 (so the integral map shares the
        normalization 2*pi/(a+b)), but with a closed-form integral and
        inverse instead of an essential-singularity bump. Requires
        0 <= a < b. Vectorized.
        """
        if not (0 <= a < b):
            raise ValueError(f"Parameters must satisfy 0 <= a < b (a={a}, b={b}).")
        x = np.asarray(x, dtype=float)
        scalar_input = x.ndim == 0
        absx = np.abs(x)
        result = np.where(
            absx <= a, 1.0,
            np.where(absx < b, (b - absx) / (b - a), 0.0))
        return result.item() if scalar_input else result

    @staticmethod
    def _lin_cutoff_integral(theta, a, b):
        """Forward CDF-like map F(theta; a, b) = norm * integral from 0 to
        theta of _lin_cutoff(x; a, b) dx, with norm = 2*pi/(a+b) so that
        F(+/-b) = +/-pi (matching the _smooth_cutoff_integral convention).
        Closed form and odd in theta: linear (norm*theta) on |theta| <= a,
        quadratic on a < |theta| < b, saturating to +/-pi for |theta| >= b.
        Requires 0 <= a < b. Vectorized; replaces the quad-based reference
        used for the smooth cutoff, so no spline is needed.
        """
        if not (0 <= a < b):
            raise ValueError(f"Parameters must satisfy 0 <= a < b (a={a}, b={b}).")
        theta = np.asarray(theta, dtype=float)
        scalar_input = theta.ndim == 0
        norm = 2 * np.pi / (a + b)
        s = np.sign(theta)
        at = np.abs(theta)
        # Ramp branch (a < |theta| < b). The expression is evaluated for all
        # entries but only selected on the ramp; (b - at)**2 stays finite
        # outside it, so the masked-out values are harmless.
        ramp = norm * (a + ((b - a) ** 2 - (b - at) ** 2) / (2 * (b - a)))
        result = np.where(
            at <= a, norm * at,
            np.where(at < b, ramp, np.pi))
        result = s * result
        return result.item() if scalar_input else result

    @staticmethod
    def _lin_cutoff_int_inverse(y, a, b):
        """Inverse of _lin_cutoff_integral. Domain y in [-pi, pi]; pins
        y = +/-pi to +/-b (saturation convention). Closed form and odd in y:
        linear (y/norm) on |y| <= a*norm, a single square root on the ramp.
        Exact to machine precision (no condition limit near +/-pi, unlike the
        smooth-cutoff spline). Requires 0 <= a < b. Vectorized.
        """
        if not (0 <= a < b):
            raise ValueError(f"Parameters must satisfy 0 <= a < b (a={a}, b={b}).")
        y = np.asarray(y, dtype=float)
        scalar_input = y.ndim == 0
        if np.any(y < -np.pi) or np.any(y > np.pi):
            raise ValueError("y must satisfy -pi <= y <= pi.")
        norm = 2 * np.pi / (a + b)
        s = np.sign(y)
        ay = np.abs(y)
        # Ramp inverse: theta = b - sqrt((b-a)^2 - 2(b-a)(|y|/norm - a)).
        # Clip the discriminant to guard tiny negatives from roundoff at |y|=pi.
        disc = np.clip((b - a) ** 2 - 2 * (b - a) * (ay / norm - a), 0.0, None)
        ramp = b - np.sqrt(disc)
        result = np.where(ay <= a * norm, ay / norm, ramp)
        result = s * result
        # Force exact endpoints.
        result = np.where(y == np.pi, b, result)
        result = np.where(y == -np.pi, -b, result)
        return result.item() if scalar_input else result

    @staticmethod
    def _vonmises(theta, k):
        """A von Mises pdf, smooth and bell-shaped around 0.

        f(theta) = exp(k*cos(theta)) / (2*pi*I0(k))

        where I0 is the modified Bessel function of the first kind of order 0.
        The parameter k controls the width of the bell: larger k gives a
        narrower peak. Integrates to 1 over [-pi, pi].

        Implemented directly rather than via scipy.stats.vonmises.pdf to avoid
        the rv_continuous dispatch overhead.

        Parameters
        ----------
        theta : float or array_like
            Angle(s) in radians.
        k : float
            Concentration parameter; must be positive.

        Returns
        -------
        float or ndarray : The value(s) of the pdf at the given theta.
        """

        if k <= 0:
            raise ValueError(f"Parameter k must be positive (k={k}).")
        theta = np.asarray(theta, dtype=float)
        scalar_input = theta.ndim == 0
        result = np.exp(k * np.cos(theta)) / (2*np.pi*i0(k))
        return result.item() if scalar_input else result

    @staticmethod
    def _vonmises_integral(theta, k):
        """
        Compute G(theta; k) = (1/I0(k)) * integral from 0 to theta of
        exp(k*cos(x)) dx. Maps [-pi, pi] to [-pi, pi] (i.e. G(+/-pi) = +/-pi),
        playing the role of a CDF-like transformation for the _vonmises weight.

        Equivalently (by a constant factor): G(theta; k) =
        2*pi*(vonmises_cdf(theta, k) - 0.5), which is how it is computed here
        via scipy.stats.vonmises.cdf.

        Parameters
        ----------
        theta : float or array_like
            Upper limit(s) of integration.
        k : float
            Concentration parameter; must be positive.

        Returns
        -------
        float or ndarray : The value(s) of G(theta; k).
        """
        if k <= 0:
            raise ValueError(f"Parameter k must be positive (k={k}).")
        theta = np.asarray(theta, dtype=float)
        scalar_input = theta.ndim == 0
        result = 2*np.pi*(vonmises.cdf(theta, k) - 0.5)
        return result.item() if scalar_input else result

    @staticmethod
    def _vonmises_int_inverse(y, k):
        """
        Compute G^{-1}(y; k): the value of theta such that
        G(theta; k) = y, for y in [-pi, pi].

        Uses scipy.stats.vonmises.ppf: since G = 2*pi*(cdf - 0.5),
        theta = vonmises.ppf(y/(2*pi) + 0.5, k).

        Parameters
        ----------
        y : float or array_like
            Target value(s); each must satisfy -pi <= y <= pi.
        k : float
            Concentration parameter; must be positive.

        Returns
        -------
        float or ndarray : theta value(s) satisfying G(theta; k) = y.
        """
        if k <= 0:
            raise ValueError(f"Parameter k must be positive (k={k}).")
        y = np.asarray(y, dtype=float)
        scalar_input = y.ndim == 0
        if np.any(y < -np.pi) or np.any(y > np.pi):
            raise ValueError("y must satisfy -pi <= y <= pi.")
        result = vonmises.ppf(y/(2*np.pi) + 0.5, k)
        # Force exact endpoints (scipy's ppf can return nan/inf at 0 or 1).
        result = np.where(y == np.pi, np.pi, result)
        result = np.where(y == -np.pi, -np.pi, result)
        return result.item() if scalar_input else result

    @staticmethod
    def _symmetric_beta(theta, alpha, b):
        """A symmetric Beta(alpha, alpha) pdf rescaled to [-b, b].

        With u = (theta + b)/(2b), the pdf is
            f(theta) = (1/(2b)) * u^(alpha-1) * (1-u)^(alpha-1) / B(alpha, alpha)
        on [-b, b], and zero outside. Symmetric about 0; alpha = 1 gives the
        uniform pdf 1/(2b); larger alpha gives a narrower peak at 0.

        Parameters
        ----------
        theta : float or array_like
            Angle(s) in radians.
        alpha : float
            Beta shape parameter (alpha = beta); must satisfy alpha >= 1.
        b : float
            Half-width of the support; must be positive.

        Returns
        -------
        float or ndarray : The value(s) of the pdf at the given theta.
        """
        if alpha < 1:
            raise ValueError(f"Parameter alpha must satisfy alpha >= 1 (alpha={alpha}).")
        if b <= 0:
            raise ValueError(f"Parameter b must be positive (b={b}).")
        theta = np.asarray(theta, dtype=float)
        scalar_input = theta.ndim == 0
        result = beta_dist.pdf(theta, alpha, alpha, loc=-b, scale=2*b)
        return result.item() if scalar_input else result

    @staticmethod
    def _symmetric_beta_integral(theta, alpha, b):
        """
        Compute G(theta; alpha, b) = 2*pi * (cdf(theta) - 0.5), where cdf is
        the Beta(alpha, alpha) cdf rescaled to [-b, b]. Maps [-pi, pi] to
        [-pi, pi] with G(0) = 0, G(b) = pi, G(-b) = -pi, saturating to +/- pi
        outside [-b, b].

        Parameters
        ----------
        theta : float or array_like
            Upper limit(s) of integration.
        alpha : float
            Beta shape parameter (alpha = beta); must satisfy alpha >= 1.
        b : float
            Half-width of the support; must be positive.

        Returns
        -------
        float or ndarray : The value(s) of G(theta; alpha, b).
        """
        if alpha < 1:
            raise ValueError(f"Parameter alpha must satisfy alpha >= 1 (alpha={alpha}).")
        if b <= 0:
            raise ValueError(f"Parameter b must be positive (b={b}).")
        theta = np.asarray(theta, dtype=float)
        scalar_input = theta.ndim == 0
        result = 2*np.pi*(beta_dist.cdf(theta, alpha, alpha, loc=-b, scale=2*b) - 0.5)
        return result.item() if scalar_input else result

    @staticmethod
    def _symmetric_beta_int_inverse(y, alpha, b):
        """
        Compute G^{-1}(y; alpha, b): the value of theta such that
        G(theta; alpha, b) = y, for y in [-pi, pi]. Pins y = +-pi to +-b
        (saturation convention).

        Parameters
        ----------
        y : float or array_like
            Target value(s); each must satisfy -pi <= y <= pi.
        alpha : float
            Beta shape parameter (alpha = beta); must satisfy alpha >= 1.
        b : float
            Half-width of the support; must be positive.

        Returns
        -------
        float or ndarray : theta value(s) satisfying G(theta; alpha, b) = y.
        """
        if alpha < 1:
            raise ValueError(f"Parameter alpha must satisfy alpha >= 1 (alpha={alpha}).")
        if b <= 0:
            raise ValueError(f"Parameter b must be positive (b={b}).")
        y = np.asarray(y, dtype=float)
        scalar_input = y.ndim == 0
        if np.any(y < -np.pi) or np.any(y > np.pi):
            raise ValueError("y must satisfy -pi <= y <= pi.")
        result = beta_dist.ppf(y/(2*np.pi) + 0.5, alpha, alpha, loc=-b, scale=2*b)
        # Force exact endpoints (scipy's ppf can return nan/inf at 0 or 1).
        result = np.where(y == np.pi, b, result)
        result = np.where(y == -np.pi, -b, result)
        return result.item() if scalar_input else result

    @staticmethod
    def _reg_power(theta, d, e):
        """A regularized power weight, 1 / (|theta|^d + e), for d, e > 0.

        Bounded everywhere (max = 1/e at theta = 0) and symmetric about 0.
        Approximates |theta|^(-d), the (un-normalized) derivative of the
        _power(theta, c) angle map with c = 1 - d, with the e -> 0 singularity
        at 0 regularized away. Not a normalized pdf; the constant factor cancels
        when used as a neural weight (rho = G / G.sum()) and the normalization
        used by _reg_power_integral makes the integral map [-pi, pi] -> [-pi, pi].

        Parameters
        ----------
        theta : float or array_like
            Angle(s) in radians.
        d : float
            Power exponent; must be positive.
        e : float
            Regularization parameter; must be positive.

        Returns
        -------
        float or ndarray : The value(s) of the weight at the given theta.
        """
        if d <= 0:
            raise ValueError(f"Parameter d must be positive (d={d}).")
        if e <= 0:
            raise ValueError(f"Parameter e must be positive (e={e}).")
        theta = np.asarray(theta, dtype=float)
        scalar_input = theta.ndim == 0
        result = 1.0 / (np.abs(theta)**d + e)
        return result.item() if scalar_input else result

    @staticmethod
    def _reg_power_integral(theta, d, e, tol=1.49e-10):
        """
        Compute F(theta; d, e) = pi * sign(theta) * I(|theta|) / I(pi), where
        I(t) = integral_0^t 1/(x^d + e) dx. Maps [-pi, pi] to [-pi, pi] with
        F(0) = 0, F(+/-pi) = +/-pi, saturating outside [-pi, pi]. As e -> 0,
        F(theta; d, e) converges to _power(theta, c=1-d) (analytically, since
        the antiderivative becomes |theta|^(1-d)/(1-d) for d != 1).

        Used as the reference implementation; hot-path callers use the
        precomputed forward spline (via _eval_forward_map) instead.

        Parameters
        ----------
        theta : float or array_like
            Upper limit(s) of integration; saturated outside [-pi, pi].
        d : float
            Power exponent; must be positive.
        e : float
            Regularization parameter; must be positive.
        tol : float
            Absolute and relative tolerance for scipy.integrate.quad.

        Returns
        -------
        float or ndarray : The value(s) of F(theta; d, e).
        """
        if d <= 0:
            raise ValueError(f"Parameter d must be positive (d={d}).")
        if e <= 0:
            raise ValueError(f"Parameter e must be positive (e={e}).")

        def integrand(x):
            return 1.0 / (x**d + e)

        # Normalization: integral over [0, pi]. Cached per (d, e) call site
        # via a simple memo so the spline build (one quad call per node) does
        # not recompute Z each time.
        Z = PerceptionModel._reg_power_normalization(d, e, tol)

        theta_arr = np.asarray(theta, dtype=float)
        scalar_input = theta_arr.ndim == 0
        flat = np.atleast_1d(theta_arr).astype(float)
        out = np.empty_like(flat)
        for i, t in enumerate(flat):
            if t >= np.pi:
                out[i] = np.pi
            elif t <= -np.pi:
                out[i] = -np.pi
            elif t == 0.0:
                out[i] = 0.0
            else:
                I, _err = quad(integrand, 0.0, abs(t),
                               epsabs=tol, epsrel=tol, limit=200)
                out[i] = np.pi * np.sign(t) * I / Z
        if scalar_input:
            return float(out[0])
        return out

    # Tiny memo for the (d, e) -> Z normalization integral. Keyed by the
    # exact float bit pattern; cleared rarely. Avoids n_nodes redundant quad
    # calls during a spline build.
    _reg_power_norm_cache = {}

    @staticmethod
    def _reg_power_normalization(d, e, tol=1.49e-10):
        key = (float(d), float(e), float(tol))
        cache = PerceptionModel._reg_power_norm_cache
        Z = cache.get(key)
        if Z is None:
            Z, _err = quad(lambda x: 1.0/(x**d + e), 0.0, np.pi,
                           epsabs=tol, epsrel=tol, limit=200)
            cache[key] = Z
        return Z

    @staticmethod
    def _reg_power_int_inverse(y, d, e, tol=1.0e-8):
        """
        Compute F^{-1}(y; d, e): the value of theta such that
        F(theta; d, e) = y, for y in [-pi, pi]. Pins y = +-pi to +-pi.
        Used as the reference implementation; hot-path callers use the
        precomputed inverse spline (via _eval_inverse_map) instead.

        Parameters
        ----------
        y : float or array_like
            Target value(s); each must satisfy -pi <= y <= pi.
        d : float
            Power exponent; must be positive.
        e : float
            Regularization parameter; must be positive.
        tol : float
            Absolute and relative tolerance for brentq.

        Returns
        -------
        float or ndarray : theta value(s) satisfying F(theta; d, e) = y.
        """
        if d <= 0:
            raise ValueError(f"Parameter d must be positive (d={d}).")
        if e <= 0:
            raise ValueError(f"Parameter e must be positive (e={e}).")
        y_arr = np.asarray(y, dtype=float)
        scalar_input = y_arr.ndim == 0
        if np.any(y_arr < -np.pi) or np.any(y_arr > np.pi):
            raise ValueError("y must satisfy -pi <= y <= pi.")
        flat = np.atleast_1d(y_arr).astype(float)
        out = np.empty_like(flat)
        for i, yv in enumerate(flat):
            if yv == np.pi:
                out[i] = np.pi
            elif yv == -np.pi:
                out[i] = -np.pi
            elif yv == 0.0:
                out[i] = 0.0
            else:
                target = abs(yv)
                func = lambda t: PerceptionModel._reg_power_integral(
                    t, d, e, tol=1.49e-10) - target
                # F(0) = 0 < target < pi = F(pi); strictly monotone.
                eps = np.pi * 1e-14
                root_pos = brentq(func, eps, np.pi - eps,
                                  xtol=tol, rtol=tol, maxiter=200)
                out[i] = np.sign(yv) * root_pos
        if scalar_input:
            return float(out[0])
        return out

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

        name = self.weight_name
        if name is None:
            # Uniform weight: integral is arc length.
            return sum(hi - lo for lo, hi in intervals)

        # Weighted: integrate the weight density via its CDF-like antiderivative
        # F. The constant normalization factor in F cancels in rho = G/G.sum(),
        # so F(hi) - F(lo) is the relevant quantity. When the weight is tied to
        # the warp, reuse the warp's forward spline (no separate weight spline
        # was built); otherwise use the weight's own forward spline.
        if self._weight_tied_to_warp:
            fwd = self._warp_forward_spline
        else:
            fwd = self._weight_forward_spline
        F = lambda x: self._eval_forward_map(name, self._weight_params, fwd, x)
        return sum(F(hi) - F(lo) for lo, hi in intervals)


    def get_neural_weight(self, theta):
        '''Returns the neural weight density for given angles theta based on the
        weighting function (angle_weight). This is a proxy for the relative
        attractiveness / neural group size contributed per unit visible arc, and
        (for front-biased families) weights things in front more highly than in
        back. Returns ones for uniform weighting (angle_weight=None).

        Parameters
        ----------
        theta : float or 1D ndarray
            angle(s) to evaluate the neural weight at

        Returns
        -------
        neural weight(s) corresponding to input theta value(s)
        '''

        name = self.weight_name
        p = self._weight_params
        if name is None:
            return np.ones_like(theta)
        elif name == 'cutoff':
            return self._smooth_cutoff(theta, p['a'], p['b'])
        elif name == 'lin_cutoff':
            return self._lin_cutoff(theta, p['a'], p['b'])
        elif name == 'vonmises':
            return self._vonmises(theta, p['k'])
        elif name == 'symmetric_beta':
            return self._symmetric_beta(theta, p['alpha'], p['b'])
        elif name == 'reg_power':
            return self._reg_power(theta, p['d'], p['e'])
        else:
            raise NotImplementedError(
                f"Unknown neural weight family {name!r}.")


    def get_neural_angle(self, theta):
        '''Returns the neural position for a given angle theta based on the
        warp (neural_angle_dist). This maps the perceived center of each target
        to the neural position of the corresponding spin group, via the CDF-like
        integral of a density family, the direct power map, or identity.

        Parameters
        ----------
        theta : float or 1D ndarray
            angle(s) to evaluate the neural position transformation at

        Returns
        -------
        neural position(s) corresponding to input theta value(s)
        '''

        name = self.warp_name
        if name is None:
            return theta
        elif name == 'direct_power':
            return self._power(theta, self._warp_params['c'])
        else:
            return self._eval_forward_map(
                name, self._warp_params, self._warp_forward_spline, theta)


    def get_neural_angle_inverse(self, theta):
        '''Returns the angle corresponding to a given neural position theta,
        the inverse of the warp (neural_angle_dist). Maps neural position back to
        the perceived center of each target.

        Parameters
        ----------
        theta : float or 1D ndarray
            neural position(s) to evaluate the inverse transformation at

        Returns
        -------
        angle(s) corresponding to input neural position value(s)
        '''

        name = self.warp_name
        if name is None:
            return theta
        elif name == 'direct_power':
            return self._power_inverse(theta, self._warp_params['c'])
        else:
            return self._eval_inverse_map(
                name, self._warp_params, self._warp_inverse_spline, theta)


    def _get_target_signals(self, focal_angle=None, focal_loc=None, mesh_signal=False):
        '''Returns the egocentric angular location of the center of each VISIBLE
        target (closer targets that are not delta functions block ones behind) as
        a length N array, and a normalized neural group size (rho) for each.

        For circle (and capsule) targets, blocking and neural group sizes are
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

        elif self.targets.geom_name == 'circle' or self.targets.geom_name == 'capsule':
            ##### Extended targets: exact interval arithmetic #####
            # Sort by closest-point distance (closest first for blocking).
            # For circles this is exact. For capsules, two capsules can
            # mutually occlude each other at different angles (the one with
            # the farther closest-point can still cross in front at some
            # angles). A fully correct solution would require per-angle
            # depth comparison; this closest-point sort is a practical
            # approximation.
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
                # Unwrap up front: get_percep_angles encodes an extent that
                # straddles +-pi as a wrapping pair (lo > hi), but both
                # _integrate_neural_weight and the mesh_signal masking below
                # require non-wrapping pieces. _subtract_intervals_circle
                # unwraps its inputs, so targets with a closer blocker were
                # already safe -- but the CLOSEST target (n == 0) never enters
                # that loop, and a raw wrapping pair reaching the integrator
                # yields a negative arc length, which the G > 0 visibility
                # filter then silently discards. That dropped the nearest
                # target for the whole angular window in which it straddles
                # the rear branch cut.
                intervals = self._unwrap_interval(original_extents[n])
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


    def plot_neural_weight(self, polar=False, ax=None, wb_plot=False):
        '''Plot the neural weighting function on [-pi, pi], with the max weight
        normalized to 1. For non-polar plots, also overlays the physical-to-
        neural angle mapping (theta -> hat{theta}) on a twin y-axis (radians),
        using a dashed line in the same color as the corresponding weight
        curve.

        Parameters
        ----------
        polar : bool, optional
            If True, use a polar plot. Default False (linear plot). The angle
            mapping overlay is only drawn for non-polar plots.
        ax : matplotlib axis, optional
            If provided, the curve is added to this axis (no figure created,
            no plt.show). For polar=True, the axis must have been created
            with projection='polar'. If None, a new figure and axis are
            created and plt.show() is called. When plotting multiple models
            on the same non-polar axis, a single twin axis is reused so all
            angle-mapping curves share one right-hand y-axis.
        wb_plot : bool
            whether or not plotting in a Jupyter notebook

        Returns
        -------
        ax : matplotlib axis, if axis was provided as an argument.
            Otherwise, None.
        '''

        theta = np.linspace(-np.pi, np.pi, 361)
        weights = self.get_neural_weight(theta)
        weights = weights/weights.max()

        if ax is None:
            local_plot = True
            if wb_plot:
                fig = plt.figure(figsize=(8, 5))
            else:
                fig = plt.figure(figsize=(6, 5))
            if polar:
                ax = plt.subplot(1, 1, 1, projection='polar')
            else:
                ax = plt.subplot(1, 1, 1)
        else:
            local_plot = False

        weight_label = f'{self.weight_name or "uniform"} weight'
        weight_line, = ax.plot(theta, weights, label=weight_label)

        if polar:
            # Match the tick scheme used in plot_blocked_signals, but with
            # labels in [-pi, pi] on the back half so it mirrors the data range.
            angles_deg = np.linspace(0, 360, 8, endpoint=False)
            labels = [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',
                      r'$\frac{3\pi}{4}$', r'$\pm\pi$', r'$-\frac{3\pi}{4}$',
                      r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$']
            ax.set_thetagrids(angles_deg, labels)
            if local_plot:
                ax.set_title('Neural Weighting Function')
                ax.legend()
                plt.show()
            else:
                return ax
            return

        tick_locs = np.array([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0,
                              np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        tick_labels = [r'$-\pi$', r'$-\frac{3\pi}{4}$', r'$-\frac{\pi}{2}$',
                       r'$-\frac{\pi}{4}$', r'$0$', r'$\frac{\pi}{4}$',
                       r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel('Neural weight')
        ax.set_xlim(-np.pi, np.pi)

        # Reuse a previously-attached twin so multiple models share one
        # right-hand axis when plotted on the same ax.
        ax2 = getattr(ax, '_neural_angle_twin', None)
        if ax2 is None:
            ax2 = ax.twinx()
            ax._neural_angle_twin = ax2
            ax2.set_ylabel(r'$\hat{\theta}$')
            ax2.set_yticks(tick_locs)
            ax2.set_yticklabels(tick_labels)
            ax2.set_ylim(-np.pi, np.pi)

        neural_theta = self.get_neural_angle(theta)
        angle_label = f'{self.warp_name or "identity"} angle map'
        ax2.plot(theta, neural_theta, color=weight_line.get_color(),
                 linestyle='--', label=angle_label)

        # Build a combined legend so callers do not need to know about the
        # twin axis. Calling this each time means the legend stays in sync
        # as additional models are added to the same ax.
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left')

        if local_plot:
            ax.set_title('Neural Weighting Function')
            plt.show()
        else:
            return ax


    def plot_blocked_signals(self, wb_plot=False, ax=None):
        '''Plots visible targets, their angular direction from the observer,
        their associated neural angles, and also the signal distribution from
        the point of view of the observer normalized so that the maximum signal
        strength is 1.

        Use as a test for _get_target_signals and get_neural_angle.

        Set wb_plot to True if plotting in a Jupyber notebook

        Parameters
        ----------
        wb_plot : bool
            whether or not plotting in a Jupyter notebook
        ax : matplotlib axis, optional
            If provided, only the target-geometry panel (targets plus the
            black perception lines and red dashed neural-direction lines)
            is drawn onto this axis; the polar perception-signal panel is
            skipped and no figure is created or shown. If None (default),
            a new figure is created with both the target-geometry and the
            polar perception-signal subplots, and plt.show() is called.

        Returns
        -------
        ax : matplotlib axis, if ax was provided as an argument.
            Otherwise, None.
        '''

        vis_angles, signals = self._get_target_signals(mesh_signal=True)
        neur_angles = self.get_neural_angle(vis_angles)

        if ax is None:
            local_plot = True
            if wb_plot:
                plt.figure(figsize=(8,5))
            else:
                plt.figure(figsize=(12,6))
            ax1 = plt.subplot(121)
        else:
            local_plot = False
            ax1 = ax

        ###### Target Geometry Plot ######
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

        if not local_plot:
            return ax1

        ###### Perception Signal Plot ######
        ax2 = plt.subplot(122, projection='polar')

        if signals.max() == 0:
            p_func = signals.sum(axis=0)
        else:
            p_func = signals.sum(axis=0)/signals.max()

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


    @contextmanager
    def signal_cache(self):
        '''Memoize get_neural_signals on (focal_angle, focal_loc) within a block.

        The perception signals are a pure function of the observer state
        (heading, location) and the perception geometry, so a root finder that
        evaluates the model repeatedly at states it has already visited pays
        for identical interval arithmetic and spline evaluations again and
        again. Inside this context manager each distinct state is computed once
        and reused; outside it (the default) nothing is cached, and no caller
        that does not opt in is affected.

        The cache lives only for the duration of the block, so it cannot go
        stale against a later warp/weight/target change. Nesting is safe: an
        inner block reuses the cache an outer block already installed. The
        cached arrays are shared between callers, so treat them as read-only.

        The key is the *exact* float state, deliberately not a tolerance: the
        Jacobians probe theta +- h (h = 1e-6 in _coupled_jacobian, 1e-7 in
        _self_consistent_jac), and a key that merged those probes with the base
        point would hand back the same signals for all three. Since perception
        is the ONLY theta dependence in dgamma_dt, that would zero the theta
        column of the coupled Jacobian, making det(J) = 0 and reporting every
        equilibrium unstable.
        '''
        if self.__dict__.get('_signal_cache') is not None:
            yield                      # an enclosing block already owns one
            return
        self._signal_cache = {}
        try:
            yield
        finally:
            del self._signal_cache     # falls back to the class attribute None


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

        See Also
        --------
        signal_cache : context manager that memoizes this call on the observer
            state, for solvers that revisit the same states many times.
        '''

        if focal_angle is None:
            focal_angle = self.focal_angle
        if focal_loc is None:
            focal_loc = self.focal_loc

        # Inside a signal_cache() block, hand back the arrays already computed
        # for this exact observer state (see signal_cache for the exact-key
        # requirement).
        cache = self._signal_cache
        if cache is not None:
            key = (float(focal_angle), float(focal_loc[0]), float(focal_loc[1]))
            cached = cache.get(key)
            if cached is not None:
                return cached

        angles, rho = self._get_target_signals(focal_angle=focal_angle,
                                              focal_loc=focal_loc)
        if angles.size == 0:
            signals = (angles, rho)
        else:
            signals = (self.get_neural_angle(angles), rho)

        if cache is not None:
            cache[key] = signals
        return signals



class NeuralBandModel:
    '''This class takes in a PerceptionModel and uses the neural angles and group 
    sizes to chose a consensus neural direction based on a discrete Ising 
    Hamiltonian. It then translates this into a consensus physical direction. 
    Relies on get_neural_signals from the PerceptionModel to obtain 
    neural angles and relative neural group size.'''

    def __init__(self, percep_model=None, beta=10, K=2):
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
        beta : float
            Inverse neural temperature: the Boltzmann factor
            beta = E/(k_B*temp) of the Glauber dynamics, where E is the energy
            scale of the Ising Hamiltonian, k_B is the Boltzmann constant, and
            temp is the temperature in Kelvin. It sets how sharply the neural
            ring commits to the consensus direction: large beta is cold (low
            noise, sharp commitment), small beta is hot (diffuse). Because it
            is a property of the neural ring and not of the scene, it does not
            scale with the number of targets. Default 10.
        K : float
            Coupling strength for Kuramoto turning speed. Used in models of
            walkers only. Default of 2 in order to match sine argument of ego/2.
        '''

        self.beta = beta
        self.K = K

        if percep_model is None:
            self.percep_model = PerceptionModel()
        else:
            assert isinstance(percep_model,PerceptionModel),\
            "percep_model must be a PerceptionModel object."
            self.percep_model = percep_model

        # Initial gamma value with a small coherence strength of 1e-5.
        # Phase is 0 in neural angle space (corresponding to straight ahead).
        self.gamma = complex(1e-5)

        # Random number generator for certain processes within the class;
        #   Can seed here for reproducability.
        #seed = 3
        self.rng = np.random.default_rng()


    def dgamma_dt(self, t=None, gamma=None, focal_angle=None, focal_loc=None,
                  signals=None):
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
            If None, use self.gamma.
        focal_angle : float, optional
            the current heading (angle) of the observer. If None, use 
            self.percep_model.focal_angle.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use the 
            self.percep_model.focal_loc.
        signals : tuple of (neural_angles, rho), optional
            Precomputed neural angles and normalized group sizes for each 
            visible target (for when the heading is held fixed across evaluations).
            If None, compute them from the perception model.

        Returns
        -------
        dgamma_dt : complex
            time derivative of gamma according to the Ising model
        '''

        if gamma is None:
            gamma = self.gamma
        Theta = np.angle(gamma)
        R = np.abs(gamma)

        if signals is None:
            neur_angles, rho = self.percep_model.get_neural_signals(
                focal_angle, focal_loc)
        else:
            neur_angles, rho = signals
        if neur_angles.size == 0:
            return -gamma
        
        # Compute the sum over all target angles.
        # Suppress overflow warnings: they indicate R is too large and are 
        #   likely the result of whatever numerical algorithm is being used
        #   trying out a bad R value.
        with np.errstate(over='ignore'):
            summands = rho*np.exp(1j*neur_angles)/(1+np.exp(
                -2*self.beta*R*np.cos(neur_angles-Theta)))

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


    def _self_consistent_jac(self, x, focal_loc, h=1e-7):
        '''Central-difference Jacobian of `_self_consistent_eq`.

        Supplied to the `hybr` polish in `sc_equilib` to make root-finding
        equivariant under the theta -> -theta mirror. `_self_consistent_eq`
        is y-flip symmetric to ~1e-20 (negating theta negates only the
        imaginary residual), but hybr's *built-in* Jacobian uses one-sided
        forward differences whose positive `+h` step direction breaks that
        symmetry: the iterate trajectory from a start is then not the mirror
        of the trajectory from the sign-flipped start, so near a saddle-node
        the polish can converge to different members of the equilibrium pair
        at (x, y) vs (x, -y) -- visible as y-asymmetry in bifurcation diagrams
        that should be mirror-symmetric. A central difference perturbs both
        +h and -h, so it is invariant under theta -> -theta; with it the whole
        hybrj Newton/dogleg iteration is equivariant and the self-consistent
        equilibrium set is exactly mirror-symmetric. (The forward/backward
        dgamma_dt evaluations also make the Jacobian estimate second-order
        accurate, which the polish doesn't need but doesn't hurt.)

        Parameters
        ----------
        x : length 2 ndarray
            [theta, R] as in `_self_consistent_eq`.
        focal_loc : array-like of length 2
            (x,y) location of the observer.
        h : float, optional
            finite-difference step (default 1e-7).

        Returns
        -------
        2x2 ndarray : d[Re,Im] / d[theta, R]
        '''
        x = np.asarray(x, dtype=float)
        n = x.size
        J = np.empty((n, n))
        for i in range(n):
            step = np.zeros(n)
            step[i] = h
            f_plus = self._self_consistent_eq(x + step, focal_loc)
            f_minus = self._self_consistent_eq(x - step, focal_loc)
            J[:, i] = (f_plus - f_minus) / (2 * h)
        return J


    def gamma_equilib(self, focal_angle=None, focal_loc=None,
                      stability_criterion='reduced'):
        '''Find gamma equilibria of dgamma/dt at a fixed observer heading.

        Multistart root-finding seeded on the polar circle of radius 0.5.
        Treats the observer's heading as given (not solved-for) and finds
        gamma values that are equilibria of dgamma_dt at that heading.
        Useful for analyzing the gamma landscape as the observer rotates,
        but NOT the right tool for bifurcation diagrams in (x, y): see
        sc_equilib() for that.

        Parameters
        ----------
        focal_angle : float, optional
            The observer's heading. If None, use self.percep_model.focal_angle.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use
            self.percep_model.focal_loc.
        stability_criterion : {'reduced', 'discrim_a'}
            'reduced' (default): timescale-separated test -- fast gamma block
              A Hurwitz AND slow Schur complement d - c A^{-1} b < 0. The
              criterion consistent with gamma slaved to equilibrium (see
              `_discrim_reduced`).
            'discrim_a': legacy 2x2 gamma-only discriminant (fast block only).

        Returns
        -------
        gamma_eqs : list of complex
            equilibrium gamma values at the fixed heading.
        stability : list of bool
            stability of each equilibrium.
        '''
        if focal_angle is True:
            raise TypeError(
                "gamma_equilib(focal_angle=True) is no longer supported; "
                "call sc_equilib(focal_loc=..., "
                "stability_criterion=...) instead to find self-consistent "
                "equilibria (heading = allocentric consensus direction).")
        if stability_criterion == 'reduced':
            stability_test = self._discrim_reduced
        elif stability_criterion == 'discrim_a':
            stability_test = self._discrim_A
        else:
            raise ValueError(
                f"stability_criterion must be 'reduced' or 'discrim_a', "
                f"got {stability_criterion!r}")
        if focal_angle is None:
            focal_angle = self.percep_model.focal_angle
        init_angles = np.linspace(-np.pi, np.pi-0.01)
        final_gammas = []
        stability = []
        # The heading is held fixed here, so every dgamma_dt evaluation in the
        # whole multistart -- all 50 hybr solves -- shares one perception
        # state; only the stability test steps off it (theta +- h). Memoizing
        # the signals collapses ~1400 identical recomputations to one.
        with self.percep_model.signal_cache():
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
                        stability.append(stability_test(
                            gamma_eq, focal_angle, focal_loc))
        return final_gammas, stability


    def sc_equilib(self, focal_loc=None, stability_criterion='reduced',
                   return_R=False):
        '''Find self-consistent equilibria where heading = consensus direction.

        A self-consistent equilibrium is a fixed point of the coupled
        (gamma_re, gamma_im, theta) system: the observer is facing its
        own consensus direction, so the egocentric consensus angle is zero
        and gamma = R + 0j in the neural frame. Solve
        dgamma_dt(R + 0j, theta, focal_loc) = 0 for both theta (the
        allocentric heading) and R (coherence strength).

        Use this for bifurcation diagrams in (x, y): it finds the physically
        meaningful equilibria of the coupled heading-consensus system.

        Strategy: scan Im(dgamma_dt) across theta at a probe R to find
        sign changes, use brentq to pin down candidate theta values, then
        polish each candidate with a 2D hybr root finder. The Jacobian is
        near-block-diagonal (Im depends mostly on theta, Re depends mostly
        on R), so the Im scan seeds theta well and hybr converges reliably.

        Parameters
        ----------
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use
            self.percep_model.focal_loc.
        stability_criterion : {'reduced', 'discrim_a'}
            'reduced' (default): timescale-separated ("block") test. Stable
              iff the fast gamma block A is Hurwitz (gamma-equilibrium branch
              attracting) AND the slow Schur complement
              lam_slow = d - c A^{-1} b < 0 (the slaved heading flow
              dtheta/dt = g(h(theta)) is stable). This is the criterion
              consistent with gamma run to equilibrium before each heading
              step -- the dynamics the walker actually integrates. There is
              deliberately no fully-coupled criterion; see `_discrim_reduced`.
            'discrim_a': legacy 2x2 gamma-only discriminant (the original
              `_discrim_A` test; the fast block alone). Retained for
              comparison plots.
        return_R : bool
            If True, also return the coherence R = |gamma| at each
            equilibrium (needed for (x, R) branch diagrams). Default False
            preserves the 2-tuple return.

        Returns
        -------
        angle_eqs : list of float
            allocentric heading at each self-consistent equilibrium.
        R_eqs : list of float
            coherence R at each equilibrium. ONLY returned if return_R=True
            (then the return is the 3-tuple (angle_eqs, R_eqs, stability)).
        stability : list of bool
            stability of each equilibrium.
        '''
        if stability_criterion == 'reduced':
            stability_test = self._discrim_reduced
        elif stability_criterion == 'discrim_a':
            stability_test = self._discrim_A
        else:
            raise ValueError(
                f"stability_criterion must be 'reduced' or 'discrim_a', "
                f"got {stability_criterion!r}")

        # theta is a solve variable here, not a constant, so the signals
        # cannot be hoisted -- but the same headings recur constantly: each
        # scan node is evaluated at all three probe radii, the hybr polish
        # revisits its iterates, and the residual check and stability test
        # land back on the converged theta. Memoizing on the exact state
        # removes those repeats (~60-70% of the perception calls) without
        # changing what is computed.
        with self.percep_model.signal_cache():
            theta_mesh = np.linspace(-np.pi, np.pi, 100)
            final_angles = []
            stability = []

            # Collect candidate theta seeds from sign changes in Im(dgamma_dt),
            # scanning at SEVERAL probe radii. A single probe (formerly R=0.5)
            # only "sees" equilibria whose R sits near it: near a saddle-node a
            # stable/unstable pair can sit at R~0.6 and produce NO Im
            # sign-change at R=0.5, so its stable member was silently dropped
            # -- undercounting the diagram and leaving a spurious no-commit
            # ("grey") basin where the slaved flow in fact commits to that
            # attractor. Pooling seeds over probe radii catches these; every
            # seed is still hybr-polished, residual-verified and deduped below,
            # so extra probes can only surface genuine equilibria, never
            # spurious ones. Each seed carries the probe R it was found at so
            # the polish starts near the right coherence.
            candidates = []   # (theta_seed, R_seed)
            for R_probe in (0.3, 0.5, 0.7):
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
                            candidates.append((theta_c, R_probe))
                        except ValueError:
                            pass
            # Always include theta=0 and theta=+-pi as candidates since
            # Im(dgamma_dt) is often zero there by symmetry but the sign
            # change can be narrower than the mesh spacing.
            for theta_extra in [0.0, np.pi, -np.pi]:
                candidates.append((theta_extra, 0.5))

            # Polish each candidate with the 2D root finder
            final_Rs = []
            for theta_c, R_seed in candidates:
                sol = root(self._self_consistent_eq, [theta_c, R_seed],
                           args=(focal_loc,), method='hybr', tol=1e-10,
                           jac=self._self_consistent_jac)
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
                if np.abs(residual) > 1e-4:
                    continue

                # Dedup by both theta and R: near a saddle-node bifurcation
                # two genuine equilibria can share a theta to within the
                # angular tolerance while differing in R, so theta alone is
                # not enough to distinguish them.
                close_idx = None
                for j, (existing_angle, existing_R) in enumerate(
                        zip(final_angles, final_Rs)):
                    angle_diff = np.abs(convert_angles(
                        theta_eq - existing_angle))
                    R_diff = np.abs(R_eq - existing_R)
                    if angle_diff < 0.02 and R_diff < 0.01:
                        close_idx = j
                        break
                gamma_eq = R_eq + 0j
                if close_idx is None:
                    final_angles.append(theta_eq)
                    final_Rs.append(R_eq)
                    stability.append(stability_test(
                        gamma_eq, theta_eq, focal_loc))
                elif not stability[close_idx]:
                    # The first-kept member of this near-coincident pair is
                    # UNSTABLE. Near a saddle-node a stable and an unstable
                    # equilibrium can collide to within the (angle, R) dedup
                    # tolerance even with the full (theta, R) test; which
                    # member is found first depends on the brentq candidate
                    # order, which REVERSES under the y-mirror, so keeping "the
                    # first" makes the survivor's stability -- and the stable
                    # count -- flip between (x, y) and (x, -y). Prefer the
                    # STABLE member (a mirror-invariant choice) so the count
                    # stays correct and y-symmetric.
                    if stability_test(gamma_eq, theta_eq, focal_loc):
                        final_angles[close_idx] = theta_eq
                        final_Rs[close_idx] = R_eq
                        stability[close_idx] = True

        if return_R:
            return final_angles, final_Rs, stability
        return final_angles, stability
    

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


    def run_dgamma_dt(self, focal_angle=None, focal_loc=None, init_gamma=None,
                      t_Final=100, warn=True):
        '''Solve the dgamma/dt equation and look for solutions to approach a
        stable equilibrium for the neural band model. Uses LSODA solver from
        scipy with a real-valued reformulation so that stiff-capable solvers
        can be applied.

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
            maximum total time for integration (default 100).

        Returns
        -------
        gamma_equilib : complex float
            Equilibrium gamma value reached by integration.
        '''

        if init_gamma is None:
            init_gamma = self.gamma

        # The heading is fixed for the whole integration, so the perception
        # signals are invariant: fetch them ONCE and reuse on every RHS
        # evaluation (the solver calls the RHS dozens of times per
        # relaxation). Exact (~1e-14) and ~9x faster than recomputing per
        # call -- speeds every caller, walkers included (dtheta_dt ->
        # run_dgamma_dt each heading step).
        signals = self.percep_model.get_neural_signals(focal_angle, focal_loc)

        # Reformulate the complex ODE as a real 2D system so that LSODA
        # (a stiff-capable solver) can be used. Near-zero Jacobian eigenvalues
        # in the slow manifold between equilibria make LSODA significantly
        # more efficient than explicit RK45 with restarted windows.
        def dgamma_real(t, y):
            gamma = y[0] + 1j*y[1]
            dg = self.dgamma_dt(t, gamma, focal_angle, focal_loc,
                                signals=signals)
            return [dg.real, dg.imag]

        y0 = [np.real(init_gamma), np.imag(init_gamma)]
        sol = solve_ivp(dgamma_real, [0, t_Final], y0, method='LSODA')

        result = sol.y[0,-1] + 1j*sol.y[1,-1]
        tol = 1e-4
        if warn and np.abs(self.dgamma_dt(None, result, focal_angle,
                                          focal_loc, signals=signals)) > tol:
            print("Warning: Integration may not have reached equilibrium.")
        return result


    def dtheta_dt(self, t=None, theta=None, gamma=None, focal_loc=None,
                  t_Final=100):
        '''Turning rate based on the neural band model coherence value.

        Runs dgamma_dt to steady state and returns the Kuramoto-style torque
        K*R*sin(Theta/2), where Theta = arg(gamma) is the neural consensus
        angle and R = |gamma|.

        Parameters
        ----------
        t : float, optional
            time variable for ODE solver compatibility. Not used.
        theta : float, optional
            current heading angle of the observer in allocentric coordinates.
            If None, use self.percep_model.focal_angle.
        gamma : complex float, optional
            Equilibrium coherence value from neural band. If None, it will be
            computed by solving dgamma_dt to t_Final (using run_dgamma_dt) and
            taking the final solution as an equilibrium value. The IC for
            run_dgamma_dt will be self.gamma, and self.gamma will be updated
            to the result. If a complex value is provided, it is used directly
            (no ODE solve).
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use the percep_model's
            focal_loc.
        t_Final : float, optional (default=100)
            final time for integration of dgamma_dt if gamma is None.
        '''
        if theta is None:
            theta = self.percep_model.focal_angle
        if gamma is None:
            self.gamma = self.run_dgamma_dt(focal_angle=theta, focal_loc=focal_loc,
                                             init_gamma=None, t_Final=t_Final)
            gamma = self.gamma
        # Half-angle torque in the NEURAL consensus angle Theta = arg(gamma).
        # Theta is the canonical [-pi, pi] coordinate (the warp maps the
        # facing-away heading to +-pi for every perception model), so the
        # "/2" normalization is perception-model-independent and the torque
        # is zero only when consensus is straight ahead (Theta=0) and maximal
        # +-K*R when consensus is directly behind (Theta=+-pi). The +-pi point
        # carries an intentional +K*R <-> -K*R jump discontinuity (the physical
        # left/right fork facing away), resolved by roundoff/noise. We use
        # arg(gamma) directly rather than its inverse-warped egocentric angle:
        # the turning law is then a direct function of gamma* with no
        # dependence on the perception model's inverse mapping.
        Theta = np.angle(gamma)
        R = np.abs(gamma)
        return self.K * R * np.sin(Theta/2)


    def _coupled_jacobian(self, gamma_star, focal_angle, focal_loc, h=1e-6):
        '''Central-difference 3x3 Jacobian of the coupled
        (gamma_re, gamma_im, theta) system at (gamma_star, focal_angle),
        with dtheta/dt = K*R*sin(arg(gamma)/2).

        Used by `_discrim_reduced`, which needs the fast gamma block A and
        the sign of det(J). NOTE: the FULL eigenvalues of this matrix are NOT
        a stability verdict for the model -- see `_discrim_reduced` for why.
        '''
        gr0 = float(gamma_star.real)
        gi0 = float(gamma_star.imag)
        th0 = float(focal_angle)

        def coupled_rhs(gr, gi, th):
            gamma = gr + 1j*gi
            dg = self.dgamma_dt(gamma=gamma, focal_angle=th,
                                focal_loc=focal_loc)
            R = np.abs(gamma)
            dth = self.K * R * np.sin(np.angle(gamma)/2)
            return np.array([dg.real, dg.imag, dth])

        J = np.zeros((3, 3))
        for k, (dr, di, dt) in enumerate([(h, 0, 0), (0, h, 0), (0, 0, h)]):
            f_plus = coupled_rhs(gr0+dr, gi0+di, th0+dt)
            f_minus = coupled_rhs(gr0-dr, gi0-di, th0-dt)
            J[:, k] = (f_plus - f_minus) / (2*h)
        return J


    def _discrim_reduced(self, gamma_star, focal_angle, focal_loc,
                         h=1e-6, tol=1e-8):
        '''Timescale-separated ("block") stability test: the criterion that
        is consistent with the slaved gamma dynamics used everywhere else in
        the model (mean-field neural ring, gamma run to equilibrium before
        each dtheta step in plot_walkers, no gamma-noise).

        Partition the coupled Jacobian J = [[A, b], [c, d]] with A the 2x2
        gamma block, b = d(dgamma)/dtheta, c = d(dtheta)/dgamma,
        d = d(dtheta)/dtheta. The equilibrium is stable iff BOTH:

          1. Fast layer -- the gamma-equilibrium branch gamma = h(theta) is
             attracting: A is Hurwitz (all eig(A) real-part < -tol). For the
             cosine kernel this coincides with `_discrim_A`.
          2. Slow layer -- the slaved flow dtheta/dt = g(h(theta)) is stable:
             its linearization is the Schur complement of A in J,
                 lam_slow = d - c A^{-1} b,
             the leading (lambda->0) term of the exact eigenvalue equation
             lam = d - c (A - lam I)^{-1} b; setting lambda = 0 inside the
             resolvent is exactly the adiabatic elimination of gamma. We need
             only its SIGN, which we read off without inverting A: by the
             block-determinant identity det(J) = det(A) * lam_slow, and
             det(A) > 0 for a Hurwitz 2x2 A, so
                 lam_slow < 0   <=>   det(J) < 0.
             Using det(J) keeps the slow test well-conditioned near a
             gamma-fold (eig(A) -> 0), where the Schur complement itself
             diverges (|lam_slow| -> inf) even though the sign is unambiguous
             and det(J) stays bounded. The A-Hurwitz gate above already
             rejects the genuinely singular case before this point.

        The slow flow is 1-D in theta and cannot oscillate, so this test
        does not see gamma-theta Hopf instabilities. That is correct, not a
        limitation -- see below.

        Why there is no fully-coupled criterion
        ---------------------------------------
        `dgamma_dt` is not a first-principles equation of motion: it is the
        rank-2 readout of the K-dimensional Glauber population dynamics
        (one n_k per visible target), obtained by differentiating
        gamma = sum_k n_k exp(i*neural_angle_k) and keeping ONLY the dn_k/dt
        term. The dropped term,

            -i * dtheta/dt * sum_k n_k * U'(theta_k) * exp(i*neural_angle_k),

        arises because the neural angles themselves move as the observer
        turns. It vanishes identically when dtheta/dt = 0, so every
        self-consistent equilibrium -- and this criterion, and the fast-block
        `_discrim_A` -- is unaffected. But OFF equilibrium the gamma-ODE is
        only valid under the timescale separation, which is exactly the
        assumption made here (and in the project preprint, whose navigational
        model is driven by the EQUILIBRIUM consensus gamma*, and in Sridhar et
        al. 2021 SI sec 1.7, whose Eq [15] likewise drops it and is used only
        at fixed target bearings).

        Consequently there is deliberately NO 'coupled' criterion. Taking the
        full 3x3 eigenvalues of `_coupled_jacobian` linearizes an incomplete
        equation and reports instabilities the model does not have: on the
        documented vonmises k=0.55 island it flagged 6 equilibria as
        Hopf-unstable (with an attracting limit cycle) where the exact
        population system is stable at every one. Nor can it be patched at the
        gamma level -- under a non-identity warp U the dropped term is not
        proportional to gamma (U' differs per target), so it is not a function
        of gamma at all.

        A genuine coupled analysis therefore requires the (K+1)-dimensional
        (n_1..n_K, theta) population system, where it also needs an explicit
        neural timescale tau_0: the verdict depends on kappa*tau_0, and the
        gamma-ODE implicitly assumes kappa*tau_0 = 2 (turning FASTER than
        neural relaxation), the opposite of the separation assumed everywhere
        else. This criterion is the tau_0 -> 0 limit, which is the dynamics
        `plot_walkers` and `_basin_destination` actually integrate.
        '''
        J = self._coupled_jacobian(gamma_star, focal_angle, focal_loc, h)
        A = J[:2, :2]
        # Fast (gamma) layer must be attracting for the slow reduction to be
        # meaningful: if A is not Hurwitz the branch h(theta) is not a stable
        # sheet. This gate also means det(A) > 0 below.
        if not np.all(np.real(np.linalg.eigvals(A)) < -tol):
            return False
        # Slow layer: sign(lam_slow) = sign(det J) (since det(A) > 0), computed
        # without the ill-conditioned A^{-1} of the Schur complement.
        return bool(np.linalg.det(J) < 0.0)


    def _discrim_A(self, gamma_star, focal_angle, focal_loc):
        '''Determines gamma-stability from the analytic 2x2 fast block (the
        free-energy Hessian H_Fhat) at a self-consistent equilibrium.

        gamma is stable iff H_Fhat is positive definite, which for a symmetric
        2x2 means BOTH the transverse curvature is positive (the classic
        A < 1 condition) AND det H_Fhat > 0. A < 1 alone is necessary but NOT
        sufficient: it is only the tangential diagonal entry and misses radial
        folds (H_rr < 0) and off-diagonal (R-Theta) saddles (det < 0). This is
        the complete fast-block test -- identical to the A-Hurwitz gate of
        _discrim_reduced for the cosine kernel -- but WITHOUT the slow
        heading-tracking (Schur) mode; use 'reduced' for that. See
        theory/free_energy_derivation.md sec 6.1.

        This is based on a cosine kernel.

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
        with np.errstate(over='ignore'):
            # w_j = (beta/2) rho_j sech^2(beta R cos(Theta - theta_j)) >= 0 is
            # the per-target weight in the Cartesian free-energy Hessian
            # H_Fhat = I - sum_j w_j vhat_j vhat_j^T (gradient flow: fast
            # block A_block = -H_Fhat). Work in the frame rotated to gamma so
            # x = radial (R), y = tangential (arg gamma); det is rotation-
            # invariant. cc/ss are cos/sin of the target angle relative to
            # the consensus.
            w = (self.beta/2*rho
                 / np.cosh(self.beta*R*np.cos(Theta-neur_angles))**2)
            cc = np.cos(Theta - neur_angles)
            ss = np.sin(Theta - neur_angles)
            A = np.sum(w*ss**2)            # transverse: 1 - H_tt (the old scalar)
            H_rr = 1 - np.sum(w*cc**2)     # radial curvature
            H_rt = -np.sum(w*cc*ss)        # R-Theta coupling
            detH = H_rr*(1 - A) - H_rt**2  # det H_Fhat
        # Full gamma-stability = H_Fhat positive definite = transverse
        # curvature positive (A < 1) AND det > 0. Strict (matches the
        # A-Hurwitz gate of _discrim_reduced).
        return bool((A < 1) and (detH > 0))


    def _process_point(self, args):
        '''Helper function for processing mesh points in plot_direction_mesh.

        Parameters
        ----------
        args : tuple
            (ii, jj, X, Y, stability_criterion) where ii and jj are the mesh
            indices, X and Y are the mesh coordinate arrays, and
            stability_criterion is forwarded to gamma_equilib.

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

        ii, jj, X, Y, stability_criterion = args
        focal_loc = np.array([X[jj,ii], Y[jj,ii]])

        final_angles, stability = self.sc_equilib(
            focal_loc=focal_loc,
            stability_criterion=stability_criterion)

        return ii, jj, final_angles, stability
    

    def plot_direction_mesh(self, xlim=(0,6), num_x=19, ylim=(-3.5,3.5), num_y=19,
                            pool=None, ax=None, title=None, wb_plot=False,
                            stability_criterion='reduced'):
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
        title : str, optional
            title for the quiver plot.
        wb_plot : bool
            whether or not plotting in a Jupyter notebook
        stability_criterion : {'reduced', 'discrim_a'}
            Which stability test to apply (forwarded to sc_equilib).
            'reduced' (default) is the timescale-separated test (fast gamma
            block Hurwitz + slow Schur complement < 0); 'discrim_a' uses the
            legacy gamma-only test for comparison plots.

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
                    results.append(self._process_point(
                        (ii, jj, X, Y, stability_criterion)))
        else:
            args_list = [(ii, jj, X, Y, stability_criterion)
                         for ii in range(num_x) for jj in range(num_y)]
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
        if not U_list:
            warnings.warn(
                "plot_direction_mesh: no equilibria found at any mesh point; "
                "rendering targets only.")
        else:
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
        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title('Neural band equilibrium plot')
        if local_plot:
            fig.legend(loc='outside center right')
            plt.show()
        else:
            return ax


    def _count_stable_at(self, args):
        '''Helper function for plot_bifurcation_diagram: evaluate the number
        of stable self-consistent equilibria at a single (x,y) location.

        Parameters
        ----------
        args : tuple
            (key, x, y, stability_criterion), where key is an arbitrary
            hashable identifier (used by the caller to reassemble results),
            (x,y) is the focal location in world coordinates, and
            stability_criterion is forwarded to gamma_equilib.

        Returns
        -------
        key : hashable
            echoed back from the input for lookup
        count : int
            number of stable self-consistent equilibria at (x,y)
        '''
        key, x, y, stability_criterion = args
        focal_loc = np.array([x, y])
        _, stability = self.sc_equilib(
            focal_loc=focal_loc,
            stability_criterion=stability_criterion)
        return key, int(sum(stability))


    def plot_bifurcation_diagram(self, xlim=(0,6), num_x=29, ylim=(-3.5,3.5),
                                 num_y=29, refinement_levels=3, max_count=None,
                                 boundary_dilation=1,
                                 pool=None, ax=None, title=None, wb_plot=False,
                                 stability_criterion='reduced',
                                 overlay_basins=False, basin_R_seed=0.15,
                                 basin_n_coarse=64, basin_n_bisect=12,
                                 basin_placement='lattice', basin_nx=6,
                                 basin_ny=4, basin_min_sep_factor=2.2,
                                 basin_max_sep_factor=5.5, basin_min_area=4,
                                 basin_target_margin=0.15,
                                 basin_wheel_radius=None, basin_bg_alpha=0.9):
        '''Plot a 2D colormap showing the number of stable self-consistent
        equilibria as a function of observer (x,y) location.

        Starts from a coarse ``num_x`` by ``num_y`` grid and adaptively
        subdivides cells whose four corners disagree on the number of stable
        equilibria, up to ``refinement_levels`` times. Evaluated points are
        cached so that corners shared between cells are not recomputed.

        Parameters
        ----------
        xlim : (xmin,xmax) tuple of floats
            x limits for the base mesh, inclusive
        num_x : int
            number of steps in x direction for the base mesh
        ylim : (ymin,ymax) tuple of floats
            y limits for the base mesh, inclusive
        num_y : int
            number of steps in y direction for the base mesh
        refinement_levels : int
            number of adaptive subdivision passes. 0 => base mesh only.
            Each pass halves cell size at boundaries where the stable count
            changes. Final virtual grid resolution is
            ((num_x-1)*2**L + 1) by ((num_y-1)*2**L + 1).
        max_count : int, optional
            maximum expected number of stable equilibria. Pins the color
            scale so that count=N maps to the same color across multiple
            calls (e.g. side-by-side subplots comparing models). If None,
            auto-detected from the data. Values in the data exceeding
            max_count are clipped with a warning.
        boundary_dilation : int
            At each refinement pass, also refine cells that share a corner
            with a cell whose own corners disagree, ``boundary_dilation``
            steps outward. Default 1 widens the refined band by one cell
            per side, smoothing stair-step artifacts where a fine-grained
            transition meets a coarse settled neighbour. 0 reproduces the
            legacy strict-corner-disagreement behaviour. Cost roughly
            doubles per dilation step at the base pass; later passes are
            largely unaffected.
        pool : multiprocessing.Pool, optional
            If provided, evaluate new points at each refinement level in
            parallel.
        ax : matplotlib axis, optional
            If provided, plot on this axis instead of creating a new figure.
        title : str, optional
            title for the plot
        wb_plot : bool
            whether or not plotting in a Jupyter notebook
        stability_criterion : {'reduced', 'discrim_a'}
            Which stability test to apply when counting stable equilibria.
            'reduced' (default) is the timescale-separated test consistent
            with the slaved-gamma walker: fast gamma block Hurwitz AND slow
            Schur complement d - c A^{-1} b < 0. 'discrim_a'
            uses the legacy gamma-only test and over-reports stability where
            heading coupling contributes a positive eigenvalue; intended for
            side-by-side comparison plots.
        overlay_basins : bool
            If True, dim the count map (``basin_bg_alpha``) and draw a
            basin-of-attraction wheel per sampled location (heading-basin
            annulus + direction arrows, colored by basin rank).
        basin_placement : {'lattice', 'grid'}
            How wheel locations are chosen when overlay_basins is True.
            'lattice' (default): a structure-aware, symmetric irregular
            rectilinear lattice seeded on the multistable regions (capped at
            ``basin_nx`` x-lines and ``basin_ny`` y-lines per side of the
            axis). 'grid': a plain regular grid, ``basin_nx`` columns by
            ``2*basin_ny - 1`` rows, evenly spaced over the domain. In both
            cases wheels whose disk would overlap a target (within
            ``basin_target_margin``) are dropped.

        Returns
        -------
        ax : matplotlib axis, if ax was provided as an argument.
            Otherwise, None.
        '''
        assert refinement_levels >= 0, "refinement_levels must be >= 0"
        assert boundary_dilation >= 0, "boundary_dilation must be >= 0"

        L = refinement_levels
        step0 = 2**L
        # Virtual fine-grid resolution (points, not cells)
        NI = (num_x - 1)*step0 + 1
        NJ = (num_y - 1)*step0 + 1

        def idx_to_xy(I, J):
            x = xlim[0] + (xlim[1] - xlim[0])*I/(NI - 1)
            y = ylim[0] + (ylim[1] - ylim[0])*J/(NJ - 1)
            return x, y

        cache = {}  # (I, J) -> stable count

        def evaluate_points(keys):
            keys = [k for k in keys if k not in cache]
            if not keys:
                return
            args_list = [(k, *idx_to_xy(*k), stability_criterion)
                         for k in keys]
            if pool is None:
                for args in args_list:
                    key, count = self._count_stable_at(args)
                    cache[key] = count
            else:
                results = pool.map(self._count_stable_at, args_list)
                for key, count in results:
                    cache[key] = count

        # 1. Evaluate base grid
        base_keys = [(i*step0, j*step0)
                     for i in range(num_x) for j in range(num_y)]
        evaluate_points(base_keys)

        # 2. Initial cells: each cell is (I_ll, J_ll, side) in virtual units
        cells = [(i*step0, j*step0, step0)
                 for i in range(num_x - 1) for j in range(num_y - 1)]

        # 3. Refinement loop
        for _ in range(L):
            # Pass A: classify each cell purely by its own corner agreement.
            to_refine = []
            settled = []
            for cell in cells:
                I, J, s = cell
                corner_counts = {cache[(I, J)], cache[(I+s, J)],
                                 cache[(I, J+s)], cache[(I+s, J+s)]}
                if len(corner_counts) == 1:
                    settled.append(cell)
                else:
                    to_refine.append(cell)

            # Pass B: dilate the refinement set so that quiet cells touching
            # a disagreement cell (sharing any corner index) get promoted
            # too. Each round expands the refined band by one cell. Corner
            # indices live on the same virtual grid for all cell sizes, so
            # this naturally couples mixed-grain neighbours that share an
            # edge or corner.
            for _round in range(boundary_dilation):
                if not to_refine:
                    break
                boundary_corners = set()
                for I, J, s in to_refine:
                    boundary_corners.update(
                        [(I, J), (I+s, J), (I, J+s), (I+s, J+s)])
                promoted = []
                kept_settled = []
                for cell in settled:
                    I, J, s = cell
                    cell_corners = {(I, J), (I+s, J),
                                    (I, J+s), (I+s, J+s)}
                    if cell_corners & boundary_corners:
                        promoted.append(cell)
                    else:
                        kept_settled.append(cell)
                if not promoted:
                    break
                to_refine.extend(promoted)
                settled = kept_settled

            # Pass C: schedule midpoint evaluations for every cell that
            # will be refined, then evaluate as a single batch (one pool
            # round-trip per refinement pass).
            new_points = set()
            for I, J, s in to_refine:
                half = s // 2
                for m in [(I+half, J), (I, J+half),
                          (I+s, J+half), (I+half, J+s),
                          (I+half, J+half)]:
                    if m not in cache:
                        new_points.add(m)
            evaluate_points(new_points)

            cells = settled
            for I, J, s in to_refine:
                half = s // 2
                cells.append((I, J, half))
                cells.append((I+half, J, half))
                cells.append((I, J+half, half))
                cells.append((I+half, J+half, half))

        # 4. Rasterize leaf cells into an int image at virtual-pixel
        #    resolution. img[row=J, col=I] -> count at virtual pixel (I,J).
        data_max = max(cache.values())
        if max_count is None:
            effective_max = data_max
        else:
            effective_max = max_count
            if data_max > effective_max:
                warnings.warn(
                    "Data contains stable-equilibrium counts up to "
                    f"{data_max} but max_count={effective_max}; "
                    "values above max_count will be clipped.")
        img = np.zeros((NJ - 1, NI - 1), dtype=int)
        for (I, J, s) in cells:
            cLL = cache[(I, J)]
            cLR = cache[(I+s, J)]
            cUL = cache[(I, J+s)]
            cUR = cache[(I+s, J+s)]
            if cLL == cLR == cUL == cUR:
                img[J:J+s, I:I+s] = cLL
            elif s == 1:
                # max depth, single pixel cell, 4 corner values disagree;
                # use lower-left arbitrarily
                img[J, I] = cLL
            else:
                half = s // 2
                img[J:J+half, I:I+half] = cLL
                img[J:J+half, I+half:I+s] = cLR
                img[J+half:J+s, I:I+half] = cUL
                img[J+half:J+s, I+half:I+s] = cUR

        # 5. Plot
        if ax is None:
            local_plot = True
            if wb_plot:
                fig = plt.figure(figsize=(12,6))
            else:
                fig = plt.figure(figsize=(5.5,5))
            ax = plt.subplot(1,1,1)
        else:
            local_plot = False

        cmap = plt.get_cmap('viridis', effective_max + 1)
        norm = BoundaryNorm(boundaries=np.arange(-0.5, effective_max + 1.5),
                            ncolors=effective_max + 1)
        img = np.clip(img, 0, effective_max)
        ax.imshow(img, origin='lower',
                  extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                  aspect='equal', interpolation='nearest',
                  cmap=cmap, norm=norm,
                  alpha=basin_bg_alpha if overlay_basins else None)

        self.percep_model.targets.plot_targets_to_axis(ax)

        # Attach labeled proxy artists so that a later ax.legend() or
        # plt.legend() call picks up one entry per integer count. These are
        # zero-data plots -- they render nothing, but the legend machinery
        # sees them as labeled handles.
        for n in range(effective_max + 1):
            ax.plot([], [], marker='s', markersize=10, linestyle='',
                    color=cmap(norm(n)), label=f'{n}')

        if title is not None:
            ax.set_title(title)
        elif stability_criterion == 'discrim_a':
            ax.set_title('Neural band bifurcation diagram (discrim_A)')
        else:
            ax.set_title('Neural band bifurcation diagram (reduced)')

        if overlay_basins:
            self._overlay_basin_wheels(
                ax, img, lambda i, j: idx_to_xy(i + 0.5, j + 0.5),
                xlim, ylim, pool=pool,
                stability_criterion=stability_criterion,
                R_seed=basin_R_seed, n_coarse=basin_n_coarse,
                n_bisect=basin_n_bisect, placement=basin_placement,
                min_sep_factor=basin_min_sep_factor,
                max_sep_factor=basin_max_sep_factor, min_area=basin_min_area,
                nx_max=basin_nx, ny_max=basin_ny,
                target_margin=basin_target_margin,
                wheel_radius=basin_wheel_radius)

        if local_plot:
            ax.legend(title='# stable\nequilibria', loc='center left',
                      bbox_to_anchor=(1.02, 0.5), frameon=False)
            fig.tight_layout()
            plt.show()
        else:
            return ax


    # ------------------------------------------------------------------
    # Basin-of-attraction wheels: optional overlay for the bifurcation
    # diagram (overlay_basins=True). At an observer location, sweeping the
    # heading around the circle from a neutral (uncommitted) neural state,
    # each heading commits under the slaved-gamma flow to one stable
    # self-consistent direction -- the "basin" of that direction. A wheel
    # shows those heading-basins (annulus arcs) and the directions (arrows).
    # Migrated from the basin_estimation/ vetting prototype.
    # ------------------------------------------------------------------

    def _basin_destination(self, focal_loc, theta0, gamma_seed, stable,
                           dt=0.1, n_steps=2000, conv_tol=1e-5,
                           min_R=0.05, max_dist=0.15):
        """Index into ``stable`` of the direction a neutral observer at
        heading ``theta0`` commits to under the slaved flow (re-equilibrate
        gamma each heading step via run_dgamma_dt, then step theta by the
        half-angle torque), or -1 if it does not commit (no convergence,
        collapsed coherence, or lands away from any stable direction)."""
        theta = theta0
        gamma = gamma_seed
        R = abs(gamma)
        converged = False
        for _ in range(n_steps):
            gamma = self.run_dgamma_dt(focal_angle=theta, focal_loc=focal_loc,
                                       init_gamma=gamma, warn=False)
            R = abs(gamma)
            dtheta = self.dtheta_dt(theta=theta, gamma=gamma,
                                    focal_loc=focal_loc)
            theta = convert_angles(theta + dtheta * dt)
            if abs(dtheta) < conv_tol:
                converged = True
                break
        if (not converged) or R < min_R or len(stable) == 0:
            return -1
        dists = [abs(convert_angles(theta - s)) for s in stable]
        j = int(np.argmin(dists))
        return j if dists[j] < max_dist else -1

    def basin_arcs_at_focal_loc(self, focal_loc, R_seed=0.15, n_coarse=64,
                                n_bisect=12, stability_criterion='reduced'):
        """Heading-basin partition for the stable directions at one location.

        Sweeps the heading from a fixed neutral seed (arg 0, |gamma|=R_seed)
        and records which stable direction each heading commits to under the
        slaved flow, refining basin boundaries by destination-flip bisection.

        Returns a dict with keys 'focal_loc', 'stable' (sorted stable
        directions), 'arcs' (list of (theta_start, theta_end, label) covering
        the circle; label indexes 'stable', or -1 for a no-commit arc), and
        'widths' ({label: total arc width}).
        """
        focal_loc = np.asarray(focal_loc, dtype=float)
        ang, stab = self.sc_equilib(focal_loc=focal_loc,
                                    stability_criterion=stability_criterion)
        stable = sorted(a for a, s in zip(ang, stab) if s)
        if len(stable) == 0:
            return {'focal_loc': tuple(focal_loc), 'stable': [],
                    'arcs': [(-np.pi, np.pi, -1)], 'widths': {-1: 2 * np.pi}}
        if len(stable) == 1:
            return {'focal_loc': tuple(focal_loc), 'stable': stable,
                    'arcs': [(-np.pi, np.pi, 0)], 'widths': {0: 2 * np.pi}}

        seed = R_seed + 0j
        thetas = np.linspace(-np.pi, np.pi, n_coarse, endpoint=False)
        labels = [self._basin_destination(focal_loc, th, seed, stable)
                  for th in thetas]

        step = 2 * np.pi / n_coarse
        seps = []
        for i in range(n_coarse):
            a_lab, b_lab = labels[i], labels[(i + 1) % n_coarse]
            if a_lab == b_lab:
                continue
            lo, hi = thetas[i], thetas[i] + step
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                if self._basin_destination(focal_loc, convert_angles(mid),
                                           seed, stable) == a_lab:
                    lo = mid
                else:
                    hi = mid
            seps.append(convert_angles(0.5 * (lo + hi)))

        if not seps:
            lab = int(labels[0])
            return {'focal_loc': tuple(focal_loc), 'stable': stable,
                    'arcs': [(-np.pi, np.pi, lab)], 'widths': {lab: 2 * np.pi}}

        seps.sort()
        arcs, widths = [], {}
        for k in range(len(seps)):
            s_start = seps[k]
            s_end = seps[(k + 1) % len(seps)]
            span = (s_end - s_start) % (2 * np.pi)
            mid = convert_angles(s_start + 0.5 * span)
            lab = self._basin_destination(focal_loc, mid, seed, stable)
            arcs.append((s_start, convert_angles(s_start + span), lab))
            widths[lab] = widths.get(lab, 0.0) + span
        return {'focal_loc': tuple(focal_loc), 'stable': stable,
                'arcs': arcs, 'widths': widths}

    def _basin_arcs_worker(self, args):
        """Pool-friendly wrapper (one (x,y) per task; mirrors the
        _count_stable_at dispatch pattern)."""
        focal_loc, R_seed, n_coarse, n_bisect, crit = args
        return self.basin_arcs_at_focal_loc(
            focal_loc, R_seed=R_seed, n_coarse=n_coarse, n_bisect=n_bisect,
            stability_criterion=crit)

    @staticmethod
    def _basin_lattice_placement(count_field, xy_of_pixel, wheel_radius,
                                 nx_max=6, ny_max=4, min_sep_factor=2.2,
                                 max_sep_factor=5.5, min_area=4):
        """Symmetric, irregular rectilinear lattice of wheel locations.

        ``nx_max`` / ``ny_max`` are **hard caps** on the line count:
        ``nx_max`` vertical (x) lines, and ``ny_max`` horizontal (y) lines on
        each side of (and including) the x-axis -- so up to ``2*ny_max - 1``
        horizontal lines and ``nx_max * (2*ny_max - 1)`` cells overall.

        For each axis ~0.6*cap lines are seeded from significant regions --
        labeled per count level and represented by the centroid of each
        component's y>=0 half (so symmetric crescent arms, not the on-axis
        full centroid, drive the y-lines), largest area first, no two within
        min_sep; the remaining budget is spent filling the widest gaps
        (> max_sep) largest-first, placing each fill line at the within-gap
        spot whose WORST-placed lattice point stays farthest from a
        count-region boundary (maximin -- boundaries are near-bifurcation and
        give flaky wheels) -- so the big 1-stable expanse still gets covered
        without over-populating. Returns the product X x Y as (x, y) cells.
        Assumes dynamics symmetric about y=0.
        """
        from scipy.ndimage import label, distance_transform_edt
        nrows, ncols = count_field.shape
        xs = np.array([xy_of_pixel(i, 0)[0] for i in range(ncols)])
        ys = np.array([xy_of_pixel(0, j)[1] for j in range(nrows)])
        min_sep = min_sep_factor * wheel_radius
        max_sep = max_sep_factor * wheel_radius

        # boundary-distance field (data units): large = deep in a region
        bnd = np.zeros(count_field.shape, dtype=bool)
        dv = count_field[:-1, :] != count_field[1:, :]
        bnd[:-1, :] |= dv
        bnd[1:, :] |= dv
        dh = count_field[:, :-1] != count_field[:, 1:]
        bnd[:, :-1] |= dh
        bnd[:, 1:] |= dh
        px = 0.5 * (abs(xs[1] - xs[0]) + abs(ys[1] - ys[0]))
        Dmap = distance_transform_edt(~bnd) * px

        def col(x):
            return int(np.clip(np.argmin(np.abs(xs - x)), 0, ncols - 1))

        def row(y):
            return int(np.clip(np.argmin(np.abs(ys - y)), 0, nrows - 1))

        # Seed lines from significant regions, labeling each count level
        # (2, 3, 4, ...) separately so a higher-count pocket nested inside a
        # lower-count region still drives its own line (otherwise everything
        # count>=2 is one connected blob and only its centroid seeds a line).
        # Represent each component by the centroid of its y>=0 HALF, not its
        # full centroid: the multistable regions are ~symmetric about y=0, so a
        # full centroid lands on the axis and never seeds a y-line up on a
        # crescent arm -- the y>=0-half centroid gives the arm's |y| instead,
        # so the lattice samples those arms (where the interesting pockets are).
        ymask = (ys >= 0)[:, None]
        comps = []
        for k in sorted(set(count_field[count_field >= 2].tolist())):
            lab, n = label(count_field == k)
            for c in range(1, n + 1):
                m = (lab == c)
                a = int(m.sum())
                if a < min_area:
                    continue
                mh = m & ymask
                if not mh.any():
                    continue
                js, is_ = np.where(mh)
                comps.append((a, float(xs[int(round(is_.mean()))]),
                              abs(float(ys[int(round(js.mean()))]))))
        comps.sort(key=lambda t: -t[0])

        # reserve ~0.6 of each cap for centroid seeds; the rest is fill budget
        seed_nx = max(1, int(round(0.6 * nx_max)))
        seed_ny = max(1, int(round(0.6 * ny_max)))
        X = []
        for _a, cx, _cy in comps:
            if len(X) >= seed_nx:
                break
            if all(abs(cx - xx) >= min_sep for xx in X):
                X.append(cx)
        Yh = [0.0]
        for _a, _cx, cy in comps:
            if len(Yh) >= seed_ny:
                break
            ay = abs(cy)
            if all(abs(ay - yy) >= min_sep for yy in Yh):
                Yh.append(ay)

        def fill(lines, lo, hi, score, cap):
            lines = sorted(lines)
            while len(lines) < cap:
                edges = [lo] + lines + [hi]
                big = [(edges[k + 1] - edges[k], edges[k], edges[k + 1])
                       for k in range(len(edges) - 1)]
                big = [g for g in big if g[0] > max_sep + 1e-9]
                if not big:
                    break
                _g, a, b = max(big)
                center = 0.5 * (a + b)
                half = 0.5 * min(_g, max_sep)
                lo_c = max(a + min_sep, center - half)
                hi_c = min(b - min_sep, center + half)
                if hi_c <= lo_c:
                    pos = center
                else:
                    cc = np.linspace(lo_c, hi_c, 25)
                    pos = float(cc[int(np.argmax([score(v) for v in cc]))])
                lines = sorted(lines + [pos])
            return lines

        # score a candidate line by the MIN boundary-distance over the points
        # where it crosses the other axis' lines (maximin): keep the
        # WORST-placed lattice point as far from a region edge as possible.
        # (Median let deep large regions dominate and pushed lines toward the
        # edges of the thin crescent arms.)
        fullY = sorted(set(Yh) | {-y for y in Yh})
        X = fill(X, float(xs.min()), float(xs.max()),
                 lambda x: min(Dmap[row(y), col(x)] for y in fullY), nx_max)
        Yh = fill(Yh, 0.0, float(ys.max()),
                  lambda y: min(Dmap[row(y), col(x)] for x in X), ny_max)
        Y = sorted(set(Yh) | {-y for y in Yh})
        return [(x, y) for x in sorted(X) for y in Y]

    @staticmethod
    def _basin_grid_placement(nx, ny, xlim, ylim, wheel_radius):
        """Regular rectangular grid of wheel locations, independent of the
        count field: ``nx`` columns evenly spaced across xlim and ``2*ny - 1``
        rows evenly spaced and symmetric about y=0 (one row on the axis), all
        inset by a wheel radius so edge wheels are not clipped. A uniform
        sampling of the domain -- the simple alternative to the structure-aware
        lattice. Assumes dynamics symmetric about y=0.
        """
        x0, x1 = xlim[0] + wheel_radius, xlim[1] - wheel_radius
        ytop = min(-ylim[0], ylim[1]) - wheel_radius
        xs = (np.linspace(x0, x1, nx) if nx > 1
              else np.array([0.5 * (x0 + x1)]))
        yh = np.linspace(0.0, ytop, ny) if ny > 1 else np.array([0.0])
        ys = sorted(set(yh.tolist()) | {-y for y in yh.tolist()})
        return [(float(x), float(y)) for x in xs for y in ys]

    @staticmethod
    def _reflect_wheel(w):
        """Mirror a wheel across the x-axis for the y<0 rows (valid under the
        x-axis symmetry assumption): negate the focal y, the stable
        directions, and each arc's heading bounds; labels/widths unchanged."""
        return {'focal_loc': (w['focal_loc'][0], -w['focal_loc'][1]),
                'stable': [-s for s in w['stable']],
                'arcs': [(-b, -a, lab) for (a, b, lab) in w['arcs']],
                'widths': dict(w['widths'])}

    @staticmethod
    def _render_basin_wheels(ax, wheels, r_out):
        """Draw basin wheels on an axis. Contained -- hand it the axis and the
        fully-computed wheel data and it does only matplotlib.

        Each wheel is a thin annulus partitioned into a location's
        heading-basin arcs plus an arrow per reachable stable direction;
        color is categorical by basin RANK (largest->smallest: gold, blue,
        vermilion, green, purple), a no-commit arc is light gray, and a
        single-direction cell draws a lone arrow with no annulus. The rank
        legend is added with add_artist so it coexists with the count legend.
        """
        from matplotlib.patches import Wedge, Patch
        import matplotlib.patheffects as pe

        # warm "autumn" categorical colors (gold -> orange -> brick red ->
        # brown). viridis (the count background) owns the whole cool->yellow
        # range, so cool wheel colors wash out against it; these warm tones
        # contrast at every count level.
        palette = ['#F5D742', '#DD6B0E', '#BC3B26', '#8A4B1E', '#5C3A1E']

        def rank_color(r):
            return palette[r % len(palette)]

        r_in = 0.83 * r_out
        ring_w = r_out - r_in
        stroke = [pe.withStroke(linewidth=1.8, foreground='black')]
        max_rank = 1
        for w in wheels:
            cx, cy = w['focal_loc']
            stable, arcs, widths = w['stable'], w['arcs'], w['widths']
            nz = [lab for lab in range(len(stable))
                  if widths.get(lab, 0.0) > 1e-9]
            if not nz:
                continue
            order = sorted(nz, key=lambda l: -widths[l])
            rank_of = {lab: r for r, lab in enumerate(order)}
            max_rank = max(max_rank, len(order))
            multi = len(nz) >= 2
            if multi:
                for (s_start, s_end, lab) in arcs:
                    span = (s_end - s_start) % (2 * np.pi)
                    if len(arcs) == 1:
                        span = 2 * np.pi
                    if span <= 1e-9:
                        continue
                    th1 = np.degrees(s_start)
                    col = ('lightgray' if lab < 0
                           else rank_color(rank_of.get(lab, 0)))
                    ax.add_patch(Wedge((cx, cy), r_out, th1,
                                       th1 + np.degrees(span), width=ring_w,
                                       facecolor=col, edgecolor='0.3',
                                       lw=0.3, zorder=5))
                cap = r_in
            else:
                cap = r_out
            for lab in nz:
                ang = stable[lab]
                frac = widths[lab] / (2 * np.pi)
                L = cap * (0.30 + 0.70 * frac) if multi else cap
                ax.annotate('', xy=(cx + L * np.cos(ang), cy + L * np.sin(ang)),
                            xytext=(cx, cy), zorder=6,
                            arrowprops=dict(arrowstyle='-|>',
                                            color=rank_color(rank_of[lab]),
                                            lw=1.3, shrinkA=0, shrinkB=0,
                                            mutation_scale=5,
                                            path_effects=stroke))
        names = ['largest basin', '2nd largest', '3rd largest',
                 '4th largest', '5th largest']
        handles = [Patch(facecolor=rank_color(i), edgecolor='0.3',
                         label=names[i]) for i in range(max_rank)]
        leg = ax.legend(handles=handles, loc='lower left', fontsize=7,
                        framealpha=0.9, title='basin wheel: rank by size',
                        title_fontsize=7)
        ax.add_artist(leg)

    def _overlay_basin_wheels(self, ax, count_field, xy_of_pixel, xlim, ylim,
                              pool=None, stability_criterion='reduced',
                              R_seed=0.15, n_coarse=64, n_bisect=12,
                              placement='lattice', min_sep_factor=2.2,
                              max_sep_factor=5.5, min_area=4, nx_max=6,
                              ny_max=4, target_margin=0.15, wheel_radius=None):
        """Place basin wheels, compute their arcs (parallel via ``pool``,
        exploiting x-axis symmetry: compute y>=0 and mirror), and render onto
        ``ax``. placement='lattice' -> structure-aware symmetric rectilinear
        lattice seeded on the multistable regions; 'grid' -> a plain regular
        nx x (2*ny-1) grid over the domain. Either way, wheels whose disk would
        overlap a target are dropped."""
        if wheel_radius is None:
            wheel_radius = 0.03 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        if placement == 'lattice':
            cells = self._basin_lattice_placement(
                count_field, xy_of_pixel, wheel_radius, nx_max=nx_max,
                ny_max=ny_max, min_sep_factor=min_sep_factor,
                max_sep_factor=max_sep_factor, min_area=min_area)
        elif placement == 'grid':
            cells = self._basin_grid_placement(
                nx_max, ny_max, xlim, ylim, wheel_radius)
        else:
            raise ValueError(
                "placement must be 'lattice' or 'grid', got %r" % (placement,))
        # drop wheels whose DISK overlaps a target (center within R_target +
        # r_wheel + margin), not just centers strictly inside
        tg = self.percep_model.targets
        cells = [c for c in cells
                 if (tg.get_dist_to_targets(np.asarray(c, dtype=float))
                     >= wheel_radius + target_margin).all()]
        # x-axis symmetry: compute wheels for y >= 0, mirror them to y < 0
        tol = 1e-9
        upper = [c for c in cells if c[1] >= -tol]
        args = [(tuple(c), R_seed, n_coarse, n_bisect, stability_criterion)
                for c in upper]
        if pool is None:
            up = [self._basin_arcs_worker(a) for a in args]
        else:
            up = pool.map(self._basin_arcs_worker, args)
        by_xy = {(round(w['focal_loc'][0], 6), round(w['focal_loc'][1], 6)): w
                 for w in up}
        wheels = list(up)
        for c in cells:
            if c[1] < -tol:
                src = by_xy.get((round(c[0], 6), round(-c[1], 6)))
                if src is not None:
                    wheels.append(self._reflect_wheel(src))
        self._render_basin_wheels(ax, wheels, wheel_radius)
        # draw the targets fully opaque, on top of the wheels
        tg.plot_targets_to_axis(ax, zorder=7)


    def _simulate_one_walk(self, args):
        '''Simulate a single walker trajectory for plot_walkers.

        Pulled out of plot_walkers so the repetitions can be dispatched to a
        multiprocessing pool (one task per repetition). Uses a per-walk RNG
        seeded from the supplied seed (an independent np.random.SeedSequence
        spawned in the parent) -- self.rng is deliberately NOT used here, since
        it would be duplicated across pool workers and yield identical walks.

        Mutates self.percep_model.focal_loc/focal_angle and self.gamma as
        scratch (re-initialized from the start_* args at the top of each call);
        the caller restores the originals once all walks finish. In a pool each
        worker holds its own pickled copy of self, so this is safe.

        Parameters
        ----------
        args : tuple
            (n, seed, start_loc, start_angle, start_gamma, dt, v, std,
             walk_std, noise_exp, R_exp, max_steps, target_tol). n is the
             repetition index (used only in the max_steps warning); seed is a
             np.random.SeedSequence (or any np.random.default_rng seed).

        Returns
        -------
        walk : (2, n_steps) float ndarray
            Column-stacked (x, y) positions of the trajectory.
        warn : str or None
            max_steps-exceeded warning message, or None if a target was found.
            Returned rather than warned in-process so warnings surface from the
            parent regardless of whether a pool was used.
        '''
        (n, seed, start_loc, start_angle, start_gamma, dt, v, std, walk_std,
         noise_exp, R_exp, max_steps, target_tol) = args
        rng = np.random.default_rng(seed)
        targets = self.percep_model.targets

        self.percep_model.focal_loc = np.array(start_loc, dtype=float).copy()
        self.percep_model.focal_angle = start_angle
        self.gamma = start_gamma
        walk = [self.percep_model.focal_loc.copy()]
        warn = None
        for step in range(max_steps):
            if np.any(targets.get_dist_to_targets(
                      self.percep_model.focal_loc) < target_tol):
                break
            old_loc = self.percep_model.focal_loc.copy()
            # Blind fast-path: if no target is visible at all, gamma
            # collapses to 0, the deterministic torque vanishes, and the
            # walker searches at the random-walk intensity walk_std -- set
            # INDEPENDENTLY of the committed std (orthogonal knobs), so it
            # re-acquires even in gentle constant-noise mode. walk_std=0
            # freezes the blind drift. A cheap fast-path skips the dgamma/dt
            # solve on a blind step.
            neur, _ = self.percep_model.get_neural_signals()
            if neur.size == 0:
                self.gamma = 0 + 0j
                dtheta = 0.0
                sigma_eff = walk_std
            else:
                dtheta = self.dtheta_dt()
                R = np.abs(self.gamma)
                # Visible: gated noise sigma*(1-R)^noise_exp -- a random walk
                # when R->0 (undecided) and low-noise homing when R->1
                # (committed); noise_exp=0 recovers a constant sigma*dW.
                sigma_eff = std * np.clip(1.0 - R, 0.0, 1.0) ** noise_exp
                if R > 0.0:
                    s = dtheta / (self.K * R)        # = sin(Theta/2), in [-1,1]
                    if noise_exp != 0:
                        # Heading-aligned modulation cos(Theta/2), Theta = the
                        # torque's angle (NBM: arg(gamma); IEM: the egocentric
                        # consensus). Derived from dtheta = K*R*sin(Theta/2):
                        #   cos(Theta/2) = sqrt(1 - (dtheta/(K*R))^2)
                        # (the +root; Theta/2 in (-pi/2, pi/2]). Full noise
                        # facing consensus (Theta=0), zero facing away
                        # (Theta=+-pi); in quadrature with the torque so
                        # corrective swings back are noise-free.
                        sigma_eff *= np.sqrt(max(0.0, 1.0 - s * s))
                    if R_exp != 1:
                        # Exponent on the drift's coherence: the walker's
                        # pursuit torque becomes K*R^R_exp*sin(Theta/2)
                        # (R_exp=1 is the model's dtheta_dt). Affects only the
                        # walker's drift here -- NOT the deterministic SC /
                        # bifurcation / basin machinery (which keeps R^1).
                        dtheta = self.K * R ** R_exp * s
            # dtheta_dt() is a turning RATE, not an angle, so it is NOT wrapped
            # here (only the resulting heading is wrapped on assignment below).
            # The diffusion term scales as sqrt(dt) (Wiener increment), NOT dt,
            # so the per-unit-time angular variance is independent of step size.
            if sigma_eff > 0.0:
                noise = sigma_eff * rng.normal() * np.sqrt(dt)
            else:
                noise = 0.0
            theta = self.percep_model.focal_angle + dtheta*dt + noise
            mv_vec = v*dt*np.array([np.cos(theta),np.sin(theta)])
            self.percep_model.focal_loc += mv_vec
            self.percep_model.focal_angle = convert_angles(theta)
            walk.append(self.percep_model.focal_loc.copy())
            if np.any(targets.check_trajectory_intersection(
                      old_loc, self.percep_model.focal_loc)):
                break
        else:
            dists = targets.get_dist_to_targets(self.percep_model.focal_loc)
            warn = (
                f"Walker {n} reached max_steps ({max_steps}). "
                f"Final position: ({self.percep_model.focal_loc[0]:.2f}, "
                f"{self.percep_model.focal_loc[1]:.2f}), "
                f"closest target distance: {dists.min():.4f}")
        return np.column_stack(walk), warn


    def plot_walkers(self, dt=0.1, v=1, std=None, walk_std=0.5*np.pi,
                     noise_exp=0, R_exp=1, repetitions=20, max_steps=1500,
                     start_loc=None, start_angle=None, target_tol=None,
                     alpha=1.0, pool=None, ax=None, title=None, wb_plot=False):
        '''Plot a walker that starts at a specified location looking in a
        specified angle (defaults to the focal_loc and focal_angle in attached
        PerceptionModel) and moves according to the neural band torque model on
        a dt step size with state-gated angular Gaussian noise. Repeat for a
        number of repetitions and plot the resulting trajectories in 2D space.

        The heading noise amplitude is sigma*(1-R)^noise_exp, where R=|gamma| is
        the coherence strength: a random walk when R->0 (undecided / no targets
        visible) and low-noise homing when R->1 (committed). noise_exp=0 recovers
        a plain constant sigma*dW. See the std/noise_exp parameters below.

        At each step, dgamma_dt is run to steady state to find the local
        equilibrium gamma; the resulting torque K*R*sin(arg(gamma)/2) in the
        neural consensus angle arg(gamma) drives the heading update.

        The walker stops when it is within target_tol of a target surface,
        when its trajectory passes through a target, or after max_steps.

        Set wb_plot to True if plotting in a Jupyter notebook

        Parameters
        ----------
        dt : float
            Time step for the walk
        v : float
            Speed of the walker, assumed constant
        std : float or None
            Sigma: the heading-noise intensity for VISIBLE steps (amplitude
            std*(1-R)^noise_exp). The per-step kick is sigma_eff*sqrt(dt)
            (Euler-Maruyama, per-unit-time angular variance sigma_eff**2,
            independent of dt). If None (default), a regime-aware default is used
            for the VISIBLE scale: 0.1 when noise_exp==0 (a gentle constant noise)
            and walk_std when noise_exp>0 (the random-walk intensity the
            (1-R)^noise_exp gate tames once the walker commits). std=0 makes the
            VISIBLE steps deterministic. (Blind steps use walk_std, not std --
            the two are orthogonal; see walk_std.)
        walk_std : float, optional (default 0.5*pi)
            Random-walk intensity used on BLIND steps (no targets visible: gamma
            collapses to 0, deterministic torque vanishes). Set INDEPENDENTLY of
            std so a lost walker re-acquires even in gentle constant-noise mode.
            walk_std=0 freezes the blind drift (deterministic search-off). Fully
            deterministic walk: std=0 AND walk_std=0. The 0.5*pi default makes the
            per-unit-time heading-change 2-sigma span the full circle (+-pi).
        noise_exp : float, optional (default 0)
            Gate exponent p in the VISIBLE-step noise amplitude
            std*(1-R)^p*cos(Theta/2), where Theta is the consensus angle relative
            to heading (the torque's angle). p=0 recovers a constant sigma*dW (no
            gating, no cos factor). p>0 interpolates between a random walk
            (R->0, undecided) and low-noise homing (R->1, committed) -- larger p
            closes the gate faster -- and the cos(Theta/2) factor (applied only
            for p!=0) zeros the noise when facing away from consensus (Theta=+-pi)
            and leaves it full facing consensus (Theta=0), in quadrature with the
            sin(Theta/2) torque so corrective swings back are noise-free.
        R_exp : float, optional (default 1)
            Exponent on the coherence R in the WALKER's drift (pursuit) torque:
            the heading update uses K*R^R_exp*sin(Theta/2). R_exp=1 is the model's
            K*R*sin(Theta/2). This affects ONLY the walker's drift -- the
            deterministic SC equilibria / bifurcation / basin machinery keep R^1.
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
        target_tol : float, optional
            Proximity threshold for declaring a target "found". The walker
            stops when the distance to any target surface is less than this
            value. If None (default), uses v*dt (one step size).
        alpha : float, optional (default 1.0)
            Opacity passed to the track plotting call (0 = fully transparent,
            1 = fully opaque). Lower values let overlapping trajectories reveal
            path density when many walks are aggregated.
        pool : multiprocessing.Pool, optional
            If provided, the repetitions are distributed across this pool (one
            task per walk) via pool.map. Each walk uses its own independent RNG
            stream (see Notes), so parallel and serial runs are statistically
            equivalent. If None (default), the walks run serially.
        ax : matplotlib axis, optional
            If provided, plot on this axis instead of creating a new figure and
            axis.
        title : str, optional
            Title for the plot. If not provided, a default title is used.
        wb_plot : bool
            Whether or not plotting in a Jupyter notebook (adjusts size of figure)

        Returns
        -------
        ax : matplotlib axis, if ax was provided as an argument. Otherwise, None.

        Notes
        -----
        Each repetition is driven by an independent RNG stream spawned from
        self.rng (via np.random.SeedSequence.spawn), not by self.rng directly.
        This is what makes pooling safe: a bound method dispatched to a
        multiprocessing pool carries a pickled copy of self -- and hence of
        self.rng -- to every worker, so drawing from self.rng inside the walk
        would give duplicate noise across workers. Spawning per-walk seeds in
        the parent keeps the walks independent and reproducible regardless of
        whether a pool is used, while still drawing from self.rng so successive
        plot_walkers calls differ.
        '''

        if start_loc is None:
            start_loc = self.percep_model.focal_loc.copy()
        else:
            start_loc = np.array(start_loc, dtype=float)
        if start_angle is None:
            start_angle = self.percep_model.focal_angle
        if target_tol is None:
            target_tol = v * dt
        if std is None:
            # Regime-aware default for the VISIBLE-step scale: a gentle constant
            # noise when ungated, else the random-walk intensity walk_std that the
            # (1-R)^p gate tames once the walker commits. (Blind steps use
            # walk_std directly, independent of this choice.)
            std = 0.1 if noise_exp == 0 else walk_std
        orig_loc = self.percep_model.focal_loc.copy()
        orig_angle = self.percep_model.focal_angle
        orig_gamma = self.gamma

        # Independent, reproducible per-walk RNG seeds spawned from self.rng so
        # repetitions are statistically independent (each walk seeds its own
        # Generator -- self.rng is not used in the walk loop, which would
        # otherwise be duplicated across pool workers; see Notes). Drawing the
        # base entropy from self.rng keeps successive plot_walkers calls
        # different.
        child_seeds = np.random.SeedSequence(
            int(self.rng.integers(0, 2**63 - 1))).spawn(repetitions)
        walk_args = [
            (n, child_seeds[n], start_loc, start_angle, orig_gamma, dt, v, std,
             walk_std, noise_exp, R_exp, max_steps, target_tol)
            for n in range(repetitions)]

        # One task per repetition; dispatch to the pool if provided, else run
        # serially. The helper mutates self.percep_model.focal_loc/angle and
        # self.gamma as scratch (restored below in the serial path).
        if pool is None:
            results = [self._simulate_one_walk(a) for a in walk_args]
        else:
            results = pool.map(self._simulate_one_walk, walk_args)

        all_walks = [walk for walk, _ in results]
        # Surface any max_steps warnings from the parent so they are visible
        # whether or not a pool (separate processes) was used.
        for _, warn in results:
            if warn is not None:
                warnings.warn(warn)

        # Restore focal location, angle, and gamma
        self.percep_model.focal_loc = orig_loc
        self.percep_model.focal_angle = orig_angle
        self.gamma = orig_gamma

        if ax is None:
            local_plot = True
            if wb_plot:
                fig = plt.figure(figsize=(6.5,4))
            else:
                fig = plt.figure(figsize=(5.5,5))
            ax = fig.add_subplot(aspect='equal')
        else:
            local_plot = False
            ax.set_aspect('equal')

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title('Random walker paths')

        self.percep_model.targets.plot_targets_to_axis(ax)

        # Plot each walker trajectory (already column-stacked by the walk
        # helper). alpha sets track opacity: lower values let overlapping paths
        # reveal where the walkers concentrate.
        for walk in all_walks:
            ax.plot(walk[0,:], walk[1,:], 'k', alpha=alpha)

        if local_plot:
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

    def __init__(self, percep_model=None, T=0.2, K=2, nu=1):
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
            Temperature for Ising model. Note that IEM's coupling carries a
            factor of the number of visible targets (see `dgamma_dt`), so its
            effective Boltzmann factor is N/T. This is NOT the same quantity as
            `NeuralBandModel.beta`, which is a scene-independent E/(k_B*temp);
            the two coincide only when N targets are visible and beta = N/T.
        K : float
            Coupling strength for Kuramoto turning speed. Default 2 to match
            the near-heading gain of the half-angle torque law
            dtheta/dt = K*|gamma|*sin((angle(gamma)-theta)/2); see
            NeuralBandModel.__init__.
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
            # IEM lives in allocentric/physical coordinates: gamma is a sum of
            # exp(i*angle) over PHYSICAL target angles, so the perception model
            # must apply no warp (identity) and uniform weight. A non-identity
            # neural_angle_dist (e.g. the default) is a category error for IEM.
            self.percep_model = PerceptionModel(neural_angle_dist=None,
                                                angle_weight=None)
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
        # Half-angle torque (see NBM.dtheta_dt). The egocentric consensus
        # angle (np.angle(gamma) - theta) is NOT pre-wrapped, so it must be
        # wrapped to (-pi, pi] before halving: sin(x/2) is 4*pi-periodic and
        # would otherwise be discontinuous/incorrect across 2*pi boundaries.
        ego = convert_angles(np.angle(gamma) - theta)
        return self.K*np.abs(gamma)*np.sin(ego/2)
    

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
        '''Find gamma equilibria of dgamma/dt at a fixed observer heading.

        Multistart root-finding seeded on the polar circle of radius 0.5.
        Treats the observer's heading as given (not solved-for) and finds
        gamma values that are equilibria of dgamma_dt at that heading.
        For self-consistent equilibria (heading = allocentric consensus
        direction), see sc_equilib().

        Parameters
        ----------
        focal_angle : float, optional
            The observer's heading. If None, use self.percep_model.focal_angle.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use
            self.percep_model.focal_loc.

        Returns
        -------
        gamma_eqs : list of complex
            equilibrium gamma values at the fixed heading.
        '''
        if focal_angle is True:
            raise TypeError(
                "gamma_equilib(focal_angle=True) is no longer supported; "
                "call sc_equilib(focal_loc=...) instead to find "
                "self-consistent equilibria.")

        init_angles = np.linspace(-np.pi, np.pi-0.01)
        init_gamma_vecs = np.zeros((init_angles.size, 2), dtype=np.double)
        init_gamma_vecs[:,0] = 0.5*np.cos(init_angles)
        init_gamma_vecs[:,1] = 0.5*np.sin(init_angles)
        final_gammas = []
        for init_val in init_gamma_vecs:
            sol = root(self.dgamma_dt_vec, init_val, args=(focal_angle, focal_loc),
                       method='hybr', tol=1e-7)
            if sol.success:
                gamma_eq = sol.x[0] + 1j*sol.x[1]
                # Check if close to any existing solution
                close_check = False
                for existing_gamma in final_gammas:
                    if np.abs(gamma_eq - existing_gamma) < 0.01:
                        close_check = True
                        break
                if not close_check:
                    final_gammas.append(gamma_eq)
        return final_gammas


    def sc_equilib(self, focal_loc=None):
        '''Find self-consistent equilibria where heading = consensus direction.

        For each init point on the polar circle of radius 0.5, set
        focal_angle to the angle of the init point and run root-finding.
        After convergence, verify the candidate is a self-consistent
        equilibrium by requiring dgamma_dt to vanish when focal_angle is
        set to the allocentric consensus direction angle(gamma_eq).

        Use this for bifurcation diagrams in (x, y).

        Parameters
        ----------
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use
            self.percep_model.focal_loc.

        Returns
        -------
        gamma_eqs : list of complex
            self-consistent equilibrium gamma values. Each gamma has
            |gamma| = R (coherence strength) and angle(gamma) = allocentric
            heading at the equilibrium.
        '''
        init_angles = np.linspace(-np.pi, np.pi-0.01)
        init_gamma_vecs = np.zeros((init_angles.size, 2), dtype=np.double)
        init_gamma_vecs[:,0] = 0.5*np.cos(init_angles)
        init_gamma_vecs[:,1] = 0.5*np.sin(init_angles)
        final_gammas = []
        for init_val in init_gamma_vecs:
            focal_angle = np.angle(init_val[0] + 1j*init_val[1])
            sol = root(self.dgamma_dt_vec, init_val,
                       args=(focal_angle, focal_loc),
                       method='hybr', tol=1e-7)
            if sol.success:
                gamma_eq = sol.x[0] + 1j*sol.x[1]
                # Verify self-consistency: dgamma_dt must be zero when
                # focal_angle equals the allocentric consensus direction
                # angle(gamma).
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

        final_gammas = self.sc_equilib(focal_loc=focal_loc)

        return ii, jj, final_gammas
    

    def _coupled_jacobian(self, gamma_star, focal_angle, focal_loc, h=1e-6):
        '''Central-difference 3x3 Jacobian of the coupled
        (gamma_re, gamma_im, theta) system at (gamma_star, focal_angle), with
        dtheta/dt = K*|gamma|*sin(convert_angles(angle(gamma) - theta)/2).

        Used by `_discrim_reduced`, which needs the fast gamma block A and
        the sign of det(J). Note that, unlike NBM, the IEM heading torque
        depends on theta directly, so the (theta, theta) entry
        d = d(dtheta)/dtheta is nonzero (= -K*R/2 at a self-consistent eq).
        NOTE: the FULL eigenvalues of this matrix are NOT a stability verdict
        for the model -- see `_discrim_reduced` for why.
        '''
        gr0 = float(gamma_star.real)
        gi0 = float(gamma_star.imag)
        th0 = float(focal_angle)

        def coupled_rhs(gr, gi, th):
            gamma = gr + 1j*gi
            dg = self.dgamma_dt(gamma=gamma, focal_angle=th,
                                focal_loc=focal_loc)
            R = np.abs(gamma)
            ego = convert_angles(np.angle(gamma) - th)
            dth = self.K * R * np.sin(ego/2)
            return np.array([dg.real, dg.imag, dth])

        J = np.zeros((3, 3))
        for k, (dr, di, dt) in enumerate([(h, 0, 0), (0, h, 0), (0, 0, h)]):
            f_plus = coupled_rhs(gr0+dr, gi0+di, th0+dt)
            f_minus = coupled_rhs(gr0-dr, gi0-di, th0-dt)
            J[:, k] = (f_plus - f_minus) / (2*h)
        return J


    def _discrim_reduced(self, gamma_star, focal_angle, focal_loc,
                         h=1e-6, tol=1e-8):
        '''Timescale-separated ("block") stability test for IEM, consistent
        with gamma slaved to equilibrium (mean field; gamma run to steady
        state before each heading step; no gamma-noise).

        Partition the coupled Jacobian J = [[A, b], [c, d]] (A the 2x2 gamma
        block). Stable iff BOTH the fast layer is attracting (A Hurwitz, all
        eig(A) real-part < -tol -- coincides with `_discrim_A_nu` for the
        cosine kernel) AND the slow layer is stable. The slow eigenvalue is
        the Schur complement lam_slow = d - c A^{-1} b, the linearization of
        the slaved heading flow dtheta/dt = g(h(theta), theta). Because IEM's
        torque depends on theta directly, d != 0 here (= -K*R/2 at a
        self-consistent eq), so the slow eigenvalue carries that direct
        heading restoring term in addition to the gamma-mediated feedback.
        We test its SIGN via the block-determinant identity without inverting
        A (which is ill-conditioned near a gamma-fold): det(J) = det(A)*lam_slow
        with det(A) > 0 for Hurwitz A, so lam_slow < 0 <=> det(J) < 0. Like the
        NBM version, the slow flow is 1-D in theta and cannot oscillate, so
        this test does not see gamma-theta Hopf instabilities. There is
        deliberately no 'coupled' criterion; see `NeuralBandModel.
        _discrim_reduced` ("Why there is no fully-coupled criterion").
        '''
        J = self._coupled_jacobian(gamma_star, focal_angle, focal_loc, h)
        A = J[:2, :2]
        if not np.all(np.real(np.linalg.eigvals(A)) < -tol):
            return False
        return bool(np.linalg.det(J) < 0.0)


    def _discrim_A_nu(self, gamma_star, focal_loc, h=1e-6, tol=1e-8):
        '''Determines gamma-stability from the full 2x2 fast block of the
        coupled Jacobian at a self-consistent equilibrium (nu-warped cosine
        kernel).

        Stable iff the fast gamma block A = d(dgamma)/dgamma is Hurwitz (all
        eigenvalues have real part < -tol) -- equivalently the free-energy
        Hessian is positive definite: both the transverse curvature (the old
        A < 1 scalar) AND det > 0.

        The block is taken NUMERICALLY from _coupled_jacobian (the nu warp
        enters through dgamma_dt) rather than the old analytic transverse-only
        A < 1 scalar, which was necessary but not sufficient -- it missed
        radial folds and off-diagonal (R-Theta) saddles. This is the fast-block
        gate of _discrim_reduced without the slow Schur mode; use 'reduced' for
        that. See theory/free_energy_derivation.md sec 6.1.

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
        J = self._coupled_jacobian(gamma_star, np.angle(gamma_star),
                                   focal_loc, h)
        A = J[:2, :2]
        return bool(np.all(np.real(np.linalg.eigvals(A)) < -tol))


    def _count_stable_at(self, args):
        '''Helper for plot_bifurcation_diagram: count stable self-consistent
        equilibria at a single (x, y) location.

        Parameters
        ----------
        args : tuple
            (key, x, y, stability_criterion). ``key`` is an arbitrary hashable
            identifier echoed back to the caller for result reassembly.

        Returns
        -------
        key : hashable
        count : int
        '''
        key, x, y, stability_criterion = args
        focal_loc = np.array([x, y])
        gamma_eqs = self.sc_equilib(focal_loc=focal_loc)
        if stability_criterion == 'reduced':
            stable = [self._discrim_reduced(g, np.angle(g), focal_loc)
                      for g in gamma_eqs]
        elif stability_criterion == 'discrim_a':
            stable = [self._discrim_A_nu(g, focal_loc) for g in gamma_eqs]
        else:
            raise ValueError(
                f"stability_criterion must be 'reduced' or 'discrim_a', "
                f"got {stability_criterion!r}")
        return key, int(sum(stable))


    def plot_bifurcation_diagram(self, xlim=(0,6), num_x=29, ylim=(-3.5,3.5),
                                 num_y=29, refinement_levels=3, max_count=None,
                                 boundary_dilation=1,
                                 pool=None, ax=None, title=None, wb_plot=False,
                                 stability_criterion='reduced'):
        '''Plot a 2D colormap showing the number of stable self-consistent
        equilibria of the Ising-ext model as a function of observer (x,y)
        location.

        Starts from a coarse ``num_x`` by ``num_y`` grid and adaptively
        subdivides cells whose four corners disagree on the number of stable
        equilibria, up to ``refinement_levels`` times. Evaluated points are
        cached so that corners shared between cells are not recomputed.

        Parameters
        ----------
        xlim : (xmin,xmax) tuple of floats
            x limits for the base mesh, inclusive
        num_x : int
            number of steps in x direction for the base mesh
        ylim : (ymin,ymax) tuple of floats
            y limits for the base mesh, inclusive
        num_y : int
            number of steps in y direction for the base mesh
        refinement_levels : int
            number of adaptive subdivision passes. 0 => base mesh only.
            Each pass halves cell size at boundaries where the stable count
            changes. Final virtual grid resolution is
            ((num_x-1)*2**L + 1) by ((num_y-1)*2**L + 1).
        max_count : int, optional
            maximum expected number of stable equilibria. Pins the color
            scale so that count=N maps to the same color across multiple
            calls (e.g. side-by-side subplots comparing models). If None,
            auto-detected from the data. Values in the data exceeding
            max_count are clipped with a warning.
        boundary_dilation : int
            At each refinement pass, also refine cells that share a corner
            with a cell whose own corners disagree, ``boundary_dilation``
            steps outward. Default 1 widens the refined band by one cell
            per side, smoothing stair-step artifacts where a fine-grained
            transition meets a coarse settled neighbour. 0 reproduces the
            legacy strict-corner-disagreement behaviour.
        pool : multiprocessing.Pool, optional
            If provided, evaluate new points at each refinement level in
            parallel.
        ax : matplotlib axis, optional
            If provided, plot on this axis instead of creating a new figure.
        title : str, optional
            title for the plot
        wb_plot : bool
            whether or not plotting in a Jupyter notebook
        stability_criterion : {'reduced', 'discrim_a'}
            Which stability test to apply when counting stable equilibria.
            'reduced' (default) is the timescale-separated test: fast gamma
            block Hurwitz AND slow Schur complement d - c A^{-1} b < 0
            (``_discrim_reduced``). 'discrim_a' routes to ``_discrim_A_nu``,
            the cosine-aware gamma-only test (the fast block alone); for IEM
            the heading torque adds a direct -K*R/2 self-term to the slow
            mode, so 'reduced' can differ from 'discrim_a' where that term
            and the gamma-mediated feedback compete.

        Returns
        -------
        ax : matplotlib axis, if ax was provided as an argument.
            Otherwise, None.
        '''
        assert refinement_levels >= 0, "refinement_levels must be >= 0"
        assert boundary_dilation >= 0, "boundary_dilation must be >= 0"

        L = refinement_levels
        step0 = 2**L
        NI = (num_x - 1)*step0 + 1
        NJ = (num_y - 1)*step0 + 1

        def idx_to_xy(I, J):
            x = xlim[0] + (xlim[1] - xlim[0])*I/(NI - 1)
            y = ylim[0] + (ylim[1] - ylim[0])*J/(NJ - 1)
            return x, y

        cache = {}

        def evaluate_points(keys):
            keys = [k for k in keys if k not in cache]
            if not keys:
                return
            args_list = [(k, *idx_to_xy(*k), stability_criterion)
                         for k in keys]
            if pool is None:
                for args in args_list:
                    key, count = self._count_stable_at(args)
                    cache[key] = count
            else:
                results = pool.map(self._count_stable_at, args_list)
                for key, count in results:
                    cache[key] = count

        base_keys = [(i*step0, j*step0)
                     for i in range(num_x) for j in range(num_y)]
        evaluate_points(base_keys)

        cells = [(i*step0, j*step0, step0)
                 for i in range(num_x - 1) for j in range(num_y - 1)]

        for _ in range(L):
            to_refine = []
            settled = []
            for cell in cells:
                I, J, s = cell
                corner_counts = {cache[(I, J)], cache[(I+s, J)],
                                 cache[(I, J+s)], cache[(I+s, J+s)]}
                if len(corner_counts) == 1:
                    settled.append(cell)
                else:
                    to_refine.append(cell)

            for _round in range(boundary_dilation):
                if not to_refine:
                    break
                boundary_corners = set()
                for I, J, s in to_refine:
                    boundary_corners.update(
                        [(I, J), (I+s, J), (I, J+s), (I+s, J+s)])
                promoted = []
                kept_settled = []
                for cell in settled:
                    I, J, s = cell
                    cell_corners = {(I, J), (I+s, J),
                                    (I, J+s), (I+s, J+s)}
                    if cell_corners & boundary_corners:
                        promoted.append(cell)
                    else:
                        kept_settled.append(cell)
                if not promoted:
                    break
                to_refine.extend(promoted)
                settled = kept_settled

            new_points = set()
            for I, J, s in to_refine:
                half = s // 2
                for m in [(I+half, J), (I, J+half),
                          (I+s, J+half), (I+half, J+s),
                          (I+half, J+half)]:
                    if m not in cache:
                        new_points.add(m)
            evaluate_points(new_points)

            cells = settled
            for I, J, s in to_refine:
                half = s // 2
                cells.append((I, J, half))
                cells.append((I+half, J, half))
                cells.append((I, J+half, half))
                cells.append((I+half, J+half, half))

        data_max = max(cache.values())
        if max_count is None:
            effective_max = data_max
        else:
            effective_max = max_count
            if data_max > effective_max:
                warnings.warn(
                    "Data contains stable-equilibrium counts up to "
                    f"{data_max} but max_count={effective_max}; "
                    "values above max_count will be clipped.")
        img = np.zeros((NJ - 1, NI - 1), dtype=int)
        for (I, J, s) in cells:
            cLL = cache[(I, J)]
            cLR = cache[(I+s, J)]
            cUL = cache[(I, J+s)]
            cUR = cache[(I+s, J+s)]
            if cLL == cLR == cUL == cUR:
                img[J:J+s, I:I+s] = cLL
            elif s == 1:
                img[J, I] = cLL
            else:
                half = s // 2
                img[J:J+half, I:I+half] = cLL
                img[J:J+half, I+half:I+s] = cLR
                img[J+half:J+s, I:I+half] = cUL
                img[J+half:J+s, I+half:I+s] = cUR

        if ax is None:
            local_plot = True
            if wb_plot:
                fig = plt.figure(figsize=(12,6))
            else:
                fig = plt.figure(figsize=(5.5,5))
            ax = plt.subplot(1,1,1)
        else:
            local_plot = False

        cmap = plt.get_cmap('viridis', effective_max + 1)
        norm = BoundaryNorm(boundaries=np.arange(-0.5, effective_max + 1.5),
                            ncolors=effective_max + 1)
        img = np.clip(img, 0, effective_max)
        ax.imshow(img, origin='lower',
                  extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                  aspect='equal', interpolation='nearest',
                  cmap=cmap, norm=norm)

        self.percep_model.targets.plot_targets_to_axis(ax)

        for n in range(effective_max + 1):
            ax.plot([], [], marker='s', markersize=10, linestyle='',
                    color=cmap(norm(n)), label=f'{n}')

        if title is not None:
            ax.set_title(title)
        elif stability_criterion == 'discrim_a':
            ax.set_title('Ising-ext bifurcation diagram (discrim_A)')
        else:
            ax.set_title('Ising-ext bifurcation diagram (reduced)')

        if local_plot:
            ax.legend(title='# stable\nequilibria', loc='center left',
                      bbox_to_anchor=(1.02, 0.5), frameon=False)
            fig.tight_layout()
            plt.show()
        else:
            return ax


    def plot_direction_mesh(self, xlim=(0,6), num_x=19, ylim=(-3.5,3.5), num_y=19,
                            pool=None, ax=None, wb_plot=False,
                            stability_criterion='reduced'):
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
        stability_criterion : {'reduced', 'discrim_a'}
            Which stability test to apply to each equilibrium. 'reduced'
            (default) is the timescale-separated test (fast gamma block
            Hurwitz + slow Schur complement < 0), consistent with gamma
            slaved to equilibrium. 'discrim_a' uses the cosine-aware
            gamma-only test `_discrim_A_nu` (the fast block alone).

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
                if stability_criterion == 'reduced':
                    stability_list[n][jj,ii] = self._discrim_reduced(
                        gamma, np.angle(gamma), focal_loc)
                elif stability_criterion == 'discrim_a':
                    stability_list[n][jj,ii] = self._discrim_A_nu(
                        gamma, focal_loc)
                else:
                    raise ValueError(
                        f"stability_criterion must be 'reduced' or "
                        f"'discrim_a', got {stability_criterion!r}")
            if len(final_gammas) > 1:
                multi_sol[jj,ii] = True

        if local_plot:
            ax = plt.subplot(1,1,1)

        # Plot targets
        self.percep_model.targets.plot_targets_to_axis(ax)
        if not U_list:
            warnings.warn(
                "plot_direction_mesh: no equilibria found at any mesh point; "
                "rendering targets only.")
        else:
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
        ax.set_aspect('equal')
        ax.set_title('Equilibrium plot')
        if local_plot:
            fig.legend(loc='outside center right')
            plt.show()
        else:
            return ax
        

    def _simulate_one_walk(self, args):
        '''Simulate a single walker trajectory for plot_walkers.

        Pulled out of plot_walkers so the repetitions can be dispatched to a
        multiprocessing pool (one task per repetition). Uses a per-walk RNG
        seeded from the supplied seed (an independent np.random.SeedSequence
        spawned in the parent) -- self.rng is deliberately NOT used here, since
        it would be duplicated across pool workers and yield identical walks.

        Mutates self.percep_model.focal_loc/focal_angle and self.gamma as
        scratch (re-initialized from the start_* args at the top of each call);
        the caller restores the originals once all walks finish. In a pool each
        worker holds its own pickled copy of self, so this is safe.

        Parameters
        ----------
        args : tuple
            (n, seed, start_loc, start_angle, start_gamma, dt, v, std,
             walk_std, noise_exp, R_exp, max_steps, target_tol). n is the
             repetition index (used only in the max_steps warning); seed is a
             np.random.SeedSequence (or any np.random.default_rng seed).

        Returns
        -------
        walk : (2, n_steps) float ndarray
            Column-stacked (x, y) positions of the trajectory.
        warn : str or None
            max_steps-exceeded warning message, or None if a target was found.
            Returned rather than warned in-process so warnings surface from the
            parent regardless of whether a pool was used.
        '''
        (n, seed, start_loc, start_angle, start_gamma, dt, v, std, walk_std,
         noise_exp, R_exp, max_steps, target_tol) = args
        rng = np.random.default_rng(seed)
        targets = self.percep_model.targets

        self.percep_model.focal_loc = np.array(start_loc, dtype=float).copy()
        self.percep_model.focal_angle = start_angle
        self.gamma = start_gamma
        walk = [self.percep_model.focal_loc.copy()]
        warn = None
        for step in range(max_steps):
            if np.any(targets.get_dist_to_targets(
                      self.percep_model.focal_loc) < target_tol):
                break
            old_loc = self.percep_model.focal_loc.copy()
            # Blind fast-path: if no target is visible at all, gamma
            # collapses to 0, the deterministic torque vanishes, and the
            # walker searches at the random-walk intensity walk_std -- set
            # INDEPENDENTLY of the committed std (orthogonal knobs), so it
            # re-acquires even in gentle constant-noise mode. walk_std=0
            # freezes the blind drift. A cheap fast-path skips the dgamma/dt
            # solve on a blind step.
            neur, _ = self.percep_model.get_neural_signals()
            if neur.size == 0:
                self.gamma = 0 + 0j
                dtheta = 0.0
                sigma_eff = walk_std
            else:
                dtheta = self.dtheta_dt()
                R = np.abs(self.gamma)
                # Visible: gated noise sigma*(1-R)^noise_exp -- a random walk
                # when R->0 (undecided) and low-noise homing when R->1
                # (committed); noise_exp=0 recovers a constant sigma*dW.
                sigma_eff = std * np.clip(1.0 - R, 0.0, 1.0) ** noise_exp
                if R > 0.0:
                    s = dtheta / (self.K * R)        # = sin(Theta/2), in [-1,1]
                    if noise_exp != 0:
                        # Heading-aligned modulation cos(Theta/2), Theta = the
                        # torque's angle (NBM: arg(gamma); IEM: the egocentric
                        # consensus). Derived from dtheta = K*R*sin(Theta/2):
                        #   cos(Theta/2) = sqrt(1 - (dtheta/(K*R))^2)
                        # (the +root; Theta/2 in (-pi/2, pi/2]). Full noise
                        # facing consensus (Theta=0), zero facing away
                        # (Theta=+-pi); in quadrature with the torque so
                        # corrective swings back are noise-free.
                        sigma_eff *= np.sqrt(max(0.0, 1.0 - s * s))
                    if R_exp != 1:
                        # Exponent on the drift's coherence: the walker's
                        # pursuit torque becomes K*R^R_exp*sin(Theta/2)
                        # (R_exp=1 is the model's dtheta_dt). Affects only the
                        # walker's drift here -- NOT the deterministic SC /
                        # bifurcation / basin machinery (which keeps R^1).
                        dtheta = self.K * R ** R_exp * s
            # dtheta_dt() is a turning RATE, not an angle, so it is NOT wrapped
            # here (only the resulting heading is wrapped on assignment below).
            # The diffusion term scales as sqrt(dt) (Wiener increment), NOT dt,
            # so the per-unit-time angular variance is independent of step size.
            if sigma_eff > 0.0:
                noise = sigma_eff * rng.normal() * np.sqrt(dt)
            else:
                noise = 0.0
            theta = self.percep_model.focal_angle + dtheta*dt + noise
            mv_vec = v*dt*np.array([np.cos(theta),np.sin(theta)])
            self.percep_model.focal_loc += mv_vec
            self.percep_model.focal_angle = convert_angles(theta)
            walk.append(self.percep_model.focal_loc.copy())
            if np.any(targets.check_trajectory_intersection(
                      old_loc, self.percep_model.focal_loc)):
                break
        else:
            dists = targets.get_dist_to_targets(self.percep_model.focal_loc)
            warn = (
                f"Walker {n} reached max_steps ({max_steps}). "
                f"Final position: ({self.percep_model.focal_loc[0]:.2f}, "
                f"{self.percep_model.focal_loc[1]:.2f}), "
                f"closest target distance: {dists.min():.4f}")
        return np.column_stack(walk), warn


    def plot_walkers(self, dt=0.1, v=1, std=None, walk_std=0.5*np.pi,
                     noise_exp=0, R_exp=1, repetitions=20, max_steps=1500,
                     start_loc=None, start_angle=None, target_tol=None,
                     alpha=1.0, pool=None, ax=None, title=None, wb_plot=False):
        '''Plot a walker that starts at a specified location looking in a
        specified angle (defaults to the focal_loc and focal_angle in attached
        PerceptionModel) and moves according to the Ising torque model on a dt
        step size with state-gated angular Gaussian noise. Repeat for a number
        of repetitions and plot the resulting trajectories in 2D space.

        The heading noise amplitude is sigma*(1-R)^noise_exp, where R=|gamma| is
        the coherence strength: a random walk when R->0 (undecided / no targets
        visible) and low-noise homing when R->1 (committed). noise_exp=0 recovers
        a plain constant sigma*dW. See the std/noise_exp parameters below.

        The walker stops when it is within target_tol of a target surface,
        when its trajectory passes through a target, or after max_steps.

        Set wb_plot to True if plotting in a Jupyter notebook

        Parameters
        ----------
        dt : float
            Time step for the walk
        v : float
            Speed of the walker, assumed constant
        std : float or None
            Sigma: the heading-noise intensity for VISIBLE steps (amplitude
            std*(1-R)^noise_exp). The per-step kick is sigma_eff*sqrt(dt)
            (Euler-Maruyama, per-unit-time angular variance sigma_eff**2,
            independent of dt). If None (default), a regime-aware default is used
            for the VISIBLE scale: 0.1 when noise_exp==0 (a gentle constant noise)
            and walk_std when noise_exp>0 (the random-walk intensity the
            (1-R)^noise_exp gate tames once the walker commits). std=0 makes the
            VISIBLE steps deterministic. (Blind steps use walk_std, not std --
            the two are orthogonal; see walk_std.)
        walk_std : float, optional (default 0.5*pi)
            Random-walk intensity used on BLIND steps (no targets visible: gamma
            collapses to 0, deterministic torque vanishes). Set INDEPENDENTLY of
            std so a lost walker re-acquires even in gentle constant-noise mode.
            walk_std=0 freezes the blind drift (deterministic search-off). Fully
            deterministic walk: std=0 AND walk_std=0. The 0.5*pi default makes the
            per-unit-time heading-change 2-sigma span the full circle (+-pi).
        noise_exp : float, optional (default 0)
            Gate exponent p in the VISIBLE-step noise amplitude
            std*(1-R)^p*cos(Theta/2), where Theta is the consensus angle relative
            to heading (the torque's angle). p=0 recovers a constant sigma*dW (no
            gating, no cos factor). p>0 interpolates between a random walk
            (R->0, undecided) and low-noise homing (R->1, committed) -- larger p
            closes the gate faster -- and the cos(Theta/2) factor (applied only
            for p!=0) zeros the noise when facing away from consensus (Theta=+-pi)
            and leaves it full facing consensus (Theta=0), in quadrature with the
            sin(Theta/2) torque so corrective swings back are noise-free.
        R_exp : float, optional (default 1)
            Exponent on the coherence R in the WALKER's drift (pursuit) torque:
            the heading update uses K*R^R_exp*sin(Theta/2). R_exp=1 is the model's
            K*R*sin(Theta/2). This affects ONLY the walker's drift -- the
            deterministic SC equilibria / bifurcation / basin machinery keep R^1.
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
        target_tol : float, optional
            Proximity threshold for declaring a target "found". The walker
            stops when the distance to any target surface is less than this
            value. If None (default), uses v*dt (one step size).
        alpha : float, optional (default 1.0)
            Opacity passed to the track plotting call (0 = fully transparent,
            1 = fully opaque). Lower values let overlapping trajectories reveal
            path density when many walks are aggregated.
        pool : multiprocessing.Pool, optional
            If provided, the repetitions are distributed across this pool (one
            task per walk) via pool.map. Each walk uses its own independent RNG
            stream (see Notes), so parallel and serial runs are statistically
            equivalent. If None (default), the walks run serially.
        ax : matplotlib axis, optional
            If provided, plot on this axis instead of creating a new figure and
            axis.
        title : str, optional
            Title for the plot. If not provided, a default title is used.
        wb_plot : bool
            Whether or not plotting in a Jupyter notebook (adjusts size of figure)

        Returns
        -------
        ax : matplotlib axis, if ax was provided as an argument. Otherwise, None.

        Notes
        -----
        Each repetition is driven by an independent RNG stream spawned from
        self.rng (via np.random.SeedSequence.spawn), not by self.rng directly.
        This is what makes pooling safe: a bound method dispatched to a
        multiprocessing pool carries a pickled copy of self -- and hence of
        self.rng -- to every worker, so drawing from self.rng inside the walk
        would give duplicate noise across workers. Spawning per-walk seeds in
        the parent keeps the walks independent and reproducible regardless of
        whether a pool is used, while still drawing from self.rng so successive
        plot_walkers calls differ. As part of this, each repetition now also
        resets self.gamma to its call-time value (the IC for the first visible
        dgamma/dt solve), so the walks no longer carry gamma over from one to
        the next.
        '''

        if start_loc is None:
            start_loc = self.percep_model.focal_loc.copy()
        else:
            start_loc = np.array(start_loc, dtype=float)
        if start_angle is None:
            start_angle = self.percep_model.focal_angle
        if target_tol is None:
            target_tol = v * dt
        if std is None:
            # Regime-aware default for the VISIBLE-step scale: a gentle constant
            # noise when ungated, else the random-walk intensity walk_std that the
            # (1-R)^p gate tames once the walker commits. (Blind steps use
            # walk_std directly, independent of this choice.)
            std = 0.1 if noise_exp == 0 else walk_std
        orig_loc = self.percep_model.focal_loc.copy()
        orig_angle = self.percep_model.focal_angle
        orig_gamma = self.gamma

        # Independent, reproducible per-walk RNG seeds spawned from self.rng so
        # repetitions are statistically independent (each walk seeds its own
        # Generator -- self.rng is not used in the walk loop, which would
        # otherwise be duplicated across pool workers; see Notes). Drawing the
        # base entropy from self.rng keeps successive plot_walkers calls
        # different.
        child_seeds = np.random.SeedSequence(
            int(self.rng.integers(0, 2**63 - 1))).spawn(repetitions)
        walk_args = [
            (n, child_seeds[n], start_loc, start_angle, orig_gamma, dt, v, std,
             walk_std, noise_exp, R_exp, max_steps, target_tol)
            for n in range(repetitions)]

        # One task per repetition; dispatch to the pool if provided, else run
        # serially. The helper mutates self.percep_model.focal_loc/angle and
        # self.gamma as scratch (restored below in the serial path).
        if pool is None:
            results = [self._simulate_one_walk(a) for a in walk_args]
        else:
            results = pool.map(self._simulate_one_walk, walk_args)

        all_walks = [walk for walk, _ in results]
        # Surface any max_steps warnings from the parent so they are visible
        # whether or not a pool (separate processes) was used.
        for _, warn in results:
            if warn is not None:
                warnings.warn(warn)

        # Restore focal location, angle, and gamma
        self.percep_model.focal_loc = orig_loc
        self.percep_model.focal_angle = orig_angle
        self.gamma = orig_gamma

        if ax is None:
            local_plot = True
            if wb_plot:
                fig = plt.figure(figsize=(6.5,4))
            else:
                fig = plt.figure(figsize=(5.5,5))
            ax = fig.add_subplot(aspect='equal')
        else:
            local_plot = False
            ax.set_aspect('equal')

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title('Random walker paths')

        self.percep_model.targets.plot_targets_to_axis(ax)

        # Plot each walker trajectory (already column-stacked by the walk
        # helper). alpha sets track opacity: lower values let overlapping paths
        # reveal where the walkers concentrate.
        for walk in all_walks:
            ax.plot(walk[0,:], walk[1,:], 'k', alpha=alpha)

        if local_plot:
            plt.show()
        else:
            return ax
