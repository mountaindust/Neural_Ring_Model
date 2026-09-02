# Neural Ring Model: Ising-type dynamics of spatial decision-making.
# Copyright (C) 2026 Christopher Strickland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Target geometry: placement, apparent angular extent, and occlusion.

``Targets`` holds every target in the scene and answers the geometric
questions perception needs -- the arc each target subtends from an observer,
which targets block which, and whether a step of the walker would run into
one. Three geometries are supported: ``circle``, ``delta`` (point) and
``capsule`` (line-segment spine with semicircular endcaps).
"""

import numpy as np
import matplotlib.pyplot as plt

from .angles import convert_angles, _smallest_enclosing_arc


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
