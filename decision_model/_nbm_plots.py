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

"""Rasters and diagnostic plots for ``NeuralBandModel`` (mixin).

Everything here consumes the model's dynamics and stability machinery and
renders it: the (x, y) bifurcation raster and its refinement, the direction
mesh, the dtheta/dt profile, and the coupling-kernel plot.  Mixed into
``NeuralBandModel``; not useful on its own.

The basin-wheel overlay that ``plot_bifurcation_diagram(overlay_basins=True)``
draws is in ``_nbm_basins``.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


class _PlotMixin:
    '''Plotting and raster methods of ``NeuralBandModel``.'''

    def plot_nu_cosine(self, wb_plot=False):
        """Plot the coupling kernel `nu_cosine` over [-pi, pi].

        Set wb_plot to True if plotting in a Jupyter notebook.
        """

        xmesh = np.linspace(-np.pi, np.pi, 1000)
        ymesh = self.nu_cosine(xmesh)

        if wb_plot:
            plt.figure(figsize=(6.5,3.25))
        else:
            plt.figure(figsize=(8,4))
        plt.plot(xmesh, ymesh)
        if self.angle_distortion_nu is None:
            nu_label = 1
        else:
            nu_label = self.angle_distortion_nu
        plt.title('Cosine Coupling Kernel, $\\nu={}$'.format(nu_label))
        plt.xlabel('Angular separation (radians)')
        plt.ylabel('Coupling')
        plt.ylim(-1.1,1.1)
        plt.grid()
        plt.show()


    def plot_dtheta_dt(self, gamma=None, focal_loc=None, wb_plot=False):
        '''Plot the turning rate dtheta/dt as a function of heading at a fixed
        focal location.

        Set wb_plot to True if plotting in a Jupyter notebook.

        Parameters
        ----------
        gamma : complex float or False, optional
            Coherence value driving the torque at each heading.
            None (default) leaves `dtheta_dt` to relax gamma at every heading,
            starting each solve from the value the previous heading landed on
            -- a swept curve, so it carries hysteresis through any bistable
            stretch, and self.gamma is left at the last value found.
            False re-seeds the solve from a neutral gamma = 0.1 + 0j (weak
            coherence, consensus straight ahead) at every heading instead, so
            each point is independent of the sweep direction, and restores
            self.gamma afterwards. This is the same neutral-seed protocol the
            basin machinery uses; the two curves differ exactly where the
            heading-basin structure is nontrivial.
            A complex value is used directly at every heading with no solve.
        focal_loc : array-like of length 2, optional
            (x,y) location of the observer. If None, use the percep_model's
            focal_loc.
        wb_plot : bool
            whether or not plotting in a Jupyter notebook
        '''

        thetas = np.linspace(-np.pi, np.pi, 1000)
        if gamma is False:
            saved_gamma = self.gamma
            dthetas = []
            for theta in thetas:
                self.gamma = 0.1 + 0j
                dthetas.append(self.dtheta_dt(theta=theta,
                                              focal_loc=focal_loc))
            self.gamma = saved_gamma
            dthetas = np.array(dthetas)
        elif gamma is None:
            # Swept: gamma is carried from one heading to the next, so it is
            # transported through each turn exactly as in the walker.
            dthetas = []
            for i, theta in enumerate(thetas):
                if i > 0:
                    self.gamma *= np.exp(-1j*(theta - thetas[i-1]))
                dthetas.append(self.dtheta_dt(theta=theta,
                                              focal_loc=focal_loc))
            dthetas = np.array(dthetas)
        else:
            dthetas = np.array([self.dtheta_dt(theta=theta, gamma=gamma,
                                               focal_loc=focal_loc)
                                for theta in thetas])

        if wb_plot:
            plt.figure(figsize=(6.5,4.5))
        else:
            plt.figure(figsize=(8,4))
        plt.plot(thetas, dthetas)
        plt.axhline(0, color='k', linestyle='--')
        plt.title('$d\\theta/dt$ vs $\\theta$')
        plt.xlabel('$\\theta$ (observer heading, radians)')
        plt.ylabel('$d\\theta/dt$')
        plt.grid()
        plt.show()


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
