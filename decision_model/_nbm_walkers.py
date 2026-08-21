"""SDE walker simulation for ``NeuralBandModel`` (mixin).

``plot_walkers`` runs an ensemble of observers through the heading SDE --
deterministic torque plus state-gated angular noise -- from a common start,
until each reaches a target, goes blind, or hits ``max_steps``.  One walk is
``_simulate_one_walk``, split out so repetitions can be handed to a
multiprocessing pool; it is also called directly by several scripts in
plots/ and walker_analysis/.

The noise law, the blind-spot search, target detection and the loss mechanisms
are documented in .claude/rules/walker-dynamics.md.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt

from .angles import convert_angles


class _WalkerMixin:
    '''Walker-simulation methods of ``NeuralBandModel``.'''

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
                        # Heading-aligned modulation cos(Theta/2), Theta =
                        # the torque's angle arg(gamma). Derived from
                        # dtheta = K*R*sin(Theta/2):
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
            turn = dtheta*dt + noise
            theta = self.percep_model.focal_angle + turn
            mv_vec = v*dt*np.array([np.cos(theta),np.sin(theta)])
            self.percep_model.focal_loc += mv_vec
            self.percep_model.focal_angle = convert_angles(theta)
            # Transport the neural state through the turn. gamma is the
            # readout sum_k n_k exp(i*neural_angle_k); turning moves every
            # neural angle by -turn while the populations n_k stay put, so
            # the carried gamma rotates with them. See "Carrying gamma
            # between steps" in plot_walkers.
            self.gamma *= np.exp(-1j*turn)
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
        a dt step size with angular Gaussian noise -- constant by default,
        optionally state-gated. Repeat for a number of repetitions and plot the
        resulting trajectories in 2D space.

        The Langevin system
        -------------------
        The walker is a constant-speed particle whose heading is the only
        stochastic degree of freedom:

            d(theta) = K * R**R_exp * sin(Theta/2) dt  +  sigma(state) dW
            d(x)     = v * cos(theta) dt
            d(y)     = v * sin(theta) dt

        Theta = arg(gamma*) and R = |gamma*| come from the neural band: at
        every step dgamma_dt is run to steady state at the current heading and
        location (the timescale separation -- see the class docstring), and
        gamma* is read off. The heading is integrated Euler-Maruyama, so the
        Wiener increment enters as sigma*sqrt(dt) and the per-unit-time angular
        variance is sigma**2 regardless of dt; the position step then advances
        along the UPDATED heading.

        The noise amplitude has three factors and a separate blind branch:

            sigma(state) = std * (1-R)**noise_exp * cos(Theta/2)   [visible]
            sigma(state) = walk_std                                [blind]

        "Blind" means no target is visible at all, where gamma collapses to 0
        and the deterministic torque vanishes with it. cos(Theta/2) is applied
        only when noise_exp != 0.

        Choosing a noise regime
        -----------------------
        The knobs overlap, so the recipes are worth stating outright:

          * CONSTANT noise while a target is in view: noise_exp=0. That kills
            the (1-R) gate and the cos(Theta/2) factor together, leaving
            sigma = std exactly. This is the default, with std defaulting to
            0.1. Blind steps still use walk_std.
          * CONSTANT noise everywhere: noise_exp=0 AND walk_std=std, with std
            given explicitly. The blind branch is otherwise a second, much
            larger noise level (walk_std defaults to 0.5*pi) that switches in
            whenever the walker loses sight of every target.
          * STATE-GATED noise (the "commit and home in" regime): noise_exp>0.
            Noise is then full when undecided (R->0) and shuts off as the
            walker commits (R->1), and the cos(Theta/2) factor additionally
            silences it when facing away from consensus. std then defaults to
            walk_std rather than 0.1.
          * DETERMINISTIC: std=0 AND walk_std=0.

        R_exp is a drift knob, not a noise knob: it rescales the pursuit
        torque only, and only for the walker.

        Carrying gamma between steps
        ----------------------------
        Each step warm-starts its dgamma_dt solve from the gamma the previous
        step ended on, rotated by minus the turn just taken.

        The rotation is what keeps the carried state consistent with what
        gamma means. gamma is the readout sum_k n_k exp(i*neural_angle_k),
        one population n_k per visible target. A step moves the observer, not
        the neurons: the populations are unchanged by the turn itself (they
        relax afterwards, on the fast neural timescale), while every neural
        angle shifts by -turn. Re-evaluating the readout at the new angles is
        therefore a rotation of gamma by exp(-i*turn) -- exact whenever the
        angles shift rigidly, which is the case for an identity warp.

        This matters because relaxing to equilibrium does not by itself name a
        gamma: where the landscape is multistable the warm start selects which
        equilibrium, and near a fold -- where a branch the walker is riding
        ceases to exist, which is precisely the decision point the model is
        built to study -- that selection is what sets the outcome. Carrying
        the rotated gamma is the timescale-separated statement: the neural
        state stays on its branch, tracking it until the branch itself
        disappears.

        Scope: the walker and `_basin_destination`. The deterministic
        machinery (sc_equilib, both stability criteria, bifurcation diagrams,
        direction meshes) sits at dtheta/dt = 0, where there is no turn to
        transport through.

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
