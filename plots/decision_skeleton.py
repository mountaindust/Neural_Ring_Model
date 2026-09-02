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

"""Deterministic decision-track *skeleton* from the model's bifurcation structure.

Instead of drawing the PNAS main-decision tracks by hand or sampling them from
noisy random walkers (``NeuralBandModel.plot_walkers``), build them deterministically:
start where there is a single stable consensus-heading direction and follow it; where
a bifurcation makes that branch go unstable and two stable branches appear, FORK and
follow both; repeat down the cascade until the tracks arrive at all three targets.

The deterministic direction field is read straight from
``NeuralBandModel.sc_equilib(focal_loc=(x, y), stability_criterion='reduced')`` --
the allocentric stable consensus-heading directions at any observer position, with
``'reduced'`` the criterion the deterministic (slaved) walker actually obeys. The
skeleton is then a forking streamline integration of that field.

This module is standalone (imports ``decision_model`` but edits nothing in it). The
fly/locust model setups below *mirror* the walker scripts ``three_target_fly.py`` /
``three_target_locust.py`` (and, for the two-target cases, ``two_target_fly_refine.py``)
-- figure scripts with ``plot_walkers`` side effects in ``__main__``, so they cannot be
imported. **Keep the two in sync**; only the deterministic warp/weight/geometry/K/beta are
reproduced here (the noise knobs std/v/noise_exp/R_exp are irrelevant to the
deterministic skeleton).

Two-target cases (``fly2`` / ``locust2``) trace a simple **Y**: a single midline trunk
rides the straight-ahead compromise heading up to one pitchfork, where it forks into
one arm per target and stops -- there is no on-midline (centre) target, so no
straight-through centre route and no second bifurcation. See ``trace_skeleton``.

Phase 0 deliverable: ``plot_branch_diagram`` -- a proper (x, theta) + (x, R)
bifurcation branch diagram along a horizontal cut, showing which equilibrium
*directions* are born and which die, and where (what the count-only
``plot_bifurcation_diagram`` throws away).

Run:  python plots/decision_skeleton.py fly --branch-diagram
      python plots/decision_skeleton.py diagram-both   # combined fly+locust headings
"""
import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decision_model as model

pi = np.pi

# ----------------------------------------------------------------------------
# Model setups -- mirror the walker scripts three_target_{fly,locust}.py. Only the
# deterministic structure (geometry, warp, weight, K, beta) is reproduced here; the noise
# knobs (std/v/noise_exp/R_exp) live only in the walker scripts. Keep the two in sync.
# a_warp sits the branches on the empirical heatmap ridge; a_weight is tuned to the
# centre/outer split; K does NOT affect the SC structure (skeleton is K-independent).
# Tuning rationale + the data match: three_target_findings.md.
# ----------------------------------------------------------------------------

# Fly: three circle targets 40 deg apart at radius 5, target radius 0.5.
FLY_LOCS = np.array([[5.0000,  0.0000],
                     [3.8302,  3.2139],
                     [3.8302, -3.2139]])
FLY = dict(locs=FLY_LOCS, r=0.5,
           a_warp=0.65*pi, b_warp=0.92*pi,    # a_warp 0.65pi (refit): first bifurcation
           a_weight=0.20*pi, b_weight=0.80*pi, #   pushed out to the empirical x
           K=2.0, beta=30.0,                   # K=2 (refit); K does not affect the skeleton
                                              # beta = N/T_old = 3/0.10
           xlim=(-0.3, 5.3), ylim=(-3.6, 3.6), godm_case='fly3')

# Locust: 3 targets at radius 3, target radius 0.1. The EMPIRICAL locust3 separation
# is 35 deg (verified from the GODM data posts), NOT the 40 deg the original
# three_target_locust.py assumed -- matching it makes the model->data alignment
# identity (skeleton starts at the origin, not offset).
_la = np.radians(35.0)
LOCUST_LOCS = np.array([[3.0,             0.0],
                        [3.0*np.cos(_la),  3.0*np.sin(_la)],
                        [3.0*np.cos(_la), -3.0*np.sin(_la)]])
LOCUST = dict(locs=LOCUST_LOCS, r=0.1,
              a_warp=0.50*pi, b_warp=0.90*pi,
              a_weight=0.10*pi, b_weight=0.80*pi,
              K=6.0, beta=30.0,               # beta = N/T_old = 3/0.10
              xlim=(-0.3, 3.3), ylim=(-2.4, 2.4), godm_case='locust3')

# Fly two-target: GODM fly2 (60 deg separation, distance 5) -> two circle targets at
# +-30 deg, radius 5 = (4.330, +-2.5), target radius 0.5. Mirrors two_target_fly_refine.py
# (which imports the 3-target fly params verbatim -- same fly, same setup, two targets):
# the SAME warp/weight/K as FLY, only the geometry differs -- and with it beta,
# which reproduces the old N_targets/T coupling and so is 20 here against FLY's 30.
# The model splits at x~2.2.
_fa2 = pi/6.0                                   # 30 deg (half of the 60 deg separation)
FLY2_LOCS = np.array([[5.0*np.cos(_fa2),  5.0*np.sin(_fa2)],
                      [5.0*np.cos(_fa2), -5.0*np.sin(_fa2)]])   # (4.330, +-2.5)
FLY2 = dict(locs=FLY2_LOCS, r=0.5,
            a_warp=0.65*pi, b_warp=0.92*pi,
            a_weight=0.20*pi, b_weight=0.80*pi,
            K=2.0, beta=20.0,                 # beta = N/T_old = 2/0.10
            xlim=(-0.3, 4.8), ylim=(-3.0, 3.0), godm_case='fly2')

# Locust two-target: GODM locust2 (45 deg separation, distance 2) -> two targets at
# +-22.5 deg. Mirrors LOCUST exactly as FLY2 mirrors FLY: same distance 3 model frame
# and the SAME warp/weight/K as LOCUST, only the separation differs (35->45 deg) --
# and beta, which is 20 here against LOCUST's 30 for the same reason as FLY2.
# CAVEAT: unlike fly2, locust2 has no walker-refine/findings validation -- these knobs
# are ASSUMED (reused from locust3). The --branch-diagram pitchfork and the make_figure
# arrival check are the validation hooks; re-tune a_warp if the split is not clean.
_la2 = np.radians(45.0)
LOCUST2_LOCS = np.array([[3.0*np.cos(_la2/2),  3.0*np.sin(_la2/2)],
                         [3.0*np.cos(_la2/2), -3.0*np.sin(_la2/2)]])
LOCUST2 = dict(locs=LOCUST2_LOCS, r=0.1,
               a_warp=0.50*pi, b_warp=0.90*pi,
               a_weight=0.10*pi, b_weight=0.80*pi,
               K=6.0, beta=20.0,              # beta = N/T_old = 2/0.10
               xlim=(-0.3, 3.3), ylim=(-2.0, 2.0), godm_case='locust2')

CASES = {'fly': FLY, 'fly2': FLY2, 'locust': LOCUST, 'locust2': LOCUST2}


def _build_model(cfg):
    targets = model.Targets(locs=cfg['locs'], geom_name='circle', r=cfg['r'])
    pm = model.PerceptionModel(targets, (0, 0), 0,
                               neural_angle_dist='lin_cutoff', angle_weight='lin_cutoff',
                               a_warp=cfg['a_warp'], b_warp=cfg['b_warp'],
                               a_weight=cfg['a_weight'], b_weight=cfg['b_weight'])
    return model.NeuralBandModel(pm, beta=cfg['beta'], K=cfg['K'])


def build_fly_model(r=0.5):
    cfg = dict(FLY, r=r)
    return _build_model(cfg)


def build_locust_model(r=0.1):
    cfg = dict(LOCUST, r=r)
    return _build_model(cfg)


# ----------------------------------------------------------------------------
# Phase 0 diagnostic: (x, theta) + (x, R) bifurcation branch diagram.
# ----------------------------------------------------------------------------

def _cluster(values, gap):
    """Merge sorted values whose neighbours are within ``gap`` into cluster means.
    The second bifurcation is a tight cluster of saddle-nodes (plus some near-SN
    solver jitter); reporting cluster centres is cleaner and more honest than a
    flood of individual count-change x-locations."""
    values = np.sort(np.asarray(values, dtype=float))
    if values.size == 0:
        return values
    out, cur = [], [values[0]]
    for v in values[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            out.append(np.mean(cur)); cur = [v]
    out.append(np.mean(cur))
    return np.array(out)


def _branch_scan(nm, y0, xs, criterion):
    """Scan a horizontal cut y=y0 over xs. Returns lists of arrays:
    per-x (theta, R, stable) and the per-x stable count."""
    th_s, th_u, R_s, R_u, x_s, x_u, n_stable = [], [], [], [], [], [], []
    for x in xs:
        angles, Rs, stab = nm.sc_equilib((x, y0), criterion, return_R=True)
        ns = 0
        for a, r, s in zip(angles, Rs, stab):
            if s:
                x_s.append(x); th_s.append(a); R_s.append(r); ns += 1
            else:
                x_u.append(x); th_u.append(a); R_u.append(r)
        n_stable.append(ns)
    return (dict(x=np.array(x_s), th=np.array(th_s), R=np.array(R_s)),
            dict(x=np.array(x_u), th=np.array(th_u), R=np.array(R_u)),
            np.array(n_stable))


def plot_branch_diagram(nm, *, y0=0.0, xlim=None, num_x=400, criterion='reduced',
                        extra_y=(), title=None, save=None):
    """Bifurcation branch diagram of SC equilibria along horizontal cuts.

    For each cut y in (y0,) + tuple(extra_y), sweep the observer x across ``xlim``
    and plot every self-consistent equilibrium's heading ``theta`` (top panel) and
    coherence ``R = |gamma|`` (bottom panel) vs x. Stable = filled, unstable = open.
    Saddle-node BIRTHS appear as a stable+unstable pair emerging together; DEATHS as
    branches colliding and vanishing. Vertical dashed lines mark x where the stable
    count changes (the bifurcation locations). This is the directional information
    the count-only ``plot_bifurcation_diagram`` discards.

    Parameters
    ----------
    nm : NeuralBandModel
    y0 : float                  primary cut (default midline 0.0)
    xlim : (xmin, xmax)         x sweep range (defaults to the model's framing)
    num_x : int                 samples along x
    criterion : stability criterion forwarded to the solver
    extra_y : iterable of float additional cuts (off-axis), one column each
    save : path or None         if given, savefig(save)
    """
    if xlim is None:
        xlim = (0.0, 5.6)
    y_cuts = (y0,) + tuple(extra_y)
    xs = np.linspace(xlim[0], xlim[1], num_x)

    ncols = len(y_cuts)
    fig, axes = plt.subplots(2, ncols, figsize=(5.2 * ncols, 7.0),
                             squeeze=False, sharex='col')
    for col, yc in enumerate(y_cuts):
        stable, unstable, n_stable = _branch_scan(nm, yc, xs, criterion)
        ax_th, ax_R = axes[0, col], axes[1, col]

        # bifurcation x-locations: where the stable count changes (clustered, since
        # the second bifurcation is a tight saddle-node cluster + near-SN jitter)
        bif_x = _cluster(xs[1:][np.diff(n_stable) != 0], gap=0.12)

        for ax in (ax_th, ax_R):
            for bx in bif_x:
                ax.axvline(bx, color='0.8', lw=0.8, ls='--', zorder=0)

        ax_th.scatter(unstable['x'], np.degrees(unstable['th']), s=4,
                      facecolors='none', edgecolors='tab:red', lw=0.5,
                      label='unstable', zorder=2)
        ax_th.scatter(stable['x'], np.degrees(stable['th']), s=5,
                      color='tab:blue', label='stable', zorder=3)
        ax_R.scatter(unstable['x'], unstable['R'], s=4, facecolors='none',
                     edgecolors='tab:red', lw=0.5, zorder=2)
        ax_R.scatter(stable['x'], stable['R'], s=5, color='tab:blue', zorder=3)

        ax_th.set_ylabel('SC equilibrium heading  $\\theta$  (deg)')
        ax_R.set_ylabel('coherence  $R=|\\gamma|$')
        ax_R.set_xlabel('observer x')
        ax_th.set_ylim(-185, 185)
        ax_th.set_yticks([-180, -90, -67, -22, 0, 22, 67, 90, 180])
        ax_R.set_ylim(0, 1.02)
        ax_th.set_title(f'cut y = {yc:.2f}'
                        + (f'   (bif x ~ {", ".join(f"{b:.2f}" for b in bif_x)})'
                           if bif_x.size else ''))
        ax_th.grid(True, alpha=0.25)
        ax_R.grid(True, alpha=0.25)
    axes[0, 0].legend(loc='lower left', fontsize=8, framealpha=0.9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print('wrote', save)
    return fig


def plot_diagram_both(*, y_cuts=(0.0, 0.5, 1.0), num_x=400, criterion='reduced',
                      save=None):
    """Combined fly-over-locust SC-equilibrium branch diagram.

    Stacks the *heading* (x, theta) panel of the fly branch diagram (top row) over
    that of the locust (bottom row), one column per y-cut -- i.e. the first row of
    each case's ``plot_branch_diagram`` output, with the (x, R) coherence row of
    both dropped. Each case keeps its own x-extent (fly to ~5.3, locust to ~3.3), so
    the rows do not share an x-axis. The cut titles are drawn on the top (fly) row
    only; the locust row uses the SAME cuts, so repeating them would be redundant.
    Per-panel rendering mirrors ``plot_branch_diagram``.
    """
    cases = [('fly', 'fly equilib. heading $\\theta$ (deg)'),
             ('locust', 'locust equilib. heading $\\theta$ (deg)')]
    ncols = len(y_cuts)
    fig, axes = plt.subplots(2, ncols, figsize=(5.2 * ncols, 7.0), squeeze=False)
    for row, (case, ylabel) in enumerate(cases):
        cfg = CASES[case]
        nm = _build_model(cfg)
        xlim = (0.0, cfg['xlim'][1])
        xs = np.linspace(xlim[0], xlim[1], num_x)
        for col, yc in enumerate(y_cuts):
            stable, unstable, n_stable = _branch_scan(nm, yc, xs, criterion)
            ax_th = axes[row, col]

            bif_x = _cluster(xs[1:][np.diff(n_stable) != 0], gap=0.12)
            for bx in bif_x:
                ax_th.axvline(bx, color='0.8', lw=0.8, ls='--', zorder=0)

            ax_th.scatter(unstable['x'], np.degrees(unstable['th']), s=4,
                          facecolors='none', edgecolors='tab:red', lw=0.5,
                          label='unstable', zorder=2)
            ax_th.scatter(stable['x'], np.degrees(stable['th']), s=5,
                          color='tab:blue', label='stable', zorder=3)

            ax_th.set_ylim(-185, 185)
            ax_th.set_yticks([-180, -90, -67, -22, 0, 22, 67, 90, 180])
            ax_th.grid(True, alpha=0.25)
            # y-label only on the left-most column (tighter figure)
            if col == 0:
                ax_th.set_ylabel(ylabel)
            # cut title only on the top (fly) row -- the locust row shares the cuts
            if row == 0:
                ax_th.set_title(f'cut y = {yc:.2f}')
            # x-label only on the bottom (locust) row -- the same 'observer x' for all
            if row == len(cases) - 1:
                ax_th.set_xlabel('observer x')
    axes[0, 0].legend(loc='lower left', fontsize=8, framealpha=0.9)
    fig.suptitle('Self-consistent equilibrium branches')
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print('wrote', save)
    return fig


# ----------------------------------------------------------------------------
# Phase 1: the forking-streamline tracer.
#
# The deterministic skeleton is a streamline integration of the stable
# consensus-heading field. A leaf follows ONE equilibrium branch by (theta, R)
# continuity (nearest equilibrium in the combined angle/coherence metric -- standard
# numerical continuation), which tracks a branch smoothly through (a) the centre
# branch's brief SC-*unstable* interlude between the first and the reborn-centre
# bifurcation, and (b) an arm's sharp redirection at the second bifurcation. Only
# the trunk forks, and only at the first bifurcation, where the single stem gains
# the two compromise arms (a saddle-node *detached* from the centre -- confirmed by
# the Phase-0 branch diagram); each of the resulting leaves then rides its own
# branch to a distinct target, so no merge step is needed.
# ----------------------------------------------------------------------------


class Track:
    """One streamline (polyline) of the decision skeleton."""

    def __init__(self, tid, parent_id, loc, heading, birth_reason, side=None):
        self.id = tid
        self.parent_id = parent_id
        self.points = [np.asarray(loc, dtype=float).copy()]
        self.headings = [float(heading)]
        self.stabilities = []           # per advanced step: was the branch SC-stable
        self.side = side                # follower mode: 0 centre/all, +-1 same-side arm,
        #                                 None = derive from launch heading
        self.birth = dict(reason=birth_reason, loc=np.asarray(loc, float).copy(),
                          heading=float(heading))
        self.death = None               # dict(reason=..., loc=..., target_idx=...)

    @property
    def xy(self):
        return np.array(self.points)

    @property
    def target_idx(self):
        return None if self.death is None else self.death.get('target_idx')


class SkeletonTree:
    def __init__(self, tracks, root_id, params):
        self.tracks = tracks
        self.root_id = root_id
        self.params = params

    def arrivals(self):
        return [t for t in self.tracks if t.death and t.death['reason'] == 'arrival']

    def leaves(self):
        # A decision-tree leaf is an endpoint: a track that arrives at a target, or
        # any childless terminal track. The root doubles as trunk AND centre leaf
        # (it forks the arms yet still rides on to the centre target), so it is
        # included when it arrives even though it has children.
        parents = {t.parent_id for t in self.tracks if t.parent_id is not None}
        return [t for t in self.tracks
                if t.id not in parents
                or (t.death and t.death['reason'] == 'arrival')]

    def target_assignment(self):
        return {t.target_idx: t.id for t in self.arrivals() if t.target_idx is not None}


def _metric(th, R, th0, R0, r_weight):
    """Combined (angle, coherence) continuation distance between two equilibria."""
    return np.hypot(model.convert_angles(th - th0), r_weight * (R - R0))


def trace_skeleton(nm, start_loc=(0.0, 0.0), start_heading=0.0, *, ds=0.02,
                   follow_tol=0.9, r_weight=2.0, target_tol=0.05, fold_thresh=0.4,
                   second_fork=True, xlim=(-0.5, 6.0), ylim=(-5.0, 5.0), max_steps=1600,
                   stability_criterion='reduced'):
    """Trace the deterministic decision-track skeleton.

    Returns a ``SkeletonTree``. The root marches from ``start_loc`` along the centre
    branch; at the first bifurcation it forks (seeding a leaf per newly-stable
    branch) and continues as the centre leaf; every leaf then follows its branch by
    (theta, R) continuity to a target.
    """
    targets = nm.percep_model.targets
    # The on-midline (centre) target, or None when none sits on y=0 (the 2-target /
    # even-symmetric case: both targets straddle the midline). center_idx is None is the
    # single switch the no-centre logic keys off below.
    _locs_y = np.asarray(targets.locs)[:, 1]
    _ci = int(np.argmin(np.abs(_locs_y)))
    center_idx = _ci if abs(_locs_y[_ci]) < 1e-6 else None
    # Is the whole problem mirror-symmetric about the x-axis (targets symmetric, start
    # on the axis heading along it)? If so the centre route lies EXACTLY on y=0, and we
    # pin it there -- otherwise the root, riding the SC-unstable centre branch through
    # its interlude, drifts off-axis as the transverse instability amplifies the hybr
    # solver's tiny y-asymmetry, and the reborn centre comes back at a spurious angle.
    _ys = np.sort(np.asarray(targets.locs)[:, 1])
    _symmetric = (np.allclose(_ys, -_ys[::-1])
                  and abs(start_loc[1]) < 1e-9
                  and abs(model.convert_angles(start_heading)) < 1e-9)

    def all_dirs(loc):
        a, R, s = nm.sc_equilib(loc, stability_criterion, return_R=True)
        return [(model.convert_angles(t), r, bool(k)) for t, r, k in zip(a, R, s)]

    def in_domain(loc):
        return xlim[0] <= loc[0] <= xlim[1] and ylim[0] <= loc[1] <= ylim[1]

    params = dict(ds=ds, follow_tol=follow_tol, r_weight=r_weight,
                  target_tol=target_tol, stability_criterion=stability_criterion,
                  start_loc=tuple(start_loc), start_heading=start_heading)

    root = Track(0, None, start_loc, start_heading, 'root')
    tracks = [root]
    queue = [root]
    next_id = [1]

    def integrate(tk, allow_fork):
        loc = tk.points[0].copy()
        dd = all_dirs(loc)
        if not dd:
            tk.death = dict(reason='blind', loc=loc.copy(), target_idx=None)
            return
        # seed the branch we ride from the equilibrium nearest the launch heading
        th0, R0, _ = min(dd, key=lambda d: abs(model.convert_angles(d[0] - tk.headings[0])))
        h, Rp = th0, R0
        tk.headings[0] = h
        has_forked = False
        fold_forked = False

        # Follower mode by track "side". The CENTRE leaf (side 0) follows the branch
        # nearest in (theta, R) among ALL equilibria, so it rides the centre branch
        # straight through its brief SC-*unstable* interlude (between the first and
        # the reborn-centre bifurcation). An ARM leaf (side +-1) follows the nearest
        # SAME-SIDE *stable* branch: this rides the compromise branch and, when that
        # branch dies at the second bifurcation, JUMPS across the fold to the
        # same-side outer-target branch (committing to the outer target) instead of
        # riding the dying inner saddle down to R->0 or grabbing the opposite-side
        # centre branch.
        if tk.side is not None:
            side = tk.side
        else:
            side = 0
            if abs(h) > np.radians(8.0):
                side = int(np.sign(h))
        # the root's centre route is pinned to y=0 in a mirror-symmetric problem
        lock_root = _symmetric and tk.birth['reason'] == 'root'

        def candidates(dirs):
            if side == 0:
                return dirs
            same = [d for d in dirs if d[2] and int(np.sign(d[0])) == side]
            if same:
                return same
            return [d for d in dirs if d[2]]    # fallback: any stable

        for _ in range(max_steps):
            dist = targets.get_dist_to_targets(loc)
            if np.min(dist) < target_tol:
                tk.death = dict(reason='arrival', loc=loc.copy(),
                                target_idx=int(np.argmin(dist)))
                return
            dd = all_dirs(loc)
            if not dd:
                tk.death = dict(reason='blind', loc=loc.copy(), target_idx=None)
                return

            # --- first fork: seed the non-continuation stable branches. Fires at the
            #     first multistable point -- the first bifurcation for a single-stable
            #     start, or immediately if the origin is already tri-stable. ---
            stable_now = [d for d in dd if d[2]]
            if allow_fork and not has_forked and center_idx is None:
                # No centre target (2-target / even-symmetric). The two arms are born
                # (saddle-node) at the FIRST bifurcation while the straight-ahead
                # compromise (centre, theta=0) is still stable, then the centre DIES at
                # a SECOND bifurcation. A walker riding the stable centre does not peel
                # off until the centre itself goes unstable, so the trunk forks at the
                # centre's DEATH (the empirical long-trunk split), not the arms' birth.
                # The root rides the centre via lock_root until here; then fork into the
                # stable arms and stop -- there is no continuation/centre leaf.
                cen = min(dd, key=lambda d: abs(model.convert_angles(d[0])))
                centre_alive = cen[2] and abs(model.convert_angles(cen[0])) < np.radians(8.0)
                if not centre_alive and len(stable_now) >= 2:
                    for d in stable_now:
                        arm_side = int(np.sign(d[0])) if abs(d[0]) > 1e-3 else 0
                        child = Track(next_id[0], tk.id, loc, d[0], 'fork', side=arm_side)
                        next_id[0] += 1
                        tracks.append(child)
                        queue.append(child)
                    has_forked = True
                    tk.death = dict(reason='fork', loc=loc.copy(), target_idx=None)
                    return
            elif allow_fork and not has_forked and len(stable_now) >= 2:
                cont = min(stable_now, key=lambda d: _metric(d[0], d[1], h, Rp, r_weight))
                for d in stable_now:
                    if d is cont:
                        continue
                    # an arm's side is the sign of its branch angle (even if the arm
                    # is born at a small angle from a tri-stable origin)
                    arm_side = int(np.sign(d[0])) if abs(d[0]) > 1e-3 else 0
                    child = Track(next_id[0], tk.id, loc, d[0], 'fork', side=arm_side)
                    next_id[0] += 1
                    tracks.append(child)
                    queue.append(child)
                has_forked = True

            # --- advance by (theta, R) continuity within the side-appropriate set ---
            cand = candidates(dd)
            if not cand:
                tk.death = dict(reason='lost', loc=loc.copy(), target_idx=None)
                return
            th, R, st = min(cand, key=lambda d: _metric(d[0], d[1], h, Rp, r_weight))
            # the centre leaf has a sanity guard; an arm is allowed the fold jump
            if side == 0 and not lock_root and _metric(th, R, h, Rp, r_weight) > follow_tol:
                tk.death = dict(reason='lost', loc=loc.copy(), target_idx=None)
                return
            if lock_root:
                # pin to the midline: keep the theta~0 (centre) equilibrium's R and
                # stability but step exactly along +x, so y stays 0 by symmetry
                cen = min(dd, key=lambda d: abs(model.convert_angles(d[0])))
                th, R, st = 0.0, cen[1], cen[2]

            # RK2 midpoint (the actual committed heading th2 -- the fold to the outer
            # branch shows up here, not in th, because the midpoint probe lands past
            # where the compromise branch dies)
            mid = loc + 0.5 * ds * np.array([np.cos(th), np.sin(th)])
            dm = candidates(all_dirs(mid))
            if dm:
                th2, R2, _ = min(dm, key=lambda d: _metric(d[0], d[1], th, R, r_weight))
            else:
                th2, R2 = th, R
            if lock_root:
                th2, R2 = 0.0, R

            # --- second bifurcation: when an arm's compromise branch folds (its
            #     committed heading th2 jumps by > fold_thresh as it commits to the
            #     outer branch), also seed a CENTRE-bound route -- the second binary
            #     decision {outer target, centre target} (the PNAS dashed diamond). ---
            if (second_fork and center_idx is not None and side != 0 and not fold_forked
                    and abs(model.convert_angles(th2 - h)) > fold_thresh):
                cvec = np.asarray(targets.locs)[center_idx] - loc
                bearing_c = np.arctan2(cvec[1], cvec[0])
                cstable = [d for d in dd if d[2]]
                if cstable:
                    cdir = min(cstable, key=lambda d:
                               abs(model.convert_angles(d[0] - bearing_c)))
                    # only if it is a genuinely different (centre-ward) branch
                    if abs(model.convert_angles(cdir[0] - th2)) > np.radians(20):
                        child = Track(next_id[0], tk.id, loc, cdir[0], 'fork2', side=0)
                        next_id[0] += 1
                        tracks.append(child)
                        queue.append(child)
                fold_forked = True

            new_loc = loc + ds * np.array([np.cos(th2), np.sin(th2)])

            # trajectory-intersection catch (small targets / pass-through)
            hit = targets.check_trajectory_intersection(loc, new_loc)  # bool array (N,)
            if np.any(hit):
                idx = int(np.argmax(hit))
                tk.points.append(new_loc); tk.headings.append(th2); tk.stabilities.append(st)
                tk.death = dict(reason='arrival', loc=new_loc.copy(), target_idx=idx)
                return

            loc = new_loc
            h, Rp = th2, R2
            tk.points.append(loc.copy())
            tk.headings.append(h)
            tk.stabilities.append(st)
            if not in_domain(loc):
                tk.death = dict(reason='domain', loc=loc.copy(), target_idx=None)
                return
        tk.death = dict(reason='max_steps', loc=loc.copy(), target_idx=None)

    while queue:
        tk = queue.pop(0)
        integrate(tk, allow_fork=(tk.birth['reason'] == 'root'))

    return SkeletonTree(tracks, root.id, params)


# ----------------------------------------------------------------------------
# Phase 2: rendering + overlay on the empirical GODM heatmaps.
# ----------------------------------------------------------------------------

def _similarity(src, dst):
    """Least-squares similarity (rotation + uniform scale + translation) mapping
    ``src`` onto ``dst`` (both (N,2)); returns a callable transforming (M,2) points.
    For the fly the model targets equal the heatmap posts, so this is ~identity; for
    the locust (data-frame posts, slightly different separation) it is a best fit."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = (D.T @ S) / len(src)
    U, Sig, Vt = np.linalg.svd(cov)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    var_s = (S ** 2).sum() / len(src)
    scale = Sig.sum() / var_s if var_s > 0 else 1.0
    t = mu_d - scale * (Rm @ mu_s)

    def f(P):
        P = np.atleast_2d(np.asarray(P, float))
        return (scale * (Rm @ P.T)).T + t
    return f


def plot_skeleton(tree, ax, *, transform=None, color='k', lw=1.8,
                  mark_bifurcations=True, skip_unstable=False, dash_second_fork=False):
    """Draw the skeleton tracks (solid black) on ``ax``. ``transform`` optionally
    maps model-frame points into the heatmap frame. ``skip_unstable=True`` omits the
    segments traced along an SC-*unstable* branch (e.g. the centre's unstable
    interlude between the first and the reborn-centre bifurcation), so the centre
    track shows only its stable stem and reborn-to-target pieces. ``dash_second_fork``
    draws the second-bifurcation -> centre routes (the PNAS diamond) dashed."""
    def xf(P):
        return transform(P) if transform is not None else np.atleast_2d(P)

    def draw(P, ls='-'):
        ax.plot(P[:, 0], P[:, 1], color=color, lw=lw, ls=ls, solid_capstyle='round',
                solid_joinstyle='round', zorder=5)

    for tk in tree.tracks:
        ls = '--' if (dash_second_fork and tk.birth['reason'] == 'fork2') else '-'
        P = xf(tk.xy)
        if skip_unstable and tk.stabilities:
            st = np.asarray(tk.stabilities, bool)   # segment k joins points[k], points[k+1]
            k, n = 0, len(st)
            while k < n:
                if not st[k]:
                    k += 1
                    continue
                j = k
                while j < n and st[j]:
                    j += 1
                draw(P[k:j + 1], ls)
                k = j
        else:
            draw(P, ls)
    if mark_bifurcations:
        for tk in tree.tracks:
            if tk.birth['reason'] == 'fork':
                p = xf(tk.birth['loc'])[0]
                ax.plot(p[0], p[1], 'o', color=color, ms=4, zorder=6)
                break   # the two arms share one fork point


def _heatmap_overlay(godm_case, nm, ax):
    """Try to draw the empirical GODM heatmap as a background. Returns a transform
    (model-frame -> heatmap-frame) on success, or None (and draws target circles)
    on any failure -- so the figure degrades gracefully without the GODM data.
    ``godm_case`` is the GODM case string ('fly3'/'fly2'/'locust3'/'locust2')."""
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:          # so the import works regardless of CWD
            sys.path.insert(0, _here)
        # fast path: a cached heatmap (model geometry now matches the data, so the
        # frames coincide -> identity transform); falls through to a full recompute
        # if no cache exists.
        cache = os.path.join(_here, f'_hm_{godm_case}.npz')
        if os.path.exists(cache):
            d = np.load(cache)
            img, extent = d['img'], tuple(d['extent'])
            ax.imshow(img, extent=extent, origin='upper', aspect='equal', zorder=0)
            pad = 0.1 * (extent[1] - extent[0])
            ax.set_xlim(extent[0] - pad, extent[1] + pad)
            ax.set_ylim(extent[2] - pad, extent[3] + pad)
            return lambda P: np.atleast_2d(np.asarray(P, float))   # identity
        # godm_heatmaps lives in walker_analysis/ (the empirical-data engine); this
        # script is in plots/, so import it by package path (project root is on
        # sys.path from the top-of-file insert).
        from walker_analysis import godm_heatmaps
        img, extent, posts = godm_heatmaps.compute_heatmap(godm_case, verbose=False)
        if img is None:
            raise RuntimeError('compute_heatmap returned no image')
        post_xy = np.array(list(posts.values()), float)
        model_xy = np.asarray(nm.percep_model.targets.locs, float)
        # correspondence by sorted y (both layouts symmetric about the x-axis)
        src = model_xy[np.argsort(model_xy[:, 1])]
        dst = post_xy[np.argsort(post_xy[:, 1])]
        transform = _similarity(src, dst)
        ax.imshow(img, extent=extent, origin='upper', aspect='equal', zorder=0)
        pad = 0.1 * (extent[1] - extent[0])
        ax.set_xlim(extent[0] - pad, extent[1] + pad)
        ax.set_ylim(extent[2] - pad, extent[3] + pad)
        return transform
    except Exception as exc:        # GODM repo absent, import error, etc.
        warnings.warn(f"GODM heatmap unavailable for {godm_case!r} ({exc}); "
                      "drawing target circles only.")
        nm.percep_model.targets.plot_targets_to_axis(ax)
        return None


def _draw_targets(nm, ax, transform=None, color='0.5', zorder=4):
    """Filled grey target circles at their true radius, mapped through ``transform``
    (model-frame -> heatmap-frame) when one is given so they register with the
    heatmap. A similarity transform maps circles to circles, so the radius is carried
    through via the transformed edge distance -- accurate in either frame. Mirrors
    decision_model.Targets.plot_targets_to_axis (same '0.5' grey); drawn under the
    skeleton (zorder 4 < the skeleton's 5) so the tracks stay visible into the target."""
    targets = nm.percep_model.targets
    if targets.geom_name != 'circle':       # all skeleton CASES are circles; be safe
        targets.plot_targets_to_axis(ax)
        return
    locs = np.asarray(targets.locs, float)
    rr = targets.r
    for n, loc in enumerate(locs):
        r_n = float(rr[n] if isinstance(rr, np.ndarray) else rr)
        if transform is not None:
            c = np.asarray(transform(loc), float).ravel()[:2]
            edge = np.asarray(transform(loc + np.array([r_n, 0.0])), float).ravel()[:2]
            r_draw = float(np.hypot(edge[0] - c[0], edge[1] - c[1]))
        else:
            c, r_draw = loc, r_n
        ax.add_patch(plt.Circle(c, r_draw, color=color, zorder=zorder))


def make_figure(case, *, heatmap=True, ds_step=0.02, save=None, ax=None,
                skip_unstable=True):
    """Build the deterministic-skeleton figure for ``case`` in CASES ('fly', 'locust'),
    overlaid on the empirical GODM heatmap (graceful fallback to target circles).
    ``skip_unstable`` (default True) breaks the centre track across its SC-unstable
    interlude, leaving a gap; set False to draw it continuously through."""
    if case not in CASES:
        raise ValueError(f"case must be one of {sorted(CASES)}, got {case!r}")
    cfg = CASES[case]
    nm = _build_model(cfg)
    tree = trace_skeleton(nm, ds=ds_step, xlim=cfg['xlim'], ylim=cfg['ylim'])

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
    else:
        fig = ax.figure

    transform = _heatmap_overlay(cfg['godm_case'], nm, ax) if heatmap else None
    if transform is None and not heatmap:
        nm.percep_model.targets.plot_targets_to_axis(ax)
    if transform is None:
        ax.set_xlim(*cfg['xlim'])
        ax.set_ylim(*cfg['ylim'])
    else:
        # heatmap drawn: the transform-None paths already drew the targets (untransformed);
        # over the heatmap we draw them in the heatmap frame so they register with it.
        _draw_targets(nm, ax, transform=transform)

    plot_skeleton(tree, ax, transform=transform, skip_unstable=skip_unstable)
    ax.set_aspect('equal')
    ax.set_title(f'{case} stable-track skeleton'
                 + (' over empirical heatmap' if transform is not None else ''))

    hit = {t.target_idx for t in tree.arrivals()}
    if hit != set(range(len(nm.percep_model.targets.locs))):
        warnings.warn(f"{case}: not all targets reached; hit {sorted(hit)} "
                      f"(arrivals: {sorted(t.target_idx for t in tree.arrivals())})")
    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print('wrote', save)
    return fig


# ----------------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------------

def main(argv):
    import argparse
    p = argparse.ArgumentParser(
        description='Deterministic decision-track skeleton from the model bifurcation '
                    'structure (default: skeleton over the empirical GODM heatmap).')
    p.add_argument('case', choices=sorted(CASES) + ['diagram-both'],
                   help='fly or locust (skeleton/branch diagram), or "diagram-both" '
                        'for the combined fly-over-locust heading branch diagram')
    p.add_argument('--branch-diagram', action='store_true',
                   help='Phase 0: plot the (x,theta)+(x,R) bifurcation branch diagram '
                        'instead of the skeleton figure')
    p.add_argument('--no-heatmap', action='store_true',
                   help='skeleton over target circles only (skip the GODM heatmap)')
    p.add_argument('--show-unstable', action='store_true',
                   help='draw the centre track continuously THROUGH its SC-unstable '
                        'interlude (default: break it, leaving a gap)')
    p.add_argument('--ds', type=float, default=0.02, help='streamline step size')
    p.add_argument('--num-x', type=int, default=400, help='branch-diagram x samples')
    p.add_argument('--save', default=None)
    p.add_argument('--no-show', action='store_true')
    args = p.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))

    if args.case == 'diagram-both':
        # The branch diagram is an analysis diagnostic, not a publication figure, so it
        # is written to ../walker_analysis/ (the skeleton figures stay here in plots/).
        wa = os.path.join(os.path.dirname(here), 'walker_analysis')
        save = args.save or os.path.join(wa, 'branch_diagram_both.png')
        plot_diagram_both(num_x=args.num_x, save=save)
        if not args.no_show:
            plt.show()
        return

    cfg = CASES[args.case]

    if args.branch_diagram:
        nm = _build_model(cfg)
        # midline + two off-axis cuts (the compromise arms leave the midline)
        extra = (0.5, 1.0) if args.case.startswith('fly') else (0.3, 0.6)
        # The branch diagram is an analysis diagnostic, not a publication figure, so it
        # is written to ../walker_analysis/ (the skeleton figure stays here in plots/).
        wa = os.path.join(os.path.dirname(here), 'walker_analysis')
        save = args.save or os.path.join(wa, f'branch_diagram_{args.case}.png')
        plot_branch_diagram(nm, y0=0.0, xlim=(0.0, cfg['xlim'][1]),
                            num_x=args.num_x, extra_y=extra,
                            title=f'{args.case}: SC-equilibrium branches (births/deaths)',
                            save=save)
    else:
        save = args.save or os.path.join(here, f'skeleton_{args.case}.png')
        make_figure(args.case, heatmap=not args.no_heatmap, ds_step=args.ds, save=save,
                    skip_unstable=not args.show_unstable)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
