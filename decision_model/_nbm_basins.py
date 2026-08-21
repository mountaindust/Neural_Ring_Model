"""Heading-basin wheels for ``NeuralBandModel`` (mixin).

At an observer location, sweeping the heading around the circle from a neutral
(uncommitted) neural state, each heading commits under the slaved-gamma flow to
one stable self-consistent direction -- the "basin" of that direction.  A wheel
shows those heading-basins (annulus arcs) and the directions (arrows); the
overlay draws one per region of a bifurcation raster.

Migrated from the basin_estimation/ vetting prototype.  Theory and findings are
in theory/basins_of_attraction.md; the code-editing gotchas are in
.claude/rules/basin-estimation.md.
"""

import numpy as np

from .angles import convert_angles


class _BasinMixin:
    '''Basin-estimation and basin-wheel overlay methods of ``NeuralBandModel``.'''

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
            turn = dtheta * dt
            theta = convert_angles(theta + turn)
            # Same transport as the walker: the neural angles all shift by
            # -turn, so the carried gamma rotates with them.
            gamma = gamma * np.exp(-1j*turn)
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
            cells = _basin_lattice_placement(
                count_field, xy_of_pixel, wheel_radius, nx_max=nx_max,
                ny_max=ny_max, min_sep_factor=min_sep_factor,
                max_sep_factor=max_sep_factor, min_area=min_area)
        elif placement == 'grid':
            cells = _basin_grid_placement(
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
                    wheels.append(_reflect_wheel(src))
        _render_basin_wheels(ax, wheels, wheel_radius)
        # draw the targets fully opaque, on top of the wheels
        tg.plot_targets_to_axis(ax, zorder=7)

# ------------------------------------------------------------------
# Wheel placement and rendering: pure geometry, no model state.
# ------------------------------------------------------------------

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

def _reflect_wheel(w):
    """Mirror a wheel across the x-axis for the y<0 rows (valid under the
    x-axis symmetry assumption): negate the focal y, the stable
    directions, and each arc's heading bounds; labels/widths unchanged."""
    return {'focal_loc': (w['focal_loc'][0], -w['focal_loc'][1]),
            'stable': [-s for s in w['stable']],
            'arcs': [(-b, -a, lab) for (a, b, lab) in w['arcs']],
            'widths': dict(w['widths'])}

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
