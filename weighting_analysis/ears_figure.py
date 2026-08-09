"""Regenerate ears_figure.png and ear_diagnostic.png (the warp-vs-weight "ears").

The original 2026-05 figures were produced by throwaway scripts that were never
committed (README.md, "Reproduction", said the code lived in the conversation
transcript). They are now stale for two independent reasons:

  1. **The wrapping-extent fix** (decision_model.py, 2026-08-04). The closest
     target's angular extent is now unwrapped before the neural weight is
     integrated, so a target straddling the rear branch cut is no longer
     silently dropped from perception. That bug bit precisely where the ears
     live -- close to a target, under UNIFORM weight (which has support all the
     way to the rear, so nothing masked it). The right-hand column of the old
     figure therefore showed phantom structure.
  2. **The default stability criterion** moved from `'coupled'` to `'reduced'`
     (2026-06-08), and warp/weight were decoupled, retiring the
     `weight_angle_only` flag the old titles used.

This script pins the whole computation so the figures can be rebuilt. It uses
the CURRENT model defaults -- `stability_criterion='reduced'`, `K=2` -- rather
than the old `'coupled'`/`K=1` pair. (Stable counts are K-invariant at an SC
equilibrium; see .claude/rules/torque-and-stability.md. The criterion is not
invariant, so the numbers here are not directly comparable to the old table --
see README.md.)

Vocabulary: the old FULL / ANGLE-only pair is now
    FULL    = neural_angle_dist=W, angle_weight='neural_angle_dist'
    UNIFORM = neural_angle_dist=W, angle_weight=None          (the model default)

Each of the eight rasters is cached in `_cache_ears.npz` behind its own JSON
fingerprint and written as soon as it finishes, so an interrupted run resumes.

Run:
    python weighting_analysis/ears_figure.py             # both figures
    python weighting_analysis/ears_figure.py sweep       # ears_figure.png only
    python weighting_analysis/ears_figure.py diagnostic  # ear_diagnostic.png only
    python weighting_analysis/ears_figure.py --regenerate
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from multiprocessing import Pool

import decision_model as model
from parallel_config import get_n_workers

pi = np.pi
HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------- setup --
TARGET_LOCS = np.array([[4.33, 2.5], [4.33, -2.5]])
TARGET_R = 0.5
XLIM, YLIM = (0.0, 6.0), (-3.5, 3.5)
NUM_X, NUM_Y = 37, 43
# 2, not 3. The virtual grid is ((num_x-1)*2**L + 1) x ((num_y-1)*2**L + 1), so
# L=2 gives 145 x 169 -- cell area 1.7e-3 sq units, ~1200 cells inside a
# 2 sq-unit ear, which is ample for the area estimate. L=3 quadruples the
# refinement work for no measurable gain in the numbers reported here, and the
# most-peaked row ('cutoff' a=0, b=pi) is by far the slowest cell to evaluate.
REFINEMENT = 2
MAX_COUNT = 3
CRITERION = 'reduced'
# BETA is the neural Boltzmann factor. This scene has 2 targets and the shipped
# figures were produced under the earlier per-target temperature T=0.2, whose
# effective coupling was N_targets/T, so beta = 2/0.2 = 10 reproduces them.
K, BETA = 2.0, 10.0

# The ear region: to the RIGHT of the target line, where a close-but-off-axis
# target would otherwise dominate a far-target commitment by visual extent.
EAR_X_MIN = 3.0

# (label, warp family, a_warp, b_warp)
ROWS = [
    ('smooth cutoff  $a=0,\\ b=\\pi$   (most peaked)', 'cutoff', 0.0, pi),
    ('von Mises  $k=0.9$   (high peak)',               'vonmises', 0.9, None),
    ('von Mises  $k=0.5$   (moderate)',                'vonmises', 0.5, None),
    ('smooth cutoff  $a=\\pi/3,\\ b=\\pi$   (mild)',   'cutoff', pi/3, pi),
]

# The worked example from README.md, "Why the ears exist".
DIAG_LOC = (5.0, 2.0)
DIAG_ROW = 0                      # cutoff a=0, b=pi

N_WORKERS = get_n_workers()


def build(warp, a_warp, b_warp, tied):
    """tied=True -> FULL weighting; tied=False -> uniform weight."""
    targets = model.Targets(locs=TARGET_LOCS, geom_name='circle', r=TARGET_R)
    pm = model.PerceptionModel(
        targets, (0, 0), 0,
        neural_angle_dist=warp,
        angle_weight='neural_angle_dist' if tied else None,
        a_warp=a_warp, b_warp=b_warp)
    return model.NeuralBandModel(pm, beta=BETA, K=K)


# ------------------------------------------------------------------ cache ---

def compute_rasters(regenerate=False):
    """Compute (or load) the eight rasters: one per row per weighting mode.

    The cache is written **incrementally, after every raster**, and each raster
    carries its own fingerprint. The most-peaked row takes tens of minutes, so a
    run that is interrupted (or a config edit that only touches one row) must
    not throw away the rows that are already done.
    """
    path = os.path.join(HERE, '_cache_ears.npz')
    have = {}
    if not regenerate and os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as d:
                for key in d.files:
                    if key.startswith('img_'):
                        have[key] = d[key]
                    elif key.startswith('fp_'):
                        have[key] = str(d[key])
            print(f'  cache holds {sum(k.startswith("img_") for k in have)}'
                  ' raster(s)')
        except (OSError, ValueError) as e:
            print(f'  cache unreadable ({e}); recomputing')
            have = {}

    def _save():
        np.savez(path, **have)

    full, unif = [], []
    pool = None
    try:
        for ri, (label, warp, a_w, b_w) in enumerate(ROWS):
            for tied, store in ((True, full), (False, unif)):
                fp = json.dumps(_row_fp(warp, a_w, b_w, tied), sort_keys=True)
                # Key on a hash of the raster's own fingerprint, NOT on its
                # position in ROWS: --rows and --criterion both change which
                # raster a given index refers to, and an index-keyed cache
                # would silently overwrite an unrelated entry.
                tag = hashlib.md5(fp.encode()).hexdigest()[:12]
                if have.get(f'fp_{tag}') == fp and f'img_{tag}' in have:
                    store.append(have[f'img_{tag}'])
                    print(f'  raster {warp} a={a_w} '
                          f'{"FULL" if tied else "UNIFORM"}: cached')
                    continue
                if pool is None:
                    pool = Pool(N_WORKERS)
                print(f'  raster {warp} a={a_w} '
                      f"{'FULL' if tied else 'UNIFORM'} [{CRITERION}]"
                      ' -- computing ...', flush=True)
                nm = build(warp, a_w, b_w, tied)
                figt, axt = plt.subplots()
                nm.plot_bifurcation_diagram(
                    xlim=XLIM, ylim=YLIM, num_x=NUM_X, num_y=NUM_Y,
                    refinement_levels=REFINEMENT, max_count=MAX_COUNT,
                    pool=pool, ax=axt, stability_criterion=CRITERION)
                img = np.asarray(axt.images[-1].get_array(), dtype=np.int16)
                plt.close(figt)
                store.append(img)
                have[f'img_{tag}'] = img
                have[f'fp_{tag}'] = np.array(fp)
                _save()
                print(f'  raster {warp} a={a_w} '
                      f'{"FULL" if tied else "UNIFORM"}: done, cached',
                      flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return np.stack(full), np.stack(unif)


def _row_fp(warp, a_w, b_w, tied):
    """Fingerprint for a single raster: everything that affects its values."""
    return dict(locs=TARGET_LOCS.tolist(), r=TARGET_R, xlim=list(XLIM),
                ylim=list(YLIM), num_x=NUM_X, num_y=NUM_Y,
                refinement=REFINEMENT, max_count=MAX_COUNT,
                criterion=CRITERION, K=K, beta=BETA,
                warp=warp, a_warp=a_w, b_warp=b_w, tied=bool(tied))


# ------------------------------------------------------------- ear metrics --

def ear_mask(f, u, extent):
    """Cells right of EAR_X_MIN where FULL has strictly more stable equilibria."""
    ny, nx = f.shape
    xs = np.linspace(extent[0], extent[1], nx)
    right = (xs >= EAR_X_MIN)[None, :]
    return (f > u) & right


def mask_area(mask, extent):
    ny, nx = mask.shape
    cell = ((extent[1] - extent[0]) / (nx - 1)) * ((extent[3] - extent[2]) / (ny - 1))
    return mask.sum() * cell


# ---------------------------------------------------------------- figure 1 --

def figure_sweep(full, unif):
    extent = [XLIM[0], XLIM[1], YLIM[0], YLIM[1]]
    n = len(ROWS)
    fig, axes = plt.subplots(n, 4, figsize=(19.0, 4.3*n), squeeze=False)
    cmap = plt.get_cmap('viridis', MAX_COUNT + 1)
    norm = BoundaryNorm(np.arange(-0.5, MAX_COUNT + 1.5), MAX_COUNT + 1)
    targets = model.Targets(locs=TARGET_LOCS, geom_name='circle', r=TARGET_R)

    areas = []
    for r, (label, warp, a_w, b_w) in enumerate(ROWS):
        f, u = np.clip(full[r], 0, MAX_COUNT), np.clip(unif[r], 0, MAX_COUNT)
        mask = ear_mask(f, u, extent)
        area = mask_area(mask, extent)
        areas.append(area)
        diff = f.astype(int) - u.astype(int)
        lim = max(1, int(np.abs(diff).max()))

        panels = [
            (f, 'count', 'FULL weighting\n' + r"(angle_weight='neural_angle_dist')"),
            (u, 'count', 'UNIFORM weight\n(angle_weight=None -- the model default)'),
            (diff, 'diff', 'FULL $-$ UNIFORM\nred = FULL has more'),
            (mask.astype(int), 'mask',
             f'Ear mask (FULL > UNIFORM, $x \\geq {EAR_X_MIN:g}$)\n'
             f'{mask.sum()} cells = {area:.2f} sq. units'),
        ]
        for c, (img, kind, ttl) in enumerate(panels):
            ax = axes[r, c]
            if kind == 'count':
                im = ax.imshow(img, origin='lower', extent=extent, aspect='equal',
                               interpolation='nearest', cmap=cmap, norm=norm)
            elif kind == 'diff':
                im = ax.imshow(img, origin='lower', extent=extent, aspect='equal',
                               interpolation='nearest', cmap='RdBu_r',
                               vmin=-lim, vmax=lim)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            else:
                ax.imshow(img, origin='lower', extent=extent, aspect='equal',
                          interpolation='nearest', cmap='Reds', vmin=0, vmax=1)
            targets.plot_targets_to_axis(ax)
            ax.set_xlim(*XLIM)
            ax.set_ylim(*YLIM)
            ax.set_title(ttl, fontsize=9.5)
            ax.set_xlabel('observer x', fontsize=9)
            if c == 0:
                ax.set_ylabel('observer y', fontsize=9)
            ax.tick_params(labelsize=8)
        axes[r, 0].text(-0.30, 0.5, label, transform=axes[r, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=11)

    handles = [plt.Line2D([], [], marker='s', ms=12, ls='', color=cmap(norm(i)),
                          label=f'{i}') for i in range(MAX_COUNT + 1)]
    fig.legend(handles=handles, title='# stable\nequilibria', loc='upper right',
               bbox_to_anchor=(1.0, 0.985), frameon=False, fontsize=10,
               title_fontsize=10)
    fig.suptitle(
        'The far-target "ear": the one structural difference between FULL '
        'weighting and uniform weight\n'
        f'NBM, two circle targets at (4.33, $\\pm$2.5), r={TARGET_R}; '
        f"stability_criterion='{CRITERION}'; K={K:g}, $\\beta$={BETA:g}; "
        f'grid {NUM_X}x{NUM_Y} + {REFINEMENT} refinement passes',
        fontsize=13, y=0.995)
    fig.tight_layout(rect=[0.015, 0, 1, 0.975])
    out = os.path.join(HERE, 'ears_figure.png')
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', out)

    print('\n  ear areas (sq. units of (x, y) space):')
    for (label, warp, a_w, b_w), area in zip(ROWS, areas):
        pretty = label.split('  ')[0] + ' ' + label.split('  ')[1]
        print(f'    {pretty:44s} {area:.2f}')
    return areas


# ---------------------------------------------------------------- figure 2 --

def figure_diagnostic():
    """Mechanism at a single observer position inside the upper ear."""
    label, warp, a_w, b_w = ROWS[DIAG_ROW]
    o = np.array(DIAG_LOC, dtype=float)
    nm_full = build(warp, a_w, b_w, True)
    nm_unif = build(warp, a_w, b_w, False)

    bear = np.arctan2(TARGET_LOCS[:, 1] - o[1], TARGET_LOCS[:, 0] - o[0])
    dist = np.linalg.norm(TARGET_LOCS - o, axis=1)
    close, far = (0, 1) if dist[0] < dist[1] else (1, 0)

    # Visual extents (angular width of each target as seen from o).
    tg = nm_full.percep_model.targets
    ext = tg.get_percep_angles(o, 0.0)
    # An extent that straddles +-pi comes back as a WRAPPING pair (lo > hi);
    # its true angular width is hi - lo + 2*pi. (This is the same convention
    # the 2026-08-04 wrapping fix is about -- see the module docstring.)
    widths = [float(hi - lo) if hi >= lo else float(hi - lo + 2*pi)
              for lo, hi in ext]

    # rho at the FAR-target heading (the commitment that FULL rescues).
    heading = bear[far]
    rho = {}
    for name, nm in (('FULL', nm_full), ('UNIFORM', nm_unif)):
        _a, r = nm.percep_model._get_target_signals(focal_angle=heading,
                                                    focal_loc=o)
        order = np.argsort(dist)
        vals = np.zeros(2)
        for slot, v in zip(order[:len(r)], r):
            vals[slot] = v
        rho[name] = vals

    # Self-consistency scan: Re(dgamma/dt) along gamma = R + 0j at that heading.
    Rs = np.linspace(0.005, 0.995, 400)
    resid = {}
    for name, nm in (('FULL', nm_full), ('UNIFORM', nm_unif)):
        nm.percep_model.focal_loc = o
        nm.percep_model.focal_angle = heading
        vals = np.array([np.real(nm.dgamma_dt(gamma=complex(R, 0.0),
                                              focal_angle=heading, focal_loc=o))
                         for R in Rs])
        resid[name] = vals

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))

    # (a) geometry
    ax = axes[0]
    for i, (tx, ty) in enumerate(TARGET_LOCS):
        ax.annotate('', xy=(tx, ty), xytext=o,
                    arrowprops=dict(arrowstyle='-', color='0.75', lw=1.0))
        ax.add_patch(plt.Circle((tx, ty), TARGET_R, fill=False, lw=1.8,
                                color='k', zorder=4))
        # Labels go to the RIGHT of each circle: the panel is equal-aspect and
        # much taller than it is wide, and the heading arrow points nearly
        # straight down, so the right-hand strip is the only clear space.
        ax.text(tx + 0.85, ty + (0.45 if i == 0 else -0.45),
                f'target {i}\nd = {dist[i]:.2f}\n'
                f'extent {np.degrees(widths[i]):.1f}°',
                fontsize=8.5, ha='left', va='center')
    ax.plot(*o, 'o', color='0.2', ms=9, zorder=5)
    ax.annotate('', xy=o + 1.3*np.array([np.cos(heading), np.sin(heading)]),
                xytext=o, arrowprops=dict(arrowstyle='-|>', color='tab:purple',
                                          lw=2.6, mutation_scale=18))
    ax.text(o[0] - 0.15, o[1] - 0.55, f'observer\n{tuple(DIAG_LOC)}',
            fontsize=8.5, ha='right', va='center')
    ax.text(0.04, 0.98, 'candidate heading =\nthe FAR target '
                        f'({np.degrees(heading):.1f}°)',
            transform=ax.transAxes, color='tab:purple', fontsize=8.5, va='top')
    ax.set_aspect('equal')
    ax.set_xlim(2.9, 7.6)
    ax.set_ylim(-3.6, 4.4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('(a)  Inside the upper ear', loc='left')

    # (b) rho split at the far-target heading
    ax = axes[1]
    idx = np.arange(2)
    w = 0.36
    for j, name in enumerate(('FULL', 'UNIFORM')):
        ax.bar(idx + (j - 0.5)*w, rho[name], w,
               color=('tab:blue' if name == 'FULL' else 'tab:gray'), label=name)
    ax.set_xticks(idx)
    ax.set_xticklabels([f'target {close}\n(CLOSE, off-axis)',
                        f'target {far}\n(FAR, being faced)'])
    ax.set_ylabel(r'perceptual mass  $\rho$')
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='0.6', lw=0.9, ls=':')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    ax.set_title('(b)  $\\rho$ at the far-target heading:\n'
                 'front bias squashes the off-axis close target',
                 loc='left', fontsize=10)

    # (c) the self-consistency scan
    ax = axes[2]
    for name, col in (('FULL', 'tab:blue'), ('UNIFORM', 'tab:gray')):
        ax.plot(Rs, resid[name], color=col, lw=2.0, label=name)
        sgn = np.sign(resid[name])
        cross = np.nonzero(np.diff(sgn) != 0)[0]
        for c in cross:
            Rstar = Rs[c] - resid[name][c]*(Rs[c+1]-Rs[c]) / (
                resid[name][c+1]-resid[name][c])
            ax.plot([Rstar], [0], 'o', color=col, ms=8, zorder=5)
            # The near-zero crossing is the unstable partner of the R>0 branch
            # (the pair born in the saddle-node that creates the commitment);
            # label the committed one and mark the other more quietly.
            if Rstar > 0.05:
                ax.annotate(f'committed $R^*$ = {Rstar:.3f}', (Rstar, 0),
                            textcoords='offset points', xytext=(8, 16),
                            color=col, fontsize=9.5)
            else:
                ax.annotate(f'unstable partner\n$R$ = {Rstar:.3f}', (Rstar, 0),
                            textcoords='offset points', xytext=(4, -42),
                            color=col, fontsize=8)
    ax.axhline(0, color='k', lw=1.0)
    ax.set_xlabel(r'$R = |\gamma|$   (with $\gamma = R + 0j$)')
    ax.set_ylabel(r'$\mathrm{Re}\ d\gamma/dt$')
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_title('(c)  Self-consistency at that heading:\n'
                 'a zero crossing = the far-target commitment exists',
                 loc='left', fontsize=10)

    fig.suptitle('Why the ear exists: front-bias weighting rescues a far-target '
                 f'commitment that uniform weight cannot support\n'
                 f'({label.split("   ")[0]}; observer {tuple(DIAG_LOC)}; '
                 f"criterion='{CRITERION}')", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(HERE, 'ear_diagnostic.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', out)

    print(f'\n  diagnostic at {DIAG_LOC}, {label}:')
    for i in range(2):
        print(f'    target {i}: bearing {np.degrees(bear[i]):+.1f}°, '
              f'distance {dist[i]:.2f}, visual extent '
              f'{np.degrees(widths[i]):.1f}°')
    for name in ('FULL', 'UNIFORM'):
        print(f'    {name:8s} rho at far-target heading = '
              f'{np.round(rho[name], 3).tolist()}')
        sgn = np.sign(resid[name])
        n_cross = int((np.diff(sgn) != 0).sum())
        print(f'             Re dgamma/dt zero crossings on R in (0,1): '
              f'{n_cross}')
    # Equilibria actually found by the solver, for the README table.
    for name, nm in (('FULL', nm_full), ('UNIFORM', nm_unif)):
        angles, stab = nm.sc_equilib(focal_loc=o,
                                     stability_criterion=CRITERION)
        st = [np.degrees(a) for a, s in zip(angles, stab) if s]
        print(f'    {name:8s} stable SC headings: '
              f'{[f"{a:+.1f}" for a in sorted(st)]}')


def main(argv=None):
    global CRITERION, ROWS
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('what', nargs='*', default=['all'],
                    help='sweep, diagnostic, or all')
    ap.add_argument('--regenerate', action='store_true')
    ap.add_argument('--criterion', default=CRITERION,
                    choices=['reduced', 'coupled', 'discrim_a'],
                    help="stability criterion (default %(default)s -- the "
                         "model default). Use 'coupled' to reproduce the "
                         "2026-05 run's criterion.")
    ap.add_argument('--rows', type=int, nargs='*', default=None,
                    help='restrict to these row indices (default: all). Useful '
                         'for the slow most-peaked row 0 alone.')
    ap.add_argument('--areas-only', action='store_true',
                    help='print ear areas without writing the figure (for '
                         'A/B comparisons against a different --criterion)')
    args = ap.parse_args(argv)
    todo = ['sweep', 'diagnostic'] if 'all' in args.what else args.what
    CRITERION = args.criterion
    if args.rows is not None:
        ROWS = [ROWS[i] for i in args.rows]
    print(f'workers: {N_WORKERS}   criterion: {CRITERION}   '
          f'rows: {len(ROWS)}')
    if 'sweep' in todo:
        print('[sweep]')
        full, unif = compute_rasters(regenerate=args.regenerate)
        if args.areas_only:
            extent = [XLIM[0], XLIM[1], YLIM[0], YLIM[1]]
            print('\n  ear areas (sq. units):')
            for (label, *_r), f, u in zip(ROWS, full, unif):
                area = mask_area(ear_mask(np.clip(f, 0, MAX_COUNT),
                                         np.clip(u, 0, MAX_COUNT), extent),
                                 extent)
                print(f'    {label.split("   ")[0]:44s} {area:.2f}')
        else:
            figure_sweep(full, unif)
    if 'diagnostic' in todo:
        print('[diagnostic]')
        figure_diagnostic()


if __name__ == '__main__':
    main()
