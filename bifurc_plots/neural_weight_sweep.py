"""
Exploratory parameter sweep over neural-weight geometry, with full weighting
of signal contributions (the "standard" model -- ``weight_angle_only=False``).
Companion to ``neural_weight_sweep_angle_only.py``, which isolates the
angle-warping contribution from the front-bias attractiveness weighting.

These plots are intentionally NOT publication-quality. They are sized to be
fast enough for parameter exploration on a many-core machine while still
resolving Hopf regions and saddle-node curves well enough to compare across
weight families. More function-space exploration is needed before any of
these settings can be locked in for publication.

Each figure is an Nx3 panel matrix where one row corresponds to one
parameterization of the perception model:

    col 1 -- PerceptionModel.plot_neural_weight (weighting curve + angle map)
    col 2 -- PerceptionModel.plot_blocked_signals (target-geometry panel)
    col 3 -- NeuralBandModel.plot_bifurcation_diagram (stable count)

Targets are fixed at two circle targets at (4.33, +/- 2.5), r=0.5, matching
the parameterization in compare_sc_vm.ipynb. Observer scans (x, y) with
focal_angle solved self-consistently per point.

Six figures are produced:
    1. cutoff weight, b=pi, varying a:        a in {0, pi/8, pi/4, pi/3}
    2. cutoff weight, varying both a and b:   (a,b) in {(0, 4pi/5), (0, 3pi/4),
                                                          (pi/4, 4pi/5),
                                                          (pi/4, 3pi/4)}
    3. von Mises weight, low k:               k in {0.1, 0.2, 0.3, 0.4, 0.5}
    4. von Mises weight, high k:              k in {0.6, 0.7, 0.8, 0.9}
    5. symmetric Beta weight, b=pi:           alpha in {1.5, 2.0, 3.0, 5.0, 10.0}
    6. regularized power weight, e=1e-3:      d in {0.1, 0.2, 0.4, 0.5}

Bifurcation panels are pinned to max_count=3 for color comparability; any
parameterization whose data exceeds that triggers a captured warning which
is reported in a post-run summary.

Per-figure caching mirrors the angle-only script: the rasterized bifurcation
`img` array for each row is saved to ``_cache_<out_name>.npz`` next to the
figure, with a JSON fingerprint of every input. The cache is invalidated
automatically if any input changes.
"""

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from multiprocessing import Pool

import decision_model as model
from parallel_config import get_n_workers


# ---- fixed setup (matches compare_sc_vm.ipynb) ----
TARGET_LOCS = np.array([[4.33, 2.5], [4.33, -2.5]])
TARGET_GEOM = 'circle'
TARGET_RADIUS = 0.5
FOCAL_LOC = (0, 0)
FOCAL_ANGLE = 0

# Bifurcation diagram settings. Matches the companion angle-only script so
# the two are directly comparable. These are exploration settings; bump
# NUM_X/NUM_Y and REFINEMENT_LEVELS if you want finer Hopf-region resolution
# at the cost of much longer runtime.
XLIM = (0.0, 6.0)
YLIM = (-3.5, 3.5)
NUM_X = 29
NUM_Y = 29
REFINEMENT_LEVELS = 2
MAX_COUNT = 3   # pinned color scale; >3 flagged for follow-up
STABILITY_CRITERION = 'coupled'

# Default size of the multiprocessing pool. Resolved per-machine via
# parallel_config (env var NR_N_WORKERS or machine_config.N_WORKERS).
# Override on the command line with --workers.
DEFAULT_N_WORKERS = get_n_workers()

# Screen-friendly dpi; these are NOT publication plots.
DPI = 150
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Bumped whenever the cache layout or anything that would invalidate a
# previously-saved npz changes. Mismatch forces a recompute.
CACHE_VERSION = 2


# ---- figure specifications ----
# Each row spec is a dict with the parameter values for the perception model
# plus a human-readable label used for the row title and the summary log.

def fig1_spec():
    """Cutoff weight, b=pi, varying a."""
    a_values = [0.0, np.pi/8, np.pi/4, np.pi/3]
    a_labels = [r'$a=0$', r'$a=\pi/8$', r'$a=\pi/4$', r'$a=\pi/3$']
    rows = [
        dict(weight='cutoff', a=a, b=np.pi,
             label=fr'cutoff, {al}, $b=\pi$')
        for a, al in zip(a_values, a_labels)
    ]
    title = (r'Cutoff neural weight, $b=\pi$ fixed, varying $a$')
    return rows, title, 'neural_weight_sweep_cutoff_b_pi.png'


def fig2_spec():
    """Cutoff weight, varying both a and b."""
    combos = [
        (0.0,      4*np.pi/5, r'$a=0$',     r'$b=4\pi/5$'),
        (0.0,      3*np.pi/4, r'$a=0$',     r'$b=3\pi/4$'),
        (np.pi/4,  4*np.pi/5, r'$a=\pi/4$', r'$b=4\pi/5$'),
        (np.pi/4,  3*np.pi/4, r'$a=\pi/4$', r'$b=3\pi/4$'),
    ]
    rows = [
        dict(weight='cutoff', a=a, b=b,
             label=f'cutoff, {al}, {bl}')
        for a, b, al, bl in combos
    ]
    title = r'Cutoff neural weight, varying $a$ and $b$'
    return rows, title, 'neural_weight_sweep_cutoff_mixed_ab.png'


def fig3_spec():
    """von Mises weight, low k."""
    ks = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = [
        dict(weight='vonmises', k=k, label=fr'von Mises, $k={k:g}$')
        for k in ks
    ]
    title = r'von Mises neural weight, low $k$'
    return rows, title, 'neural_weight_sweep_vonmises_low_k.png'


def fig4_spec():
    """von Mises weight, high k."""
    ks = [0.6, 0.7, 0.8, 0.9]
    rows = [
        dict(weight='vonmises', k=k, label=fr'von Mises, $k={k:g}$')
        for k in ks
    ]
    title = r'von Mises neural weight, high $k$'
    return rows, title, 'neural_weight_sweep_vonmises_high_k.png'


def fig5_spec():
    """Symmetric Beta(alpha, alpha) weight, b=pi, varying alpha."""
    alphas = [1.5, 2.0, 3.0, 5.0, 10.0]
    rows = [
        dict(weight='symmetric_beta', alpha=a, b=np.pi,
             label=fr'symmetric beta, $\alpha={a:g}$, $b=\pi$')
        for a in alphas
    ]
    title = r'Symmetric Beta neural weight, $b=\pi$ fixed, varying $\alpha$'
    return rows, title, 'neural_weight_sweep_symmetric_beta_b_pi.png'


def fig6_spec():
    """Regularized power weight, e=1e-3 fixed, varying d."""
    ds = [0.1, 0.2, 0.4, 0.5]
    rows = [
        dict(weight='reg_power', d=d, e=1e-3,
             label=fr'reg power, $d={d:g}$, $e=10^{{-3}}$')
        for d in ds
    ]
    title = (r'Regularized power neural weight, $e=10^{-3}$ fixed, '
             r'varying $d$')
    return rows, title, 'neural_weight_sweep_reg_power_e_1e-3.png'


# ---- model construction ----

def build_models(row_spec):
    """Return (PerceptionModel, NeuralBandModel) configured per row_spec."""
    targets = model.Targets(locs=TARGET_LOCS, geom_name=TARGET_GEOM,
                            r=TARGET_RADIUS)
    percep = model.PerceptionModel(
        targets, FOCAL_LOC, FOCAL_ANGLE,
        neural_weight=row_spec['weight'], neural_angle='integral')
    if row_spec['weight'] == 'cutoff':
        percep.a = row_spec['a']
        percep.b = row_spec['b']
    elif row_spec['weight'] == 'vonmises':
        percep.k = row_spec['k']
    elif row_spec['weight'] == 'symmetric_beta':
        percep.alpha = row_spec['alpha']
        percep.b = row_spec['b']
    elif row_spec['weight'] == 'reg_power':
        percep.d = row_spec['d']
        percep.e = row_spec['e']
    else:
        raise ValueError(f"unsupported weight: {row_spec['weight']!r}")
    nbm = model.NeuralBandModel(percep)
    return percep, nbm


# ---- cache fingerprint helpers ----

def row_param_signature(row_spec):
    """Subset of row_spec that actually affects the bifurcation result. The
    'label' field is purely cosmetic and is intentionally excluded so that
    edits to row labels do not invalidate the cache."""
    if row_spec['weight'] == 'cutoff':
        return dict(weight='cutoff',
                    a=float(row_spec['a']),
                    b=float(row_spec['b']))
    elif row_spec['weight'] == 'vonmises':
        return dict(weight='vonmises', k=float(row_spec['k']))
    elif row_spec['weight'] == 'symmetric_beta':
        return dict(weight='symmetric_beta',
                    alpha=float(row_spec['alpha']),
                    b=float(row_spec['b']))
    elif row_spec['weight'] == 'reg_power':
        return dict(weight='reg_power',
                    d=float(row_spec['d']),
                    e=float(row_spec['e']))
    else:
        raise ValueError(f"unsupported weight: {row_spec['weight']!r}")


def figure_fingerprint(rows):
    """JSON-serializable description of every input that affects the
    bifurcation rasters for this figure. Used as the cache key."""
    return dict(
        cache_version=CACHE_VERSION,
        weight_angle_only=False,
        target_locs=TARGET_LOCS.tolist(),
        target_geom=TARGET_GEOM,
        target_radius=TARGET_RADIUS,
        focal_loc=list(FOCAL_LOC),
        focal_angle=FOCAL_ANGLE,
        xlim=list(XLIM),
        ylim=list(YLIM),
        num_x=NUM_X,
        num_y=NUM_Y,
        refinement_levels=REFINEMENT_LEVELS,
        max_count=MAX_COUNT,
        stability_criterion=STABILITY_CRITERION,
        rows=[row_param_signature(r) for r in rows],
    )


def cache_path_for(out_name):
    base, _ = os.path.splitext(out_name)
    return os.path.join(OUTPUT_DIR, f'_cache_{base}.npz')


def try_load_cache(out_name, fingerprint):
    """Return (imgs, overflow_flags) if cache hits, else None.

    imgs has shape (n_rows, H, W) with int dtype; overflow_flags is a
    length-n_rows bool array recording whether the row's data exceeded
    max_count at compute time.
    """
    path = cache_path_for(out_name)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            stored_fp = json.loads(str(d['fingerprint_json']))
            if stored_fp != fingerprint:
                print(f"  cache fingerprint mismatch for {path}; "
                      "will recompute")
                return None
            imgs = d['imgs']
            overflow_flags = d['overflow_flags'].astype(bool)
    except (OSError, KeyError, json.JSONDecodeError, ValueError) as e:
        print(f"  cache at {path} unreadable ({e}); will recompute")
        return None
    print(f"  loaded bifurcation cache from {path}")
    return imgs, overflow_flags


def save_cache(out_name, fingerprint, imgs, overflow_flags):
    path = cache_path_for(out_name)
    np.savez(path,
             fingerprint_json=np.array(json.dumps(fingerprint, sort_keys=True)),
             imgs=imgs,
             overflow_flags=np.asarray(overflow_flags, dtype=bool))
    print(f"  wrote bifurcation cache to {path}")


# ---- bifurcation rendering helpers ----

def _render_bifurcation_from_img(ax, img, targets, max_count, title):
    """Replay the visual portion of NeuralBandModel.plot_bifurcation_diagram
    from a cached `img` array. Mirrors the imshow + targets overlay + title
    + proxy-handle behaviour at decision_model.py:2500-2532 so that
    cache-replay panels are visually indistinguishable from freshly-computed
    ones.
    """
    cmap = plt.get_cmap('viridis', max_count + 1)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, max_count + 1.5),
                        ncolors=max_count + 1)
    img = np.clip(img, 0, max_count)
    ax.imshow(img, origin='lower',
              extent=[XLIM[0], XLIM[1], YLIM[0], YLIM[1]],
              aspect='equal', interpolation='nearest',
              cmap=cmap, norm=norm)
    targets.plot_targets_to_axis(ax)

    # Proxy artists so a later legend() call would pick up one entry per
    # integer count, matching plot_bifurcation_diagram behavior.
    for n in range(max_count + 1):
        ax.plot([], [], marker='s', markersize=10, linestyle='',
                color=cmap(norm(n)), label=f'{n}')

    if title is not None:
        ax.set_title(title)


def _compute_bifurcation_panel(nbm, ax, pool, title):
    """Call plot_bifurcation_diagram with the standard settings, capture
    its imshow output as a numpy array for caching, and report whether the
    data was clipped (i.e., contained >MAX_COUNT stable equilibria).

    Returns (img, overflowed).
    """
    overflowed = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nbm.plot_bifurcation_diagram(
            xlim=XLIM, ylim=YLIM,
            num_x=NUM_X, num_y=NUM_Y,
            refinement_levels=REFINEMENT_LEVELS,
            max_count=MAX_COUNT,
            pool=pool,
            ax=ax,
            title=title,
            stability_criterion=STABILITY_CRITERION)
        for w in caught:
            msg = str(w.message)
            if 'max_count' in msg and 'clipped' in msg:
                overflowed = True
            else:
                warnings.warn_explicit(
                    w.message, w.category, w.filename, w.lineno)

    # plot_bifurcation_diagram calls ax.imshow once; pull the rasterized
    # int array back out for caching. asarray strips any masked-array
    # wrapping so the saved npz stays plain int.
    img = np.asarray(ax.images[-1].get_array(), dtype=int)
    return img, overflowed


# ---- figure builder ----

def build_figure(rows, suptitle, out_name, pool, overflow_log,
                 regenerate=False):
    """Render one Nx3 figure and write it to OUTPUT_DIR at DPI dpi.

    Parameters
    ----------
    rows : list of dict
        per-row parameter specifications; see fig*_spec() above
    suptitle : str
        figure-level title
    out_name : str
        output filename (placed in OUTPUT_DIR)
    pool : multiprocessing.Pool
        shared worker pool for bifurcation evaluation
    overflow_log : list
        appended to with (out_name, row_label) tuples for any row whose
        bifurcation data exceeds MAX_COUNT
    regenerate : bool
        if True, ignore any existing cache and recompute from scratch
    """
    n = len(rows)
    fingerprint = figure_fingerprint(rows)

    cached = None if regenerate else try_load_cache(out_name, fingerprint)
    cached_imgs, cached_overflow = (None, None)
    if cached is not None:
        cached_imgs, cached_overflow = cached

    fig, axes = plt.subplots(n, 3, figsize=(16, 4.5*n),
                             squeeze=False)

    # We collect freshly-computed imgs/overflow for the rows that hit the
    # compute path. If any row recomputed, we rewrite the whole cache at
    # the end so it stays internally consistent.
    fresh_imgs = [None]*n
    fresh_overflow = [False]*n
    any_recomputed = (cached_imgs is None)

    for r, row_spec in enumerate(rows):
        percep, nbm = build_models(row_spec)

        ax_w, ax_g, ax_b = axes[r]

        # --- col 1: neural weight + angle-mapping overlay ---
        percep.plot_neural_weight(ax=ax_w)
        ax_w.set_title(f'{row_spec["label"]}\nNeural weight & angle map',
                       fontsize=11)

        # --- col 2: target geometry + neural-direction lines ---
        percep.plot_blocked_signals(ax=ax_g)
        ax_g.set_title(f'{row_spec["label"]}\nTarget geometry &'
                       ' neural directions', fontsize=11)

        # --- col 3: bifurcation diagram (cached or fresh) ---
        title = (f'{row_spec["label"]}\n'
                 '# stable self-consistent equilibria')
        if cached_imgs is not None:
            _render_bifurcation_from_img(
                ax_b, cached_imgs[r], percep.targets, MAX_COUNT, title)
            row_overflow = bool(cached_overflow[r])
            fresh_imgs[r] = cached_imgs[r]
            fresh_overflow[r] = row_overflow
        else:
            img, row_overflow = _compute_bifurcation_panel(
                nbm, ax_b, pool, title)
            fresh_imgs[r] = img
            fresh_overflow[r] = row_overflow

        if row_overflow:
            overflow_log.append(
                (out_name, row_spec['label'],
                 f"data contained >{MAX_COUNT} stable equilibria; clipped"))
            print(f"  [overflow] {row_spec['label']}: "
                  f">{MAX_COUNT} stable equilibria detected, clipped")

        ax_b.set_xlabel('observer x')
        ax_b.set_ylabel('observer y')

    # Persist the cache when anything had to be recomputed (or when no
    # cache existed before). Stacking imgs is safe because num_x/num_y/
    # refinement_levels are uniform across rows.
    if any_recomputed:
        imgs_arr = np.stack(fresh_imgs, axis=0).astype(np.int16)
        save_cache(out_name, fingerprint, imgs_arr, fresh_overflow)

    # Shared legend for the bifurcation count colors, in figure-level space.
    cmap = plt.get_cmap('viridis', MAX_COUNT + 1)
    handles = [plt.Line2D([], [], marker='s', markersize=10,
                          linestyle='', color=cmap(i / MAX_COUNT),
                          label=f'{i}')
               for i in range(MAX_COUNT + 1)]
    fig.legend(handles=handles, title='# stable\nequilibria',
               loc='center right', bbox_to_anchor=(0.995, 0.5),
               frameon=False)

    fig.suptitle(suptitle, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.96, 0.985])

    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Generate the six-figure neural-weight parameter "
                     "sweep. Bifurcation rasters are cached per-figure so "
                     "layout tweaks don't trigger recomputation."))
    parser.add_argument('--regenerate', action='store_true',
                        help="ignore existing per-figure caches and "
                             "recompute every bifurcation panel")
    parser.add_argument('--only', type=int, choices=[1, 2, 3, 4, 5, 6],
                        action='append', default=None,
                        help="only build the listed figure(s) (1-6); may "
                             "be passed multiple times. Default: all.")
    parser.add_argument('--workers', type=int, default=DEFAULT_N_WORKERS,
                        help=f"size of the multiprocessing pool. Default: "
                             f"{DEFAULT_N_WORKERS}. Bump this up when "
                             f"running on a many-core machine.")
    return parser.parse_args()


def main():
    args = parse_args()
    all_specs = [fig1_spec(), fig2_spec(), fig3_spec(), fig4_spec(),
                 fig5_spec(), fig6_spec()]
    if args.only:
        selected = sorted(set(args.only))
        figure_specs = [all_specs[i-1] for i in selected]
    else:
        figure_specs = all_specs

    print(f"Using {args.workers} worker(s) for bifurcation evaluation.")

    overflow_log = []
    with Pool(args.workers) as pool:
        for rows, suptitle, out_name in figure_specs:
            print(f"\n=== Building {out_name} ({len(rows)} rows) ===")
            build_figure(rows, suptitle, out_name, pool, overflow_log,
                         regenerate=args.regenerate)

    print("\n=== Summary ===")
    if overflow_log:
        print(f"{len(overflow_log)} parameterization(s) exceeded "
              f"max_count={MAX_COUNT}; follow up on these:")
        for out_name, label, msg in overflow_log:
            print(f"  - {out_name}: {label}")
            print(f"      ({msg})")
    else:
        print(f"No parameterization exceeded max_count={MAX_COUNT}.")


if __name__ == "__main__":
    main()
