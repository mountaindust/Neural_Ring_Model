"""
Exploratory parameter sweep over neural-weight geometry with
``weight_angle_only=True`` -- i.e. the neural-weight function is used only
to warp the neural-angle map and is NOT used to weight signal contributions
(group-size integration sees a flat weight). This isolates the contribution
of the angle warping from the front-bias attractiveness weighting.

These plots are intentionally NOT publication-quality. They are sized to be
fast enough for parameter exploration on a many-core machine while still
being good enough to resolve approximate Hopf regions in the bifurcation
diagrams. For the publication-quality counterparts (weight_angle_only=False,
high dpi, fine bifurcation grid) see ``hq_bifurc_plots/neural_weight_sweep.py``.

Each figure is an Nx3 panel matrix where one row corresponds to one
parameterization of the perception model:

    col 1 -- PerceptionModel.plot_neural_weight (weighting curve + angle map)
    col 2 -- PerceptionModel.plot_blocked_signals (target-geometry panel)
    col 3 -- NeuralBandModel.plot_bifurcation_diagram

Four figures are produced. Their parameter rows mirror a subset of the
corresponding HQ figures so the side-by-side comparison is direct:

    1. cutoff weight, b=pi, varying a:
         a in {0, pi/8, pi/4, pi/3}
         HQ counterpart: hq_bifurc_plots/neural_weight_sweep_cutoff_b_pi.png

    2. symmetric Beta(alpha, alpha) weight, b=pi, varying alpha
       (first four rows of the HQ figure):
         alpha in {1.5, 2.0, 3.0, 5.0}
         HQ counterpart:
           hq_bifurc_plots/neural_weight_sweep_symmetric_beta_b_pi.png

    3. von Mises weight, low k:
         k in {0.1, 0.2, 0.3, 0.4, 0.5}
         HQ counterpart: hq_bifurc_plots/neural_weight_sweep_vonmises_low_k.png

    4. von Mises weight, high k:
         k in {0.6, 0.7, 0.8, 0.9}
         HQ counterpart: hq_bifurc_plots/neural_weight_sweep_vonmises_high_k.png

Per-figure caching mirrors the HQ script: the rasterized bifurcation `img`
array for each row is saved to ``_cache_<out_name>.npz`` next to the figure,
alongside a JSON fingerprint of every input that affects the result. The
cache is invalidated automatically if any input (including
``weight_angle_only``) changes.

Bifurcation panels are pinned to ``max_count=3`` for color comparability;
any parameterization whose data exceeds that triggers a captured warning
which is reported in a post-run summary.
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


# ---- fixed setup (matches hq_bifurc_plots/neural_weight_sweep.py) ----
TARGET_LOCS = np.array([[4.33, 2.5], [4.33, -2.5]])
TARGET_RADIUS = 0.5
FOCAL_LOC = (0, 0)
FOCAL_ANGLE = 0

# All rows use weight_angle_only=True. This is the whole point of this
# script and is part of the cache fingerprint -- changing it forces a
# recompute.
WEIGHT_ANGLE_ONLY = True

# Bifurcation diagram settings. Lower than the HQ script (which uses
# num_x=41, num_y=41, refinement_levels=4) because these are exploratory
# plots, not publication output. The chosen settings still resolve Hopf
# regions well enough for parameter exploration.
XLIM = (0.0, 6.0)
YLIM = (-3.5, 3.5)
NUM_X = 29
NUM_Y = 29
REFINEMENT_LEVELS = 2
MAX_COUNT = 3
STABILITY_CRITERION = 'coupled'

# 4 by default for a quad-core laptop; bump up via --workers when running
# remotely on the many-core box.
DEFAULT_N_WORKERS = 4

# Screen-friendly dpi; these are NOT publication plots.
DPI = 150
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Bumped whenever the cache layout or anything that would invalidate a
# previously-saved npz changes.
CACHE_VERSION = 1


# ---- figure specifications ----

def fig1_spec():
    """Cutoff weight, b=pi, varying a -- angle-warping only.

    Companion to hq_bifurc_plots/neural_weight_sweep_cutoff_b_pi.png.
    """
    a_values = [0.0, np.pi/8, np.pi/4, np.pi/3]
    a_labels = [r'$a=0$', r'$a=\pi/8$', r'$a=\pi/4$', r'$a=\pi/3$']
    rows = [
        dict(weight='cutoff', a=a, b=np.pi,
             label=fr'cutoff (angle-only), {al}, $b=\pi$')
        for a, al in zip(a_values, a_labels)
    ]
    title = (r'Cutoff neural weight (angle warping only), $b=\pi$ fixed, '
             r'varying $a$' '\n'
             r'companion to hq_bifurc_plots/'
             r'neural_weight_sweep_cutoff_b_pi.png')
    return rows, title, 'neural_weight_sweep_cutoff_b_pi_angle_only.png'


def fig2_spec():
    """Symmetric Beta weight, b=pi, varying alpha (first four rows of the
    HQ figure) -- angle-warping only.

    Companion to hq_bifurc_plots/neural_weight_sweep_symmetric_beta_b_pi.png
    (first four rows; the alpha=10 row of the HQ figure is intentionally
    omitted here).
    """
    alphas = [1.5, 2.0, 3.0, 5.0]
    rows = [
        dict(weight='symmetric_beta', alpha=a, b=np.pi,
             label=fr'sym. beta (angle-only), $\alpha={a:g}$, $b=\pi$')
        for a in alphas
    ]
    title = (r'Symmetric Beta neural weight (angle warping only), '
             r'$b=\pi$ fixed, varying $\alpha$' '\n'
             r'companion to hq_bifurc_plots/'
             r'neural_weight_sweep_symmetric_beta_b_pi.png (first 4 rows)')
    return rows, title, \
        'neural_weight_sweep_symmetric_beta_b_pi_first4_angle_only.png'


def fig3_spec():
    """von Mises weight, low k -- angle-warping only.

    Companion to hq_bifurc_plots/neural_weight_sweep_vonmises_low_k.png.
    """
    ks = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = [
        dict(weight='vonmises', k=k,
             label=fr'von Mises (angle-only), $k={k:g}$')
        for k in ks
    ]
    title = (r'von Mises neural weight (angle warping only), low $k$' '\n'
             r'companion to hq_bifurc_plots/'
             r'neural_weight_sweep_vonmises_low_k.png')
    return rows, title, 'neural_weight_sweep_vonmises_low_k_angle_only.png'


def fig4_spec():
    """von Mises weight, high k -- angle-warping only.

    Companion to hq_bifurc_plots/neural_weight_sweep_vonmises_high_k.png.
    """
    ks = [0.6, 0.7, 0.8, 0.9]
    rows = [
        dict(weight='vonmises', k=k,
             label=fr'von Mises (angle-only), $k={k:g}$')
        for k in ks
    ]
    title = (r'von Mises neural weight (angle warping only), high $k$' '\n'
             r'companion to hq_bifurc_plots/'
             r'neural_weight_sweep_vonmises_high_k.png')
    return rows, title, 'neural_weight_sweep_vonmises_high_k_angle_only.png'


# ---- model construction ----

def build_models(row_spec):
    """Return (PerceptionModel, NeuralBandModel) configured per row_spec
    with ``weight_angle_only=WEIGHT_ANGLE_ONLY``."""
    targets = model.Targets(locs=TARGET_LOCS, geom_name=None,
                            r=TARGET_RADIUS)
    percep = model.PerceptionModel(
        targets, FOCAL_LOC, FOCAL_ANGLE,
        neural_weight=row_spec['weight'], neural_angle='integral',
        weight_angle_only=WEIGHT_ANGLE_ONLY)
    if row_spec['weight'] == 'cutoff':
        percep.a = row_spec['a']
        percep.b = row_spec['b']
    elif row_spec['weight'] == 'vonmises':
        percep.k = row_spec['k']
    elif row_spec['weight'] == 'symmetric_beta':
        percep.alpha = row_spec['alpha']
        percep.b = row_spec['b']
    else:
        raise ValueError(f"unsupported weight: {row_spec['weight']!r}")
    nbm = model.NeuralBandModel(percep)
    return percep, nbm


# ---- cache fingerprint helpers ----

def row_param_signature(row_spec):
    """Subset of row_spec that affects the bifurcation result. 'label' is
    cosmetic and intentionally excluded."""
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
    else:
        raise ValueError(f"unsupported weight: {row_spec['weight']!r}")


def figure_fingerprint(rows):
    return dict(
        cache_version=CACHE_VERSION,
        weight_angle_only=bool(WEIGHT_ANGLE_ONLY),
        target_locs=TARGET_LOCS.tolist(),
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
             fingerprint_json=np.array(json.dumps(fingerprint,
                                                  sort_keys=True)),
             imgs=imgs,
             overflow_flags=np.asarray(overflow_flags, dtype=bool))
    print(f"  wrote bifurcation cache to {path}")


# ---- bifurcation rendering helpers ----

def _render_bifurcation_from_img(ax, img, targets, max_count, title):
    cmap = plt.get_cmap('viridis', max_count + 1)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, max_count + 1.5),
                        ncolors=max_count + 1)
    img = np.clip(img, 0, max_count)
    ax.imshow(img, origin='lower',
              extent=[XLIM[0], XLIM[1], YLIM[0], YLIM[1]],
              aspect='equal', interpolation='nearest',
              cmap=cmap, norm=norm)
    targets.plot_targets_to_axis(ax)

    for n in range(max_count + 1):
        ax.plot([], [], marker='s', markersize=10, linestyle='',
                color=cmap(norm(n)), label=f'{n}')

    if title is not None:
        ax.set_title(title)


def _compute_bifurcation_panel(nbm, ax, pool, title):
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

    img = np.asarray(ax.images[-1].get_array(), dtype=int)
    return img, overflowed


# ---- figure builder ----

def build_figure(rows, suptitle, out_name, pool, overflow_log,
                 regenerate=False):
    n = len(rows)
    fingerprint = figure_fingerprint(rows)

    cached = None if regenerate else try_load_cache(out_name, fingerprint)
    cached_imgs, cached_overflow = (None, None)
    if cached is not None:
        cached_imgs, cached_overflow = cached

    fig, axes = plt.subplots(n, 3, figsize=(16, 4.5*n),
                             squeeze=False)

    fresh_imgs = [None]*n
    fresh_overflow = [False]*n
    any_recomputed = (cached_imgs is None)

    for r, row_spec in enumerate(rows):
        percep, nbm = build_models(row_spec)

        ax_w, ax_g, ax_b = axes[r]

        percep.plot_neural_weight(ax=ax_w)
        ax_w.set_title(f'{row_spec["label"]}\nNeural weight & angle map',
                       fontsize=11)

        percep.plot_blocked_signals(ax=ax_g)
        ax_g.set_title(f'{row_spec["label"]}\nTarget geometry &'
                       ' neural directions', fontsize=11)

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

    if any_recomputed:
        imgs_arr = np.stack(fresh_imgs, axis=0).astype(np.int16)
        save_cache(out_name, fingerprint, imgs_arr, fresh_overflow)

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
        description=("Generate the four-figure exploratory neural-weight "
                     "parameter sweep with weight_angle_only=True. "
                     "Bifurcation rasters are cached per-figure so layout "
                     "tweaks don't trigger recomputation."))
    parser.add_argument('--regenerate', action='store_true',
                        help="ignore existing per-figure caches and "
                             "recompute every bifurcation panel")
    parser.add_argument('--only', type=int, choices=[1, 2, 3, 4],
                        action='append', default=None,
                        help="only build the listed figure(s) (1-4); may "
                             "be passed multiple times. Default: all.")
    parser.add_argument('--workers', type=int, default=DEFAULT_N_WORKERS,
                        help=f"size of the multiprocessing pool. Default: "
                             f"{DEFAULT_N_WORKERS}. Bump this up when "
                             f"running on a many-core machine.")
    return parser.parse_args()


def main():
    args = parse_args()
    all_specs = [fig1_spec(), fig2_spec(), fig3_spec(), fig4_spec()]
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
