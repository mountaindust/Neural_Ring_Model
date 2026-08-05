"""
Merged exploratory parameter sweep over the NeuralBandModel's perception
geometry. Supersedes the old ``neural_weight_sweep.py`` /
``neural_weight_sweep_angle_only.py`` pair: the two roles of the
PerceptionModel are now configured *independently* at the top of this file,
exactly as the model itself decouples them.

    WARP   (neural_angle_dist) -- the egocentric -> neural angle map
    WEIGHT (angle_weight)      -- the density integrated over each target's
                                  visible arc to set rho

Each role has a family selector and two parameter slots (``a``/``b``). Every
parameter slot accepts EITHER a scalar OR an iterable. The figure has one row
per element of the Cartesian product of all active (length>1 or not) slots --
i.e. *the number of rows is the number of parameter combinations*. So:

  * vary the warp ``a`` only (fix ``b``):  A_WARP = [..4 values..], B_WARP = pi
  * hold the warp constant, sweep the weight: A_WARP/B_WARP scalars, WEIGHT a
    family with A_WEIGHT = [..values..]
  * sweep both a and b of one role:         A_WARP and B_WARP both iterables
    -> len(A_WARP) * len(B_WARP) rows

The DEFAULT config reproduces the old ``*_angle_only`` behavior (warp-only,
uniform weight) but with two deliberate changes: the warp family is
``lin_cutoff`` (analytic trapezoidal cutoff) and the stability criterion is
``'reduced'`` (the timescale-separated default that matches the slaved walker).

Each row is an Nx3 panel matrix:

    col 1 -- PerceptionModel.plot_neural_weight (weight curve + angle map)
    col 2 -- PerceptionModel.plot_blocked_signals (target-geometry panel)
    col 3 -- NeuralBandModel.plot_bifurcation_diagram (# stable equilibria)

These plots are intentionally NOT publication-quality; they are sized for fast
iteration on a many-core machine.

Caching: the rasterized bifurcation ``img`` for every row is saved to
``_cache_<out_name>.npz`` next to the figure, alongside a JSON fingerprint of
every input that affects the result (geometry, grid, criterion, warp family,
weight selector, and the per-row a/b values). The cache is invalidated
automatically whenever any of those change; CACHE_VERSION is the manual
backstop for changes the fingerprint fields don't capture (npz layout, or a
correctness fix inside decision_model.py).

Bifurcation panels are pinned to ``max_count=3`` for color comparability; any
parameterization whose data exceeds that triggers a captured warning reported
in a post-run summary.
"""

import argparse
import itertools
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


# ========================= CONFIG (edit me) =========================

# ---- fixed scene (two-target GODM 'fly2' geometry: dist 5, +-30 deg) ----
TARGET_LOCS = np.array([[4.33, 2.5], [4.33, -2.5]])
TARGET_GEOM = 'circle'
TARGET_RADIUS = 0.5
FOCAL_LOC = (0, 0)
FOCAL_ANGLE = 0

# ---- WARP role: neural_angle_dist (egocentric -> neural angle map) ----
# Family: one of 'cutoff', 'lin_cutoff', 'vonmises', 'symmetric_beta',
# 'reg_power', 'direct_power', or None (identity / no warp).
# A_WARP/B_WARP: scalar OR iterable. Slot meaning is family-dependent
#   cutoff/lin_cutoff: a, b   vonmises: a=k (b unused)   beta: a=alpha, b
#   reg_power: a=d, b=e   direct_power: a=c (b unused)
# Leave a slot None to take the family default. B_WARP MUST be None for a
# single-parameter family (vonmises/direct_power).
WARP_FAMILY = 'lin_cutoff'
A_WARP = [0.0, np.pi/8, np.pi/4, np.pi/3]   # vary a ...
B_WARP = np.pi                              # ... b fixed

# ---- WEIGHT role: angle_weight (rho weighting) ----
# WEIGHT: None (uniform weight, the default), 'neural_angle_dist' (tie the
# weight to the warp family+params), or an independent family key (same set as
# the warp, EXCEPT 'direct_power' which is disallowed as a weight).
# A_WEIGHT/B_WEIGHT: scalar OR iterable; used ONLY when WEIGHT is a family key.
# They MUST be None when WEIGHT is None or 'neural_angle_dist'.
#
# Example -- hold the neural density constant, sweep the weight instead:
#   WARP_FAMILY='lin_cutoff'; A_WARP=np.pi/4; B_WARP=np.pi
#   WEIGHT='lin_cutoff'; A_WEIGHT=[0.1, 0.2, 0.3]; B_WEIGHT=0.8*np.pi
WEIGHT = None
A_WEIGHT = None
B_WEIGHT = None

# ---- bifurcation diagram settings (exploration-quality) ----
XLIM = (0.0, 6.0)
YLIM = (-3.5, 3.5)
NUM_X = 29
NUM_Y = 29
REFINEMENT_LEVELS = 2
MAX_COUNT = 3
STABILITY_CRITERION = 'reduced'   # 'reduced' (default) | 'coupled' | 'discrim_a'

# Output filename. None -> auto-generated from the config (so distinct sweeps
# don't collide). Override on the command line with --out.
OUT_NAME = None

# Bumped only when the cache layout or compute semantics change in a way the
# fingerprint fields below don't already capture. 1 = first merged-script
# version (fresh fingerprint schema + filenames; old per-figure caches from the
# two predecessor scripts never match and are simply ignored).
# 2 = the _get_target_signals wrapping-extent fix in decision_model.py: the
# closest target's angular extent is now unwrapped before the neural weight is
# integrated, so a target straddling the rear branch cut is no longer dropped
# from perception. Changes stable counts near the targets under uniform weight.
CACHE_VERSION = 2

DPI = 150
# Output image formats (extensions). The auto-generated base name is shared;
# one file is written per format. ('png',) at 150 dpi for fast exploration;
# use plots/neural_weight_sweep.py (300 dpi, 49x49 grid + 3 refinement levels)
# for publication-quality output.
OUTPUT_FORMATS = ('png',)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_N_WORKERS = get_n_workers()

# ---- font sizes ----
# The column-3 stable-count legend is fixed at LEGEND3_FS (see build_figure)
# and is intentionally the largest element; the main figure title matches it,
# and the rest sit between their old sizes and that.
LEGEND3_FS = 20       # column-3 stable-count legend (markers + text + title)
SUPTITLE_FS = 20      # main figure title (matches the col-3 legend)
TITLE_FS = 16         # subplot titles
LABEL_FS = 14         # axis labels
TICK_FS = 14          # tick labels
COL1_LEGEND_FS = 12   # column-1 legend

# ====================================================================


# ---- sweep-variable helpers ----

def _as_list(v):
    """Normalize a scalar-or-iterable sweep variable to a list.

    None -> [None]; a string or a bare scalar -> a single-element list;
    a list/tuple/ndarray -> its elements as a list.
    """
    if v is None:
        return [None]
    if isinstance(v, (str, bytes)):
        return [v]
    if isinstance(v, np.ndarray):
        return list(v.ravel())
    try:
        return list(v)
    except TypeError:
        return [v]


def _uses_b(family):
    """True if `family` has a second (b) parameter slot."""
    return (family is not None
            and model._FAMILY_INFO[family]['slots'][1] is not None)


# ---- config validation (fail fast with a clear message) ----

def validate_config():
    # WARP role
    if WARP_FAMILY is not None and WARP_FAMILY not in model._FAMILY_INFO:
        raise ValueError(
            f"WARP_FAMILY must be a family key {sorted(model._FAMILY_INFO)} "
            f"or None, got {WARP_FAMILY!r}.")
    if WARP_FAMILY is None:
        if A_WARP is not None or B_WARP is not None:
            raise ValueError(
                "A_WARP/B_WARP must be None when WARP_FAMILY is None "
                "(identity warp has no parameters).")
    elif not _uses_b(WARP_FAMILY) and B_WARP is not None:
        raise ValueError(
            f"B_WARP must be None: the {WARP_FAMILY!r} warp has a single "
            "parameter (set it via A_WARP).")

    # WEIGHT role
    if WEIGHT == 'direct_power':
        raise ValueError(
            "WEIGHT='direct_power' is disallowed (a signed angle map is not a "
            "density). Use 'reg_power' for the power-map-derivative density.")
    if WEIGHT in (None, 'neural_angle_dist'):
        if A_WEIGHT is not None or B_WEIGHT is not None:
            raise ValueError(
                "A_WEIGHT/B_WEIGHT must be None when WEIGHT is None (uniform) "
                "or 'neural_angle_dist' (tied to the warp). Sweep the warp "
                "parameters instead.")
        if WEIGHT == 'neural_angle_dist' and (WARP_FAMILY is None
                                              or WARP_FAMILY == 'direct_power'):
            raise ValueError(
                "WEIGHT='neural_angle_dist' requires WARP_FAMILY to be a "
                f"density family, got {WARP_FAMILY!r}.")
    elif WEIGHT in model._FAMILY_INFO:
        if not _uses_b(WEIGHT) and B_WEIGHT is not None:
            raise ValueError(
                f"B_WEIGHT must be None: the {WEIGHT!r} weight has a single "
                "parameter (set it via A_WEIGHT).")
    else:
        raise ValueError(
            f"WEIGHT must be None, 'neural_angle_dist', or a family key "
            f"{sorted(k for k in model._FAMILY_INFO if k != 'direct_power')}, "
            f"got {WEIGHT!r}.")

    if STABILITY_CRITERION not in ('reduced', 'coupled', 'discrim_a'):
        raise ValueError(
            "STABILITY_CRITERION must be 'reduced', 'coupled', or 'discrim_a', "
            f"got {STABILITY_CRITERION!r}.")


# ---- row expansion (Cartesian product of the active parameter axes) ----

def build_rows():
    """Expand the config into one row dict per parameter combination.

    Each row carries the four generic constructor slots (a_warp, b_warp,
    a_weight, b_weight); inactive slots are None. The number of rows is the
    product of the lengths of the active axes.
    """
    weight_is_family = WEIGHT not in (None, 'neural_angle_dist')

    wa = _as_list(A_WARP) if WARP_FAMILY is not None else [None]
    wb = _as_list(B_WARP) if _uses_b(WARP_FAMILY) else [None]
    va = _as_list(A_WEIGHT) if weight_is_family else [None]
    vb = _as_list(B_WEIGHT) if (weight_is_family and _uses_b(WEIGHT)) else [None]

    rows = [dict(a_warp=a_w, b_warp=b_w, a_weight=a_v, b_weight=b_v)
            for a_w, b_w, a_v, b_v in itertools.product(wa, wb, va, vb)]
    return rows


# ---- model construction ----

def build_models(row):
    """Return (PerceptionModel, NeuralBandModel) for a single row."""
    targets = model.Targets(locs=TARGET_LOCS, geom_name=TARGET_GEOM,
                            r=TARGET_RADIUS)
    percep = model.PerceptionModel(
        targets, FOCAL_LOC, FOCAL_ANGLE,
        neural_angle_dist=WARP_FAMILY, angle_weight=WEIGHT,
        a_warp=row['a_warp'], b_warp=row['b_warp'],
        a_weight=row['a_weight'], b_weight=row['b_weight'])
    return percep, model.NeuralBandModel(percep)


# ---- labels ----

# Human-readable family names for titles/legends (the raw keys stay in
# filenames and the config). Add new families here when they are added to
# the model's _FAMILY_INFO.
_FAMILY_DISPLAY = {
    'cutoff': 'smooth cutoff',
    'lin_cutoff': 'linear cutoff',
    'vonmises': 'von Mises',
    'symmetric_beta': 'symmetric beta',
    'reg_power': 'regularized power',
    'direct_power': 'power',
}


def _pretty(family):
    """Display name for a family key (e.g. 'lin_cutoff' -> 'linear cutoff').

    Returns None for None so callers can fall back to 'identity'/'uniform'.
    Unknown keys pass through unchanged.
    """
    if family is None:
        return None
    return _FAMILY_DISPLAY.get(family, family)


def _role_label(family, a_val, b_val, role):
    """'family (key=val, key=val)' with family-default fill-in for None slots."""
    if family is None:
        return 'identity' if role == 'warp' else 'uniform'
    info = model._FAMILY_INFO[family]
    a_key, b_key = info['slots']
    defs = info['defaults']
    a_show = a_val if a_val is not None else defs[a_key]
    parts = [f"{a_key}={a_show:.3g}"]
    if b_key is not None:
        b_show = b_val if b_val is not None else defs[b_key]
        parts.append(f"{b_key}={b_show:.3g}")
    return f"{_pretty(family)} ({', '.join(parts)})"


def row_label(row):
    warp = _role_label(WARP_FAMILY, row['a_warp'], row['b_warp'], 'warp')
    if WEIGHT is None:
        weight = 'uniform neural weight'
    elif WEIGHT == 'neural_angle_dist':
        weight = 'weight tied to warp'
    else:
        weight = 'weight: ' + _role_label(
            WEIGHT, row['a_weight'], row['b_weight'], 'weight')
    return f"{warp}  |  {weight}"


# ---- cache fingerprint ----

def row_signature(row):
    return {k: (None if row[k] is None else float(row[k]))
            for k in ('a_warp', 'b_warp', 'a_weight', 'b_weight')}


def figure_fingerprint(rows):
    return dict(
        cache_version=CACHE_VERSION,
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
        warp_family=WARP_FAMILY,
        weight=WEIGHT,
        rows=[row_signature(r) for r in rows],
    )


def default_out_name():
    warp_tag = WARP_FAMILY if WARP_FAMILY is not None else 'identity'
    if WEIGHT is None:
        weight_tag = 'uniform'
    elif WEIGHT == 'neural_angle_dist':
        weight_tag = 'tied'
    else:
        weight_tag = WEIGHT
    return f"neural_weight_sweep_{warp_tag}_warp_{weight_tag}_weight.png"


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


# ---- column-1 (perception) rendering ----

_TICK_LOCS = np.array([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0,
                       np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
_TICK_LABELS = [r'$-\pi$', r'$-\frac{3\pi}{4}$', r'$-\frac{\pi}{2}$',
                r'$-\frac{\pi}{4}$', r'$0$', r'$\frac{\pi}{4}$',
                r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']


def _warp_density(percep, theta):
    """Warp (neural) density on theta, normalized so max |value| = 1.

    This is the family density underlying the egocentric->neural angle map (it
    is proportional to the map's derivative). Evaluated analytically for the
    density families; for an identity warp (None) or the direct_power angle map
    (no closed-form density) it falls back to differentiating the angle map.
    """
    name = percep.warp_name
    p = percep.warp_params
    if name == 'cutoff':
        d = percep._smooth_cutoff(theta, p['a'], p['b'])
    elif name == 'lin_cutoff':
        d = percep._lin_cutoff(theta, p['a'], p['b'])
    elif name == 'vonmises':
        d = percep._vonmises(theta, p['k'])
    elif name == 'symmetric_beta':
        d = percep._symmetric_beta(theta, p['alpha'], p['b'])
    elif name == 'reg_power':
        d = percep._reg_power(theta, p['d'], p['e'])
    else:   # identity (None) or direct_power: density = d(angle map)/dtheta
        d = np.gradient(percep.get_neural_angle(theta), theta)
    d = np.asarray(d, dtype=float)
    mx = np.nanmax(np.abs(d))
    return d / mx if mx > 0 else d


def _show_density_for(rows):
    """Decide the column-1 content for the whole figure.

    Show the neural density + angle map whenever anything about the WARP is
    changing across rows (or nothing changes but the warp is non-identity);
    show the weighting function alone when only the WEIGHT varies.
    """
    warp_varies = len({(r['a_warp'], r['b_warp']) for r in rows}) > 1
    weight_varies = len({(r['a_weight'], r['b_weight']) for r in rows}) > 1
    return warp_varies or (not weight_varies and WARP_FAMILY is not None)


def _plot_col1(percep, ax, show_density):
    """Column-1 perception panel.

    show_density=True  -> solid neural-density curve + dashed angle map (twin).
    show_density=False -> the weighting function alone.
    """
    theta = np.linspace(-np.pi, np.pi, 361)
    ax.set_xticks(_TICK_LOCS)
    ax.set_xticklabels(_TICK_LABELS)
    ax.set_xlabel(r'$\theta$', fontsize=LABEL_FS)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='both', labelsize=TICK_FS)

    if show_density:
        dens = _warp_density(percep, theta)
        line, = ax.plot(
            theta, dens,
            label=f'{_pretty(percep.warp_name) or "identity"} neural density')
        ax.set_ylabel('Neural density (norm.)', fontsize=LABEL_FS)

        ax2 = ax.twinx()
        ax2.set_ylabel(r'$\tilde{\theta}$', fontsize=LABEL_FS)
        ax2.set_yticks(_TICK_LOCS)
        ax2.set_yticklabels(_TICK_LABELS)
        ax2.set_ylim(-np.pi, np.pi)
        ax2.tick_params(axis='both', labelsize=TICK_FS)
        ax2.plot(theta, percep.get_neural_angle(theta),
                 color=line.get_color(), linestyle='--',
                 label=f'{_pretty(percep.warp_name) or "identity"} angle map')

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='lower right', fontsize=COL1_LEGEND_FS)
    else:
        w = np.asarray(percep.get_neural_weight(theta), dtype=float)
        mx = np.nanmax(np.abs(w))
        if mx > 0:
            w = w / mx
        ax.plot(theta, w,
                label=f'{_pretty(percep.weight_name) or "uniform"} weight')
        ax.set_ylabel('Neural weight (norm.)', fontsize=LABEL_FS)
        ax.legend(loc='lower right', fontsize=COL1_LEGEND_FS)


# ---- figure builder ----

def build_figure(rows, out_name, pool, overflow_log, regenerate=False):
    n = len(rows)
    fingerprint = figure_fingerprint(rows)

    cached = None if regenerate else try_load_cache(out_name, fingerprint)
    cached_imgs, cached_overflow = (None, None)
    if cached is not None:
        cached_imgs, cached_overflow = cached

    fig, axes = plt.subplots(n, 3, figsize=(16, 4.5*n), squeeze=False)

    fresh_imgs = [None]*n
    fresh_overflow = [False]*n
    any_recomputed = (cached_imgs is None)

    show_density = _show_density_for(rows)
    col1_sub = 'Neural density & angle map' if show_density else 'Neural weight'

    for r, row in enumerate(rows):
        percep, nbm = build_models(row)
        label = row_label(row)
        ax_w, ax_g, ax_b = axes[r]

        _plot_col1(percep, ax_w, show_density)
        ax_w.set_title(f'{label}\n{col1_sub}', fontsize=TITLE_FS)

        percep.plot_blocked_signals(ax=ax_g)
        ax_g.set_title('Target geometry &\nneural directions', fontsize=TITLE_FS)
        ax_g.tick_params(axis='both', labelsize=TICK_FS)

        title = '# stable self-consistent equilibria'
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
                (label, f"data contained >{MAX_COUNT} stable equilibria; "
                        "clipped"))
            print(f"  [overflow] {label}: >{MAX_COUNT} stable equilibria "
                  "detected, clipped")

        ax_b.set_xlabel('observer x', fontsize=LABEL_FS)
        ax_b.set_ylabel('observer y', fontsize=LABEL_FS)
        ax_b.tick_params(axis='both', labelsize=TICK_FS)
        ax_b.title.set_fontsize(TITLE_FS)
        # The bifurcation panel is equal-aspect and portrait, so it would
        # otherwise center in its (wider) cell and leave a large gap after
        # column 2. Anchor it to the left/west edge of its cell to pull it
        # toward column 2 without moving columns 1 and 2.
        ax_b.set_anchor('W')

    if any_recomputed:
        imgs_arr = np.stack(fresh_imgs, axis=0).astype(np.int16)
        save_cache(out_name, fingerprint, imgs_arr, fresh_overflow)

    cmap = plt.get_cmap('viridis', MAX_COUNT + 1)
    handles = [plt.Line2D([], [], marker='s', markersize=20, linestyle='',
                          color=cmap(i / MAX_COUNT), label=f'{i}')
               for i in range(MAX_COUNT + 1)]
    # bbox x chosen to halve the col-3 -> legend gap; the figure is saved with
    # bbox_inches='tight', which re-crops the right edge so it stays tight.
    fig.legend(handles=handles, title='# stable\nequilibria',
               loc='center right', bbox_to_anchor=(0.976, 0.5), frameon=False,
               fontsize=LEGEND3_FS, title_fontsize=LEGEND3_FS)

    # Title reflects what the sweep varies. When only the weight varies it is
    # the neural weight function; when the warp varies it is the neural density
    # -- and also the weighting when the weight is tied to the warp
    # (angle_weight='neural_angle_dist'), since both then change together.
    if not show_density:
        sweep_kind = 'neural weight function'
    elif WEIGHT == 'neural_angle_dist':
        sweep_kind = 'neural density and weighting'
    else:
        sweep_kind = 'neural density'
    suptitle = f'Neural band bifurcation sweep: changes in {sweep_kind}'
    fig.suptitle(suptitle, fontsize=SUPTITLE_FS, y=0.995)
    # Reserve extra room on the right for the enlarged stable-count legend.
    fig.tight_layout(rect=[0, 0, 0.93, 0.985])

    base, _ = os.path.splitext(out_name)
    for fmt in OUTPUT_FORMATS:
        out_path = os.path.join(OUTPUT_DIR, f"{base}.{fmt}")
        save_kwargs = dict(dpi=DPI, bbox_inches='tight')
        if fmt in ('tif', 'tiff'):
            save_kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(out_path, **save_kwargs)
        print(f"Wrote {out_path}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Generate the merged neural-weight parameter sweep. The "
                     "warp and weight roles are configured independently at "
                     "the top of the script; each parameter slot accepts a "
                     "scalar or an iterable, and the number of rows is the "
                     "number of parameter combinations. The bifurcation "
                     "raster is cached so layout tweaks don't recompute."))
    parser.add_argument('--regenerate', action='store_true',
                        help="ignore the existing cache and recompute every "
                             "bifurcation panel")
    parser.add_argument('--out', type=str, default=None,
                        help="output filename (default: auto-generated from "
                             "the config)")
    parser.add_argument('--workers', type=int, default=DEFAULT_N_WORKERS,
                        help=f"size of the multiprocessing pool. Default: "
                             f"{DEFAULT_N_WORKERS}.")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_config()
    rows = build_rows()
    out_name = args.out or OUT_NAME or default_out_name()

    print(f"Configuration: warp={WARP_FAMILY}, weight={WEIGHT}, "
          f"criterion={STABILITY_CRITERION}")
    print(f"Sweep expands to {len(rows)} row(s) -> {out_name}")
    if len(rows) > 12:
        print(f"  note: {len(rows)} rows makes a very tall figure "
              f"({4.5*len(rows):.0f} in); consider narrowing the sweep.")
    print(f"Using {args.workers} worker(s) for bifurcation evaluation.")

    overflow_log = []
    with Pool(args.workers) as pool:
        build_figure(rows, out_name, pool, overflow_log,
                     regenerate=args.regenerate)

    print("\n=== Summary ===")
    if overflow_log:
        print(f"{len(overflow_log)} row(s) exceeded max_count={MAX_COUNT}; "
              "follow up on these:")
        for label, msg in overflow_log:
            print(f"  - {label}\n      ({msg})")
    else:
        print(f"No row exceeded max_count={MAX_COUNT}.")


if __name__ == "__main__":
    main()
