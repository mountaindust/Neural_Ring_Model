"""Can an anti-foveal (centre-dip) neural weight bias the observer OUTWARD?

Motivation
----------
The GODM three-target data (Sridhar et al. 2021) splits very differently by
species: flies commit ~45% to the centre target (roughly the 1:2:1 a two-stage
binary cascade would give), locusts only ~29% (strongly outer-biased). The
model is intrinsically centre-biased -- a walker that peels onto a compromise
arm is recaptured by the *reborn centre branch* before the outer-target branch
is born -- so the fly matches and the locust does not (see
../walker_analysis/three_target_findings.md). The published locust model needed
an external, hand-added outward bias.

This module tests a mechanism internal to *our* perception model: keep the
foveal neural DENSITY that sets the warp, but give the WEIGHT (the density
integrated over each target's visible arc to set rho) a **dip in the middle**
instead of a bump, so that whatever sits dead ahead is under-weighted. Two
anti-foveal shapes are compared against the two incumbents, a 2x2 design that
separates the two things such a shape can do:

                     | full-weight periphery | rear falloff
    ---------------- | --------------------- | ---------------------
    no frontal dip   | uniform (None)        | 'lin_cutoff' (foveal)
    frontal dip      | 'lin_dip'             | 'lin_ring'

Geometry is the empirical locust3 layout (3 circle targets, r = 0.1, at
distance 3, bearings {0, +-35 deg}), with the shipped locust warp/K/T from
../plots/decision_skeleton.py.

Stages (run one, several, or `all`)
-----------------------------------
    mechanism  rho at the two competing commitments -> outward_bias_mechanism.png
    cascade    midline (x, theta) branch diagrams   -> outward_bias_cascade.png
    events     bifurcation-event x vs dip depth     -> outward_bias_events.png
    raster     (x, y) stable-count diagrams         -> outward_bias_rasters.png
    walkers    endpoint census + tracks             -> outward_bias_walkers.png

Every stage caches its numeric result next to the figure in
``_cache_outward_<stage>.npz`` behind a JSON fingerprint of the inputs, so
re-running to tweak a plot is free. ``--regenerate`` forces recomputation.

NOTE ``'lin_dip'`` and ``'lin_ring'`` are NOT part of `PerceptionModel` -- they
were removed once this analysis came back negative. [anti_foveal.py](anti_foveal.py)
preserves them and re-registers them at import; that is the only reason this
script still runs.

Run:
    python weighting_analysis/outward_bias.py all
    python weighting_analysis/outward_bias.py cascade events --regenerate
"""

import argparse
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anti_foveal

# The two anti-foveal families were REMOVED from decision_model.py once this
# analysis came back negative; anti_foveal.py preserves them and re-registers
# them onto PerceptionModel. This must run at module level, not under
# `if __name__ == '__main__'`: multiprocessing workers re-import the main
# module on spawn and would otherwise not know the family names.
anti_foveal.register()

pi = np.pi
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- geometry --
# Empirical locust3: 3 targets at distance 3, bearings {0, +-35 deg}, r = 0.1.
_SEP = np.radians(35.0)
TARGET_LOCS = np.array([[3.0, 0.0],
                        [3.0*np.cos(_SEP),  3.0*np.sin(_SEP)],
                        [3.0*np.cos(_SEP), -3.0*np.sin(_SEP)]])
TARGET_R = 0.1

# Shipped locust knobs (plots/decision_skeleton.py LOCUST). The WARP is held
# fixed across every case here -- only the WEIGHT changes, which is the whole
# point of the warp/weight decouple.
A_WARP, B_WARP = 0.50*pi, 0.90*pi
K, T = 6.0, 0.10

XLIM, YLIM = (0.0, 3.4), (-2.4, 2.4)
CRITERION = 'reduced'

# ------------------------------------------------------------------- cases --
# (key, label, angle_weight, a_weight, b_weight, colour)
CASES = [
    ('uniform',  'uniform\n(model default)',       None,         None,      None,   'tab:gray'),
    ('foveal',   'foveal lin_cutoff\n(shipped locust)', 'lin_cutoff', 0.10*pi, 0.80*pi, 'tab:blue'),
    ('dip',      'lin_dip  m=0.25, b=$\\pi/2$\n(dip, full-weight periphery)',
     'lin_dip',  0.25, pi/2,  'tab:red'),
    ('ring',     'lin_ring  m=0.25, p=$\\pi/2$\n(dip + rear falloff)',
     'lin_ring', 0.25, pi/2,  'tab:orange'),
]
CASE_BY_KEY = {c[0]: c for c in CASES}

# Dip depths swept in the `events` stage (m = 1 is uniform).
M_SWEEP = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1])

N_WORKERS = get_n_workers()


def build_model(weight, a_weight, b_weight):
    targets = model.Targets(locs=TARGET_LOCS, geom_name='circle', r=TARGET_R)
    pm = model.PerceptionModel(targets, (0, 0), 0,
                               neural_angle_dist='lin_cutoff', angle_weight=weight,
                               a_warp=A_WARP, b_warp=B_WARP,
                               a_weight=a_weight, b_weight=b_weight)
    return model.NeuralBandModel(pm, T=T, K=K)


# ------------------------------------------------------------------- cache --

def _cache_path(stage):
    return os.path.join(HERE, f'_cache_outward_{stage}.npz')


def load_cache(stage, fingerprint, regenerate=False):
    if regenerate:
        return None
    path = _cache_path(stage)
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=False)
        if json.loads(str(d['fingerprint_json'])) != fingerprint:
            print(f'  [{stage}] fingerprint changed; recomputing')
            return None
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
        print(f'  [{stage}] cache unreadable ({e}); recomputing')
        return None
    print(f'  [{stage}] loaded cache')
    return d


def save_cache(stage, fingerprint, **arrays):
    np.savez(_cache_path(stage),
             fingerprint_json=np.array(json.dumps(fingerprint, sort_keys=True)),
             **arrays)
    print(f'  [{stage}] wrote cache')


def _base_fp():
    return dict(locs=TARGET_LOCS.tolist(), r=TARGET_R, a_warp=A_WARP,
                b_warp=B_WARP, K=K, T=T, criterion=CRITERION,
                cases=[[c[0]] + [None if v is None else v for v in c[2:5]]
                       for c in CASES])


# ----------------------------------------------------- pooled midline scan --
# One model per worker process, rebuilt when the case changes.

_NM = None


def _init_worker(cfg):
    global _NM
    _NM = build_model(*cfg)


def _scan_one_x(x):
    """Self-consistent equilibria on the midline at observer (x, 0).

    Returns a fixed-width row so the results stack into an array: up to
    MAX_EQ (theta, R, stable) triples, NaN-padded.
    """
    angles, Rs, stab = _NM.sc_equilib((x, 0.0), CRITERION, return_R=True)
    row = np.full((MAX_EQ, 3), np.nan)
    for i, (a, r, s) in enumerate(zip(angles, Rs, stab)):
        if i >= MAX_EQ:
            break
        row[i] = (a, r, float(s))
    return row


MAX_EQ = 12


def midline_scan(cfg, xs, pool_size=None):
    with Pool(pool_size or N_WORKERS, initializer=_init_worker,
              initargs=(cfg,)) as p:
        rows = p.map(_scan_one_x, xs)
    return np.stack(rows)          # (n_x, MAX_EQ, 3)


# ------------------------------------------------ branch classification -----
# Along the midline the SC headings fall into three geometric classes. The
# outer targets sit at +-35 deg from the origin but their bearing grows without
# bound as the observer advances (90 deg at x = 3), so the class boundaries are
# angular bands, not fixed values. CENTRE_TOL is generous enough to absorb the
# solver's few-tenths-of-a-degree jitter on the exactly-0 branch.
CENTRE_TOL = 5.0        # deg: |theta| < 5 -> the centre-target branch
ARM_LO, ARM_HI = 8.0, 50.0    # deg: the two-target compromise arms
OUTER_LO, OUTER_HI = 50.0, 175.0   # deg: single outer-target commitment


def classify(rows):
    """rows -> (centre, arms, outer) boolean masks over x, stable branches only."""
    th = np.degrees(rows[:, :, 0])
    stable = rows[:, :, 2] == 1.0
    a = np.abs(th)
    centre = np.any(stable & (a < CENTRE_TOL), axis=1)
    arms = np.any(stable & (a > ARM_LO) & (a < ARM_HI), axis=1)
    outer = np.any(stable & (a >= OUTER_LO) & (a < OUTER_HI), axis=1)
    return centre, arms, outer


def cascade_events(rows, xs):
    """Locate the four cascade events on a midline scan.

    Returns dict with arm_birth, centre_lost, centre_reborn, outer_birth
    (np.nan where the event does not occur in the scanned window).
    """
    centre, arms, outer = classify(rows)
    ev = dict(arm_birth=np.nan, centre_lost=np.nan, centre_reborn=np.nan,
              outer_birth=np.nan)
    if arms.any():
        ev['arm_birth'] = xs[np.argmax(arms)]
    if outer.any():
        ev['outer_birth'] = xs[np.argmax(outer)]
    for i in range(1, len(xs)):
        if centre[i-1] and not centre[i] and np.isnan(ev['centre_lost']):
            ev['centre_lost'] = xs[i]
        if (not np.isnan(ev['centre_lost']) and not centre[i-1] and centre[i]
                and np.isnan(ev['centre_reborn'])):
            ev['centre_reborn'] = xs[i]
    ev['no_centre_frac'] = float((~centre).mean())
    return ev


# =====================================================================
# Stage: mechanism
# =====================================================================

MECH_X = [1.8, 2.0, 2.2]      # observer positions inside the second-bifurcation zone


def stage_mechanism(regenerate=False):
    print('[mechanism]')
    fp = dict(_base_fp(), stage='mechanism', xs=MECH_X)
    cached = load_cache('mechanism', fp, regenerate)
    if cached is None:
        rho_c = np.full((len(CASES), len(MECH_X), 3), np.nan)   # facing centre
        rho_o = np.full((len(CASES), len(MECH_X), 3), np.nan)   # facing upper outer
        for ci, (key, _lab, w, aw, bw, _col) in enumerate(CASES):
            nm = build_model(w, aw, bw)
            pm = nm.percep_model
            for xi, x in enumerate(MECH_X):
                o = np.array([x, 0.0])
                bear = np.arctan2(TARGET_LOCS[:, 1] - o[1],
                                  TARGET_LOCS[:, 0] - o[0])
                for arr, heading in ((rho_c, 0.0), (rho_o, bear[1])):
                    _assign_rho(arr[ci, xi], pm, heading, o, bear)
        save_cache('mechanism', fp, rho_c=rho_c, rho_o=rho_o)
    else:
        rho_c, rho_o = cached['rho_c'], cached['rho_o']
    _plot_mechanism(rho_c, rho_o)


def _assign_rho(out, pm, heading, o, bear):
    """Fill out[3] with each target's rho by TARGET INDEX; 0 where invisible.

    `_get_target_signals` returns `(c_angles, rho)` for the visible targets
    only, `c_angles` being their EGOCENTRIC centre angles (it undoes its
    internal distance sort before filtering, so the surviving entries are in
    original target order -- but the filtering means position in the returned
    array is NOT the target index). Match on the egocentric angle, which is
    exact: `bear` holds each target's allocentric bearing from `o`.
    """
    out[:] = 0.0
    ang, rho = pm._get_target_signals(focal_angle=heading, focal_loc=o)
    if len(rho) == 0:
        return
    ego = model.convert_angles(np.asarray(bear) - heading)
    for a, val in zip(np.asarray(ang), rho):
        d = np.abs(model.convert_angles(ego - a))
        slot = int(np.argmin(d))
        assert d[slot] < 1e-8, (
            f'could not match returned signal at ego {a} to a target '
            f'(closest is {d[slot]} away)')
        out[slot] = val


def _plot_mechanism(rho_c, rho_o):
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.28)

    # --- top-left: the four weight shapes ---
    ax = fig.add_subplot(gs[0, :2])
    th = np.linspace(-pi, pi, 721)
    # lin_dip(m, b) and lin_ring(m, p) are identical for |theta| <= b = p, and
    # uniform coincides with lin_dip's peripheral plateau, so three of the four
    # curves overlap somewhere. Distinct dashes keep every one visible.
    styles = {'uniform': (3.4, (1, 1.6)), 'foveal': (2.4, None),
              'dip': (3.0, (6, 2.2)), 'ring': (2.0, None)}
    for key, lab, w, aw, bw, col in CASES:
        pm = build_model(w, aw, bw).percep_model
        wv = np.asarray(pm.get_neural_weight(th), dtype=float)
        mx = np.nanmax(np.abs(wv))
        if mx > 0:
            wv = wv / mx
        lw, dashes = styles[key]
        line, = ax.plot(np.degrees(th), wv, color=col, lw=lw,
                        label=lab.replace('\n', '  '))
        if dashes:
            line.set_dashes(dashes)
    ax.axvline(0, color='0.85', lw=0.8, zorder=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_xlabel('egocentric angle (deg)')
    ax.set_ylabel('neural weight (normalised)')
    ax.set_title('(a)  The four weighting shapes compared', loc='left')
    ax.legend(fontsize=8.5, loc='lower center', ncol=2, framealpha=0.95)
    ax.grid(alpha=0.25)

    # --- top-right: the geometry of the two competing commitments ---
    ax = fig.add_subplot(gs[0, 2])
    o = np.array([MECH_X[1], 0.0])
    bear = np.arctan2(TARGET_LOCS[:, 1]-o[1], TARGET_LOCS[:, 0]-o[0])
    for i in range(3):
        ax.annotate('', xy=TARGET_LOCS[i], xytext=o,
                    arrowprops=dict(arrowstyle='-', color='0.78', lw=0.9))
    ax.plot(TARGET_LOCS[:, 0], TARGET_LOCS[:, 1], 'k*', ms=14, zorder=4)
    for i, (dx, dy) in enumerate([(9, -14), (-4, 10), (-4, -20)]):
        ax.annotate(['centre', 'upper', 'lower'][i], TARGET_LOCS[i],
                    textcoords='offset points', xytext=(dx, dy), fontsize=8.5)
    ax.plot(*o, 'o', color='0.25', ms=7, zorder=4)
    for hd, c in ((0.0, 'tab:green'), (bear[1], 'tab:purple')):
        ax.annotate('', xy=o + 0.85*np.array([np.cos(hd), np.sin(hd)]), xytext=o,
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=2.4,
                                    mutation_scale=16))
    # egocentric angle of the rival target under each candidate heading
    ax.text(0.03, 0.13, f'rival at ego {np.degrees(bear[1]):.0f}°',
            color='tab:green', fontsize=8, transform=ax.transAxes)
    ax.text(0.03, 0.05, f'rival at ego {np.degrees(bear[0]-bear[1]):.0f}°',
            color='tab:purple', fontsize=8, transform=ax.transAxes)
    ax.set_aspect('equal')
    ax.set_xlim(1.5, 3.5)
    ax.set_ylim(-2.15, 2.35)
    ax.set_title(f'(b)  Observer at ({o[0]:.1f}, 0):\n'
                 'two candidate commitments', loc='left', fontsize=10)
    ax.set_xlabel('x')

    # --- bottom: rho bars, facing centre vs facing outer ---
    xi = 1                                       # the x = 2.0 slice
    labels = ['centre', 'upper', 'lower']
    for col, (arr, ttl, sub) in enumerate((
            (rho_c, '(c)  Heading = CENTRE target',
             'the centre branch must hold itself'),
            (rho_o, '(d)  Heading = UPPER OUTER target',
             'the outer branch must hold itself'))):
        ax = fig.add_subplot(gs[1, col])
        width = 0.2
        idx = np.arange(3)
        for ci, (key, lab, *_rest) in enumerate(CASES):
            col_c = CASES[ci][5]
            ax.bar(idx + (ci - 1.5)*width, arr[ci, xi], width,
                   color=col_c, label=lab.split('\n')[0], edgecolor='none')
        faced = 0 if col == 0 else 1
        ax.axhline(1/3, color='0.6', lw=0.8, ls=':')
        ax.set_xticks(idx)
        ax.set_xticklabels([f'{l}\n(FACING)' if i == faced else l
                            for i, l in enumerate(labels)])
        ax.set_ylabel(r'perceptual mass  $\rho$')
        ax.set_ylim(0, 0.72)
        ax.set_title(f'{ttl}\n{sub}', loc='left', fontsize=10)
        ax.grid(axis='y', alpha=0.25)
        if col == 0:
            ax.legend(fontsize=7.5, ncol=2, loc='upper right')

    # --- bottom-right: the summary ratio ---
    ax = fig.add_subplot(gs[1, 2])
    for ci, (key, lab, *_rest) in enumerate(CASES):
        colr = CASES[ci][5]
        faced_c = rho_c[ci, :, 0]                # facing centre -> centre faced
        faced_o = rho_o[ci, :, 1]                # facing outer  -> upper faced
        ax.plot(MECH_X, faced_c, 'o-', color=colr, lw=1.8, ms=5)
        ax.plot(MECH_X, faced_o, 's--', color=colr, lw=1.8, ms=5, mfc='none')
    ax.plot([], [], 'ko-', label=r'$\rho$ of the faced CENTRE target')
    ax.plot([], [], 'ks--', mfc='none', label=r'$\rho$ of the faced OUTER target')
    ax.set_xlabel('observer x (on the midline)')
    ax.set_ylabel(r'$\rho$ of the target being faced')
    ax.set_title('(e)  A dip punishes both commitments', loc='left', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle('Why a centre-dip weight cannot bias the model outward: '
                 'it suppresses whatever the observer faces',
                 fontsize=13, y=0.985)
    out = os.path.join(HERE, 'outward_bias_mechanism.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', out)


# =====================================================================
# Stage: cascade
# =====================================================================

# Stop the sweep just short of the centre target (at x = 3.0, r = 0.1): once the
# observer reaches and passes it the SC set is post-decision and picks up the
# facing-away theta = +-180 branch, which is not an outer-target commitment.
CASCADE_NX = 240
CASCADE_XS = np.linspace(0.05, 2.85, CASCADE_NX)


def stage_cascade(regenerate=False):
    print('[cascade]')
    fp = dict(_base_fp(), stage='cascade', nx=CASCADE_NX,
              xlim=[float(CASCADE_XS[0]), float(CASCADE_XS[-1])])
    cached = load_cache('cascade', fp, regenerate)
    if cached is None:
        rows = []
        for key, _lab, w, aw, bw, _c in CASES:
            print(f'  scanning {key} ...')
            rows.append(midline_scan((w, aw, bw), CASCADE_XS))
        rows = np.stack(rows)
        save_cache('cascade', fp, rows=rows, xs=CASCADE_XS)
    else:
        # xs is stored for provenance only; the fingerprint already pins it to
        # CASCADE_XS, which _plot_cascade uses directly.
        rows = cached['rows']
    _plot_cascade(rows)
    return rows


def _plot_cascade(rows):
    n = len(CASES)
    fig, axes = plt.subplots(2, n, figsize=(4.1*n, 7.4), squeeze=False,
                             sharex=True, sharey='row',
                             gridspec_kw=dict(height_ratios=[2.3, 1.0]))
    for ci, (key, lab, *_r) in enumerate(CASES):
        r = rows[ci]
        ax = axes[0, ci]
        th = np.degrees(r[:, :, 0])
        stable = r[:, :, 2] == 1.0
        X = np.repeat(CASCADE_XS[:, None], r.shape[1], axis=1)
        good = ~np.isnan(th)
        ax.scatter(X[good & ~stable], th[good & ~stable], s=5,
                   facecolors='none', edgecolors='tab:red', lw=0.45, zorder=2)
        ax.scatter(X[good & stable], th[good & stable], s=6,
                   color='tab:blue', zorder=3)

        ev = cascade_events(r, CASCADE_XS)
        marks = [('arm_birth', 'arms born', 'tab:green'),
                 ('centre_lost', 'centre unstable', 'tab:orange'),
                 ('centre_reborn', 'centre reborn', 'tab:purple'),
                 ('outer_birth', 'outer born', 'k')]
        for kk, mlab, mc in marks:
            if not np.isnan(ev[kk]):
                ax.axvline(ev[kk], color=mc, lw=1.1, ls='--', alpha=0.8, zorder=1)
        ax.set_ylim(-185, 185)
        ax.set_yticks([-180, -90, -45, 0, 45, 90, 180])
        ax.set_title(lab, fontsize=9.5)
        ax.grid(alpha=0.22)
        if ci == 0:
            ax.set_ylabel(r'SC equilibrium heading $\theta$ (deg)')

        # shade the centre-unstable window
        centre, arms, outer = classify(r)
        ax.fill_between(CASCADE_XS, -185, 185, where=~centre,
                        color='tab:orange', alpha=0.08, zorder=0)

        # branch-presence ribbon
        axr = axes[1, ci]
        for j, (mask, mlab, mc) in enumerate((
                (centre, 'centre branch', 'tab:green'),
                (arms, 'compromise arms', 'tab:blue'),
                (outer, 'outer branches', 'tab:red'))):
            axr.fill_between(CASCADE_XS, j, j + 0.8, where=mask,
                             color=mc, alpha=0.75, step='mid')
        axr.set_ylim(-0.15, 3.6)
        axr.set_xlabel('observer x (on the midline)')
        axr.grid(axis='x', alpha=0.22)
        txt = ('outer branch NEVER born' if not outer.any()
               else f"outer born at x = {ev['outer_birth']:.2f}")
        axr.text(0.03, 0.97, txt, transform=axr.transAxes, va='top',
                 fontsize=9, color=('tab:red' if not outer.any() else '0.25'),
                 fontweight=('bold' if not outer.any() else 'normal'))

    axes[0, 0].scatter([], [], s=6, color='tab:blue', label='stable')
    axes[0, 0].scatter([], [], s=6, facecolors='none', edgecolors='tab:red',
                       label='unstable')
    for _kk, mlab, mc in (('a', 'arms born', 'tab:green'),
                          ('b', 'centre destabilises', 'tab:orange'),
                          ('c', 'centre reborn', 'tab:purple'),
                          ('d', 'outer born', 'k')):
        axes[0, 0].plot([], [], ls='--', lw=1.1, color=mc, label=mlab)
    axes[0, 0].plot([], [], lw=6, color='tab:orange', alpha=0.18,
                    label='no centre branch')
    axes[0, 0].legend(fontsize=7.6, loc='lower left', framealpha=0.95, ncol=2)
    # The ribbon row shares its y axis, so label it once (a per-column
    # set_yticklabels would be overwritten by the last column's empty list).
    axes[1, 0].set_yticks([0.4, 1.4, 2.4])
    axes[1, 0].set_yticklabels(['centre', 'arms', 'outer'])
    axes[1, 0].set_ylabel('stable branch\npresent at this x', fontsize=9)
    for ci in range(1, n):
        axes[1, ci].tick_params(labelleft=False)
    fig.suptitle('Midline bifurcation cascade under four weightings '
                 '(locust 3-target geometry; warp fixed)', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(HERE, 'outward_bias_cascade.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', out)


# =====================================================================
# Stage: events
# =====================================================================

EVENT_NX = 160
EVENT_XS = np.linspace(0.05, 2.85, EVENT_NX)
EVENT_FAMILIES = [('lin_dip', pi/2, 'lin_dip  (dip, full-weight periphery)', 'tab:red'),
                  ('lin_ring', pi/2, 'lin_ring  (dip + rear falloff)', 'tab:orange')]

# Control: at a fixed depth, does the dip's WIDTH matter? (It does not -- depth
# is the whole lever. Reported in the console summary, not plotted.)
WIDTH_SWEEP_M = 0.4
WIDTH_SWEEP_B = [0.15*pi, 0.25*pi, 0.5*pi, 0.75*pi, pi]


def stage_events(regenerate=False):
    print('[events]')
    fp = dict(_base_fp(), stage='events', nx=EVENT_NX,
              ms=M_SWEEP.tolist(),
              fams=[[f[0], f[1]] for f in EVENT_FAMILIES],
              width_m=WIDTH_SWEEP_M, width_b=list(WIDTH_SWEEP_B))
    cached = load_cache('events', fp, regenerate)
    keys = ['arm_birth', 'centre_lost', 'centre_reborn', 'outer_birth',
            'no_centre_frac']
    if cached is None:
        table = np.full((len(EVENT_FAMILIES), len(M_SWEEP), len(keys)), np.nan)
        for fi, (fam, second, _lab, _c) in enumerate(EVENT_FAMILIES):
            for mi, m in enumerate(M_SWEEP):
                r = midline_scan((fam, float(m), second), EVENT_XS)
                ev = cascade_events(r, EVENT_XS)
                table[fi, mi] = [ev[k] for k in keys]
                print(f'  {fam} m={m:.2f}: '
                      + ' '.join(f'{k}={ev[k]:.2f}' if not np.isnan(ev[k])
                                 else f'{k}=--' for k in keys[:4]))
        # baselines
        base = {}
        for key in ('uniform', 'foveal'):
            c = CASE_BY_KEY[key]
            ev = cascade_events(midline_scan((c[2], c[3], c[4]), EVENT_XS),
                                EVENT_XS)
            base[key] = [ev[k] for k in keys]
        base_arr = np.array([base['uniform'], base['foveal']])
        width = np.full((len(WIDTH_SWEEP_B), len(keys)), np.nan)
        for bi, b in enumerate(WIDTH_SWEEP_B):
            ev = cascade_events(
                midline_scan(('lin_dip', WIDTH_SWEEP_M, float(b)), EVENT_XS),
                EVENT_XS)
            width[bi] = [ev[k] for k in keys]
        save_cache('events', fp, table=table, base=base_arr, ms=M_SWEEP,
                   width=width)
    else:
        table, base_arr, width = cached['table'], cached['base'], cached['width']
    _plot_events(table, base_arr, keys)

    ki = {k: i for i, k in enumerate(keys)}
    print(f'\n  width control -- lin_dip at fixed m = {WIDTH_SWEEP_M}:')
    for b, row in zip(WIDTH_SWEEP_B, width):
        ob = row[ki['outer_birth']]
        print(f'    b = {b/pi:.2f}pi:  centre reborn at '
              f"{row[ki['centre_reborn']]:.2f}, outer born "
              f"{'--never--' if np.isnan(ob) else f'{ob:.2f}'}")
    print('    (depth m is the lever; the dip width barely moves anything)')
    return table


def _plot_events(table, base_arr, keys):
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    ki = {k: i for i, k in enumerate(keys)}

    # --- panels 1-2: event x vs dip depth, one per family ---
    for fi, (fam, _second, lab, _c) in enumerate(EVENT_FAMILIES):
        ax = axes[fi]
        for kk, mlab, mc, mk in (
                ('arm_birth', 'arms born', 'tab:green', 'o'),
                ('centre_lost', 'centre destabilises', 'tab:orange', 's'),
                ('centre_reborn', 'centre REBORN', 'tab:purple', '^'),
                ('outer_birth', 'outer branch born', 'k', 'D')):
            y = table[fi, :, ki[kk]]
            ax.plot(M_SWEEP, y, mk + '-', color=mc, label=mlab, lw=1.7, ms=5)
        # shade where the outer branch is extinct
        extinct = np.isnan(table[fi, :, ki['outer_birth']])
        if extinct.any():
            m_hi = M_SWEEP[extinct].max()
            ax.axvspan(M_SWEEP.min() - 0.03, m_hi + 0.025, color='tab:red',
                       alpha=0.10, zorder=0)
            ax.text(0.5*(m_hi + M_SWEEP.min()), 2.72, 'outer branch\nextinct',
                    color='tab:red', fontsize=9.5, ha='center', va='bottom',
                    fontweight='bold')
        for bi, (blab, bcol) in enumerate((('uniform', 'tab:gray'),
                                           ('foveal (shipped)', 'tab:blue'))):
            ax.axhline(base_arr[bi, ki['centre_reborn']], color=bcol, lw=1.0,
                       ls=':', alpha=0.9)
        ax.invert_xaxis()                       # deeper dip to the right
        ax.set_xlabel('central floor  $m$   (1 = uniform, deeper dip →)')
        ax.set_ylabel('observer x of the event')
        ax.set_ylim(0.5, 3.25)
        ax.set_title(lab, fontsize=10.5, loc='left')
        ax.grid(alpha=0.25)
        if fi == 0:
            ax.legend(fontsize=8.2, loc='upper left')

    # --- panel 3: the ordering that matters ---
    ax = axes[2]
    for fi, (fam, _s, lab, col) in enumerate(EVENT_FAMILIES):
        gap = table[fi, :, ki['outer_birth']] - table[fi, :, ki['centre_reborn']]
        ax.plot(M_SWEEP, gap, 'o-', color=col, lw=1.9, ms=5,
                label=lab.split('  ')[0])
        extinct = np.isnan(table[fi, :, ki['outer_birth']])
        if extinct.any():
            ax.plot(M_SWEEP[extinct], np.full(extinct.sum(), -0.03 - 0.02*fi),
                    'x', color=col, ms=8, mew=2)
    for bi, (blab, bcol) in enumerate((('uniform', 'tab:gray'),
                                       ('foveal (shipped)', 'tab:blue'))):
        gap = base_arr[bi, ki['outer_birth']] - base_arr[bi, ki['centre_reborn']]
        ax.axhline(gap, color=bcol, lw=1.4, ls='--', label=blab)
    ax.axhline(0.0, color='k', lw=1.0)
    ax.set_ylim(-0.09, 0.42)
    ax.text(0.02, 0.20, 'below 0 = outer branch wins the race\n'
                        '(what an outer bias would need)',
            transform=ax.transAxes, ha='left', fontsize=8.5, color='0.3')
    ax.text(0.02, 0.06, '×  = outer branch never born at all',
            transform=ax.transAxes, fontsize=8.5, color='0.3')
    ax.invert_xaxis()
    ax.set_xlabel('central floor  $m$   (deeper dip →)')
    ax.set_ylabel(r'$x_{\rm outer\ born} - x_{\rm centre\ reborn}$')
    ax.set_title('The dip moves the race the WRONG way', fontsize=10.5, loc='left')
    ax.legend(fontsize=8.2, loc='upper left')
    ax.grid(alpha=0.25)

    fig.suptitle('Bifurcation-cascade events vs. dip depth '
                 '(locust 3-target geometry, warp fixed)', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, 'outward_bias_events.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', out)


# =====================================================================
# Stage: raster
# =====================================================================

NUM_X, NUM_Y = 45, 61
REFINEMENT = 2
# 5, not 3: this geometry genuinely reaches 5 stable SC equilibria in the
# five-branch window of the cascade (centre + two arms + two outer), and
# clipping at 3 would hide exactly the multistability the comparison is about.
MAX_COUNT = 5


def stage_raster(regenerate=False):
    print('[raster]')
    fp = dict(_base_fp(), stage='raster', num_x=NUM_X, num_y=NUM_Y,
              refinement=REFINEMENT, xlim=list(XLIM), ylim=list(YLIM),
              max_count=MAX_COUNT)
    cached = load_cache('raster', fp, regenerate)
    if cached is None:
        imgs = []
        with Pool(N_WORKERS) as pool:
            for key, _lab, w, aw, bw, _c in CASES:
                print(f'  raster {key} ...')
                nm = build_model(w, aw, bw)
                figt, axt = plt.subplots()
                nm.plot_bifurcation_diagram(
                    xlim=XLIM, ylim=YLIM, num_x=NUM_X, num_y=NUM_Y,
                    refinement_levels=REFINEMENT, max_count=MAX_COUNT,
                    pool=pool, ax=axt, stability_criterion=CRITERION)
                imgs.append(np.asarray(axt.images[-1].get_array(), dtype=int))
                plt.close(figt)
        imgs = np.stack(imgs)
        save_cache('raster', fp, imgs=imgs)
    else:
        imgs = cached['imgs']
    _plot_rasters(imgs)
    return imgs


def _plot_rasters(imgs):
    n = len(CASES)
    fig, axes = plt.subplots(1, n, figsize=(3.5*n + 1.4, 4.9), squeeze=False)
    cmap = plt.get_cmap('viridis', MAX_COUNT + 1)
    norm = BoundaryNorm(np.arange(-0.5, MAX_COUNT + 1.5), MAX_COUNT + 1)
    targets = model.Targets(locs=TARGET_LOCS, geom_name='circle', r=TARGET_R)
    for ci, (key, lab, *_r) in enumerate(CASES):
        ax = axes[0, ci]
        ax.imshow(np.clip(imgs[ci], 0, MAX_COUNT), origin='lower',
                  extent=[XLIM[0], XLIM[1], YLIM[0], YLIM[1]], aspect='equal',
                  interpolation='nearest', cmap=cmap, norm=norm)
        targets.plot_targets_to_axis(ax)
        ax.set_title(lab, fontsize=9.5)
        ax.set_xlabel('observer x')
        if ci == 0:
            ax.set_ylabel('observer y')
    # Skip the 0-stable swatch unless some cell actually has one -- this
    # geometry never does, and an unused entry just invites the reader to hunt
    # for a colour that isn't in the panels.
    lo = 0 if (imgs == 0).any() else 1
    handles = [plt.Line2D([], [], marker='s', ms=13, ls='', color=cmap(norm(i)),
                          label=f'{i}') for i in range(lo, MAX_COUNT + 1)]
    fig.legend(handles=handles, title='# stable\nequilibria', loc='center right',
               bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=11,
               title_fontsize=11)
    fig.suptitle('Stable self-consistent equilibrium count under four weightings '
                 '(locust 3-target geometry)', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 0.91, 0.95])
    out = os.path.join(HERE, 'outward_bias_rasters.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', out)


# =====================================================================
# Stage: walkers
# =====================================================================
# Shipped locust walker knobs (walker_analysis/three_target_locust.py).
W_NOISE_EXP, W_R_EXP = 2.0, 3.0
W_STD, W_V, W_DT = 3.0, 0.20, 0.04
W_TOL, W_MAXSTEPS, W_SEED = 0.20, 5000, 3
W_REPS = 400


def stage_walkers(regenerate=False):
    print('[walkers]')
    fp = dict(_base_fp(), stage='walkers', reps=W_REPS, seed=W_SEED,
              std=W_STD, v=W_V, dt=W_DT, noise_exp=W_NOISE_EXP,
              R_exp=W_R_EXP, tol=W_TOL, max_steps=W_MAXSTEPS)
    cached = load_cache('walkers', fp, regenerate)
    if cached is None:
        counts = np.zeros((len(CASES), 4), int)     # centre, upper, lower, lost
        tracks = {}
        seeds = np.random.SeedSequence(W_SEED).spawn(W_REPS)
        with Pool(N_WORKERS) as pool:
            for ci, (key, _lab, w, aw, bw, _c) in enumerate(CASES):
                nm = build_model(w, aw, bw)
                args = [(n, seeds[n], (0.0, 0.0), 0.0, 0.3 + 0j, W_DT, W_V,
                         W_STD, 0.0, W_NOISE_EXP, W_R_EXP, W_MAXSTEPS, W_TOL)
                        for n in range(W_REPS)]
                res = pool.map(nm._simulate_one_walk, args)
                walks = [wk for wk, _ in res]
                for wk in walks:
                    end = wk[:, -1]
                    d = np.linalg.norm(TARGET_LOCS - end, axis=1)
                    counts[ci, int(np.argmin(d)) if d.min() < 0.5 else 3] += 1
                # store a subsample of tracks for the figure (ragged -> pad)
                keep = walks[:120]
                L = max(wk.shape[1] for wk in keep)
                arr = np.full((len(keep), 2, L), np.nan)
                for i, wk in enumerate(keep):
                    arr[i, :, :wk.shape[1]] = wk
                tracks[f'tracks_{ci}'] = arr
                print(f'  {key}: centre/upper/lower/lost = {counts[ci].tolist()}'
                      f'  -> centre {counts[ci,0]/max(counts[ci,:3].sum(),1):.1%}')
        save_cache('walkers', fp, counts=counts, **tracks)
        # _plot_walkers reads tracks by key, so give it the same mapping the
        # cached path would (no need to round-trip through the file).
        cached = dict(tracks, counts=counts)
    counts = cached['counts']
    _plot_walkers(counts, cached)
    return counts


def _plot_walkers(counts, cached):
    n = len(CASES)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.4*n + 4.4, 4.6),
                             gridspec_kw=dict(width_ratios=[1]*n + [1.15]))
    targets = model.Targets(locs=TARGET_LOCS, geom_name='circle', r=TARGET_R)
    for ci, (key, lab, *_r) in enumerate(CASES):
        ax = axes[ci]
        arr = cached[f'tracks_{ci}']
        for wk in arr:
            ax.plot(wk[0], wk[1], 'k', alpha=0.32, lw=0.7)
        targets.plot_targets_to_axis(ax)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect('equal')
        tot = counts[ci, :3].sum()
        pct = counts[ci, 0] / tot if tot else np.nan
        ax.set_title(f'{lab}\ncentre {pct:.0%}   (lost {counts[ci,3]})',
                     fontsize=9)
        ax.set_xlabel('x')
        if ci == 0:
            ax.set_ylabel('y')

    ax = axes[-1]
    idx = np.arange(len(CASES))
    tot = counts[:, :3].sum(axis=1).astype(float)
    centre_frac = counts[:, 0] / np.where(tot > 0, tot, 1)
    cols = [c[5] for c in CASES]
    ax.bar(idx, centre_frac, color=cols)
    ax.axhline(0.29, color='k', lw=1.6, ls='--')
    ax.text(len(CASES) - 0.55, 0.30, 'locust data 29%', fontsize=8.5, ha='right')
    ax.axhline(1/3, color='0.6', lw=1.0, ls=':')
    ax.set_xticks(idx)
    ax.set_xticklabels([c[0] for c in CASES], rotation=20)
    ax.set_ylabel('fraction committing to the CENTRE target')
    ax.set_ylim(0, 1.0)
    ax.set_title(f'Endpoint census ({W_REPS} walkers each)', fontsize=10)
    ax.grid(axis='y', alpha=0.25)

    fig.suptitle('Locust three-target walkers under the four weightings',
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, 'outward_bias_walkers.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', out)


# =====================================================================

STAGES = {'mechanism': stage_mechanism, 'cascade': stage_cascade,
          'events': stage_events, 'raster': stage_raster,
          'walkers': stage_walkers}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('stages', nargs='*', default=['all'],
                    help='one or more of: ' + ', '.join(STAGES) + ', or all')
    ap.add_argument('--regenerate', action='store_true',
                    help='ignore caches and recompute')
    args = ap.parse_args(argv)
    todo = list(STAGES) if 'all' in args.stages else args.stages
    bad = [s for s in todo if s not in STAGES]
    if bad:
        ap.error(f'unknown stage(s): {bad}; choose from {list(STAGES)}')
    print(f'workers: {N_WORKERS}   stages: {todo}')
    for s in todo:
        STAGES[s](regenerate=args.regenerate)


if __name__ == '__main__':
    main()
