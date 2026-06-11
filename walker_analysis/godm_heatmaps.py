"""Reproduce the Figure-1 (x, y) occupancy heatmaps from Sridhar, Li, Gorbonos
et al. 2021, "The geometry of decision-making in individuals and collectives"
(PNAS) -- the *heatmap layer only*, with none of the paper's overlays (black
trajectory scatter, fitted bifurcation curves, red target markers).

Why this file exists
--------------------
Figure 1 of that paper shows, for flies and locusts choosing between two or
three targets, a 2-D density of body positions in a target-centred frame. As
the animal approaches, the single occupancy ridge splits -- the geometric
"bifurcation" of decision-making. We want exactly those density fields (and
their spatial extents) so a NeuralBandModel walker ensemble can be compared to
the empirical data on the same axes, the same way the authors did.

What is reproduced
------------------
Four panels, one per (species, n_targets):

    'fly2'     flies,   2 targets (60 deg separation, distance 5)
    'fly3'     flies,   3 targets (40 deg separation, distance 5)
    'locust2'  locusts, 2 targets (45 deg separation, distance 2)
    'locust3'  locusts, 3 targets (35 deg separation, distance 2)

The pipeline is a faithful port of the GODM analysis notebooks
(``Analysis/flies/quantify_bifurcations.ipynb`` and
``Analysis/locusts/quantify_bifurcations_n{2,3}.ipynb``) restricted to the
density computation. Per case:

  1. Read each experiment's ``results.csv`` and the post (target) geometry from
     the SQLite project/experiment databases shipped in the GODM repo.
  2. Keep only the decision-phase trajectories (nStimuli stages 1..3) at the
     target angular separation used for that panel (all separations for the
     3-target locust case, which the notebook pools).
  3. Spatially discretise each trajectory: drop samples closer than 0.01 to the
     last kept sample, so standing still does not dominate the density.
  4. Rotate every trajectory (and its posts) so the targets sit symmetrically
     about the +x axis -- a target-centred frame shared across experiments.
  5. Keep trajectories that *ended* within a threshold distance of a target
     (i.e. the animal actually committed), and -- for flies -- have > 30
     samples (the notebook's swarm ``len(idx) > 30`` gate).
  6. Re-segment the pooled stream into per-trajectory events by position jumps
     and keep a duration band (the "signal"/"bifurcation" trajectories).
  7. For a window sliding over within-trajectory time, take a Gaussian-blurred,
     per-window-normalised 2-D histogram and max-project across windows. This
     is the trick that renders both the pre-split ridge and the post-split
     branches at full contrast instead of a blob.

The original code blurs with ``cv2.GaussianBlur``; OpenCV is not a dependency
of this project, so :func:`cv2_gaussian_blur` reproduces it with
``scipy.ndimage.gaussian_filter`` using OpenCV's auto sigma(ksize) rule and a
matching kernel radius and border mode.

Data location
-------------
The GODM repository is expected at ``../../GODM`` relative to this file (i.e.
a sibling of the Neural_Ring_Model project), or at ``$GODM_DIR`` if set. Only
the read-only ``Data/`` and ``Analysis/*.db`` files are touched.

Usage
-----
    python walker_analysis/godm_heatmaps.py            # all four panels -> PNGs
    python walker_analysis/godm_heatmaps.py fly2        # one panel

Programmatic (for overlaying your own model on the same axes)::

    from walker_analysis import godm_heatmaps as gh
    img, extent, posts = gh.compute_heatmap('fly2')
    # img: 2-D float array in [0, 1]; extent: (xmin, xmax, ymin, ymax) in the
    # target-centred frame; posts: dict of target (x, y) positions.
    ax.imshow(img, extent=extent, origin='upper', aspect='equal')
"""
import os
import sys
import sqlite3

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
GODM_DIR = os.environ.get(
    'GODM_DIR', os.path.normpath(os.path.join(HERE, '..', '..', 'GODM')))

NBINS = 500             # histogram resolution (notebook: nbins = 500)
SPATIAL_STEP = 0.01     # spatial discretisation distance (notebook constant)
PROJECT = 'DecisionGeometry'


# --------------------------------------------------------------------------- #
# Case configuration
# --------------------------------------------------------------------------- #
# Each case lists the experiment batches (one DB pair each), the experiment
# index window, and the panel-specific parameters lifted from the notebooks.
def _fly_db(name):
    return os.path.join(GODM_DIR, 'Analysis', 'flies', name)


def _loc_db(name):
    return os.path.join(GODM_DIR, 'Analysis', 'locusts', name)


FLY_DATA = os.path.join(GODM_DIR, 'Data', 'flies')
LOC_DATA = os.path.join(GODM_DIR, 'Data', 'locusts')

# CSV column layouts differ by species (flies carry an extra 'dir' column).
FLY_COLS = ['x', 'y', 'z', 'dir', 'event', 't', 'nStimuli']
LOC_COLS = ['x', 'y', 'z', 'event', 't', 'nStimuli']

CASES = {
    'fly2': dict(
        species='fly', n_posts=2, data_dir=FLY_DATA, csv_cols=FLY_COLS,
        batches=[(_fly_db('flyProjects.db'), _fly_db('flyExperiments.db'))],
        exp_lo=0, exp_hi=10,
        angle_index=0,          # angles[0] == 60 deg  (notebook get((0, 1)))
        dmin_thresh=0.8, min_samples=30,
        jump=0.5, time_mode='seconds', dur_lo=26.5, dur_hi=None,
        window=20, tmax=50, blur=201, orient='rot90', mirror=False,
        posts='hardcoded',
    ),
    'fly3': dict(
        species='fly', n_posts=3, data_dir=FLY_DATA, csv_cols=FLY_COLS,
        batches=[(_fly_db('flyProjects.db'), _fly_db('flyExperiments.db'))],
        exp_lo=10, exp_hi=20,
        angle_index=1,          # angles[1] == 40 deg  (notebook get((0, 2)))
        dmin_thresh=0.8, min_samples=30,
        jump=0.5, time_mode='seconds', dur_lo=24.5, dur_hi=None,
        window=30, tmax=60, blur=101, orient='rot90', mirror=True,
        posts='hardcoded',
    ),
    'locust2': dict(
        species='locust', n_posts=2, data_dir=LOC_DATA, csv_cols=LOC_COLS,
        batches=[
            (_loc_db('locustProjects_20_01_07.db'),
             _loc_db('locustExperiments_20_01_07.db')),
            (_loc_db('locustProjects_19_11_28.db'),
             _loc_db('locustExperiments_19_11_28.db')),
        ],
        exp_lo=0, exp_hi=10,
        angle_index=1,          # angles[1] == 45 deg (both batches)
        dmin_thresh=0.5, min_samples=None,
        jump=0.2, time_mode='frames', dur_lo=170, dur_hi=300,
        window=50, tmax=250, blur=201, orient='rot90', mirror=False,
        posts='data',
    ),
    'locust3': dict(
        species='locust', n_posts=3, data_dir=LOC_DATA, csv_cols=LOC_COLS,
        batches=[
            (_loc_db('locustProjects_2_3post-june.db'),
             _loc_db('locustExperiments_2_3post-june.db')),
            (_loc_db('locustProjects_2_3post.db'),
             _loc_db('locustExperiments_2_3post.db')),
        ],
        exp_lo=0, exp_hi=10,
        angle_index=None,       # pool all decision separations (notebook)
        # The 3-post locust 'angle' DB field is not the target separation, so
        # the notebook rotates by a fixed 35 deg rather than (n-1)*sep/2.
        rot_fixed=np.pi * 35.0 / 180.0,
        dmin_thresh=1.0, min_samples=None,
        jump=0.2, time_mode='frames', dur_lo=280, dur_hi=500,
        window=70, tmax=300, blur=101, orient='flipud_rot90', mirror=True,
        posts='data',
    ),
}


# --------------------------------------------------------------------------- #
# Primitive helpers
# --------------------------------------------------------------------------- #
def rotate(x, y, ang):
    """Rotate (x, y) by ``ang`` using the notebooks' [[c, s], [-s, c]] matrix."""
    c, s = np.cos(ang), np.sin(ang)
    return c * x + s * y, -s * x + c * y


def cv2_gaussian_blur(img, ksize):
    """Equivalent of ``cv2.GaussianBlur(img, (ksize, ksize), 0)``.

    OpenCV picks sigma from the kernel size when sigmaX is 0:
    ``sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8``. Its default border is
    BORDER_REFLECT_101, which matches scipy's ``mode='mirror'``. We cap the
    kernel radius at ``(ksize - 1) // 2`` so the support matches OpenCV's.
    """
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    radius = (ksize - 1) // 2
    return gaussian_filter(img, sigma=sigma, mode='mirror', radius=radius)


_ORIENT = {
    'rot90': np.rot90,
    'flipud_rot90': lambda a: np.flipud(np.rot90(a)),
}


def _discretise_mask(x, y, step):
    """Sequential radial downsample: keep a sample iff it lies more than
    ``step`` from the last kept sample. Mirrors the notebooks' per-track loop
    (the first sample of each track is left unselected, as there). Returns a
    boolean keep-mask. Runs over Python lists for speed (no numba available).
    """
    n = len(x)
    keep = np.zeros(n, dtype=bool)
    if n < 2:
        return keep
    xl = x.tolist()
    yl = y.tolist()
    sx = xl[0]
    sy = yl[0]
    s2 = step * step
    for k in range(1, n):
        dx = xl[k] - sx
        dy = yl[k] - sy
        if dx * dx + dy * dy > s2:
            keep[k] = True
            sx = xl[k]
            sy = yl[k]
    return keep


# --------------------------------------------------------------------------- #
# Data loading + preprocessing
# --------------------------------------------------------------------------- #
def _gather_angles(proj_db, exp_lo, exp_hi):
    """Sorted unique target separations (radians) over the experiment window --
    the notebooks' ``angles`` array, used to pick the panel's angle group."""
    cur = sqlite3.connect(proj_db).cursor()
    rows = cur.execute(
        'SELECT post1 FROM projects WHERE project = ? AND exp >= ? AND exp < ?',
        (PROJECT, exp_lo, exp_hi)).fetchall()
    angs = []
    for (a,) in rows:
        if a and a != 'None':
            try:
                ang = eval(a)['angle']
            except Exception:
                continue
            if ang not in angs:
                angs.append(ang)
    return np.sort(np.array(angs))


def _load_batch(proj_db, exp_db, cfg, batch_tag, target_angle):
    """Load one DB-pair batch and return a tidy frame of *decision-phase*
    trajectories at the panel's target separation, already discretised,
    rotated into the target-centred frame, end-distance filtered, and (flies)
    sample-count filtered.

    Columns: uuid, nStimuli, event, t, rx, ry, and rotated post coordinates
    (rp0x, rp0y, rp1x, rp1y[, rp2x, rp2y]).
    """
    n_posts = cfg['n_posts']
    cur_e = sqlite3.connect(exp_db).cursor()
    cur_p = sqlite3.connect(proj_db).cursor()
    exp_ids = [r[0] for r in cur_e.execute(
        'SELECT expId FROM experiments WHERE project = ? AND exp >= ? AND exp < ?',
        (PROJECT, cfg['exp_lo'], cfg['exp_hi']))]

    frames = []
    for uuid, exp_id in enumerate(exp_ids):
        exp = cur_e.execute(
            'SELECT exp FROM experiments WHERE expId = ?', (exp_id,)).fetchone()[0]
        rep = cur_e.execute(
            'SELECT replicate FROM experiments WHERE expId = ?',
            (exp_id,)).fetchone()[0]

        # Post (target) geometry for each nStimuli stage, ordered by stage.
        post0 = [r[0] for r in cur_p.execute(
            'SELECT post0 FROM projects WHERE project = ? AND exp = ? AND '
            'replicate = ? ORDER BY nStimuli', (PROJECT, exp, rep))]
        post1 = [r[0] for r in cur_p.execute(
            'SELECT post1 FROM projects WHERE project = ? AND exp = ? AND '
            'replicate = ? ORDER BY nStimuli', (PROJECT, exp, rep))]
        post2 = [r[0] for r in cur_p.execute(
            'SELECT post2 FROM projects WHERE project = ? AND exp = ? AND '
            'replicate = ? ORDER BY nStimuli', (PROJECT, exp, rep))]

        csv = os.path.join(cfg['data_dir'], exp_id, 'results.csv')
        raw = pd.read_csv(csv, names=cfg['csv_cols'])
        raw = raw[['x', 'y', 'event', 't', 'nStimuli']].apply(
            pd.to_numeric, errors='coerce')
        raw = raw.dropna()                     # drop occasional malformed rows
        raw['nStimuli'] = raw['nStimuli'].astype(int)
        raw['event'] = raw['event'].astype(int)

        stages = np.sort(raw['nStimuli'].unique())
        decision = set(stages[1:-1].tolist())   # drop first/last (controls)

        for stage in decision:
            pdict0 = eval(post0[stage])
            sep = pdict0['angle']               # target separation at this stage
            if target_angle is not None and not np.isclose(sep, target_angle):
                continue
            p0 = pdict0['position']
            p1 = eval(post1[stage])['position']
            p2 = eval(post2[stage])['position'] if (
                n_posts == 3 and post2[stage] not in (None, 'None')) else None

            sub = raw[raw['nStimuli'] == stage]
            for ev, trk in sub.groupby('event'):
                trk = trk.sort_values('t')
                xv = trk['x'].to_numpy(float)
                yv = trk['y'].to_numpy(float)
                mask = _discretise_mask(xv, yv, SPATIAL_STEP)
                if not mask.any():
                    continue
                xv, yv = xv[mask], yv[mask]
                tv = trk['t'].to_numpy(float)[mask]

                # End-near-a-target filter (notebook: dmin at max-t row).
                dists = [np.hypot(xv[-1] - p0[0], yv[-1] - p0[1]),
                         np.hypot(xv[-1] - p1[0], yv[-1] - p1[1])]
                if p2 is not None:
                    dists.append(np.hypot(xv[-1] - p2[0], yv[-1] - p2[1]))
                if min(dists) >= cfg['dmin_thresh']:
                    continue
                if cfg['min_samples'] and len(xv) <= cfg['min_samples']:
                    continue

                # Rotate trajectory + posts so targets straddle the +x axis.
                base = np.arctan2(p0[1], p0[0])
                if cfg.get('rot_fixed') is not None:
                    ang = base + cfg['rot_fixed']
                else:
                    ang = base + (n_posts - 1) * sep / 2.0
                rx, ry = rotate(xv, yv, ang)
                rp0 = rotate(p0[0], p0[1], ang)
                rp1 = rotate(p1[0], p1[1], ang)
                rec = dict(uuid=f'{batch_tag}:{uuid}', nStimuli=stage,
                           event=int(ev), t=tv, rx=rx, ry=ry,
                           rp0x=rp0[0], rp0y=rp0[1], rp1x=rp1[0], rp1y=rp1[1])
                if n_posts == 3 and p2 is not None:
                    rp2 = rotate(p2[0], p2[1], ang)
                    rec['rp2x'], rec['rp2y'] = rp2[0], rp2[1]
                frames.append(pd.DataFrame({
                    k: (v if np.ndim(v) else np.full(len(rx), v))
                    for k, v in rec.items()}))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Event re-segmentation + duration band
# --------------------------------------------------------------------------- #
def _resegment(df, jump, time_mode, dur_lo, dur_hi):
    """Pool the trajectories (sorted so each is contiguous), split into events
    by position jumps >= ``jump``, assign a within-event time, then keep events
    whose duration falls in the panel's band. Returns (rx, ry, time) arrays.

    time_mode 'seconds': within-event time is ``t - t_event_start`` (flies).
    time_mode 'frames':  within-event time is the sample index since event
                         start (locusts' ``t2``).
    band: dur_hi is None  -> keep events with max-time >  dur_lo (flies, 'long')
          dur_hi is set    -> keep events with dur_lo < max-time <= dur_hi
                              (locusts, the mid-duration "signal" band)
    """
    df = df.sort_values(['uuid', 'nStimuli', 'event', 't']).reset_index(drop=True)
    rx = df['rx'].to_numpy(float)
    ry = df['ry'].to_numpy(float)
    n = len(rx)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    step = np.hypot(np.diff(rx), np.diff(ry))
    boundary = np.concatenate(([True], step >= jump))
    event_id = np.cumsum(boundary) - 1

    idx = np.arange(n)
    start_idx = np.maximum.accumulate(np.where(boundary, idx, -1))
    if time_mode == 'frames':
        within = (idx - start_idx).astype(float)
    else:  # seconds
        t = df['t'].to_numpy(float)
        within = t - t[start_idx]

    # Per-event maximum within-time = event duration.
    n_ev = event_id[-1] + 1
    dur = np.zeros(n_ev)
    np.maximum.at(dur, event_id, within)
    ev_dur = dur[event_id]

    if dur_hi is None:
        keep = ev_dur > dur_lo
    else:
        keep = (ev_dur > dur_lo) & (ev_dur <= dur_hi)
    return rx[keep], ry[keep], within[keep]


# --------------------------------------------------------------------------- #
# Heatmap assembly
# --------------------------------------------------------------------------- #
def _density_window(x, y, hist_range, blur, orient):
    """One window's contribution: normalised, blurred, oriented density."""
    h, _, _ = np.histogram2d(x, y, bins=NBINS, range=hist_range, density=True)
    img = _ORIENT[orient](cv2_gaussian_blur(h, blur))
    m = img.max()
    if m <= 0:
        return None
    return img / m


def _max_project(xs, ys, ts, hist_range, blur, orient, window, tmax):
    """Slide a time window over the trajectories and max-project the
    per-window normalised densities (notebook ``np.fmax`` accumulation)."""
    img = None
    for t0 in range(0, tmax - window):
        sel = (ts > t0) & (ts < t0 + window)
        if not sel.any():
            continue
        win = _density_window(xs[sel], ys[sel], hist_range, blur, orient)
        if win is None:
            continue
        img = win if img is None else np.fmax(win, img)
    return img


def _reference_posts(cfg, df):
    """Target positions (rotated frame) and the (x, y) plotting window.

    'hardcoded' (flies) uses the notebook's exact reference geometry; 'data'
    (locusts) reads the rotated post coordinates straight from the trajectories
    (constant across an experiment up to rotation; median over the pool).
    """
    n_posts = cfg['n_posts']
    if cfg['posts'] == 'hardcoded' and n_posts == 2:
        px = 5.0 * np.cos(np.pi / 6)
        posts = {'p0': (px, -5.0 * np.sin(np.pi / 6)),
                 'p1': (px, 5.0 * np.sin(np.pi / 6))}
        xmax = posts['p0'][0]
        y0, y1 = posts['p0'][1], posts['p1'][1]
    elif cfg['posts'] == 'hardcoded' and n_posts == 3:
        a = 2.0 * np.pi / 9.0          # 40 deg
        posts = {'p0': (5.0 * np.cos(a), -5.0 * np.sin(a)),
                 'p1': (5.0, 0.0),
                 'p2': (5.0 * np.cos(a), 5.0 * np.sin(a))}
        xmax = posts['p1'][0]
        y0, y1 = posts['p0'][1], posts['p2'][1]
    else:  # 'data' (locusts): read rotated posts from the pooled trajectories
        posts = {'p0': (float(df['rp0x'].median()), float(df['rp0y'].median())),
                 'p1': (float(df['rp1x'].median()), float(df['rp1y'].median()))}
        if n_posts == 3:
            posts['p2'] = (float(df['rp2x'].median()),
                           float(df['rp2y'].median()))
            xmax = posts['p1'][0]
            y0, y1 = posts['p0'][1], posts['p2'][1]
        else:
            xmax = posts['p0'][0]
            y0, y1 = posts['p0'][1], posts['p1'][1]
    ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
    extent = (0.0, xmax, ymin, ymax)
    return posts, extent


def compute_heatmap(case, verbose=True):
    """Build one panel's occupancy heatmap.

    Returns ``(img, extent, posts)`` where ``img`` is a 2-D float array
    normalised to [0, 1] (NaN where no trajectory ever fell), ``extent`` is
    ``(xmin, xmax, ymin, ymax)`` in the target-centred frame for ``imshow``,
    and ``posts`` maps target labels to (x, y) positions in that frame.
    """
    cfg = CASES[case]
    target_angle = None
    if cfg['angle_index'] is not None:
        angles = _gather_angles(cfg['batches'][0][0], cfg['exp_lo'], cfg['exp_hi'])
        target_angle = float(angles[cfg['angle_index']])

    parts = []
    for bi, (proj_db, exp_db) in enumerate(cfg['batches']):
        if verbose:
            print(f'  [{case}] loading batch {bi + 1}/{len(cfg["batches"])} '
                  f'({os.path.basename(proj_db)})')
        parts.append(_load_batch(proj_db, exp_db, cfg, f'b{bi}', target_angle))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    if verbose:
        print(f'  [{case}] {df["uuid"].nunique()} experiments, '
              f'{len(df.groupby(["uuid", "nStimuli", "event"]))} trajectories, '
              f'{len(df)} samples after preprocessing')

    posts, extent = _reference_posts(cfg, df)
    xmax, ymin, ymax = extent[1], extent[2], extent[3]
    hist_range = [[0.0, xmax], [ymin, ymax]]

    xs, ys, ts = _resegment(df, cfg['jump'], cfg['time_mode'],
                            cfg['dur_lo'], cfg['dur_hi'])
    if cfg['mirror']:               # 3-target panels exploit the y-symmetry
        xs = np.concatenate((xs, xs))
        ys = np.concatenate((ys, -ys))
        ts = np.concatenate((ts, ts))
    if cfg['time_mode'] == 'seconds':
        ts = ts.astype(np.uint8).astype(int)

    if verbose:
        print(f'  [{case}] {len(xs)} samples in the duration band; '
              f'max-projecting {cfg["tmax"] - cfg["window"]} time windows')
    img = _max_project(xs, ys, ts, hist_range, cfg['blur'], cfg['orient'],
                       cfg['window'], cfg['tmax'])
    return img, extent, posts


# --------------------------------------------------------------------------- #
# Plotting / CLI
# --------------------------------------------------------------------------- #
_TITLES = {
    'fly2': 'Flies, 2 targets (60$\\degree$)',
    'fly3': 'Flies, 3 targets (40$\\degree$)',
    'locust2': 'Locusts, 2 targets (45$\\degree$)',
    'locust3': 'Locusts, 3 targets (35$\\degree$)',
}


def save_figure(case, fname=None, verbose=True):
    """Compute one panel and save it as a clean heatmap PNG (no overlays)."""
    img, extent, posts = compute_heatmap(case, verbose=verbose)
    if fname is None:
        fname = f'godm_heatmap_{case}.png'
    out = os.path.join(HERE, fname)

    width = max(extent[1], 1e-6)
    height = max(extent[3] - extent[2], 1e-6)
    fig, ax = plt.subplots(figsize=(4.0, 4.0 * height / width + 0.4))
    ax.imshow(img, extent=extent, origin='upper', aspect='equal')
    ax.set_title(_TITLES.get(case, case))
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        kb = os.path.getsize(out) / 1024
        print(f'  [{case}] wrote {fname} ({kb:.1f} KB)\n')
    return out


def main(argv):
    cases = argv[1:] if len(argv) > 1 else list(CASES)
    bad = [c for c in cases if c not in CASES]
    if bad:
        sys.exit(f'unknown case(s): {bad}; choose from {list(CASES)}')
    print(f'GODM data: {GODM_DIR}')
    if not os.path.isdir(GODM_DIR):
        sys.exit(f'GODM repo not found at {GODM_DIR} (set $GODM_DIR)')
    print('Generating GODM Figure-1 heatmaps (heatmap layer only):')
    for case in cases:
        save_figure(case)
    print('done.')


if __name__ == '__main__':
    main(sys.argv)
