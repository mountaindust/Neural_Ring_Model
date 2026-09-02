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

"""Three-target FRUIT FLY walkers, REFINED for substructure matching vs GODM.

A high-realization sibling of ``three_target_fly.py`` (which is left untouched to
preserve the shipped result). It started from three_target_fly.py's knobs but has
since been refit against the GODM heatmap (see three_target_fly_refine_findings.md):
**K lowered 3.5->2.0** (the model default; lower K gentles the peel and de-biases the
centre) and **a_warp raised 0.45->0.65pi** (pushes the first bifurcation out to the
empirical x, so walkers stop peeling toward the targets too early). These also let the
SAME parameterization fit the 2-target case (two_target_fly_refine.py). What else
changes from the shipped script is the *measurement*:

  1. Many more realizations (default 1500; full worker pool from machine_config),
     so the occupancy field is statistically smooth.
  2. The walker density is rendered with the SAME pipeline as the empirical
     heatmaps (godm_heatmaps.py): a sliding within-trajectory time window, each
     window's 2-D histogram per-window-normalised, Gaussian-blurred, and
     max-projected. THIS is what exposes the bifurcation substructure -- the trunk
     ridge and the post-split branches at uniform contrast -- instead of a plain
     histogram swamped by the origin start-point peak. The walker's step index is
     the time proxy (constant speed v => step index proportional to arc length,
     exactly the near-constant-speed assumption the GODM render makes).
  3. Rendered on the EXACT same extent, target geometry, orientation, blur, and
     y-mirror as godm_heatmap_fly3 -- so the two panels are pixel-comparable.

Run:  python plots/three_target_fly_refine.py            # default reps
      NR_REPS=3000 python plots/three_target_fly_refine.py
      python plots/three_target_fly_refine.py 800         # reps as arg
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decision_model as model
from parallel_config import get_n_workers
import walker_analysis.godm_heatmaps as gh  # reuse the exact empirical render pipeline

pi = np.pi

# --- fixed experimental geometry (do NOT change) ---
target_locs = np.array([[5.0000,  0.0000],
                        [3.8302,  3.2139],
                        [3.8302, -3.2139]])
R = 0.5

# --- model knobs (refit vs GODM; see three_target_fly_refine_findings.md) ---
# a_warp sets the first-bifurcation x: raising it pushes the up/down split outward so
# walkers stop committing to the targets too early (the empirical ridge reaches the
# outer targets from higher x than the early-peeling walkers did). 0.65pi lands the
# 3-target split ~on the data and lets the SAME warp help the 2-target case.
# IMPORTANT: pushing a_warp out only helps once K is low (K=2) -- at K=3.5 the longer
# trunk over-recaptures to the reborn centre branch. b_warp also sets the rear blind
# spot / the 2nd bifurcation. Override with NR_A_WARP.
A_WARP = float(os.environ.get('NR_A_WARP', 0.65)) * pi
B_WARP = 0.92*pi
# a_weight is the centre/outer split lever (three_target_findings.md); overridable
# for sweeping. Lower a_weight releases more walkers to the outer targets.
A_WEIGHT = float(os.environ.get('NR_A_WEIGHT', 0.20)) * pi
B_WEIGHT = 0.80*pi
# K is the turning gain dtheta/dt = K*R*sin(Theta/2). It does NOT move the bifurcation
# locations (K-invariant; see .claude/rules/torque-and-stability.md) but sets how
# SHARPLY the walker peels once past one: higher K -> tighter pull to the dead-ahead
# consensus -> MORE corner-cutting + centre recapture (worse vs GODM); lower K -> gentler
# peel, walkers ride the trunk longer and reach the outer targets along the empirical
# ridge. K=2 (the model default) fits best; 3.5 (the old refine value) over-peeled.
# Override with NR_K.
K = float(os.environ.get('NR_K', 2.0))
# BETA is the neural Boltzmann factor. This scene has 3 targets and the shipped
# results were produced under the earlier per-target temperature T=0.10, whose
# effective coupling was N_targets/T, so beta = 3/0.10 = 30 reproduces them.
# The two-target sibling therefore does NOT share this value -- see
# two_target_fly_refine.py, which sets its own beta = 2/0.10 = 20.
BETA = 30.0
NOISE_EXP, R_EXP = 2.0, 3.0
# std is the centre de-bias lever AT HIGH REALIZATION COUNT. The shipped
# three_target_fly.py uses std=2.5, which three_target_findings.md reported as ~45-49%
# centre -- but that was an 80-walker estimate; at N>=1000 std=2.5 settles to ~55%
# centre (more centre-biased). std=4.0 recovers the empirical ~45% centre (measured
# 49-50%) AND best reconstructs the GODM fly3 heatmap (a clean, balanced trident;
# std=6.0 over-corrects, fading the centre trunk below the data). a_weight does NOT
# move the split in this regime (saturated). Override with NR_STD.
STD = float(os.environ.get('NR_STD', 4.0))
V, DT = 0.30, 0.05
TARGET_TOL = 0.20
MAX_STEPS, SEED = 4000, 3

# Release scatter: the empirical flies are not all launched from a single point /
# heading, so the GODM trunk is a broad ridge near x=0 (with the nested-chevron
# beads), while a delta-function launch gives a thin streak. A modest start spread
# mimics that scatter. Documented in three_target_findings.md to barely move the
# centre/outer split (start-heading spread 0->+-55 deg: 67->62% centre), so it is a
# render-fidelity knob, not a split-tuning one. Override with NR_POS_STD / NR_HEAD_STD.
START_POS_STD = float(os.environ.get('NR_POS_STD', 0.075))     # sigma of x,y start jitter
START_HEAD_STD = float(os.environ.get('NR_HEAD_STD', np.radians(12.0)))  # sigma of heading

# Many realizations for a smooth field; override with NR_REPS or argv[1].
REPETITIONS = int(os.environ.get('NR_REPS', sys.argv[1] if len(sys.argv) > 1 else 1500))

# GODM fly3 render constants (must match godm_heatmaps CASES['fly3']).
GODM_CASE = 'fly3'
BLUR = gh.CASES[GODM_CASE]['blur']        # 101  (cv2 ksize -> sigma rule)
ORIENT = gh.CASES[GODM_CASE]['orient']    # 'rot90'
# Empirical posts (rotated frame) define the shared plotting window.
_A40 = 2.0 * pi / 9.0                      # 40 deg
EXTENT = (0.0, 5.0, -5.0*np.sin(_A40), 5.0*np.sin(_A40))   # (0, 5, -3.214, 3.214)
HIST_RANGE = [[EXTENT[0], EXTENT[1]], [EXTENT[2], EXTENT[3]]]

# Time-window render: rescale the walker step index to ~N_TIME integer units so
# the sliding window mirrors the GODM fly3 setup (window ~ half of tmax).
N_TIME = 55          # target number of integer time units (cf. GODM tmax=60)
WIN_FRAC = 0.5       # window length as a fraction of tmax (GODM: 30/60)


def build_model():
    targets = model.Targets(locs=target_locs, geom_name='circle', r=R)
    pm = model.PerceptionModel(targets, (0, 0), 0,
                               neural_angle_dist='lin_cutoff', angle_weight='lin_cutoff',
                               a_warp=A_WARP, b_warp=B_WARP,
                               a_weight=A_WEIGHT, b_weight=B_WEIGHT)
    return model.NeuralBandModel(pm, beta=BETA, K=K)


def run_walkers(nm, pool=None, reps=REPETITIONS, std=STD, v=V,
                pos_std=START_POS_STD, head_std=START_HEAD_STD):
    """Return a list of (2, n) trajectory arrays.

    Each walker is launched from a jittered start (Gaussian sigma ``pos_std`` in
    x and y, ``head_std`` in heading) to mimic the empirical release scatter; set
    both to 0 for the original single-point launch.
    """
    seeds = np.random.SeedSequence(SEED).spawn(reps)
    srng = np.random.default_rng(SEED)            # independent of the per-walk noise
    starts = srng.normal(0.0, pos_std, size=(reps, 2)) if pos_std > 0 else np.zeros((reps, 2))
    heads = srng.normal(0.0, head_std, size=reps) if head_std > 0 else np.zeros(reps)
    args = [(n, seeds[n], tuple(starts[n]), float(heads[n]), 0.3 + 0j, DT, v, std, 0.0,
             NOISE_EXP, R_EXP, MAX_STEPS, TARGET_TOL) for n in range(reps)]
    res = pool.map(nm._simulate_one_walk, args) if pool else [nm._simulate_one_walk(a) for a in args]
    return [w for w, _ in res]


def endpoint_split(walks):
    """Count which target each walker ends nearest (0=center, 1=upper, 2=lower)."""
    counts = np.zeros(len(target_locs), int)
    for w in walks:
        end = w[:, -1]
        counts[int(np.argmin(np.linalg.norm(target_locs - end, axis=1)))] += 1
    return counts


def godm_density(walks, mirror=True):
    """Render the walker ensemble through the GODM max-projection pipeline.

    Each walker contributes (x, y, t) samples with ``t`` = step index rescaled to
    integer time units. We slide a window over t, per-window-normalise a blurred
    2-D histogram, and max-project -- exactly godm_heatmaps._max_project. Returns
    an image on EXTENT directly comparable to compute_heatmap('fly3').
    """
    xs = np.concatenate([w[0] for w in walks])
    ys = np.concatenate([w[1] for w in walks])
    # Per-walker integer step-index time, rescaled so the longest walk ~ N_TIME.
    lengths = [w.shape[1] for w in walks]
    scale = N_TIME / float(np.percentile(lengths, 95))
    ts = np.concatenate([np.arange(L) for L in lengths]).astype(float) * scale
    ts = np.floor(ts).astype(int)

    if mirror:                       # GODM fly3 exploits the y-symmetry (mirror=True)
        xs = np.concatenate((xs, xs))
        ys = np.concatenate((ys, -ys))
        ts = np.concatenate((ts, ts))

    tmax = int(ts.max()) + 1
    window = max(2, int(round(WIN_FRAC * tmax)))
    img = gh._max_project(xs, ys, ts, HIST_RANGE, BLUR, ORIENT, window, tmax)
    return img


def similarity(walk_img, ref_img):
    """Pixel-wise agreement of two heatmaps on the shared grid.

    Both images come off the same NBINS/HIST_RANGE/orient/blur pipeline, so they
    align pixel-for-pixel. Returns (corr_all, corr_support): Pearson correlation
    over all finite pixels, and over the empirical support (ref > 0.05) where the
    structure actually lives.
    """
    a = np.nan_to_num(np.asarray(walk_img, float))
    b = np.nan_to_num(np.asarray(ref_img, float))
    if a.shape != b.shape:                      # guard; should not happen
        return float('nan'), float('nan')
    af, bf = a.ravel(), b.ravel()
    corr_all = float(np.corrcoef(af, bf)[0, 1])
    sup = bf > 0.05
    corr_sup = float(np.corrcoef(af[sup], bf[sup])[0, 1]) if sup.sum() > 10 else float('nan')
    return corr_all, corr_sup


def main():
    n_workers = get_n_workers()
    print(f'fly walker refine: {REPETITIONS} realizations on {n_workers} workers')
    nm = build_model()
    with Pool(n_workers) as pool:
        walks = run_walkers(nm, pool=pool)

    counts = endpoint_split(walks)
    lengths = np.array([w.shape[1] for w in walks])
    print('endpoint split (center, upper, lower):', counts.tolist(),
          ' -> centre %.1f%%  outer %.1f%%'
          % (100*counts[0]/counts.sum(), 100*counts[1:].sum()/counts.sum()))
    print('trajectory steps: mean %.0f  median %.0f  min %d  max %d'
          % (lengths.mean(), np.median(lengths), lengths.min(), lengths.max()))

    # Reference empirical panel (recomputed from the GODM data on the same axes).
    print('computing GODM fly3 reference heatmap ...')
    ref_img, ref_extent, _ = gh.compute_heatmap(GODM_CASE, verbose=False)
    walk_img = godm_density(walks)

    corr_all, corr_sup = similarity(walk_img, ref_img)
    print('heatmap similarity vs GODM fly3:  corr(all)=%.3f  corr(support)=%.3f'
          % (corr_all, corr_sup))

    # Self-contained npz: walker tracks + the empirical heatmap + every parameter
    # needed to label/plot a publication panel (so a plotting script needs neither
    # pandas nor the GODM data). Object array of (2, n) tracks -> load with
    # allow_pickle=True.
    here = os.path.dirname(os.path.abspath(__file__))
    npz = os.path.join(here, 'three_target_fly_refine.npz')
    # Build the object array by fill (np.array(list_of_(2,n), dtype=object) tries
    # to make a regular 2-D array and fails when the track lengths differ).
    walks_obj = np.empty(len(walks), dtype=object)
    for i, w in enumerate(walks):
        walks_obj[i] = w
    np.savez(npz,
             walks=walks_obj,
             ref_img=ref_img, extent=np.array(EXTENT, float),
             corr_all=corr_all, corr_support=corr_sup, split=counts,
             reps=REPETITIONS, seed=SEED,
             K=K, beta=BETA, std=STD, v=V, dt=DT,
             a_warp=A_WARP, b_warp=B_WARP, a_weight=A_WEIGHT, b_weight=B_WEIGHT,
             noise_exp=NOISE_EXP, R_exp=R_EXP, target_tol=TARGET_TOL,
             start_pos_std=START_POS_STD, start_head_std=START_HEAD_STD,
             target_locs=target_locs, target_R=R, godm_case=GODM_CASE)
    print('wrote', npz)

    fig, ax = plt.subplots(1, 4, figsize=(21, 5.5))
    # Panel 0: empirical reference
    ax[0].imshow(ref_img, extent=ref_extent, origin='upper', aspect='equal')
    ax[0].set_title('GODM fly3 (empirical, N=125)')
    # Panel 1: walker density, SAME pipeline / extent / orientation
    ax[1].imshow(walk_img, extent=EXTENT, origin='upper', aspect='equal')
    ax[1].set_title('walker density, GODM pipeline (N=%d)' % REPETITIONS)
    # Panel 2: overlay -- walker density with the empirical ridge as contours.
    # contour() takes the same extent/origin mapping as imshow, so the lines align.
    ax[2].imshow(walk_img, extent=EXTENT, origin='upper', aspect='equal')
    ax[2].contour(np.nan_to_num(ref_img), levels=[0.35, 0.6],
                  colors=['white', 'red'], linewidths=1.0, alpha=0.9,
                  extent=EXTENT, origin='upper')
    ax[2].set_xlim(EXTENT[0], EXTENT[1]); ax[2].set_ylim(EXTENT[2], EXTENT[3])
    ax[2].set_title('walker density + GODM ridge contour\n(corr_support=%.2f)' % corr_sup)
    # Panel 3: raw tracks for context
    for w in walks[:300]:
        ax[3].plot(w[0], w[1], 'k', alpha=0.10, lw=0.6)
    nm.percep_model.targets.plot_targets_to_axis(ax[3])
    ax[3].set_xlim(EXTENT[0], EXTENT[1]); ax[3].set_ylim(EXTENT[2], EXTENT[3])
    ax[3].set_aspect('equal'); ax[3].set_title('walker tracks (first 300)')
    for a in ax:
        a.set_xlabel('x'); a.set_ylabel('y')
    fig.suptitle('Fly three-target: walker substructure vs GODM  '
                 '(K=%.1f, β=%.4g, a_warp=%.2fπ, a_weight=%.2fπ, σ=%.1f, '
                 'start jitter pos=%.2f head=%.0f°)'
                 % (K, BETA, A_WARP/pi, A_WEIGHT/pi, STD, START_POS_STD,
                    np.degrees(START_HEAD_STD)), y=1.03)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'three_target_fly_refine.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
