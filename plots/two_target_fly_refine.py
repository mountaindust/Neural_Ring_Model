"""Two-target FRUIT FLY walkers, REFINED for substructure matching vs GODM fly2.

Sibling of ``three_target_fly_refine.py`` for the 2-target fly experiment (GODM
'fly2': 60 deg target separation, distance 5). The MODEL PARAMETERIZATION IS
IDENTICAL to the 3-target refine -- it is *imported* from that module (single
source of truth), testing the hypothesis that the same fly in the same setup, just
with two targets, needs no re-tuning. Only the target geometry and the GODM render
target ('fly2') differ. Same workflow as the 3-target case: high-realization
ensemble -> GODM max-projection render -> Pearson corr + self-contained npz.

Run:  python plots/two_target_fly_refine.py            # default reps
      NR_REPS=2500 python plots/two_target_fly_refine.py
      python plots/two_target_fly_refine.py 800         # reps as arg
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
import walker_analysis.godm_heatmaps as gh
# Shared parameterization -- identical to the 3-target case by import, not by copy.
# (Sibling module in this same plots/ directory.)
from three_target_fly_refine import (
    K, T, NOISE_EXP, R_EXP, STD, V, DT, A_WARP, B_WARP, A_WEIGHT, B_WEIGHT,
    TARGET_TOL, MAX_STEPS, SEED, START_POS_STD, START_HEAD_STD, N_TIME, WIN_FRAC,
    similarity)

pi = np.pi

# --- fixed experimental geometry: GODM fly2 (60 deg separation, distance 5) ---
# Two targets at +-30 deg, radius 5 -> (4.330, +-2.5); same target radius as the
# 3-target fly. Matches godm_heatmaps _reference_posts('fly2') exactly.
_HALF = pi / 6.0                                   # 30 deg (half of the 60 deg sep)
target_locs = np.array([[5.0*np.cos(_HALF),  5.0*np.sin(_HALF)],
                        [5.0*np.cos(_HALF), -5.0*np.sin(_HALF)]])   # (4.330, +-2.5)
R = 0.5

# Many realizations for a smooth field; override with NR_REPS or argv[1].
REPETITIONS = int(os.environ.get('NR_REPS', sys.argv[1] if len(sys.argv) > 1 else 1500))

# GODM fly2 render constants (must match godm_heatmaps CASES['fly2']).
GODM_CASE = 'fly2'
BLUR = gh.CASES[GODM_CASE]['blur']        # 201 (heavier blur than fly3's 101)
ORIENT = gh.CASES[GODM_CASE]['orient']    # 'rot90'
MIRROR = gh.CASES[GODM_CASE]['mirror']    # False (fly2 is NOT y-mirrored)
_PX = 5.0 * np.cos(pi / 6.0)               # 4.330 -- fly2 reference window x-max
EXTENT = (0.0, _PX, -2.5, 2.5)
HIST_RANGE = [[EXTENT[0], EXTENT[1]], [EXTENT[2], EXTENT[3]]]


def build_model():
    targets = model.Targets(locs=target_locs, geom_name='circle', r=R)
    pm = model.PerceptionModel(targets, (0, 0), 0,
                               neural_angle_dist='lin_cutoff', angle_weight='lin_cutoff',
                               a_warp=A_WARP, b_warp=B_WARP,
                               a_weight=A_WEIGHT, b_weight=B_WEIGHT)
    return model.NeuralBandModel(pm, T=T, K=K)


def run_walkers(nm, pool=None, reps=REPETITIONS, std=STD, v=V,
                pos_std=START_POS_STD, head_std=START_HEAD_STD):
    """Return a list of (2, n) trajectory arrays (jittered start, as 3-target)."""
    seeds = np.random.SeedSequence(SEED).spawn(reps)
    srng = np.random.default_rng(SEED)
    starts = srng.normal(0.0, pos_std, size=(reps, 2)) if pos_std > 0 else np.zeros((reps, 2))
    heads = srng.normal(0.0, head_std, size=reps) if head_std > 0 else np.zeros(reps)
    args = [(n, seeds[n], tuple(starts[n]), float(heads[n]), 0.3 + 0j, DT, v, std, 0.0,
             NOISE_EXP, R_EXP, MAX_STEPS, TARGET_TOL) for n in range(reps)]
    res = pool.map(nm._simulate_one_walk, args) if pool else [nm._simulate_one_walk(a) for a in args]
    return [w for w, _ in res]


def endpoint_split(walks):
    """Count which target each walker ends nearest (0=upper, 1=lower)."""
    counts = np.zeros(len(target_locs), int)
    for w in walks:
        end = w[:, -1]
        counts[int(np.argmin(np.linalg.norm(target_locs - end, axis=1)))] += 1
    return counts


def godm_density(walks, mirror=MIRROR):
    """Render the walker ensemble through the GODM max-projection pipeline on the
    fly2 grid (step index = time proxy). mirror defaults to fly2's (False)."""
    xs = np.concatenate([w[0] for w in walks])
    ys = np.concatenate([w[1] for w in walks])
    lengths = [w.shape[1] for w in walks]
    scale = N_TIME / float(np.percentile(lengths, 95))
    ts = np.concatenate([np.arange(L) for L in lengths]).astype(float) * scale
    ts = np.floor(ts).astype(int)

    if mirror:
        xs = np.concatenate((xs, xs))
        ys = np.concatenate((ys, -ys))
        ts = np.concatenate((ts, ts))

    tmax = int(ts.max()) + 1
    window = max(2, int(round(WIN_FRAC * tmax)))
    return gh._max_project(xs, ys, ts, HIST_RANGE, BLUR, ORIENT, window, tmax)


def main():
    n_workers = get_n_workers()
    print(f'fly two-target refine: {REPETITIONS} realizations on {n_workers} workers')
    nm = build_model()
    with Pool(n_workers) as pool:
        walks = run_walkers(nm, pool=pool)

    counts = endpoint_split(walks)
    lengths = np.array([w.shape[1] for w in walks])
    print('endpoint split (upper, lower):', counts.tolist(),
          ' -> upper %.1f%%  lower %.1f%%'
          % (100*counts[0]/counts.sum(), 100*counts[1]/counts.sum()))
    print('trajectory steps: mean %.0f  median %.0f  min %d  max %d'
          % (lengths.mean(), np.median(lengths), lengths.min(), lengths.max()))

    print('computing GODM fly2 reference heatmap ...')
    ref_img, ref_extent, _ = gh.compute_heatmap(GODM_CASE, verbose=False)
    walk_img = godm_density(walks)

    corr_all, corr_sup = similarity(walk_img, ref_img)
    print('heatmap similarity vs GODM fly2:  corr(all)=%.3f  corr(support)=%.3f'
          % (corr_all, corr_sup))

    here = os.path.dirname(os.path.abspath(__file__))
    npz = os.path.join(here, 'two_target_fly_refine.npz')
    walks_obj = np.empty(len(walks), dtype=object)
    for i, w in enumerate(walks):
        walks_obj[i] = w
    np.savez(npz,
             walks=walks_obj,
             ref_img=ref_img, extent=np.array(EXTENT, float),
             corr_all=corr_all, corr_support=corr_sup, split=counts,
             reps=REPETITIONS, seed=SEED,
             K=K, T=T, std=STD, v=V, dt=DT,
             a_warp=A_WARP, b_warp=B_WARP, a_weight=A_WEIGHT, b_weight=B_WEIGHT,
             noise_exp=NOISE_EXP, R_exp=R_EXP, target_tol=TARGET_TOL,
             start_pos_std=START_POS_STD, start_head_std=START_HEAD_STD,
             target_locs=target_locs, target_R=R, godm_case=GODM_CASE)
    print('wrote', npz)

    fig, ax = plt.subplots(1, 4, figsize=(18, 5.5))
    ax[0].imshow(ref_img, extent=ref_extent, origin='upper', aspect='equal')
    ax[0].set_title('GODM fly2 (empirical)')
    ax[1].imshow(walk_img, extent=EXTENT, origin='upper', aspect='equal')
    ax[1].set_title('walker density, GODM pipeline (N=%d)' % REPETITIONS)
    ax[2].imshow(walk_img, extent=EXTENT, origin='upper', aspect='equal')
    ax[2].contour(np.nan_to_num(ref_img), levels=[0.35, 0.6],
                  colors=['white', 'red'], linewidths=1.0, alpha=0.9,
                  extent=EXTENT, origin='upper')
    ax[2].set_xlim(EXTENT[0], EXTENT[1]); ax[2].set_ylim(EXTENT[2], EXTENT[3])
    ax[2].set_title('walker density + GODM ridge contour\n(corr_support=%.2f)' % corr_sup)
    for w in walks[:300]:
        ax[3].plot(w[0], w[1], 'k', alpha=0.10, lw=0.6)
    nm.percep_model.targets.plot_targets_to_axis(ax[3])
    ax[3].set_xlim(EXTENT[0], EXTENT[1]); ax[3].set_ylim(EXTENT[2], EXTENT[3])
    ax[3].set_aspect('equal'); ax[3].set_title('walker tracks (first 300)')
    for a in ax:
        a.set_xlabel('x'); a.set_ylabel('y')
    fig.suptitle('Fly two-target: walker substructure vs GODM  '
                 '(K=%.1f, T=%.2f, a_warp=%.2fπ, a_weight=%.2fπ, σ=%.1f, '
                 'start jitter pos=%.3f head=%.0f°)'
                 % (K, T, A_WARP/pi, A_WEIGHT/pi, STD, START_POS_STD,
                    np.degrees(START_HEAD_STD)), y=1.03)
    fig.tight_layout()
    out = os.path.join(here, 'two_target_fly_refine.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
