"""Three-target LOCUST walkers, tuned to the GODM data (Sridhar et al. 2021).

Three circle targets 35 deg apart at radius 3, target radius 0.1 -- positions and size
FIXED by the experiment. (The empirical locust separation is 35 deg, verified from the
GODM posts, NOT the 40 deg an earlier prototype assumed.) The walker starts at the
origin and makes two sequential binary decisions, reaching all three targets.

Tuning: lower T and a HIGH turning gain K=6 (locusts move stop-and-go and reorient
fast, so the fly's K=2 is too sluggish), with a narrow foveal weight window (a_weight)
to push commitment toward the outer targets (the data is outer-biased, ~29% centre).

FINDING: the model cannot fully reproduce the locust's clean-yet-outer-biased split --
reaching 29% centre needs noise that muddies the tight trident, so this config favours
a clean trident (residual ~61% centre). See three_target_findings.md.

Run:  python walker_analysis/three_target_locust.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decision_model as model

pi = np.pi

# --- fixed experimental geometry (do NOT change) ---
# Empirical locust3 separation is 35 deg at distance 3 (verified from the GODM data
# posts) -- NOT the 40 deg the original three_target_locust.py assumed.
_a = np.radians(35.0)
target_locs = np.array([[3.0,            0.0],
                        [3.0*np.cos(_a),  3.0*np.sin(_a)],
                        [3.0*np.cos(_a), -3.0*np.sin(_a)]])
R = 0.1

# --- retuned model knobs ---
A_WARP, B_WARP = 0.40*pi, 0.90*pi      # a_warp near original: branches sit on the ridge
A_WEIGHT, B_WEIGHT = 0.10*pi, 0.80*pi  # narrow foveal weight: pushes walkers toward the
#   outer targets (the data is outer-biased, ~29% centre). NOTE: the model cannot fully
#   reach the locust's clean-yet-outer-biased split -- see three_target_findings.md.
K, T = 6.0, 0.10                       # locusts turn fast -> high K; low T for cleanliness
NOISE_EXP, R_EXP = 2.0, 3.0
STD, V, DT = 3.0, 0.20, 0.04           # std raised modestly (clean trident); higher std
#   reduces centre bias further but muddies the tracks -- this config favours cleanliness.
TARGET_TOL = 0.20
REPETITIONS, MAX_STEPS, SEED = 80, 5000, 3

XLIM, YLIM = (-0.3, 3.3), (-2.4, 2.4)


def build_model():
    targets = model.Targets(locs=target_locs, geom_name='circle', r=R)
    pm = model.PerceptionModel(targets, (0, 0), 0,
                               neural_angle_dist='lin_cutoff', angle_weight='lin_cutoff',
                               a_warp=A_WARP, b_warp=B_WARP,
                               a_weight=A_WEIGHT, b_weight=B_WEIGHT)
    return model.NeuralBandModel(pm, T=T, K=K)


def run_walkers(nm, pool=None, std=STD, v=V):
    seeds = np.random.SeedSequence(SEED).spawn(REPETITIONS)
    args = [(n, seeds[n], (0.0, 0.0), 0.0, 0.3 + 0j, DT, v, std, 0.0,
             NOISE_EXP, R_EXP, MAX_STEPS, TARGET_TOL) for n in range(REPETITIONS)]
    res = pool.map(nm._simulate_one_walk, args) if pool else [nm._simulate_one_walk(a) for a in args]
    return [w for w, _ in res]


def endpoint_split(walks):
    counts = np.zeros(len(target_locs), int)
    for w in walks:
        counts[int(np.argmin(np.linalg.norm(target_locs - w[:, -1], axis=1)))] += 1
    return counts


def density(walks, xlim=XLIM, ylim=YLIM, bins=220, blur=2.5):
    xs = np.concatenate([w[0] for w in walks])
    ys = np.concatenate([w[1] for w in walks])
    H, _, _ = np.histogram2d(xs, ys, bins=bins, range=[list(xlim), list(ylim)])
    H = gaussian_filter(H.T, blur)
    if H.max() > 0:
        H = np.sqrt(H / H.max())   # sqrt: compress the origin-convergence peak, show prongs
    return H, (xlim[0], xlim[1], ylim[0], ylim[1])


def main():
    nm = build_model()
    with Pool(10) as pool:
        walks = run_walkers(nm, pool=pool)
    counts = endpoint_split(walks)
    print('endpoint split (center, upper, lower):', counts.tolist(),
          ' -> outer fraction %.2f' % (counts[1:].sum() / counts.sum()))

    fig, ax = plt.subplots(1, 2, figsize=(11, 6))
    for w in walks:
        ax[0].plot(w[0], w[1], 'k', alpha=0.45, lw=0.8)
    nm.percep_model.targets.plot_targets_to_axis(ax[0])
    ax[0].set_xlim(*XLIM); ax[0].set_ylim(*YLIM); ax[0].set_aspect('equal')
    ax[0].set_title('locust tracks  (K=%.1f, T=%.2f, a_warp=%.2fpi)' % (K, T, A_WARP/pi))
    img, extent = density(walks)
    ax[1].imshow(img, extent=extent, origin='lower', aspect='equal')
    ax[1].set_title('locust walker density (cf. godm_heatmap_locust3.png)')
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'three_target_locust.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
