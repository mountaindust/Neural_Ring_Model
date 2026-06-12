"""Three-target FRUIT FLY walkers, tuned to the GODM data (Sridhar et al. 2021).

Three circle targets 40 deg apart at radius 5, target radius 0.5 -- positions and size
FIXED by the experiment. The walker starts at the origin and makes two sequential
binary decisions (up/down, then outer/centre), reaching all three targets. Tuning vs
the early prototype: lower T (sharper, cleaner commitment), higher K (tighter turns),
and a NARROW foveal weight window (a_weight) chosen so the centre/outer target split
matches the data (~45% centre). Bifurcation locations are set by the warp; a_warp keeps
the branches on the empirical heatmap ridge (godm_heatmap_fly3.png).

See three_target_findings.md for the data match, the split-tuning sweeps (a_weight is
the lever), and the matching deterministic skeleton (decision_skeleton.py).

Run:  python walker_analysis/three_target_fly.py
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
target_locs = np.array([[5.0000,  0.0000],
                        [3.8302,  3.2139],
                        [3.8302, -3.2139]])
R = 0.5

# --- retuned model knobs ---
A_WARP, B_WARP = 0.45*pi, 0.92*pi      # a_warp kept near the original so the consensus
#   branches sit on the empirical heatmap ridge; b_warp keeps the rear blind spot.
A_WEIGHT, B_WEIGHT = 0.20*pi, 0.80*pi  # narrow foveal weight window: tuned so the
#   centre/outer target split matches the data (~45% centre); a smaller a_weight
#   releases more walkers to the outer targets (uniform weight is the most
#   centre-biased). See three_target_findings.md.
K, T = 3.5, 0.10                       # higher K (tighter turns), lower T (cleaner)
# Gated noise (decoupled R^p drift / (1-R)^q gate):
NOISE_EXP, R_EXP = 2.0, 3.0
STD, V, DT = 2.5, 0.30, 0.05           # std set with a_weight for the ~45% centre split
TARGET_TOL = 0.20
REPETITIONS, MAX_STEPS, SEED = 80, 4000, 3

XLIM, YLIM = (-0.3, 5.3), (-3.6, 3.6)


def build_model():
    targets = model.Targets(locs=target_locs, geom_name='circle', r=R)
    pm = model.PerceptionModel(targets, (0, 0), 0,
                               neural_angle_dist='lin_cutoff', angle_weight='lin_cutoff',
                               a_warp=A_WARP, b_warp=B_WARP,
                               a_weight=A_WEIGHT, b_weight=B_WEIGHT)
    return model.NeuralBandModel(pm, T=T, K=K)


def run_walkers(nm, pool=None, std=STD, v=V, k=None):
    """Return a list of (2, n) trajectory arrays."""
    seeds = np.random.SeedSequence(SEED).spawn(REPETITIONS)
    args = [(n, seeds[n], (0.0, 0.0), 0.0, 0.3 + 0j, DT, v, std, 0.0,
             NOISE_EXP, R_EXP, MAX_STEPS, TARGET_TOL) for n in range(REPETITIONS)]
    res = pool.map(nm._simulate_one_walk, args) if pool else [nm._simulate_one_walk(a) for a in args]
    return [w for w, _ in res]


def endpoint_split(nm, walks):
    """Count which target each walker ends nearest (0=center, 1=upper, 2=lower)."""
    counts = np.zeros(len(target_locs), int)
    for w in walks:
        end = w[:, -1]
        counts[int(np.argmin(np.linalg.norm(target_locs - end, axis=1)))] += 1
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
    counts = endpoint_split(nm, walks)
    print('endpoint split (center, upper, lower):', counts.tolist(),
          ' -> outer fraction %.2f' % (counts[1:].sum() / counts.sum()))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for w in walks:
        ax[0].plot(w[0], w[1], 'k', alpha=0.45, lw=0.8)
    nm.percep_model.targets.plot_targets_to_axis(ax[0])
    ax[0].set_xlim(*XLIM); ax[0].set_ylim(*YLIM); ax[0].set_aspect('equal')
    ax[0].set_title('fly tracks  (K=%.1f, T=%.2f, a_warp=%.2fpi)' % (K, T, A_WARP/pi))
    img, extent = density(walks)
    ax[1].imshow(img, extent=extent, origin='lower', aspect='equal')
    ax[1].set_title('fly walker density (cf. godm_heatmap_fly3.png)')
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'three_target_fly.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
