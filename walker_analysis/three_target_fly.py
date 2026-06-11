"""Three-target random walkers on the NeuralBandModel -- FRUIT FLY geometry.

Three circle targets 40 degrees apart at radius 5, each of radius 0.5 (diameter 1;
the fruit-fly target size from the paper's supplementary). The walker starts at the
origin facing the center target and produces a double bifurcation: an up/down split
near x~1.5, then a split toward the individual targets near x~3.5, reaching all
three -- with tight, loop-free tracks that match the empirical occupancy heatmap
(godm_heatmap_fly3.png).

Bifurcation LOCATIONS are set by the warp (a_warp/b_warp) and the visual-field
weight window (a_weight/b_weight); they are independent of K, std, v, and the
noise exponents (so those are free to tune the split). See
walker_analysis/three_target_analysis.md for the mechanism (the reborn-center-
branch recapture) and walker_analysis/gated_pq_analysis.md for the noise-exponent
analysis the gated tuning below is based on.

Tuning summary (this layout, radius 5, target radius 0.5):
  - a_warp = 0.47*pi : wide neural plateau -> first bifurcation at x~1.5.
  - b_warp = 0.92*pi : warp saturates +-166 deg ego, i.e. a ~+-14 deg rear BLIND
                       SPOT (a plausible fruit-fly value); also sets the second
                       bifurcation to x~3.5.
  - a_weight/b_weight = 0.40*pi / 0.80*pi : near-panoramic vision.
  - GATED noise (decoupled R^p drift / (1-R)^q gate, see gated_pq_analysis.md):
      noise_exp (q) = 2 : steep gate -> noise->0 on commitment (clean homing).
      R_exp (p)     = 3 : drift exponent decoupled up from the gate; cancels the
                          steep gate's center bias and makes the split dt-robust.
      std (sigma)   = 2.5 : de-skew strength, set to MATCH THE EMPIRICAL HEATMAP
                          (godm_heatmap_fly3.png) rather than a fixed ratio. Lower
                          sigma -> more center-dominant; higher -> more even.
  - v = 0.3 : low speed shrinks the turn radius (~v/(K*R)) so an off-axis approach
      curves into the target instead of swinging wide -> ZERO loops around the
      targets (verified: angular sweep < 2.5 rad for every walker, capture 1.0).
  - dt=0.05 : with R_exp=3 the drift step K*R^3*sin*dt <= 0.1 rad (R<=1), well
              within the Euler margin; the high p makes the ratio dt-robust. v/std
              do not constrain dt (the diffusion term is the exact sqrt(dt) Wiener
              increment; heading accuracy is governed by K*dt).

Run:  python walker_analysis/three_target_fly.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decision_model as model

pi = np.pi

# Three targets 40 deg apart at radius 5 (exact ipynb locations).
target_locs = np.array([[5.0000,  0.0000],
                        [3.8302,  3.2139],
                        [3.8302, -3.2139]])
R = 0.5  # target radius: flies use diameter-1 targets (per the paper's supplementary;
#          the main figures make all targets look equal-sized -- they are not).

# --- model knobs (warp re-tuned for r=0.5; see _check / gated_pq_analysis.md) ---
A_WARP, B_WARP = 0.47*pi, 0.92*pi      # first bif ~1.5, second ~3.5; ~+-14deg blind spot
A_WEIGHT, B_WEIGHT = 0.40*pi, 0.80*pi  # near-panoramic visual field
K, T = 2.0, 0.2                        # do NOT move bifurcation locations
# Gated noise (see module docstring / gated_pq_analysis.md):
NOISE_EXP = 2.0                        # q: gate steepness -> clean homing
R_EXP = 3.0                            # p: drift exponent -> de-skew + dt-robust
STD, V, DT = 2.5, 0.3, 0.05            # sigma -> heatmap match; v=0.3 -> tight turns
TARGET_TOL = 0.2                       # arrival radius (r=0.5 targets already catch
#   most approaches via trajectory intersection; this just trims fly-by overshoot).
REPETITIONS, MAX_STEPS, SEED = 60, 4000, 3


def build_model():
    targets = model.Targets(locs=target_locs, geom_name='circle', r=R)
    pm = model.PerceptionModel(targets, (0, 0), 0,
                               neural_angle_dist='lin_cutoff', angle_weight='lin_cutoff',
                               a_warp=A_WARP, b_warp=B_WARP,
                               a_weight=A_WEIGHT, b_weight=B_WEIGHT)
    nm = model.NeuralBandModel(pm, T=T, K=K)
    nm.rng = np.random.default_rng(seed=SEED)
    return nm


if __name__ == "__main__":
    nm = build_model()
    fig, ax = plt.subplots(figsize=(8, 6))
    with Pool(10) as pool:
        nm.plot_walkers(dt=DT, v=V, std=STD, noise_exp=NOISE_EXP, R_exp=R_EXP,
                        target_tol=TARGET_TOL, repetitions=REPETITIONS,
                        max_steps=MAX_STEPS, alpha=0.5, pool=pool, ax=ax,
                        title='Fruit fly: three targets (diam 1), double bifurcation')
    ax.set_xlim(-0.5, 5.6); ax.set_ylim(-4, 4)
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'three_target_fly.png')
    fig.savefig(outpath, dpi=120, bbox_inches='tight')
    print('wrote', outpath)
    plt.show()
