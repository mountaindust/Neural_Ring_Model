"""Three-target random walkers on the NeuralBandModel -- LOCUST geometry.

Three circle targets 40 degrees apart at radius 3 (the fly LAYOUT scaled by 3/5,
center target at (3,0)), but each of radius 0.1 (diameter 0.2 -- the LOCUST target
size from the paper's supplementary, much smaller than the flies' diameter-1
targets). The walker starts at the origin facing the center target and produces a
double bifurcation: an up/down split near x~0.9, then a split toward the individual
targets near x~2, reaching all three -- with tight, loop-free tracks that match the
empirical occupancy heatmap (godm_heatmap_locust3.png).

The smaller targets change the angular extent, so the warp is re-tuned for this
geometry (it is NOT identical to the fly script -- the different target sizes break
the scale-invariance). Bifurcation LOCATIONS are set by the warp (a_warp/b_warp)
and the weight window (a_weight/b_weight), independent of K, std, v, and the noise
exponents. See walker_analysis/three_target_analysis.md for the mechanism and
walker_analysis/gated_pq_analysis.md for the noise-exponent analysis.

Tuning summary (this layout, radius 3, target radius 0.1):
  - a_warp = 0.46*pi : first bifurcation at x~0.9.
  - b_warp = 0.90*pi : warp saturates +-162 deg ego (~+-18 deg rear blind spot);
                       second bifurcation at x~2.
  - a_weight/b_weight = 0.40*pi / 0.80*pi : near-panoramic vision.
  - GATED noise (decoupled R^p drift / (1-R)^q gate, see gated_pq_analysis.md):
      noise_exp (q) = 2 : steep gate -> noise->0 on commitment (clean homing).
      R_exp (p)     = 3 : drift exponent decoupled up from the gate; cancels the
                          steep gate's center bias and makes the split dt-robust.
      std (sigma)   = 2.5 : de-skew strength, set to MATCH THE EMPIRICAL HEATMAP
                          (godm_heatmap_locust3.png). Lower sigma -> more
                          center-dominant; higher -> more even.
  - v = 0.3, target_tol = 0.25 : the tiny r=0.1 targets have a small capture
      cross-section, so the 0.25 arrival radius (capture on entry to within 0.35 of
      center) plus the low-v tight turns give ZERO loops (verified: angular sweep
      < 2.5 rad for every walker, capture 1.0).
  - dt=0.05 : with R_exp=3 the drift step K*R^3*sin*dt <= 0.1 rad; the high p makes
              the ratio dt-robust. v/std do not constrain dt.

Run:  python walker_analysis/three_target_locust.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decision_model as model

pi = np.pi

# Three targets 40 deg apart at radius 3 (fly layout scaled by 3/5).
_a = np.radians(40.0)
target_locs = np.array([[3.0,            0.0],
                        [3.0*np.cos(_a),  3.0*np.sin(_a)],
                        [3.0*np.cos(_a), -3.0*np.sin(_a)]])
R = 0.1  # target radius: locusts use diameter-0.2 targets (per the supplementary).

# --- model knobs (warp re-tuned for r=0.1; NOT shared with the fly script) ---
A_WARP, B_WARP = 0.46*pi, 0.90*pi      # first bif ~0.9, second ~2.0
A_WEIGHT, B_WEIGHT = 0.40*pi, 0.80*pi  # near-panoramic visual field
K, T = 2.0, 0.2                        # do NOT move bifurcation locations
# Gated noise (see module docstring / gated_pq_analysis.md):
NOISE_EXP = 2.0                        # q: gate steepness -> no circling/loops
R_EXP = 3.0                            # p: drift exponent -> de-skew + dt-robust
STD, V, DT = 2.0, 0.2, 0.05            # sigma=2.0: tighter spine to match the tight
#   locust heatmap; v=0.2 -> turn radius ~v/(K*R)~0.1 ~ the r=0.1 target, so an
#   off-axis approach curves INTO the tiny target instead of past it.
TARGET_TOL = 0.2                       # arrival radius kept small (capture within
#   0.3 of center) so tracks reach the tiny targets without a stop-short crescent.
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
                        title='Locust: three targets (diam 0.2), double bifurcation')
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-2.6, 2.6)
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'three_target_locust.png')
    fig.savefig(outpath, dpi=120, bbox_inches='tight')
    print('wrote', outpath)
    plt.show()
