# walker_analysis

Walker (physical-space, x–y) random-walk analysis for `NeuralBandModel`.

Here the noise is angular heading noise and the object of
study is the walker's trajectory in physical space via
`NeuralBandModel.plot_walkers`.

## Model state documented here

Heading torque is the **half-angle law** in the neural consensus angle:
`dθ/dt = K·R·sin(arg(γ)/2)` (`arg(γ)` is the neural consensus angle, which
is `0` straight ahead and `±π` facing away — no inverse-warp mapping).
`plot_walkers` also has a blind-spot search (`walk_std`, default π/2) that
turns a walker with no visible target into a diffusive search until a target
re-enters view.

These figures use the shipped constant-noise default (`noise_exp=0`, i.e.
plain `σ·dW` with `σ=std=0.1`). The model has since gained opt-in gated-noise
machinery (`noise_exp`, `R_exp`, `cos(Θ/2)` modulation) that these figures
intentionally do not exercise.

## Contents

- `rndwalk_figures_orig.py` — regenerates the representative K-series figures below.
- `godm_heatmaps.py` — reproduces the empirical (x, y) occupancy heatmaps from
  Sridhar et al. 2021 for side-by-side comparison with the model (see
  "Empirical comparison heatmaps" below). This is the empirical-data engine; the
  publication scripts in `../plots/` import it.
- `three_target_analysis.md` — why three-target walkers consistently pick the
  middle target (the reborn-center-branch recapture); the early (~80-walker)
  split-tuning exploration. **Its "`a_weight` is the lever" conclusion is superseded
  for the fly** by the high-realization refit (see `../plots/`).
- `three_target_fly.py` / `three_target_locust.py` — three-target walker quick-look
  models on the GODM geometry (fly 40°/radius 5/r=0.5; **locust 35°**/radius 3/r=0.1).
  The fly was GODM-refit (**K=2, a_warp=0.65π, std=4** → ~45% centre); the locust is
  still the older `a_weight`/`std` tuning (a clean near-match — a documented tension).
  Run each directly (10-core pool).
- `three_target_findings.md` — the data match and the split levers. **At high
  realization count the levers are `a_warp` / `K` / `std`, not `a_weight`** (which is
  saturated); plus the locust clean-vs-outer tension.
- `birth_mechanism.md` / `skeleton_birth_analysis.py` — where the bifurcations are
  born/die on the y=0 cut (pitchfork-vs-saddle-node mechanism).

**Publication figures live in [`../plots/`](../plots/)** (moved out of here): the
GODM-substructure walker match (`fly_results.py` → `fly_results_{2,3}target.png`, fed
by the npz producers `three_target_fly_refine.py` / `two_target_fly_refine.py`) and the
deterministic decision-track skeletons (`decision_skeleton.py` →
`skeleton_{fly,locust}.png`, `branch_diagram_{fly,locust}.png`). Those scripts import
`decision_model`, `parallel_config`, and this directory's `godm_heatmaps`.

Run:

```
python walker_analysis/rndwalk_figures_orig.py
```

Each figure overlays the tracks of 30 walkers. Shared setup: observer
starts at (0,0) facing +x (east); cutoff warp `a=0, b=π` with **uniform**
weighting (`angle_weight=None`); `std=0.1`, `v=1`, `dt=0.1`; seed 0;
`max_steps=1500`. Targets are drawn as red stars. Two target layouts are used,
both at `x=4.33`: **FOUR** deltas at `y=±0.75, ±2.25` and **TWO** at `y=±2.5`;
circle variants give each target radius `r=0.5`.

The figures sweep the turning gain `K` to show how it shapes the approach. 

The K=4 and K=6 figures use a clamped view (`xlim=(-1,7)`, `ylim=(-5,5)`) so a
single wide track does not blow up the auto-scaled axes.

**K-series summary.** `K` sets how hard the walker turns toward its perceived
consensus direction. `K=2` (the default) turns in smoothly; larger `K` turns in
more sharply and can take wider curved excursions on the way, most visibly at
`K=10` with zero-radius (delta) targets. Across the whole sweep no walkers are
lost — finite-radius (circle) targets are, if anything, even more forgiving
because their capture radius absorbs near-misses.

## Empirical comparison heatmaps (Sridhar et al. 2021)

`godm_heatmaps.py` regenerates the (x, y) occupancy heatmaps from Figure 1 of
Sridhar, Li, Gorbonos et al. 2021, *The geometry of decision-making in
individuals and collectives* (PNAS) — **the density layer only**, with none of
the paper's overlays (black trajectory scatter, fitted bifurcation curves, red
target markers). The point is to put the empirical decision-bifurcation density
on the same target-centred (x, y) axes the model's walker ensembles use, so the
two can be compared directly.

Four panels, one PNG each (`godm_heatmap_<case>.png`):

| case | animal | targets | separation | distance |
|---|---|---|---|---|
| `fly2` | flies | 2 | 60° | 5 |
| `fly3` | flies | 3 | 40° | 5 |
| `locust2` | locusts | 2 | 45° | 2 |
| `locust3` | locusts | 3 | 35° | 3 |

Each shows the characteristic bifurcation: a single occupancy ridge along the
midline far from the targets that splits into branches toward each target as the
animal commits.

Run (all four, or one case):

```
python walker_analysis/godm_heatmaps.py
python walker_analysis/godm_heatmaps.py fly2
```

The script reads the GODM data/code repo at `../../GODM` (a sibling of this
project; override with `$GODM_DIR`). It is a faithful port of the analysis
notebooks' density computation — spatial discretisation, rotation into the
target-centred frame, end-near-a-target and duration filtering, then a
Gaussian-blurred per-window-normalised 2-D histogram max-projected over a
sliding within-trajectory time window. `cv2.GaussianBlur` is reproduced with
`scipy.ndimage.gaussian_filter` (OpenCV is not a dependency here). For
overlaying the model on these axes, call `compute_heatmap(case)`, which returns
`(img, extent, posts)`. The locust cases read ~60–100 large CSVs, so a full run
takes a few minutes.
