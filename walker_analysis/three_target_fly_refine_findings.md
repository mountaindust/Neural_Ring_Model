# Fly three-target: substructure match vs GODM (refinement)

A high-realization refinement of the fly three-target walker
([three_target_fly_refine.py](three_target_fly_refine.py)), aimed at the question
*how much of the GODM fly3 heatmap substructure does the walker ensemble reconstruct?*
The shipped [three_target_fly.py](three_target_fly.py) is left untouched.

## The decisive change was the *measurement*, not the model

The original `three_target_fly.py` rendered the walker ensemble as a plain
`sqrt`-compressed 2-D histogram. That render is swamped by the origin start-point
peak (every walker launches from exactly `(0,0)`), so the branches read as faint
prongs and the bifurcation structure is invisible — which is why the match to the
empirical heatmap looked unclear.

The empirical heatmap is **not** a plain histogram. `godm_heatmaps.py` reproduces the
paper's pipeline: a window sliding over within-trajectory time, each window's 2-D
histogram **per-window-normalised**, Gaussian-blurred, and max-projected. That
per-window normalisation is exactly what renders the trunk ridge and the post-split
branches at uniform contrast instead of a blob.

`three_target_fly_refine.py` runs the walker tracks through that **same** pipeline
(reusing `godm_heatmaps._max_project`), on the **same** extent / target geometry /
orientation / blur / y-mirror as `fly3`. The walker's step index is the time proxy
(constant speed `v` ⇒ step index ∝ arc length — the near-constant-speed assumption
the GODM render already makes). The result is pixel-comparable to
`compute_heatmap('fly3')`.

**Once rendered this way, the substructure reconstructs well:** broad trunk ridge →
first bifurcation at x≈2 → a balanced trident reaching all three targets. See
`three_target_fly_refine.png` (4 panels: empirical | walker | overlay | tracks).

## Quantitative match

Both heatmaps live on the identical 500×500 grid, so a direct Pearson correlation is
meaningful (`similarity()` in the script). Final run, **2500 realizations** on 32
workers, std=4.0:

- centre/outer split **46.3% / 53.7%** (1157 / 684 / 659) — the data is **45% centre**.
- heatmap correlation **corr(all)=0.77, corr(support)=0.74** (support = empirical ridge > 0.05).

The deterministic midline cascade pins the model's bifurcations at **x≈1.8** (the
straight-ahead centre branch dies → up/down split) and **x≈2.55** (centre reborn +
outer branches appear → centre-vs-outer decision), bracketing the empirical trunk
split at x≈2.

## What controls the split at high N — std, not a_weight

`three_target_findings.md` reported `a_weight=0.20π` as ~45–49% centre, but that was
an 80-walker estimate. **At N≥1000 the centre fraction is robustly ~55% at std=2.5**
(more centre-biased than the low-N estimate), and **`a_weight` does NOT move it** in
the 0.13–0.20π range (saturated, and lowering it slightly *worsens* the heatmap corr).

The lever that works at high N is **std** (heading-noise intensity):

| std | centre % | corr(support) | note |
|---|---|---|---|
| 2.5 (shipped) | ~55% | 0.73 | clean trident, centre over-bright |
| **4.0 (refined default)** | **~46–50%** | **0.74** | balanced trident — best match to data + heatmap |
| 6.0 | ~36% | 0.76 | over-corrected: outer-dominant, centre trunk fades below the data |

std=6.0 has a marginally higher correlation, but that comes from diffuse
envelope-filling — its centre trunk fades *below* the empirical brightness, so it is a
worse *substructure* match despite the number. **std=4.0 is the sweet spot.** This is
the only knob changed from the shipped config; `K`, `T`, warp, `a_weight`, and the
gated-noise exponents are identical. (`three_target_fly_std_sweep.png` shows the
tradeoff; reproduce any single point with `NR_STD=<v> python ... three_target_fly_refine.py`.)

A modest **start scatter** (`pos_std=0.20`, `head_std=12°`, override `NR_POS_STD`/
`NR_HEAD_STD`) mimics the empirical release variability and broadens the walker trunk
to match GODM's broad near-origin ridge; it is documented not to move the split.

## Residual mismatches (the honest limits)

- **The nested-chevron "beads"** along GODM's midline trunk are not reproduced. These
  are almost certainly a per-window-normalisation artifact of the *sparse* empirical
  data (125 trajectories, integer-second time bins); the dense, finely time-resolved
  walker ensemble produces a smooth ridge instead. Not a dynamics mismatch.
- The walker trident interior is slightly **fuller / more diffuse** than GODM, and the
  empirical outer branches curve a touch higher toward the corners while the walker's
  are straighter. Minor.

## Reproduce

```
python walker_analysis/three_target_fly_refine.py            # 1500 reps, std=4.0 default
NR_REPS=2500 python walker_analysis/three_target_fly_refine.py
NR_STD=2.5  python walker_analysis/three_target_fly_refine.py # shipped-noise comparison
```

Worker count comes from `machine_config.py` (`get_n_workers`, reserves a few cores);
the GODM data is read from `../../GODM`. `pandas` is required for the GODM read.
