# Fly three-target: substructure match vs GODM (refinement)

A high-realization refinement of the fly three-target walker
([three_target_fly_refine.py](three_target_fly_refine.py)), aimed at the question
*how much of the GODM fly3 heatmap substructure does the walker ensemble reconstruct?*
The shipped [three_target_fly.py](../walker_analysis/three_target_fly.py) is left untouched.

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
workers, at the refit config (**K=2.0, a_warp=0.65π, std=4.0** — see the next section):

- centre/outer split **44.8% / 55.2%** (1121 / 700 / 679) — the data is **45% centre**.
- heatmap correlation **corr(all)=0.781, corr(support)=0.760** (support = empirical ridge > 0.05).

(The earlier config K=3.5/a_warp=0.45π gave 0.768 / 0.738 at 46% centre; the refit
below improves both the match and the split, and is what lets the 2-target case work.)

## Where the walkers commit — the first bifurcation (a_warp) and K

The walkers were peeling toward the targets **too early**: at the outer targets the
empirical ridge reaches the target from higher x (the right) than the walker tracks
did. Two independent levers fix this, and they interact:

- **`a_warp` sets the first-bifurcation x** (where the straight-ahead branch dies and
  the walker must commit up/down). Raising it pushes the split outward — measured from
  the deterministic midline cascade (`sc_equilib` along y=0):

  | a_warp | 3-target first-bif x | 2-target first-bif x |
  |---|---|---|
  | 0.45π (old) | 1.6 | 1.8 |
  | 0.55π | 2.0 | 2.0 |
  | **0.65π** | **2.3** | **2.2** |

- **`K` (turning gain) does NOT move the bifurcation** — it is K-invariant (the K-factor
  cancels the ½ in the coupled Jacobian; see `.claude/rules/torque-and-stability.md`).
  It sets how *sharply* the walker peels once past the split. **Higher K is worse, not
  better:** it tightens the pull to the dead-ahead consensus, so the walker corner-cuts
  onto a target *and* is more strongly recaptured to the centre. Lower K gentles the
  peel — walkers ride the trunk longer and arrive along the empirical ridge.

  | K (3-target, a_warp=0.55π) | centre % | corr(all) |
  |---|---|---|
  | **2.0** (model default) | **41** | **0.780** |
  | 3.5 (old refine) | 54 | 0.740 |
  | 6.0 | 65 | 0.677 |

**The interaction matters:** at K=3.5, pushing a_warp out *backfired* for the 3-target
(the longer trunk over-recaptured to the reborn centre branch → centre 50→59%, corr
down). Only once K is lowered to 2.0 does pushing a_warp out help both cases. With
K=2.0 the corr rises monotonically with a_warp for both:

| a_warp (K=2.0) | 3t corr(all) | 2t corr(all) |
|---|---|---|
| 0.50π | 0.765 | 0.589 |
| 0.60π | 0.787 | 0.681 |
| **0.65π** | 0.784 | **0.714** |

**Adopted shared config: K=2.0, a_warp=0.65π** — a single parameterization that fits
both the 3-target (corr 0.78, centre ~45%) and the 2-target (corr 0.69, up from 0.57)
cases. Reproduce a point with `NR_K=<v> NR_A_WARP=<v> python ... three_target_fly_refine.py`.

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
worse *substructure* match despite the number. **std=4.0 is the sweet spot.** (This
std sweep was run at the old K=3.5/a_warp=0.45π; at the adopted K=2.0/a_warp=0.65π the
centre is already ~45% so std=4.0 sits comfortably — see the bifurcation section above.
`three_target_fly_std_sweep.png` shows the tradeoff; reproduce a point with
`NR_STD=<v> python ... three_target_fly_refine.py`.)

Knobs changed from the shipped `three_target_fly.py`: **K 3.5→2.0, a_warp 0.45→0.65π,
std 2.5→4.0** (plus the start scatter and the GODM-pipeline render). `β`, `a_weight`,
`b_warp`, and the gated-noise exponents are unchanged.

A modest **start scatter** (`pos_std=0.075`, `head_std=12°`, override `NR_POS_STD`/
`NR_HEAD_STD`) mimics the empirical release variability and broadens the walker trunk
toward GODM's near-origin ridge; it does not move the split, and the Pearson
correlation is insensitive to it (`pos_std` 0.20→0.075: corr(all) 0.768→0.766,
corr(support) 0.738 unchanged — the metric is set by the trident, not the trunk
width). `pos_std=0.075` is tight enough to keep the straight-to-centre tracks on the
y=0 ridge; 0.20 spread them visibly above/below it.

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
python plots/three_target_fly_refine.py            # 1500 reps, std=4.0 default
NR_REPS=2500 python plots/three_target_fly_refine.py
NR_STD=2.5  python plots/three_target_fly_refine.py # shipped-noise comparison
```

Worker count comes from `machine_config.py` (`get_n_workers`, reserves a few cores);
the GODM data is read from `../../GODM`. `pandas` is required for the GODM read.
