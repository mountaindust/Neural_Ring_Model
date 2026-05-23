# Weighting vs warping in NBM — exploratory analysis

**Date:** 2026-05-22
**Status:** exploratory; no code changes proposed yet.

## Question

`PerceptionModel.neural_weight` plays two roles in `NeuralBandModel` (NBM):

1. **Weighting** — `neural_weight(theta)` is integrated over each target's
   visible angular interval to produce `rho` (the perceptual mass that target
   contributes to gamma).
2. **Warping** — the integral of `neural_weight` is used (CDF-like) to map
   egocentric angles to neural angles via `get_neural_angle`.

The flag `weight_angle_only=True` keeps role (2) but switches off role (1)
(integration uses uniform weight). The hypothesis under test:

> Does the warping alone do most of the work? In particular, is the
> "symmetry-breaking" that originally motivated the front-bias weighting
> already provided by the front-stretching effect of the warp?

## Headline finding

Across the standard two-circle symmetric setup
(`targets at (4.33, ±2.5), r=0.5`, observer in `[0,6] × [-3.5, 3.5]`,
`stability_criterion='coupled'`, K=1, T=0.2):

- The bifurcation rasters with and without weighting agree on
  **~85-97% of pixels** depending on how peaked the weighting function is.
- Most disagreement is **boundary jitter** (a 1-2 pixel shift in saddle-node
  arcs on the left near the targets).
- One **genuine structural difference** appears as two "ears" of extra
  bistability on the right side of the figure (`x ≈ 4-6, y ≈ ±2`),
  where FULL weighting produces a 2-stable region and ANGLE-only produces
  only a 1-stable region.

Ear area scales monotonically with weighting peakedness:

| weighting parameterization      | ear area (sq. units of x,y) |
|---------------------------------|------------------------------|
| cutoff `a=0`, `b=π` (sharpest)  | 2.12                         |
| vonMises `k=0.9`                | 2.46                         |
| vonMises `k=0.5`                | 1.72                         |
| cutoff `a=π/3`, `b=π` (mild)    | 1.01                         |

See `ears_figure.png` for the per-row layouts and `ear_diagnostic.png` for
the mechanism.

## Why the ears exist (corrected interpretation)

Initial intuition was wrong. The "extra stable equilibrium" in the ear is
**not** a compromise heading; it's a *commitment* to the **far** target that
ANGLE-only mode loses.

Worked example at observer = (5.0, 2.0) with cutoff `a=0, b=π`:

- target 0 at allo direction +143.3°, distance 0.84, visual extent **73.5°**
- target 1 at allo direction −98.5°, distance 4.55, visual extent **12.6°**

Self-consistent stable equilibria found by `gamma_equilib(focal_angle=True)`:

| mode  | heading_allo | facing       | stable? |
|-------|--------------|--------------|---------|
| FULL  | +143.3°      | target 0     | yes     |
| FULL  | −98.5°       | target 1     | yes     |
| ANGLE | +143.3°      | target 0     | yes     |

ANGLE-only loses the target-1 equilibrium. The mechanism (panel c of
`ear_diagnostic.png`):

At the hypothesized heading −98.5° (facing target 1), target 0 sits at
egocentric ≈ −118° (behind/left of the observer). Its arc length is still
the same ~73°, so:

- ANGLE-only assigns ρ = (0.853, 0.147) — the close target dominates by ~5.8×
  *purely from visual extent*, even though it's behind the observer.
  `dgamma_dt(R+0j, θ=−98.5°)` is negative for all R > 0; never crosses zero;
  the consensus is pulled back toward target 0 regardless of R. No
  self-consistent equilibrium at this heading.
- FULL weighting evaluates the cutoff at |ego| ≈ 118° (w ≈ 0.02), squashing
  the off-axis target's contribution. ρ rebalances to (0.575, 0.425).
  `dgamma_dt` now crosses zero at R\* ≈ 0.425. Self-consistent equilibrium.

So the ears occupy observer positions where one target is much closer/bigger
than the other, *and* where committing to the far target would put the close
target off-axis behind. There, **front-bias weighting suppresses the close
target's pull enough to let a far-target commitment be self-consistent.**
Visual extent alone (the only thing ANGLE-only keeps) cannot do this — the
big target wins no matter where the observer hypothesizes facing.

## Two clarifications worth recording

1. **`weight_angle_only=True` does NOT make ρ equal across targets.** ρ is
   the integral of (uniform) weight over each target's visible angular
   extent. Closer targets have larger extent and therefore larger ρ. At
   (5, 2.0) the close target has ~5.8× more ρ than the far target under
   ANGLE-only — purely from being close.
2. **The single ANGLE-only equilibrium in the ear region is not a
   compromise.** It points exactly at the close/big target. The bistability
   loss is the disappearance of the *far*-target option, not the collapse
   of two committed options into a midway one.

## Reframing the original hypothesis

The warping-does-what-weighting-does hypothesis is correct **wherever
visual extent and front-bias agree** — most of the standard sweep, including
all the symmetric-observer-on-y=0 regimes. In those positions, the close
target is also the most-frontal target, and the two effects are redundant.

The hypothesis breaks **where extent and front-bias disagree** — observer
positions where the close/big target is off-axis at the candidate heading.
There, weighting changes the equilibrium count in a real qualitative way
(rescues the far-target commitment) that warping alone cannot reproduce.

Whether that asymmetric regime is worth keeping the weighting machinery for
is the design call. Options sketched in the original conversation:

- Remove the weighting role entirely. Loses the far-target rescue in
  asymmetric configurations but the model is simpler.
- Decouple `neural_weight` from the rho-integration: let `neural_weight`
  drive only the angle warping (which is the conceptually clean role for
  "density of the neural band"), and use uniform weighting *or* a separate,
  much milder front-bias for ρ. Preserves the warping interpretation while
  letting the weighting strength be tuned independently to whatever level
  asymmetric-configuration behavior requires.

## Delta targets — same structure, threshold shifts outward

Re-running the same observer sweep with `geom_name=None` (delta targets)
instead of circles gives a quite different qualitative picture from the
circle case — but the difference between FULL and ANGLE-only is much less
dramatic than the circle ears suggested.

| weighting parameterization | disagreement (delta) | sign of FULL−ANGLE |
|---------------------------|----------------------|--------------------|
| cutoff `a=0`, `b=π`        | 1.9%                 | FULL has fewer     |
| vonMises `k=0.9`           | 8.7%                 | FULL has fewer     |
| vonMises `k=0.5`           | 5.5%                 | FULL has fewer     |
| cutoff `a=π/3`, `b=π`      | 0.0%                 | (essentially identical) |

See `delta_sweep_comparison.png`. The **qualitative bifurcation structure
is the same in both modes** — same 1-stable / 2-stable / 3-stable regions
in the same nested arrangement around the targets. The only difference is
that **the bifurcation arcs separating these regions are pushed slightly
outward from the targets under FULL weighting**, so a band of observer
positions that are 3-stable under ANGLE-only is already 2-stable under
FULL. No "ears"; no new structural features.

### The shift is a heading-sensitivity effect

FULL weighting makes the dynamics more sensitive to where the observer is
hypothesized to face: targets directly in front get weighted significantly
more than targets to the side. Under ANGLE-only, each target's `rho`
contribution is fixed at `1/N_visible` regardless of heading, so heading
asymmetries propagate only through the warping. So as the observer moves
closer to the targets (and the angular separation between them grows),
the moment at which a small heading perturbation creates a "critical"
discrepancy between the two targets' contributions arrives at a smaller
separation under FULL than under ANGLE. The midway / compromise heading
loses stability sooner; the corresponding bifurcation arc sits further
out in physical space.

`delta_threshold_shift.png` shows this directly. Walking the observer
along the y=0 axis from x=0 toward the target line at x=4.33 (so the
half-angle α between targets seen from the observer grows from 30° to
90°), the third (heading-coupled) eigenvalue of the midway equilibrium's
3×3 Jacobian crosses zero at:

- **FULL: x = 0.534 (α ≈ 33.4°)** — midway destabilizes here
- **ANGLE: x = 0.783 (α ≈ 35.2°)** — midway destabilizes here

The disagreement band x ∈ (0.534, 0.783) on the y=0 axis matches the
visible disagreement strip in the sweep raster.

### Why FULL crosses sooner

The midway equilibrium (heading exactly halfway between the two targets,
so both at egocentric ±α) is symmetric: `w(+α) = w(−α)`, so `rho` is
(0.5, 0.5) in both modes, and the equilibrium R\* is identical in both
modes too. The eigenvalue analysis at observer (0.76, 0) shows the
difference:

- FULL eigenvalues: (−0.84, −0.49, **+0.38**) — destabilized
- ANGLE eigenvalues: (−0.84, −0.054 ± 0.175i) — stable spiral
- `dρ_target0/dθ` at midway: FULL = +0.258, ANGLE = exactly 0

ANGLE's `dρ/dθ = 0` is structural: when neural_weight is ignored, ρ
depends only on which targets are visible, not on where they sit, so a
small heading perturbation doesn't shift any target's contribution.
Under FULL, perturbing heading toward target 0 moves it closer to
egocentric 0° where w is larger, boosting its ρ at the expense of
target 1's. That's an extra positive-feedback term in the third row of
the Jacobian, which kicks one eigenvalue positive and destabilizes midway
earlier (smaller α, smaller x).

So the front-bias **weighting** function is doing roughly what it was
designed to do: making the model more decisive in physical space by
amplifying the contribution of frontal targets. The model produces a
"decision" (loss of the compromise option) sooner as the observer
approaches the targets, by exactly this mechanism.

## Combined picture across geometries

| geometry | direction of disagreement | where it lives | what's happening |
|----------|---------------------------|----------------|--------------------|
| delta    | FULL < ANGLE              | inner arc (between observer and targets) | same structure as ANGLE; bifurcation arcs pushed slightly outward → midway destabilizes sooner |
| circle   | FULL > ANGLE              | behind targets (asymmetric observer positions) | qualitatively new structural feature: front-bias suppresses a close-but-off-axis target's extent-dominance, rescuing the far-target commitment |

The delta picture is more or less the "graded amplifier" interpretation of
the weighting: ANGLE-only already supports all the equilibria; FULL just
makes their bifurcations happen at less extreme observer positions. The
circle picture is something stronger: the weighting opens up a regime of
asymmetric bistability that ANGLE-only cannot produce at all. The
asymmetry is because circle ρ also has a visual-extent component, and the
extent ratio can dominate so heavily under ANGLE-only that no value of R
can make a far-target commitment self-consistent — whereas the front-bias
under FULL squashes the extent-dominant close target enough to let the
far-target equilibrium exist.

The decoupled-weighting option (let `neural_weight` drive the warp only;
use a separate, milder front-bias for ρ — or none) is still attractive:
it would let the front-bias strength be tuned independently of the
warping shape, which is the conceptually clean role for "neural-band
density." Removing the front-bias entirely (true ANGLE-only) is more
conservative for deltas than for circles — for deltas it costs only a
small shift in where bifurcation arcs sit, but for circles it gives up
the asymmetric-position bistability that produces the ears.

## What's NOT covered here

- **3+ symmetric targets with observer at the center** (the configuration
  the user said originally motivated weighting). A first pass found that
  both modes produce similar equilibrium counts at the exact center, but the
  brentq scan in `gamma_equilib` was missing 3-fold-symmetric copies, so
  the comparison was not conclusive. Worth a focused follow-up before any
  decision about removing the weighting.
- **Walker dynamics, Hopf islands, basin sizes.** We've only looked at
  counts and stability of self-consistent equilibria. The ANGLE-only delta
  midway equilibrium is a slowly-decaying spiral (eigenvalues at
  −0.054 ± 0.175i — a Hopf bifurcation could be nearby). Worth checking
  whether deltas with ANGLE-only develop Hopf-island behavior similar to
  the vonMises `k=0.55` circle case in `VM_bifurcations/VERDICT.md`.
- **Capsule targets.** Likely lie between deltas and circles depending on
  `l`/`w`; not characterized here.

## Reproduction

- `ears_figure.png` is built from cached `_cache_*.npz` files in
  `bifurc_plots/` (companion scripts `neural_weight_sweep.py` and
  `neural_weight_sweep_angle_only.py`). The build script is inline in the
  conversation transcript that produced this folder; not re-saved here
  because the raster data is already cached upstream.
- `ear_diagnostic.png` runs three diagnostics on the single observer
  position `(5.0, 2.0)` with cutoff `a=0, b=π`. Reproduction script
  is in the same transcript.
- `delta_sweep_comparison.png` was computed from scratch (no companion
  scripts existed for delta targets); rasters are cached in
  `_delta_rasters.npz` next to the PNG. Reproduction script in the
  transcript.
- `delta_diagnostic.png` runs the same three-panel mechanism layout as
  `ear_diagnostic.png` but for deltas at observer (0.76, 0) with
  vonMises `k=0.9`.
- `delta_threshold_shift.png` walks the observer along the y=0 axis and
  reports the max-real-part eigenvalue of the midway equilibrium's
  Jacobian and the `dρ/dθ` magnitude, for both modes. The crossings of
  the eigenvalue through zero define the FULL/ANGLE disagreement band
  on the y=0 axis. Reproduction script in the transcript.
