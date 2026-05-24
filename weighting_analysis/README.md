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

Self-consistent stable equilibria found by `sc_equilib`:

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

## Delta + ANGLE-only: Hopf-unstable foci exist, but no limit cycle

Follow-up to the open question above about whether the ANGLE-only delta
midway spiral signals a nearby Hopf island like the vonMises `k=0.55`
circle case in [VM_bifurcations/VERDICT.md](../VM_bifurcations/VERDICT.md).

**Headline:** **No.** Hopf-unstable foci do appear in this configuration
(more than one might expect, in fact), but they are *not* the unique
attractor in any cell — every Hopf-positive cell coexists with at least one
coupled-stable focus, and long-time integration from the Hopf focus
escapes to the stable focus rather than orbiting a limit cycle. There is
no observable head-bobbing behavior.

### What was scanned

Same observer window `[0, 6] × [-3.5, 3.5]` and delta-target geometry as
the rest of this folder. Resolution 121×141 cells, multiprocessing pool.
For each cell, `NeuralBandModel.sc_equilib(focal_loc=fl,
stability_criterion='coupled')` returns all self-consistent equilibria;
the coupled 3×3 Jacobian is built by finite difference for each. Hopf
indicator = max real part among complex-pair eigenvalues, across all
non-saddle eqs in the cell.

All four weighting choices from the earlier delta sweep:

| weight                  | Hopf+ cells / 17061 | max Re(complex eig) |
|-------------------------|---------------------|----------------------|
| **vonMises k=0.9**      | 86                  | **+0.1741**          |
| **vonMises k=0.5**      | 57                  | +0.1493              |
| cutoff `a=0`, `b=π`     | 2                   | +0.0166              |
| cutoff `a=π/3`, `b=π`   | 0                   | (none)               |

vonMises ANGLE-only is the unambiguous winner: Hopf-unstable foci with
positive real parts that comfortably exceed the VM-circle island's max
(+0.082 from [VERDICT.md](../VM_bifurcations/VERDICT.md)). Cutoff
weighting barely shows it.

The Hopf-positive cells are spread broadly — `x ∈ [0.1, 6.0]`,
`|y| ∈ [0.5, 2.9]` — and are not confined to a thin curve like the
VM-circle Hopf loop. They line up roughly along the inner saddle-node
arcs of the bifurcation diagram, where new equilibria are being born.

`hopf_overview.png` (top row) shows the four Hopf-indicator rasters
with the level-zero contour drawn where it exists. Bottom row of the
same figure is a representative trajectory at the strongest Hopf+ cell
(see "No stable limit cycle is born" below).

### Crucial distinction from the VM/circle case

In the VM-circle Hopf island, the Hopf-unstable focus is the **only**
self-consistent equilibrium in the cell — coupled-stable count is 0, and
the stable limit cycle is the unique attractor.

In delta + ANGLE, **no cell has Hopf+ AND `n_stable=0` simultaneously**
in any of the four weightings. Concretely, at the strongest Hopf+ cell
(observer `(1.35, 2.40)`, vonMises k=0.9), there are three
self-consistent eqs:

| θ (deg) | R     | classification     | eigenvalues                                    |
|---------|-------|--------------------|-----------------------------------------------|
| −54.1°  | 0.480 | **Hopf-unstable focus** | (−0.981, **+0.174 ± 0.009j**)          |
| −28.4°  | 0.512 | stable focus            | (−0.973, −0.463 ± 0.453j)              |
| −3.4°   | 0.477 | saddle                  | (−0.983, +0.574, −0.054)               |

The Hopf-unstable focus coexists with a stable focus and a saddle.

### No stable limit cycle is born

Long-time integration of the coupled 3-eq ODE
(`dγ/dt = nbm.dgamma_dt(γ, θ, focal_loc)`,
`dθ/dt = K·R·sin(ego_angle)`)
from `[R_u + ε, 0, θ_u]` at the (1.35, 2.40) Hopf focus, for
`ε ∈ {1e-5, 1e-3, 1e-1}` and `T=200`, settles in every case onto the
stable focus at `(θ=-0.4954, R=0.5124)`. No orbit of any amplitude
persists, even though the Hopf eigenvalue real part `+0.174` corresponds
to a time constant of only ~6 units (compared to ~12 units for the
VM-circle island, which orbits with period ≈ 17.4).

Trajectories shown in the bottom row of `hopf_overview.png`.

### Criticality: exit-via-saddle, not exit-via-LC

The (1.35, 2.40) cell has Re=+0.174 but Im=+0.0094 — nearly
Bogdanov-Takens. Re-evaluating all 86 Hopf+ cells for the (Re, Im) of
their complex pair, the Hopf-unstable foci trace a curve from
large-Re/small-Im (near-BT) to small-Re/large-Im (just past the Hopf
curve, Im ≈ 0.39). At observer (1.45, 2.45) the eigenvalues are
`+0.0517 ± 0.3388j` — modest unstable spiral with period 2π/0.339 ≈ 18.5
and growth e-folding 1/0.052 ≈ 19, comparable to the VM-circle Hopf
island (period ≈ 17.4).

Forward integration from `ε ∈ {1e-4, 5e-3, 5e-2}` perturbations:
log-distance from the focus grows **almost exactly as `exp(Re·t)`** with
no intermediate saturation. Around t ≈ 120 (for ε=1e-4), the trajectory
reaches the saddle at θ=-0.928 (only 0.053 rad from the Hopf focus at
θ=-0.981); the saddle's unstable manifold then sweeps the trajectory to
the global stable focus within ~30 more time units. **No limit-cycle
plateau interposes at any amplitude scale.**

Backward integration diverges to e+37 within t=100 (the global stable
focus becomes repelling in reverse time and pushes trajectories
unboundedly), so backward integration cannot reveal a putative unstable
LC in this multistable landscape.

Mechanistically: the Hopf-unstable focus is the *local* picture in the
linearization, but the *global* escape route runs through the
neighboring saddle's stable manifold rather than around a limit cycle.
The Hopf bifurcation here is "soft" — linearly Hopf-unstable but with
no robust limit-cycle attractor or repeller, because the saddle
structure intervenes before any LC can develop. This is fundamentally
different from the VM-circle case, where the Hopf focus sits inside a
1-equilibrium region with no nearby saddles, so the unstable manifold
*must* close into a stable cycle.

`hopf_criticality.png` shows the exp(Re·t) growth and the absence of
any saturation plateau.

### Why the picture is so different from VM/circle

In the VM-circle case, the Hopf-unstable focus appears inside a
1-equilibrium region — a thin "wedge" between the saddle-node curve and
the Hopf curve where only one self-consistent eq exists. There is
nowhere else for a trajectory to go, so a limit cycle must exist (and
does).

In delta + ANGLE, the Hopf-positive cells sit inside multistable regions
(3 to 5 self-consistent eqs). The Hopf bifurcation removes the local
stability of one focus but the other stable focus remains a global
attractor; the basin of the stable focus extends across the formerly
local basin of the now-unstable focus, with at most a tiny unstable
limit cycle separating them. There's no closed bounded region where
trajectories must orbit.

### On the y=0 midway spiral specifically

The original open question was prompted by the ANGLE-only midway
equilibrium at (0.76, 0) having eigenvalues `−0.054 ± 0.175i` — a
slowly-decaying spiral that looked Hopf-adjacent. Walking the observer
along y=0 and tracking the midway-equilibrium eigenvalues shows that
the complex pair never actually crosses zero from below: the spiral
collides on the real axis (the complex pair merges into two real
eigenvalues) and one of the resulting real eigenvalues then crosses
zero — i.e., the destabilization is a real-eigenvalue crossing
(pitchfork-like, consistent with the y=0 symmetry), not a Hopf. The
Hopf-positive cells found above are all off-axis.

## Walker blind-spot trap under cutoff weighting (delta targets)

**Date:** 2026-05-23

Investigation of why delta-target walkers sometimes wander off and never
return to any target, triggered by the walker termination criterion work.

### Setup

Same four-target delta geometry as the rest of this folder:
`target_locs = [(4.33, ±2.25), (4.33, ±0.75)]`, `geom_name=None`.
`neural_weight='cutoff'`, `a=0`, `b=pi`, `neural_angle='integral'`,
`K=10`, `dt=0.1`, `v=1`, `std=0.5`.  30 walkers from `(0, 0)` facing 0.

5 of 30 walkers hit `max_steps` without finding any target.  All five
ended up 50-75 units from the targets — not grazing, but genuinely lost.

### Mechanism: neural-angle collapse at ±pi

Detailed step-by-step diagnostic of seed=3 (reproduces the behavior):

1. **Overshoot (steps 0-57).**  The walker approaches target 0, steered by
   strong torque (`K*R*sin(ego) ≈ 8-10 rad/s`).  But it overrotates —
   heading swings past the target direction to 122.7° at closest approach
   (distance 0.165 from target 0).

2. **Noise kick (step 57).**  A large negative noise draw (-1.247) swings
   the heading back to 99.3°.  Now all four targets are behind the walker
   (physical ego angles 130-170°).

3. **Neural angle collapse (steps 58+).**  The integral neural mapping
   with `a=0, b=pi` maps physical ego angles near ±180° to neural angles
   of **exactly ±180°**.  By step 60, all four neural angles are pinned
   at ±180° — the mapping has lost all directional differentiation.

4. **Gamma locks onto the branch cut.**  With all neural angles at ±180°,
   `dgamma_dt` converges to `gamma = -1 + 0j` (angle = ±180°, R = 1.0).
   This sits exactly on the ±pi branch cut of `np.angle`.

5. **Torque death.**  `convert_gamma(gamma)` returns `ego_angle ≈ ±180°`.
   `sin(±180°) ≈ 0`, so `K*R*sin(ego_angle) ≈ 0`.  The restoring torque
   drops from ~8 rad/s (target in front) to **< 0.3 rad/s** (all targets
   behind).  Floating-point noise at the branch cut also causes the torque
   to **oscillate in sign** between steps — the model can't decide whether
   to turn left or right.

6. **Pure random walk.**  With negligible net torque, the walker drifts
   under noise.  Cumulative torque after 40 steps past closest approach:
   -0.507 rad.  Cumulative noise: -0.383 rad.  Neither dominates; the
   heading oscillates between ~85° and ~127° with no net progress toward
   facing the targets.

### Root cause

This is a genuine physical feature of the model, not a numerical artifact.
The cutoff weighting + integral neural mapping creates a **perceptual dead
zone** directly behind the walker: physical angles near ±180° are all
mapped to the same neural angle (±180°), destroying the directional
information the torque equation needs to steer the walker back.

The model's prediction: an agent with foveal (forward-biased) perception
that overshoots its target and gets everything behind it will be unable to
navigate back.  Whether this is a desirable prediction or a modeling
artifact to be addressed is an open question.

### Possible remedies (not yet implemented)

- **Minimum torque floor or explicit U-turn behavior:** if all targets are
  behind the walker, impose a minimum turning rate to force a U-turn.
- **Wider neural mapping:** use `a > 0` or a less peaked weighting so that
  targets at ±150° still have distinct neural angles (not collapsed to
  ±180°).
- **Heading noise coupling:** make the noise term heading-dependent so that
  walkers in the dead zone get larger random kicks, modeling increased
  "searching" behavior when all targets are behind.

### Reproduction

Seed 3 with the setup above reproduces the behavior deterministically.
Seeds 10 and 16 also wander off with 400-step walks.  Diagnostic script
is in the conversation transcript that produced this section.

## What's NOT covered here

- **3+ symmetric targets with observer at the center** (the configuration
  the user said originally motivated weighting). A first pass found that
  both modes produce similar equilibrium counts at the exact center, but the
  brentq scan in `sc_equilib` was missing 3-fold-symmetric copies, so
  the comparison was not conclusive. Worth a focused follow-up before any
  decision about removing the weighting.
- **Walker SDE dynamics, basin sizes, noise-induced switching.** All the
  long-time integrations above are deterministic. With noise (the
  walker's T term), the Hopf-unstable focus + stable focus pair may show
  noise-driven switching between near-orbiting and settled states; this
  has not been characterized.
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
### Hopf-island investigation — numerical methods

The Hopf-island section above (`hopf_overview.png`, `hopf_criticality.png`)
was produced by a set of one-off scripts that have since been deleted;
this paragraph captures the parameterization for reproducibility.

- **Equilibrium finder.** `NeuralBandModel.sc_equilib(focal_loc=fl,
  stability_criterion='coupled')`. This is the canonical self-consistent
  finder in the library (see [decision_model.py:2250](../decision_model.py)) —
  brentq sign-change scan over θ at `R_probe = 0.5`, hybr polish each
  candidate, residual filter `1e-4`, dedup with circle-distance(θ) <
  0.02 AND `|R − R'| < 0.01`. At all test cells above it returns the
  same eqs as my one-off custom finder, with machine-precision residuals.
- **Coupled Jacobian.** 3×3 finite-difference Jacobian of
  `(γ_re, γ_im, θ) → (Re dγ/dt, Im dγ/dt, K·R·sin(ego_angle))` at the
  equilibrium, with step `h = 1e-6`. Stability classification:
  *Hopf-unstable focus* if max real part is positive and any eigenvalue
  has `|Im| > 1e-6`; *saddle* if all eigenvalues are real with exactly
  one positive; *stable* if max real part `< −1e-8`.
- **Bifurcation rasters.** Observer window `[0, 6] × [-3.5, 3.5]` at
  121 × 141 cells. Multiprocessing pool sized via `parallel_config.get_n_workers()`.
- **Forward / backward integration.** `scipy.integrate.solve_ivp` with
  `method='LSODA'`, `rtol=1e-10`, `atol=1e-12`, `max_step=0.2`. Time
  windows: `T_fwd = 200` and `T_bwd = 100` for the criticality test at
  (1.45, 2.45); longer windows (T=300–4000) for the (1.35, 2.40)
  trajectory in `hopf_overview.png`.
- **Test cells.** (1.35, 2.40) for the headline trajectory (Re=+0.174,
  Im=+0.009, near-BT); (1.45, 2.45) for the criticality test (Re=+0.052,
  Im=+0.339, period ≈ 18.5, e-folding ≈ 19); (2.0, 2.7) as a
  just-past-Hopf-curve probe (Re=+0.0005, Im=+0.392).
