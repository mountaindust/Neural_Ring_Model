# Weighting vs warping in NBM

**Started:** 2026-05-22 · **Last revised:** 2026-08-19

> **Decision (2026-06):** keep uniform weight (`angle_weight=None`); do not adopt
> a foveal vonMises weight. The `1/N` pathology is exclusive to the *exact delta*
> (a singular limit), and a foveal weight buys a marginal commitment-signal lift
> for the cost of the far-target "ears" below. See
> [foveal_decision.md](foveal_decision.md).

> **Anti-foveal follow-up (2026-08):** whether a weight with a **dip** in the
> middle (rather than a bump) can bias the observer *outward* — the mechanism
> the locust three-target data would need — is a separate, self-contained
> investigation: [outward_bias.md](outward_bias.md)
> (script [outward_bias.py](outward_bias.py)). Short answer: **no**. An
> egocentric weight suppresses whatever is dead ahead, which penalises every
> single-target commitment, and the outer ones are the fragile ones; it also
> lowers `R`, which is the same quantity the foveal-weight commitment-signal
> argument wants to *raise*. Both point the same way: **concentrate the weight,
> don't spread it.** The two weight families written for it were removed from
> the model afterwards and preserved in [anti_foveal.py](anti_foveal.py).

### Reading note — vocabulary

This folder predates the **warp/weight decouple**. `PerceptionModel` used to
take a single `neural_weight` function serving both roles plus a
`weight_angle_only` switch; it now takes two independent arguments. The old
terms are mapped to the current API as follows, and the current names are used
from here on:

| old (in the original write-up) | current API |
|---|---|
| `neural_weight=W, neural_angle='integral'` ("FULL weighting") | `neural_angle_dist=W, angle_weight='neural_angle_dist'` |
| `weight_angle_only=True` ("ANGLE-only") | `angle_weight=None` — **uniform weight, and now the model default** |
| `neural_angle='power'` | `neural_angle_dist='direct_power'` |

Where the tables and prose below still say **FULL** / **ANGLE-only**, read them
as **weight tied to the warp** / **uniform weight**.

## Question

The perception model has two roles:

1. **Weighting** (`angle_weight`) — a density integrated over each target's
   visible angular interval to produce `rho` (the perceptual mass that target
   contributes to gamma).
2. **Warping** (`neural_angle_dist`) — a density integrated CDF-like to map
   egocentric angles to neural angles via `get_neural_angle`.

Setting `angle_weight=None` keeps role (2) but makes role (1) uniform
(integration weights every visible radian equally). The hypothesis under test:

> Does the warping alone do most of the work? In particular, is the
> "symmetry-breaking" that originally motivated the front-bias weighting
> already provided by the front-stretching effect of the warp?

## Headline finding

Across the standard two-circle symmetric setup
(`targets at (4.33, ±2.5), r=0.5`, observer in `[0,6] × [-3.5, 3.5]`):

- The bifurcation rasters with and without weighting agree on
  **~82-95% of cells** depending on how peaked the weighting function is
  (regenerated 2026-08: 88.4 / 82.1 / 88.5 / 95.0% for the four rows below).
- Most disagreement is **boundary jitter** (a 1-2 cell shift in saddle-node
  arcs on the left near the targets).
- One **genuine structural difference** appears as two "ears" of extra
  bistability on the right side of the figure (`x ≳ 3`, `|y| ≈ 1.5-2.6`),
  where FULL weighting produces a 2-stable region and uniform weight produces
  only a 1-stable region. In the regenerated rasters each ear is a **single
  continuous band wrapping around and behind its target**; in the 2026-05
  version it was split into two lobes by a notch at the target's own `x` —
  that notch was the wrapping bug (see below), not structure.

Ear area (`x ≥ 3` cells where FULL has strictly more stable equilibria than
uniform) still scales with weighting peakedness. **Regenerated 2026-08-06** by
[ears_figure.py](ears_figure.py). Two things changed since the 2026-05 run — the
wrapping-extent fix and the default criterion (`'coupled'` → `'reduced'`) — so
the middle column re-runs the *old* criterion on the *fixed* code to separate
them:

| weighting parameterization | 2026-05: `coupled`, **pre-fix** | current code, `coupled` | current code, `reduced` (shipped) |
|---|---|---|---|
| cutoff `a=0`, `b=π` (sharpest) | 2.12 | 3.35 | **3.36** |
| vonMises `k=0.9` | 2.46 | 3.69 | **3.72** |
| vonMises `k=0.5` | 1.72 | 2.63 | **2.65** |
| cutoff `a=π/3`, `b=π` (mild) | 1.01 | 2.08 | **2.08** |

**The criterion change accounts for essentially none of it** (columns 3 and 4
agree to ≤0.03 sq units, i.e. a couple of boundary cells). The ears grew by
**+58% to +106%**, and that is the wrapping fix. Reproduce column 3 with
`python weighting_analysis/ears_figure.py sweep --criterion discrim_a --areas-only`.

Peakedness ordering and the qualitative reading are unchanged — but every ear
area quoted in the 2026-05 write-up is an **undercount**.

See `ears_figure.png` for the per-row layouts and `ear_diagnostic.png` for
the mechanism. (The 2026-05 versions of both are in git history; what changed
and why is quantified above and in the next section, which is the part worth
having.)
(`K` went 1 → 2 as well, but stable counts are K-invariant at an SC equilibrium —
see `.claude/rules/torque-and-stability.md` — so it cannot contribute. The grid
also differs: the current run is 37×43 + 2 refinement passes; areas are
cells × cell-area, so they are comparable up to discretization.)

### The wrapping-extent fix, and why it mattered here

`decision_model._get_target_signals` used to hand the **closest** target's raw
angular extent to `_integrate_neural_weight` without unwrapping it. An extent
straddling ±π comes back from `get_percep_angles` as a *wrapping* pair
(`lo > hi`), which integrates to a **negative** arc length, which the `G > 0`
visibility filter then silently discarded — so the nearest target vanished from
perception for the whole angular window in which it straddled the rear branch
cut. (Targets with a closer blocker were already safe: `_subtract_intervals_circle`
unwraps its own inputs. The closest target never enters that loop.) Fixed
2026-08-04, regression-tested in [../tests/test_intervals.py](../tests/test_intervals.py).

It required an extended target (circle/capsule — deltas use a pointwise weight
and were never affected) that is also the **closest** one; the ears live exactly
where that happens, an observer close to one target and considering a commitment
that puts it behind them.

**Worked demonstration.** Observer (4.5, 2.0) — just outside the upper target,
in the vertical notch that split the old ear mask into two lobes — cutoff warp
`a=0, b=π`, at the candidate heading facing the *far* (lower) target. The near
target is 0.53 away at egocentric **−159.1°** with a **142.4°** visual extent,
so its extent runs past −180° and comes back as a wrapping pair:

| | FULL weighting | uniform weight |
|---|---|---|
| raw wrapping-pair integral (old code) | **−5.779** | **−3.797** |
| → `G > 0` discards the near target? | yes | yes |
| correct (unwrapped) integral | +0.504 | +2.486 |
| far target's integral | +0.445 | +0.223 |
| **correct** ρ(near) | 0.531 | **0.918** |
| **buggy** ρ(near) | 0 (dropped) | 0 (dropped) |

The bug fires in *both* columns — but it only changes the **count** in one.
With the near target dropped, ρ = (0, 1) makes the far-target commitment
trivially self-consistent. Under FULL weighting that commitment already exists
(it *is* the ear), so the count is unchanged; under uniform weight the near
target correctly outweighs the far one 11:1 and no such commitment exists, so
the bug **invents** one. `FULL − UNIFORM` collapses to zero there: that is the
notch.

**Measured directly** by [wrapping_fix_effect.py](wrapping_fix_effect.py), which
overrides `PerceptionModel._unwrap_interval` in a subclass to revert exactly the
fixed line and nothing else (`_get_target_signals` calls it through `self`,
while `_subtract_intervals_circle` calls it on the class, so the blocking
arithmetic is untouched). Recounting the upper ear on a 31×19 grid over
`[3,6] × [1.2,3.0]`, `cutoff a=0,b=π`, `criterion='coupled'`:

| | ear area | cells |
|---|---|---|
| current code | **1.70** | 170 / 589 |
| with the bug restored | **1.18** | 118 / 589 |

and the per-column damage is exactly as predicted — the **uniform** column
changes in **53 of 589 cells (9.0%)**, all in `x ∈ [3.9, 4.8]`, `y ∈ [1.5, 2.3]`
(beside and behind the upper target at (4.33, 2.5)); the **FULL** column changes
in **1** cell. So the bug erased ~31% of the ear, on the uniform side, in a band
at the target's own `x`. That band is the notch, and closing it is why the
regenerated ears are both larger and continuous.

The mechanism worked out below is *not* affected: at the (5.0, 2.0) worked
example neither target straddles the cut at the headings involved, and the
regenerated diagnostic reproduces the original numbers to the digit.

## Why the ears exist (corrected interpretation)

Initial intuition was wrong. The "extra stable equilibrium" in the ear is
**not** a compromise heading; it's a *commitment* to the **far** target that
uniform weight loses.

Worked example at observer = (5.0, 2.0) with cutoff `a=0, b=π`:

- target 0 at allo direction +143.3°, distance 0.84, visual extent **73.5°**
- target 1 at allo direction −98.5°, distance 4.55, visual extent **12.6°**

Self-consistent stable equilibria found by `sc_equilib` (re-verified 2026-08-06
under `criterion='reduced'` — unchanged):

| mode    | heading_allo | facing       | stable? |
|---------|--------------|--------------|---------|
| FULL    | +143.3°      | target 0     | yes     |
| FULL    | −98.5°       | target 1     | yes     |
| uniform | +143.3°      | target 0     | yes     |

Uniform weight loses the target-1 equilibrium. The mechanism (panel c of
`ear_diagnostic.png`):

At the hypothesized heading −98.5° (facing target 1), target 0 sits at
egocentric ≈ −118° (behind/left of the observer). Its arc length is still
the same ~73°, so:

- Uniform weight assigns ρ = (0.853, 0.147) — the close target dominates by ~5.8×
  *purely from visual extent*, even though it's behind the observer.
  `dgamma_dt(R+0j, θ=−98.5°)` is negative for all R > 0; never crosses zero;
  the consensus is pulled back toward target 0 regardless of R. No
  self-consistent equilibrium at this heading.
- FULL weighting evaluates the cutoff at |ego| ≈ 118° (w ≈ 0.02), squashing
  the off-axis target's contribution. ρ rebalances to (0.574, 0.426).
  `dgamma_dt` now crosses zero at R\* ≈ 0.425. Self-consistent equilibrium.
  (It crosses a second time at R ≈ 0.019 — the unstable partner born with it in
  the saddle-node that creates the commitment; see panel c.)

So the ears occupy observer positions where one target is much closer/bigger
than the other, *and* where committing to the far target would put the close
target off-axis behind. There, **front-bias weighting suppresses the close
target's pull enough to let a far-target commitment be self-consistent.**
Visual extent alone (the only thing uniform weight keeps) cannot do this — the
big target wins no matter where the observer hypothesizes facing.

## Two clarifications worth recording

1. **`angle_weight=None` does NOT make ρ equal across targets.** ρ is
   the integral of (uniform) weight over each target's visible angular
   extent. Closer targets have larger extent and therefore larger ρ. At
   (5, 2.0) the close target has ~5.8× more ρ than the far target under
   uniform weight — purely from being close.
2. **The single uniform-weight equilibrium in the ear region is not a
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

> **Outcome:** the second option was taken. `neural_angle_dist` and
> `angle_weight` are now fully independent, uniform weight is the default (so
> the ears are opt-in), and the "separate, much milder front-bias" is available
> by choosing any weight family and parameters you like — including, since
> 2026-08, anti-foveal ones ([outward_bias.md](outward_bias.md)).

## Delta targets — same structure, threshold shifts outward

Re-running the same observer sweep with `geom_name=None` (delta targets)
instead of circles gives a quite different qualitative picture from the
circle case — but the difference between FULL and uniform is much less
dramatic than the circle ears suggested.

| weighting parameterization | disagreement (delta) | sign of FULL−uniform |
|---------------------------|----------------------|--------------------|
| cutoff `a=0`, `b=π`        | 1.9%                 | FULL has fewer      |
| vonMises `k=0.9`           | 8.7%                 | FULL has fewer      |
| vonMises `k=0.5`           | 5.5%                 | FULL has fewer      |
| cutoff `a=π/3`, `b=π`      | 0.0%                 | (essentially identical) |

See `delta_sweep_comparison.png`. The **qualitative bifurcation structure
is the same in both modes** — same 1-stable / 2-stable / 3-stable regions
in the same nested arrangement around the targets. The only difference is
that **the bifurcation arcs separating these regions are pushed slightly
outward from the targets under FULL weighting**, so a band of observer
positions that are 3-stable under uniform weight is already 2-stable under
FULL. No "ears"; no new structural features.

### The shift is a heading-sensitivity effect

FULL weighting makes the dynamics more sensitive to where the observer is
hypothesized to face: targets directly in front get weighted significantly
more than targets to the side. Under uniform weight, each target's `rho`
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
3×3 Jacobian crosses zero at (a **real** crossing, hence a `det J` sign
change and invariant under the reduction error — see the retraction section
below):

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
- ANGLE eigenvalues: (−0.84, −0.054 ± 0.175i) — stable
- `dρ_target0/dθ` at midway: FULL = +0.258, ANGLE = exactly 0

(These are γ-level 3×3 spectra. The **signs** are trustworthy — the positive
`+0.38` is real, so it is a `det J` sign change, and `eig(A)` is invariant —
but the *complex pair* `−0.054 ± 0.175i` in the ANGLE row is an artifact of
the reduction and should not be read as a spiral rate. See the retraction
section below.)

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
the weighting: uniform weight already supports all the equilibria; FULL just
makes their bifurcations happen at less extreme observer positions. The
circle picture is something stronger: the weighting opens up a regime of
asymmetric bistability that uniform weight cannot produce at all. The
asymmetry is because circle ρ also has a visual-extent component, and the
extent ratio can dominate so heavily under uniform weight that no value of R
can make a far-target commitment self-consistent — whereas the front-bias
under FULL squashes the extent-dominant close target enough to let the
far-target equilibrium exist.

The decoupled-weighting option (let `neural_weight` drive the warp only;
use a separate, milder front-bias for ρ — or none) is still attractive:
it would let the front-bias strength be tuned independently of the
warping shape, which is the conceptually clean role for "neural-band
density." Removing the front-bias entirely (uniform weight) is more
conservative for deltas than for circles — for deltas it costs only a
small shift in where bifurcation arcs sit, but for circles it gives up
the asymmetric-position bistability that produces the ears.

## Delta + uniform weight: Hopf investigation — RETRACTED (2026-08-19)

This section reported a scan for Hopf-unstable foci in the delta + uniform
configuration (86 "Hopf+" cells over a 121×141 raster, an
exit-via-saddle criticality analysis, a near-Bogdanov-Takens claim at
(1.35, 2.40), figures `hopf_overview.png` and `hopf_criticality.png`).
**It has been deleted, and the two figures with it.**

Every Hopf classification in it came from the **full eigenvalues of the
γ-level `(γ_re, γ_im, θ)` Jacobian** — the `'coupled'` criterion, removed
from the model on 2026-08-19. `dγ/dt` is the rank-2 readout of the
K-dimensional Glauber population dynamics and drops a term proportional to
`dθ/dt`; taking the full eigenvalues linearizes an incomplete equation. The
question the section was answering ("is there a Hopf island here like the
vonMises k=0.55 circle case?") is void at both ends: **the vonMises circle
island was itself the same artifact** — see
[stale_coupled_model_starting_code/README.md](../stale_coupled_model_starting_code/README.md)
and the retraction banner in
[VM_bifurcation_old_dtheta/VERDICT.md](../VM_bifurcation_old_dtheta/VERDICT.md).

### What this does *not* touch — the triage rule

The dropped term adds, to each γ-row of the Jacobian, a multiple of the
θ-row. That is a determinant-preserving row operation, so **`sign(det J)`
and `eig(A)` are exactly invariant.** Verified over 1950 self-consistent
equilibria across four configurations (vonMises circle, delta + uniform,
delta + tied weight, cutoff circle): det-sign matched 1950/1950 and
`eig(A)` to ≤ 2.7e-9.

Therefore:

- **Everything determined by a real eigenvalue crossing zero is
  unaffected** — saddle-node curves, pitchfork/threshold crossings, and
  every stable-count boundary that is not a Hopf boundary. This includes
  the delta threshold-shift numbers above (`x = 0.534` FULL, `x = 0.783`
  uniform): those are real-eigenvalue crossings, i.e. `det J` sign changes,
  and they stand.
- **Only complex-pair (Hopf) classifications are corrupted.** In the
  delta + uniform configuration the full-spectrum verdict differed from the
  invariant one at 4 of 544 equilibria in the check above — precisely the
  Hopf cells.

So the `'coupled'`-era rasters elsewhere in this folder are correct except
in those few cells, and the `'reduced'` and `'discrim_a'` criteria — which
use only `eig(A)` and `sign(det J)` — were never affected at all.

### The open question it was meant to answer

The prompt was the uniform-weight midway equilibrium at (0.76, 0) having a
slowly-decaying complex pair that "looked Hopf-adjacent". That complex pair
is a γ-level object and is not trustworthy. What *is* trustworthy is the
destabilization itself: along y=0 the midway equilibrium loses stability
through a **real** eigenvalue crossing zero (pitchfork-like, consistent with
the y=0 mirror symmetry) — a `det J` sign change, hence invariant. Whether
any genuine oscillatory instability exists in this configuration is
**unanswered**, and answering it requires the `(n⃗, θ)` population system
plus an explicit neural timescale `τ₀`.

## Walker blind-spot trap under cutoff weighting (delta targets)

**Date:** 2026-05-23 · **Status: RESOLVED — kept as the diagnosis that motivated
two fixes. Steps 5–6 below describe the OLD torque law and no longer happen.**

> Two changes have since removed this failure mode, and it is worth keeping the
> two apart because they are different bugs at the same place on the circle:
>
> 1. **The half-angle torque law** (`dθ/dt = K·R·sin(Θ/2)`, `K=2`) replaced
>    `sin(ego)`. The old law was zero *both* straight ahead and directly behind;
>    the "torque death" in step 5 below is exactly that spurious behind-zero.
>    Under the half-angle law the torque at `Θ = ±π` is **maximal** (`±K·R`), so
>    a walker with everything behind it turns hard rather than drifting. The
>    remaining `±π` behaviour is an intentional left/right **fork** (a jump
>    between `+K·R` and `−K·R`), not a dead zone — see
>    `.claude/rules/torque-and-stability.md`.
> 2. **The true blind spot** (no visible target at all, which a `b_weight < π`
>    cutoff really does produce behind the walker) is now handled by the `R = 0`
>    fast-path, which searches diffusively at the independent `walk_std`
>    parameter (default `π/2`) until a target re-enters view.
>
> Separately, note that a target *straddling* the rear branch cut used to vanish
> from perception entirely because of an interval-wrapping bug — a **third**,
> unrelated way to lose sight of a target behind you, fixed 2026-08-04. See
> "The wrapping-extent fix" below. The analysis in this section was done under
> a rear-excluding `cutoff` weight, where the targets really were outside the
> visible cone, so it is not an instance of that bug.

Investigation of why delta-target walkers sometimes wander off and never
return to any target, triggered by the walker termination criterion work.

### Setup

Same four-target delta geometry as the rest of this folder:
`target_locs = [(4.33, ±2.25), (4.33, ±0.75)]`, `geom_name=None`.
In current API terms: `neural_angle_dist='cutoff'` with `a_warp=0`,
`b_warp=pi` and `angle_weight='neural_angle_dist'`,
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

### Root cause (as diagnosed then)

The cutoff weighting + integral neural mapping creates a **perceptual dead
zone** directly behind the walker: physical angles near ±180° are all
mapped to the same neural angle (±180°), destroying the directional
information the torque equation needs to steer the walker back.

The conclusion drawn at the time was that this might be a genuine prediction
of the model — an agent with foveal perception that overshoots its target and
gets everything behind it cannot navigate back.

### What was actually done

The angle *collapse* (steps 3–4) is real and unchanged: a peaked warp really
does compress the rear into a narrow neural range. What was wrong was
concluding that this had to cost the walker its torque. Of the three remedies
sketched below, the first was adopted in a stronger form and the third became
the blind-spot search:

- ~~**Minimum torque floor or explicit U-turn behavior**~~ → **superseded by the
  half-angle law**, which does not need a special case: `sin(Θ/2)` is *maximal*
  at `Θ = ±π`, so a walker facing away turns hardest, and the sign oscillation
  of step 5 became a well-defined left/right fork of magnitude `2·K·R`.
- **Wider neural mapping** (`a > 0`, less peaked) still works and is worth
  remembering, but is no longer required.
- ~~**Heading noise coupling**~~ → **implemented** as the `R = 0` blind-spot
  fast-path plus the state-gated noise law (`walk_std`, `noise_exp`,
  `cos(Θ/2)` modulation): see `.claude/rules/walker-dynamics.md`.

Tracked as resolved in [TODO.md](../TODO.md).

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
  long-time integrations above are deterministic; noise-induced switching
  between coexisting stable headings has not been characterized.
- **Capsule targets.** Likely lie between deltas and circles depending on
  `l`/`w`; not characterized here.

## Reproduction

**Committed scripts** (these regenerate their figures from scratch, with npz
caching behind a JSON fingerprint of every input that affects the result):

| script | figures / output |
|---|---|
| [ears_figure.py](ears_figure.py) | `ears_figure.png`, `ear_diagnostic.png` |
| [outward_bias.py](outward_bias.py) | `outward_bias_{mechanism,cascade,events,rasters,walkers}.png` |
| [wrapping_fix_effect.py](wrapping_fix_effect.py) | console only — the buggy-vs-fixed ear recount above |
| [anti_foveal.py](anti_foveal.py) | not a script — the two retired weight families, re-registered onto `PerceptionModel` so `outward_bias.py` still runs |
| [anti_foveal_selftest.py](anti_foveal_selftest.py) | console only — 250 numerics checks on those families |

```
python weighting_analysis/ears_figure.py              # both ear figures
python weighting_analysis/ears_figure.py diagnostic   # just the mechanism panel
python weighting_analysis/ears_figure.py sweep --criterion discrim_a --areas-only
python weighting_analysis/outward_bias.py all
python weighting_analysis/wrapping_fix_effect.py
python weighting_analysis/anti_foveal_selftest.py
```

Note `anti_foveal.py` monkeypatches `PerceptionModel` for the life of the
process. That is contained: only `outward_bias.py` and the self-test import it,
neither is collected by `pytest tests/`, and the self-test is deliberately not
named `test_*` so it cannot be swept into a run that also exercises the real
model.

`ears_figure.py` caches each of its eight rasters separately (in
`_cache_ears.npz`, keyed by a hash of that raster's own inputs) and writes the
cache as soon as each finishes, so an interrupted run resumes and a
`--criterion` A/B does not evict the shipped rasters. The most-peaked row is by
far the slowest; a full cold run is well over an hour on 10 workers.

**Not yet re-scripted.** The remaining figures below were produced by one-off
scripts that were never committed (their parameterizations are recorded here so
they can be rebuilt, but there is no runnable file):

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
  the eigenvalue through zero define the FULL/uniform disagreement band
  on the y=0 axis. Reproduction script in the transcript.

Note that the three delta figures predate both the `'reduced'` default and the
wrapping-extent fix. The fix cannot touch them — delta targets use a *pointwise*
weight, never the arc integral, so they were never subject to the wrapping bug —
and the criterion change is now known to be almost inert here: `'coupled'` and
`'reduced'` differ only where a *complex pair* crosses zero, since `eig(A)` and
`sign(det J)` are invariant under the reduction error (see the retraction
section above). So these rasters and threshold crossings are correct except in
the handful of Hopf cells. The `ears_figure.py` circle sweep has been
regenerated under the current defaults; these have not.

The `_delta_rasters.npz` cache referenced above is **not present** in this
folder, so `delta_sweep_comparison.png` cannot currently be rebuilt even from
the recorded parameterization.
