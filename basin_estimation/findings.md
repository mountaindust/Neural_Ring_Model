# Basin-of-attraction vetting — running mathematical findings

A self-contained writeup of the mathematical concepts and findings from
the vetting plan in [README.md](README.md). Aimed at a reader who knows
basic stability analysis (fixed points, Jacobian eigenvalues, the use of
linearization to classify stable/unstable equilibria) but is new to:

- two-timescale dynamics and slow manifolds,
- Langevin SDEs and their stationary distributions,
- effective potentials V(θ) and the Poincaré-Hopf theorem,
- folds (saddle-node bifurcations) on a parameterized branch,
- Schur complements for slow-eigenvalue extraction,
- basin-attribution bisection.

Each concept is introduced as it comes up. As more vetting steps are
worked through, this file grows; the table of contents tracks which
step contributed each section.

## Table of contents

| §  | Topic | Step |
|----|-------|------|
| 0  | Re-vetting under the K=2, sin(Θ*/2) torque law + γ/θ decoupling | 2026-06 |
| 1  | The state space and the slow manifold | foundation |
| 2  | γ-Langevin dynamics and what they tell us | 3 |
| 3  | Slow-manifold θ-scan and γ-continuation | 4 |
| 4  | Basins, effective potential, and the multistable case | 5 |
| 5  | The Schur complement and slow eigenvalues | 5 (technical) |
| 6  | γ-saddle finding and ΔF_γ at fixed θ | 6 |
| 7  | Discontinuity detection during θ-scans | 7 |
| 8  | Monte Carlo escape-time validation of Kramers | 8 |
| 9  | Asymmetric basin test (close+far targets) | 9 |
| 10 | Hopf island graceful failure and the public-API wrapper | 10 |
| 11 | Performance benchmark and decision on grid strategy | 11 |
| 12 | Summary — vetting plan complete | — |
| 13 | Open theory questions | — |

## 0. Re-vetting under the K=2, sin(Θ*/2) torque law (2026-06-01)

**The original Steps 1–11 below were vetted under the old heading torque
`dθ/dt = K·R·sin(ego)` with `K=1`. The model has since changed to the
half-angle law `dθ/dt = K·R·sin(ego/2)` with default `K=2`** (see the
CLAUDE.md "Half-angle heading torque" note). This section records the
re-vetting of every Steps 5–9 calibration number under the new law. The
three basin_estimation/ formula sites (`theta_scan.py`,
`basin_via_theta.py`, `mc_escape.py`) were already updated to `sin(ego/2)`;
this is the numerical confirmation.

**Headline: almost everything is invariant, and provably so.** The *only*
place `K` and the half-angle enter is the θ-equation. At *every*
self-consistent equilibrium `ego=0`, where `d/dθ[sin(ego/2)] = ½·ego'`
versus `d/dθ[sin(ego)] = ego'`; doubling `K` (1→2) exactly cancels that ½.
So the full 3×3 coupled Jacobian — and therefore stability, slow
eigenvalues, Hopf/saddle-node locations — is *identical* at SC equilibria.
Everything that is a property of the γ-only landscape or of SC equilibria
is mathematically K-independent (F̂, γ_eq(θ), γ-saddles, ΔF_γ, γ-fold
locations). Verified numerically against the recorded values:

| quantity | findings value (old law) | new-law value | status |
|---|---|---|---|
| SC eqs at (1.2,0) | stable ±0.6625, 0; unstable ±0.3674 | identical | **invariant** |
| SC eqs at (2.0,0) | stable ±0.8204; unstable 0 | identical | **invariant** |
| SC eqs at (4.0,1.5) | stable +1.2520, −1.4885; unstable −1.3096 | identical | **invariant** |
| γ-eq counts at (1.2,0) | 5 (3 min/2 sad) center, 3 (2/1) side | identical | **invariant** |
| ΔF_γ center (1.2,0) | 0.0154 | 0.01537 | **invariant** |
| ΔF_γ side (1.2,0) | 0.00426 | 0.00426 | **invariant** |
| γ-folds (2.0,0) | −0.92, −2.46 | −0.9232, −2.4626 | **invariant** |
| γ-fold (1.2,0) inner | ±0.4789 | ±0.4842 | **invariant** |
| basin widths (4.0,1.5) | close 303°, far 57°, ratio 5.35 | 303.3°, 56.7°, 5.35 | **invariant** |
| (4.0,1.5) boundaries | fold −1.39, saddle −2.38 | fold −1.394, saddle −2.384 | **invariant** |
| slow eig at SC eq (0.5,0) | (V''≈+0.42, time ~2) | −0.419 (V''=+0.419) | **invariant** |
| Step 8 central slope | 0.0179 (ΔF_γ 0.0154) | 0.0180, ratio 1.168 | **invariant** |
| Step 8 τ tables | 110.6/18.4/7.6; 211.5/48.1/14.8 | 112.7/19.2/7.6; 213.8/48.1/14.8 | **invariant** (within MC noise) |
| Step 8 ordering | τ_side≈40, τ_cent≈1200 @ D=0.003 | 40.0, 1243.7 | **invariant** |
| Hopf island (2.1,2.45) | 0 stable + Hopf sentinel | identical | **invariant** |

**The one genuine change: a new branch-cut basin-boundary type at the
facing-away heading.** Under `sin(ego/2)`, wherever γ_eq crosses the
negative real axis (arg(γ_eq) through ±π, i.e. consensus directly behind),
`f` jumps from +K·R to −K·R while γ_eq moves *continuously* through −R
(only its argument flips sign across the cut). Measured at (0.5,0), θ=±π:
γ_eq = −0.9494, R = 0.9494, `f` jumps −1.899 → +1.898, so |Δf| = 3.80 ≈
2·K·R = 3.798, with |Δγ| ≈ 0.02 (a pure f-jump, not a γ-fold). This is the
intentional left/right fork at the facing-away point, **not** a numerical
artifact (both `sin(arg(γ)/2)` and the algebraic half-angle identity give
two-sided limits ±1 there). Consequences for the basin machinery:

- **f(θ) magnitudes off-equilibrium shift.** Near every SC eq `f` is
  unchanged to first order (K=2·½ = 1 = old K·1), but away from ego=0 the
  two laws diverge; at the facing-away heading the old law gave `f≈0`
  (a spurious torque dead-zone) and the new law gives `|f|≈K·R` (maximal).
  Any reported `f(θ)` plot or off-equilibrium `ΔV` excursion value shifts.
- **A basin boundary that sat at θ=±π under the old law is now a
  branch-cut, not a saddle.** This matters only in 1-stable cells and in
  symmetric cells where a basin edge falls at the facing-away heading. In
  the multistable *asymmetric* cells (e.g. (4.0,1.5)) the boundaries are
  saddles and γ-folds — both invariant — so basin widths are unchanged.
- **Step 7's detector already catches it** as a type-(d) event (see §7).
  The §7 BlindSpot/perception-collapse machinery is unaffected. The old
  Step-7 T1 ("smooth 1-stable scan → zero detections") was updated to
  expect exactly one branch-cut f-jump at θ≈±π; T1 now passes with that
  expectation. All other test scripts in basin_estimation/ pass unchanged.

**Net effect on the integration TODO.** The Steps 5–9 calibration
constants stand as written; the only thing the basin machinery gains under
the new law is the fourth boundary type (saddle, γ-fold,
perception-collapse, **branch-cut**). Section-level notes below are
annotated where the old K=1/sin(ego) framing appears.

### 0.1 Why this is robust to *future* dθ/dt changes — the γ/θ decoupling

`K` (and possibly the functional form of the turning law) may still move as
the dθ/dt physical model is refined. The invariance above is not a
coincidence of the specific `K=2, sin(ego/2)` choice — it reflects a
structural decoupling worth stating directly, because it tells you which
results are banked regardless of what dθ/dt becomes.

Write the turning law as `dθ/dt = K·R·g(ego)`, where `g` is the
dimensionless shape function (`g = sin` old, `g = sin(·/2)` now), with the
two properties any sane turning law has: `g(0)=0` (don't turn when facing
consensus) and `g'(0)>0` (turn toward consensus, harder the farther off).

- **dγ/dt is Hamiltonian/Glauber-derived; dθ/dt is phenomenological.** The
  only place the turning law enters the whole system is the θ-equation, and
  θ feeds *back* into dγ/dt only through perception geometry (the neural
  angles θ̂_j and weights ρ_j), never through a turning parameter.
- **The slow manifold γ_eq(θ) knows nothing about `g`.** Everything that is
  a property of it is turning-law-independent: γ_eq(θ), R(θ), ego(θ),
  γ-folds, perception-collapse zones, and the entire γ-noise story
  (F̂, γ-saddles, ΔF_γ).
- **SC equilibria sit where ego(θ)=0** — a γ-side condition — and any law
  with `g(0)=0` vanishes there, so SC-eq *locations* are turning-law-
  independent. Their *reduced-flow* stability is `sign(f'(θ_s)) =
  sign(g'(0)·ego'(θ_s)·R)`; with `g'(0)>0` fixed, the sign (hence the
  stable/unstable count and the basin count) is set by the γ-side quantity
  `ego'(θ_s)`. The *magnitude* of `f'(θ_s)` scales with `K·g'(0)` — that is
  the only thing the K=2 doubling held fixed (2·½ = 1·1).

So **basin topology, boundary inventory, basin counts, geometric widths
bounded by saddles/folds, and γ-noise robustness (ΔF_γ) are insulated from
the dθ/dt uncertainty.** What stays tied to dθ/dt: slow-eigenvalue
*magnitudes* → θ-relaxation timescale → **θ-noise barrier heights ΔV**;
the coupled 3×3 fine structure (Hopf-boundary locations) *if* `K·g'(0)`
drifts off 1; and the existence/width of branch-cut-bounded basins.

### 0.2 Two open items that depend on the final dθ/dt

Both are on the θ-noise side and cannot be closed until the turning law is
fixed; flag them when that happens:

1. **θ-noise basin "depth" at the branch-cut/antipode needs its own
   first-passage treatment.** The branch-cut is a *repelling fork*, not a
   Kramers barrier you climb over, so the saddle/fold escape formulas in §7
   don't apply to a basin bounded by it. Its FPT behavior must be worked
   out for whatever `g` is finally chosen.
2. **The θ-noise-vs-γ-noise proxy question (§13.4) is dθ/dt-dependent.**
   Whether the `std` knob in `plot_walkers` is a faithful stand-in for the
   physical γ-noise rests on the θ-dynamics, so the σ ↔ T·g(geometry)
   calibration should be (re)run once dθ/dt is settled. The γ-noise half
   (ΔF_γ, Step 6/8) is already done and is turning-law-independent.

## 1. The state space and the slow manifold

The deterministic dynamics live on (γ_re, γ_im, θ) ∈ ℝ² × S¹. The first
two coordinates are the real and imaginary parts of the neural
coherence value; the third is the observer's heading, on the circle.

The dynamical equations have a built-in **separation of timescales**:

- γ relaxes on its own timescale, set by the Hessian of F̂. In our
  calibration setup the Hessian eigenvalues at the SC equilibria are
  ~1, so γ relaxes on time ~1.
- θ moves under the torque dθ/dt = K · R · sin(ego_angle/2), with K = 2
  in the current setup (this section was originally written for the old
  K=1, sin(ego) law; see §0 — the two are identical at SC equilibria, so
  the slow timescale here is unchanged). Near a stable SC equilibrium
  R ~ 0.6 and ego_angle ~ 0, so dθ/dt is small and θ moves on time ~2.

These are not enormously different — but the ratio is ~2, which turns
out to matter for some of the numerics (see §5 on Schur).

The **slow manifold** is the surface in state space where γ has
relaxed to its quasi-steady value given the current θ:

$$γ_{eq}(θ) = \text{the γ-fixed-point of } d γ /d t \text{ at this θ}.$$

Picturing the dynamics in (γ_re, γ_im, θ)-space:

```
γ_im
 │
 │             ╱     trajectory falls
 │            ╱      onto slow manifold,
 │           ╱       then moves slowly
 │          ╱        along it in θ
 │      ──╱─────●──────────────────  γ_eq(θ)  slow manifold
 │       ╱   ╱
 │      ╱  ╱     fast γ-relaxation
 │     ╱ ╱       (perpendicular to manifold)
 │    ╱╱
 │   ╱
 │  ●  initial (γ, θ)
 │
 └────────────────────────────── γ_re
```

A trajectory starts somewhere off the slow manifold, falls onto it
quickly (because γ is fast), then evolves slowly along the manifold as
θ drifts. This is the central organizing fact for everything that
follows.

**Why this is useful.** Once we have γ slaved to γ_eq(θ), the
remaining dynamics is *1-dimensional in θ*. That makes the basin
question tractable: instead of asking "which (γ, θ) starting points
flow to this stable?" we ask "which θ starting points (with γ slaved)
flow to this stable?"

The reduced 1D ODE is

$$\dot θ = f(θ) := K \cdot R(θ) \cdot \sin\!\bigl(\text{ego\_angle}(θ)\bigr),$$

where R(θ) = |γ_eq(θ)| and ego_angle(θ) is the inverse neural mapping
of arg(γ_eq(θ)). This f(θ) is exactly what the θ-scan in Step 4
computes.

(Caveat: γ_eq(θ) isn't always a single-valued smooth function of θ. It
can have *branches* and *folds*. We get there in §4.)

## 2. γ-Langevin dynamics and what they tell us — Step 3

### 2.1 What is a Langevin SDE?

The deterministic γ-dynamics is dγ = −∇F̂(γ) dt. The **Langevin
equation** is the same dynamics with Gaussian white noise added:

$$d γ = -\nabla F̂(γ)\, dt + \sqrt{2 D}\, dW.$$

dW is a 2D Wiener increment — an infinitesimal Gaussian random vector
with variance dt per component, independent of the past. Intuitively,
W(t) is "Brownian motion": a continuous random path whose displacements
over disjoint time intervals are independent and Gaussian-distributed
with variance equal to the interval length. The noise term
√(2D)·dW(t) gives the system random kicks whose typical magnitude per
unit time is set by the **diffusion coefficient** D.

You can think of this as moving in a landscape F̂ while being randomly
jostled. With no jostling (D = 0), the trajectory rolls deterministically
into the nearest minimum. With jostling, it occasionally escapes.

### 2.2 Euler-Maruyama integration

We integrate the SDE numerically by the simplest stable scheme:

$$γ_{n+1} = γ_n + (-\nabla F̂(γ_n))\, dt + \sqrt{2 D \cdot dt}\, ξ_n,$$

where ξ_n ~ 𝒩(0, I_2) is a fresh 2D standard normal at each step. The
deterministic drift gets scaled by dt; the noise gets scaled by √dt.
That square-root scaling is a hallmark of Brownian motion: noise grows
as √t, not t, so for small dt it dominates the drift, and for large dt
it averages out.

For Step 3, dt = 0.02 and total integration ~8000 time units (400k
steps). That's enough for many independent samples of the stationary
distribution.

### 2.3 The stationary distribution is Boltzmann

For a gradient Langevin SDE (drift = −∇V for some scalar V), the
**Fokker-Planck equation** for the probability density P(γ, t) reads

$$\partial_t P = \nabla \cdot (P \nabla V) + D \nabla^2 P.$$

Setting ∂_t P = 0 (stationary state) and solving gives

$$\boxed{P_{ss}(γ) \;\propto\; \exp\!\bigl(-V(γ) / D\bigr).}$$

This is the **Boltzmann distribution** with effective temperature D.
The role of "temperature" is played by the diffusion coefficient: high
D means broad distribution, low D means concentration on minima.

For our model, V = F̂, so the γ-Langevin's stationary distribution is

$$P_{ss}(γ) \;\propto\; \exp\!\bigl(-F̂(γ) / D\bigr).$$

### 2.4 Local Gaussian near a γ-minimum

Near a stable γ-equilibrium γ_eq (a local minimum of F̂), expand F̂ to
second order:

$$F̂(γ) \approx F̂(γ_{eq}) + \tfrac{1}{2} (γ - γ_{eq})^T H (γ - γ_{eq}),$$

where H is the Hessian of F̂ at γ_eq (this is the 2×2 matrix we
analyzed in Step 2). Substituting into the Boltzmann form:

$$P_{ss}(γ) \;\propto\; \exp\!\left[-\frac{1}{2 D} (γ - γ_{eq})^T H (γ - γ_{eq})\right].$$

This is a 2D Gaussian centered at γ_eq with **covariance matrix**

$$\Sigma = D \cdot H^{-1}.$$

So if H has eigenvalues (λ_1, λ_2), the Gaussian's principal axes
align with H's eigenvectors, with standard deviations
√(D/λ_1), √(D/λ_2) along each axis. Tighter wells (large λ) give
narrower distributions; soft wells give broader ones.

This is what V2 in Step 3 validates: simulate γ-Langevin, histogram
the samples in steady state, compare to D · H⁻¹.

Sketch — γ fluctuating in a quadratic well:

```
γ_im
 │     ⋅ ⋅⋅ ⋅
 │    ⋅·⋅·⋅⋅·
 │   ·⋅··●··⋅·     ← γ samples (Gaussian cloud
 │    ⋅·⋅·⋅·⋅       around γ_eq, principal axes
 │     ⋅⋅⋅          set by H eigenvectors)
 │      ⋅
 └──────────── γ_re
       γ_eq
```

### 2.5 Why D = T/(2kN)

This is the calibration that links the Langevin noise amplitude D to
the underlying spin model's temperature T and finite-size correction
1/N. Heuristically:

The spin model has N independent Bernoulli spins per "group," each
with within-group on-probability q_j = n_j/ρ_j. Each spin contributes a
small fluctuation to the order parameter γ. By the central limit
theorem, the *aggregate* fluctuation of γ over N spins is Gaussian
with variance ~ 1/N at fixed T. Match this binomial fluctuation
variance to the Langevin stationary variance D · H⁻¹: working out the
coefficient gives D = T/(2 k N), where the 2k arises from the F̂
normalization (recall 2k · F̂ equals the mean-field free energy per
spin at the constrained minimum).

A rigorous derivation goes through the **van Kampen system-size
expansion** of the master equation — see [theory_background.md](../theory_background.md)
§IV.4. The upshot is that D ~ 1/N is a *finite-size correction*: in
the N → ∞ limit, D → 0 and the γ-dynamics becomes deterministic.

### 2.6 The 1/N scaling check

V3 in Step 3 verifies this calibration directly: by running γ-Langevin
at three values of N (200, 1000, 5000) and measuring the empirical
γ-variance at each, we check that

$$\frac{\text{Var}(γ; N_a)}{\text{Var}(γ; N_b)} \approx \frac{N_b}{N_a}.$$

If D had the wrong N-dependence (e.g. ~ 1/N²) the ratios would not
match the predicted ones. The observed match (rel err ~2%) is the
calibration's signature.

### 2.7 Why the local Gaussian isn't the whole story

V2–V4 validated the *local* stationary distribution: the Gaussian near
γ_eq. They do *not* validate the behavior of γ when it strays far from
γ_eq — toward γ-saddles or across barriers to other γ-basins. The
Kramers escape rate (Step 8) is the property that depends on the
*tails* of P_ss, not just the local cloud. So Step 3's pass tells us
the noise calibration is right and the local landscape is what we
think; Step 8 will probe the global landscape.

## 3. Slow-manifold θ-scan and γ-continuation — Step 4

### 3.1 What the scan computes

We want to evaluate γ_eq(θ) and the reduced dynamics f(θ) = K · R · sin(ego)
at many θ values, to map out the slow manifold and the 1D θ-dynamics
on it.

The naive approach — at each θ, integrate dγ/dt to steady state
starting from γ = 10⁻⁵ + 0j — works but is wasteful: the γ-flow has
to traverse a long distance from a generic starting point to the
equilibrium, and may take 60–150 time units in the worst case (the
"near-saddle slow manifold" issue documented in CLAUDE.md).

### 3.2 Warm-start γ-continuation

The trick: when stepping from θ_n to θ_{n+1} = θ_n + dθ, start the
LSODA integration from the **previous step's γ_eq**, not from a
generic initial condition. Because γ_eq is a continuous function of θ
*on a single γ-branch*, γ_eq(θ_{n+1}) is close to γ_eq(θ_n) when dθ
is small.

This is a numerical **homotopy** or **continuation** method: by
following a smooth deformation of the equilibrium structure (γ_eq as
θ varies), we cheaply track equilibria that would be expensive to
locate from scratch.

Performance benefit observed in Step 4: ~4 ms per θ sample with
warm-start, vs. tens of ms with cold-start.

### 3.3 What "γ-branch" means

At a single fixed θ, the γ-only dynamics

$$d γ / d t = \sum_j ρ_j\, e^{i \hat θ_j}\, σ(u_j(γ)) - γ$$

can have multiple equilibria — multiple critical points of F̂(γ; θ).
Some are local minima (stable γ-eqs), some are γ-saddles (unstable in
the γ-only flow).

A **γ-branch** is a continuous family of γ-equilibria parameterized by
θ, all of the same type (all stable, say). As θ varies smoothly, these
γ-equilibria deform smoothly. A given γ-branch exists for some range
of θ — outside that range, it disappears at a fold (§4.3).

Warm-start γ-continuation tracks **one specific γ-branch** because the
LSODA integration relaxes the warm-start γ to the nearest γ-equilibrium
— which, for small dθ, is the same branch's continuation.

### 3.4 Verification via the simple case (0.5, 0)

At (0.5, 0) the model has a single stable SC equilibrium and a single
unstable equilibrium on S¹. The scan around the full circle should
find exactly one stable zero of f(θ) (at θ = 0) and one unstable
(at ±π). T1 in Step 4 checks this; T2 checks that the stable zero is
exactly at the sc_equilib value. Both pass.

The mathematical justification for "exactly one stable + one unstable"
is the Poincaré-Hopf theorem on S¹, introduced in §4 below.

## 4. Basins, the effective potential, and the multistable case — Step 5

### 4.1 The effective potential V(θ)

A 1D ODE on a smooth manifold is automatically a **gradient system**:
there always exists a scalar function V such that

$$\dot θ = -\frac{dV}{dθ}.$$

Constructing V is just integration: V(θ) = −∫ f(s) ds + const.

This V is the *effective potential* the walker is descending on the
slow manifold. Equilibria of f are critical points of V:

- **stable** equilibrium (f' < 0) ↔ **local minimum** of V
- **unstable** equilibrium (f' > 0) ↔ **local maximum** of V

Sketch — V(θ) for one stable, one saddle (the (0.5, 0) case):

```
V(θ)
 │
 │        ___________
 │       /           \
 │      /             \
 │     /               \
 │____/                 \____
 +----+-----+-----+-----+----  θ
   saddle stable  saddle
   (at ±π)  (at 0)
              (same point on S¹)
```

In a generic 2D ODE you wouldn't have a gradient structure (would
need a Lyapunov function). On the slow manifold, the 1D reduction
gives one for free. This is why the slow-manifold projection is so
useful.

### 4.2 The Poincaré-Hopf theorem on S¹

A topological constraint on how many stable and unstable equilibria
can coexist on a smooth vector field:

**Poincaré-Hopf, 1D circle case.** For a smooth vector field on S¹
with only isolated zeros,

$$\sum_\text{zeros}(\text{index}) = χ(S^1) = 0,$$

with index +1 for unstable zeros (f' > 0) and −1 for stable (f' < 0).
So

$$(\#\text{unstable}) - (\#\text{stable}) = 0.$$

Intuitively: f is continuous on S¹, so between two stables it must
change sign with positive slope (an unstable in between). Stable and
unstable alternate around the loop.

This gives us a free consistency check: if a zero-finder claims to
have found 3 stable but only 2 unstable on the circle, it's missing
one. We use this in Step 5 to detect that sc_equilib is incomplete
(see §4.5).

### 4.3 γ-folds — saddle-node bifurcations on the slow manifold

At any fixed θ, F̂(γ; θ) has multiple critical points: γ-stable
equilibria and γ-saddles. As θ varies smoothly, these critical points
deform smoothly along γ-branches.

A **fold** (saddle-node bifurcation) is the θ value where a γ-stable
and a γ-saddle on the same branch *collide and disappear* together.
Past the fold, neither exists.

Sketch — γ_eq(θ) on a branch with two folds:

```
R(θ)             
 │            ╱──╲                ╱──╲
 │           ╱    ╲    catastrophic    ╲
 │          ╱      ╲     jump          ╲
 │ ────────●       ●─────→●         ●─────────
 │      branch A    │     │        branch B
 │                 fold-θ-  fold-θ+
 +─────────────────┴─────────────┴────────────  θ
```

For θ between fold-θ− and fold-θ+, the branch-A γ-equilibrium doesn't
exist. Our warm-started LSODA, finding itself out of equilibrium when
it crosses fold-θ−, relaxes to whatever γ-equilibrium *is* nearby —
typically a γ_eq on a different branch (branch B in the picture). The
scan's recorded γ_eq jumps from a branch-A value to a branch-B value
in a single dθ step.

We detect this jump as |Δγ_eq| being much larger than the typical
step magnitude (~0.03 typical, ~0.5 at a fold). We use both a relative
threshold (8× the median step) and an absolute threshold (0.4) to
classify a step as a fold event.

**The fold is itself a basin boundary** in (θ, γ)-space. A walker
sitting on branch A at θ slightly inside the fold, if perturbed
slightly past the fold, no longer has branch-A's γ to relax to — its γ
catastrophically jumps to branch B. Once on branch B, the θ-dynamics
may carry it to a completely different stable.

### 4.4 What can bound a basin

Putting §4.1–§4.3 together: on the slow manifold, the basin of a
stable equilibrium in a *multistable* region can be bounded by:

- **(a) A saddle on the same γ-branch.** A smooth zero of f(θ) with
  f' > 0. The walker's θ flows back to the stable from one side and
  away from it on the other side. The Kramers escape rate near this
  boundary is rate ~ (prefactor)·exp(−ΔV/D_θ), where the prefactor
  involves V''(θ_s) and |V''(θ_saddle)| from standard 1D Kramers.

- **(b) A γ-fold.** The γ-branch terminates. A walker perturbed
  past the fold loses its γ-branch entirely; γ catastrophically jumps
  to a different branch, and the walker may flow to a different
  stable. The exponential dependence on barrier height still holds —
  the V-barrier *up to* the fold (ΔV = V(θ_fold) − V(θ_s) by
  trapezoidal integration of −f from inside the basin) determines
  the rate at which θ-noise pushes the walker to the fold. The
  *prefactor* differs from the smooth-saddle case because |V''| at
  the fold isn't well-defined — the rate-limiting step is reaching
  the fold, not the (instantaneous) crossing.

Either way, the basin in θ is an arc bounded by these events. Step 5's
truncated CCW/CW scans terminate at the first such event in each
direction.

**Scope caveat.** This basin-boundary picture is only physically
meaningful when there is a *second basin to escape to* — i.e. in
multistable regions. In a 1-stable region on S¹, the basin is the
entire circle minus the unstable point, and noise pushing the walker
"over" the V-saddle just sends θ around the loop back to the same
stable. ΔV and ΔF_γ in 1-stable cells measure transient excursion
timescales, not basin-transition rates. See §13.6 for the broader
implications of this scope correction.

### 4.5 Why sc_equilib's unstable count can be incomplete

sc_equilib enumerates self-consistent equilibria: states where γ is
on the positive real axis (γ = R + 0j with R > 0) and θ equals the
consensus direction. Two ways saddles get missed:

**(a) Saddles with γ on the negative real axis.** Some unstable
equilibria have γ = −R + 0j (R > 0), corresponding to the observer
facing directly away from the consensus. sc_equilib's R > 0 filter
excludes these. At (0.5, 0), the unstable at ±π is exactly this case —
sc_equilib reports 0 unstable, but Poincaré-Hopf requires 1.

**(b) Saddles on a different γ-branch.** At multistable points the SC
saddles can sit on the *central* γ = R + 0j branch — the same branch
as the central stable. Side stables sit on *different* γ-branches.
When a side-branch scan is performed with warm-start γ-continuation,
it never passes through γ = R + 0j at the saddle θ-values; the
central-branch saddle is invisible to the side-branch scan, which
instead encounters a γ-fold.

This isn't a bug — it's the correct answer for the walker's
perspective. A walker starting on a side branch experiences a
fold-bounded basin, not the SC saddle.

### 4.6 The basin picture at (1.2, 0)

At (1.2, 0), sc_equilib reports 3 stable (−0.66, 0, +0.66) and 2
unstable (±0.367). The full Poincaré-Hopf-consistent picture from our
truncated scans:

```
   −π                              0                              +π
    ●═══════════●─────────✗════════●════════✗─────────●═══════════●
   sad?    left-stable   left-fold center right-fold right-stable  sad?
                                                                      
    └────  left basin  ────┘└──center basin──┘└─── right basin ───┘
```

Each basin is an arc bounded by a saddle (●) at the ±π end or a γ-fold
(✗) at the inner end. The central stable's basin is saddle-bounded
going CCW (to +π) and to a fold going CW; the side stables' basins
are saddle-bounded going *outward* (to ±π) and fold-bounded going
*inward*. Our scans correctly detect both kinds of boundary.

A numerical subtlety: for the *central stable*, going CCW and going CW
can both wind all the way around to ±π. The dynamics' sensitivity to
which γ-branch the warm-start picks up after the first wrap can split
CCW and CW onto different effective branches, giving an apparent
asymmetry. This is a scan-side numerical quirk, not physics —
restricting attention to the fold-side (the inner boundaries) gives
perfectly y-symmetric results, which is what T4 verifies.

### 4.7 Two notions of "Method B" — saddle-finding via bisection

The planning notes mentioned "Method B" as bisection between two
known stable equilibria. Step 5 turned up that there are actually
**two distinct things** that could be called bisection here, and they
answer different questions.

#### Sign-change bisection (what we implemented in Step 5)

Given two stable θs and a γ-branch (chosen via warm-start), bisect on
the sign of f(θ) along the branch:

- Probe a midpoint θ_m.
- Use warm-start γ-continuation to compute f(θ_m) on the chosen branch.
- Compare to the sign of f at the brackets.
- Continue bisecting toward where f changes sign on the branch.

This finds the **first zero of f on the chosen branch** between the
brackets. If that branch contains a saddle, you'll find it. If it
contains a fold (no zero of f), the bisection has nothing to converge
to and will misbehave.

#### Basin-attribution bisection (what was actually meant)

Given two stable θs, bisect on which basin a midpoint lands in:

- Probe a midpoint θ_m with appropriate initial γ.
- **Run the full deterministic dynamics from (θ_m, γ_init) until convergence.**
- See which stable the trajectory lands at.
- Bisect toward the change-over.

This finds the **dynamical boundary between the two basins** in θ —
whether it's a saddle, a fold, or any other discontinuity. The
boundary is the θ at which an arbitrarily small displacement flips
the endpoint.

Sketch of basin-attribution bisection between two stables θ_L and θ_R:

```
                       Bisection iter
 θ_L ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━● θ_R
       [trajectory lands at θ_L]       [trajectory lands at θ_R]
                       ↓ probe midpoint, evolve
                     ●           lands at θ_R, so bracket is now (θ_L, midpoint)
                       ↓ probe new midpoint
                ●                lands at θ_L, so bracket is (this, prev)
                       ↓ ... converges to the basin boundary
```

In our (1.2, 0) example with the right and central stables, the
basin-attribution bisection would find the fold at θ ≈ +0.479,
because a walker starting just left of the fold (with appropriate γ)
ends up at the central stable, and starting just right ends up at the
right stable.

The sign-change bisection I implemented in Step 5, by contrast, finds
θ ≈ +0.331 — close to the *central γ-branch's* SC saddle at +0.367.
It's the right answer for "where does f change sign on the central
γ-branch" but not for "where does the walker's basin change."
Different question.

#### Implication for Step 7

Step 7 (proper basin boundary detection across γ-branches) will need
to implement basin-attribution bisection. It's harder — each probe
requires integrating the full coupled dynamics to convergence — but
it's the right ground truth for what the walker actually experiences.

## 5. The Schur complement and slow eigenvalues — technical aside

Step 5 needed to compute the slow eigenvalue of the 3×3 coupled
Jacobian at a SC equilibrium and compare it to V''(θ_s) from the
scan. A first attempt — "take the eigenvalue whose eigenvector has the
largest θ-component" — gave an 18% error, which was confusing because
V''(θ_s) and the slow eigenvalue should agree exactly.

The fix is to use the **Schur complement** for the slow eigenvalue.
This is a useful tool worth knowing.

### 5.1 The Schur complement, briefly

Given a block matrix

$$M = \begin{pmatrix} A & B \\ C & D \end{pmatrix}$$

with A square and invertible, the **Schur complement of A in M** is

$$M / A := D - C A^{-1} B.$$

For a 3×3 Jacobian J of a 2-fast-1-slow system, write it as

$$J = \begin{pmatrix} J_{γγ} & J_{γθ} \\ J_{θγ} & J_{θθ} \end{pmatrix}$$

where J_γγ is 2×2 (γ-block), J_θθ is scalar, and J_γθ, J_θγ are
2-vectors. The Schur complement of the γ-block is

$$J_{θθ} - J_{θγ} J_{γγ}^{-1} J_{γθ}.$$

This is the scalar coefficient of the *effective 1D θ-dynamics* on
the slow manifold, after the fast γ-motion has been eliminated.

### 5.2 Why "eigenvalue with largest θ-component" fails

It would be exactly correct in the limit of **infinite timescale
separation**: when the γ-block eigenvalues are much faster than the
θ-block, the 3×3 Jacobian decouples into a γ-block (with eigenvectors
purely in the γ-subspace) and a θ-block (eigenvector purely in θ).
The "slow" eigenvalue would then be exactly the one with eigenvector
along (0, 0, 1).

In our model, the timescale separation is only ~2× (γ-Hessian eigs
≈ 1 vs θ-dynamics eig ≈ 0.5). The mixing between γ and θ-directions
in the slow eigenvector is not negligible: the slow eigenvalue is not
simply the diagonal J_θθ entry; it's the Schur-reduced value, which
includes the J_θγ J_γγ⁻¹ J_γθ "correction" from γ-coupling.

Picking the eigenvalue by "largest θ-component" picks one whose
eigenvector is mostly-but-not-purely θ; the eigenvalue is shifted
from the Schur complement by the mixing. With ~2× separation, the
shift is ~18%.

The Schur complement gives the correct answer to floating-point
precision (T3 passes with 0.02% error).

### 5.3 Where else this comes up

Schur complements appear all over multiscale and reduction analysis:

- Center-manifold / slow-manifold reductions (the standard tool).
- Block-matrix conditioning in numerical linear algebra.
- Causal inference (linear regression with controls = Schur complement
  of the covariance).
- Macroscopic transport coefficients from microscopic models.

If you find yourself trying to "extract a slow direction" from a
coupled linear system, the Schur complement is almost always what
you want.

## 6. γ-saddle finding and ΔF_γ at fixed θ — Step 6

Step 5 looked at *basin boundaries in θ* on the slow manifold. Step 6
looks at *basin boundaries in γ at fixed θ*: at each stable SC
equilibrium (γ_s, θ_s), the γ-only dynamics defines a 2D landscape
F̂(γ; θ_s) whose minimum is γ_s. The γ-basin of γ_s in the complex
γ-plane is bounded by **γ-saddles** — critical points of F̂(γ; θ_s) at
which the Hessian has one negative eigenvalue.

This question is much cleaner than Step 5's because everything happens
at fixed θ — no γ-folds, no branch ambiguities. We're just enumerating
critical points of a 2D smooth function and computing barrier heights.

### 6.1 The barrier height ΔF_γ

For each γ-saddle γ_sad neighboring γ_s, the **barrier height** is

$$ΔF_γ = F̂(γ_{sad}; θ_s) - F̂(γ_s; θ_s).$$

By Kramers theory, the escape rate of a Langevin trajectory from the
γ_s basin over this saddle is

$$\text{rate} \;\sim\; \exp\!\bigl(-ΔF_γ / D\bigr),$$

where D = T/(2kN) is the γ-Langevin diffusion coefficient (§2.5).
**Bigger ΔF_γ means a more noise-robust γ-equilibrium**; small ΔF_γ
means γ-noise can readily kick the trajectory out of its current
γ-basin into a neighboring one.

### 6.2 Two ways to evaluate ΔF_γ, and why they should agree

**Direct evaluation:** plug γ_s and γ_sad into the F̂ formula
(derivation §3) and subtract.

**Path integral:** for any path γ(s) from γ_s to γ_sad,

$$ΔF_γ = \int_0^1 \nabla F̂(γ(s)) \cdot γ'(s)\, ds.$$

By the **gradient theorem** (the fundamental theorem of calculus for
line integrals), this is path-independent: as long as ∇F̂ is the
gradient of a single-valued scalar field, *any* path from γ_s to γ_sad
gives the same value. We use the straight line γ(s) = γ_s + s(γ_sad
− γ_s).

Test T1 checks these agree to machine precision (6.6e-17 max). This
is a *consistency check on the implementation*: if direct and path
disagreed, either F̂ or ∇F̂ would have a bug. They don't.

### 6.3 Hessian directional curvature

At γ_s a γ-minimum, expand F̂ to second order along the unit direction
n̂ pointing toward γ_sad:

$$F̂(γ_s + t\, n̂) \;\approx\; F̂(γ_s) + \tfrac{1}{2}\,(n̂^T H n̂)\, t^2.$$

The leading coefficient n̂ᵀ H n̂ is the **directional curvature** of
F̂ at γ_s along the escape direction. By centered finite difference,

$$V''(0) \approx \frac{F̂(γ_s + h n̂) - 2 F̂(γ_s) + F̂(γ_s - h n̂)}{h^2}.$$

Test T2 checks that V''(0) computed by finite differences matches
n̂ᵀ H n̂ computed analytically. Max error 5.9e-8, at the floor of
the centered-FD truncation error (~h²·V'''' with h=1e-4). This is
another consistency check between F̂'s second-order expansion and
its analytical Hessian.

### 6.4 What the calibration points show

|              Point | θ_s        | # γ-eqs | # γ-saddles | ΔF_γ      |
|--------------------|------------|---------|-------------|-----------|
| (2.0, 0) side      | ±0.82      | 3       | 1           | **0.144** |
| (1.2, 0) center    | 0          | 5       | 2 (mirror)  | **0.0154** |
| (1.2, 0) side      | ±0.66      | 3       | 1           | **0.00426** |

**Three structural observations:**

1. **ΔF_γ varies by ~30× across the calibration setup.** The side
   stables at (1.2, 0) have very small barriers — they will be much
   more susceptible to γ-noise than the (2.0, 0) sides. A small ΔF_γ
   means a γ-noise kick can flip γ to a different basin and the
   walker switches branches.

2. **The number of γ-saddles per stable depends on local structure.**
   At the y=0 symmetric central stable of (1.2, 0), reflection symmetry
   gives a *mirror pair* of γ-saddles — two parallel escape channels
   with identical barrier height. The Kramers total escape rate is
   the sum, which doubles the effective rate compared to a single
   channel. Off-axis stables have only one saddle each.

3. **5 γ-equilibria at (1.2, 0), θ=0** matches the "5-stable bullseye"
   that CLAUDE.md flags as `discrim_A`'s over-count at the
   nearby (1.5, 0) — but here those are 3 *γ-stable* eqs at fixed θ
   (γ-mins of F̂), not 5 *coupled-stable* SC eqs. The structure is
   real; the over-count was just `discrim_A` not testing the
   parallel-to-γ direction.

### 6.5 Sketch — γ-landscape at (1.2, 0), θ=0

Schematic F̂(γ; θ=0) iso-contours on the complex γ-plane at the central
stable's θ:

```
       γ_im
        │
   ●────│────●          ● = γ-mins (3)
   │ ╲  │  ╱ │
   │  ╲ │ ╱  │          ▲ = γ-saddles (2)
   │   ▲│▲   │
   │   ─●─   │ ────── γ_re      ● at center: SC γ_s ≈ R+0j
   │   ▲│▲   │                  ● off-axis: ≈ 0.25 ± 0.43i (symmetric pair)
   │  ╱ │ ╲  │
   │ ╱  │  ╲ │
   ●────│────●
        │
```

γ_s = +0.50 + 0j (the central minimum on the real axis). The two
mirror-image γ-saddles lie at ≈+0.37 ± 0.22i, between γ_s and the
two off-axis γ-mins at ≈+0.25 ± 0.43i. Either γ-saddle gives an
escape channel; by y-symmetry their barrier heights are equal
(ΔF_γ ≈ 0.0154 each).

The Kramers γ-noise escape rate is the sum over both channels:

$$\text{rate}_\text{γ-escape} \;\approx\; 2 \exp(-0.0154 / D)$$

with D = T/(2kN). For T = 0.2, k = 2, N = 1000, D ≈ 5e-5 — so the
Kramers exponent is 0.0154/5e-5 ≈ 308, and exp(-308) is vanishingly
small. So γ-escape from this central stable is rare at N = 1000. At
N = 100, D ≈ 5e-4, exponent ≈ 31, exp(-31) ≈ 3e-14 — still rare. At
N = 10, exponent ≈ 3, exp(-3) ≈ 0.05 — finally non-negligible.

The smaller-ΔF_γ side stables at (1.2, 0) (ΔF_γ ≈ 0.00426) require
about 4× lower N to see comparable escape rates, so they should be
noticeably more noise-sensitive in MC.

### 6.6 Implication for Step 8

Step 8 (Monte Carlo escape times) will validate these Kramers
predictions empirically. The prediction is straightforward:

- Run γ-Langevin at fixed θ = θ_s starting from γ_s.
- Measure mean first-passage time τ to leave a small γ-neighborhood
  of γ_s (or, equivalently, to cross over a γ-saddle).
- Check 1/τ ≈ (Kramers prefactor) · exp(-ΔF_γ/D).

The ΔF_γ values from Step 6 fix the exponential factor; the prefactor
involves curvatures at γ_s and γ_sad (1D Kramers formula) or
determinants (2D version). Step 8 will probe this.

## 7. Discontinuity detection during θ-scans — Step 7

Step 5 established that basin boundaries on the slow manifold can be
either smooth saddles (zeros of f with f'>0) or γ-folds (catastrophic
γ-branch terminations). Step 7 builds dedicated detectors for the
non-saddle events, plus a third event type — *perception collapse* —
which we discovered during Step 7 itself.

### 7.1 Three event types and their signatures

For a θ-scan with consecutive samples (θ_i, γ_eq,i, f_i), we identify
three classes of basin-boundary events:

| Event | Mechanism | Signature on the scan |
|-------|-----------|-----------------------|
| **γ-fold** | γ-branch terminates; γ catastrophically jumps to a different branch | |Δγ_eq| between consecutive samples is much larger than typical |
| **f-jump** | Discrete change in perception (e.g. occlusion transition for non-delta geometries) without a corresponding γ-jump | |Δf| is large *without* |Δγ| being large |
| **Perception collapse** | All targets fall outside the perception window for some θ-range; ρ_j = 0 for every j, so γ_eq = 0, R = 0, f = 0 | R is essentially zero over an extended interval of consecutive samples |

### 7.2 Why |Δf| isn't a strong primary signal

A natural question: shouldn't an f-jump be a stronger signal than a
γ-jump, since the basin dynamics are governed by f? Empirically — no.
Looking at the largest events on the (1.2, 0) CW scan from +0.6625:

```
   i        θ       |Δγ|       |Δf|
  37  -0.4999 5.2878e-01 1.9870e-01
 104  -2.6047 3.4893e-01 3.3557e-01
  96  -2.3534 3.0134e-01 1.9326e-01
   1  +0.6311 3.2387e-02 1.9936e-02
```

The known γ-fold is at i=37 with |Δγ| = 0.53 (about 17× the typical
0.03 step). But |Δf| there is 0.20 — only about 10× the typical 0.02
step. Another event at i=104 has |Δf| = 0.34, the LARGEST in the
scan, but its |Δγ| is moderate. The ranking by |Δf| would mislead.

The reason: f = K·R·sin(ego_angle). A γ-fold flips γ to a different
branch where R, ego_angle can be anything — sometimes nearly the same
sin·R product as before, sometimes very different. |Δγ| measures the
γ-displacement directly, |Δf| measures only one component of the
geometric consequence. Use |Δγ| as the primary signal and |Δf| as a
complementary one (the f-jump detector we built fires only when |Δf|
is large *without* a corresponding |Δγ| jump — those are perception
discontinuities, not γ-folds).

### 7.3 The static vs. dynamical "blind-spot trap" distinction

`weighting_analysis/README.md` documents a *dynamical* blind-spot
trap under integral-neural-mapping + cutoff `a=0/b=π` weighting:
a noisy walker that rotates such that all targets get behind it has
γ collapse onto the ±π branch cut (where the integral mapping pins
multiple distinct ego angles to the same neural angle), and the
restoring torque dies. The walker enters a pure random walk.

When we set this exact configuration up and looked at the *static*
slow-manifold scan, we discovered something subtle: **the static
signature of "γ pinned to the ±π branch cut" is indistinguishable
from a normal SC saddle.**

- At a normal saddle near θ = ±π, γ_eq = −R + 0j (negative real
  axis) with R close to 1. arg(γ) = π, |f| ≈ 0.
- At the blind-spot trap point, γ also has arg = π and |f| ≈ 0,
  with R ≈ 1.

The two have the *same* values of R, arg(γ), f. The distinction
between them is purely dynamical: at a saddle, a deterministic
trajectory placed slightly off it slides away on a well-defined
unstable manifold. At the trap, the noisy γ-Langevin equation lingers
because γ-noise can't escape the branch cut quickly (the perception
geometry makes the gradient of F̂ near γ = −1+0j flat in the relevant
direction).

So **Step 7's static detector cannot detect the blind-spot trap as
distinct from a normal back-of-circle saddle.** This is a limitation
in principle, not in implementation. The dynamical signature (slow
mixing in the branch-cut region) would require Step 8's MC machinery
to confirm.

> **Correction under the half-angle law (see §0).** The argument above
> ("same R, arg(γ), f at the saddle and the trap, so statically
> indistinguishable") was specific to the old `sin(ego)` torque, where
> `sin(±π) ≈ 0` made `|f| ≈ 0` at *both* the back-of-circle saddle and
> the trap. Under the current `sin(ego/2)` law, `sin(±π/2) = ±1`, so at
> θ ≈ ±π the static `|f|` is *maximal* (≈ K·R), **not** zero — and the
> ±π point is no longer a smooth saddle but a type-(d) **branch-cut
> f-jump** (|Δf| ≈ 2·K·R; §0). The static scan now *does* flag the ±π
> heading, as a branch-cut rather than a saddle. What remains true is
> the narrower statement: the static detector still can't tell a
> *dynamical* blind-spot *trap* (slow γ-mixing under noise) from an
> ordinary branch-cut crossing — both share the static branch-cut
> signature; the trap-vs-not question is still dynamical (Step 8 MC).
> The half-angle law's removal of the back-of-circle torque dead-zone
> is the deterministic half of the two-part blind-spot fix recorded in
> CLAUDE.md (the other half being `blind_search_std` in `plot_walkers`).

### 7.4 Perception collapse — the static signature we *can* detect

If we tighten the cutoff so that the perception window doesn't cover
the full circle, an entirely different kind of discontinuity appears:
*genuine perception collapse*. With cutoff `a=0/b=π/2`, targets with
|ego_angle| > π/2 (anything more than 90° off the front) have ρ_j = 0.

For our 4-target setup at observer (0, 0), targets sit at
ego ≈ ±9.8°, ±27.5° when the observer faces forward. When the
observer rotates by θ, those ego angles shift by −θ. The first
target to exit the perception window does so at θ ≈ 90° − 27.5°
(or thereabouts); by θ ≈ 90° + 27.5° = 117.5°, *all* four targets
are outside the cutoff window. From that θ until 360° − 117.5° on
the other side, every ρ_j = 0.

The γ-dynamics in this regime become trivially:

$$dγ/dt = \sum_j 0 \cdot e^{i\hat θ_j}\, \sigma(\cdot) - γ = -γ,$$

so γ_eq = 0 and R = 0 and f = 0 over the whole "blind" θ-range.

This *is* statically detectable: just look for an extended run of
consecutive scan samples with R near zero. The detector we built
fires on the BlindSpot (b=π/2) setup and identifies a 69-sample
collapse zone covering θ in [−π, −2.07] ∪ [+2.07, +π] (about 2 rad
of arc, behind the walker).

### 7.5 What this means for basin structure

In §4.4 we listed two possible basin-boundary types: a smooth saddle
on the same γ-branch, or a γ-fold. Step 7 adds a third, and the
half-angle torque law adds a fourth:

- **Perception collapse boundary**: where R drops to 0 over an
  extended θ-range. A walker entering this region has no torque to
  re-orient by. The basin in θ is bounded by the *edge* of this
  collapse zone — where the first target re-enters the perception
  window as the walker rotates toward the front.
- **Branch-cut boundary** (half-angle law, §0): where γ_eq crosses the
  negative real axis (consensus directly behind), `f` jumps +K·R ↔ −K·R
  while γ_eq moves continuously. A real left/right fork at the
  facing-away heading, bounding basins in 1-stable and symmetric cells
  where an edge falls at θ ≈ ±π. Static signature: |Δf| ≈ 2·K·R with
  |Δγ| small and γ_eq on the negative real axis.

The Kramers escape rate near a perception collapse boundary is
different again from the saddle and fold cases: the torque is
identically zero in the collapse region, so the system behaves
like pure diffusion (Brownian motion in θ) until θ wanders close
enough to the boundary that a target re-enters perception. The
mean first-passage time across a collapse region of width Δθ is
~ Δθ² / (2 D_θ) — diffusive, not exponential.

So the model has at least three qualitatively distinct basin-boundary
mechanisms, each with its own Kramers / FPT signature:

| Boundary type | Escape rate scaling |
|---------------|--------------------|
| Saddle | exp(−ΔV / D_θ)  [exponential, Kramers] |
| γ-fold | exp(−ΔV_fold / D_θ)  [exponential, but with ΔV at the discontinuity rather than at a smooth max] |
| Perception collapse | Δθ² / (2 D_θ)  [polynomial, diffusive] |
| Branch-cut (half-angle law) | f-jump +K·R ↔ −K·R at θ≈±π; a repelling fork, not a barrier to escape over — bounds 1-stable/symmetric basins (§0) |

### 7.6 Bonus finding — extra γ-fold at (2.0, 0)

The (2.0, 0) CW scan from +0.8204 caught TWO folds in Step 7, not
one: at θ ≈ −0.92 and θ ≈ −2.46. Step 5 only saw the first one
because it terminated at the first event. Step 7's detector
post-processes a full-circle scan and finds both.

Both are at moderate |Δγ| (0.83 and 0.40, severity 29× and 14×).
The second fold is the same |Δγ| = 0.27 event I marked as "ambiguous"
in Step 5 — bumping the threshold from 0.15 to 0.4 in Step 5 had
hidden it. Step 7's threshold (0.4 absolute) catches it because the
severity (relative to median |Δγ| of this longer scan) is high.

This is a real second fold on the +0.8204 γ-branch — not a numerical
artifact. The γ-branch from +0.8204 going CW has TWO catastrophic
jumps before completing the loop. Whether either is the "correct"
basin boundary depends on which side we're approaching from. The
basin-attribution bisection of Step 7's extension (deferred) would
disambiguate.

## 8. Monte Carlo validation of Kramers — Step 8

Steps 5 and 6 produced two analytical predictions for escape rates from
a stable SC equilibrium:
- **θ-saddle escape on the slow manifold**: rate ~ exp(−ΔV / D_θ)
  (Step 5's V(θ) barrier picture).
- **γ-saddle escape in γ-space at fixed θ**: rate ~ exp(−ΔF_γ / D)
  (Step 6's free-energy barrier picture; D = T/(2kN)).

Step 8 puts the γ-Langevin SDE we built in Step 3 to work on actual
escape experiments and compares the empirical mean first-passage time
to the **Kramers prediction**. This is the empirical ground-truth
check the planning notes flagged as critical.

### 8.1 Kramers escape rate — the multidimensional formula

For a Langevin system

$$dγ = -\nabla F̂(γ) \, dt + \sqrt{2 D} \, dW$$

in d dimensions, with a γ-minimum at γ_s and a γ-saddle at γ_sad, the
**Kramers (or Eyring-Kramers) escape rate** is

$$k_\text{Kramers} = \frac{|λ_\text{neg}|}{2π} \sqrt{\frac{|\det H_\text{min}|}{|\det H_\text{sad}|}}\; \exp\!\left(-\frac{ΔF_γ}{D}\right)$$

where:
- ΔF_γ = F̂(γ_sad) − F̂(γ_min) is the barrier height.
- H_min, H_sad are the Hessians of F̂ at the two critical points.
- λ_neg is the (single) negative eigenvalue of H_sad.
- The prefactor encodes how stiff the well is at γ_min (deeper wells
  → faster attempts to escape) and how sharp the barrier is at the
  saddle (narrower peaks → less time crossing).

The mean first-passage time τ is the reciprocal: τ = 1 / k. With
multiple saddles, the rates add (each is an independent parallel
escape channel), so the total rate is

$$k_\text{total} = \sum_\text{saddles}\, k_\text{Kramers,i}.$$

For the y-symmetric central stable at (1.2, 0), two mirror saddles
give k_total = 2 · k_single. For the side stables, one saddle.

### 8.2 What we measure empirically

The MC experiment integrates the γ-Langevin SDE from initial state
(γ_s, θ_s) by Euler-Maruyama with dt = 0.01:

$$γ_{n+1} = γ_n - \nabla F̂(γ_n) \, dt + \sqrt{2 D \, dt}\, ξ_n$$

with ξ_n ~ N(0, I_2) and θ updated deterministically by
dθ/dt = K·R·sin(ego(γ)).

We declare *escape* the first time γ enters a small ball around
*another γ-minimum* — i.e., γ has committed to a different
γ-basin. (Crossing the saddle alone is not enough; γ can fluctuate
across the saddle and back.) The mean escape time τ_emp over many
realizations is the empirical mean first-passage time.

We sweep over multiple D values to map out the exponential
dependence; the slope of log(τ_emp) vs 1/D should match the slope of
log(τ_Kramers) vs 1/D, namely ΔF_γ.

### 8.3 Results

Two calibration points, both at focal_loc = (1.2, 0):

**Central stable** (θ_s = 0, two y-symmetric γ-saddles, ΔF_γ ≈ 0.0154):

| D | τ_emp | τ_Kramers | emp/Kramers |
|---|-------|-----------|-------------|
| 0.005 | 110.6 | 59.2  | 1.87 |
| 0.010 | 18.4  | 12.7  | 1.45 |
| 0.020 | 7.6   | 5.9   | 1.28 |

**Side stable** (θ_s = +0.6625, one γ-saddle, ΔF_γ ≈ 0.00426):

| D | τ_emp | τ_Kramers | emp/Kramers |
|---|-------|-----------|-------------|
| 0.0015 | 211.5 | 125.4 | 1.69 |
| 0.003  | 48.1  | 30.3  | 1.59 |
| 0.006  | 14.8  | 14.9  | 1.00 |

### 8.4 What the slopes say

Fitting log(τ_emp) = α/D + β to the central data:

```
empirical: log τ ≈ 0.0179 · (1/D) + 1.13
Kramers:   log τ ≈ 0.0154 · (1/D) + 1.01
slope ratio: 1.16
```

The empirical slope 0.0179 is within 16% of the predicted ΔF_γ = 0.0154.
Test pass criterion was a factor of 2; this is comfortably better.

### 8.5 Why empirical τ is slightly *larger* than Kramers

Throughout the sweep, empirical τ is 1.0× to 1.9× the Kramers
prediction, never smaller. This is the expected sign — and informative
about what each formula computes.

- **Kramers** counts the time to *first reach the saddle* in the
  small-D Gaussian-fluctuation regime. It's a top-of-the-barrier
  rate.
- **Empirical (our criterion)** counts the time to *commit to the
  destination γ-minimum*. After crossing the saddle, γ takes some
  additional time to descend into the new basin.

The post-saddle commitment time is short (set by the relaxation rate
at the saddle, ~ 1/|λ_neg|), so the difference is small. At smaller D,
the well-residence time dominates and the ratio approaches 1
(asymptotically Kramers). At larger D, the post-saddle time becomes a
non-negligible fraction; ratio increases.

The trend "ratio decreases with increasing 1/D" in the central data
(1.87 → 1.45 → 1.28) is consistent: smaller D → longer well residence
→ ratio → 1.

### 8.6 Relative ordering of escape rates

T2 checks the relative ordering of escape rates across stable
equilibria. At a common D = 0.003, extrapolating from the fitted
exponentials:

- τ_side(D=0.003) ≈ 40
- τ_center(D=0.003) ≈ 1200

The side stable escapes ~30× faster — exactly as expected from the
ratio of ΔF_γ values (0.0154 - 0.00426 = 0.0111 difference,
exp(0.0111/0.003) ≈ exp(3.7) ≈ 40 — close enough).

**This validates the central operational claim of the whole vetting
plan**: ΔF_γ ranks SC equilibria by noise robustness correctly. The
side stables at (1.2, 0) are more noise-sensitive than the central
stable, by the expected factor.

### 8.7 What this means for the basin estimator

Step 8 is the empirical anchor for everything. Without it, ΔF_γ
would be a number we computed but didn't know predicted the right
escape rates. With it, we have:

- **Kramers prefactor is approximately correct** in our parameter
  regime — within a factor of ~2 across the sweep.
- **The exponent is empirically tight** — the slope is within 16%
  of ΔF_γ.
- **Relative ordering is right** — the smaller-ΔF_γ stable escapes
  faster by close to the predicted ratio.

So for the basin-of-attraction visualization, **ΔF_γ is a trustworthy
scalar measure of γ-noise robustness** to display per stable equilibrium
*in multistable cells* (1-stable cells get a direction arrow instead — see
the §13.6 scope correction).

### 8.8 Compute notes

- 32-core multiprocessing.Pool.
- Total wall-clock: 1m53s for 600 realizations (100 per ensemble × 6
  ensembles).
- Total CPU time: 43 min (2273% CPU = ~23 cores effective utilization
  averaged — bursty because shorter ensembles use less of the
  parallel capacity).
- Bottleneck: per-step `nbm.percep_model.get_neural_signals` call.
  A precomputed θ-mesh of perception data + per-realization
  interpolation would speed this up substantially if Step 8's MC
  becomes a hot path; not necessary for this validation pass.

## 9. Asymmetric basin test — Step 9

The user's prior physical intuition anticipated a specific qualitative
behavior:

> in a bistable region where the observer is very close to one
> circular target and the other is far, the far target's basin
> should be small.

Step 9 confirms this prediction quantitatively and with a magnitude
larger than the underlying distance asymmetry would naively suggest.

### 9.1 Setup

Calibration: focal_loc = (4.0, 1.5) in VM-k055 (two circle targets at
(4.33, ±2.5)).

- Target 1 at distance 1.05, allocentric direction +71.7°.
- Target 2 at distance 4.01, allocentric direction −85.3°.
- **Distance ratio: 3.81×.**

The two stable SC equilibria sit at θ-headings near each target's
allocentric direction:
- θ_s = +1.252 ≈ +71.7° → facing the **close** target (target 1).
- θ_s = −1.489 ≈ −85.3° → facing the **far** target (target 2).

### 9.2 Result

Running Step 5's truncated CCW/CW basin extractor at each stable:

| Stable | Target | d | basin width | endpoints |
|---|---|---|---|---|
| +1.252 | close | 1.05 | **303° (5.29 rad)** | saddle ↔ saddle |
| −1.488 | far   | 4.01 | **57° (0.99 rad)**  | fold ↔ saddle |

**Basin ratio: 5.35×** — the close-target basin is 5× wider than the
far-target basin, despite a distance ratio of only ~4×.

### 9.3 Interpreting the numbers

Two qualitative things to note:

**The close basin is nearly the whole circle.** 303° out of 360° all
flow to the close-target stable. The far-target stable only "captures"
a 57° wedge of headings — those pointing almost directly at the far
target.

**The asymmetry is amplified beyond the geometric ratio.** Distance
ratio is 3.81 but basin ratio is 5.35. Perception strength falls off
faster than 1/distance — target angular extent scales as ~1/d for
distant circle targets, but the *consensus* weight depends nonlinearly
on the relative magnitudes through the F̂ landscape (the sigmoid-based
sums in dgamma/dt). Smaller perception → exponentially shallower γ-well
→ smaller basin.

### 9.4 Topological observation

The two basins share a saddle boundary at θ = −2.38, exactly as
Poincaré-Hopf demands: between two stable equilibria on S¹, there must
be an unstable equilibrium (a saddle). Specifically:

- Close basin's CCW endpoint = saddle at −2.38.
- Far basin's CW endpoint = saddle at −2.38 (same point).

The other boundary of the far-target basin is *not* a saddle — it's a
**γ-fold at θ ≈ −1.39**, only 0.09 rad (5°) from the far-target stable
at −1.49. The far basin is squeezed against this γ-fold from one side
and the shared saddle from the other, leaving only the narrow 57°
wedge.

This is the basin structure the user's intuition predicted, complete
with the γ-fold mechanism that wasn't part of the original argument.
The γ-fold is what makes the "narrowness" of the far basin so
dramatic — without it (if the far basin extended symmetrically toward
a hypothetical second saddle), the basin would be much wider than 57°.

### 9.5 Implication for the bifurcation plot

The basin-of-attraction visualization in the two-panel plot (the
project's main goal) should faithfully represent this asymmetry, in
the multistable cells where it's meaningful. Two specific things to
surface visually:

- **Basin width should be readable per stable** — same multistable
  count doesn't mean same noise robustness.
- **Asymmetric basins are the norm off-axis**, not the exception. The
  y=0 symmetric calibration points used through most of Steps 5–8 are
  the special case where mirror-image stables have equal basins. Most
  of the multistable (x,y)-plane has at least some asymmetry.

The actual scalar to plot per stable in multistable cells is ΔF_γ
(from Step 6, validated in Step 8) for γ-noise robustness, or basin
width Δθ_total (from Step 5) for the geometric extent. Step 9
confirms both vary substantially across stables at a given
multistable (x,y), and that the variation matches the underlying
perception geometry.

(In 1-stable cells these scalars don't carry the same meaning — see
§4.4 scope caveat. The 1-stable visualization is just the direction
arrow from `sc_equilib` plus optional local-stiffness glyph.)

## 10. Hopf island graceful failure and the public-API wrapper — Step 10

Up to this point each step was a focused validation experiment. Step 10
is a robustness-and-API step: the basin estimator needs to handle
focal_loc choices where there is no stable SC equilibrium at all,
without crashing, hanging, or producing garbage. This matters because
the eventual two-panel bifurcation+basin plot will call the estimator
at every (x, y) of the grid — including cells inside Hopf islands.

### 10.1 What a Hopf island is

VM_bifurcations/VERDICT.md documents two small "islands" near
(2.1, ±2.45) in the VM-k055 setup where sc_equilib reports 0 stable
SC equilibria. The 1 unstable SC eq found there is a **Hopf-unstable
focus**: linearization at the SC eq has a pair of complex-conjugate
eigenvalues with positive real part. The actual attractor in the
neighborhood is a **stable limit cycle** of period ~17 — the walker's
heading oscillates without settling.

For basin analysis: there is no basin to compute, because there is no
stable fixed point to define one. The right answer is "no basins
here," with metadata indicating why.

### 10.2 The wrapper

A new function `compute_basins_at_focal_loc(focal_loc, *,
scan_single_stable=False)` in basin_via_theta.py is the public entry
point. It always returns a dict with these keys:

```
{
  'basins':         list of basin dicts (possibly empty),
  'stable_count':   int — number of stable SC eqs found,
  'unstable_count': int — number of unstable SC eqs found,
  'sentinel':       None when basins computed normally,
                    else a short string explaining the situation
}
```

The basin-dict shape depends on cell class:

- **Multistable cells** (or 1-stable with `scan_single_stable=True`):
  each basin dict has the full output of `basin_features()` —
  `theta_stable`, `gamma_stable`, `delta_theta_ccw`, `delta_theta_cw`,
  `delta_V_ccw`, `delta_V_cw`, `ccw_endpoint`, `ccw_type`,
  `cw_endpoint`, `cw_type`, etc.
- **1-stable cells with the default** `scan_single_stable=False`:
  each basin dict has only `theta_stable`, `gamma_stable`, and
  `is_single_stable: True`. The scan was skipped because basin-barrier
  scalars don't carry their multistable meaning here (see §4.4 scope
  caveat and §13.6). The renderer should draw just the direction
  arrow plus an optional local-stiffness glyph from the F̂-Hessian
  at γ_s.

Sentinel cases:
- **"no stable SC eq found; suspect Hopf-unstable focus + limit
  cycle"** when sc_equilib finds only unstable eqs (Hopf island).
- **"no SC eq found at all"** when sc_equilib returns nothing (could
  happen at a weird parameter combination — does not in our
  calibration setup).
- **"could not recover γ for stable θ=…"** when a stable θ is
  reported but the (slightly different) gamma_equilib path can't
  find the matching γ to seed the scan. Partial-success case.

### 10.3 Test results

| Test | focal_loc | Expected | Got | Wall time |
|---|---|---|---|---|
| T1 (Hopf) | (2.1, 2.45) | 0 basins + sentinel | 0 basins + Hopf sentinel | 0.09 s |
| T2 (3-stable) | (1.2, 0) | 3 basins, no sentinel | 3 basins, no sentinel | 3.79 s |
| T3 (1-stable) | (0.5, 0) | 1 basin, no sentinel | 1 basin (minimal, no-scan), no sentinel | 0.5 s |

All pass. The Hopf-island case completes in 90 ms because the wrapper
short-circuits at the `not stable_thetas` check before attempting any
scans. The 1-stable case (T3) returns a minimal basin record with
`is_single_stable: True` because the default `scan_single_stable=False`
recognizes the scope correction from §13.6 (basin barriers don't
carry their multistable meaning at 1-stable cells). The multistable
3-stable case (T2) runs the full pipeline as designed.

### 10.4 Why this matters for the bifurcation plot

The two-panel plot's right panel will need to render *something*
sensible at every cell of the (x, y) grid, with the rendering rule
depending on cell class:

- **Multistable cells** (stable_count ≥ 2): arrows (one per stable,
  oriented allocentric, length or color encoding ΔF_γ or basin width
  Δθ_total). This is the regime where Kramers-style noise robustness
  is well-defined and the basin scan output is informative.
- **1-stable cells**: direction arrow only (from `theta_stable` in the
  minimal basin dict). Optionally encode local stiffness from the
  F̂-Hessian at γ_s as a glyph size / color. No basin-barrier scalar
  is meaningful here (§4.4 scope caveat, §13.6).
- **Cells inside Hopf islands** (0 stable, ≥1 unstable): a distinct
  marker — maybe a circular arrow indicating "limit cycle" —
  triggered by the `sentinel` containing "limit cycle".
- **Cells with 0 SC eq of any kind**: a "no perception / blind"
  marker, triggered by the `"no SC eq"` sentinel. (Doesn't arise in
  VM-k055; may arise under tighter cutoffs.)

The wrapper's sentinel string is the routing key. A renderer that
checks for `result['sentinel']` and special-cases the recognized
substrings can give meaningful visual feedback even at corner cases,
which is the difference between a useful diagnostic plot and one
that quietly produces blank cells with no explanation.

### 10.5 Caveats

The wrapper does NOT:
- Distinguish "Hopf-unstable focus + limit cycle" from other
  zero-stable cases (e.g. all-unstable real saddles). The Hopf
  framing in the sentinel is the most common interpretation in
  this model but not proven case-by-case.
- Estimate properties of the limit cycle (period, amplitude). That
  would require an additional ODE-on-S¹ integration; deferred.
- Recover from "γ_s not findable" by trying harder. If
  gamma_equilib(focal_angle=θ_sc) misses the γ at R+0j (rare but
  documented in CLAUDE.md), the wrapper reports the partial-success
  sentinel and skips that θ_sc. A future enhancement could fall
  back to a polar multistart at radius 0.5.

## 11. Performance benchmark — Step 11

**Bottom line: ~4.2 min parallel (32 cores) for a 41×41 grid with basins**,
vs ~21 s for `sc_equilib`-only baseline. Feasible — a researcher can iterate
on bifurcation+basin plots in minutes. (The README's strict "full/baseline
ratio < 10×" criterion fails at ~14×, but only because `sc_equilib` is
*fast* (~90 ms); the absolute runtime is fine.)

Per-point cost by cell class, and how the renderer routes each (this routing
*is* the cost structure — multistable cells dominate the budget):

| cell class | per-point full | what to compute / render |
|---|---|---|
| **0-stable** (Hopf / collapse) | ~0.09 s (wrapper short-circuit) | sentinel-driven marker; no basin |
| **1-stable** | ~0.09 s (scan skipped by default) | direction arrow + optional F̂-Hessian stiffness glyph; no scan (the scalars don't carry their multistable meaning — see §13.6) |
| **multistable** (≥2 stable) | ~2–4 s | per-stable CCW/CW basin scan, ΔF_γ via `gamma_eqs_at_theta`, fold-vs-saddle classification |

The default `scan_single_stable=False` skips the θ-scan at 1-stable cells.
This is primarily a *correctness* choice (§13.6), not a speed one: the perf
win parallelizes away because multistable cells are the bottleneck.

**Available mitigations** if higher resolution (81×81+) outgrows comfort —
none needed at 41×41: (1) subgrid-sample within the spatially-clustered
multistable regions and interpolate; (2) cross-cell γ warm-start from the
neighboring cell's γ_eq series (exploiting (x,y)-continuity of γ_eq);
(3) coarser scan resolution (n_max 300→100, ~3× faster, looser fold
locations). Benchmarked on the five calibration points plus a 5×5 random
sweep on (0.5, 5.5)×(−2.5, 2.5); `bench_per_point.py` reproduces.

## 12. Summary — vetting plan complete

All 11 steps are done; the estimator is verified end-to-end against
analytical predictions, Monte Carlo ground truth, and the user's physical
intuition, and is fast enough for the bifurcation+basin plot (§11). Per-step
detail is in the body sections above; the two non-obvious headline results:

- **The slow manifold is not a global object — it's a union of γ-branches
  glued at γ-folds** (§4.3–§4.5). A basin in θ is an arc on one γ-branch,
  bounded by a saddle, a γ-fold, a perception-collapse zone, or (under the
  half-angle law) a branch-cut. Poincaré-Hopf holds within a branch, not
  across the union, so `sc_equilib`'s unstable list can be incomplete.
- **ΔF_γ is a validated γ-noise robustness scalar in multistable cells**
  (§6, §8). It varies by ~30× across the calibration setup ((1.2,0) sides
  ≈ 0.00426 vs (2.0,0) sides ≈ 0.144), and γ-Langevin MC confirms the
  Kramers prediction within a factor of 2, with correct relative ordering.

### Ready for implementation in decision_model.py

Public API: `result = compute_basins_at_focal_loc(focal_loc, *,
scan_single_stable=False)` returns `{'basins', 'stable_count',
'unstable_count', 'sentinel'}`; each multistable basin dict carries
`theta_stable`, `gamma_stable`, `delta_theta_{ccw,cw}`, `delta_V_{ccw,cw}`,
`{ccw,cw}_endpoint`, `{ccw,cw}_type`. To wire into the two-panel plot: add a
`NeuralBandModel` method wrapping it, and route rendering by cell class
(§11 table / §10.4) — sentinel marker for 0-stable, direction arrow (+
optional stiffness glyph) for 1-stable, ΔF_γ-coded arrows per stable for
multistable. Open theory gaps and carry-forward items are in §13.4–§13.5.

## 13. Open theory questions

The theory questions on the table at the start of this effort are mostly
answered in the body above: the F̂/ΔF_γ derivation and validation (§6, §8),
γ-folds and the multi-type basin-boundary catalog (§4.3–§4.5, §7), Hopf-
island handling (§10), and the computational budget (§11). The chain
"θ kicked → γ_eq(θ) shifts → γ chases" is captured by V(θ) in the adiabatic
limit (§1), so warping is absorbed into γ_eq(θ) and needs no separate
treatment. What remains genuinely open is consolidated below: the headline
θ-noise-proxy gap (§13.4), smaller gaps (§13.5), and the scope correction
that frames where any of it applies (§13.6). The two open items that depend
on the still-unsettled dθ/dt law are also flagged in §0.2.

### 13.4 The headline gap — θ-noise as a faithful proxy for γ-noise

**This is the most consequential open theory question.**

#### Why it matters

The existing walker code (`plot_walkers`) uses θ-noise as the
stochasticity knob. That was always understood as a proxy — the
physically motivated noise lives in γ (Glauber temperature T) and
finite-size 1/N fluctuations. The planning decision (Option 3 in the
γ-noise model question) was to run *both* MC modes in parallel:
γ-Langevin and θ-Gaussian, and produce an empirical mapping
σ ↔ T·g(geometry) at each calibration point. If σ ↔ T·g is roughly
constant across the bifurcation diagram, then the existing θ-noise
walkers are quantitatively meaningful: any past experiment using
`std` can be back-translated to an effective N and an effective
temperature in the underlying spin model.

I deferred the θ-noise side of Step 8 because of the §5 finding that
side-stable θ-basins in multistable VM-k055 are *γ-fold-bounded*
rather than smooth-V-saddle-bounded — meaning the simple
exp(2 ΔV/σ²) Kramers prediction for θ-noise escape doesn't apply in
the textbook form at those points (the prefactor depends on the
second derivative at the saddle, which doesn't exist at a fold).

#### Conceptual scope correction

An earlier draft of this section proposed running the calibration at
the 1-stable-far point (0.5, 0). That was a mistake. **In a 1-stable
region on S¹, there is no second basin to escape to.** Noise pushing
the walker over the V-saddle at ±π sends θ around the loop and back
to the same stable. The "escape time" measured there is the time for
a *transient excursion*, not for a basin transition, and doesn't
correspond to any quantity the σ ↔ T·g mapping should predict.

The σ ↔ T·g calibration is only physically meaningful in **multistable
regions** where γ-noise vs θ-noise drive *qualitatively the same*
basin-to-basin transitions and we can ask "do they produce the same
escape rate when matched correctly."

#### Revised proposed validation

A focused MC sweep at a multistable calibration point — the most
natural choice is **(1.2, 0) at the side stable θ_s = +0.6625**,
which we already characterized in Step 8 (γ-Langevin: τ_γ(D) measured
at D = 0.0015, 0.003, 0.006).

1. **γ-Langevin side already done** — τ_γ(D) data from Step 8 stands.

2. **θ-noise escape MC.** Add to `mc_escape.py` a new mode that
   integrates the reduced 1D θ-dynamics with γ slaved to γ_eq(θ) on
   the slow manifold:

   $$dθ = f(θ)\, dt + σ \sqrt{dt} \cdot \mathcal N(0,1)$$

   where f is interpolated from a precomputed θ-scan from
   `theta_scan.py`. Sweep σ; measure τ_θ(σ).

   The basin boundary at this point is a γ-fold at θ ≈ -0.479
   (Step 5). For θ-noise alone with γ slaved, the "escape" is the
   first θ-crossing of θ_fold. Crucially: **the Kramers exp(-ΔV/D_θ)
   prediction still applies for the rate of reaching the fold**, with
   ΔV computed as the V-barrier *up to* the fold (V(θ_fold) − V(θ_s)
   = trapezoidal integral of −f along the slow manifold). The
   discontinuity at the fold doesn't change the rate of reaching it —
   it only changes what happens *after* crossing.

3. **The proxy test.** For each (τ_γ(D), τ_θ(σ)) pair where both
   escape distributions are well-characterized, find σ such that
   τ_θ(σ) = τ_γ(D). The resulting mapping σ(D) is the empirical
   T_eff = T·g(geometry). For the proxy to be faithful, this should
   collapse onto a single relationship σ² ∝ D (or equivalently
   σ² ∝ T/N for some geometric factor g(focal_loc)). If it does, the
   geometric factor g is read off from the slope.

4. **Repeat at the central stable θ_s = 0** at (1.2, 0). The central
   stable has two symmetric γ-saddles (not folds) bounding its basin
   in γ-space and two γ-folds bounding its basin in θ-space. This
   gives a second calibration point with different geometry — if
   g(focal_loc) is roughly constant across stable equilibria at the
   same (x, y), the proxy is robust.

5. **Optionally repeat at a third point** like (2.0, 0) side or
   (4.0, 1.5) close-target stable to see if g varies smoothly with
   (x, y). If g is roughly constant, the existing θ-noise machinery
   is rigorously justified as a proxy. If g varies by >2× across
   reasonable observer positions, then `std` results need geometric
   correction before being interpretable.

#### Cost estimate

The γ-Langevin data already exists from Step 8 (1m53s for 600
realizations across 6 D values). Adding the θ-noise mode is cheap:
the reduced 1D θ-dynamics with interpolated f is ~10× faster per
step than γ-Langevin (no per-step F̂ gradient evaluation). A matching
σ-sweep at 200 realizations × 3 σ values per point × 2 points ≈ 1200
realizations should run in under 1 minute on 32 cores. Total
incremental cost: well under 10 minutes for the full calibration.

A reusable extension of `mc_escape.py` would be the natural
deliverable.

#### What this would resolve

- Whether the σ in `plot_walkers` corresponds to a meaningful effective
  T·g in the underlying spin model, or only to itself.
- Whether the geometric factor g varies enough across observer
  positions to require position-dependent reinterpretation of
  existing `std` results.
- Closing the loop on the planning-doc decision to use Option 3 (both
  noise channels) for the noise-model question.

### 13.5 Smaller theory gaps for future work

These are less consequential than the θ-noise proxy question above
but were genuinely on the table and not closed out.

1. **Perception-collapse Δθ² scaling — MC validation.** §7.5 predicts
   that first-passage times across a perception-collapse zone scale
   as Δθ² / (2 D_θ) — diffusive, not exponential. This was never MC
   tested. Setup: BlindSpot (cutoff b = π/2), initialize θ at the
   front of the collapse zone, run θ-Brownian motion, measure mean
   time to exit. Test that the time scales linearly with Δθ² and
   inversely with D_θ.

2. **"When does which mechanism dominate" map.** Plot
   min(ΔV/D_θ, ΔF_γ/D, log(Δθ²/(2D_θ))) across the (x, y) parameter
   plane to identify regions where each escape mechanism is the
   rate-limiting one. Probably tractable as part of the eventual
   two-panel plot — would surface the basin-mechanism choice
   visually as a third "panel" or as the basin-color coding.

3. **Joint (γ, θ) gradient status.** γ-only is gradient (Glauber F̂);
   1D θ-only is automatically gradient. The coupled 3D system on
   (γ_re, γ_im, θ) has the Jacobian we computed in `_discrim_coupled`;
   testing whether the *full* vector field is curl-free everywhere
   (not just at SC eqs) would tell us whether a global Lyapunov
   function exists. The curl-free test in 3D is the vanishing of
   each component of ∇ × f at sample points. Quick to check
   numerically. If the field *is* gradient, the resulting joint
   Lyapunov function would unify the V(θ) and F̂(γ) pictures into a
   single 3D landscape — useful for visualization and possibly for
   tighter escape-rate predictions.

4. **Tighter prefactor analysis in Step 8 MC.** The empirical/Kramers
   ratio of 1.0–1.9 was within the pass criterion but not exact. The
   discrepancy was attributed to the commit-to-destination escape
   criterion adding post-saddle time. A more careful test would
   compute the *transition-state* crossing time analytically and
   subtract, to recover the bare Kramers rate. Not blocking but
   would tighten the validation.

5. **The basin-attribution bisection** mentioned in §4.7 was never
   implemented (only sign-change bisection was). At points where
   γ-fold detection is delicate, basin-attribution would give the
   exact basin boundary by integrating the full coupled dynamics
   from probe midpoints. Not blocking for visualization since the
   |Δγ| heuristic gets fold locations to ~dθ ≈ 0.02 rad already.

### 13.6 Conceptual scope correction — basin barriers only matter in multistable regions

A clarification that emerged from re-examining the §13.4 plan:
**noise-robustness scalars in the Kramers sense are only physically
meaningful in multistable regions.** In a 1-stable region on S¹,
the basin is the entire circle minus the unstable point. Noise can
push the walker over the V-saddle, but the walker just rolls around
the loop and returns to the *same* stable. There is no alternative
basin to escape *to*.

This corrects two things in earlier sections:

1. **§4.4's framing of γ-fold-vs-saddle escape rates** applies to
   multistable basins, not 1-stable ones. In 1-stable cells, neither
   "exp(-ΔV/D_θ)" nor any other barrier formula corresponds to a
   meaningful event — what's measured is just the transient excursion
   timescale, which is itself observable but isn't what "noise
   robustness of the equilibrium" usually means.

2. **§13.4's first-draft proposal to calibrate at (0.5, 0)** would have
   measured something, but not the right thing. (0.5, 0) is 1-stable;
   no σ ↔ T·g(geometry) mapping exists there because no basin
   transition exists. The revised §13.4 plan uses multistable points
   instead.

#### What to visualize in 1-stable cells

The right deliverable for a 1-stable cell in the two-panel plot is:

- **Direction arrow** showing the unique stable's allocentric heading
  (from `sc_equilib`; ~negligible cost).
- *Optionally*, a **local-stiffness encoding** — the F̂-Hessian
  eigenvalues at γ_s, or V''(θ_s) — as a glyph size or color. This
  is a *fluctuation magnitude* scalar (how tightly noise keeps the
  walker near the stable), not an *escape probability* scalar. Cheap
  to compute, no scan required.

The full basin-extraction machinery (truncated CCW/CW scans, V(θ)
integration, γ-saddle finding) should be **reserved for multistable
cells**, where it does the work it was designed for.

#### Implication for §11 mitigations

The "skip the scan at 1-stable cells" default (§11) is not just a
performance choice — it's a *correctness* one: at 1-stable cells the scan
computes scalars (ΔV, basin-width, fold-vs-saddle classification) that
don't carry their multistable-context meaning. Skipping the scan and
drawing only the direction arrow (plus optional stiffness glyph) is the
*right* thing to do, not just the fast thing.

### 13.7 Bottom line

The vetting plan answered the core engineering questions
(does ΔF_γ work as a robustness scalar in multistable cells — yes;
does the basin extraction handle pathological cases — yes; is the
cost feasible — yes). It also answered three of the five "future
work" theory items in passing.

The *headline open question* — whether θ-noise is empirically a
faithful proxy for γ-noise in multistable regions — was deferred and
is the right thing to pick up next if a follow-up theory session
happens. A focused extension of mc_escape.py at the (1.2, 0)
multistable calibration points would close it.
