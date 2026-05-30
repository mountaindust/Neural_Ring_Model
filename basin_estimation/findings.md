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
| 1  | The state space and the slow manifold | foundation |
| 2  | γ-Langevin dynamics and what they tell us | 3 |
| 3  | Slow-manifold θ-scan and γ-continuation | 4 |
| 4  | Basins, effective potential, and the multistable case | 5 |
| 5  | The Schur complement and slow eigenvalues | 5 (technical) |

## 1. The state space and the slow manifold

The deterministic dynamics live on (γ_re, γ_im, θ) ∈ ℝ² × S¹. The first
two coordinates are the real and imaginary parts of the neural
coherence value; the third is the observer's heading, on the circle.

The dynamical equations have a built-in **separation of timescales**:

- γ relaxes on its own timescale, set by the Hessian of F̂. In our
  calibration setup the Hessian eigenvalues at the SC equilibria are
  ~1, so γ relaxes on time ~1.
- θ moves under the torque dθ/dt = K · R · sin(ego_angle), with K = 1
  in our setup. Near a stable SC equilibrium R ~ 0.6 and ego_angle ~ 0,
  so dθ/dt is small and θ moves on time ~2.

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
stable equilibrium can be bounded by:

- **(a) A saddle on the same γ-branch.** A smooth zero of f(θ) with
  f' > 0. The walker's θ flows back to the stable from one side and
  away from it on the other side. The Kramers escape rate near this
  boundary involves the V-barrier height: rate ~ exp(−ΔV/D_θ).

- **(b) A γ-fold.** The γ-branch terminates. A walker perturbed
  past the fold loses its γ-branch entirely; γ jumps to a different
  branch, and the walker may flow to a different stable. The
  Kramers exponential doesn't apply in the standard form here —
  the escape isn't over a smooth barrier but past a discontinuity.

Either way, the basin in θ is an arc bounded by these events. Step 5's
truncated CCW/CW scans terminate at the first such event in each
direction.

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

## 6. Summary and what's next

**Solid foundations (Steps 1–4):**

- F̂(γ) derivation: verified to machine precision against `dgamma_dt`.
- γ-Langevin SDE: Boltzmann stationary distribution, calibration
  D = T/(2kN), local Gaussian, 1/N scaling all verified.
- Slow-manifold θ-scan with warm-start γ-continuation: tracks one
  γ-branch correctly.

**The substantive finding (Step 5):**

The slow manifold projecting the (γ, θ) dynamics onto 1D heading
dynamics is not a global object — it's a *union of γ-branches* glued
at γ-folds. A basin in θ around a stable SC equilibrium is an arc on
one γ-branch, bounded by either a saddle (smooth zero of f on the
same branch) or a γ-fold (a discontinuity where the branch terminates
and γ jumps). The Poincaré-Hopf theorem (#stable = #unstable on S¹)
holds within each branch but not across the branch union. sc_equilib's
saddle list is necessarily incomplete because it restricts to γ on
the positive real axis and to one γ-branch at a time. Sign-change
bisection and basin-attribution bisection answer different questions;
the latter is what the walker actually experiences.

**Outstanding:**

- Step 6: γ-saddle finding at fixed θ and ΔF_γ. At fixed θ, the
  γ-fold issue doesn't apply — we're looking at the γ-Hessian
  landscape of F̂ at a single SC equilibrium, finding γ-saddles
  in the complex γ-plane.
- Step 7: proper basin-attribution bisection for finding fold
  boundaries from the dynamics rather than from γ-jump heuristics.
- Step 8: Monte Carlo validation of basin sizes via γ-Langevin
  escape times — empirical ground truth for everything above.
