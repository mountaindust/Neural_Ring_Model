# Basin-of-attraction estimator — planning session transcript

Session on planning the two-panel bifurcation + basin-of-attraction plot
(the TODO at the bottom of CLAUDE.md). No implementation yet. This file
preserves the design discussion, including theory points raised that the
user wants to revisit in a separate session.

---

## Original request

> I would like to work on the two-panel bifurcation + basin-of-attraction
> plot mentioned in the CLAUDE.md todo next. Before we embark on any
> implementation in decision_model.py, I would like to plan this out,
> particularly in terms of how we can get good (and computationally
> efficient and stable) estimates for the basin of attraction for each
> stable equilibrium point. First, please review the relevant theory, the
> notes on the project, and the relevant parts of the code including how
> equilibria are found. Then, propose ways that we could estimate the
> basin of attraction for each stable equilibrium in a bifurcation diagram
> with potential pros and cons (assuming there is more than one obvious
> approach). My thinking is that any such method (or methods) should be
> tested and vetted before we make an actual implementation.

---

## Initial proposal

### 1. What "basin" actually means in this model

State-space candidates:

| candidate | description | how walkers move in it |
|---|---|---|
| (γ_re, γ_im, θ) ∈ ℝ²×S¹ | full coupled 3-equation system | not directly — γ relaxes faster than θ |
| θ ∈ S¹ along the slow manifold γ = γ_eq(θ) | reduced 1D dynamics on the heading | exactly what `plot_walkers` simulates |

`plot_walkers` calls `dtheta_dt` → `run_dgamma_dt` to steady state at
every step (decision_model.py:2573–2576, 3199). γ is slaved adiabatically
to θ; noise is added to θ only (3196–3199). So the physically meaningful
basin is in θ-space on the slow manifold, and basin boundaries are
points where the reduced θ-dynamics have unstable equilibria (or, in
Hopf cases, where the slow manifold disappears or develops a limit
cycle).

Reduced dynamics:
$$\dot\theta = f(\theta) := K \cdot R(\theta) \cdot \sin\bigl(\text{ego\_angle}(\theta)\bigr).$$

A 1D ODE is automatically gradient: V'(θ) = −f(θ), so an effective
potential V(θ) exists for free. SC stable equilibria are local minima of
V; saddles bounding their basins are local maxima of V.

Noise is angular Gaussian with std `std`, applied as `noise*dt` per
Euler step (`plot_walkers`, line 3199). In Langevin form
θ̇ = f(θ) + std·dW (variance per unit time = std²), so Kramers escape
rate ~ exp(−2·ΔV/std²) where ΔV = V(saddle) − V(stable).

### 2. What we need at each grid point

For each (x,y) and each stable SC equilibrium θₛᵢ, one or two scalars:

- **Half-arcs** Δθ⁻, Δθ⁺ — distance in θ to the left/right saddle (or
  boundary feature). Geometric, intuitive.
- **Barrier heights** ΔV⁻, ΔV⁺ — energy to escape each direction.
  Quantitative; converts directly to a Kramers exponent.

### 3. Candidate methods

**Method A — trust the existing unstable-equilibrium output of `sc_equilib`.**
Zero extra code, but user has already flagged that the finder misses
unstable equilibria; gives only geometric basin width; says nothing
about Hopf-island boundaries.

**Method B — θ-bisection between known stable equilibria.**
Robust; only needs forward integration; works even if `sc_equilib`
misses the saddle. Doesn't give barrier height; ~10 s per pair worst
case.

**Method C — θ-scan along the slow manifold (recommended primary).**
For each grid point:

1. Sample θ on a dense circular mesh (100–200 points).
2. At each θ, compute γ_eq(θ) using γ-continuation (LSODA seeded from
   the previous θ's γ_eq, not from γ=10⁻⁵).
3. Compute f(θ) = K·R(θ)·sin(ego_angle(θ)) at every mesh point.
4. Numerically integrate V'(θ) = −f(θ) to get V(θ).
5. Locate sign changes of f to refine saddle/stable θ via brentq.
6. For each stable θₛ, the two adjacent saddles give Δθ⁻, Δθ⁺, ΔV⁻, ΔV⁺.

Gives both saddle locations and barrier heights in one pass. Continuation
makes γ_eq evaluation 5–10× cheaper. Picks up saddles `sc_equilib`
misses. Slow manifold can have folds; doesn't handle Hopf cases natively.

**Method D — Monte Carlo escape times (ground truth).**
Most faithful but very expensive per grid point. Best as a vetting tool
at a handful of representative points.

**Method E — local curvature at the stable equilibrium.**
V''(θₛ) = −f'(θₛ). Cheap but local only; says nothing about how far the
saddle is.

### 4. Recommendation

Implement Method C as the primary estimator; use Methods A, B, D as
vetting tools.

### 5. Suggested vetting plan

Standalone exploratory script in a `basin_estimation/` companion folder
mirroring `VM_bifurcations/`. Calibration points:

- 2-stable symmetric (expect equal basins).
- Asymmetric bistable (close to one target, far from the other).
- 3-stable.
- Hopf-island point from VERDICT.md (expect Method C to fail
  gracefully).

Cross-checks: Method A vs C on saddles; B vs C on saddle θ; MC at 1–2
points; explicit Hopf behaviour check; performance check.

---

## First round of fine points

### Fine point 1 — what γ-branch does V(θ) actually live on?

The walker doesn't track γ along the slow manifold the way a textbook
adiabatic theorem says. Each step calls `run_dgamma_dt` from the
previous γ. So the walker follows the γ-branch that is continuous in θ
from where it started.

Strictly, basin of attraction of a stable SC equilibrium is not just an
arc in θ — it's an arc in (θ, γ-branch) space.

### Fine point 2 — saddle-node folds in γ_eq(θ)

At a fold the γ-branch terminates; warm-started LSODA jumps to a
different γ-branch. f(θ) is discontinuous; V(θ) integrated through that
discontinuity is ill-defined. Physically real: walker's γ catastrophically
jumps. Basin extends to fold-θ; the Kramers formula doesn't apply in
simple V(θ)-exp form there.

### Fine point 3 — the two noise sources

- `T` in `dgamma_dt` — Ising temperature, controls γ-relaxation, but
  does not add noise during walker sim.
- `std` in `plot_walkers` — explicit Gaussian noise on θ at each step.

The Kramers exponent for our V(θ) picture uses `std`, not `T`. T
affects the *shape* of f(θ) and therefore V(θ), but doesn't appear in
the exponent.

### Fine point 4 — Hopf island and limit-cycle attractors

In Hopf islands (VM k=0.55 around (2.1, ±2.45)) the SC equilibrium is
unstable; attractor is a stable limit cycle. No basin to estimate for
SC eq. Adjacent cells with barely-Hopf-stable SC eq have fragile
Kramers estimates.

### Fine point 5 — computational budget

~0.5–2 s per stable eq for γ-continuation around the circle.
On a 41×41 grid with refinement_levels=3–4, ~5000–10000 evaluated
cells. ~5–10 min extra in parallel. Acceptable.

---

## First AskUserQuestion round — answers

- **Model scope:** NBM only
- **γ-continuation:** warm-start
- **MC setup:** deferred ("let's discuss some of the fine points of the
  approach first")

---

## User's three deep theory points

> gamma is very dependent on theta through the translation of physical
> angles to neural angles. So to some extent that I'm currently unsure
> of, noise in theta is noise in gamma. Does this matter? Also, some
> context: this model is mean-field Glauber dynamics of an underlying,
> inherently noisy process (in gamma). By introducing noise in theta,
> we hope to recover some of that lost stochasticity in the walker
> process. So, at least from the perspective of theory, we are in fact
> interested in how robust any given stable equilibrium is to neural
> noise - not just the theta noise we have put in as a proxy. Does this
> change things? Finally, due to certain discontinuities in the model,
> basins of attraction may be defined by more than just saddle points.
> Does this change the approach?

### Response to point 1 — θ-noise is γ-noise via warping

When θ gets kicked by δθ, target signals shift in neural space, so
γ_eq(θ) shifts off the real axis, and γ chases.

For basin analysis on the slow manifold, this is captured by computing
γ_eq(θ) via the actual perception model at each θ in our scan. The chain
"θ kicked → γ_eq shifts → γ relaxes" plays out on fast γ-timescale; once
γ relaxes, we're back on the slow manifold at (θₛ+δθ, γ_eq(θₛ+δθ)), and
slow θ dynamics takes over. V(θ) integrates f(θ) which already includes
all warping-induced nonlinearity.

Conclusion: warping doesn't break the V(θ) picture **as long as θ-noise
is the only noise source.** Point 2 challenges that assumption.

### Response to point 2 — γ-noise is the physical noise

This changes things substantially. Two qualitatively distinct escape
mechanisms out of a stable SC equilibrium:

**(a) θ-saddle escape:** γ tracks γ_eq(θ); θ wanders along slow
manifold until crossing a θ-saddle. Rate scales with V(θ) barrier.

**(b) γ-branch escape:** at fixed θ, γ excurses far enough to leave
its current γ-basin and lands on a different γ-branch. Once on a
different branch, f(θ) flow drives θ to a different SC equilibrium.
Barrier is the free-energy barrier in γ at fixed θ:
ΔF_γ(θ) = F_γ(γ_saddle; θ) − F_γ(γ_eq; θ), rate ~ exp(−ΔF_γ/T).

For cosine-kernel mean-field, F_γ is the Lyapunov function for
deterministic γ-flow, of roughly the form:

$$F(\gamma; \theta) \;=\; \tfrac{1}{2}|\gamma|^2 \;-\; \frac{T}{N}\sum_i \rho_i \log\bigl(2\cosh(N\,R\cos(\theta_i^{\text{neur}} - \Theta)/T)\bigr)$$

(Algebra to be verified against `Hamiltonian ideas.tex`.) γ-equilibria
are critical points of F at fixed θ; stable γ-eqs are local minima;
γ-saddles found by `gamma_equilib` with unstable label bound the
γ-basins.

Implications:

- Right noise robustness scalar is **not just ΔV(θ).** It's the
  smaller of {ΔV(θ)/σ², ΔF_γ/T} — whichever escape is easier dominates.
- At a typical bistable grid point with strong γ-wells, γ-escape is
  rare; θ-saddle escape dominates → V(θ) is right.
- In regions where γ has multiple basins at fixed θ (decision-paralysis,
  cluttered-multi-target), γ-escape can dominate → ΔF_γ is right.
- Both barriers depend on the same T. Natural exponents become
  ΔV(θ)/T_eff vs ΔF_γ/T, with T_eff from projecting γ-noise onto θ
  along slow manifold (T_eff = T · geometric factor in strict
  adiabatic limit).

Also suggests **MC vetting should use γ-noise**, not (or in addition to)
θ-noise. Clean stochastic version: add Gaussian noise to `dgamma_dt`
with amplitude calibrated to T.

### Response to point 3 — discontinuities define basin boundaries

At least three types in NBM:

1. **γ-branch folds.** Slow-manifold γ-continuation jumps; basin
   terminates at fold-θ. No Kramers barrier — escape rate set by how
   close γ_eq(θ) is to the fold.

2. **Perception discontinuities — blind spots.** Under `a=0, b=π`
   cutoff, weight is zero for ego angles outside [−π+ε, π−ε]; cross
   that boundary and a target instantly drops out. dgamma_dt is
   *discontinuous* in θ there. Walker blind-spot trap (γ locks to
   −1+0j on branch cut, sin(±π)≈0): once walker crosses, no torque to
   pull it back. Geometrically the basin has a sharp edge.

3. **Occlusion / target-blocking discontinuities.** When observer moves
   through a position where one target abruptly starts/stops occluding
   another, visible-target set changes discretely. ρ for unblocked
   target is discontinuous. Less dramatic than (2) but same character.

So basin estimator can't just look for sign changes in f(θ) — it also
needs to look for **discontinuities in γ_eq(θ) and in f(θ)** as θ
varies, and treat those as basin boundaries with a different label.

In the θ-scan, both can be detected from the same diagnostic: |Δγ_eq|
small for ordinary stretches and large at folds; |Δf| small for ordinary
stretches and large at perception discontinuities. Threshold catches
both.

### Revised approach

| Component | What it captures | How to compute |
|---|---|---|
| **V(θ) along slow manifold** | θ-saddle escape barrier (mechanism a) | θ-scan with warm-start γ-continuation, ∫f dθ |
| **ΔF_γ(θₛ)** at each stable SC eq | γ-branch escape barrier (mechanism b) | F_γ evaluated at γ_eq(θₛ) and at γ-saddle(s) found by `gamma_equilib` at fixed θ=θₛ |
| **Fold-θ detector** | γ-fold boundary | |Δγ_eq| jumps during θ-scan |
| **Discontinuity-θ detector** | perception discontinuity boundary | |Δf(θ)| jumps during θ-scan |
| **MC ground truth** | empirical escape rates | walker with γ-noise (modified `run_dgamma_dt`) |

Single robustness scalar per stable SC eq: `min(ΔV(θ)/T_eff, ΔF_γ/T)` if
both boundary types are smooth, or "fold-bounded basin, escape rate
~ |γ_eq(θₛ) − γ_fold|" if a fold dominates, with explicit "no estimate"
sentinel inside Hopf islands. May want to display all three (θ-barrier,
γ-barrier, geometry) as separate panels first.

---

## Second AskUserQuestion round — answers

- **F_γ source:** Re-derive as part of vetting
- **γ-noise model:** (deferred; clarification requested)
- **Discontinuities:** (deferred)

### Clarification on γ-noise model — option 1 vs option 3

**Option 1 (Langevin γ-noise only):** Modify `run_dgamma_dt` so γ ODE
becomes γ SDE with noise amplitude calibrated to T. Walker is otherwise
deterministic (`std=0` in `plot_walkers`). Measure escape rates driven
purely by γ-noise. The physically motivated model.

**Option 3 (both Langevin γ-noise and existing θ-noise):** Same
Langevin modification, but retain the existing `std` heading-noise knob.
Can drive escapes with either noise channel alone, or both together, and
compare.

| | Option 1 | Option 3 |
|---|---|---|
| Cleanliness | One noise model. Fewer knobs. | Two channels; care needed. |
| What you learn from MC vetting | Whether basin estimates correctly predict γ-noise escape rates. | Same as option 1, plus empirical mapping σ → T·g(geometry). |
| Cost | Lower. | Marginally higher. |
| Backward continuity | Existing `std` results become uncalibrated. | Existing `std` results retroactively convert to effective T. |

The reason option 3 was included: user mentioned θ-noise is "introduced
to recover some of the lost stochasticity in γ" — i.e. a proxy. Option 3
quantifies how good the proxy is. If σ → T·g is approximately constant
across the bifurcation diagram, existing θ-noise machinery is justified.
If g varies wildly with (x,y), existing walker plots may have been
misleading in some regimes.

**Recommendation:** option 3, with order (i) get γ-Langevin running and
use as primary vetting tool, then (ii) at calibration points, also run
θ-noise alone and produce σ↔T·g mapping as secondary diagnostic.

### User decision

Option 3.

---

## Current design state

- **Scope:** NBM only
- **γ-continuation:** warm-start
- **F_γ:** re-derive as part of vetting
- **Noise model:** γ-Langevin (primary) + θ-noise (for calibration)
- **Discontinuity catalog:** proceeding with the three named (γ-folds,
  blind-spot crossings under cutoff weighting, occlusion transitions).
  **Loose end:** user is not certain this is exhaustive; revisit if
  vetting surfaces something else.

---

## Theory items flagged for a future session

The user wants to revisit these in more depth in a separate session:

- The mean-field free energy F_γ(γ; θ) for the NBM cosine kernel with
  weighting and warping — derive from `Hamiltonian ideas.tex` and
  verify the sketch in this transcript.
- The two-timescale (γ fast, θ slow) Langevin framework and how
  γ-noise projects onto θ-noise in the strict adiabatic limit
  (the T_eff = T · g(geometry) factor).
- Comparison of θ-saddle vs γ-branch escape rates: when each
  dominates as a function of where you are in the bifurcation diagram.
- The status of the (γ, θ) system as gradient-or-not: γ subsystem is
  gradient (Glauber free energy); θ dynamics is added separately;
  whether a joint Lyapunov function exists for the coupled system, or
  only local quadratic forms near each stable SC equilibrium.
- The Kramers-vs-discontinuity escape distinction at γ-folds and
  perception discontinuities: what replaces the exponential barrier
  formula there.
