# Basins of attraction & noise robustness — findings catalogue

Durable mathematical findings on the Neural Ring Model's basin structure,
noise, and the geometry at basin boundaries. Distilled from the completed
basin-estimation vetting effort (the old `basin_estimation/` directory, now
retired — its step-by-step vetting log lives in git history). Companion docs:

- [free_energy_derivation.md](free_energy_derivation.md) — the closed-form
  Lyapunov / free energy F̂(γ) this catalogue summarizes.
- [theory_background.md](theory_background.md) — self-contained Lyapunov /
  stat-mech / Langevin / Kramers tutorial behind the math here.

The visualization this work fed is **implemented**: `NBM.plot_bifurcation_diagram(overlay_basins=True)`
([decision_model.py](../decision_model.py)). See **"What the code does now"** below.

> **Scope of validity (read first).** The model splits cleanly into a
> Hamiltonian/Glauber-derived **γ-side** (neural coherence dynamics) and a
> phenomenological **θ-side** (the heading turning law dθ/dt, still actively
> evolving). Everything labelled *γ-side* below is **turning-law-invariant**
> and banked; θ-side magnitudes rescale if the turning law changes. See
> "The γ/θ decoupling" for exactly which is which.

---

## 1. The two-timescale picture and the slow manifold

The deterministic dynamics live on `(γ_re, γ_im, θ) ∈ ℝ² × S¹`: the complex
neural coherence γ, and the observer heading θ. There is a built-in
**separation of timescales** — γ relaxes fast (Hessian eigenvalues of F̂ ≈ 1
in the calibration setup, so time ~1), θ moves slowly under the torque
`dθ/dt = K·R·sin(Θ/2)` (time ~2 near a stable equilibrium). The ratio is only
~2×, which matters numerically (see §6 on the Schur complement).

The **slow manifold** is `γ_eq(θ)` — γ relaxed to its quasi-steady value at
the current θ. A trajectory falls fast onto the manifold, then drifts slowly
along it in θ. **The walker *is* this slaved system:** `plot_walkers` /
`dtheta_dt` re-equilibrate γ to steady state (warm-started from the previous
γ) on every heading step, then advance θ. This is the implementation of
"neural noise averages out," and it is why the slow-manifold framework is the
correct one for the real walker.

With γ slaved, the remaining dynamics is **1-D in θ**:

```
θ̇ = f(θ) := K · R(θ) · sin(Θ(θ)/2),   R(θ)=|γ_eq(θ)|,  Θ(θ)=arg(γ_eq(θ)).
```

**Crucial caveat:** `γ_eq(θ)` is *not* a single global smooth function. It is
a **union of γ-branches glued at γ-folds** (§4). A basin in θ is an arc on one
branch. Poincaré-Hopf holds *within* a branch, not across the union — which is
why `sc_equilib`'s unstable list can be incomplete (§3, §4).

## 2. The free energy F̂ and γ-noise (summary)

The γ-flow is **exact gradient descent** of a closed-form Lyapunov function,
`dγ/dt = −∇F̂(γ; θ, focal_loc)`, with

```
F̂(γ) = ½|γ|² − (1/2β) Σⱼ ρⱼ ln(1 + e^{uⱼ(γ)}),   uⱼ(γ) = 2β v̂ⱼ·γ.
```

F̂ is **Hamiltonian / Glauber-derived**, hence *independent of dθ/dt*. Two
independent routes (gradient-flow integration; mean-field projection
`2β·F̂ = β·F_mf` per spin at the constrained minimum) give the same object. Full
derivation + numerical validation: [free_energy_derivation.md](free_energy_derivation.md).

Adding physical γ-noise gives the **γ-Langevin SDE**

```
dγ = −∇F̂(γ) dt + √(2D) dW,    D = 1/(2βN),
```

with stationary distribution `P_ss(γ) ∝ exp(−F̂(γ)/D)` (Boltzmann with
effective temperature D). Near a γ-minimum the cloud is Gaussian with
covariance `Σ = D·H⁻¹` (H the F̂-Hessian). `D = 1/(2βN)` is a finite-size
(1/N) correction — verified by the empirical γ-variance scaling as 1/N
(~2% rel. err.) — vanishing as N→∞ (deterministic mean field). The 2β arises
from the F̂ normalization; rigorous route is van Kampen's system-size
expansion ([theory_background.md](theory_background.md) §IV.4).

**Note on the implemented walker.** The *simulated* walker has **only heading
(θ) noise** (the `std` knob); there is no γ-noise term in `plot_walkers`. The
γ-Langevin above was the vetting tool that calibrated and validated F̂; it is
the right model for *physical* neural noise, but it is not what the walker
integrates. This distinction is load-bearing for §5 and §7.

## 3. The reduced θ-dynamics: effective potential & topology

A 1-D ODE on S¹ is automatically a **gradient system**: `θ̇ = −dV/dθ` with
`V(θ) = −∫f`. So on the slow manifold the walker descends an effective
potential V(θ); stable equilibria (f′<0) are minima, unstable (f′>0) are
maxima.

**Poincaré-Hopf on S¹:** `(#unstable) − (#stable) = χ(S¹) = 0` — stable and
unstable equilibria alternate around the loop. This is a free consistency
check: 3 stable claimed with only 2 unstable ⇒ one is missing. `sc_equilib`
misses unstable equilibria two ways: (a) saddles with γ on the **negative**
real axis (`γ=−R+0j`, observer facing directly away) are excluded by its R>0
filter — e.g. at (0.5,0) the unstable at ±π; (b) saddles on a **different
γ-branch** than the one a side-stable scan rides. Neither is a bug — the
walker on a side branch genuinely experiences a fold-bounded basin, not the
SC saddle.

## 4. What bounds a basin — the four boundary kinds

In a multistable region the basin of a stable equilibrium (an arc in θ) is
bounded by one of **four** mechanisms. The first three are **γ-side /
turning-law-invariant**; the fourth is set by the turning law at the antipode.

| Boundary | Mechanism | First-passage / escape scaling |
|---|---|---|
| **Smooth saddle** | zero of f(θ) with f′>0 on the same γ-branch; a genuine separatrix | `exp(−ΔV/D_θ)` — textbook Kramers |
| **γ-fold** (saddle-node) | the γ-min collides with a γ-saddle and annihilates; γ jumps to another branch (R collapses) | `exp(−ΔV_fold/D_θ)` — exponential, but ΔV is taken *up to* the fold; no smooth-saddle prefactor |
| **Perception collapse** | all targets leave the visible cone (tight cutoff); ρⱼ=0 ⇒ γ_eq=0, R=0, f=0 over a θ-range | `Δθ²/(2D_θ)` — **diffusive**, not exponential |
| **Branch-cut** (half-angle law) | at the facing-away heading Θ=arg(γ)=±π, f jumps ±K·R while γ_eq passes continuously through −R | a **repelling fork**, not a barrier to climb; bounds 1-stable / symmetric basins where an edge falls at θ≈±π |

**γ-fold detail.** Detected on a scan as `|Δγ_eq|` ≫ typical (~0.03 step;
~0.5 at a fold) — flagged by a relative threshold (8× median) and absolute
threshold (0.4). Example: at (2.0,0) the +0.82 branch has **two** folds going
CW (θ≈−0.92, −2.46); at (1.2,0) the central basin's inner edges are folds at
θ≈±0.479.

**Branch-cut detail.** New under the half-angle torque law. Measured at
(0.5,0), θ=±π: `γ_eq=−0.9494`, R=0.9494, f jumps −1.899→+1.898, so
`|Δf|≈3.80 ≈ 2·K·R` with `|Δγ|≈0.02` (a pure f-jump, *not* a γ-fold). It is
the intentional left/right fork at the facing-away point — both `sin(arg(γ)/2)`
and the algebraic half-angle identity give two-sided limits ±1 there. Under
the *old* `sin(ego)` law this point was a torque dead-zone (f≈0); the
half-angle law makes |f| maximal there, removing the spurious back-of-circle
dead-zone (the deterministic half of the walker blind-spot fix).

**Scope caveat — boundaries only matter when there is a second basin.** In a
**1-stable** region the basin is the whole circle minus the unstable point;
noise pushing θ "over" the V-saddle just sends it around the loop back to the
same stable. ΔV and ΔF_γ in 1-stable cells measure *transient excursion*
timescales, not basin-transition rates. So the right 1-stable deliverable is a
**direction arrow** (+ optional local-stiffness glyph from the F̂-Hessian),
**not** a robustness color.

## 5. γ-noise robustness: ΔF_γ and Kramers validation

For a γ-saddle `γ_sad` neighboring a stable `γ_s` at fixed θ, the **barrier
height** `ΔF_γ = F̂(γ_sad) − F̂(γ_s)` sets the γ-noise escape rate
`~exp(−ΔF_γ/D)`. Bigger ΔF_γ ⇒ more γ-noise-robust. (Direct and path-integral
evaluations agree to ~1e-8, a gradient-theorem consistency check on the F̂
implementation.)

**ΔF_γ varies ~30× across the calibration setup:**

| point | θ_s | #γ-eqs | #γ-saddles | ΔF_γ |
|---|---|---|---|---|
| (2.0,0) side | ±0.82 | 3 | 1 | **0.144** |
| (1.2,0) center | 0 | 5 | 2 (mirror) | **0.0154** |
| (1.2,0) side | ±0.66 | 3 | 1 | **0.00426** |

A y=0 symmetric central stable has a **mirror pair** of γ-saddles — two
parallel escape channels, so the Kramers total rate doubles.

**Monte Carlo validation (γ-Langevin, vs Eyring-Kramers):**
- **Exponent tight** — empirical slope of `log τ` vs 1/D is 0.0179 vs the
  predicted ΔF_γ=0.0154 (within **16%**; pass criterion was a factor of 2).
- **Prefactor ~right** — empirical/Kramers τ ratio stays in **1.0–1.9** across
  the D-sweep (empirical always ≥ Kramers, because the MC criterion counts
  *commit-to-destination* time, not just top-of-barrier).
- **Ordering correct** — at D=0.003, τ_side≈40 vs τ_center≈1200: the
  smaller-ΔF_γ side stable escapes **~30×** faster, matching
  `exp(Δ(ΔF_γ)/D)`.

⇒ **ΔF_γ is a trustworthy γ-noise robustness scalar — but only in multistable
cells** (1-stable cells have no second basin).

## 6. The Schur complement (slow-eigenvalue technical note)

The slow eigenvalue of the 3×3 coupled Jacobian at an SC equilibrium is the
**Schur complement** of the γ-block, `λ_slow = J_θθ − J_θγ J_γγ⁻¹ J_γθ`, not
"the eigenvalue whose eigenvector has the largest θ-component." The latter is
exact only at infinite timescale separation; at the model's ~2× separation the
γ–θ mixing shifts it by ~18%. The Schur form matches V″(θ_s) from the scan to
~0.02%. (This is exactly the `'reduced'` stability criterion's slow test —
see [.claude/rules/torque-and-stability.md](../.claude/rules/torque-and-stability.md),
evaluated robustly as `sign(det J)` via `det J = det A · λ_slow`.)

## 7. The scan-fold vs decision-boundary distinction (headline gotcha)

The single most important — and counterintuitive — finding. **A γ-branch's
fold is NOT where the decision changes.** Crossing a γ-fold is **necessary but
not sufficient** for a θ-noise decision switch.

Worked at (4.0,1.5), VM-k055 (far stable θ_far=−1.489, γ_far≈1.0; close stable
θ_close=+1.252):

- The warm-start scan reports the far basin's CCW boundary as a **γ-fold at
  θ_fold=−1.394**, only **5°** from the far stable.
- Kick θ just past the fold (holding γ committed) and run the slaved flow: γ
  *does* jump branches (γ: 1.0+0j → 0.40−0.08j, R: 1.0→0.41) — it genuinely
  leaves its γ-basin. **Yet the flow returns to FAR:** the post-fold low-R
  branch still produces restoring torque toward the far target. The jump is
  *recoverable*.
- Bisecting on the slaved-flow *outcome* locates the **true decision boundary
  at θ_dyn=−0.307**, **68°** (1.087 rad) past the fold. A **0.06 rad** heading
  change flips the chosen target — with γ-noise identically zero. θ_dyn sits at
  **no SC equilibrium**; it is a fold-mediated separatrix, not a smooth zero of
  f (so θ-noise escape across it is not textbook Kramers).

**Why heading noise can flip a decision at all.** With γ slaved, the θ-noise
walker never climbs the fixed-θ barrier ΔF_γ (γ sits at its well bottom). The
channel is instead the **γ-fold**: a θ-excursion past a fold makes the slaved
γ jump basins. So γ-basin structure *enables* a θ-noise switch as a branch-jump
event — but the **decision robustness of a stable equilibrium is its
basin-attribution width, not its fold distance.**

Two physically distinct quantities, which coincide at smooth-saddle edges and
**diverge at fold edges**:
- **γ-branch fragility** — fold close to the stable, R collapsing on a small
  heading change — is near-saddle-node marginality of the *committed neural
  state*.
- **Decision-basin width** — how far the heading can stray and still return —
  is the *behavioral* robustness.

The far stable has a *fragile γ-branch* (R falls within 5°) but a *robust
decision* (68° to flip).

## 8. Asymmetric basins, and what R does *not* measure

**Asymmetry is the norm off-axis.** At (4.0,1.5) the close target (d=1.05) and
far target (d=4.01) give very unequal basins. Correcting §7's scan-fold
artifact gives the true picture:

| far-stable basin | scan / γ-branch | dynamical (decision) | neutral-seed |
|---|---|---|---|
| CCW-side width | 5° | 68° | — |
| total far basin | ~57° | ~119° | 106° |
| close basin | ~303° | ~241° | 254° |
| **close/far ratio** | **5.35×** | **≈ 2.0×** | **2.40×** |

The *qualitative* result (close basin ≫ far basin) is robust; the **5.35×
magnitude was largely a scan artifact** — the true dynamical asymmetry is ~2×.

**R = |γ| is NOT a basin-size proxy** (tempting shortcut, ruled out). It can
*anti-correlate*: at (4.0,1.5) the wide-basin close stable has the **lower**
R (0.918) and the narrow-basin far stable has R=1.000. Basin width (θ-arc
between bounding saddles/folds) and R (committed angular sharpness) are set by
different things — a close, wide target spreads its warped phasors (R<1) even
when committed; a far, point-like target is sharp (R→1). R is closer to a
**well-curvature / local-stiffness** proxy (`V″(θ_s) ∝ R·Θ′(θ_s)`). The grain
of truth: `R→0` flags an equilibrium approaching saddle-node death (basin→0) —
a *near-SN marginality* flag, not a width measure.

## 9. Commitment: neutral vs committed seed

With the consensus pinned straight-ahead (`arg(γ)=0`), the seed **R** is the
only knob that selects between coexisting γ-branches at a fold — and it bites
only in thin near-fold wedges. Comparing an **uncommitted** seed (R≈0.15) vs a
**committed** one (R≈1.0) isolates the commitment-sensitive wedge.

At (4.0,1.5): the seed is decision-relevant on only **~4% of headings**. In
that wedge the uncommitted seed → **close** target, the committed seed → **far**
target ⇒ **commitment is required to hold the farther / weaker-perception
target**; an uncommitted walker defaults to the closer/stronger one. Since the
two SC-eq R's are nearly equal (far 1.000, close 0.918), the discriminator here
is **perception strength, not equilibrium R**.

## 10. The γ/θ decoupling — what is banked vs turning-law-dependent

Write the turning law as `dθ/dt = K·R·g(ego)` with `g(0)=0`, `g′(0)>0`
(old `g=sin`; current `g=sin(·/2)`). dγ/dt is Hamiltonian-derived and never
sees g; θ feeds back into dγ/dt *only* through perception geometry. Therefore:

- **Banked (γ-side, turning-law-invariant):** the slow manifold γ_eq(θ), R(θ),
  ego(θ), γ-folds, perception-collapse zones, F̂, γ-saddles, **ΔF_γ**, SC-eq
  *locations*, stable/unstable counts, basin *topology*, and geometric basin
  widths bounded by **saddles/folds**.
- **Tied to dθ/dt (θ-side, rescales with K·g′(0)):** slow-eigenvalue
  *magnitudes* → θ-relaxation timescale → **θ-noise barrier heights ΔV**; the
  and the existence/width of branch-cut-bounded basins.

The K=2 default was chosen so K·g′(0) = 2·½ = 1·1, holding even the θ-side
magnitudes fixed *at SC equilibria* across the old→new law change. Under a
non-identity warp the off-equilibrium θ-side numbers rescale by `ν(0)=W′(0)`
(neural density at center; e.g. vonmises k=0.55: ×1.609).

**Two open items that stay tied to dθ/dt** (cannot be closed until the turning
law settles): (1) the θ-noise basin "depth" at a branch-cut/antipode needs its
own first-passage treatment (a repelling fork, not a Kramers barrier);
(2) the θ-noise-as-γ-noise proxy question (§11).

## 11. Open / deferred questions

- **Headline gap — is θ-noise a faithful proxy for γ-noise?** The walker uses
  θ-noise (`std`); the physical noise is γ-side (Glauber β, 1/N). Whether the
  `std ↔ g(geometry)/β` mapping is roughly constant across the (x,y) plane
  (so existing `std` results back-translate to an effective N and β) is
  **deferred** and dθ/dt-dependent. Only meaningful in multistable cells.
- **Committed-walker (history-dependent) decision basins** — the full
  both-sides-at-both-stables recompute via slaved-flow bisection. The
  first-paper plot uses the simpler *neutral-seed* protocol instead (§12).
- **Fold/R-collapse "fragility" glyph** — render γ-branch fragility separately
  from decision width (§7).
- **θ-noise decision-flip escape rate** — needs the noise characterized.
- Perception-collapse `Δθ²` diffusive scaling — MC test not run.
- "Which mechanism dominates" map — `min(ΔV/D_θ, ΔF_γ/D, log(Δθ²/2D_θ))` over
  (x,y).
- Joint (γ,θ) global Lyapunov function — curl-free test of the full 3-D field.

Engineering follow-ups are tracked in [TODO.md](../TODO.md).

## 12. What the code does now (implementation status)

The basin visualization is **live in the model** (no separate `basin_estimation/`
module): `NBM.plot_bifurcation_diagram(overlay_basins=True)` in
[decision_model.py](../decision_model.py).

- **Single-panel wheel overlay.** The count colormap is dimmed
  (`basin_bg_alpha`, default 0.7) and a basin **wheel** is drawn per region.
  (The separate count panel was dropped — the wheels already encode the count.)
  Placement is deterministic: deepest-interior (distance transform) +
  min-area filter + richest-first min-separation merging
  (`_basin_wheel_placement` / `_overlay_basin_wheels` / `_render_basin_wheels`).
- **Basin arcs — `basin_arcs_at_focal_loc(focal_loc, R_seed=0.15, n_coarse=64,
  n_bisect=12, stability_criterion='reduced')`.** Implements the first-paper
  **fixed-seed protocol**: a fixed **neutral seed** (`arg(γ)=0`,
  `|γ|=R_seed=0.15` — the *indecision range* 0.1–0.2, below the committed
  R≳0.4 seen in walkers, above the R≈0 arg-degeneracy). Headings are swept
  (`n_coarse=64`) and each is run under the **slaved flow** to its destination
  stable; basin boundaries are refined by **destination-flip bisection**
  (`n_bisect=12`). The neutral fixed seed makes the basin map single-valued /
  history-independent (removes the §7 commitment ambiguity).
- **Slaved-flow destination — `_basin_destination`.** Re-equilibrates γ via
  `run_dgamma_dt` (warm-started) each heading step, advances θ by the
  half-angle torque (`dtheta_dt`), converges at `|dθ|<1e-5`
  (`n_steps=2000`, `dt=0.1`); returns the nearest stable index, or −1 for
  no-commit (no convergence, `R<min_R=0.05`, or lands `>max_dist=0.15` from any
  stable). 1-stable cells return the whole circle as one arc; 0-stable cells a
  single no-commit arc.
- **Rendering.** Wheel = θ-basin annulus partitioned into arcs (categorical
  color by basin **rank**, largest→smallest) + one direction arrow per stable
  direction, length-modulated by basin width. Single-stable cells draw a lone
  full-length arrow, no annulus.

**Validated** at (4.0,1.5): neutral-seed widths close=254°, far=106° ⇒ 2.40×,
matching the §8 dynamical ~2× (not the 5.35× scan artifact).

**Superseded / removed.** The scan-based prototype
`compute_basins_at_focal_loc` (and the whole `basin_estimation/` vetting
module) is gone; the slaved-flow overlay above replaces it. The scan-based
`basin_features` widths were *γ-branch extents* — correct at saddle edges,
wrong at fold edges (§7) — which is precisely why the overlay uses slaved-flow
destination bisection instead.

**Deferred** (→ [TODO.md](../TODO.md)): committed-walker history-dependent
basin recompute, the fold/R-collapse fragility glyph, and the θ-noise
escape-rate.

## Calibration setups & cost

- **VM-k055** (primary): vonmises warp `k=0.55`, two **circle** targets at
  (4.33, ±2.5); `β=10` (two targets at the earlier `T=0.2`), `N=1000`
  ⇒ `D≈5e-5` in the γ-Langevin tests.
- **BlindSpot** (discontinuity stress test): tight cutoff weight `b=π/2`, used
  to exhibit perception-collapse zones.
- **Cost:** ~4.2 min parallel (32 cores) for a 41×41 grid with basins, vs ~21 s
  for `sc_equilib`-only. Multistable cells dominate (~2–4 s each); 1-stable and
  collapse cells are nearly free (~0.09 s, scan skipped /
  short-circuited). Mitigations if higher resolution is needed: subgrid-sample
  the clustered multistable regions; cross-cell γ warm-start; coarser scans.

## Further reading

- [free_energy_derivation.md](free_energy_derivation.md) — F̂(γ) closed form,
  gradient-flow proof, mean-field cross-check, D = 1/(2βN) calibration.
- [theory_background.md](theory_background.md) — Lyapunov functions, free
  energy, Langevin & Kramers, for a reader new to statistical mechanics.
- [.claude/rules/torque-and-stability.md](../.claude/rules/torque-and-stability.md)
  — the half-angle torque, the two stability criteria, the coupled Jacobian.
- [CLAUDE.md](../CLAUDE.md) — model architecture, coordinate frames, conventions.
