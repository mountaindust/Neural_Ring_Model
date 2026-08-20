# Folding `IsingExtModel` into `NeuralBandModel`

**Status:** reviewed and verified 2026-08-18/19; **implemented 2026-08-19 as
`NeuralBandModel(angle_distortion_nu=...)`.** `IsingExtModel` is deliberately
still present, awaiting review before deletion — steps 5 and 6 below are the
only ones outstanding. This document is the plan and the measurements it
rested on; §10 records what was actually built and what the implementation
measured.

**Verdict: yes, IEM can be folded into NBM and deleted.** IEM is not merely
*reproducible* by NBM — it is NBM written in a rotated coordinate frame. The
one thing NBM cannot express today is the ν-warped coupling kernel, which is
about a dozen lines to add.

---

## 1. The exact correspondence

With `neural_angle_dist=None`, `angle_weight=None` (the configuration CLAUDE.md
already mandates for IEM), ν = 1, and β = N_visible/T:

    dγ_IEM(γ·e^{iθ}, θ)  ==  e^{iθ} · dγ_NBM(γ, θ)

i.e. `γ_IEM = γ_NBM · e^{iθ}` — the same model, allocentric vs egocentric
frame. Measured:

| quantity | agreement |
|---|---|
| `dgamma_dt` identity, 1200 random (loc, heading, γ) states, delta/circle/capsule × uniform/cutoff weight | max err **1.08e-15** |
| `dtheta_dt`, 500 random (θ, γ) | **6.1e-16** |
| SC equilibria (θ, R) on 238-pt grids, delta and circle | **0** count mismatches, ≤ 2e-9 |
| stability (`reduced`, `discrim_a`) over 788 equilibria | **0** disagreements |
| `gamma_equilib` at fixed heading, 63 (loc, heading) states | **5.8e-10** |
| `IEM._discrim_A_nu` vs `NBM._discrim_A` at ν=1 | **0 / 142** disagree |

Incidental: IEM's `/signals.sum()` is a no-op — `rho` already sums to 1 to
1.1e-16.

## 2. The frame difference is inert

Changing variables `γ_E = γ_A·e^{−iθ}` turns IEM's vector field into NBM's
**plus an advection term** `−i·θ̇·γ_E`. It vanishes at every equilibrium
(θ̇ = 0), and `sign(det J)` and `eig(A)` are invariant under it — so
equilibria, `'reduced'` and `'discrim_a'` are untouched. It was visible only in
the full 3×3 eigenvalues, i.e. the `'coupled'` criterion, **which was removed
2026-08-19** (see `NeuralBandModel._discrim_reduced`). Nothing in the codebase
integrates the coupled system, so after that removal the frame choice has **no
observable footprint at all**.

Walker trajectories agree to ~1e-4 (that residual is `run_dgamma_dt`
tolerance); the one large divergence found was the intentional ±π branch-cut
fork at a symmetric start, resolved by roundoff.

> **Amended by §10/§11.** "No observable footprint at all" holds for
> everything deterministic, but the frame also decides how each walker step
> *warm-starts* its γ relaxation. Where γ is multistable that IC picks the
> branch, so walks can end at different targets — and this is a modelling
> difference that survives `dt → 0`, not a tolerance artifact. See §11.

## 3. The ν kernel is the one genuine extension

`cosine(x) = cos(π·(|x|/π)^ν)` bends the coupling only, leaving the `e^{iθ̂}`
phase alone. **No PerceptionModel setting can imitate it:**

- Every warp/weight combination keeps the fast γ-block **symmetric**
  (max |A₀₁−A₁₀| ≤ 8.3e-10 across 10 configs incl. `direct_power`,
  `vonmises`, `cutoff`, tied weight) — i.e. the γ-flow is gradient.
- ν = 0.5 and ν = 2 give asymmetry ≈ **4.3**, the same order as ‖A‖. Not
  gradient.

Warp and ν are different mechanisms: a warp moves the phase too, ν does not.

**Cost of adopting ν:** ν ≠ 1 destroys the free-energy structure, so **F̂, the
analytic `_discrim_A`, ΔF_γ basin barriers and the γ-Langevin noise calibration
are ν=1-only.** `_discrim_reduced` is numerical and stays valid.

It changes structure materially: at (10,10) ν=1 gives 1 equilibrium, ν=0.5
gives 3 (2 stable); ν=0.5 grids reach 5 SC equilibria.

**Prototype result.** An NBM subclass with `nu` + a `kernel()` using
`convert_angles` reproduced IEM over 238 points × {delta, circle} ×
ν ∈ {1, 0.5, 2}: max |Δθ|, |ΔR| ≈ 2e-9 and **0 stability mismatches at ν = 1
and ν = 2**. At ν = 0.5, 2/238 points per geometry differ — a solver seeding
gap, see §5.

## 4. What is being removed is worse than what replaces it

IEM is 1298 lines. **360 are AST-identical to NBM** (`plot_walkers` 188,
`_simulate_one_walk` 108, `_discrim_reduced` 26, `dgamma_dt_vec` 24). Its
`plot_bifurcation_diagram` and `plot_direction_mesh` differ only by NBM's basin
overlay, title strings, and where stability is computed. **Genuinely novel:
~98 lines** — `cosine` (7), `_discrim_A_nu` (32), `plot_cosine` (20),
`plot_dtheta_dt` (39); the last two are generic utilities NBM would *gain*, and
neither is called anywhere.

**`IEM.sc_equilib` is broken for any heading-dependent perception.** It
root-finds `dgamma_dt` at a *fixed seed heading*, then only *filters* for
self-consistency — it never solves for the heading. That works solely because
with identity warp + uniform weight `dgamma_dt` is heading-independent
(measured: spread **8e-16** over 9 headings; with `'cutoff'` weight, **1.58**).
With `angle_weight='cutoff'` it returns **0 equilibria** at points where
genuine ones exist with residual ~1e-16. This is why
`tests/test_broad_validation.py` reports **38 mismatches** (14 delta + 24
circle), *all* "IEM found fewer" — every one vanishes with
`angle_weight=None` (0/238 both geometries).

Performance: `sc_equilib` **3.8×** slower (7.99 s vs 2.08 s / 100 pts — no
`signal_cache`); `run_dgamma_dt` **7.8×** slower (0.39 s vs 0.05 s / 50 solves
— restarted RK45 vs LSODA) and it emits "may not have reached equilibrium"
during walker runs.

## 5. Two real caveats

**(a) β = N/T is not a globally exact reparameterization.** IEM's coupling
carries the *visible* target count; NBM's β does not. Under uniform weight
N_visible = N_total almost everywhere — but not in the full-occlusion shadow
behind a target. For two r=0.5 circles that shadow is a band of width exactly
2r extending indefinitely behind the near target: **2.16%** of a
[−10,40]×[−25,45] scan. At (15,−20): β = N_total/T gives residual 2.5e-3,
β = N_visible/T gives **0.0**. Under a cutoff weight cone N is constant at only
~69% of states. Keep β scene-independent (the physical argument in CLAUDE.md
holds) and state that exact IEM/Sridhar reproduction inside an occlusion shadow
is not available.

**(b) `NBM.sc_equilib` has a ν≠1 seeding gap.** At ν = 0.5 equilibria migrate
to R ≈ 0.77–0.85, above the probe radii `(0.3, 0.5, 0.7)`. Over a 476-point
ν=0.5 grid the union of both finders is 764 equilibria: NBM finds 758, IEM 760.
**Adding a 0.85 probe takes NBM to 762 — strictly better than IEM.** (2
residual misses are the known near-saddle-node pair-selection case.) The
missed ones are genuine roots of each model's own residual; NBM's Im-scan at
R_probe ∈ {0.8, 0.85, 0.9} exposes the sign changes it misses at {0.3,0.5,0.7}.

## 6. Consumers

- **Both IEM notebooks are already dead.** `ising_workbook.ipynb` and
  `neural_band.ipynb` use the removed `neural_weight=` / `neural_angle=`
  PerceptionModel API and raise `TypeError` on construction. No live notebook
  consumers.
- **Tests:** `tests/test_broad_validation.py` (the whole cross-model
  comparison, excluded from pytest); three signature/torque assertions in
  `tests/test_half_angle_torque.py` (`_identity_iem`, `K==2`, `walk_std`,
  `R_exp`); `_iem_plain`, `test_schur_block_determinant_identity_iem` and two
  `test_defaults_are_reduced` entries in `tests/test_reduced_criterion.py`;
  `test_no_coupled_criterion` also asserts on IEM.
- **TODO.md:** the two open IEM items — the `run_dgamma_dt` LSODA port and the
  `signal_cache` mirror — become **moot** on the fold; delete them with the
  class.

## 7. Fold steps

1. Add `nu=1` to `NeuralBandModel.__init__` and a `kernel(x)` method.
   **Wrapping with `convert_angles` is load-bearing** — the ν kernel is not
   2π-periodic (`cosine(3.283) != cosine(wrap(3.283))`). Use it in
   `dgamma_dt`.
2. Gate the ν=1-only machinery: `_discrim_A` falls back to the numerical fast
   block when ν ≠ 1 (that *is* `_discrim_A_nu`; they agree 0/142 at ν=1). Same
   for the F̂ / ΔF_γ basin path.
3. Add a `0.85` probe radius to `sc_equilib` (§5b).
4. Move `plot_cosine` and `plot_dtheta_dt` across (neither is currently
   called).
5. Replace `test_broad_validation.py`'s cross-class comparison with a direct
   test of the rotation identity `dγ_A = e^{iθ}·dγ_E` — which is what it was
   really probing — or retire it.
6. Update CLAUDE.md: the model IEM represented = NBM with identity warp,
   uniform weight, β = N/T, plus ν. The only thing dropped is the
   world-anchored γ frame, whose sole footprint was the removed `'coupled'`
   criterion.

## 8. Re-verifying

Every number above came from short standalone scripts, none committed. The
cheap re-checks, in order of value:

- **Rotation identity** — random (loc, θ, γ); compare
  `IEM.dgamma_dt(γ·e^{iθ}, θ)` against `e^{iθ}·NBM.dgamma_dt(γ, θ)` with
  `β = N_visible/T` recomputed per state. Should be ~1e-15.
- **Gradient test** — central-difference the fast block and check
  `|A₀₁ − A₁₀|`; ~1e-10 for any warp/weight, ~4 for ν ≠ 1.
- **Equilibrium/stability parity** — `NBM.sc_equilib` vs `IEM.sc_equilib` on a
  grid, `angle_weight=None` (with a non-uniform weight IEM's finder fails, §4).

## 9. One thing not verified

The review established that NBM+ν reproduces **IEM**, not that it reproduces
**Sridhar et al. (2021)**. If the ν=0.5 + delta-target configuration is meant
to match published numbers, check their coupling normalization: their SI
Eq. [9]–[12] carry the option count `k` in `2k·V⃗·p̂_i/T`, exactly IEM's
`2·angles.size·R·cosine(...)/T`, whereas the project preprint deliberately
drops that factor (targets are not identical). That is caveat (5a) restated,
and it is the one place the two conventions genuinely diverge.

---

## 10. What was implemented (2026-08-19)

**API.** `NeuralBandModel(percep_model=None, beta=10, K=2,
angle_distortion_nu=None)`. The argument is named for the draft paper's
terminology rather than the plan's bare `nu`.

- `None` (default) or an explicit `1` is the plain cosine kernel and is
  **bit-identical** to the pre-fold model. Verified against the HEAD source
  over delta/circle/capsule × {lin_cutoff, identity, vonmises+tied,
  direct_power+cutoff}: `dgamma_dt` `0.0`, `sc_equilib` angles/R/stability
  under both criteria `0.0` and zero count mismatches, `dtheta_dt` `0.0`,
  seeded walker trajectories bitwise equal.
- Anything else **requires** `neural_angle_dist=None, angle_weight=None` and
  raises `ValueError` otherwise; with no `percep_model` supplied, one of that
  form is built (the plain path still gets the class default). The guard keys
  on the argument being *set*, so `nu=1` on a warped model is refused too and
  the configuration means the same thing at every nu. Everything *numerical*
  keys instead on `_nu_active` (set **and** != 1), which is what keeps the
  default path untouched.

**Steps 1-4 as planned:** `nu_cosine` (step 1, with the load-bearing
`convert_angles` wrap in `dgamma_dt`), the `_discrim_A` fallback (step 2), the
`0.85` probe (step 3, gated on `_nu_active`), and `plot_cosine` ->
`plot_nu_cosine` plus `plot_dtheta_dt` (step 4). `plot_dtheta_dt` keeps IEM's
`gamma=False` neutral-seed mode but implements it in its own loop
(re-seeding `self.gamma = 0.1 + 0j` per heading, restoring afterwards) so
`NBM.dtheta_dt`'s contract is unchanged; the seed is the egocentric image of
IEM's `0.1*exp(i*theta)`.

**Verification (new measurements).**

| check | result |
|---|---|
| rotation identity, 3 geometries x nu in {1, 0.5, 2}, 150 random states each | max err **2.4e-15** |
| `dtheta_dt` identity, same grid | **8.9e-16** |
| stable-count raster, 20x21 x {delta, circle} x nu x both criteria (5040 cells) | **2 cells differ**, both NBM reporting one MORE stable equilibrium |
| every IEM equilibrium accounted for, same raster (~3200 equilibria) | **0 genuinely missing** |
| `_discrim_A` analytic vs numerical fallback at nu=1 | **0/446** disagreements |
| full pytest suite | 143 passed |

The three residual differences, all understood:

1. **Occlusion shadow (caveat 5a).** 22 of ~950 circle equilibria per nu sit
   at `N_visible = 1 < N_total = 2`; there the heading still agrees to 1e-14
   and only `R` differs, by **4.5e-5**. Setting `beta = N_visible/T` at those
   states collapses it to **1.7e-14**, confirming the mechanism exactly.
   Per the decision recorded in CLAUDE.md, `beta` stays scene-independent.
2. **Seeding (caveat 5b).** At delta/nu=0.5 the `0.85` probe finds a genuine
   *stable* equilibrium at `R = 0.7615` (residual **3.3e-15** under IEM's own
   `dgamma_dt`) that IEM's radius-0.5 multistart misses — the 2 differing
   raster cells, at the mirror pair (1.75, +-1). NBM never found fewer.
3. **Near-saddle-node dedup.** At circle/nu=0.5 two equilibria 0.018 rad and
   0.009 in R apart fall inside `sc_equilib`'s `(0.02, 0.01)` merge window, so
   NBM reports the stable member where IEM lists both. Pre-existing NBM
   behaviour, unchanged by the fold, and it never moves a stable count (the
   dropped member is the unstable one). If nu work makes this bite, the fix is
   the dedup tolerance, not the kernel.

**Walkers — the frame's one observable footprint.** §2 says the frame
difference has no footprint after the `'coupled'` removal. That is right for
everything deterministic, but not for the walker: the two frames *warm-start*
each γ relaxation differently. Measured with IEM given NBM's LSODA relaxation,
circle and capsule walks agree to **1e-8** (the remaining 1e-4 in §2 was RK45 vs
LSODA), but delta walks diverge to **O(1)** and end at different targets. That
divergence is reproduced **exactly** by a single model, solver and noise stream
with only the carry frame swapped. §11 works out what it actually is.

---

## 11. The walker carry frame (2026-08-20)

Prompted by qualitatively different `plot_walkers` output between the two
models in `ising_workbook.ipynb`: 2 delta targets at (4.33, +-2.5), observer at
the origin, nu = 0.5, `beta = 10 = N_total/T`, `K = 3`, `std = 0.1`,
`noise_exp = 0`, 40 walkers, both models seeded `default_rng(3)` so the noise
streams are identical.

**§10's "finite-relaxation artifact" was wrong.** It is a modelling
difference, and the faithful convention is IEM's.

**What it is.** Each walker step warm-starts its `dgamma_dt` solve from the
gamma the previous step ended on. NBM holds that fixed in the EGOcentric
frame; IEM holds it fixed in the ALLOcentric frame. Since

    gamma = sum_k n_k exp(i * neural_angle_k)

carries one population `n_k` per *target* (the project preprint's Glauber
formulation, and the basis of the note in `_discrim_reduced`), freezing the
populations and turning the observer by `d` sends `neural_angle_k ->
neural_angle_k - d` and hence `gamma -> gamma * exp(-i*d)`. Verified directly:
over headings and turn sizes, `|gamma_new - gamma_old*exp(-i*d)| <= 2.5e-16`
while `|gamma_new - gamma_old| = 0.22`. **The allocentric carry is exact for an
identity warp; the egocentric carry makes the neural state lag the turn by `d`
every step.** Under a non-identity warp `neural_angle_k = U(theta_k - theta)`
does not shift rigidly and *neither* carry is exact -- the same obstruction
that rules out a coupled criterion at the gamma level.

**Which knob is responsible.** A 2x2 over {carry frame} x {gamma solver}, all
running NBM's math, 40 seeded walkers, censused by which target was reached:

| carry | solver | +y | -y | neither |
|---|---|---|---|---|
| ego  | LSODA    | 18 | 17 | 5 |
| ego  | IEM RK45 | 18 | 17 | 5 |
| allo | LSODA    | 22 | 16 | 2 |
| allo | IEM RK45 | 23 | 15 | 2 |

The real IEM gives 23/15/2 and the real NBM 18/17/5, so the carry frame
accounts for the whole difference and the solver for none of it -- 40/40 walks
pick the same target when only the solver changes. This is despite IEM's
restarted RK45 leaving gamma about 2000x further from equilibrium: over 651
walker steps the returned gamma has median residual **4.2e-5** (p95 7.6e-5,
0.5% above its own 1e-4 tolerance) against LSODA's **2.0e-11**. Being slaved
*well enough* is evidently not the binding constraint; *which branch* is.

**Why it survives `dt -> 0`.** The lag is systematic -- always opposite the
turn -- not random, so it does not average out. Noise-free, K=3, refining dt:

| dt | max turn/step (rad) | \|ego - allo\| endpoint, per start |
|---|---|---|
| 0.1     | 0.288 | 0.112, 0.122, 0, 0.135, 0 |
| 0.05    | 0.143 | 0.129, **4.892**, 0, 0.134, 0 |
| 0.025   | 0.071 | 0.130, **4.887**, 0, 0.128, 0 |
| 0.0125  | 0.035 | 0.130, **4.881**, 0, 0.132, 0 |
| 0.00625 | 0.018 | 0.132, **4.874**, 0, 0.138, 0 |

Two starts agree to 0.0000 at every dt (single-basin paths -- the warm start is
irrelevant there). The rest converge to a *nonzero* limit, and the step at
which the two carries first land on different `gamma*` converges to a fixed
time `t = 0.95` as dt halves, i.e. a genuine fold crossing in the continuum
limit rather than a discretization event. On one path 84 of 170 steps then sit
on different branches, `|delta gamma*|` up to 1.0 (opposed consensus
directions) -- hysteresis holding the split open.

This scene is close to a worst case for it: the walk seeds `gamma = 1e-5`
(undecided) at a point equidistant from two symmetric targets, so the first
commitment is made right at a basin boundary.

**Scope.** `plot_walkers` and `_basin_destination` only. Every deterministic
result sits at `dtheta/dt = 0`, where the advection term is identically zero --
consistent with the 2/5040 raster agreement in §10, both of which were seeding,
not frame.

**Only the turn is rigid.** A walker step turns AND translates. Identity warp
makes the turn shift every `neural_angle_k` by the same `-dtheta`
(residual 5.6e-17), but translation moves each target's bearing by a different
amount -- measured +0.0043 and -0.0178 rad for the two targets in one step,
opposite signs -- which no rotation of gamma can represent. Along the walk:

| phase | median turn/step | median translation residual |
|---|---|---|
| before the branch split (t < 0.95) | 0.0254 | 0.0149 |
| after it, still > 1 from a target  | 0.0189 | 0.0229 |
| final approach (< 1 from a target) | 0.1190 | 0.1102 |

So the allocentric carry is **not exact either** -- it captures the turn and
drops the translation, where the egocentric carry drops both. The turn
dominates ~1.7:1 in the decision-critical phase, which is where it counts; the
large residuals sit in the final approach, after the decision is made.

**Open.** Whether to switch NBM's walker (and `_basin_destination`) to the
allocentric carry is undecided; it would change published NBM walker and basin
output. If adopted, gate it on `percep_model.warp_name is None`, NOT on
`angle_distortion_nu`: keying on nu would make nu=None and nu=1 produce
different walks despite being the same model, and would miss the plain-NBM
identity-warp configurations that get the same exactness for free. The only
exact fix is to carry the populations `n_k` rather than gamma. Tracked in
TODO.md.

---

## 12. What Sridhar et al. actually do (SI read 2026-08-20)

From `sridhar_liang_Gorbonos_2021_PNAS_supp.pdf`, on the two questions the
carry-frame issue raises.

### 12.1 Their spins are indexed by GOAL, not by ring position

- SS1.2, l.66: "Each spin i **encodes direction to one of the presented goals**
  p_i, and exists in one of two states."
- SS1.2, l.85-87, the movement rule: "The agent moves along V and **spins
  update their goal vector p_i to reflect the agent's movement. The goal vector
  p_i now points from the agent's updated location to the spin's preferred
  goal**" (plus wrapped-Gaussian directional noise sigma_e).
- SS1.7, Eq [7]-[8]: `n_i = (1/N) sum_{j in G_i} sigma_j`,
  `V = v_0 sum_i p_i n_i` -- literally `gamma = sum_k n_k exp(i*angle_k)`.
- SS1.7, Eq [15]: `dV/dt = sum_i (dn_i/dt) p_i` -- the p_i-dot term is dropped,
  exactly the advection term noted in `_discrim_reduced`.

So the spin STATE persists across a step and the DIRECTION it encodes is
recomputed from the new position. Commitment attaches to the target, not to a
ring position. **That is the allocentric/population carry.**

SS1.8 (their embedding in ring-attractor models) points the same way: l.268-270,
an activity bump "appears on a specified sector of the ring, and **rotates
concurrently with the landmark as the animal turns**" -- the bump follows the
object rather than sitting still while the object slides out from under it.

**Correction (2026-08-20).** An earlier draft of this section claimed the
carry convention and the dropped advection term were "the same question", with
a per-target reading forcing the allocentric carry and a fixed-ring-position
reading forcing the egocentric one. That was wrong. The project's own
derivation (1st draft SS2.3-2.4, Eqns 2.5, 2.9-2.10) is per-target throughout
-- the state is `n_vec`, K ODEs, one per target -- so no fixed-ring-position
alternative is in play, and the advection term is dropped for the ordinary
reason: writing `dgamma_j/dtau = sum_k [(dn_k/dtau) Q_j(theta_k) +
n_k (d/dtau) Q_j(theta_k)]` on the FAST timescale `tau = t/tau_0`, the neural
angles are frozen, so the second term vanishes. Timescale separation, not
interpretation.

**The warm start is not a frame convention at all.** Stated in the draft's own
notation it is an evaluation. The readout is

    gamma_j = sum_k n_k Q_j(theta_k)        (Q+ = cos U, Q- = sin U; SS2.5)

A slow step moves the OBSERVER, not the neurons -- the neurons relax afterwards
on `tau`. So `n_vec` is unchanged across the step and the warm start is simply
that same formula evaluated at the new neural angles:

    gamma_j(warm) = sum_k n_k Q_j(theta_k^new)

Nothing is being chosen. For an identity warp and a pure turn by `d`,
`theta_k -> theta_k - d` and `Q+ + iQ- = exp(i*theta)`, so this collapses to
`gamma -> gamma*exp(-i d)`. The egocentric carry corresponds to *not*
re-evaluating `Q_j` at the new angles at all.

Verified: under a pure turn the rotated point IS the new equilibrium (max err
**2.3e-10** over random states and turn sizes; no relaxation needed at all),
while the unrotated point sits off-branch by up to **0.303**. The equilibrium
branch is equivariant under rigid rotation of the neural angles, so rotating is
not an approximation to branch-following -- it is branch-following.

**Why exactness stops there.** With a translation (or a non-identity warp) the
`theta_k` move by DIFFERENT amounts, and `sum_k n_k Q_j(theta_k^new)` then
cannot be computed from `gamma` alone -- it needs all K components of `n_vec`.
That is the same rank-2 obstruction as the missing coupled criterion, now in
one line.

### 12.2 Their walker is not this project's walker

Sridhar's agent, Eq [4]: `V = (v_0/N) sum_i p_i sigma_i`, and "the agent moves
along V".

| | Sridhar | NBM / IEM |
|---|---|---|
| direction of travel | `arg(V)` directly | separate heading `theta` |
| turning | none -- no heading state at all | `dtheta/dt = K R sin(Theta/2)` |
| speed | `v_0 * coherence` (Eq [4]: \|V\| scales with R) | constant `v` |
| noise | wrapped Gaussian `sigma_e` on the GOAL directions `p_i` | Gaussian on the HEADING |
| neural update per step | one Metropolis-Hastings sweep (SS1.2 l.79); the equilibrated network in SS1.5 l.135 is a *separate* comparison network run 1000 steps | `run_dgamma_dt` to steady state |

The heading variable and the half-angle torque are **this project's extension**;
Sridhar has neither. Which means the carry-frame ambiguity **only exists because
of that extension** -- with no separate heading to turn, there is no "does gamma
rotate?" question to answer.

A walker closer to Sridhar would relax gamma, then step along `arg(gamma)` with
step length proportional to `R` -- no turning law, no heading noise. That is
neither model's `plot_walkers`.

---

## 13. Resolution (2026-08-20): gamma is transported

Implemented option (b). `NeuralBandModel._simulate_one_walk`,
`_basin_destination` and `plot_dtheta_dt`'s swept mode now rotate the carried
gamma by minus the turn just taken. Applied unconditionally -- not gated on
`angle_distortion_nu`, not gated on the warp -- so one convention holds across
the whole class.

**What settled it.** The decisive step in the `ising_workbook` walk is a
saddle-node, not a basin crossing. Tracking the stable landscape across it:

| step | heading | gamma at | stable equilibria |
|---|---|---|---|
| 7 | +7.35 deg | -8.89 deg | -41.68, **-8.89** (R=0.789), +23.90 |
| 8 | +6.30 deg | -7.97 deg | -41.81, **-7.97** (R=0.771), +25.86 |
| 9 | +5.38 deg | -7.17 deg | -42.01, +27.68 -- **middle branch gone** |

gamma was riding the compromise branch and that branch dies -- the
compromise-breaking bifurcation of Sridhar SS1.7. Two consequences:

1. **Away from folds the warm start is irrelevant.** Both conventions relax to
   the same equilibrium on **58/58** steps, agreeing to ~1e-11. There is no
   accuracy question to answer there.
2. **At the fold the tau_0 -> 0 limit is singular and refinement does not
   help.** Sub-stepping the crossing at M = 16, 64, 256, 1024 leaves the two
   conventions at -42.309 deg vs +29.372 deg with no sign of converging. So
   this was never a dt-truncation error that shrinks away.

The transport is the statement consistent with the model's own assumption:
gamma rides its branch, tracking it until the branch itself disappears. The
alternative leaves gamma behind by `(dtheta/dt)*dt` -- a lag whose size is the
integrator's step, i.e. an implicit `tau_0 ~ dt`, where the model has set
`tau_0 -> 0`. Anyone wanting a genuine lag should put `tau_0` in explicitly
(carry `n_vec`, integrate at finite `tau_0`), not obtain it from the step size.

**Side effect:** NBM's walker now agrees with IEM's, whose allocentric gamma
was already transported implicitly by its frame -- 37/40 matched-seed walks
pick the same target (was 32/40), the remainder being IEM's RK45-vs-LSODA
relaxation tolerance.
