# Folding `IsingExtModel` into `NeuralBandModel`

**Status:** reviewed and verified 2026-08-18/19; **not yet implemented.** This
is the working plan plus the measurements it rests on, so the fold can be done
without re-deriving anything.

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
