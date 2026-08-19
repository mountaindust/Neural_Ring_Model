# Stale: γ-level "coupled model" starting code

**Status: stale. Nothing in this directory is a current result.** It is kept
only as *starting code* for an eventual, correct comparison between a fully
coupled model and the strict-timescale-separation model the project actually
uses.

---

## Why it is stale

These scripts were written to compare the `'reduced'` (timescale-separated)
stability criterion against a `'coupled'` criterion that took the **full 3×3
eigenvalues** of the `(γ_re, γ_im, θ)` Jacobian. **That criterion was removed
from `decision_model.py` on 2026-08-19, because it is not a stability test for
the model.**

`dγ/dt` is not an equation of motion. It is the rank-2 *readout* of the
K-dimensional Glauber population dynamics — one population `n_k` per visible
target (see the project preprint's Glauber section). The γ-ODE is
obtained by differentiating

    γ = Σₖ nₖ · exp(i·θ̂ₖ)

and keeping **only** the `dnₖ/dt` term. The term that is dropped,

    −i · θ̇ · Σₖ nₖ · U′(θₖ) · exp(i·θ̂ₖ)

is nonzero whenever the observer is turning, because the neural angles `θ̂ₖ`
move with the heading. Sridhar et al. (2021) drop the analogous term in SI
Eq. [15] and use it only at fixed target bearings ("targets at infinity",
SI §1.7); the preprint's own route is the timescale separation — its
navigational model is driven by the equilibrium consensus γ*.

At `θ̇ = 0` the dropped term vanishes identically. So **self-consistent
equilibria, the `'reduced'` criterion, and `'discrim_a'` are all unaffected** —
they are as correct as they ever were. But the *full* eigenvalues linearize the
incomplete equation, and they report instabilities the model does not have.

**Measured on this directory's own setup** (vonmises k=0.55, targets at
(4.33, ±2.5), r=0.5): across 403 self-consistent equilibria in the island
region, the γ-level 3×3 Jacobian flags **6** as Hopf-unstable — with an
attracting limit cycle of |γ|-amplitude 2.7e-2 — while the exact
`(n₁, n₂, θ)` population system is **stable at every one of them**. The same
holds under both older torque laws (`K·R·sin(ego)` and `K·R·sin(ego/2)`):
7 flagged by the γ-3×3, 0 by the exact system.

So the "Hopf island", the Bautin/degenerate-Hopf points, the head-bobbing
limit cycle and its period are **artifacts of the reduction**, not properties
of the model. The write-up that used to be in this file asserted them as
findings; it has been deleted rather than corrected.

It cannot be repaired at the γ level. Under a non-identity warp `U` the
dropped term is not proportional to γ (because `U′` differs from target to
target), so it is not a function of γ at all.

---

## What a correct comparison would require

The object to linearize or integrate is the **(K+1)-dimensional population
system**, state `(n₁, …, n_K, θ)`:

    ṅₖ = (1/τ₀) · [ ρₖ(θ) · σ( 2β Σℓ nℓ J(θ̂ₖ(θ), θ̂ℓ(θ)) ) − nₖ ]
    θ̇  = κ · |γ| · sin( arg(γ) / 2 ),        γ = Σₖ nₖ · exp(i·θ̂ₖ(θ))

Notes for whoever picks this up:

- **It is cheap.** K is the number of *visible* targets, typically 2–3, so the
  Jacobian is 3×3 or 4×4 — no more expensive than what these scripts already
  do. At a self-consistent equilibrium the populations are known in closed
  form (`Θ = 0` there, so `nₖ* = ρₖ·σ(2βR·cos θ̂ₖ)`), so it can be evaluated
  at every equilibrium `sc_equilib` already returns, with no extra
  root-finding.
- **It needs an explicit neural timescale `τ₀`**, which the model does not
  currently expose. This is not optional: the verdict depends on the
  dimensionless ratio `κτ₀`. The γ-ODE implicitly assumes `κτ₀ = 2` — turning
  *faster* than neural relaxation, the opposite of the separation assumed
  everywhere else in the project. `'reduced'` is the `τ₀ → 0` limit, which is
  the dynamics `plot_walkers` and `_basin_destination` actually integrate.
- **There is a genuine Hopf, but not where the γ-ODE put it.** At the island
  centre (2.10, 2.45) the exact system is stable for `κτ₀ ≳ 0.4`, has a true
  unstable complex pair around `κτ₀ ≈ 0.2` (neural relaxation ~5× faster than
  turning), and goes over to the slaved picture as `κτ₀ → 0`. Mapping that
  window across the island is the obvious first job — one location is not a
  result.
- **Sanity check to reproduce first:** with θ frozen the rank-2 reduction is
  exact, so the γ-ODE and the population system must agree to solver
  tolerance (~1e-12). If they do not, the population implementation is wrong.
  A second check: for K > 2 the population Jacobian's n-block spectrum is
  `eig(A) ∪ {−1/τ₀}^(K−2)`, the extra directions being transverse to the
  rank-2 γ subspace.

---

## Files

| file | state |
|---|---|
| `compare_reduced_vs_coupled.py` | **Broken.** Calls `sc_equilib(stability_criterion='coupled')`, which now raises. Scan/plot/report scaffolding is reusable. |
| `island_anatomy.py` | **Runs, partly invalid.** Its per-equilibrium `saddle` / `slow-unstable` classes and the near-γ-fold conditioning report are valid (they use only the γ block and `det J`); its `hopf` class and the Hopf-cell detector are the artifact. Good scaffolding for the per-equilibrium eigenvalue table. |
| `cycle_birth_death.py` | **Runs, invalid.** Traces the "supercritical Hopf" of the γ-level ODE. The transverse-parameter-sweep + amplitude/period measurement machinery is exactly what a `κτ₀` sweep needs. |
| `reduced_vs_full_dynamics.py` | **Runs, half invalid.** Its "FULL" system is the γ-level coupled ODE (artifact); its "REDUCED" (slaved) half is valid. The side-by-side figure layout is reusable. Its reference to `reduced_dynamics_anatomy.py` now points at [theory/reduced_dynamics_anatomy.py](../theory/reduced_dynamics_anatomy.py). |

Figures previously generated here (`island_anatomy.png`, `reduced_vs_full.png`,
`compare_vonmises_island.png`, …) are not committed; regenerating them from the
current scripts would reproduce the artifact.

---

## What survived elsewhere

Not everything from this analysis was wrong. These results stand and live in
the main docs:

- The **block-determinant identity** and the `sign(det J)` formulation of the
  slow test — well-conditioned at a γ-fold where `λ_slow` itself diverges.
  Unit-tested in [`tests/test_reduced_criterion.py`](../tests/test_reduced_criterion.py).
- **`discrim_a` over-counts** by exactly the slow heading-tracking mode; the
  worked (1.5, 0) example (reduced = 3, discrim_a = 5) is regression-tested.
- The **near-saddle-node skin** where a γ-block eigenvalue passes through zero
  and the separation genuinely breaks down — a property of `A`, unaffected.
- The **γ-bistability relaxation oscillation** in the 0-stable band: a
  property of the slaved system, and the thing the walker actually does.

See the "Stability criterion" section of [CLAUDE.md](../CLAUDE.md) and
`NeuralBandModel._discrim_reduced` for the current picture.
