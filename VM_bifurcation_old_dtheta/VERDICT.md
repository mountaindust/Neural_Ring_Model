# Bifurcation diagram verdict: VM k=0.55, two circle targets

> **⚠️ STALE, AND ITS HOPF/LIMIT-CYCLE RESULTS ARE RETRACTED (2026-08-19).**
>
> 1. **Retraction.** Everything here that rests on the *full 3×3 eigenvalues*
>    of the `(γ_re, γ_im, θ)` Jacobian — the Hopf curve, the Bautin points, the
>    "stable limit cycle of period ≈ 17.4", the head-bobbing interpretation —
>    is an **artifact of an incomplete reduction**, not a property of the model.
>    `dγ/dt` is the rank-2 readout of the K-dimensional Glauber population
>    dynamics and drops a term proportional to `dθ/dt`; taking the full
>    eigenvalues linearizes the incomplete equation. Measured on this very
>    setup under *both* old torque laws (`K·R·sin(ego)`, `K·R·sin(ego/2)`): the
>    γ-3×3 flags 7 equilibria Hopf-unstable where the exact `(n₁,n₂,θ)` system
>    is stable at every one. The `'coupled'` criterion has been removed from the
>    model; see `NeuralBandModel._discrim_reduced` for the full note. The
>    retracted sections have been deleted from this document; the scripts that
>    produced them are unchanged and still in this folder.
> 2. **Stale torque law.** This directory (formerly `VM_bifurcations/`) was
>    produced under an older heading torque and the diagnostic scripts still
>    reconstruct dθ/dt as `K·R·sin(ego)` / `K·R·sin(ego/2)` using the
>    inverse-warped egocentric angle. The model has since moved to
>    `dθ/dt = K·R·sin(arg(γ)/2)`. SC-equilibrium locations and stable/unstable
>    *counts* are provably invariant under both changes (sign-preserving).

## TL;DR

When the standard self-consistent bifurcation diagram is computed with
`NeuralBandModel.sc_equilib` for the parameter set
below, two non-obvious features appear:

1. **A small "4 / 5-stable" bullseye near (1.5, 0)** — *numerical artifact.*
   The `_discrim_A` stability test only checks the 2×2 γ-Jacobian at fixed
   heading and misses the slow heading-tracking mode; the extra "stable"
   equilibria are unstable in the heading direction. The correct count there
   is 3. **This finding stands** — it is a statement about the slow (Schur)
   mode, which the current `'reduced'` criterion tests directly, and
   `reduced` reproduces the 3-vs-5 split (regression-tested in
   [tests/test_reduced_criterion.py](../tests/test_reduced_criterion.py)).
2. **Two symmetric "0-stable" islands near (2.1, ±2.45).** The *0-stable*
   classification stands: at those cells the γ-block has a positive
   eigenvalue, so no self-consistent heading is stable under the slaved
   dynamics the walker integrates. **What was inside them is retracted** —
   the claimed Hopf-unstable focus and stable limit cycle of period ≈ 17.4
   were artifacts of the removed `'coupled'` criterion (see banner). The
   slaved walker does not head-bob there; where the symmetric SC equilibrium
   is γ-bistable it performs a smaller, faster relaxation oscillation between
   two γ-branches instead.

---

## Reproducibility

### Geometry and model parameters

```python
import numpy as np
import decision_model as model

target_locs = np.array([[4.33,  2.5],
                        [4.33, -2.5]])

targets = model.Targets(
    locs=target_locs,
    geom_name='circle',
    r=0.5,
)

percep = model.PerceptionModel(
    targets,
    focal_loc=(0, 0),     # any value; overridden per-cell during sweep
    focal_angle=0,        # any value; overridden per-cell during sweep
    neural_weight='vonmises',
    neural_angle='integral',
)
percep.k = 0.55           # vonmises concentration; broad front bias

nbm = model.NeuralBandModel(percep)   # uses defaults β=10, K=1
```

That's it for the model side — every result in this folder uses these
exact settings.

### Refinement needed in spatial scans

The 4/5-stable bullseye is *very small*. With the default
`NeuralBandModel.plot_bifurcation_diagram(...)` settings, you must use
`refinement_levels >= 3` and a base grid `num_x, num_y >= 41` to see it
at all.

Concrete scan resolutions used in the diagnostics here:

| diagnostic | grid | comment |
|---|---|---|
| Initial scan ([diagnostic_bifurc_vm.py](diagnostic_bifurc_vm.py)) | 41×41, no refinement | catches 5-stable in 1 cell, 0-stable in 2 cells; misses 4-stable entirely |
| Recount with coupled Jacobian ([diagnostic_recount_grid.py](diagnostic_recount_grid.py)) | 61×61, no refinement | sees 5-stable (1 cell) and 4-stable (2 cells); 0-stable in 4 cells; the 1D y=0 scan uses 121 points on x∈[0, 3] |
| Island map ([diagnostic_island_final.py](diagnostic_island_final.py)) | 81×61 over x∈[1, 3], y∈[1.5, 3] | resolves the upper 0-stable island as a thin arc of 26 cells |
| Skeleton ([diagnostic_arc_skeleton.py](diagnostic_arc_skeleton.py)) | 121×79 over x∈[1, 3.5], y∈[1.5, 2.8] | resolves the Hopf curve and SN curve as separate features |
| Arc slices ([diagnostic_arc_bifurcation.py](diagnostic_arc_bifurcation.py)) | 1D, 301 x-samples on [1, 3.5], at six y-values | resolves the narrowing of the Hopf interval (Δx ≈ 0.05 at y=2.55 down to ≈ 0.01 at y=2.05) |

To reproduce the user's notebook view (`compare_sc_vm.ipynb` last cell), the
notebook calls:
```python
neur_model_vm.plot_bifurcation_diagram(
    pool=pool, wb_plot=True, ax=None,
    refinement_levels=4, max_count=None,
    num_x=41, num_y=41,
    title='VM model equilibrium plot',
)
```
That's effective resolution `(41-1)·2⁴+1 = 641` virtual pixels per side, fine
enough to render the bullseye and the 0-stable arcs visibly.

### Stability criterion used by these scripts

`sc_equilib`'s historical default `_discrim_A` checks only the γ-Jacobian at
fixed heading. The scripts here instead form the full 3×3 Jacobian of the
(γ_re, γ_im, θ) system with `dθ/dt = K·R·sin(ego_angle)` by finite difference
(see e.g. `coupled_jacobian_max_re` in
[diagnostic_arc_skeleton.py](diagnostic_arc_skeleton.py)) and take **all
three eigenvalues**. That test is the removed `'coupled'` criterion and is
**not valid** (see banner): only its *fast-block* content (the 2×2 γ block)
and the *sign of det J* — i.e. what the current `'reduced'` criterion uses —
carry over. Results below that depend on the third eigenvalue have been
removed.

### How to run the scripts

The scripts import `decision_model` from the parent directory. From the
`VM_bifurcations/` folder:

```
python diagnostic_bifurc_vm.py
python diagnostic_recount_grid.py     # pool size from parallel_config.get_n_workers
python diagnostic_island_final.py
python diagnostic_arc_bifurcation.py
python diagnostic_arc_skeleton.py
python diagnostic_island_dynamics.py
python diagnostic_coupled_jacobian.py
```

Each script writes its `.png` outputs into the current directory.

---

## Why the 4/5-stable region is spurious

`NeuralBandModel.sc_equilib` historically called `_discrim_A` (decision_model.py:1942)
to classify each self-consistent equilibrium. `_discrim_A` is the
discriminant of the **2×2 γ-Jacobian at fixed heading** — it asks "is this
γ-equilibrium stable as a fixed point of `dgamma_dt` with `focal_angle`
held constant?"

But the bifurcation plot is supposed to count fixed points of the
**coupled 3D system** (γ_re, γ_im, θ), where
`dθ/dt = K·R·sin(ego_angle)`. The two criteria can disagree.

At (1.5, 0), 5 self-consistent equilibria exist:

| θ_eq | R | _discrim_A | coupled 3×3 eigenvalues | verdict |
|---|---|---|---|---|
| ±0.7209 (toward target) | 0.616 | True | -0.99, -0.48 ± 0.61j | **stable** |
| 0.0000 (midpoint) | 0.433 | True | -0.92, -0.57, -0.10 | **stable** |
| ±0.1606 (intermediate) | 0.435 | True | -0.98, -0.34, **+0.31** | **saddle** |

The two intermediate equilibria pass `_discrim_A` (largest γ-eigenvalue
≈ -0.34) but acquire a positive eigenvalue along the heading direction.
Confirmed dynamically: integrating the coupled ODE from each "saddle"
with a small dθ perturbation, the trajectory drifts to either the
midpoint or the on-target equilibrium.

When the 61×61 grid is rescanned using the full coupled Jacobian as the
criterion ([diagnostic_recount_grid.py](diagnostic_recount_grid.py)):

| count | _discrim_A | coupled |
|---|---|---|
| 0 | 4 | 4 |
| 1 | 2042 | 2042 |
| 2 | 1577 | 1639 |
| 3 | 95 | 36 |
| **4** | **2** | **0** |
| **5** | **1** | **0** |

The 4 and 5 stable regions vanish entirely. A few cells near (3.8, ±1.5)
also drop from 3 → 2 for the same reason.

See [diagnostic_recount_compare.png](diagnostic_recount_compare.png) (3-panel:
current plot, true count, difference) and
[diagnostic_y0_slice.png](diagnostic_y0_slice.png) (1D bifurcation along y=0).

---

## The 0-stable islands (retracted content removed)

There is no occlusion at (2.10, 2.45) — both targets are visible, and the
configuration is asymmetric. The cells are genuinely 0-stable under the
slaved dynamics: the γ-block at the unique self-consistent equilibrium has a
positive eigenvalue, so no fixed heading is attracting.

**Everything this document previously said about what happens *inside* the
islands has been deleted** (Hopf curve, Bautin points, the period-17.4 limit
cycle, the bifurcation cascade across the arc, the head-bobbing
interpretation). All of it came from the third eigenvalue of the γ-level 3×3
Jacobian — the removed `'coupled'` criterion — which linearizes an incomplete
equation. Re-measured on this setup under both old torque laws, the exact
`(n₁, n₂, θ)` population system has **no** Hopf instability at any of the
cells the γ-3×3 flagged (7 → 0). The saddle-node curve and the equilibrium
counts are unaffected and stand.

See [diagnostic_arc_slices.png](diagnostic_arc_slices.png) and
[diagnostic_island_final.png](diagnostic_island_final.png) for the raw scans;
read the stability colouring in them as `'coupled'` output, i.e. not current.

---

## Are the saddle equilibria really equilibria of the 3-eq system?

Yes, exactly. Every self-consistent equilibrium γ = R + 0j at heading
θ* gives `angle(γ) = 0`, hence `ego_angle = inv_neural(0) = 0` (the
integral neural map sends 0 → 0 by symmetry of the weight). Then
`dθ/dt = K·R·sin(0) = 0` and `dγ/dt = 0` by the search criterion. All
three RHS components vanish identically, not just to numerical
tolerance.

The converse also holds: any 3-eq fixed point with R > 0 must satisfy
`sin(ego_angle) = 0`. Taking the principal branch
(`ego_angle = 0`; the π branch is "looking 180° away from consensus"),
this forces `angle(γ) = 0`, i.e. γ is real positive. So self-consistent
search captures **every** non-trivial 3-eq fixed point — no extras, no
missing ones.

Numerical confirmation: at all 13 saddles checked across three test
points, |dγ/dt| = O(10⁻¹⁶) and |dθ/dt| = 0.0 exactly. The "saddles"
are genuine 3-eq equilibria with positive Jacobian eigenvalues, not
artifacts of the search.

---

## Suggested fix to `decision_model.py`

In `_count_stable_at` (decision_model.py:2163), add a coupled-Jacobian
eigenvalue check after `_discrim_A` passes. ~6 extra `dgamma_dt`
evaluations per candidate (cheap relative to root-finding). The same
overcounting almost certainly affects `IsingExtModel`'s bifurcation
plots — its `_discrim_A` is structurally identical.

---

## Diagnostic scripts in this folder

- [diagnostic_bifurc_vm.py](diagnostic_bifurc_vm.py) — initial coarse scan
  + per-point sweep with both `_discrim_A` and a coupled-ODE drift check.
  Generates a text report with the per-point breakdown.
- [diagnostic_coupled_jacobian.py](diagnostic_coupled_jacobian.py) —
  coupled 3×3 Jacobian at suspect points, plus an ODE probe inside the
  0-stable island starting from many initial headings.
- [diagnostic_recount_grid.py](diagnostic_recount_grid.py) — full 61×61
  grid recount with both criteria. Produces
  [diagnostic_recount_compare.png](diagnostic_recount_compare.png) and
  [diagnostic_y0_slice.png](diagnostic_y0_slice.png).
- [diagnostic_island_dynamics.py](diagnostic_island_dynamics.py) —
  verifies saddles are 3-eq equilibria; long-time integration (its "limit
  cycle" is the retracted γ-level artifact); x-slice at y=2.45 through the
  island. Produces
  [diagnostic_island_long_dynamics.png](diagnostic_island_long_dynamics.png).
- [diagnostic_island_final.py](diagnostic_island_final.py) — fine 81×61
  map of the upper island. Produces
  [diagnostic_island_final.png](diagnostic_island_final.png).
- [diagnostic_arc_bifurcation.py](diagnostic_arc_bifurcation.py) — slice
  scans at y=2.05, 2.15, 2.25, 2.35, 2.45, 2.55 with full
  per-equilibrium stability tracking, plus bifurcation event detection
  (SN-births, Hopf transitions). Produces
  [diagnostic_arc_slices.png](diagnostic_arc_slices.png).
- [diagnostic_arc_skeleton.py](diagnostic_arc_skeleton.py) — separates the
  (retracted) Hopf curve from the SN curve (boundary of n_eqs jump) on a
  121×79 grid. Only the SN curve is meaningful. Produces
  [diagnostic_arc_skeleton.png](diagnostic_arc_skeleton.png).
