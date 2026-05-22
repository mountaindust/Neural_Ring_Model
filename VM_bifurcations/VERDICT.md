# Bifurcation diagram verdict: VM k=0.55, two circle targets

## TL;DR

When the standard self-consistent bifurcation diagram is computed with
`NeuralBandModel.gamma_equilib(focal_angle=True)` for the parameter set
below, two non-obvious features appear:

1. **A small "4 / 5-stable" bullseye near (1.5, 0)** — *numerical artifact.*
   The `_discrim_A` stability test only checks the 2×2 γ-Jacobian at fixed
   heading; the full coupled 3×3 (γ_re, γ_im, θ) Jacobian shows the extra
   "stable" equilibria are saddles in the heading direction. True
   coupled-stable count there is 3.
2. **Two symmetric "0-stable" islands near (2.1, ±2.45)** — *real.* Inside
   each, the unique self-consistent equilibrium is Hopf-unstable and a
   small-amplitude **stable limit cycle of period ≈ 17.4** is the actual
   attractor. The walker would settle into a steady "head-bobbing"
   oscillation rather than reaching any fixed heading.

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

nbm = model.NeuralBandModel(percep)   # uses defaults T=0.2, K=1
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

### Coupled-stability criterion

`gamma_equilib`'s default `_discrim_A` checks only the γ-Jacobian at fixed
heading. The "true" coupled stability requires forming the 3×3 Jacobian of
the (γ_re, γ_im, θ) system, where `dθ/dt = K·R·sin(ego_angle)`. All
diagnostic scripts in this folder do this check by finite difference; see
e.g. `coupled_jacobian_max_re` in [diagnostic_arc_skeleton.py](diagnostic_arc_skeleton.py).

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

`NeuralBandModel.gamma_equilib` calls `_discrim_A` (decision_model.py:1942)
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

## Why the 0-stable islands are real (and what's inside them)

There is no occlusion at (2.10, 2.45) — both targets are visible. The
configuration is asymmetric: target 0 (close) is at distance 2.23,
bearing +1.3°, half-extent 13°; target 1 (far) is at distance 5.43,
bearing -65.8°, half-extent 5.3°. The observer sits almost level with
the upper target's y-coordinate.

At this point only one self-consistent equilibrium exists
(θ ≈ -0.088, R ≈ 0.754). Coupled-Jacobian eigenvalues are
**-1.00 and +0.082 ± 0.197j** — a Hopf-unstable focus.

Long-time integration (t=4000, LSODA, rtol=1e-10) reveals a clean
**stable limit cycle** with period ≈ **17.378 ± 0.04** time units.
Phase portrait in (θ, ego_angle) is a closed orbit encircling the
unstable equilibrium:

- θ oscillates between -0.155 and -0.006 (amplitude ≈ 0.075 rad ≈ 4.3°)
- ego_angle oscillates between -0.040 and +0.037 rad
- |γ| stays nearly constant at 0.755

So a walker placed inside the 0-stable island does not converge to any
heading — it settles into a slow heading-oscillation ("head-bobbing").
This is qualitatively different from "decision paralysis" (which would
be slow drift); it's a genuine periodic attractor.

See [diagnostic_island_long_dynamics.png](diagnostic_island_long_dynamics.png).

### Shape of the island and the bifurcation skeleton

A fine 121×79 scan over (x∈[1, 3.5], y∈[1.5, 2.8]) — see
[diagnostic_arc_skeleton.png](diagnostic_arc_skeleton.png) — shows that
**the Hopf curve and the saddle-node curve are entirely separate** in
this parameter window:

- **Hopf curve (magenta loop in the skeleton plot)** is a closed-loop
  curve inside the 1-equilibrium region, spanning roughly (1.75, 2.18)
  up to (2.5, 2.63) and back. It encloses the 0-stable island. Crossing
  it, the unique self-consistent equilibrium loses stability via
  Andronov-Hopf and a small-amplitude stable limit cycle is born. The
  width of the Hopf-unstable interval narrows monotonically with y:
  Δx ≈ 0.05 at y=2.55, Δx ≈ 0.04 at y=2.45, Δx ≈ 0.03 at y=2.35,
  Δx ≈ 0.02 at y=2.15, Δx ≈ 0.01 at y=2.05. Both ends of the curve
  close to a *degenerate Hopf* point where the eigenvalue's positive
  peak just touches zero — these are codim-2 generalised Hopf (Bautin)
  points.

- **Saddle-node curve** is the boundary between the 1-equilibrium region
  (blue, upper-left) and the 3-equilibrium region (yellow, lower-right).
  Crossing it from above to below, two new equilibria are born: one
  stable node + one saddle.

The two curves do **not** intersect in the resolved window. There is a
thin sliver of "1-equilibrium, stable" between the SN curve and the
Hopf arc — the equilibrium has just survived SN annihilation but has
not yet been Hopf-destabilised.

### Bifurcation cascade across the arc

Following a path from deep inside the 3-equilibrium region (e.g.
(3.0, 1.8)) up across the saddle-node curve, through the 1-stable
wedge, and into the 0-stable arc:

1. **Inside the 3-eq region**: 3 equilibria coexist.
   (a) the *original* upper-branch equilibrium (θ ≈ +0.05 at y=2.05),
   (b) a SN-born stable node (θ ≈ -1.13 at (3.0, 2.05)),
   (c) a SN-born saddle between them. (a) and (b) are stable.
2. **Crossing the SN curve upward**: (b) and (c) collide and annihilate
   in a standard saddle-node bifurcation. Only (a) remains, still
   stable. Coupled-stable count = 1.
3. **Wedge between SN and Hopf curves**: 1 stable equilibrium (a).
4. **Crossing the Hopf curve into the arc**: (a)'s complex eigenvalues
   pass through the imaginary axis — Andronov-Hopf bifurcation. (a)
   becomes an unstable focus and a small-amplitude stable limit cycle
   appears around it. Coupled-stable count = 0.
5. **Inside the arc**: 1 unstable focus + 1 stable limit cycle.
6. **Crossing the Hopf curve again** (other side of the loop):
   (a) regains stability. Limit cycle shrinks back into (a). 1 stable eq.

So **only one stable equilibrium genuinely "disappears"** in the
cascade — the SN-born node (b) — and it does so by colliding with the
saddle (c) in an ordinary saddle-node bifurcation. The other stable
equilibrium (a) persists across all three regions; what changes is its
stability, via a Hopf bifurcation on a totally separate curve.

The lower symmetric island (y < 0) has the same structure mirrored
across y=0. See [diagnostic_arc_slices.png](diagnostic_arc_slices.png) for
the per-y x-sweeps and [diagnostic_island_final.png](diagnostic_island_final.png)
for the upper-island close-up.

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
  verifies saddles are 3-eq equilibria; long-time integration showing
  the limit cycle; x-slice at y=2.45 through the island. Produces
  [diagnostic_island_long_dynamics.png](diagnostic_island_long_dynamics.png).
- [diagnostic_island_final.py](diagnostic_island_final.py) — fine 81×61
  map of the upper island. Produces
  [diagnostic_island_final.png](diagnostic_island_final.png).
- [diagnostic_arc_bifurcation.py](diagnostic_arc_bifurcation.py) — slice
  scans at y=2.05, 2.15, 2.25, 2.35, 2.45, 2.55 with full
  per-equilibrium stability tracking, plus bifurcation event detection
  (SN-births, Hopf transitions). Produces
  [diagnostic_arc_slices.png](diagnostic_arc_slices.png).
- [diagnostic_arc_skeleton.py](diagnostic_arc_skeleton.py) — separates
  Hopf curve (level set of complex-eigenvalue real part) from SN curve
  (boundary of n_eqs jump) on a 121×79 grid. Produces
  [diagnostic_arc_skeleton.png](diagnostic_arc_skeleton.png).
