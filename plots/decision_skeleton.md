# Deterministic decision-track skeleton

> **Note (later work):** this documents the original skeleton feature. The tracer has
> since gained the **second-bifurcation diamond** (each arm splits into {outer, centre},
> the "two sequential binary decisions"), the `skip_unstable` option, a midline pin for
> symmetry, and the data-matched parameters (fly **`a_warp=0.65/a_weight=0.20, K=2`** —
> GODM-refit to push the first bifurcation out, see
> [three_target_fly_refine_findings.md](three_target_fly_refine_findings.md); **locust
> 35°**, `a_warp=0.40/a_weight=0.10`). For the current state and the GODM data match see
> [three_target_findings.md](../walker_analysis/three_target_findings.md).

Draws the PNAS-style "main decision tracks" for the three-target experiments
**deterministically from the model's bifurcation structure** — no hand-drawing, no
noisy random-walk sampling. Implemented in
[decision_skeleton.py](decision_skeleton.py).

The idea (from the collaborator): start where there is a single stable
consensus-heading direction and follow it; where a bifurcation makes that branch go
unstable and new stable branches appear, fork and follow each; repeat down the
cascade until the tracks arrive at all three targets. The deterministic direction
field is read straight from
`NeuralBandModel.sc_equilib(focal_loc=(x, y), stability_criterion='reduced')`
(`'reduced'` is the criterion the deterministic/slaved walker obeys).

## Deliverable figures

- [skeleton_fly.png](skeleton_fly.png), [skeleton_locust.png](skeleton_locust.png) —
  the skeleton over the empirical GODM heatmaps. The black tracks land on the bright
  heatmap ridges in both species.
- [branch_diagram_fly.png](../walker_analysis/branch_diagram_fly.png),
  [branch_diagram_locust.png](../walker_analysis/branch_diagram_locust.png) — the (x, θ) + (x, R)
  bifurcation **branch diagram** (equilibrium *directions* and coherence vs observer
  position, not just stable counts).

## CLI

```
python plots/decision_skeleton.py fly                 # skeleton over heatmap
python plots/decision_skeleton.py locust --branch-diagram
python plots/decision_skeleton.py fly --no-heatmap    # skeleton over target circles
```

## The midline bifurcation cascade (what is born/dies where)

The branch diagram shows the cascade is richer than the "one stable → goes unstable →
two appear" sketch. For the **fly** (at the refit `a_warp=0.65π`; the locust cascade is
qualitatively the same at its own `a_warp=0.40π`):

| x | event |
|---|---|
| ~1.3 | **First bifurcation**: the two compromise arms (±25°) are *born* by saddle-node — while the center (0°) is **still stable**. Two branches appear *alongside* center; they don't replace it. |
| ~2.4 | Center (0°) **destabilizes** (the inner saddles merge into it). |
| ~3.0  | Center is **reborn stable**. |
| ~3.3 | **Outer-target branches** (±80°) are *born* by a **separate** saddle-node. |

(Raising `a_warp` from the old 0.45π to 0.65π pushed every event in this cascade
outward — first bifurcation ~0.9→~1.3, etc. — which is the point: the walker commits
later and stops peeling toward the targets too early.)

**The reborn center and the outer-target branches are born at *different* x, in
distinct bifurcation events.** The reborn center comes *first* (fly ≈3.0 vs ≈3.3;
locust ≈1.55 vs ≈1.8). The second-bifurcation region is a tight cluster of
saddle-nodes (with some near-fold solver jitter, visible in the diagram).

The *mechanism* of this separation — the center is a **pitchfork** (re-stabilizes
once the outer targets swing past neural ±90° broadside) while the outer is an
**Ising saddle-node** (a marginal single-target commitment, fragile to T / weight /
separation) — is worked out in [birth_mechanism.md](../walker_analysis/birth_mechanism.md)
(parameter sweeps, kernels, the 9-gon ring; figure
[birth_mechanism.png](../walker_analysis/birth_mechanism.png)), via
[skeleton_birth_analysis.py](../walker_analysis/skeleton_birth_analysis.py).

## A modeling finding worth flagging

A **noise-free walker seeded on an arm recaptures to the *center* target** — it does
*not* ride the arm out to its outer target. The outer "main tracks" are the
stable-consensus **compromise ridges** that only *noisy* walkers populate. So you
cannot get the outer tracks from deterministic walkers; you need the
bifurcation / SC-equilibrium skeleton, which is what this module computes. (This is
the deterministic counterpart of the reborn-center recapture documented in
[three_target_analysis.md](../walker_analysis/three_target_analysis.md).)

## How the tracer works

The trunk marches from the origin along the center branch. At the first bifurcation
it **forks**, seeding a leaf per newly-stable branch, and continues as the center
leaf. Each leaf then follows ONE equilibrium branch by **(θ, R) continuity** (standard
numerical continuation):

- The **center leaf** (side 0) follows the nearest branch among *all* equilibria, so
  it rides the center branch straight through its brief SC-*unstable* interlude
  (between the first and the reborn-center bifurcation) to the center target.
- An **arm leaf** (side ±1) follows the nearest *same-side stable* branch: it rides
  the compromise branch and, when that branch dies at the second bifurcation, **jumps
  across the fold** to the same-side outer-target branch (committing to the outer
  target) instead of riding the dying inner saddle down to R→0 or grabbing the
  opposite-side center branch.

No merge step is needed — the three leaves ride distinct branches to distinct targets.

## Verification

- 3 leaves → the 3 targets, each ending at the target surface (`surf_dist` ≈ 0.04).
- **Exact** mirror symmetry between the upper/lower arms (dx = 0, dy_sum = 0).
- Center leaf stays exactly on the midline (max |y| = 0).
- Robust across `ds ∈ [0.015, 0.03]` (fly) and `ds ∈ [0.02, 0.03]` (locust at the
  default `ds = 0.02`). The fragility at coarse `ds` was under-resolution of the sharp
  second-bifurcation turn.

## Notes / caveats

- **Model parameters are copied** from the current [three_target_fly.py](../walker_analysis/three_target_fly.py)
  / [three_target_locust.py](../walker_analysis/three_target_locust.py) (which already carry the corrected
  radii — fly 0.5, locust 0.1 — and per-geometry warps). Those scripts are figure
  programs with `plot_walkers` side effects in `__main__`, so they cannot be imported;
  the constants are duplicated and flagged to **keep in sync**.
- **Locust heatmap alignment** is a best-fit similarity transform (the empirical
  locust separation ≈35° differs from the model's 40°), so that overlay is
  approximate; the fly heatmap posts equal the model targets, so the fly aligns
  ~exactly.
- The branch diagram needs R alongside θ, which `NeuralBandModel.sc_equilib` discards,
  so the module includes `sc_equilib_with_R` mirroring the canonical solver
  ([../decision_model.py](../decision_model.py)) — also flagged to keep in sync.

## Public API

- `build_fly_model(r=0.5)`, `build_locust_model(r=0.1)` — the corrected setups.
- `trace_skeleton(nm, ...) -> SkeletonTree` — the forking-streamline tracer.
  `SkeletonTree.arrivals()`, `.leaves()`, `.target_assignment()`.
- `plot_skeleton(tree, ax, transform=None, ...)` — draw the solid black tracks.
- `plot_branch_diagram(nm, y0=0.0, xlim=..., extra_y=(...), ...)` — the (x, θ)+(x, R)
  branch diagram.
- `make_figure(case, heatmap=True, ...)` — skeleton over the GODM heatmap (graceful
  fallback to target circles if the `../../GODM` data repo is absent).
- `sc_equilib_with_R(nm, focal_loc, ...)` — `sc_equilib` augmented with R.
