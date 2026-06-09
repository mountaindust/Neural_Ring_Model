---
paths:
  - "basin_estimation/**"
---

# Basin-of-attraction estimator — working notes

Auto-loads when working under `basin_estimation/`. Full detail is in [findings.md](../../basin_estimation/findings.md) (eleven-step plan, results) and [free_energy_derivation.md](../../basin_estimation/free_energy_derivation.md) (the F̂ derivation); all test scripts pass.

**Status: vetted, not yet wired into `decision_model.py`.** Public entry point `compute_basins_at_focal_loc(focal_loc, *, scan_single_stable=False)` (in `basin_estimation/basin_via_theta.py`) returns a dict with `basins`, `stable_count`, `unstable_count`, and `sentinel` (None, or a reason string for Hopf-island / perception-collapse / partial-success cells; basin-dict shape per cell class in findings.md §10.2). Both prerequisite modeling changes (sin(Θ*/2) dθ/dt with K=2; warp/weight decouple) are **DONE**, and the Steps 5–9 calibration points were **re-vetted under the new law and confirmed invariant** (findings.md §0). What remains is integration + per-cell rendering (rules in findings.md §10.4), not re-vetting.

**Load-bearing facts** (full detail in findings.md):
- **ΔF_γ** — the F̂-barrier from a stable to a γ-saddle at fixed θ — is the γ-noise robustness scalar, MC-validated against Kramers within a factor of 2, **but only in multistable cells**; 1-stable cells have no second basin, so the deliverable there is a direction arrow + optional stiffness glyph, not a robustness color (§6, §8, §13.6). F̂ is Hamiltonian/Glauber-derived, independent of dθ/dt.
- **Basin boundaries** come in four kinds: smooth saddle, γ-fold (γ-branch jump), perception-collapse (R→0 under tight cutoffs), and — under the half-angle law — a **branch-cut** at the facing-away heading (f jumps ±K·R as γ_eq crosses the negative real axis). The first three are γ-side / turning-law-invariant; the branch-cut's existence is set by dθ/dt at the antipode (§4, §7).
- **Cost**: ~4.2 min parallel (32 cores) for a 41×41 grid with basins; multistable cells dominate, 1-stable/Hopf cells are nearly free (§11).

**Open, and dθ/dt-dependent.** dγ/dt is Hamiltonian-derived; dθ/dt is a phenomenological turning law (K and possibly its form may still move). This decouples cleanly: the γ-side outputs (slow manifold, γ-folds, ΔF_γ, SC-eq locations, basin counts/widths bounded by saddles/folds) are insulated from the turning-law uncertainty, as is γ-noise robustness. What stays tied to dθ/dt: θ-noise barrier magnitudes, and two specific open items — (a) whether θ-noise (the `std` knob in `plot_walkers`) is a faithful proxy for physical γ-noise (Glauber T, 1/N), deferred from Step 8 and only resolvable once dθ/dt is fixed (§13.4); (b) the θ-noise basin "depth" at the branch-cut/antipode, which needs its own first-passage treatment (the branch-cut is a repelling fork, not a Kramers barrier) once dθ/dt is fixed.

**Basin-width gotcha (findings.md §14, 2026-06-08).** The basin *widths* `basin_features` returns are γ-*branch* extents from the warm-start scan — correct at saddle edges but **wrong at γ-fold edges** (e.g. (4.0,1.5) far stable: 5° branch extent vs 68° true decision basin). For a θ-noise walker the deliverable must render **basin-attribution** decision widths via the slaved-flow bisection in `basin_estimation/fold_kick_demo.py` (the §4.7/§13.5(5) method, now implemented), and keep the scan-fold location + R-collapse as a *separate* "γ-branch fragility / near-SN" glyph — not as the basin width. See [TODO.md](../../TODO.md) for the two-panel-plot and §9-recompute work items.
