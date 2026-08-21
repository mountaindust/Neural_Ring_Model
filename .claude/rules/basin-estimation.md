---
paths:
  - "decision_model/_nbm_basins.py"
---

# Basin-of-attraction overlay — code-editing gotchas

Auto-loads with `decision_model/_nbm_basins.py`. This is the thin "don't break it while
coding" note for the basin-wheel overlay (`plot_bifurcation_diagram(overlay_basins=True)`,
`basin_arcs_at_focal_loc`, `_basin_destination`, `_overlay_basin_wheels`).
**Full theory, findings, and derivations are in
[theory/basins_of_attraction.md](../../theory/basins_of_attraction.md)** (and
[free_energy_derivation.md](../../theory/free_energy_derivation.md)). The
standalone `basin_estimation/` vetting module was retired 2026-06 (in git
history).

**What the overlay computes.** A fixed **neutral seed** (`arg(γ)=0`,
`|γ|=R_seed=0.15` — the *indecision* range, below committed R≳0.4, above the
R≈0 arg-degeneracy) is swept over headings; each runs the **slaved flow**
(`_basin_destination`: re-equilibrate γ via `run_dgamma_dt` warm-started each
step, then half-angle torque) to its destination stable, with
**destination-flip bisection** for boundaries. This yields single-valued,
history-independent basin arc-widths per stable direction.

**Two load-bearing gotchas:**

1. **Basin-width gotcha.** The overlay renders **decision** basin widths from
   the neutral-seed slaved-flow bisection — **not** γ-*branch* scan-fold
   extents. The two coincide at smooth-saddle edges but **diverge at γ-fold
   edges** (e.g. (4.0,1.5) far stable: 5° branch extent vs 68° true decision
   basin). Crossing a γ-fold makes the slaved γ jump branches but is often
   *recoverable* — necessary, not sufficient, for a decision switch. Never
   render a scan-fold width as the basin width. (theory §7.)

2. **ΔF_γ robustness is multistable-only.** The γ-noise barrier ΔF_γ is a
   validated robustness scalar **only in cells with ≥2 stable equilibria**.
   1-stable cells have no second basin — deliver a direction arrow (+ optional
   F̂-Hessian stiffness glyph), never a robustness color. (theory §4, §5.)

dγ/dt (hence F̂, ΔF_γ, γ-folds, basin counts) is Hamiltonian-derived and
**turning-law-invariant**; only θ-noise barrier magnitudes track dθ/dt.
Deferred work (committed-walker basin, fragility glyph, θ-noise escape-rate)
is in [TODO.md](../../TODO.md).
