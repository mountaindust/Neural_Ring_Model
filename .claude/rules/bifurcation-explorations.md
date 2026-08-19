---
paths:
  - "VM_bifurcation_old_dtheta/**"
  - "bifurc_plots/**"
  - "stale_coupled_model_starting_code/**"
---

# Bifurcation explorations — script roles & physical phenomena

Auto-loads under the exploratory parameter-sweep directories. Authoritative write-up: [VM_bifurcation_old_dtheta/VERDICT.md](../../VM_bifurcation_old_dtheta/VERDICT.md) (old `sin(ego)` law).

## Script roles

- **`bifurc_plots/`** — exploratory parameter-sweep scripts. `neural_weight_sweep.py` (full weighting) and `neural_weight_sweep_angle_only.py` (warping-only) are companions; `bifurcation_compare_discrim_vs_coupled.py` and `arc_skeleton_and_island_dynamics.py` are **stale** — both call the removed `'coupled'` criterion and will raise. Not publication-quality — settings are sized for fast iteration on a many-core machine. More function-space exploration is needed before any of these can be locked in for publication.
- **`stale_coupled_model_starting_code/`** — **stale.** Scripts written against the removed `'coupled'` criterion; kept only as starting code for an eventual `(n⃗, θ)` population-level comparison. See its [README.md](../../stale_coupled_model_starting_code/README.md).
- **`VM_bifurcation_old_dtheta/`** — diagnostic scripts + [VERDICT.md](../../VM_bifurcation_old_dtheta/VERDICT.md) for the saddle-node skeleton near (2.1, ±2.45) in the vonmises-k0.55 / two-target setup. **Old-law (`sin(ego)`) analysis.** Its Hopf/limit-cycle findings were retracted (see VERDICT.md).

Mesh sweeps and bifurcation refinement use `multiprocessing` extensively (see [PARALLEL_CONFIG.md](../../PARALLEL_CONFIG.md) for tuning).

## 0-stable "decision paralysis" band under power warping

With `c=0.5` power warping, a transition band between 1-stable and 2-stable regions has **0 stable equilibria**. This is a **genuine pitchfork-like bifurcation**, not a numerical artifact:

- `_discrim_A` (old transverse `A<1` form) reported 0 stability-label mismatches vs the numerical Jacobian across 1,084 `ν=1` cosine-kernel equilibria (delta/circle × warped/unwarped) — but that sample did not hit the off-axis / low-R cases where `A<1` is necessary-not-sufficient. On a broader sweep the transverse-only test over-counts (148/5582 equilibria: radial folds + off-diagonal saddles); `_discrim_A` was completed to the full fast block (`A<1` **and** `det>0`) in 2026-07. See [free_energy_derivation.md](../../theory/free_energy_derivation.md) §6.1.
- **Untested at the population level.** The supporting "full coupled 3×3 Jacobian also shows instability" check used the removed `'coupled'` criterion and no longer counts as evidence; the fast-block and `reduced` results above stand on their own. Whether the exact `(n⃗, θ)` system also lacks a stable equilibrium here has not been checked.
- The bifurcation occurs because power warping stretches physical angles to wider neural angles (~112–138° separation), destabilizing the transverse perturbation mode.

These are real "decision paralysis" locations where no stable consensus exists under self-consistent heading-consensus coupling.

## Bifurcation-diagram rendering conventions (also in CLAUDE.md)

- Default colormap is viridis keyed on stable count alone, with a `max_count` kwarg. **Don't reintroduce two-axis `(n_stable, n_unstable)` color coding** without an explicit ask — it was tried and made boundaries harder to read. (The `stability_criterion='coupled'` plumbing it introduced is preserved.)
- **`boundary_dilation` kwarg** (default 1) widens each refinement pass by promoting cells sharing a corner with a disagreement cell. **Cell-center sampling deferred** — propose it (rather than further enlarging dilation) only if thin-feature artifacts persist after `boundary_dilation` and modest `num_x`/`num_y` increases.
