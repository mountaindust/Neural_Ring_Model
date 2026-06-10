---
paths:
  - "VM_bifurcation_old_dtheta/**"
  - "bifurc_plots/**"
  - "reduced_criterion/**"
---

# Bifurcation explorations — script roles & physical phenomena

Auto-loads under the exploratory parameter-sweep directories. Authoritative write-ups: [VM_bifurcation_old_dtheta/VERDICT.md](../../VM_bifurcation_old_dtheta/VERDICT.md) (old `sin(ego)` law) and [reduced_criterion/README.md](../../reduced_criterion/README.md) (current half-angle-law analysis of `reduced` vs `coupled`).

## Script roles

- **`bifurc_plots/`** — exploratory parameter-sweep scripts. `neural_weight_sweep.py` (full weighting) and `neural_weight_sweep_angle_only.py` (warping-only) are companions; `bifurcation_compare_discrim_vs_coupled.py` compares stability criteria; `arc_skeleton_and_island_dynamics.py` is the upper-arc / island-dynamics combined figure. Not publication-quality — settings are sized for fast iteration on a many-core machine. More function-space exploration is needed before any of these can be locked in for publication.
- **`reduced_criterion/`** — analysis of the `'reduced'` (timescale-separated) stability criterion vs `'coupled'`, and the dynamics where they differ. Canonical scripts + figures + [README.md](../../reduced_criterion/README.md); the block-determinant-identity derivation is in [block_determinant_identity.tex](../../reduced_criterion/block_determinant_identity.tex).
- **`VM_bifurcation_old_dtheta/`** — diagnostic scripts + [VERDICT.md](../../VM_bifurcation_old_dtheta/VERDICT.md) for the Hopf-island / saddle-node skeleton near (2.1, ±2.45) in the vonmises-k0.55 / two-target setup. **Old-law (`sin(ego)`) analysis** — see `reduced_criterion/` for the current picture.

Mesh sweeps and bifurcation refinement use `multiprocessing` extensively (see [PARALLEL_CONFIG.md](../../PARALLEL_CONFIG.md) for tuning).

## 0-stable "decision paralysis" band under power warping

With `c=0.5` power warping, a transition band between 1-stable and 2-stable regions has **0 stable equilibria**. This is a **genuine pitchfork-like bifurcation**, not a numerical artifact:

- `_discrim_A` formula verified correct for `ν=1` cosine kernel across 1,084 equilibria (0 mismatches vs numerical Jacobian) in all configs (delta/circle × warped/unwarped).
- Full coupled 3×3 Jacobian also shows instability — heading coupling doesn't rescue stability.
- The bifurcation occurs because power warping stretches physical angles to wider neural angles (~112–138° separation), destabilizing the transverse perturbation mode.

These are real "decision paralysis" locations where no stable consensus exists under self-consistent heading-consensus coupling.

## Hopf island with stable limit cycle (vonmises k=0.55, two-target setup)

In the parameter window analyzed in [VM_bifurcation_old_dtheta/VERDICT.md](../../VM_bifurcation_old_dtheta/VERDICT.md) (old `sin(ego)` law), the bifurcation skeleton near `(2.1, ±2.45)` includes:

- A closed **Hopf curve** (magenta loop) inside the 1-equilibrium region, ending at degenerate Hopf (codim-2 Bautin) points where the eigenvalue's positive peak just touches zero.
- A separate **saddle-node curve** between the 1- and 3-equilibrium regions.
- Inside the Hopf loop: 1 Hopf-unstable focus + 1 **stable limit cycle** (period ~17.4 old-law). Walkers exhibit steady "head-bobbing" oscillation, not convergence.
- Cascade across the arc (3-eq → SN crossing → wedge → Hopf crossing → unstable-focus+cycle → Hopf back): only one stable eq genuinely disappears (via SN); the other temporarily loses stability via Hopf and recovers.

This is a **coupled-system** phenomenon: the smooth head-bobbing limit cycle requires γ to lag θ, so it is seen by `stability_criterion='coupled'` but not by the default `'reduced'`, which correctly reports those cells as 0-stable (no fixed heading). The slaved walker the model actually integrates does **not** head-bob there; where the symmetric SC eq is γ-bistable it does a much smaller, faster **γ-bistability relaxation oscillation** instead (mechanism: fold/hysteresis between two stable γ-branches, not a Hopf).

**Current status under the half-angle law** ([reduced_criterion/](../../reduced_criterion/)): the Hopf island persists but is **near-degenerate** — the only Hopf-unstable focus found is one cell ≈(2.467, ±2.633), Re ≈ +0.0005, at the fold/Bautin tip; full-system cycle period ~14. The equilibria across the 0-stable band are unstable **saddle-foci** (γ-block has a positive eigenvalue, but coupling θ makes the unstable pair complex). γ-side structure (the saddle-node/fold curve, basin widths) is turning-law-invariant; θ-side fine structure rescaled by ν(0).

**How to apply:** in similar parameter regimes, `'coupled'` exposes Hopf-unstable foci + stable limit cycles, while the default `'reduced'` (and the walker) instead show 0-stable cells with at most a small relaxation oscillation. [VERDICT.md](../../VM_bifurcation_old_dtheta/VERDICT.md) is the old-law write-up (scripts, PNGs); [reduced_criterion/README.md](../../reduced_criterion/README.md) is the current half-angle-law analysis.

## Bifurcation-diagram rendering conventions (also in CLAUDE.md)

- Default colormap is viridis keyed on stable count alone, with a `max_count` kwarg. **Don't reintroduce two-axis `(n_stable, n_unstable)` color coding** without an explicit ask — it was tried and made boundaries harder to read. (The `stability_criterion='coupled'` plumbing it introduced is preserved.)
- **`boundary_dilation` kwarg** (default 1) widens each refinement pass by promoting cells sharing a corner with a disagreement cell. **Cell-center sampling deferred** — propose it (rather than further enlarging dilation) only if thin-feature artifacts persist after `boundary_dilation` and modest `num_x`/`num_y` increases.
