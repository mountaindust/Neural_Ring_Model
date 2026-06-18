# Neural Ring Model — Project Guide

Mathematical model of collective decision-making based on Ising-type dynamics on a neural ring. This file is the durable project context loaded into every Claude Code session in this repo — theory, model architecture, and known limitations that are hard to re-derive from code or git history. **Deep solver / walker / stability internals now live in path-scoped `.claude/rules/*.md` that auto-load only when you open the matching code** — see the reference index at the bottom for what lives where.

## ⚠️ Git policy (do not deviate)

**Never `git commit` automatically, and never `git push` automatically.** Commit only when the user explicitly asks for a commit, each time — a prior commit request does not authorize future ones. **A request to commit is NOT a request to push:** push only when the user explicitly asks to push, separately. When in doubt, stage/show the diff and ask.

## Codebase layout

- [decision_model.py](decision_model.py) — the project's research code (~4200 lines). Two model classes (`NeuralBandModel`, `IsingExtModel`), a `PerceptionModel`, and a `Targets` helper.
- Jupyter notebooks (`compare_sc_vm.ipynb`, `compare_sc_beta.ipynb`, `neural_band.ipynb`, `neural_band_walker.ipynb`, `ising_workbook.ipynb`, `debug_all_unstable.ipynb`) — testing, exploration, and visualization.
- [VM_bifurcation_old_dtheta/](VM_bifurcation_old_dtheta/), [reduced_criterion/](reduced_criterion/), [bifurc_plots/](bifurc_plots/) — bifurcation/stability exploration scripts + write-ups (`VERDICT.md` is the old-`sin(ego)`-law Hopf-island skeleton; `reduced_criterion/README.md` is the current half-angle-law `reduced`-vs-`coupled` analysis). Script roles + the physical-phenomena findings are in the auto-loading `.claude/rules/bifurcation-explorations.md`. Not publication-quality — sized for fast iteration on a many-core machine.
- [basin_estimation/](basin_estimation/) — basin-of-attraction estimator vetting (see "Basin-of-attraction estimator" below; running findings in [findings.md](basin_estimation/findings.md)).
- [weighting_analysis/](weighting_analysis/) — warp-vs-weight "ears" analysis ([README.md](weighting_analysis/README.md)).
- [tests/](tests/) — unit tests (`test_broad_validation.py`, `test_intervals.py`, `test_segments.py`, `test_half_angle_torque.py`, `test_reduced_criterion.py`, …). **Running convention in [tests/README.md](tests/README.md):** `pytest tests/` runs the whole fast suite; `test_broad_validation.py` is a slow multiprocessing cross-model script excluded from pytest — run it deliberately with `python tests/test_broad_validation.py`.
- [Matlab/](Matlab/), [early_ideas/](early_ideas/) — legacy / prototype code, not part of the active model.
- [PARALLEL_CONFIG.md](PARALLEL_CONFIG.md), [parallel_config.py](parallel_config.py), [machine_config.py](machine_config.py) — per-machine worker-count settings; mesh sweeps and bifurcation refinement use `multiprocessing` extensively.
- [TODO.md](TODO.md) — engineering TODO; [TODO_Jan_2026.md](TODO_Jan_2026.md) — archived research-direction prose.

## Coordinate systems

The model lives in three distinct angular coordinate frames. Keeping them straight is essential.

1. **Allocentric** — absolute directions in the world (compass bearings to targets).
2. **Egocentric** — directions relative to the observer's current heading. `egocentric = allocentric − heading`. Center of visual field is 0.
3. **Neural** — egocentric angles passed through a neural mapping (e.g. power mapping `f(θ) = π·sign(θ)·|θ/π|^c`, or the integral-spline mapping). Models denser neural representation near the center of the visual field (foveal region).

**Why this matters mathematically:** with a non-identity warp (`c ≠ 1`), `f(a+b) ≠ f(a) + f(b)`. The neural landscape is therefore *heading-dependent*: rotating the observer changes not just which direction is "center" but the relative spacing of all perceived target angles in neural space. The two model classes handle this nonlinearity differently — which is the entire reason both exist.

## Two model classes

### `NeuralBandModel` (NBM)

The current/preferred model. γ lives in **egocentric/neural** space.

- The coupling kernel *and* the pull direction in `dγ/dt` both use neural (warped egocentric) angles.
- Walker torque: `dθ/dt = K·R·sin(Θ/2)` where `Θ = arg(γ)` is the **neural** consensus angle and `R = |γ|` — the **half-angle law in the neural angle directly**, no inverse-warp mapping. `Θ ∈ (−π, π]` is already wrapped before halving; neural angle 0 is straight ahead (the SC equilibrium), `±π` is facing-away. Default `K=2`. (Derivation + the 2026-06-02 move off the inverse-warp form: `.claude/rules/torque-and-stability.md`.)
- Self-consistent equilibria have γ = R + 0j (real positive): heading=consensus ⇒ egocentric consensus = 0 ⇒ Θ_neural = f(0) = 0 ⇒ γ = R + 0j.

Key methods:
- [`sc_equilib`](decision_model.py#L2250) — self-consistent equilibrium finder (heading = consensus); used for bifurcation diagrams in (x, y). Returns `(allocentric_angles, stability_booleans)`.
- [`gamma_equilib`](decision_model.py#L2181) — γ-equilibrium finder at a fixed observer heading. *Not* the self-consistent finder.
- [`run_dgamma_dt`](decision_model.py#L2360) — ODE to steady state.
- [`dtheta_dt`](decision_model.py#L2411) — heading dynamics.
- [`convert_gamma`](decision_model.py#L2339) — inverse neural mapping from γ to `(ego_angle, R)`. Introspection only — **no longer on the torque path** (dθ/dt uses `arg(γ)` directly).
- [`plot_walkers`](decision_model.py#L2979) — SDE walker simulation.
- [`plot_direction_mesh`](decision_model.py#L2549), [`plot_bifurcation_diagram`](decision_model.py#L2734).

### `IsingExtModel` (IEM)

Closest to the published PNAS paper. γ lives in **allocentric** space.

- When `ν ≠ 1`, the power transform applies only to the cosine-kernel coupling strength, not to the pull direction in `dγ/dt`. The exponential term uses physical (allocentric) angles for the pull direction.
- Walker torque: `dθ/dt = K·|γ|·sin(convert_angles(angle(γ) − θ)/2)`. **Half-angle law.** Both `angle(γ)` and `θ` are allocentric; the egocentric argument is *not* pre-wrapped, so `convert_angles` must wrap it to `(−π, π]` before halving (`sin(x/2)` is 4π-periodic — load-bearing). Default `K=2`.
- Self-consistent equilibria satisfy `dgamma_dt(γ, focal_angle=angle(γ)) = 0`.

Key methods: [`sc_equilib`](decision_model.py#L3567) (self-consistent finder, returns gammas), [`gamma_equilib`](decision_model.py#L3502) (fixed-heading γ-finder), [`run_dgamma_dt`](decision_model.py#L3337), [`dtheta_dt`](decision_model.py#L3390), [`plot_bifurcation_diagram`](decision_model.py#L3668), [`plot_direction_mesh`](decision_model.py#L3891), [`plot_walkers`](decision_model.py#L4057).

### `PerceptionModel` API — warp and weight are decoupled

`PerceptionModel` has **two independent roles**, set by two constructor args:

- **`neural_angle_dist`** (WARP): the distribution integrated CDF-like to map egocentric→neural angles. One of `{'cutoff','lin_cutoff','vonmises','symmetric_beta','reg_power','direct_power', None}`. `'lin_cutoff'` (the **default**) is the trapezoidal/piecewise-linear analog of `'cutoff'` — same support/plateau (`a`,`b`) and normalization, but with a closed-form integral *and* inverse (no spline); it agrees with `'cutoff'` exactly at `0,±a,±b` and differs only on the ramp interior (bifurcation structure differs by <1.1% of cells, boundary-pixel jitter only). `'direct_power'` is the power angle map `f(θ)=π·sign(θ)(|θ|/π)^c` (NOT a CDF-integral); `None` is identity (no warp).
- **`angle_weight`** (WEIGHT): the density integrated over each target's visible arc to set ρ. One of `{'cutoff','lin_cutoff','vonmises','symmetric_beta','reg_power','neural_angle_dist', None}`. `'neural_angle_dist'` ties the weight to the warp (old full-weighting behavior); `None` (the **default**) is uniform weight. `'direct_power'` is **disallowed** as a weight.

Change parameters post-init by **assigning** the generic two-slot properties `a_warp`/`b_warp`/`a_weight`/`b_weight` (`pm.a_warp = 0.55`, etc.) — this rebuilds only the affected role's splines. `warp_params`/`weight_params` are **read-only** views. Full parameter detail (`_FAMILY_INFO` per-family mapping, Old→new migration table, the strict setter / permissive getter, diagnostics) is in `.claude/rules/perception-and-solver.md`.

**IEM must be used with `neural_angle_dist=None, angle_weight=None`.** In IEM γ lives in allocentric/physical coordinates; the model assumes the observer perceives target angles directly. A non-identity `neural_angle_dist` on IEM is a category error — its `dγ/dt` math doesn't apply, and `sc_equilib` then returns near-empty diagrams (model rejecting an invalid input, not a solver bug).

**NBM is the model for warped perception.** Foveal density, egocentric warping, any non-identity neural mapping → use NBM. The two are *not* substitutes under warping; they were designed to handle the warping nonlinearity in incompatible ways.

## Self-consistent equilibria

At a physical turning equilibrium the observer has stopped turning, so heading = allocentric consensus direction ⇒ egocentric consensus = 0 ⇒ (for any reasonable warp) Θ_neural = 0 ⇒ γ = R + 0j. The allocentric consensus direction *is* `θ` (the heading) at a self-consistent equilibrium — no inverse mapping needed to recover the physical direction. `NBM.sc_equilib(focal_loc=..., stability_criterion=...)` returns `(allocentric_angles, stability_booleans)`; `gamma_equilib` is a *separate*, fixed-heading γ-finder.

For NBM the SC turning equilibria are where `dθ/dt = K·R·sin(Θ/2) = 0` with `Θ = arg(γ)`: the smooth zero is `Θ = 0` (heading = consensus, stable); `Θ = ±π` (facing directly *away*) is an intentional branch-cut fork, not a smooth equilibrium. (The three design approaches considered — and why egocentric-warp + heading-dependent SC equilibria was Option 3 — are in `.claude/rules/torque-and-stability.md`.)

## Stability criterion

Three criteria via `stability_criterion=`, all built from the same numerically-differenced 3×3 Jacobian on `(γ_re, γ_im, θ)` ([`_coupled_jacobian`](decision_model.py#L2759)):

- **`'reduced'` (DEFAULT, since 2026-06-08)** — the timescale-separated test, consistent with γ slaved to equilibrium everywhere else in the model. Stable iff the fast γ block `A` is Hurwitz **and** the slow Schur complement `λ_slow < 0` (evaluated as `sign(det J)` via the block-determinant identity `det J = det A · λ_slow`, which stays well-conditioned at a γ-fold). This is the right default because the simulated walker *is* the slaved system.
- **`'coupled'`** — the full 3×3 eigenvalue test; additionally flags coupled γ–θ **Hopf / limit-cycle** instabilities the slaved walker never realizes. Use when studying the continuous coupled ODE rather than the walker.
- **`'discrim_a'`** — legacy analytic γ-only discriminant (fast block alone); **over-counts** stable equilibria (misses the slow heading-tracking instability). Kept for comparison plots.

`sc_equilib`, `gamma_equilib`, `plot_direction_mesh`, `plot_bifurcation_diagram` (and the count helpers) accept all three; NBM and IEM defaults are now `'reduced'`. SC equilibria (γ = R+0j) are *exactly* equilibria of the full 3-eq system. Derivations, the why-reduced consistency argument, the worked over-counting example at (1.5, 0), and the half-angle/`K=2` invariance proofs are in `.claude/rules/torque-and-stability.md`.

## Geometry: targets

Three target geometries: `circle`, `delta` (point), and `capsule`.

**Capsule** (line-segment spine + semicircular endcaps of radius `w/2`) replaced an earlier buggy `segment` geometry whose flaw was zero angular extent viewed end-on. Capsule gives nonzero extent for any `w > 0`, uses distance-to-spine ≤ `w/2` for overlap, and degenerates to a circle when `l=0`.

**Known limitation — capsule blocking approximation:** blocking order uses closest-point distance sorting, approximate for capsules that mutually occlude at different angles. A fully correct solution needs per-angle depth comparison. Acceptable for current use; flag if pathological cases come up.

## Bifurcation diagrams — conventions

[`plot_bifurcation_diagram`](decision_model.py#L2734) (NBM) / [`plot_bifurcation_diagram`](decision_model.py#L3668) (IEM) render `(x, y)` sweeps colored by stable-equilibrium count.

- **Default colormap is viridis keyed on stable count alone**, with a `max_count` kwarg.
- **Don't reintroduce two-axis `(n_stable, n_unstable)` color coding** without an explicit ask — it was tried and made boundaries harder to read. (The `stability_criterion='coupled'` plumbing it introduced is preserved — a real correctness fix.)
- **`boundary_dilation` kwarg** (default 1) widens each refinement pass by promoting cells sharing a corner with a disagreement cell — addresses stair-step artifacts and partially helps thin features.
- **Cell-center sampling deferred** — propose it (rather than further enlarging dilation) only if thin-feature artifacts persist after `boundary_dilation` + modest `num_x`/`num_y` increases.

## Physical phenomena

(Detail + reproducibility in the auto-loading exploration rule / linked docs.)

- **0-stable "decision paralysis" band under power warping** (`c=0.5`): a genuine pitchfork-like bifurcation, not a numerical artifact — power warping stretches physical angles to wide neural angles, destabilizing the transverse mode, so no stable consensus exists. → `.claude/rules/bifurcation-explorations.md`.
- **Hopf island with stable limit cycle** (vonmises k=0.55, two-target, near (2.1, ±2.45)): a *coupled-system* phenomenon (head-bobbing limit cycle needs γ to lag θ) — seen by `'coupled'`, reported 0-stable by the default `'reduced'` (and the walker, which at most does a small γ-bistability relaxation oscillation). → [VM_bifurcation_old_dtheta/VERDICT.md](VM_bifurcation_old_dtheta/VERDICT.md) (old law), [reduced_criterion/README.md](reduced_criterion/README.md) (current law), `.claude/rules/bifurcation-explorations.md`.
- **Weighting vs warping — the "ears":** warping alone reproduces full-weighting bifurcation structure *except* two "ears" of extra far-target bistability behind two circle targets — present under non-uniform `angle_weight`, absent under uniform (the default), hence now opt-in. → [weighting_analysis/README.md](weighting_analysis/README.md).

## Basin-of-attraction estimator (vetted, not yet wired in)

End-to-end vetting of the basin-estimation machinery lives in [basin_estimation/](basin_estimation/) ([findings.md](basin_estimation/findings.md), [free_energy_derivation.md](basin_estimation/free_energy_derivation.md)); the eleven-step plan is complete and all test scripts pass. Public entry point `compute_basins_at_focal_loc(focal_loc, *, scan_single_stable=False)` (`basin_estimation/basin_via_theta.py`) returns `basins`/`stable_count`/`unstable_count`/`sentinel`. Both prerequisite modeling changes (sin(Θ*/2) dθ/dt with K=2; warp/weight decouple) are DONE and Steps 5–9 re-vetted invariant. Load-bearing facts (ΔF_γ robustness only in multistable cells; four basin-boundary kinds incl. the branch-cut; ~4.2 min/41×41 cost) and the basin-width / fold-vs-decision-boundary gotcha are in `.claude/rules/basin-estimation.md`; remaining integration work is in [TODO.md](TODO.md).

## Open TODOs

Engineering TODOs (IEM `run_dgamma_dt` LSODA port; cell-center sampling; residual heading-noise floor; foveal `angle_weight` commitment signal; two-panel bifurcation+basin plot) now live in **[TODO.md](TODO.md)**.

## Common gotchas

- When modifying `sc_equilib`, `gamma_equilib`, or `_get_target_signals`, **preserve exact interval arithmetic**. Mesh-based fallback paths exist for plotting only.
- When changing warp or weight parameters, **assign the `a_warp`/`b_warp`/`a_weight`/`b_weight` properties** — they trigger the correct (and only the affected) spline rebuild. `warp_params`/`weight_params` are read-only views; item assignment and rebinding both raise.
- When modifying `run_dgamma_dt`, **preserve the real-valued reformulation** for LSODA compatibility.
- When discussing model differences or debugging warping issues, **always check which coordinate system each quantity lives in**. Most subtle bugs trace back to a coordinate-frame mismatch.

## Detailed references (auto-loaded rules + pull-only docs)

Path-scoped rules under `.claude/rules/` load **only** when you open a matching file (zero cost otherwise). Pull-only docs load only when Read.

| Topic | Auto-loads when you open… | Full detail |
|---|---|---|
| PerceptionModel deep API, interval arithmetic, integral splines, `sc_equilib` solvers, `run_dgamma_dt` LSODA, `R<0.01` filter | `decision_model.py` | `.claude/rules/perception-and-solver.md` |
| Walker `plot_walkers`: state-gated noise law, `walk_std` blind-spot search, target detection, `R_exp`, loss mechanisms | `decision_model.py` | `.claude/rules/walker-dynamics.md` |
| dθ/dt half-angle torque + `K=2` derivation, the three stability criteria (full), coupled-Jacobian proofs, SC-design history | `decision_model.py` | `.claude/rules/torque-and-stability.md` |
| Bifurcation-script roles + physical phenomena (decision paralysis, Hopf island) | `VM_bifurcation_old_dtheta/`, `bifurc_plots/`, `reduced_criterion/` | `.claude/rules/bifurcation-explorations.md`; [VERDICT.md](VM_bifurcation_old_dtheta/VERDICT.md), [reduced_criterion/README.md](reduced_criterion/README.md) |
| Basin estimator facts + basin-width gotcha | `basin_estimation/` | `.claude/rules/basin-estimation.md`; [findings.md](basin_estimation/findings.md), [free_energy_derivation.md](basin_estimation/free_energy_derivation.md) |
| Weighting-vs-warping "ears" | `weighting_analysis/` | `.claude/rules/weighting.md`; [weighting_analysis/README.md](weighting_analysis/README.md) |
| Engineering TODOs | — | [TODO.md](TODO.md) |
| Archived research-direction notes | — | [TODO_Jan_2026.md](TODO_Jan_2026.md) |
