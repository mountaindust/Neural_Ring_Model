# Neural Ring Model — Project Guide

Mathematical model of collective decision-making based on Ising-type dynamics on a neural ring. This file is the durable project context loaded into every Claude Code session in this repo — theory, model architecture, and known limitations that are hard to re-derive from code or git history. **Deep solver / walker / stability internals now live in path-scoped `.claude/rules/*.md` that auto-load only when you open the matching code** — see the reference index at the bottom for what lives where.

## ⚠️ Git policy (do not deviate)

**Never `git commit` automatically, and never `git push` automatically.** Commit only when the user explicitly asks for a commit, each time — a prior commit request does not authorize future ones. **A request to commit is NOT a request to push:** push only when the user explicitly asks to push, separately. When in doubt, stage/show the diff and ask.

## Codebase layout

- [decision_model.py](decision_model.py) — the project's research code (~4200 lines). Two model classes (`NeuralBandModel`, `IsingExtModel`), a `PerceptionModel`, and a `Targets` helper.
- Jupyter notebooks (`compare_sc_vm.ipynb`, `compare_sc_beta.ipynb`, `neural_band.ipynb`, `neural_band_walker.ipynb`, `ising_workbook.ipynb`, `debug_all_unstable.ipynb`) — testing, exploration, and visualization.
- [VM_bifurcation_old_dtheta/](VM_bifurcation_old_dtheta/), [bifurc_plots/](bifurc_plots/) — bifurcation/stability exploration scripts + write-ups. Script roles + the physical-phenomena findings are in the auto-loading `.claude/rules/bifurcation-explorations.md`. Not publication-quality — sized for fast iteration on a many-core machine.
- [stale_coupled_model_starting_code/](stale_coupled_model_starting_code/) — **stale.** Scripts written against the removed `'coupled'` criterion. Kept only as starting code for an eventual `(n⃗, θ)` population-level comparison; see its README.
- [theory/](theory/) — durable theory: the basin-of-attraction findings catalogue ([basins_of_attraction.md](theory/basins_of_attraction.md)), the F̂ free-energy derivation ([free_energy_derivation.md](theory/free_energy_derivation.md)), the block-determinant identity behind `'reduced'` ([block_determinant_identity.tex](theory/block_determinant_identity.tex)), a Lyapunov/Langevin/Kramers background tutorial ([theory_background.md](theory/theory_background.md)), and [reduced_dynamics_anatomy.py](theory/reduced_dynamics_anatomy.py) — derives the γ-bistability relaxation-oscillation period by branch integration (the paper figure for the same phenomenon is `plots/stability_comparison_figure.py`). The basin *visualization* is implemented in [decision_model.py](decision_model.py) (`NBM.plot_bifurcation_diagram(overlay_basins=True)`); see "Basin-of-attraction overlay" below.
- [weighting_analysis/](weighting_analysis/) — warp-vs-weight "ears" analysis ([README.md](weighting_analysis/README.md)) and the anti-foveal outward-bias study ([outward_bias.md](weighting_analysis/outward_bias.md)). Also the home of `anti_foveal.py`, which preserves two weight families that were tried and then removed from the model — nothing outside this directory should import it.
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
- **Neural coupling is `beta`** (default 10): `dγ/dt = Σⱼ ρⱼ e^{iθ̂ⱼ}·σ(2·β·R·cos(θ̂ⱼ−Θ)) − γ`. See "Neural temperature" below.
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

## Neural temperature: `beta` (NBM)

`NeuralBandModel(beta=10)` is the **Boltzmann factor of the Glauber dynamics**, `β = 𝓔/(k_B·temp)`, where `𝓔` is the energy scale of the Ising Hamiltonian, `k_B` the Boltzmann constant, and `temp` the temperature in Kelvin. Large β is cold (sharp commitment), small β is hot (diffuse). It appears in `dγ/dt` as the logistic argument `2·β·R·cos(θ̂ⱼ−Θ)` and in the analytic free-energy Hessian of `_discrim_A` as `w_j = (β/2)·ρⱼ·sech²(β·R·cos(θ̂ⱼ−Θ))`.

**β is a property of the neural ring, not of the scene** — it does not scale with the number of targets. The Hamiltonian is normalized `𝓔/N` over the N *neurons*; an extra factor of the target count would make the per-neuron energy scale depend on how many targets happen to be in view, which is not physical.

**Reading pre-β scripts and write-ups.** The earlier parameterization used a temperature `T` with the target count folded into the coupling, giving an effective `β = N/T` where `N` was the number of **currently visible** targets (`neur_angles.size`). To reproduce an old result, set `β = N_total/T` — exact wherever every target contributes to perception (partial occlusion is fine; only a target contributing *nothing*, `G == 0`, shrinks the old `N`). Under a restricted weight cone the two genuinely differ: in the 3-target foveal setups a target is fully out of view at ~34–48% of (position, heading) states, and there the old law's effective β dropped to `2/T` or lower while the new one holds. Scripts in `plots/`, `walker_analysis/`, `weighting_analysis/` carry their converted `BETA` with the arithmetic in a comment; note that a 2-target and a 3-target scene at the same old `T` need **different** β (e.g. `two_target_fly_refine.py` β=20 vs `three_target_fly_refine.py` β=30).

**IEM still uses `T`** and still carries the visible-target factor (`2·N·R·cos(...)/T`); `IsingExtModel.T` and `NeuralBandModel.beta` are not the same quantity.

## Self-consistent equilibria

At a physical turning equilibrium the observer has stopped turning, so heading = allocentric consensus direction ⇒ egocentric consensus = 0 ⇒ (for any reasonable warp) Θ_neural = 0 ⇒ γ = R + 0j. The allocentric consensus direction *is* `θ` (the heading) at a self-consistent equilibrium — no inverse mapping needed to recover the physical direction. `NBM.sc_equilib(focal_loc=..., stability_criterion=...)` returns `(allocentric_angles, stability_booleans)`; `gamma_equilib` is a *separate*, fixed-heading γ-finder.

For NBM the SC turning equilibria are where `dθ/dt = K·R·sin(Θ/2) = 0` with `Θ = arg(γ)`: the smooth zero is `Θ = 0` (heading = consensus, stable); `Θ = ±π` (facing directly *away*) is an intentional branch-cut fork, not a smooth equilibrium. (The three design approaches considered — and why egocentric-warp + heading-dependent SC equilibria was Option 3 — are in `.claude/rules/torque-and-stability.md`.)

## Stability criterion

Two criteria via `stability_criterion=`, both built from the same numerically-differenced 3×3 Jacobian on `(γ_re, γ_im, θ)` (`_coupled_jacobian`):

- **`'reduced'` (DEFAULT, since 2026-06-08)** — the timescale-separated test, consistent with γ slaved to equilibrium everywhere else in the model. Stable iff the fast γ block `A` is Hurwitz **and** the slow Schur complement `λ_slow < 0` (evaluated as `sign(det J)` via the block-determinant identity `det J = det A · λ_slow`, which stays well-conditioned at a γ-fold). This is the right default because the simulated walker *is* the slaved system.
- **`'discrim_a'`** — the fast γ-block test alone (A-Hurwitz ⇔ free-energy Hessian positive-definite: transverse `A<1` **and** `det>0`); **over-counts** stable equilibria by exactly the slow heading-tracking mode it omits. Kept for comparison plots. (Until 2026-07 this checked only the transverse `A<1`, so it *additionally* over-counted on the fast layer — radial folds + off-diagonal saddles; `_discrim_A` now adds the analytic `det>0` and `_discrim_A_nu` takes the block from `_coupled_jacobian`. See `theory/free_energy_derivation.md` §6.1.)

`sc_equilib`, `gamma_equilib`, `plot_direction_mesh`, `plot_bifurcation_diagram` (and the count helpers) accept both; NBM and IEM defaults are `'reduced'`. SC equilibria (γ = R+0j) are *exactly* equilibria of the full 3-eq system. Derivations, the why-reduced consistency argument, the worked over-counting example at (1.5, 0), and the half-angle/`K=2` invariance proofs are in `.claude/rules/torque-and-stability.md`.

**There is deliberately no fully-coupled criterion (`'coupled'` removed 2026-08-19).** `dγ/dt` is not an equation of motion: it is the rank-2 readout of the K-dimensional Glauber population dynamics (one `n_k` per visible target; see the project preprint's Glauber section), obtained by differentiating `γ = Σₖ nₖ e^{iθ̂ₖ}` and keeping **only** the `dnₖ/dt` term. The dropped term `−i·θ̇·Σₖ nₖ·U′(θₖ)·e^{iθ̂ₖ}` vanishes identically at `θ̇ = 0`, so **equilibria, `'reduced'` and `'discrim_a'` are all unaffected** — but off equilibrium the γ-ODE is only valid under the timescale separation. Taking the full 3×3 eigenvalues therefore linearizes an incomplete equation, and it cannot be patched at the γ level (under a warp the dropped term is not proportional to γ, so it is not a function of γ at all). A genuine coupled analysis needs the `(n₁…n_K, θ)` system **and** an explicit neural timescale `τ₀`. See `NeuralBandModel._discrim_reduced` for the full note and the measured evidence.

## Geometry: targets

Three target geometries: `circle`, `delta` (point), and `capsule`.

**Capsule** (line-segment spine + semicircular endcaps of radius `w/2`) replaced an earlier buggy `segment` geometry whose flaw was zero angular extent viewed end-on. Capsule gives nonzero extent for any `w > 0`, uses distance-to-spine ≤ `w/2` for overlap, and degenerates to a circle when `l=0`.

**Known limitation — capsule blocking approximation:** blocking order uses closest-point distance sorting, approximate for capsules that mutually occlude at different angles. A fully correct solution needs per-angle depth comparison. Acceptable for current use; flag if pathological cases come up.

## Bifurcation diagrams — conventions

[`plot_bifurcation_diagram`](decision_model.py#L2734) (NBM) / [`plot_bifurcation_diagram`](decision_model.py#L3668) (IEM) render `(x, y)` sweeps colored by stable-equilibrium count.

- **Default colormap is viridis keyed on stable count alone**, with a `max_count` kwarg.
- **Don't reintroduce two-axis `(n_stable, n_unstable)` color coding** without an explicit ask — it was tried and made boundaries harder to read. (The `stability_criterion=` plumbing it introduced is preserved — a real correctness fix.)
- **`boundary_dilation` kwarg** (default 1) widens each refinement pass by promoting cells sharing a corner with a disagreement cell — addresses stair-step artifacts and partially helps thin features.
- **Cell-center sampling deferred** — propose it (rather than further enlarging dilation) only if thin-feature artifacts persist after `boundary_dilation` + modest `num_x`/`num_y` increases.

## Physical phenomena

(Detail + reproducibility in the auto-loading exploration rule / linked docs.)

- **0-stable "decision paralysis" band under power warping** (`c=0.5`): a genuine pitchfork-like bifurcation, not a numerical artifact — power warping stretches physical angles to wide neural angles, destabilizing the transverse mode, so no stable consensus exists. → `.claude/rules/bifurcation-explorations.md`.
- **No γ–θ Hopf / head-bobbing** (vonmises k=0.55, two-target, near (2.1, ±2.45)): the "Hopf island with stable limit cycle" reported here through 2026-08 was an **artifact of the removed `'coupled'` criterion**, which took the full eigenvalues of the γ-level `(γ_re, γ_im, θ)` Jacobian — an incomplete equation off `θ̇ = 0`. The exact `(n⃗, θ)` population system is stable at every equilibrium the γ-3×3 flagged (6 → 0 under the current law, 7 → 0 under both older torque laws). **Don't re-derive a coupled γ-ODE and rediscover it**; a genuine coupled analysis needs the population system plus an explicit `τ₀` (see "Stability criterion" above). The cells are still 0-stable under `'reduced'` — the γ-block has a positive eigenvalue there — and the slaved walker does a small γ-bistability *relaxation* oscillation, not head-bobbing. → [stale_coupled_model_starting_code/README.md](stale_coupled_model_starting_code/README.md).
- **Weighting vs warping — the "ears":** warping alone reproduces full-weighting bifurcation structure *except* two "ears" of extra far-target bistability behind two circle targets — present under non-uniform `angle_weight`, absent under uniform (the default), hence now opt-in. → [weighting_analysis/README.md](weighting_analysis/README.md).
- **An anti-foveal (centre-dip) `angle_weight` cannot bias the observer outward** — the mechanism the locust 3-target split (29% centre vs the fly's 45%) would need. A weight is a function of *egocentric* angle, so a dip at ego 0 suppresses **whatever the observer currently faces**, penalizing every single-target commitment equally; since the outer-target branches are the fragile ones (marginal Ising saddle-nodes) they die first, annihilating entirely below a floor of ~0.5. Walker census goes to **100% centre** (data 29%, shipped foveal 65%). Second, independent penalty: spreading ρ lowers `R`, and the gated walker drift `K·R^{R_exp}` collapses with it, so the ensemble gets *narrower*, not wider. **Concentrate the weight, don't spread it** — the same direction the foveal commitment-signal question wants. The locust gap therefore stays in the **recapture mechanism**, not the perception weighting. The two families written to test this (`lin_dip`, `lin_ring`) were **removed from `decision_model.py`** afterwards and preserved, with their tests, in [weighting_analysis/anti_foveal.py](weighting_analysis/anti_foveal.py). → [weighting_analysis/outward_bias.md](weighting_analysis/outward_bias.md).

## Basin-of-attraction overlay (vetted and implemented)

The basin-estimation effort is complete and the visualization is **wired into the model**: `NBM.plot_bifurcation_diagram(overlay_basins=True)` draws a single-panel basin-**wheel** overlay (per-region θ-basin annulus + direction arrows, categorical color by basin rank), computed from a fixed neutral seed (`basin_R_seed=0.15`, neural arg=0) under the slaved flow with destination-flip bisection. The standalone `basin_estimation/` vetting module has been retired (its 11-step log is in git history). Both prerequisite modeling changes (sin(Θ*/2) dθ/dt with K=2; warp/weight decouple) are DONE; the γ-side findings are turning-law-invariant. Durable findings + derivations now live in [theory/](theory/): [basins_of_attraction.md](theory/basins_of_attraction.md) (the four basin-boundary kinds incl. the branch-cut; ΔF_γ robustness, multistable-only; the scan-fold-vs-decision-boundary gotcha; ~4.2 min/41×41 cost; **what the code does now**) and [free_energy_derivation.md](theory/free_energy_derivation.md) (the F̂ derivation). The two code-editing gotchas are in `.claude/rules/basin-estimation.md`; remaining work (committed-walker basin, fragility glyph, θ-noise escape-rate) is in [TODO.md](TODO.md).

## Open TODOs

Engineering TODOs (IEM `run_dgamma_dt` LSODA port; cell-center sampling; residual heading-noise floor; foveal `angle_weight` commitment signal; basin-overlay refinement + deferred basin work) now live in **[TODO.md](TODO.md)**.

## Common gotchas

- When modifying `sc_equilib`, `gamma_equilib`, or `_get_target_signals`, **preserve exact interval arithmetic**. Mesh-based fallback paths exist for plotting only.
- **A target extent that straddles ±π comes back from `get_percep_angles` as a *wrapping* pair (`lo > hi`) — unwrap it before integrating.** `_integrate_neural_weight` (and the `mesh_signal` masking) require non-wrapping pieces; a raw wrapping pair gives a *negative* arc length, which the `G > 0` visibility filter then silently discards, dropping the target from perception entirely. `_subtract_intervals_circle` unwraps its own inputs, so only paths that bypass it need `_unwrap_interval` explicitly — which is how the **closest** target (no closer blocker, so it never enters that loop) went missing for the whole angular window in which it straddled the rear branch cut. Fixed 2026-08-04; regression-tested in [tests/test_intervals.py](tests/test_intervals.py). Weights whose support ends before the rear (`cutoff`/`lin_cutoff`) mask it except within `d < r/sin(π−b)` of a target **centre** (≈0.85 for `r=0.5, b=4π/5`; ≈1.62 for `b=0.9π`); **uniform weight is exposed at every distance**.
- When changing warp or weight parameters, **assign the `a_warp`/`b_warp`/`a_weight`/`b_weight` properties** — they trigger the correct (and only the affected) spline rebuild. `warp_params`/`weight_params` are read-only views; item assignment and rebinding both raise.
- When modifying `run_dgamma_dt`, **preserve the real-valued reformulation** for LSODA compatibility.
- When discussing model differences or debugging warping issues, **always check which coordinate system each quantity lives in**. Most subtle bugs trace back to a coordinate-frame mismatch.

## Detailed references (auto-loaded rules + pull-only docs)

Path-scoped rules under `.claude/rules/` load **only** when you open a matching file (zero cost otherwise). Pull-only docs load only when Read.

| Topic | Auto-loads when you open… | Full detail |
|---|---|---|
| PerceptionModel deep API, interval arithmetic, integral splines, `sc_equilib` solvers, `run_dgamma_dt` LSODA, `R<0.01` filter | `decision_model.py` | `.claude/rules/perception-and-solver.md` |
| Walker `plot_walkers`: state-gated noise law, `walk_std` blind-spot search, target detection, `R_exp`, loss mechanisms | `decision_model.py` | `.claude/rules/walker-dynamics.md` |
| dθ/dt half-angle torque + `K=2` derivation, the two stability criteria (full), coupled-Jacobian proofs, SC-design history | `decision_model.py` | `.claude/rules/torque-and-stability.md` |
| Bifurcation-script roles + physical phenomena (decision paralysis) | `VM_bifurcation_old_dtheta/`, `bifurc_plots/` | `.claude/rules/bifurcation-explorations.md`; [VERDICT.md](VM_bifurcation_old_dtheta/VERDICT.md) |
| Basin overlay code-editing gotchas (basin-width; multistable-only robustness; neutral-seed protocol) | `decision_model.py` | `.claude/rules/basin-estimation.md` |
| Basin findings, F̂ derivation, block-determinant identity, Lyapunov/Langevin/Kramers background (pull-only) | — | [theory/basins_of_attraction.md](theory/basins_of_attraction.md), [theory/free_energy_derivation.md](theory/free_energy_derivation.md), [theory/block_determinant_identity.tex](theory/block_determinant_identity.tex), [theory/theory_background.md](theory/theory_background.md) |
| Weighting-vs-warping "ears" | `weighting_analysis/` | `.claude/rules/weighting.md`; [weighting_analysis/README.md](weighting_analysis/README.md) |
| Engineering TODOs | — | [TODO.md](TODO.md) |
| Archived research-direction notes | — | [TODO_Jan_2026.md](TODO_Jan_2026.md) |
