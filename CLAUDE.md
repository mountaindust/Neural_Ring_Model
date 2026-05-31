# Neural Ring Model — Project Guide

Mathematical model of collective decision-making based on Ising-type dynamics on a neural ring. This file is the durable project context loaded into every Claude Code session in this repo. It captures theory, model architecture, solver design, and known limitations — the things that are hard to re-derive from code or git history alone.

## Codebase layout

- [decision_model.py](decision_model.py) — the project's research code (~4200 lines). Two model classes (`NeuralBandModel`, `IsingExtModel`), a `PerceptionModel`, and a `Targets` helper.
- Jupyter notebooks (`compare_sc_vm.ipynb`, `compare_sc_beta.ipynb`, `neural_band.ipynb`, `neural_band_walker.ipynb`, `ising_workbook.ipynb`, `debug_all_unstable.ipynb`) — testing, exploration, and visualization.
- [VM_bifurcations/](VM_bifurcations/) — diagnostic scripts and the [VERDICT.md](VM_bifurcations/VERDICT.md) write-up of the Hopf-island / saddle-node bifurcation skeleton near (2.1, ±2.45) in the vonmises-k0.55 / two-target setup.
- [bifurc_plots/](bifurc_plots/) — exploratory parameter-sweep scripts. `neural_weight_sweep.py` (full weighting) and `neural_weight_sweep_angle_only.py` (warping-only) are companions; `bifurcation_compare_discrim_vs_coupled.py` compares stability criteria; `arc_skeleton_and_island_dynamics.py` is the upper-arc / island-dynamics combined figure. Not publication-quality — settings are sized for fast iteration on a many-core machine. More function-space exploration is needed before any of these can be locked in for publication.
- [basin_estimation/](basin_estimation/) — vetting work for the basin-of-attraction estimator that supports the two-panel bifurcation+basin plot TODO. Eleven-step vetting plan complete; public API ready but **not yet wired into `decision_model.py`** (deferred behind pending modeling changes — see "Basin-of-attraction estimator" section below). Running mathematical findings in [basin_estimation/findings.md](basin_estimation/findings.md).
- [tests/](tests/) — unit tests (e.g. `test_broad_validation.py`, `test_intervals.py`, `test_segments.py`).
- [Matlab/](Matlab/), [early_ideas/](early_ideas/) — legacy / prototype code, not part of the active model.
- [PARALLEL_CONFIG.md](PARALLEL_CONFIG.md), [parallel_config.py](parallel_config.py), [machine_config.py](machine_config.py) — per-machine worker-count settings for multiprocessing.

Mesh sweeps and bifurcation refinement use `multiprocessing` extensively (see PARALLEL_CONFIG.md for tuning).

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
- Walker torque: `dθ/dt = K·R·sin(ego_angle)` where `(ego_angle, R) = convert_gamma(γ)`. No subtraction because `convert_gamma` applies the inverse neural mapping to get an egocentric angle already relative to the current heading. Neural angle 0 always corresponds to egocentric angle 0 (straight ahead).
- Self-consistent equilibria have γ = R + 0j (real positive): heading=consensus ⇒ egocentric consensus = 0 ⇒ Θ_neural = f(0) = 0 ⇒ γ = R + 0j.

Key methods:
- [`sc_equilib`](decision_model.py#L2250) — self-consistent equilibrium finder (heading = consensus). Used for bifurcation diagrams in (x, y).
- [`gamma_equilib`](decision_model.py#L2181) — γ-equilibrium finder at a fixed observer heading. *Not* the self-consistent finder.
- [`run_dgamma_dt`](decision_model.py#L2360) — ODE to steady state.
- [`dtheta_dt`](decision_model.py#L2411) — heading dynamics.
- [`convert_gamma`](decision_model.py#L2339) — inverse neural mapping from γ to `(ego_angle, R)`.
- [`plot_walkers`](decision_model.py#L2979) — SDE walker simulation.
- [`plot_direction_mesh`](decision_model.py#L2549), [`plot_bifurcation_diagram`](decision_model.py#L2734).

### `IsingExtModel` (IEM)

Closest to the published PNAS paper. γ lives in **allocentric** space.

- When `ν ≠ 1`, the power transform applies only to the cosine-kernel coupling strength, not to the pull direction in `dγ/dt`. The exponential term uses physical (allocentric) angles for the pull direction.
- Walker torque: `dθ/dt = K·|γ|·sin(angle(γ) − θ)`. Both `angle(γ)` and `θ` are allocentric; subtraction is needed because both live in the same absolute frame.
- Self-consistent equilibria satisfy `dgamma_dt(γ, focal_angle=angle(γ)) = 0` where `angle(γ)` is the allocentric consensus direction.

Key methods: [`sc_equilib`](decision_model.py#L3567) (self-consistent finder, returns gammas; bifurcation diagrams), [`gamma_equilib`](decision_model.py#L3502) (fixed-heading γ-finder), [`run_dgamma_dt`](decision_model.py#L3337), [`dtheta_dt`](decision_model.py#L3390), [`plot_bifurcation_diagram`](decision_model.py#L3668), [`plot_direction_mesh`](decision_model.py#L3891), [`plot_walkers`](decision_model.py#L4057).

### Model usage constraint — read this before configuring `PerceptionModel`

**IEM must be used with `neural_weight=None, neural_angle=None`.** In IEM, γ lives in allocentric/physical coordinates; the model assumes the observer perceives target angles directly without neural warping. Running IEM with `neural_angle='integral'` (or any non-identity warp) is a category error — the math behind IEM's `dγ/dt` doesn't apply.

**NBM is the model for warped perception.** If you need foveal density, egocentric warping, or any non-identity neural mapping, use NBM. The two are *not* substitutes for each other under warping; they were designed to handle the warping nonlinearity in incompatible ways.

**Diagnostic note:** if an IEM bifurcation diagram or direction mesh comes back almost entirely zero/empty, the most likely cause is accidentally driving IEM with `neural_angle='integral'`. IEM's polar-init multistart in `sc_equilib` doesn't find roots under that (invalid) configuration. This is *not* a solver bug — it's the model rejecting an invalid input. Switch to `neural_angle=None` to confirm.

## Self-consistent equilibria

At a physical turning equilibrium, the observer has stopped turning, so heading = allocentric consensus direction. This makes egocentric consensus = 0, which (for any reasonable warp) makes Θ_neural = 0, which makes γ = R + 0j.

Three approaches were considered for handling neural warping with self-consistency; **Option 3 was chosen:**

1. ~~Make warping allocentric.~~ Rejected: experimental evidence shows the denser neural band near the center of the visual field is intrinsically egocentric. The warping has to be egocentric.
2. ~~Hybrid with mismatch from Hamiltonian.~~ Rejected: suboptimal, no good error estimates.
3. **Accept heading-dependent equilibria and find the self-consistent ones.** Adopted because we already accept that consensus direction should depend on where the observer is looking; the self-consistent solution is the physically meaningful subset.

The allocentric consensus direction *is* `θ` (the heading) at a self-consistent equilibrium — no inverse mapping needed to recover the physical direction. `NBM.sc_equilib(focal_loc=..., stability_criterion=...)` returns `(allocentric_angles, stability_booleans)`. (`NBM.gamma_equilib` is a separate method that finds γ-eqs at a fixed observer heading; it is *not* the self-consistent finder.)

For NBM, the mathematical proof that only `θ = n·π` self-consistent equilibria exist for turning: `sin(power_inverse(Θ)) = 0` requires `Θ = n·π`. Only `n=0` is stable for turning; `n=±1` corresponds to facing directly *away* from consensus.

## Solver architecture

### Perception: exact interval arithmetic for blocking

[`PerceptionModel._get_target_signals`](decision_model.py#L1650) uses exact interval arithmetic, not a mesh discretization:

- [`_subtract_intervals_circle`](decision_model.py#L1457) computes visible angular intervals after blocking by closer targets.
- [`_integrate_neural_weight`](decision_model.py#L1497) integrates neural weight (cutoff or vonmises) over those intervals analytically.

The original implementation Riemann-summed over a discrete θ-mesh and produced equilibrium residuals of ~1e-3 — not roundoff but genuine discretization error that caused convergence failures in `sc_equilib`. Switching to interval arithmetic dropped residuals to machine precision (~1e-14) and gave a 4.5× speedup for circle targets.

The `full_signal` parameter was renamed to `mesh_signal` (the mesh path is still used by `plot_blocked_signals`). The `G.sum()==0` case returns empty arrays instead of NaN division.

### Integral antiderivatives precomputed as splines

[`PerceptionModel._build_integral_splines`](decision_model.py#L615) tabulates forward + inverse `CubicSpline`s at 2001 nodes for both `'cutoff'` (F(θ; a, b)) and `'vonmises'` (G(θ; k)) weights. Built once at `__init__`; the `a`, `b`, `k` attributes are properties whose setters trigger a rebuild.

- **Accuracy:** forward direction matches the reference `quad`/`cdf` to ~5e-11 everywhere; end-to-end `_get_target_signals` ρ values match the reference path to ~1e-16 (machine precision).
- **Inverse direction is condition-limited to ~1e-8 near `y = ±π`** because `dF/dx → 0` at the boundary. This only affects `get_neural_angle_inverse` via `convert_gamma(γ)` with `np.angle(γ)`; 1e-8 error in `ego_angle` is negligible for walker dynamics and the walker rotates out of the poorly-conditioned region whenever it matters.
- **Performance:** circle/cutoff `test_broad_validation` went from 91.8s → 34.8s (~2.6×). Per-point cost for circle targets now comparable to delta targets (~7ms vs ~15ms).
- **Cutoff spline construction (non-obvious):** `F(x)` saturates to ±π in floating point once `b − |x| < ~0.05` (the `exp(−norm/(b−x))` tail underflows). Naïve `CubicSpline` fails the strict-monotonicity requirement. `_build_integral_splines` uses a greedy monotone filter to drop saturated boundary nodes while preserving exact ±π endpoints.
- **Domain restriction:** inverse splines raise `ValueError` on `y` outside `[−π, π]`; forward splines saturate safely. Callers are domain-clean by construction.
- **Reference kernels retained for testing:** `_smooth_cutoff_integral` and `_smooth_cutoff_int_inverse` (static methods) are still used by tests to validate the splines against `quad`/`brentq`. `scipy.stats.vonmises.cdf/ppf` are the vonmises reference.

### NBM `sc_equilib`: single-pass solver

Simplified from an earlier two-pass `brentq + multistart` to a single-pass strategy:

1. Scan `Im(dgamma_dt)` across a 100-point θ mesh at `R_probe = 0.5`.
2. Find sign changes → `brentq` for precise `θ_c`.
3. Add `θ = 0, ±π` as explicit candidates.
4. Polish each with 2D `hybr` (`tol=1e-10`), require `sol.success`.
5. Residual threshold **1e-4**. The `hybr+logistic` combination can produce residuals up to ~2e-5 due to exponential amplification; a tighter 1e-6 threshold was silently dropping ~10% of valid equilibria and creating apparent holes in direction meshes.
6. Deduplicate with both circular angle distance < 0.02 **and** R distance < 0.01. Both axes are required: near a saddle-node bifurcation, two genuine equilibria of opposite stability can share θ to within ~1e-3 rad while differing in R by ~0.02, so θ-only dedup silently discards one of the pair. Which one survives depends on the brentq sign-change scan order, which flips under coordinate symmetries — producing visible chirality (anti-symmetric "1-stable invading 2-stable" intrusions) in bifurcation diagrams that should be y-symmetric. `IEM.sc_equilib` was always correct here because it dedups by full complex `|γ_eq − existing_γ|`; NBM regressed in the cf6af66 self-consistent rewrite when the kept-list became θ-only (γ = R+0j made θ feel like the natural identifier). The broad-validation grid in [tests/test_broad_validation.py](tests/test_broad_validation.py) doesn't reach near-SN configurations, so this kind of regression won't show up there — y-symmetry of `plot_bifurcation_diagram` output on a symmetric target setup is the real diagnostic.

**Residual asymmetry (known):** even with full (θ, R) dedup, ~25–30% of the y-flip pixel asymmetry persists in `weight_angle_only=True` cutoff/beta/vonmises sweeps. Traced to `scipy.optimize.root(method='hybr')` itself: `_self_consistent_eq` is y-flip symmetric to 1e-20, but hybr's internal Jacobian estimation uses positive forward-difference steps, so the trajectory from a starting point is not the mirror of its trajectory from the sign-flipped starting point. Two scaffolds for fixing this if it ever matters: symmetrize the multistart by also trying mirrored starts for every candidate, or densify the brentq candidate seeding so both members of a near-SN pair are reachable from independent starts.

### IEM `sc_equilib`: multistart polar-init

[`IEM.sc_equilib`](decision_model.py#L3567) uses multistart root finding seeded on a polar grid at radius 0.5 around the unit circle. This is the strategy appropriate for the *allocentric, unwarped* problem IEM is designed for. After the interval-arithmetic refactor, the smoother ρ landscape allowed the root finder to converge to spurious boundary solutions (e.g. `R=1.0` with self-consistent residual ~0.5) that mesh noise had previously masked.

**Fix in place:** after finding `γ_eq`, re-evaluate `dgamma_dt` with `focal_angle=angle(γ_eq)` and reject if `|residual| > 1e-6`. This is the correct self-consistency criterion: at equilibrium, the observer faces its consensus direction.

### `run_dgamma_dt`: LSODA with real-valued reformulation

[`NBM.run_dgamma_dt`](decision_model.py#L2360) reformulates the complex γ ODE as a real 2D system (scipy's stiff solvers reject complex `y0`), uses a single LSODA call (not restarted RK45 windows), and checks convergence via the actual `|dgamma_dt|` at the endpoint.

**Why:** at certain walker positions (e.g. `x ~ 1.1` with two symmetric targets at `~±30°`), `dgamma_dt` has 3 equilibria with a **near-saddle slow manifold** between them where a Jacobian eigenvalue is `~−1.7e-5`. Trajectories crossing this manifold need 60–150 time units to escape. The previous restarted-RK45 implementation had `t_Final=30` (too short), lost adaptive step history at each restart, and used a less reliable finite-difference convergence check. Default `t_Final` is now 100 in both `run_dgamma_dt` and `dtheta_dt`. LSODA uses ~170 nfev (~10ms) for hard cases — comparable to or faster than the old approach when it succeeded.

**Known TODO:** [`IEM.run_dgamma_dt`](decision_model.py#L3337) still uses the restarted-RK45 pattern and likely suffers from the same near-saddle behavior. Apply matching LSODA fix when warnings appear there.

### The `R < 0.01` filter

At `θ = π` with cutoff weighting, targets behind the observer have zero neural weight, so `dgamma_dt = −γ` and the only equilibrium is `R = 0`. The filter correctly excludes this trivial state.

### Walker target detection (`plot_walkers`)

Both `NBM.plot_walkers` and `IEM.plot_walkers` use a two-layer target detection system:

1. **Proximity check** — `Targets.get_dist_to_targets(loc) < target_tol` at the start of each step. Works for all geometry types (circle returns `center_dist - r`, capsule returns `max(spine_dist - w/2, 0)`, delta returns Euclidean distance). Default `target_tol = v*dt` (one step size).
2. **Trajectory intersection** — `Targets.check_trajectory_intersection(old_loc, new_loc)` after each step. Catches pass-throughs where the walker steps entirely through a target in one Euler step. Uses point-to-segment distance for circles, segment-to-segment distance (`Targets._min_dist_segments`) for capsules.

If a walker exhausts `max_steps` (default 1500) without finding a target, a `warnings.warn` is issued with the repetition number, final position, and closest target distance. The walk is still included in the heatmap aggregation.

## Stability criterion

Default stability test is the **3×3 coupled Jacobian** on `(γ_re, γ_im, θ)`, built numerically with `h=1e-6`, `tol=1e-8`.

- NBM uses `dθ/dt = K·R·sin(ego_angle)` (via `convert_gamma`).
- IEM uses `dθ/dt = K·|γ|·sin(angle(γ) − θ)`.

Implemented in [`NBM._discrim_coupled`](decision_model.py#L2449) and [`IEM._discrim_coupled`](decision_model.py#L3565). `sc_equilib`, `gamma_equilib`, `_count_stable_at`, `_process_point`, `plot_direction_mesh`, and `plot_bifurcation_diagram` all accept `stability_criterion='coupled'` (default) or `'discrim_a'` (legacy 2D test, kept for side-by-side comparison plots). (NBM only — `IEM.sc_equilib` and `IEM.gamma_equilib` return just gammas without a stability list; the IEM plot/count helpers do their own stability test.)

**Why the coupled criterion is correct:** for self-consistent equilibria, the physically meaningful question is stability of the **coupled 3D system**, not the 2D γ subsystem at fixed `focal_angle`. The two criteria disagree wherever the heading dimension contributes a positive eigenvalue while the γ subsystem alone is stable.

**Self-consistent equilibria are exactly equilibria of the 3-eq system.** Proof: γ = R+0j gives `angle(γ) = 0`, so `ego_angle = inv_neural(0) = 0` (integral neural map sends 0→0 by symmetry), so `dθ/dt = K·R·sin(0) = 0`. Combined with `dγ/dt = 0` from the search, all three RHS components vanish. So "saddles" by the coupled criterion are real 3-eq equilibria with positive Jacobian eigenvalues, not numerical artifacts.

The legacy 2D criterion (`_discrim_A` on NBM, `_discrim_A_nu` on IEM) was *over-counting* stable equilibria — at (1.5, 0) in the vonmises k=0.55 setup, coupled reports (3 stable, 2 unstable) while `discrim_a` reports (5, 0). The 2 extra "stable" equilibria at `θ ≈ ±0.16` pass the 2D test but the coupled 3×3 Jacobian has a positive eigenvalue `≈ +0.31`.

## Geometry: targets

Three target geometries are supported: `circle`, `delta` (point), and `capsule`.

**Capsule** (line-segment spine + semicircular endcaps of radius `w/2`) replaced an earlier `segment` geometry that had multiple bugs (operator precedence in overlap check, self-assignment in distance calc, broken angle sorting, scalar param handling) and a fundamental issue: zero angular extent when viewed end-on. Capsule solves end-on vanishing (`w > 0` always gives nonzero extent), simplifies overlap detection (distance-to-spine ≤ `w/2`), and degenerates to a circle when `l=0`.

**Known limitation — capsule blocking approximation:** blocking order uses closest-point distance sorting, which is approximate for capsules that mutually occlude at different angles. A fully correct solution would need per-angle depth comparison. Acceptable for current use; flag if pathological cases come up.

## Bifurcation diagrams — conventions

[`plot_bifurcation_diagram`](decision_model.py#L2734) (NBM) and [`plot_bifurcation_diagram`](decision_model.py#L3668) (IEM) render `(x, y)` parameter sweeps colored by stable-equilibrium count.

- **Default colormap is viridis keyed on stable count alone**, with a `max_count` kwarg.
- An earlier experiment categorized cells by `(n_stable, n_unstable)` pairs with a fixed 60-color palette. **This was reverted** — two-axis color coding made boundaries harder to read, not easier. Don't reintroduce two-axis categorization without an explicit ask. The `stability_criterion='coupled'` plumbing from that work is preserved (real correctness fix).
- **`boundary_dilation` kwarg** (default 1) widens each refinement pass by promoting cells sharing a corner with a disagreement cell. Addresses stair-step artifacts at region boundaries and partially helps thin features.
- **Cell-center sampling deferred.** Evaluating each cell's center as a 5th disagreement test was discussed and deferred — some features are genuinely thin and there's not much getting around that without finer base sampling. If thin-feature artifacts persist after `boundary_dilation` and modest `num_x`/`num_y` increases, propose adding center sampling rather than further enlarging dilation.

## Physical phenomena

### 0-stable "decision paralysis" band under power warping

With `c=0.5` power warping, a transition band between 1-stable and 2-stable regions has **0 stable equilibria**. This is a **genuine pitchfork-like bifurcation**, not a numerical artifact:

- `_discrim_A` formula verified correct for `ν=1` cosine kernel across 1,084 equilibria (0 mismatches vs numerical Jacobian) in all configs (delta/circle × warped/unwarped).
- Full coupled 3×3 Jacobian also shows instability — heading coupling doesn't rescue stability.
- The bifurcation occurs because power warping stretches physical angles to wider neural angles (~112–138° separation), destabilizing the transverse perturbation mode.

These are real "decision paralysis" locations where no stable consensus exists under self-consistent heading-consensus coupling.

### Hopf island with stable limit cycle (vonmises k=0.55, two-target setup)

In the parameter window analyzed in [VM_bifurcations/VERDICT.md](VM_bifurcations/VERDICT.md), the bifurcation skeleton near `(2.1, ±2.45)` includes:

- A closed **Hopf curve** (magenta loop) inside the 1-equilibrium region, ending at degenerate Hopf (codim-2 Bautin) points where the eigenvalue's positive peak just touches zero.
- A separate **saddle-node curve** between the 1- and 3-equilibrium regions.
- Inside the Hopf loop: 1 Hopf-unstable focus + 1 **stable limit cycle** of period ~17.4. Walkers exhibit steady "head-bobbing" oscillation, not convergence.
- Cascade across the arc (3-eq → SN crossing → wedge → Hopf crossing → unstable-focus+cycle → Hopf back): only one stable eq genuinely disappears (via SN); the other temporarily loses stability via Hopf and recovers.

**How to apply:** in similar parameter regimes, expect to see Hopf-unstable foci coexisting with stable limit cycles, not just simple "decision paralysis." [VERDICT.md](VM_bifurcations/VERDICT.md) is the authoritative write-up with reproducibility, scripts, and PNGs.

## Basin-of-attraction estimator (vetted in basin_estimation/, not yet wired in)

An end-to-end vetting effort for the basin-estimation machinery that supports the two-panel bifurcation+basin plot TODO lives in [basin_estimation/](basin_estimation/). The eleven-step vetting plan is complete; everything load-bearing is documented in [basin_estimation/findings.md](basin_estimation/findings.md) (§1–§13). The folder also contains [free_energy_derivation.md](basin_estimation/free_energy_derivation.md) (the F̂(γ) derivation), [README.md](basin_estimation/README.md) (the vetting plan), and standalone test scripts that all pass.

**Status:** not wired into `decision_model.py`. The user has chosen to make the modeling changes described in the next subsection *before* integrating, so the integration doesn't have to be redone.

**Public API.** The vetted entry point is

```python
result = compute_basins_at_focal_loc(focal_loc, *, scan_single_stable=False)
# in basin_estimation/basin_via_theta.py
```

Always returns a dict with `basins` (list per stable SC eq), `stable_count`, `unstable_count`, and `sentinel` (None for normal cases; a short reason string for Hopf islands, perception-collapse cells, or partial-success cases). The basin-dict shape depends on cell class — see findings.md §10.2.

**Key results worth knowing without reading findings.md in full:**

- **F̂(γ; θ, focal_loc)** has a closed-form expression derived from the Hamiltonian and Glauber dynamics; verified to machine precision against `dgamma_dt`. F̂ is the Lyapunov function for the deterministic γ-flow.
- **The slow manifold is not a global object.** Projecting (γ, θ) dynamics onto 1D heading dynamics gives a *union of γ-branches glued at γ-folds*, not a single smooth f(θ). Basin boundaries in θ can be smooth saddles, γ-folds (catastrophic γ-branch jumps), perception-collapse zones (R → 0 over an interval under tight cutoffs), or — under the proposed sin(Θ*/2) dynamics — branch-cut discontinuities. See findings.md §4 and §7.
- **ΔF_γ** (the F̂-barrier between the γ-min at γ_s and a γ-saddle at fixed θ_s) is the empirically validated noise-robustness scalar **but only in multistable cells**. γ-Langevin MC at (1.2, 0) confirms Kramers prediction within factor of 2; slope of log(τ) vs 1/D within 16% of ΔF_γ; relative ordering correct. See findings.md §6, §8.
- **Conceptual scope correction (findings.md §13.6).** Noise-robustness scalars (ΔV, ΔF_γ) only carry their Kramers meaning in multistable regions on S¹. In 1-stable regions the basin is the entire circle minus the unstable point; noise pushing the walker over the V-saddle just sends θ around the loop back to the same stable. The right 1-stable visualization is direction arrow only (optionally with a stiffness glyph from the F̂-Hessian at γ_s), no Kramers-style robustness coloring. The default `scan_single_stable=False` reflects this — 1-stable cells return a minimal basin record without running the scan.
- **Per-cell-class rendering rules** for the two-panel plot are in findings.md §10.4 and §12: 0-stable cells get sentinel-marker glyphs (limit cycle, blind zone); 1-stable cells get a direction arrow; multistable cells get one arrow per stable colored or sized by ΔF_γ.
- **Cost benchmark**: parallel runtime ~4.2 min on 32 cores for a 41×41 grid with basins (vs ~20 s for sc_equilib-only baseline). See findings.md §11. Multistable cells dominate the budget; 1-stable cells are essentially free.

**Headline open theory question** (findings.md §13.4): is θ-noise (the existing `std` knob in `plot_walkers`) a faithful proxy for γ-noise (the physically motivated noise from Glauber T and finite-size 1/N fluctuations)? Deferred from Step 8. A focused MC sweep at the (1.2, 0) multistable calibration points would close it. Other smaller theory gaps and the carry-forward list are in findings.md §13.5.

### How the proposed modeling changes affect the basin-estimation findings

Two modeling changes are planned (in a future session) *before* wiring the basin estimator into `decision_model.py`. Below is what they do and don't change about the vetting work above. The full discussion (including the half-angle-identity discontinuity correction) is in the session that produced this CLAUDE.md update; the summary follows.

**Change (1): replace `dθ/dt = K·R·sin(Θ*)` with `dθ/dt = K·R·sin(Θ*/2)`.** Motivation: under the current sin(Θ*) form, torque is zero both when consensus is straight ahead (Θ*=0, the SC equilibrium) AND when consensus is directly behind (Θ*=±π). The behind case is a *spurious* zero — the walker becomes undecided. With sin(Θ*/2), torque is zero only at Θ*=0 and maximal at the facing-away point.

What is unchanged in the vetting:
- F̂(γ; θ, focal_loc) derivation — the θ-equation does not enter F̂ at all (F̂ comes from γ-only Glauber dynamics).
- γ-Langevin SDE, γ-saddle finding at fixed θ, ΔF_γ values, all of Steps 1–3 and Step 6.
- Discontinuity detection framework (Step 7) — the detector handles a new event type naturally.
- All sentinel-handling and graceful-failure behavior (Step 10).

What changes:
- f(θ) on the slow manifold has a new functional form; numerical values shift.
- The Schur-complement slow eigenvalue (Step 5 §5) scales by ½ at SC eqs because d/dθ[sin(ego/2)] at ego=0 is ½ vs 1. Stability classification (sign) is unchanged.
- **A new basin-boundary type appears.** Under sin(Θ*/2), wherever γ_eq crosses the negative real axis (consensus directly behind), f has a *jump discontinuity* from +K·R to −K·R. **This is not a numerical artifact.** Both sin(arg(γ)/2) and the algebraic half-angle identity `sign(γ_im)·sqrt((|γ|−γ_re)/(2|γ|))` give two-sided limits +1 and −1 at γ = −|γ|+0j. Any single-valued angle representation with range (−π, π] inherits this discontinuity. Physically it represents the left/right ambiguity at the facing-away point — a real fork in the deterministic dynamics. Step 7's |Δf|-jump detector already catches discontinuities of this kind; the basin-extraction machinery would gain a fourth boundary type (saddle, γ-fold, perception-collapse, branch-cut).
- Poincaré-Hopf in its smooth form (#stable = #unstable smooth zeros) becomes a generalized version that counts sign-change discontinuities with the same parity as smooth unstable zeros.
- Specific findings to re-verify after the change: Hopf island location at (2.1, 2.45) (depends on 3×3 Jacobian eigenvalues, which change), specific slow-eigenvalue magnitudes, specific basin widths. Qualitative existence of Hopf islands somewhere is likely preserved.

Code that needs updating for Change (1): `decision_model.py` (`dtheta_dt`, `_discrim_coupled`, and the analogous IEM functions); `basin_estimation/theta_scan.py`, `basin_estimation/basin_via_theta.py`, `basin_estimation/mc_escape.py` (the `_eval_f` formula and the θ-update in the MC integrator).

**Change (2): default `weight_angle_only=True` in `PerceptionModel`; possibly decouple weighting from warping entirely.** Motivation: cleaner API, more interpretable defaults.

What is unchanged in the vetting:
- F̂ derivation, ∇F̂, Hessian formulas — all are functions of (θ̂_j, ρ_j), the perception-model outputs. New defaults give different (θ̂_j, ρ_j) values but the closed forms are identical.
- All framework machinery (γ-Langevin, basin extraction, discontinuity detection, MC validation, Kramers checks, sentinel handling). Every basin-estimation test script in basin_estimation/ remains parameter-agnostic and would still pass.

What changes:
- Numerical values at every calibration point shift. The vetting was done with VM-k055 (vonmises k=0.55 + integral warping); the existing calibration points like "(1.2, 0) is 3-stable" may no longer be true under the new defaults.
- Bifurcation structure on (x, y) shifts. Multistable regions probably shrink (less perception nonlinearity); Hopf islands may move or disappear under default uniform-weighting.
- The blind-spot trap (see Open TODOs) becomes an opt-in scenario rather than the default behavior. The Step 7 BlindSpot test is still valid as a stress test of the perception-collapse detector but requires explicit cutoff weighting.

Code that needs updating for Change (2): `PerceptionModel` constructor defaults; possible factoring of weighting and warping into separate optional callables. The basin_estimation/ scripts don't need code changes for Change (2); they just need new calibration points re-identified.

**Bottom line.** The vetting *framework* (every methodology and test script in basin_estimation/) survives both changes intact. The specific *numerical results* at the calibration points used in Steps 5–9 would change under either change. Doing the model work first and then wiring the basin estimator avoids re-vetting against parameters that are about to move.

## Open TODOs

- **`IEM.run_dgamma_dt` LSODA port** — same restarted-RK45 pattern as the old NBM version; apply matching real-valued LSODA fix when warnings appear.
- **Cell-center sampling for bifurcation refinement** — deferred; propose if `boundary_dilation` + grid increases are insufficient for thin features.
- **Walker blind-spot trap under cutoff weighting.** With `a=0, b=pi` cutoff + integral neural mapping and delta targets, walkers that overshoot a target and get all targets behind them lose the ability to navigate back. The integral neural mapping collapses all behind-the-walker ego angles to ±180° in neural space, gamma locks onto -1+0j (the ±π branch cut), and `sin(±180°) ≈ 0` kills the restoring torque. The walker enters a pure random walk. Detailed analysis in [weighting_analysis/README.md](weighting_analysis/README.md). Possible remedies: minimum torque floor / U-turn behavior, wider neural mapping (`a > 0`), or heading-dependent noise. Not yet implemented.
- **Two-panel bifurcation + basin-of-attraction plot.** The vetting work for the basin estimator is complete in [basin_estimation/](basin_estimation/); see the "Basin-of-attraction estimator" section above for status, public API, and key results. ΔF_γ (γ-noise free-energy barrier from the F̂ Lyapunov function) is the empirically validated noise-robustness scalar in multistable cells. Public API `compute_basins_at_focal_loc` ready to call from a new `NeuralBandModel` method. **Pending modeling changes (sin/2 dθ/dt and weighting/warping defaults) before integration** — see the next TODO and the "How the proposed modeling changes affect the basin-estimation findings" subsection above.

- **Modeling changes pending before basin-estimator integration:**
  - **(1) Change `dθ/dt` from `K·R·sin(Θ*)` to `K·R·sin(Θ*/2)`.** Gives the walker maximum torque when consensus is directly behind (instead of zero torque, the current "undecided" pathology). Introduces a deliberate left/right discontinuity in f(θ) at the facing-away configuration — this is intrinsic to sin(Θ*/2) on a single-valued angle representation and represents a real physical fork in the dynamics, not a numerical artifact. Affects `dtheta_dt`, `_discrim_coupled` in NBM (and IEM analogs); affects a few basin_estimation/ test scripts. Impact analysis: see "Basin-of-attraction estimator" section above.
  - **(2) Default `weight_angle_only=True` in `PerceptionModel`; possibly decouple weighting from warping entirely** (warping as user-specified neural-angle map; weighting as user-specified μ(θ) defaulting to uniform). Purely parametric API improvement from the basin-estimator framework's perspective — the math machinery is parameter-agnostic — but numerical values at calibration points shift, and the blind-spot trap (next TODO) becomes opt-in rather than default. Impact analysis: see "Basin-of-attraction estimator" section above.

## Common gotchas

- When modifying `sc_equilib`, `gamma_equilib`, or `_get_target_signals`, preserve exact interval arithmetic. Mesh-based fallback paths exist for plotting only.
- When changing the neural-angle transform, the `a`/`b`/`k` setters trigger `_build_integral_splines` rebuild automatically — don't bypass them.
- When modifying `run_dgamma_dt`, preserve the real-valued reformulation for LSODA compatibility.
- When discussing model differences or debugging warping-related issues, always check which coordinate system each quantity lives in. Most subtle bugs trace back to a coordinate-frame mismatch.
