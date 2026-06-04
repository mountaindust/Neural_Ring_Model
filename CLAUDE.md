# Neural Ring Model — Project Guide

Mathematical model of collective decision-making based on Ising-type dynamics on a neural ring. This file is the durable project context loaded into every Claude Code session in this repo. It captures theory, model architecture, solver design, and known limitations — the things that are hard to re-derive from code or git history alone.

## Codebase layout

- [decision_model.py](decision_model.py) — the project's research code (~4200 lines). Two model classes (`NeuralBandModel`, `IsingExtModel`), a `PerceptionModel`, and a `Targets` helper.
- Jupyter notebooks (`compare_sc_vm.ipynb`, `compare_sc_beta.ipynb`, `neural_band.ipynb`, `neural_band_walker.ipynb`, `ising_workbook.ipynb`, `debug_all_unstable.ipynb`) — testing, exploration, and visualization.
- [VM_bifurcations/](VM_bifurcations/) — diagnostic scripts and the [VERDICT.md](VM_bifurcations/VERDICT.md) write-up of the Hopf-island / saddle-node bifurcation skeleton near (2.1, ±2.45) in the vonmises-k0.55 / two-target setup.
- [bifurc_plots/](bifurc_plots/) — exploratory parameter-sweep scripts. `neural_weight_sweep.py` (full weighting) and `neural_weight_sweep_angle_only.py` (warping-only) are companions; `bifurcation_compare_discrim_vs_coupled.py` compares stability criteria; `arc_skeleton_and_island_dynamics.py` is the upper-arc / island-dynamics combined figure. Not publication-quality — settings are sized for fast iteration on a many-core machine. More function-space exploration is needed before any of these can be locked in for publication.
- [basin_estimation/](basin_estimation/) — vetting work for the basin-of-attraction estimator (the two-panel bifurcation+basin plot TODO). Eleven-step plan complete; public API ready but **not yet wired into `decision_model.py`**. Status, results, and the re-vetting under the new torque law are in the "Basin-of-attraction estimator" section below; running findings in [findings.md](basin_estimation/findings.md).
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
- Walker torque: `dθ/dt = K·R·sin(Θ/2)` where `Θ = arg(γ)` is the **neural** consensus angle and `R = |γ|`. **Half-angle law in the neural angle directly** (see "Half-angle heading torque" below) — no inverse-warp mapping. `Θ ∈ (−π, π]` (from `np.angle`) is already wrapped before halving; neural angle 0 is straight ahead (the SC equilibrium), `±π` is facing-away. Default `K=2`. (Earlier the torque used the inverse-warped egocentric angle, `sin(ego/2)` via `convert_gamma`; see the dated note in "Half-angle heading torque" for the change and its consequences.)
- Self-consistent equilibria have γ = R + 0j (real positive): heading=consensus ⇒ egocentric consensus = 0 ⇒ Θ_neural = f(0) = 0 ⇒ γ = R + 0j.

Key methods:
- [`sc_equilib`](decision_model.py#L2250) — self-consistent equilibrium finder (heading = consensus). Used for bifurcation diagrams in (x, y).
- [`gamma_equilib`](decision_model.py#L2181) — γ-equilibrium finder at a fixed observer heading. *Not* the self-consistent finder.
- [`run_dgamma_dt`](decision_model.py#L2360) — ODE to steady state.
- [`dtheta_dt`](decision_model.py#L2411) — heading dynamics.
- [`convert_gamma`](decision_model.py#L2339) — inverse neural mapping from γ to `(ego_angle, R)`. Introspection only — **no longer on the torque path** (dθ/dt uses `arg(γ)` directly); kept for external analysis/diagnostics that want the egocentric direction.
- [`plot_walkers`](decision_model.py#L2979) — SDE walker simulation.
- [`plot_direction_mesh`](decision_model.py#L2549), [`plot_bifurcation_diagram`](decision_model.py#L2734).

### `IsingExtModel` (IEM)

Closest to the published PNAS paper. γ lives in **allocentric** space.

- When `ν ≠ 1`, the power transform applies only to the cosine-kernel coupling strength, not to the pull direction in `dγ/dt`. The exponential term uses physical (allocentric) angles for the pull direction.
- Walker torque: `dθ/dt = K·|γ|·sin(convert_angles(angle(γ) − θ)/2)`. **Half-angle law.** Both `angle(γ)` and `θ` are allocentric; subtraction is needed because both live in the same absolute frame. Unlike NBM, the egocentric argument `angle(γ) − θ` is *not* pre-wrapped, so `convert_angles` must wrap it to `(−π, π]` before halving (`sin(x/2)` is 4π-periodic). Default `K=2`.
- Self-consistent equilibria satisfy `dgamma_dt(γ, focal_angle=angle(γ)) = 0` where `angle(γ)` is the allocentric consensus direction.

Key methods: [`sc_equilib`](decision_model.py#L3567) (self-consistent finder, returns gammas; bifurcation diagrams), [`gamma_equilib`](decision_model.py#L3502) (fixed-heading γ-finder), [`run_dgamma_dt`](decision_model.py#L3337), [`dtheta_dt`](decision_model.py#L3390), [`plot_bifurcation_diagram`](decision_model.py#L3668), [`plot_direction_mesh`](decision_model.py#L3891), [`plot_walkers`](decision_model.py#L4057).

### `PerceptionModel` API — warp and weight are decoupled

`PerceptionModel` has **two independent roles**, set by two constructor args:

- **`neural_angle_dist`** (WARP): the distribution integrated CDF-like to map egocentric→neural angles. One of `{'cutoff','vonmises','symmetric_beta','reg_power','direct_power', None}`. `'direct_power'` is the power angle map `f(θ)=π·sign(θ)(|θ|/π)^c` (NOT a CDF-integral); `None` is identity (no warp).
- **`angle_weight`** (WEIGHT): the density integrated over each target's visible arc to set ρ. One of `{'cutoff','vonmises','symmetric_beta','reg_power','neural_angle_dist', None}`. `'neural_angle_dist'` ties the weight to the warp (old full-weighting behavior); `None` (the **default**) is uniform weight (old `weight_angle_only=True` behavior). `'direct_power'` is **disallowed** as a weight (it is a signed angle map, not a density — use `reg_power`).

Per-role parameters use generic two-slot kwargs `a_warp/b_warp` and `a_weight/b_weight`, mapped per family by `_FAMILY_INFO` (cutoff: a,b; vonmises: a=k; symmetric_beta: a=alpha,b; reg_power: a=d,b=e; direct_power: a=c). Unset slots take family defaults. Change parameters post-init by **assigning the same-named properties** — `pm.a_warp = 0.55`, `pm.b_weight = π`, etc. — which auto-rebuild only the affected role's splines (a tied weight is mirrored + rebuilt when its warp changes). The setter is strict (raises on an unused slot, identity warp, uniform weight, or tied-weight target); the getter is permissive (returns `None` for an unused slot / identity / uniform). `warp_params`/`weight_params` are **read-only** views of the current canonical-keyed params (e.g. `pm.warp_params == {'k': 0.55}`); both mutation (`['k']=…`) and rebinding raise with a message pointing back to the `a_*`/`b_*` properties. The old `neural_weight`/`neural_angle`/`weight_angle_only` args and the `a/b/k/alpha/d/e` properties were **removed** (non-backward-compatible; see git history for the decouple session). Old→new: `neural_weight=W, neural_angle='integral'` (full weighting) → `neural_angle_dist=W, angle_weight='neural_angle_dist'`; `weight_angle_only=True` → `angle_weight=None`; `neural_angle='power'` → `neural_angle_dist='direct_power'`.

**IEM must be used with `neural_angle_dist=None, angle_weight=None`.** In IEM, γ lives in allocentric/physical coordinates; the model assumes the observer perceives target angles directly without neural warping. Running IEM with a non-identity `neural_angle_dist` is a category error — the math behind IEM's `dγ/dt` doesn't apply.

**NBM is the model for warped perception.** If you need foveal density, egocentric warping, or any non-identity neural mapping, use NBM. The two are *not* substitutes for each other under warping; they were designed to handle the warping nonlinearity in incompatible ways.

**Diagnostic note:** if an IEM bifurcation diagram or direction mesh comes back almost entirely zero/empty, the most likely cause is accidentally driving IEM with a non-identity `neural_angle_dist`. IEM's polar-init multistart in `sc_equilib` doesn't find roots under that (invalid) configuration. This is *not* a solver bug — it's the model rejecting an invalid input. Switch to `neural_angle_dist=None` to confirm.

## Self-consistent equilibria

At a physical turning equilibrium, the observer has stopped turning, so heading = allocentric consensus direction. This makes egocentric consensus = 0, which (for any reasonable warp) makes Θ_neural = 0, which makes γ = R + 0j.

Three approaches were considered for handling neural warping with self-consistency; **Option 3 was chosen:**

1. ~~Make warping allocentric.~~ Rejected: experimental evidence shows the denser neural band near the center of the visual field is intrinsically egocentric. The warping has to be egocentric.
2. ~~Hybrid with mismatch from Hamiltonian.~~ Rejected: suboptimal, no good error estimates.
3. **Accept heading-dependent equilibria and find the self-consistent ones.** Adopted because we already accept that consensus direction should depend on where the observer is looking; the self-consistent solution is the physically meaningful subset.

The allocentric consensus direction *is* `θ` (the heading) at a self-consistent equilibrium — no inverse mapping needed to recover the physical direction. `NBM.sc_equilib(focal_loc=..., stability_criterion=...)` returns `(allocentric_angles, stability_booleans)`. (`NBM.gamma_equilib` is a separate method that finds γ-eqs at a fixed observer heading; it is *not* the self-consistent finder.)

For NBM, the SC turning equilibria are where `dθ/dt = K·R·sin(Θ/2) = 0` with `Θ = arg(γ)` the neural consensus angle: the smooth zero is `Θ = 0` (heading = consensus, stable); `Θ = ±π` (facing directly *away*) is the intentional branch-cut fork, not a smooth equilibrium.

## Solver architecture

### Perception: exact interval arithmetic for blocking

[`PerceptionModel._get_target_signals`](decision_model.py#L1650) uses exact interval arithmetic, not a mesh discretization:

- [`_subtract_intervals_circle`](decision_model.py#L1457) computes visible angular intervals after blocking by closer targets.
- [`_integrate_neural_weight`](decision_model.py#L1497) integrates neural weight (cutoff or vonmises) over those intervals analytically.

The original implementation Riemann-summed over a discrete θ-mesh and produced equilibrium residuals of ~1e-3 — not roundoff but genuine discretization error that caused convergence failures in `sc_equilib`. Switching to interval arithmetic dropped residuals to machine precision (~1e-14) and gave a 4.5× speedup for circle targets.

The mesh path is retained only for `plot_blocked_signals` (the `mesh_signal` flag); the `G.sum()==0` case returns empty arrays, not NaN.

### Integral antiderivatives precomputed as splines

`PerceptionModel._make_integral_spline(name, params)` tabulates forward + inverse `CubicSpline`s at 2001 nodes for the CDF-like integral map of a density family (`'cutoff'` F(θ; a, b), `'vonmises'` G(θ; k), `'reg_power'` F(θ; d, e); `'symmetric_beta'` is analytic, no spline). Since the warp/weight decouple there are **two** spline sets built once at `__init__`: `_build_warp_splines` (forward+inverse, for `get_neural_angle`/`_inverse`) and `_build_weight_splines` (forward only, the ρ arc-integral antiderivative — skipped when the weight is uniform or tied to the warp, in which case the warp forward spline is reused). Assigning the `a_warp`/`b_warp`/`a_weight`/`b_weight` properties rebuilds only the affected role's splines (via `_set_slot`). The generic forward/inverse evaluators are `_eval_forward_map`/`_eval_inverse_map`.

- **Accuracy:** forward direction matches the reference `quad`/`cdf` to ~5e-11 everywhere; end-to-end `_get_target_signals` ρ values match the reference path to ~1e-16 (machine precision).
- **Inverse direction is condition-limited to ~1e-8 near `y = ±π`** because `dF/dx → 0` at the boundary. This only affects `get_neural_angle_inverse` via `convert_gamma(γ)` with `np.angle(γ)`; 1e-8 error in `ego_angle` is negligible for walker dynamics and the walker rotates out of the poorly-conditioned region whenever it matters.
- **Performance:** circle/cutoff `test_broad_validation` went from 91.8s → 34.8s (~2.6×). Per-point cost for circle targets now comparable to delta targets (~7ms vs ~15ms).
- **Cutoff spline construction (non-obvious):** `F(x)` saturates to ±π in floating point once `b − |x| < ~0.05` (the `exp(−norm/(b−x))` tail underflows). Naïve `CubicSpline` fails the strict-monotonicity requirement. `_make_integral_spline` uses a greedy monotone filter (in the cutoff branch) to drop saturated boundary nodes while preserving exact ±π endpoints.
- **Domain restriction:** inverse splines raise `ValueError` on `y` outside `[−π, π]`; forward splines saturate safely. Callers are domain-clean by construction.
- **Reference kernels retained for testing:** `_smooth_cutoff_integral` and `_smooth_cutoff_int_inverse` (static methods) are still used by tests to validate the splines against `quad`/`brentq`. `scipy.stats.vonmises.cdf/ppf` are the vonmises reference.

### NBM `sc_equilib`: single-pass solver

Simplified from an earlier two-pass `brentq + multistart` to a single-pass strategy:

1. Scan `Im(dgamma_dt)` across a 100-point θ mesh at `R_probe = 0.5`.
2. Find sign changes → `brentq` for precise `θ_c`.
3. Add `θ = 0, ±π` as explicit candidates.
4. Polish each with 2D `hybr` (`tol=1e-10`), require `sol.success`.
5. Residual threshold **1e-4**. The `hybr+logistic` combination can produce residuals up to ~2e-5 due to exponential amplification; a tighter 1e-6 threshold was silently dropping ~10% of valid equilibria and creating apparent holes in direction meshes.
6. Deduplicate with both circular angle distance < 0.02 **and** R distance < 0.01. Both axes are required: near a saddle-node bifurcation, two genuine equilibria of opposite stability can share θ to within ~1e-3 rad while differing in R by ~0.02, so θ-only dedup silently discards one of the pair. Which one survives depends on the brentq sign-change scan order, which flips under coordinate symmetries — producing visible chirality (anti-symmetric "1-stable invading 2-stable" intrusions) in bifurcation diagrams that should be y-symmetric. (`IEM.sc_equilib` dedups on the full complex `|γ_eq − existing_γ|`, which sidesteps this.) The broad-validation grid in [tests/test_broad_validation.py](tests/test_broad_validation.py) doesn't reach near-SN configurations, so y-symmetry of `plot_bifurcation_diagram` output on a symmetric target setup is the real diagnostic.

**Residual asymmetry (known):** even with full (θ, R) dedup, ~25–30% of the y-flip pixel asymmetry persists in uniform-weight (`angle_weight=None`) cutoff/beta/vonmises sweeps. Traced to `scipy.optimize.root(method='hybr')` itself: `_self_consistent_eq` is y-flip symmetric to 1e-20, but hybr's internal Jacobian estimation uses positive forward-difference steps, so the trajectory from a starting point is not the mirror of its trajectory from the sign-flipped starting point. Two scaffolds for fixing this if it ever matters: symmetrize the multistart by also trying mirrored starts for every candidate, or densify the brentq candidate seeding so both members of a near-SN pair are reachable from independent starts.

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

**Blind-spot search (now free, via the `R=0` fast-path).** On any step where *no* target is visible — `percep_model.get_neural_signals()` returns empty arrays (e.g. under a cutoff weight with `b_weight < π`, all targets fall outside the visible cone) — a cheap fast-path sets `self.gamma = 0`, deterministic torque `= 0`, and `R = 0` (no wasted `dγ/dt` solve). Because the heading noise is `σ·(1−R)^p` (see "Euler-Maruyama heading step" below), `R=0` makes the gate `(1−R)^p = 1` for any exponent, so a blind step automatically becomes a full-`σ` pure-rotational-diffusion search walk until a target re-enters view (proximity/trajectory checks still fire normally). The dedicated `blind_search_std` knob was **removed**: with `std`'s gated default of `0.75·π` (when `noise_exp>0`) the search is recovered for free at the same intensity. `std=0` gives the frozen-drift behavior (deterministic). Validated in [tests/test_half_angle_torque.py](tests/test_half_angle_torque.py): a lone-target blind start is re-acquired across seeds; `std=0` marches straight off.

**Note — degenerate-histogram guard.** The heatmap path uses `RectBivariateSpline` (bicubic, needs ≥4 points/axis); the fallback-to-`imshow` guard now triggers at `< 4` bin centers (was `< 2`) so a single near-straight walk no longer crashes the plot.

**Euler-Maruyama heading step with state-gated noise.** The heading update is `θ ← θ + dθ/dt·dt + σ·(1−R)^p·√dt·Z` with `Z ~ N(0,1)`, `R = |γ|`, `σ = std`, `p = noise_exp`. The diffusion term scales as **`√dt`, not `dt`**: a Wiener increment has variance `dt`, so the kick gives a per-unit-time angular variance of `σ²·(1−R)^{2p}` that is **independent of the step size** (an earlier version multiplied by `dt`, variance `∝ dt²`, which made the effective temperature shrink as `dt` was refined). `σ` is a rotational-diffusion *intensity*, not a per-step σ — but now a **state-gated** one.

- **`noise_exp=0` → constant `σ·dW`** (the old additive law; `(1−R)^0=1` everywhere).
- **`noise_exp>0`** interpolates a random walk (`R→0`, undecided/blind) ↔ low-noise homing (`R→1`, committed); larger `p` closes the gate faster. The drift is *also* `R`-weighted (`dθ/dt = K·R·sin(arg γ/2)`), so the step is literally `R·(pursuit) + (1−R)^p·(diffusion)`.
- **Regime-aware default for `std=None`:** `0.1` when `noise_exp==0` (a gentle constant noise) and `0.75·π` when `noise_exp>0` (the random-walk/blind intensity the gate tames once committed). `σ` plays two roles with two natural scales; the default picks the right one. `std=0` ⇒ fully deterministic.
- **Shipped default is `noise_exp=0`** (constant noise); the gated law is opt-in pending the foveal-weight commitment-signal analysis (see TODO). `R` saturates to ~1 on commitment for extended targets but is pinned near `1/N` for uniform-weight point targets — i.e. the gate's *ability to close* is inherited from the perception model (geometry/`angle_weight`), not the noise law.

The deterministic stability machinery (SC equilibria, Jacobians, bifurcation rasters) is noise-free and untouched by this.

**Two distinct loss mechanisms, and what is/isn't fixed.** The half-angle torque eliminates the *dead-zone* loss (targets behind → restoring torque was ~0; now maximal). The blind-spot search eliminates the *true blind-spot* loss (zero visible targets). A **third, unrelated** mechanism survives and should not be mistaken for either: with **point (delta) targets and large K (e.g. K=10)** a walker can settle into a stable wide *orbit* around the cluster (high R≈0.97, all targets visible, consensus persistently ~90° to the side), circling outside all the zero-radius targets without landing. This is the strong-coupling limit-cycle regime (cf. VM Hopf islands), not a perception failure. It vanishes at the new default `K=2` or with any finite target radius (a 4-delta sweep loses 8/30 → 2/30 under the dead-zone fix at K=10, and 0/30 at K=2 or with circle targets).

## Stability criterion

Default stability test is the **3×3 coupled Jacobian** on `(γ_re, γ_im, θ)`, built numerically with `h=1e-6`, `tol=1e-8`.

- NBM uses `dθ/dt = K·R·sin(arg(γ)/2)` (the neural consensus angle directly).
- IEM uses `dθ/dt = K·|γ|·sin(convert_angles(angle(γ) − θ)/2)`.

Implemented in [`NBM._discrim_coupled`](decision_model.py#L2449) and [`IEM._discrim_coupled`](decision_model.py#L3565). `sc_equilib`, `gamma_equilib`, `_count_stable_at`, `_process_point`, `plot_direction_mesh`, and `plot_bifurcation_diagram` all accept `stability_criterion='coupled'` (default) or `'discrim_a'` (legacy 2D test, kept for side-by-side comparison plots). (NBM only — `IEM.sc_equilib` and `IEM.gamma_equilib` return just gammas without a stability list; the IEM plot/count helpers do their own stability test.)

**Why the coupled criterion is correct:** for self-consistent equilibria, the physically meaningful question is stability of the **coupled 3D system**, not the 2D γ subsystem at fixed `focal_angle`. The two criteria disagree wherever the heading dimension contributes a positive eigenvalue while the γ subsystem alone is stable.

**Self-consistent equilibria are exactly equilibria of the 3-eq system.** Proof: γ = R+0j gives `arg(γ) = 0`, so `dθ/dt = K·R·sin(0/2) = 0`. Combined with `dγ/dt = 0` from the search, all three RHS components vanish. So "saddles" by the coupled criterion are real 3-eq equilibria with positive Jacobian eigenvalues, not numerical artifacts.

### Half-angle heading torque and the `K=2` default

The heading torque is the **half-angle** law `dθ/dt = K·R·sin(arg(γ)/2)` (NBM — the neural consensus angle) / `K·|γ|·sin(ego/2)` with `ego = convert_angles(angle(γ)−θ)` (IEM — physical angle difference). Rationale: a walker that has committed to a neural direction should turn *faster* the farther that direction is from its heading. `sin(ego)` was zero both straight-ahead (ego=0, the wanted SC equilibrium) and directly-behind (ego=±π) — the behind-zero was a spurious "dead zone" that let overshooting walkers wander off. `sin(ego/2)` is zero only at ego=0 and maximal at the facing-away point, so a walker facing away turns hard back toward consensus. There is an intentional **jump discontinuity** at ego=±π (torque flips `+K·R ↔ −K·R`), a real left/right fork resolved by roundoff/noise; the existing Step-7 `|Δf|` detector treats it as a basin-boundary event.

**Update (2026-06-02) — NBM torque moved to the neural angle.** NBM now uses the neural consensus angle `arg(γ)` directly (`sin(arg(γ)/2)`), dropping the inverse-warp mapping `ego = get_neural_angle_inverse(arg(γ))` that the old `sin(ego/2)` form applied. *Why:* in neural coordinates the facing-away heading is always `±π`, so the `÷2` normalization (zero ahead, maximal `±K·R` facing away) is **perception-model-independent** — the old ego form's effective normalization was instead the warp-dependent `ν(0)=W'(0)` (the neural density at center; ≈ "÷b" for a cutoff of half-width `b`). It also makes dθ/dt a direct function of γ with no ill-conditioned inverse near `±π`. *Invariant under the change:* SC-equilibrium locations, stability classification, stable/unstable counts, saddle-node curves, and the branch-cut jump magnitude `2·K·R` — the change rescales only the θ-row of the coupled Jacobian by the positive constant `ν(0)`, preserving eigenvalue signs. *Changed (θ-side magnitudes):* slow eigenvalues, `V''(θ_s)`, θ-relaxation timescales, θ-noise barriers `ΔV`, and Hopf-curve fine structure all rescale by `ν(0)` under a non-identity warp (e.g. vonmises k=0.55: ×1.609, confirmed). **dθ/dt is phenomenological and actively evolving:** treat θ-side calibration numbers in [basin_estimation/findings.md](basin_estimation/findings.md) and [VM_bifurcation_old_dtheta/](VM_bifurcation_old_dtheta/VERDICT.md) as needing a re-run to be current; γ-side outputs (F̂, ΔF_γ, γ-folds, basin counts/widths) are turning-law-invariant and stand. IEM is unchanged (its `ego` is a physical angle difference, not warped). The `K=2` invariance note below concerns the earlier `sin(ego)`→`sin(ego/2)` half-angle change and still holds.

**Default `K` was raised 1 → 2.** `sin(x/2)` has slope ½ at x=0 vs 1 for `sin(x)`, so K=2 restores the near-front turning gain. This is exact, not cosmetic: **at every SC equilibrium (ego=0) the K-doubling cancels the ½ in the full 3×3 coupled Jacobian** (the surviving partials are `K·R·cos(0)·½·∂ego/∂·`; the `sin(0)=0` terms drop), so eigenvalues — and therefore stability classification, Hopf and saddle-node locations, and every `plot_bifurcation_diagram` stable-count — are **identical to the old `K=1, sin(ego)` model**. Verified two ways in [tests/test_half_angle_torque.py](tests/test_half_angle_torque.py): a per-eq Jacobian-eigenvalue equality test, and an end-to-end stable-count raster that is bit-identical before/after. Only the *global* flow (walker trajectories, basin shapes off ego=0) changes. The IEM `convert_angles` wrap is load-bearing: `sin(x/2)` is 4π-periodic, so omitting it would make the torque wrong across 2π boundaries (regression-tested).

The legacy 2D criterion (`_discrim_A` on NBM, `_discrim_A_nu` on IEM) was *over-counting* stable equilibria — at (1.5, 0) in the vonmises k=0.55 setup, coupled reports (3 stable, 2 unstable) while `discrim_a` reports (5, 0). The 2 extra "stable" equilibria at `θ ≈ ±0.16` pass the 2D test but the coupled 3×3 Jacobian has a positive eigenvalue `≈ +0.31`.

## Geometry: targets

Three target geometries are supported: `circle`, `delta` (point), and `capsule`.

**Capsule** (line-segment spine + semicircular endcaps of radius `w/2`) replaced an earlier buggy `segment` geometry whose fundamental flaw was zero angular extent when viewed end-on. Capsule gives nonzero extent for any `w > 0`, uses distance-to-spine ≤ `w/2` for overlap detection, and degenerates to a circle when `l=0`.

**Known limitation — capsule blocking approximation:** blocking order uses closest-point distance sorting, which is approximate for capsules that mutually occlude at different angles. A fully correct solution would need per-angle depth comparison. Acceptable for current use; flag if pathological cases come up.

## Bifurcation diagrams — conventions

[`plot_bifurcation_diagram`](decision_model.py#L2734) (NBM) and [`plot_bifurcation_diagram`](decision_model.py#L3668) (IEM) render `(x, y)` parameter sweeps colored by stable-equilibrium count.

- **Default colormap is viridis keyed on stable count alone**, with a `max_count` kwarg.
- **Don't reintroduce two-axis `(n_stable, n_unstable)` color coding** without an explicit ask — it was tried and made boundaries harder to read, not easier. (The `stability_criterion='coupled'` plumbing it introduced is preserved — a real correctness fix.)
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

### Weighting vs warping: the "ears" (detail in weighting_analysis/)

Now that warp and weight are decoupled (Change (2)), the full write-up lives in
[weighting_analysis/README.md](weighting_analysis/README.md) and need not be loaded every
session. One-line takeaway: warping alone reproduces the bifurcation structure of
full weighting **except** for two "ears" of extra far-target bistability at off-axis
observer positions behind two circle targets — present under a non-uniform `angle_weight`,
absent under uniform (`angle_weight=None`, the default). The ears are therefore now
opt-in. The README also covers the delta-target threshold shift, a delta+ANGLE Hopf
follow-up (Hopf-unstable foci but no limit cycle), and the cutoff blind-spot trap. The
README still uses the pre-decouple `neural_weight`/`weight_angle_only` vocabulary; map it
via the Old→new table in the "`PerceptionModel` API" section.

## Basin-of-attraction estimator (vetted in basin_estimation/, not yet wired in)

End-to-end vetting of the basin-estimation machinery for the two-panel bifurcation+basin plot TODO lives in [basin_estimation/](basin_estimation/): the eleven-step plan is complete and documented in [findings.md](basin_estimation/findings.md) (plus [free_energy_derivation.md](basin_estimation/free_energy_derivation.md) for the F̂ derivation); all test scripts pass.

**Status: vetted, not yet wired into `decision_model.py`.** Public entry point `compute_basins_at_focal_loc(focal_loc, *, scan_single_stable=False)` (in `basin_estimation/basin_via_theta.py`) returns a dict with `basins`, `stable_count`, `unstable_count`, and `sentinel` (None, or a reason string for Hopf-island / perception-collapse / partial-success cells; basin-dict shape per cell class in findings.md §10.2). Both prerequisite modeling changes (sin(Θ*/2) dθ/dt with K=2; warp/weight decouple) are **DONE**, and the Steps 5–9 calibration points were **re-vetted under the new law and confirmed invariant** (findings.md §0). What remains is integration + per-cell rendering (rules in findings.md §10.4), not re-vetting.

**Load-bearing facts** (full detail in findings.md):
- **ΔF_γ** — the F̂-barrier from a stable to a γ-saddle at fixed θ — is the γ-noise robustness scalar, MC-validated against Kramers within a factor of 2, **but only in multistable cells**; 1-stable cells have no second basin, so the deliverable there is a direction arrow + optional stiffness glyph, not a robustness color (§6, §8, §13.6). F̂ is Hamiltonian/Glauber-derived, independent of dθ/dt.
- **Basin boundaries** come in four kinds: smooth saddle, γ-fold (γ-branch jump), perception-collapse (R→0 under tight cutoffs), and — under the half-angle law — a **branch-cut** at the facing-away heading (f jumps ±K·R as γ_eq crosses the negative real axis). The first three are γ-side / turning-law-invariant; the branch-cut's existence is set by dθ/dt at the antipode (§4, §7).
- **Cost**: ~4.2 min parallel (32 cores) for a 41×41 grid with basins; multistable cells dominate, 1-stable/Hopf cells are nearly free (§11).

**Open, and dθ/dt-dependent.** dγ/dt is Hamiltonian-derived; dθ/dt is a phenomenological turning law (K and possibly its form may still move). This decouples cleanly: the γ-side outputs (slow manifold, γ-folds, ΔF_γ, SC-eq locations, basin counts/widths bounded by saddles/folds) are insulated from the turning-law uncertainty, as is γ-noise robustness. What stays tied to dθ/dt: θ-noise barrier magnitudes, and two specific open items — (a) whether θ-noise (the `std` knob in `plot_walkers`) is a faithful proxy for physical γ-noise (Glauber T, 1/N), deferred from Step 8 and only resolvable once dθ/dt is fixed (§13.4); (b) the θ-noise basin "depth" at the branch-cut/antipode, which needs its own first-passage treatment (the branch-cut is a repelling fork, not a Kramers barrier) once dθ/dt is fixed.

## Open TODOs

- **`IEM.run_dgamma_dt` LSODA port** — same restarted-RK45 pattern as the old NBM version; apply matching real-valued LSODA fix when warnings appear.
- **Cell-center sampling for bifurcation refinement** — deferred; propose if `boundary_dilation` + grid increases are insufficient for thin features.
- **Walker blind-spot trap under cutoff weighting — RESOLVED.** The half-angle law removes the behind-walker *dead zone*; the *true blind spot* (zero visible targets) is now handled by the state-gated noise itself — at `R=0` the gate `(1−R)^p=1` gives a full-`σ` search for free (see "Euler-Maruyama heading step" / "Blind-spot search"). The old dedicated `blind_search_std` knob was removed. Background in [weighting_analysis/README.md](weighting_analysis/README.md).
- **Residual heading-noise floor (idea, deferred).** Generalize the gated noise `σ·(1−R)^p` to `floor + (σ−floor)·(1−R)^p`, so a small noise persists even at full commitment (`R→1`) instead of vanishing. Appealing for several reasons but kept simple for now; revisit before merging this branch or before publication. Distinct from the two-scale `std` default (which is one scale, two regimes).
- **Foveal `angle_weight` as a commitment signal (analysis pending).** For uniform-weight *point* targets `R` is pinned near `1/N`, so the noise gate can't close; a concentrating (foveal) weight is what makes `R→1` on commitment. Plan: (A) sweep weight concentration to find the largest mild vonMises weight that lifts committed `R` for a small target (r≈0.1) while staying below the "ears" bistability; (B) fix the noise at `σ(1−R)^p` and show graceful degradation across target radius (delta = soft limit). Then decide whether to ship a gated default (`noise_exp>0`) + a default non-uniform `angle_weight`.
- **Two-panel bifurcation + basin-of-attraction plot.** Basin estimator vetted (see the "Basin-of-attraction estimator" section above); public API `compute_basins_at_focal_loc` ready to wire into a `NeuralBandModel` method. Both prerequisite modeling changes (sin(Θ*/2) dθ/dt with K=2; warp/weight decouple) are DONE and the Steps 5–9 calibration points re-vetted/invariant (findings.md §0); remaining work is integration + per-cell rendering, not re-vetting. The one new behavior to surface is the branch-cut basin-boundary type at θ≈±π (Step 7 already detects it).

## Common gotchas

- When modifying `sc_equilib`, `gamma_equilib`, or `_get_target_signals`, preserve exact interval arithmetic. Mesh-based fallback paths exist for plotting only.
- When changing warp or weight parameters, assign the `a_warp`/`b_warp`/`a_weight`/`b_weight` properties — they trigger the correct (and only the affected) spline rebuild. `warp_params`/`weight_params` are read-only views; both item assignment and rebinding raise (pointing you at the `a_*`/`b_*` properties).
- When modifying `run_dgamma_dt`, preserve the real-valued reformulation for LSODA compatibility.
- When discussing model differences or debugging warping-related issues, always check which coordinate system each quantity lives in. Most subtle bugs trace back to a coordinate-frame mismatch.
