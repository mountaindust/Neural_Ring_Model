---
paths:
  - "decision_model.py"
---

# Perception model & solver numerics

Deep detail for `PerceptionModel`, the interval-arithmetic blocking path, the integral-spline maps, and the equilibrium/ODE solvers in `decision_model.py`. The durable summary is in [CLAUDE.md](../../CLAUDE.md); this rule auto-loads whenever `decision_model.py` is open.

## `PerceptionModel` API — full parameter detail

`PerceptionModel` has two independent roles set by two constructor args: **`neural_angle_dist`** (the WARP, egocentric→neural angle map) and **`angle_weight`** (the WEIGHT, density integrated over each target's visible arc to set ρ). See CLAUDE.md for the role overview and the allowed family values.

Per-role parameters use generic two-slot kwargs `a_warp/b_warp` and `a_weight/b_weight`, mapped per family by `_FAMILY_INFO` (cutoff: a,b; lin_cutoff: a,b; vonmises: a=k; symmetric_beta: a=alpha,b; reg_power: a=d,b=e; direct_power: a=c). `lin_cutoff` (the **default warp**) is the analytic trapezoidal sibling of `cutoff` — same `(a,b)` slots/defaults and `0≤a<b` validation, but spline-free: closed-form `_lin_cutoff_integral`/`_lin_cutoff_int_inverse` (the inverse is a single `sqrt`, exact to machine precision — no `±π` condition limit), so `_make_integral_spline` returns `(None,None)` for it like `symmetric_beta`. Unset slots take family defaults. Change parameters post-init by **assigning the same-named properties** — `pm.a_warp = 0.55`, `pm.b_weight = π`, etc. — which auto-rebuild only the affected role's splines (a tied weight is mirrored + rebuilt when its warp changes). The setter is strict (raises on an unused slot, identity warp, uniform weight, or tied-weight target); the getter is permissive (returns `None` for an unused slot / identity / uniform). `warp_params`/`weight_params` are **read-only** views of the current canonical-keyed params (e.g. `pm.warp_params == {'k': 0.55}`); both mutation (`['k']=…`) and rebinding raise with a message pointing back to the `a_*`/`b_*` properties. The old `neural_weight`/`neural_angle`/`weight_angle_only` args and the `a/b/k/alpha/d/e` properties were **removed** (non-backward-compatible; see git history for the decouple session). Old→new: `neural_weight=W, neural_angle='integral'` (full weighting) → `neural_angle_dist=W, angle_weight='neural_angle_dist'`; `weight_angle_only=True` → `angle_weight=None`; `neural_angle='power'` → `neural_angle_dist='direct_power'`.


## Perception: exact interval arithmetic for blocking

[`PerceptionModel._get_target_signals`](../../decision_model.py#L1650) uses exact interval arithmetic, not a mesh discretization:

- [`_subtract_intervals_circle`](../../decision_model.py#L1457) computes visible angular intervals after blocking by closer targets.
- [`_integrate_neural_weight`](../../decision_model.py#L1497) integrates neural weight (cutoff or vonmises) over those intervals analytically.

The original implementation Riemann-summed over a discrete θ-mesh and produced equilibrium residuals of ~1e-3 — not roundoff but genuine discretization error that caused convergence failures in `sc_equilib`. Switching to interval arithmetic dropped residuals to machine precision (~1e-14) and gave a 4.5× speedup for circle targets.

The mesh path is retained only for `plot_blocked_signals` (the `mesh_signal` flag); the `G.sum()==0` case returns empty arrays, not NaN.

## Integral antiderivatives precomputed as splines

`PerceptionModel._make_integral_spline(name, params)` tabulates forward + inverse `CubicSpline`s at 2001 nodes for the CDF-like integral map of a density family (`'cutoff'` F(θ; a, b), `'vonmises'` G(θ; k), `'reg_power'` F(θ; d, e); `'symmetric_beta'` is analytic, no spline). Since the warp/weight decouple there are **two** spline sets built once at `__init__`: `_build_warp_splines` (forward+inverse, for `get_neural_angle`/`_inverse`) and `_build_weight_splines` (forward only, the ρ arc-integral antiderivative — skipped when the weight is uniform or tied to the warp, in which case the warp forward spline is reused). Assigning the `a_warp`/`b_warp`/`a_weight`/`b_weight` properties rebuilds only the affected role's splines (via `_set_slot`). The generic forward/inverse evaluators are `_eval_forward_map`/`_eval_inverse_map`.

- **Accuracy:** forward direction matches the reference `quad`/`cdf` to ~5e-11 everywhere; end-to-end `_get_target_signals` ρ values match the reference path to ~1e-16 (machine precision).
- **Inverse direction is condition-limited to ~1e-8 near `y = ±π`** because `dF/dx → 0` at the boundary. This only affects `get_neural_angle_inverse` via `convert_gamma(γ)` with `np.angle(γ)`; 1e-8 error in `ego_angle` is negligible for walker dynamics and the walker rotates out of the poorly-conditioned region whenever it matters.
- **Performance:** the circle/cutoff cross-model validation grid went from 91.8s → 34.8s (~2.6×). Per-point cost for circle targets now comparable to delta targets (~7ms vs ~15ms).
- **Cutoff spline construction (non-obvious):** `F(x)` saturates to ±π in floating point once `b − |x| < ~0.05` (the `exp(−norm/(b−x))` tail underflows). Naïve `CubicSpline` fails the strict-monotonicity requirement. `_make_integral_spline` uses a greedy monotone filter (in the cutoff branch) to drop saturated boundary nodes while preserving exact ±π endpoints.
- **Domain restriction:** inverse splines raise `ValueError` on `y` outside `[−π, π]`; forward splines saturate safely. Callers are domain-clean by construction.
- **Reference kernels retained for testing:** `_smooth_cutoff_integral` and `_smooth_cutoff_int_inverse` (static methods) are still used by tests to validate the splines against `quad`/`brentq`. `scipy.stats.vonmises.cdf/ppf` are the vonmises reference.

## NBM `sc_equilib`: single-pass solver

Simplified from an earlier two-pass `brentq + multistart` to a single-pass strategy:

1. Scan `Im(dgamma_dt)` across a 100-point θ mesh at each probe radius `R_probe ∈ (0.3, 0.5, 0.7)` — plus **`0.85` when the ν kernel is active** (`angle_distortion_nu` set and ≠ 1). A sharp kernel migrates equilibria out to `R ≈ 0.77–0.85`, above every plain probe, and the Im-scan then reports **no sign change at all** for them; over a ν=0.5 grid the extra probe recovers roots the plain radii miss entirely. Gated on `_nu_active` so the plain-cosine path keeps its cost and its output unchanged.
2. Find sign changes → `brentq` for precise `θ_c`.
3. Add `θ = 0, ±π` as explicit candidates.
4. Polish each with 2D `hybr` (`tol=1e-10`), require `sol.success`. **An equivariant Jacobian is supplied** (`jac=self._self_consistent_jac`, central-difference): hybr's *built-in* Jacobian uses one-sided forward differences whose `+h` step direction breaks the θ→−θ mirror, so without it the polish trajectory from a start is not the mirror of the trajectory from the sign-flipped start. A central difference perturbs `±h`, so the whole hybrj Newton/dogleg iteration is mirror-equivariant — see "y-symmetry" below.
5. Residual threshold **1e-4**. The `hybr+logistic` combination can produce residuals up to ~2e-5 due to exponential amplification; a tighter 1e-6 threshold was silently dropping ~10% of valid equilibria and creating apparent holes in direction meshes.
6. Deduplicate with both circular angle distance < 0.02 **and** R distance < 0.01. Both axes are required: near a saddle-node bifurcation, two genuine equilibria of opposite stability can share θ to within ~1e-3 rad while differing in R by ~0.02, so θ-only dedup silently discards one of the pair. **When a merge cluster has mixed stability the survivor is chosen stability-canonically** (the stable member wins, replacing a first-kept unstable one), not by scan order — because the brentq scan order *reverses* under the y-mirror, so order-dependent "keep first" makes the survivor's stability (and the stable count) flip between (x, y) and (x, −y). y-symmetry of `plot_bifurcation_diagram` output on a symmetric target setup is the real diagnostic.

**y-symmetry (fixed 2026-06-12):** the stable-count output of `sc_equilib` on an x-axis-symmetric target setup is now **exactly** y-flip symmetric (`count(x,y) == count(x,−y)` with angles negated). Two coupled sources were fixed: (1) the hybr forward-difference Jacobian (step 4 above) — the dominant source; supplying the central-difference `jac` made the polish equivariant; (2) the mixed-stability dedup chirality (step 6) — a genuine near-SN stable/unstable pair within the (0.02, 0.01) tolerance was merged to whichever member came *first*, and the scan order reverses under the mirror. `_self_consistent_eq` is itself y-flip symmetric to ~1e-20, so with the algorithm made equivariant the equilibrium *set* is mirror-exact. Verified: uniform-weight cutoff and fly `lin_cutoff` sweeps go from 1–5 mismatched stable-count pixels to **0**; no stable-count pixel changed off the corrected asymmetries (before/after grid diff); total/unstable-count asymmetry also dropped (fly 4→0). **Residual (out of scope):** under the **`cutoff` warp spline** the *unstable*-count near a few SN pixels can still flip y (uniform 16→12), because the warp forward spline itself is only ~5e-11 mirror-symmetric (not the hybr/dedup path); invisible in the stable-count diagram. `lin_cutoff` (analytic, machine-precision symmetric) shows none of this. The `walker_analysis/decision_skeleton.py` midline pin is now belt-and-suspenders (still guards the SC-*unstable* centre-branch drift) rather than the only remedy.

## `run_dgamma_dt`: LSODA with real-valued reformulation

[`NBM.run_dgamma_dt`](../../decision_model.py#L2360) reformulates the complex γ ODE as a real 2D system (scipy's stiff solvers reject complex `y0`), uses a single LSODA call (not restarted RK45 windows), and checks convergence via the actual `|dgamma_dt|` at the endpoint. **Preserve the real-valued reformulation when modifying this** (LSODA compatibility).

**Why:** at certain walker positions (e.g. `x ~ 1.1` with two symmetric targets at `~±30°`), `dgamma_dt` has 3 equilibria with a **near-saddle slow manifold** between them where a Jacobian eigenvalue is `~−1.7e-5`. Trajectories crossing this manifold need 60–150 time units to escape. The previous restarted-RK45 implementation had `t_Final=30` (too short), lost adaptive step history at each restart, and used a less reliable finite-difference convergence check. Default `t_Final` is now 100 in both `run_dgamma_dt` and `dtheta_dt`. LSODA uses ~170 nfev (~10ms) for hard cases — comparable to or faster than the old approach when it succeeded.


## The `R < 0.01` filter

At `θ = π` with cutoff weighting, targets behind the observer have zero neural weight, so `dgamma_dt = −γ` and the only equilibrium is `R = 0`. The filter correctly excludes this trivial state.
