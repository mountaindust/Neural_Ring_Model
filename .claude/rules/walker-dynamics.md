---
paths:
  - "decision_model.py"
---

# Walker dynamics (`plot_walkers`) — noise law, blind-spot search, target detection

Deep detail for the SDE walker simulation in `decision_model.py` (`NBM.plot_walkers`). Auto-loads with `decision_model.py`. The deterministic stability machinery (SC equilibria, Jacobians, bifurcation rasters) is noise-free and untouched by everything here — see [torque-and-stability.md](torque-and-stability.md) for that.

## Carrying γ between heading steps

Each step warm-starts `run_dgamma_dt` from the previous step's γ, **rotated by minus the turn just taken** (`self.gamma *= np.exp(-1j*turn)` in `_simulate_one_walk`, matching `_basin_destination` and `plot_dtheta_dt`'s swept mode). `turn` is the *full* heading change `dtheta*dt + noise` — noise turns the observer too.

**Why.** `γ = Σₖ nₖ e^{iθ̂ₖ}` carries one population per visible target. A step moves the observer, not the neurons: `nₖ` is unchanged by the turn (it relaxes afterwards, on the fast timescale) while every `θ̂ₖ` shifts by `−turn`, so re-evaluating the readout at the new angles *is* a rotation of γ. Exact when the shift is rigid (identity warp); under a warp each target shifts by `−U′(θₖ−θ)·turn`, so the rotation is the leading term.

**When it bites.** Never, where γ is single-basin — the relaxation erases the warm start (measured: 58/58 steps identical either way, ~1e-11). At a **fold**, where the branch the walker is riding disappears, the warm start decides which surviving attractor γ falls to, and that is the decision the model is about. Sub-stepping the crossing 1024× does not make the conventions agree — the τ₀→0 limit is singular there.

**Noise context.** The per-step heading kick is `std·√dt`; where that is larger than the transport (≈ the turn per step) the choice is sub-dominant for *ensemble statistics* but still changes individual tracks. `_basin_destination` runs noise-free, so there it is the only thing deciding fold outcomes.

## Walker target detection

`plot_walkers` uses a two-layer target detection system:

1. **Proximity check** — `Targets.get_dist_to_targets(loc) < target_tol` at the start of each step. Works for all geometry types (circle returns `center_dist - r`, capsule returns `max(spine_dist - w/2, 0)`, delta returns Euclidean distance). Default `target_tol = v*dt` (one step size).
2. **Trajectory intersection** — `Targets.check_trajectory_intersection(old_loc, new_loc)` after each step. Catches pass-throughs where the walker steps entirely through a target in one Euler step. Uses point-to-segment distance for circles, segment-to-segment distance (`Targets._min_dist_segments`) for capsules.

If a walker exhausts `max_steps` (default 1500) without finding a target, a `warnings.warn` is issued with the repetition number, final position, and closest target distance. The walk is still plotted as a track.

**Plot output — tracks only.** `plot_walkers` plots every walker trajectory as a black line (`ax.plot(..., 'k', alpha=alpha)`); there is no histogram/heatmap. The `alpha` kwarg (default `0.5`) sets track opacity — lower values let overlapping paths reveal density. (The earlier `np.histogram2d` + `RectBivariateSpline`-interpolated `imshow` heatmap, its degenerate-bin `imshow` fallback, and the `plot_tracks` toggle were all removed 2026-06-10; the `RectBivariateSpline` import went with them.)

## Blind-spot search (via the `R=0` fast-path, at the independent `walk_std`)

On any step where *no* target is visible — `percep_model.get_neural_signals()` returns empty arrays (e.g. under a cutoff weight with `b_weight < π`, all targets fall outside the visible cone) — a cheap fast-path sets `self.gamma = 0`, deterministic torque `= 0`, `R = 0` (no wasted `dγ/dt` solve), and the heading noise to **`walk_std`** (default `0.5·π`) — a pure-rotational-diffusion search walk until a target re-enters view (proximity/trajectory checks still fire normally). `walk_std` is a **separate `plot_walkers` parameter, orthogonal to the committed `std`**: the visible-step gated formula `σ·(1−R)^p` gives only `σ` at `R=0`, which in gentle constant mode (`std=0.1`) is too weak to re-acquire — so the blind intensity is set independently. This is the un-removal of the old `blind_search_std` knob (the unification over-reached): you can now run e.g. `std=0.2, walk_std=π/2`. `walk_std=0` freezes the blind drift; a **fully deterministic** walk is `std=0` *and* `walk_std=0`. Validated in [tests/test_half_angle_torque.py](../../tests/test_half_angle_torque.py): a lone-target blind start re-acquires under the *default* constant mode (`std=None→0.1`, `walk_std=0.5·π`); `walk_std=0` marches straight off even when `std>0` (orthogonality test). The `0.5·π` default makes the per-unit-time heading-change 2σ span the full circle (±π) without over-rotating (`0.75·π` gave 2σ=1.5π).

## Euler-Maruyama heading step with state-gated noise

The visible-step heading update is `θ ← θ + K·R^{R_exp}·sin(Θ/2)·dt + σ·(1−R)^p·cos(Θ/2)·√dt·Z` with `Z ~ N(0,1)`, `R = |γ|`, `Θ` = the consensus angle relative to heading (the torque's angle), `σ = std`, `p = noise_exp`, `R_exp` the drift exponent (the `cos(Θ/2)` factor applies only when `p≠0`; `R_exp=1` is the model torque `dθ/dt`). The diffusion term scales as **`√dt`, not `dt`**: a Wiener increment has variance `dt`, so the kick gives a per-unit-time angular variance **independent of the step size** (an earlier version multiplied by `dt`, variance `∝ dt²`, which made the effective temperature shrink as `dt` was refined). `σ` is a rotational-diffusion *intensity*, not a per-step σ — but now a **state-gated** one.

- **`noise_exp=0` → constant `σ·dW`** (the old additive law; `(1−R)^0=1`, no `cos` factor).
- **`noise_exp>0`** interpolates a random walk (`R→0`, undecided/blind) ↔ low-noise homing (`R→1`, committed); larger `p` closes the gate faster. The drift is *also* `R`-weighted (`dθ/dt = K·R·sin(Θ/2)`), so the step is literally `R·(pursuit) + (1−R)^p·(diffusion)`.
- **Heading-aligned `cos(Θ/2)` modulation (`p≠0` only):** noise is **full facing consensus** (`Θ=0`) and **zero facing away** (`Θ=±π`) — in quadrature with the `sin(Θ/2)` torque, so corrective swings back are noise-free while exploration near course is preserved (a "hot at the well bottom, cold at the barrier" temperature profile). Implemented model-agnostically via the torque: `cos(Θ/2)=√(1−(dθ/dt /(K·R))²)`, so it auto-pairs with the torque (`Θ = arg γ`). Stress-tested at `noise_exp=1`, 4 circle targets: ~12% lower path tortuosity and steps-to-capture, ~11% less facing-away time, capture unchanged; effect vanishes where the walker rarely faces away (`p=2/3`, 2 targets). Strictly benign (paths only tighten).
- **Drift exponent `R_exp` (walker-only).** The walker's pursuit torque is `K·R^{R_exp}·sin(Θ/2)`; **default `R_exp=1` is the model's `dθ/dt`** (changed 2026-06-11 from the old regime-aware `None → 1/noise_exp`). **Affects only the walker's drift** — `dtheta_dt` and the deterministic SC/bifurcation/basin machinery keep `R^1`. Two opt-in deviations: **(1)** `R_exp = 1/noise_exp` (`<1`, the old gated default) boosts pursuit at intermediate coherence — it reduces facing-away *time* on small targets (4 circle r=0.1: −58%/−77% at `noise_exp` 2/3) but the reduction is cosmetic (capture, steps-to-capture, and tortuosity all unchanged vs `R_exp=1`; the absolute facing-away fraction is already only a few %), which is why it is no longer the default; **(2)** `R_exp ≈ noise_exp` (`>1`, *suppresses* contested-zone drift) de-skews multi-target splits under a steep `(1−R)^q` gate and makes the split ratio dt-robust — see [gated_pq_analysis.md](../../walker_analysis/gated_pq_analysis.md). (If ever wanted globally in the torque law it's deterministic-structure-benign: SC-eq locations and stability signs preserved, θ-timescales rescale.)
- **Regime-aware default for `std=None`** (the *visible*-step scale): `0.1` when `noise_exp==0` (a gentle constant noise) and `walk_std` when `noise_exp>0` (the random-walk intensity the gate tames once committed). `std=0` ⇒ visible steps deterministic. **Blind steps use `walk_std`, not `std`** — the two are orthogonal knobs (see Blind-spot search); the `R=0` value of `σ·(1−R)^p` would otherwise be only `σ`, too weak in constant mode.
- **Shipped default is `noise_exp=0`** (constant noise); the gated law is opt-in pending the foveal-weight commitment-signal analysis (see [TODO.md](../../TODO.md)). `R` saturates to ~1 on commitment for extended targets but is pinned near `1/N` for uniform-weight point targets — i.e. the gate's *ability to close* is inherited from the perception model (geometry/`angle_weight`), not the noise law.

## Two distinct loss mechanisms, and what is/isn't fixed

The half-angle torque eliminates the *dead-zone* loss (targets behind → restoring torque was ~0; now maximal). The blind-spot search eliminates the *true blind-spot* loss (zero visible targets). A **third, unrelated** mechanism survives and should not be mistaken for either: with **point (delta) targets and large K (e.g. K=10)** a walker can settle into a stable wide *orbit* around the cluster (high R≈0.97, all targets visible, consensus persistently ~90° to the side), circling outside all the zero-radius targets without landing. This is the strong-coupling limit-cycle regime (cf. VM Hopf islands), not a perception failure. It vanishes at the new default `K=2` or with any finite target radius (a 4-delta sweep loses 8/30 → 2/30 under the dead-zone fix at K=10, and 0/30 at K=2 or with circle targets).
