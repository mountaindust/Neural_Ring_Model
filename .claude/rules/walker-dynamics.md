---
paths:
  - "decision_model.py"
---

# Walker dynamics (`plot_walkers`) — noise law, blind-spot search, target detection

Deep detail for the SDE walker simulation in `decision_model.py` (`NBM.plot_walkers` / `IEM.plot_walkers`). Auto-loads with `decision_model.py`. The deterministic stability machinery (SC equilibria, Jacobians, bifurcation rasters) is noise-free and untouched by everything here — see [torque-and-stability.md](torque-and-stability.md) for that.

## Walker target detection

Both `NBM.plot_walkers` and `IEM.plot_walkers` use a two-layer target detection system:

1. **Proximity check** — `Targets.get_dist_to_targets(loc) < target_tol` at the start of each step. Works for all geometry types (circle returns `center_dist - r`, capsule returns `max(spine_dist - w/2, 0)`, delta returns Euclidean distance). Default `target_tol = v*dt` (one step size).
2. **Trajectory intersection** — `Targets.check_trajectory_intersection(old_loc, new_loc)` after each step. Catches pass-throughs where the walker steps entirely through a target in one Euler step. Uses point-to-segment distance for circles, segment-to-segment distance (`Targets._min_dist_segments`) for capsules.

If a walker exhausts `max_steps` (default 1500) without finding a target, a `warnings.warn` is issued with the repetition number, final position, and closest target distance. The walk is still included in the heatmap aggregation.

**Degenerate-histogram guard.** The heatmap path uses `RectBivariateSpline` (bicubic, needs ≥4 points/axis); the fallback-to-`imshow` guard now triggers at `< 4` bin centers (was `< 2`) so a single near-straight walk no longer crashes the plot.

## Blind-spot search (via the `R=0` fast-path, at the independent `walk_std`)

On any step where *no* target is visible — `percep_model.get_neural_signals()` returns empty arrays (e.g. under a cutoff weight with `b_weight < π`, all targets fall outside the visible cone) — a cheap fast-path sets `self.gamma = 0`, deterministic torque `= 0`, `R = 0` (no wasted `dγ/dt` solve), and the heading noise to **`walk_std`** (default `0.5·π`) — a pure-rotational-diffusion search walk until a target re-enters view (proximity/trajectory checks still fire normally). `walk_std` is a **separate `plot_walkers` parameter, orthogonal to the committed `std`**: the visible-step gated formula `σ·(1−R)^p` gives only `σ` at `R=0`, which in gentle constant mode (`std=0.1`) is too weak to re-acquire — so the blind intensity is set independently. This is the un-removal of the old `blind_search_std` knob (the unification over-reached): you can now run e.g. `std=0.2, walk_std=π/2`. `walk_std=0` freezes the blind drift; a **fully deterministic** walk is `std=0` *and* `walk_std=0`. Validated in [tests/test_half_angle_torque.py](../../tests/test_half_angle_torque.py): a lone-target blind start re-acquires under the *default* constant mode (`std=None→0.1`, `walk_std=0.5·π`); `walk_std=0` marches straight off even when `std>0` (orthogonality test). The `0.5·π` default makes the per-unit-time heading-change 2σ span the full circle (±π) without over-rotating (`0.75·π` gave 2σ=1.5π).

## Euler-Maruyama heading step with state-gated noise

The visible-step heading update is `θ ← θ + K·R^{R_exp}·sin(Θ/2)·dt + σ·(1−R)^p·cos(Θ/2)·√dt·Z` with `Z ~ N(0,1)`, `R = |γ|`, `Θ` = the consensus angle relative to heading (the torque's angle), `σ = std`, `p = noise_exp`, `R_exp` the drift exponent (the `cos(Θ/2)` factor applies only when `p≠0`; `R_exp=1` is the model torque `dθ/dt`). The diffusion term scales as **`√dt`, not `dt`**: a Wiener increment has variance `dt`, so the kick gives a per-unit-time angular variance **independent of the step size** (an earlier version multiplied by `dt`, variance `∝ dt²`, which made the effective temperature shrink as `dt` was refined). `σ` is a rotational-diffusion *intensity*, not a per-step σ — but now a **state-gated** one.

- **`noise_exp=0` → constant `σ·dW`** (the old additive law; `(1−R)^0=1`, no `cos` factor).
- **`noise_exp>0`** interpolates a random walk (`R→0`, undecided/blind) ↔ low-noise homing (`R→1`, committed); larger `p` closes the gate faster. The drift is *also* `R`-weighted (`dθ/dt = K·R·sin(Θ/2)`), so the step is literally `R·(pursuit) + (1−R)^p·(diffusion)`.
- **Heading-aligned `cos(Θ/2)` modulation (`p≠0` only):** noise is **full facing consensus** (`Θ=0`) and **zero facing away** (`Θ=±π`) — in quadrature with the `sin(Θ/2)` torque, so corrective swings back are noise-free while exploration near course is preserved (a "hot at the well bottom, cold at the barrier" temperature profile). Implemented model-agnostically via the torque: `cos(Θ/2)=√(1−(dθ/dt /(K·R))²)`, so it auto-pairs with whatever the torque is (NBM: `Θ=arg γ`; IEM: the egocentric consensus). Stress-tested at `noise_exp=1`, 4 circle targets: ~12% lower path tortuosity and steps-to-capture, ~11% less facing-away time, capture unchanged; effect vanishes where the walker rarely faces away (`p=2/3`, 2 targets). Strictly benign (paths only tighten).
- **Drift exponent `R_exp` (walker-only).** The walker's pursuit torque is `K·R^{R_exp}·sin(Θ/2)`; `R_exp=1` is the model's `dθ/dt`. **Regime-aware default `R_exp=None` → `1` when `noise_exp==0`, else `1/noise_exp`** — `R^{1/p} > R` for `R<1`, so the walker pursues *harder* at intermediate coherence, balancing the `(1−R)^p` noise gate (a one-number explore→commit handoff). Strongly reduces facing-away time (4 circle, r=0.1: −58% at p=2, −77% at p=3; tighter paths, faster capture, capture unchanged). **Affects only the walker's drift** — `dtheta_dt` and the deterministic SC/bifurcation/basin machinery keep `R^1`. (If ever wanted globally in the torque law it's deterministic-structure-benign: SC-eq locations and stability signs preserved, θ-timescales rescale.)
- **Regime-aware default for `std=None`** (the *visible*-step scale): `0.1` when `noise_exp==0` (a gentle constant noise) and `walk_std` when `noise_exp>0` (the random-walk intensity the gate tames once committed). `std=0` ⇒ visible steps deterministic. **Blind steps use `walk_std`, not `std`** — the two are orthogonal knobs (see Blind-spot search); the `R=0` value of `σ·(1−R)^p` would otherwise be only `σ`, too weak in constant mode.
- **Shipped default is `noise_exp=0`** (constant noise); the gated law is opt-in pending the foveal-weight commitment-signal analysis (see [TODO.md](../../TODO.md)). `R` saturates to ~1 on commitment for extended targets but is pinned near `1/N` for uniform-weight point targets — i.e. the gate's *ability to close* is inherited from the perception model (geometry/`angle_weight`), not the noise law.

## Two distinct loss mechanisms, and what is/isn't fixed

The half-angle torque eliminates the *dead-zone* loss (targets behind → restoring torque was ~0; now maximal). The blind-spot search eliminates the *true blind-spot* loss (zero visible targets). A **third, unrelated** mechanism survives and should not be mistaken for either: with **point (delta) targets and large K (e.g. K=10)** a walker can settle into a stable wide *orbit* around the cluster (high R≈0.97, all targets visible, consensus persistently ~90° to the side), circling outside all the zero-radius targets without landing. This is the strong-coupling limit-cycle regime (cf. VM Hopf islands), not a perception failure. It vanishes at the new default `K=2` or with any finite target radius (a 4-delta sweep loses 8/30 → 2/30 under the dead-zone fix at K=10, and 0/30 at K=2 or with circle targets).
