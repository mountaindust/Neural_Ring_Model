# Tests — running convention

This directory mixes **two kinds of tests by design**. There is no
`pytest.ini`/`pyproject.toml`; pytest runs on naming-convention defaults. [`conftest.py`](conftest.py) only adds the repo root to `sys.path` so
`import decision_model` resolves.

## TL;DR

```sh
pytest tests/                      # runs the whole suite
```

Every file here is collected and run by `pytest tests/`.

## The two tiers

### 1. pytest-native correctness/unit tests
Discrete `test_*` functions with `assert`s. Run under pytest; some also re-invoke pytest or
their own functions from a `__main__` block, so `python tests/<file>.py` works too.

- [`test_half_angle_torque.py`](test_half_angle_torque.py) — dθ/dt half-angle torque law (NBM
  neural-angle form), torque shape + the ±π branch-cut jump, the
  `convert_angles` 4π-wrap regression, K-doubling Jacobian invariance, and walker
  noise/blind-spot behavior.
- [`test_reduced_criterion.py`](test_reduced_criterion.py) — correctness of the `'reduced'`
  stability criterion: Schur block-determinant identity, Schur == slaved slow flow, the
  documented vonmises (1.5, 0) `reduced`-vs-`discrim_a` disagreement, the absence of any
  `'coupled'` criterion, and the defaults.
- [`test_trajectory_intersection.py`](test_trajectory_intersection.py) —
  `Targets.check_trajectory_intersection` and `_min_dist_segments` across circle/delta/capsule.
- [`test_signal_cache.py`](test_signal_cache.py) — `PerceptionModel.signal_cache`, the
  perception memo used by `NBM.sc_equilib`/`gamma_equilib`: cache-hit contents and lifetime
  (dropped on exit/exception, nesting), the exact-key requirement (the Jacobians' θ±h probes
  must not collide with the base state), and bit-identical solver output with the memo
  disabled across setups × all three stability criteria.
- [`test_beta_coupling.py`](test_beta_coupling.py) — the neural Boltzmann factor
  `NeuralBandModel.beta`: `dγ/dt` against an independent reference, the coupling being
  independent of the target count (and of a target dropping out of view), agreement of
  the analytic `_discrim_A` free-energy Hessian with the numerically-differenced fast
  block (the only cross-check on that Hessian's β), and equivalence with the earlier
  `N/T` coupling where every target is perceived.
- [`test_angle_distortion_nu.py`](test_angle_distortion_nu.py) —
  `NeuralBandModel(angle_distortion_nu=...)`, the distorted cosine coupling kernel folded
  in from the retired `IsingExtModel`. ν=None and ν=1 leave the model bit-identical to the
  plain-cosine one; the constructor's *and* the property setter's identity-warp/uniform-weight
  requirement (plus non-positive/non-finite ν); the kernel's zero crossing; the `convert_angles`
  wrap regression (the ν kernel is not 2π-periodic); the `_discrim_A` numerical fallback vs the
  analytic Hessian; and `plot_dtheta_dt`'s neutral-seed sweep not leaking `self.gamma`.

### 2. Numerics-verification scripts (also pytest-discoverable)
These run their checks at **module import** using `check_*`/`ok`/`raises` helpers that tally
module-level `passed`/`failed` counters and print a per-check banner (max-diff vs tolerance,
per-family sections). The rich printed output is the point — keep them script-style.

Each exposes a single bridging `def test_<name>(): assert failed == 0` so that **`pytest`
genuinely fails when a check fails** (pytest captures the banner and shows it on failure), and
guards its `exit(1)` under `if __name__ == '__main__'` so running directly still returns a
non-zero exit code. `python tests/<file>.py` and `pytest` therefore agree.

- [`test_perception_spline.py`](test_perception_spline.py) — precomputed integral splines
  (forward/inverse/roundtrip/symmetry/endpoints/validation) vs quad/brentq/scipy references for
  every warp family, plus end-to-end `_integrate_neural_weight` invariance.
- [`test_lin_cutoff.py`](test_lin_cutoff.py) — the analytic `'lin_cutoff'` (trapezoidal) family:
  closed-form integral + inverse vs a quad reference, saturation/normalization, and end-to-end
  as both a warp and a weight.
- [`test_intervals.py`](test_intervals.py) — occlusion/blocking interval arithmetic
  (`_subtract_intervals_circle`, `_unwrap_interval`, `_subtract_interval_pair`) and
  `_integrate_neural_weight`.
- [`test_segments.py`](test_segments.py) — capsule target geometry (`check_target_overlap`,
  `get_dist_to_targets`, `get_percep_angles`, `_get_target_signals`; zero/finite width, end-on
  viewing, ±π wrap, partial occlusion, l=0 → circle degeneracy).
- [`test_warp_weight_decouple.py`](test_warp_weight_decouple.py) — the original example of this
  bridging pattern. New decoupled warp/weight API: cross-family warp+weight, per-role params,
  tied-vs-uniform equivalence, respline isolation, read-only `warp_params`/`weight_params`
  views, and error paths.

## Why some tests aren't `assert`-style

The numerics scripts cross-check spline/quad/brentq paths across whole parameter sweeps and
report per-check max-error vs tolerance. That diagnostic narrative is far more useful for
debugging a tolerance regression than a bare pass/fail would be, which is why they stay
script-style and are bridged into pytest rather than rewritten as many tiny `test_*` functions.
