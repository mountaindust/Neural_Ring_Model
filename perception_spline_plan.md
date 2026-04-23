# Precompute spline lookups for PerceptionModel integral transforms

## Context

`PerceptionModel` in [decision_model.py](decision_model.py) implements a mapping between perceived angles and "neural" angles via a CDF-like integral of the neural weight function. Today, every evaluation pays real compute:

- [decision_model.py:545-605](decision_model.py#L545-L605) — `_smooth_cutoff_integral_scalar` calls `scipy.integrate.quad` on the smooth transition zone `a < |theta| < b`.
- [decision_model.py:635-672](decision_model.py#L635-L672) — `_smooth_cutoff_int_inverse_scalar` calls `brentq` to invert that same integral, and each iteration of brentq calls `_smooth_cutoff_integral_scalar` (which calls `quad`).
- [decision_model.py:734-760](decision_model.py#L734-L760) and [decision_model.py:763-792](decision_model.py#L763-L792) — `_vonmises_integral` / `_vonmises_int_inverse` route through `scipy.stats.vonmises.cdf` / `.ppf`, which are vectorized C but still carry per-call `rv_continuous` dispatch overhead.

These are called on hot paths: `get_neural_angle` at [decision_model.py:1041](decision_model.py#L1041), `get_neural_angle_inverse` at [decision_model.py:1072](decision_model.py#L1072), and especially `_integrate_neural_weight` at [decision_model.py:967](decision_model.py#L967), which is invoked for every visible interval of every visible target inside `_get_target_signals`. In the walker code, `get_neural_angle_inverse` is called on every ODE step via [decision_model.py:1659](decision_model.py#L1659).

The integrals (both forward and inverse) are smooth, one-dimensional, and fully determined once the weight parameters (`a`, `b`, or `k`) are fixed. We can precompute spline lookups at model initialization and drop `quad`/`brentq`/`vonmises.ppf` off the hot path. The goal: O(1) evaluation for forward and inverse, with deviation vs. the current implementation at roundoff level (target: max abs error <= 1e-10, matching the old `quad` tolerance of `1.49e-10`).

## Design

### What gets splined

| `neural_weight` | `neural_angle` | Forward transform | Inverse transform |
|---|---|---|---|
| `None` | `'integral'` | identity (no spline) | identity |
| `'cutoff'` | `'integral'` | **spline** | **spline** |
| `'vonmises'` | `'integral'` | **spline** | **spline** |
| any | `'power'` | analytical (no spline) | analytical |
| any | `None` | identity | identity |

The `power` and identity cases stay analytical — there's nothing to gain.

### Spline construction

In `PerceptionModel.__init__`, after `self._a`, `self._b`, `self._k` are set, call a new helper `self._build_integral_splines()`.

- **Cutoff case** (`neural_weight == 'cutoff'`):
  - Build 2001 equispaced nodes `x_nodes` on `[-b, b]`.
  - Evaluate `y_nodes[i] = _smooth_cutoff_integral(x_nodes[i], a, b)` via a Python loop (build-time only; uses the renamed scalar reference).
  - Snap exact values: `y_nodes[0] = -pi`, `y_nodes[-1] = pi`, and force the center node (at `x=0`) to `y=0` to preserve the antisymmetry F(-x) = -F(x) at the midpoint.
  - Forward spline: `self._cutoff_forward_spline = CubicSpline(x_nodes, y_nodes, bc_type='natural')`.
  - Inverse spline: `self._cutoff_inverse_spline = CubicSpline(y_nodes, x_nodes, bc_type='natural')`. `y_nodes` is strictly monotone (cutoff > 0 on (-b, b)), so this is well-defined.

- **Vonmises case** (`neural_weight == 'vonmises'`):
  - Build 2001 equispaced nodes `theta_nodes` on `[-pi, pi]`.
  - Evaluate `y_nodes = 2*pi*(vonmises.cdf(theta_nodes, k) - 0.5)` (vectorized, cheap).
  - Snap exact values: `y_nodes[0] = -pi`, `y_nodes[-1] = pi`, and the center node to 0.
  - Forward: `self._vonmises_forward_spline = CubicSpline(theta_nodes, y_nodes, bc_type='natural')`.
  - Inverse: `self._vonmises_inverse_spline = CubicSpline(y_nodes, theta_nodes, bc_type='natural')`.

The construction cost is a few ms at init time; it runs once per configuration.

### Making `a`, `b`, `k` first-class mutable attributes

The current docstring lists `focal_loc`, `focal_angle`, and `targets` as freely reassignable at runtime. It omits `a`, `b`, `k` — but reassigning them is in fact the only way to reconfigure the weight function on an existing model, so this is a documentation oversight rather than a deliberate restriction. With splines in play we need to rebuild them whenever these parameters change, which is a good fit for Python `property` setters.

- Convert `a`, `b`, `k` into `@property` / `@<name>.setter` pairs on `PerceptionModel`. Backing storage: `self._a`, `self._b`, `self._k`.
- Each setter assigns the backing attribute and then calls `self._build_integral_splines()`.
- Update the `PerceptionModel` docstring to note that `a`, `b`, `k` may be reassigned at runtime and that doing so automatically rebuilds the integral splines. Cross-reference the existing note that `focal_loc`, `focal_angle`, `targets` are also mutable.

**Single build during `__init__`**: the constructor sets defaults by writing directly to the backing attributes (`self._a = ...`, `self._b = ...`, `self._k = ...`), which bypasses the setters entirely. After all attributes are in place, call `self._build_integral_splines()` exactly once.

```python
def __init__(self, ...):
    # ... existing attribute setup ...
    if neural_weight == 'cutoff':
        self._a = np.pi/3
        self._b = 4*np.pi/5
    elif neural_weight == 'vonmises':
        self._k = 2.0
    # ... remaining init (targets, theta_mesh, etc.) ...
    self._build_integral_splines()   # one build, final state
```

Post-construction, a user who updates both `pm.a` and `pm.b` in sequence pays for two rebuilds. At a few ms each that is acceptable for an interactive workflow; if it ever matters, a `set_cutoff_params(a, b)` convenience method can be added later.

**Out of scope**: `neural_weight` itself and `neural_angle` remain immutable post-init (changing them changes which spline, or whether any spline, exists). Document this explicitly in the docstring to avoid confusion.

### New evaluation paths

Add four small instance methods that wrap the splines and handle out-of-range saturation:

```python
def _neural_angle_cutoff(self, theta):
    theta = np.asarray(theta, dtype=float)
    scalar = theta.ndim == 0
    b = self._b
    clamped = np.clip(theta, -b, b)
    result = self._cutoff_forward_spline(clamped)
    result = np.where(theta >= b, np.pi, result)
    result = np.where(theta <= -b, -np.pi, result)
    return float(result) if scalar else result

def _neural_angle_cutoff_inverse(self, y):
    y = np.asarray(y, dtype=float)
    scalar = y.ndim == 0
    if np.any((y < -np.pi) | (y > np.pi)):
        raise ValueError("y must satisfy -pi <= y <= pi.")
    result = self._cutoff_inverse_spline(y)
    result = np.where(y == np.pi, self._b, result)
    result = np.where(y == -np.pi, -self._b, result)
    return float(result) if scalar else result

def _neural_angle_vonmises(self, theta): ...        # analogous; saturate at +/-pi
def _neural_angle_vonmises_inverse(self, y): ...    # analogous
```

### Docstring update

In `PerceptionModel.__init__`'s docstring (around [decision_model.py:405](decision_model.py#L405)), extend the existing "can be changed at any time as attributes" note to also cover `a`, `b`, `k`, and mention that reassigning them triggers an automatic rebuild of the integral splines. Explicitly note that `neural_weight` and `neural_angle` are NOT mutable post-init.

### Routing changes

- **`get_neural_angle`** [decision_model.py:1041](decision_model.py#L1041): replace `self._smooth_cutoff_integral(theta, self.a, self.b)` with `self._neural_angle_cutoff(theta)`, and `self._vonmises_integral(theta, self.k)` with `self._neural_angle_vonmises(theta)`.
- **`get_neural_angle_inverse`** [decision_model.py:1072](decision_model.py#L1072): analogous replacements for the two inverse branches.
- **`_integrate_neural_weight`** [decision_model.py:967](decision_model.py#L967): the cutoff and vonmises branches currently call the static `_smooth_cutoff_integral_scalar` / `_vonmises_integral` as antiderivatives inside a `sum(F(hi) - F(lo) for ...)` comprehension. Swap those for calls to `self._neural_angle_cutoff` / `self._neural_angle_vonmises`. The constant factor in the cutoff case was already absorbed into the antiderivative (via `norm = 2*pi/(a+b)`), and the new spline returns the same antiderivative, so the per-target ratios stay identical.

### Static-method cleanup (unrelated but adjacent)

After the refactor, the `np.vectorize`-based array wrappers `_smooth_cutoff_integral` and `_smooth_cutoff_int_inverse` have no remaining callers:

- All hot-path callers move to the spline.
- `test_intervals.py` lines 367, 368, 375, 376, 382, 383, 398, 399, 402, 403 all pass scalars to `PM._smooth_cutoff_integral(...)` — these calls continue to work if we rename the scalar kernel to take over the short name.

Changes:
- Remove `_smooth_cutoff_integral` (the vectorized wrapper at [decision_model.py:608-632](decision_model.py#L608-L632)).
- Remove `_smooth_cutoff_int_inverse` (the wrapper at [decision_model.py:674-699](decision_model.py#L674-L699)).
- Rename `_smooth_cutoff_integral_scalar` -> `_smooth_cutoff_integral`.
- Rename `_smooth_cutoff_int_inverse_scalar` -> `_smooth_cutoff_int_inverse`.
- Update the internal `func` closure inside the inverse to call the renamed forward method (one reference).
- `_vonmises_integral` and `_vonmises_int_inverse` stay as-is (they're already vectorized via scipy's C-level cdf/ppf, no scalar/wrapper split exists).

## Files to modify

- [decision_model.py](decision_model.py) — all edits listed above (add spline build, add property setters for a/b/k, add instance-level lookups, route high-level methods through splines, squash static-method wrappers, docstring update).

## Files to create

- `test_perception_spline.py` — new test, described below.

## Verification plan

### New test: `test_perception_spline.py`

Mirrors the style of `test_intervals.py` (plain script with `check_scalar`, exits non-zero on failure). For each spline configuration, compare the spline path against the reference quad/brentq/scipy path:

1. **Cutoff forward vs. reference**
   - Instantiate `pm = PerceptionModel(neural_weight='cutoff', neural_angle='integral')` (defaults give `a=pi/3`, `b=4*pi/5`).
   - Sample 5000 random `theta` in `[-pi, pi]` (seeded) plus fixed stress points: `0`, `+/-a`, `+/-b`, `+/-a*0.99`, `+/-b*0.99`, `+/-b*1.01`, `+/-pi`.
   - For each, compute `spline = pm._neural_angle_cutoff(theta)` and `ref = PerceptionModel._smooth_cutoff_integral(theta, pm.a, pm.b)` (renamed scalar reference).
   - Assert `max(|spline - ref|) < 1e-10`.

2. **Cutoff inverse vs. reference**
   - Sample 5000 random `y` in `[-pi, pi]`, plus `0`, `+/-a*norm`, `+/-pi`, `+/-(pi - 1e-12)`.
   - Compare `pm._neural_angle_cutoff_inverse(y)` vs. `PerceptionModel._smooth_cutoff_int_inverse(y, pm.a, pm.b)` (renamed scalar reference).
   - Assert `max(|spline - ref|) < 1e-10`.

3. **Cutoff roundtrip**
   - For 5000 random `x` in `[-b, b]`: `|_neural_angle_cutoff_inverse(_neural_angle_cutoff(x)) - x| < 1e-10`.
   - For 5000 random `y` in `[-pi, pi]`: the dual direction, same tolerance.

4. **Cutoff symmetry and endpoints**
   - `pm._neural_angle_cutoff(0.0) == 0.0` exactly.
   - `pm._neural_angle_cutoff(+/-b) == +/-pi` exactly.
   - `pm._neural_angle_cutoff(1.5 * b)` saturates to `pi` exactly.
   - `pm._neural_angle_cutoff(-x) ~= -pm._neural_angle_cutoff(x)` for sampled `x`, within 1e-12.

5. **Vonmises forward/inverse/roundtrip/endpoints** — analogous to 1-4, using `neural_weight='vonmises'` (default `k=2.0`) and `scipy.stats.vonmises.cdf` / `.ppf` as reference.

6. **`_integrate_neural_weight` invariance** — end-to-end check that the routing swap preserves behavior:
   - Build a small `Targets` scenario with a few circles, `pm = PerceptionModel(targets=..., neural_weight='cutoff')`.
   - Call `pm._get_target_signals()`; record `(c_angles, rho)`.
   - Temporarily monkey-patch `pm._neural_angle_cutoff = lambda t: PM._smooth_cutoff_integral(t, pm.a, pm.b)` to force the reference path; record the same output.
   - Assert both `rho` arrays agree to `1e-10`.
   - Repeat for `neural_weight='vonmises'`.

7. **Parameter sweep** — run the above accuracy checks for a handful of non-default `(a, b)` and `k` values to ensure the node count of 2001 holds up across the parameter range the user actually uses.

8. **Property-setter rebuild** — verify that mutating `a`, `b`, or `k` after construction rebuilds the splines:
   - Construct `pm = PerceptionModel(neural_weight='cutoff')`, record `id(pm._cutoff_forward_spline)`.
   - Assign `pm.a = 0.8`; confirm `id(pm._cutoff_forward_spline)` changed (rebuilt) and that `pm._neural_angle_cutoff(0.5)` now matches `PerceptionModel._smooth_cutoff_integral(0.5, 0.8, pm.b)` to 1e-10.
   - Repeat for `pm.b` and for the vonmises case (`pm.k`).
   - Verify that during `__init__` only one build occurs (instrument `_build_integral_splines` with a counter via monkey-patch, construct a model, assert the counter equals 1 — since defaults are written to the backing attributes, no setter-triggered builds should happen).

### Existing tests continue to pass

- [test_intervals.py](test_intervals.py) — all current asserts continue to hold after the static-method renames (it calls `PM._smooth_cutoff_integral(scalar, a, b)` which now resolves to the renamed scalar function with identical numeric output).
- [test_broad_validation.py](test_broad_validation.py) and [test_segments.py](test_segments.py) — should be unaffected; run them after the edit to confirm no downstream regressions.

### How to run

```bash
python test_intervals.py
python test_segments.py
python test_broad_validation.py
python test_perception_spline.py
```

All four should exit 0.
