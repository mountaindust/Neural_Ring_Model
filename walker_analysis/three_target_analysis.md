# Three-target walkers and the center-target bias

Why three-target walkers in `NeuralBandModel.plot_walkers` consistently choose the
**middle** target instead of producing a clean double bifurcation to all three —
and how to recover an arbitrary (e.g. 1:2:1) split while keeping the bifurcation
locations fixed by the data.

## Setup this analysis refers to

The layout from `neural_band_walker.ipynb` (locust/fly reconstruction):

- Three `circle` targets (`r=0.3`) at allocentric bearings **0°, +40°, −40°**, all
  at radius 5: `[[5,0],[3.8302,3.2139],[3.8302,-3.2139]]`.
- Walker starts at the **origin facing 0°** — i.e. *facing the center target head-on*.
  This is the crux of everything below.
- Perception: `neural_angle_dist='lin_cutoff'` (`a_warp=0.25π, b_warp=0.9π`),
  `angle_weight='lin_cutoff'` (`a_weight=0.25π, b_weight=0.30π`, a ±54° visual
  window). `K=4.5`, `T=0.2`, `v=1`, `std=0.025`, `dt=0.1`.

The bifurcation *x-locations* are fixed by experiment (and set with the
`a_warp`/`b_warp` knobs): for locusts the first bifurcation is at x≈0.7–1.2 and
the second at x≈2; for flies, first at x≈1–1.5, second at x≈3.5. **None of the
fixes below move those locations** — they only rebalance which target wins.

With the notebook settings, an endpoint census over 60–80 seeds is **100% center**
(`center/upper/lower = 80/0/0`). The goal is to understand that and reach a
controllable split (the user's target is **1:2:1** top:center:bottom).

## What is actually happening (the bifurcation mechanism)

The walker rides outward roughly along the midline (y≈0) because it starts facing
the center target. Probe the self-consistent equilibria along that midline
(`sc_equilib(focal_loc=[x,0])`, allocentric stable consensus directions) and the
structure goes through this cascade:

| x (along midline) | # stable | stable consensus directions (deg) |
|---|---|---|
| 0.5 – 1.5  | **2** | ±22°  — *straight-ahead (0°) is UNSTABLE here* |
| 1.75 – 2.25 | **5** | −57, −25, **0**, +25, +57 |
| ≥ 2.5      | **3** | −67, **0**, +67 |

Read top to bottom, this is the whole story:

1. **First bifurcation (x≈0.5–1.5).** The straight-ahead consensus is unstable;
   only the two "pair-compromise" directions ±22° are stable. The walker must
   peel up or down. So far, so good — this is the left/right split you want.

2. **The center branch is *reborn* (x≈1.7).** A brand-new stable branch at **0°**
   (pointing dead at the center target) is created — together with the ±57°
   outer-target branches — only a short distance past where the up/down split
   began. The walker has had almost no room to diverge from the midline in that
   gap, so it is **recaptured by the reborn 0° branch** and homes on center.

3. **Compromise branches die (x≈2.5).** The ±25° branches annihilate, leaving
   `{center 0°, upper +67°, lower −67°}`. The walker is already locked on 0°.

The decisive event is step 2: **the center target's own consensus branch
reappears directly in the walker's path before the up/down commitment carries it
clear of the midline.** A target sitting on the symmetry axis *is* the
"go-to-the-average" direction, so the average attractor and the center target are
the same object.

### Corroborating detail

- **Deterministic basin watershed.** With noise off, a walker launched with
  initial heading ≤18° lands on **center**; ≥20° lands on **upper**. The watershed
  is ~19°. The +22° compromise branch sits just above it, but a walker approaching
  from the origin (facing 0°) curves up onto that branch from *below* and drifts to
  y just under the +20° ray — i.e. it sits on the **center side** of the watershed
  and never crosses it. Hence center, deterministically.

- **The Ising gate does the switching.** Tracing an actual walker, at the second
  bifurcation the sigmoid gate `1/(1+exp(−2NR·cos(Θ−θ_i)/T))` on the *outer*
  target collapses to 0 and the consensus snaps back to the center target. The
  outer target is switched off, not the center.

### What it is *not*

- **Not the third (far) target's pull.** Once the walker commits up, the lower
  target is suppressed twice over — by the Ising gate (anti-aligned → gate≈0) and
  by falling outside the ±54° `angle_weight` window. It contributes ≈nothing at the
  second bifurcation. `angle_weight` is a fixed *egocentric* window with no notion
  of "the pair under consideration," so it cannot gate out "the target not in this
  pair"; narrowing/widening it only relocates the bifurcations.

- **Not the warp.** Repeating with identity warp (`neural_angle_dist=None`) keeps
  the center bias (watershed still ~15–20°). The bias is geometric/dynamical, born
  of the initial condition (facing the center target), not of the neural mapping.
  (Removing the weight window entirely — uniform weight — makes it *worse*: every
  walker goes center even when launched at +30°, because all three targets stay
  visible and the average is pinned to center.)

## Options for overcoming it

The bifurcation *locations* (warp/weight) are pinned by the data, so they cannot
be used to fix the skew. The skew is a **basin-size asymmetry** set by the walker
starting out facing the center target, and it is rebalanced only by the
**dynamical / noise knobs**: `std`, `v`, `K`, `T`, and the noise law (`noise_exp`).

Restating the target as a number: **1:2:1 (top:center:bottom) is exactly "the
second bifurcation is a fair 50/50 coin."** Half the walkers go up, half down (by
symmetry); on each branch a fair coin sends 50% to the outer target and 50% back
to center, netting 25% top / 50% center / 25% bottom. So the job is to make the
center-vs-outer choice on each branch unbiased.

Empirically, the center fraction tunes smoothly from heavily-center-skewed to
even. Constant-noise census (default warp/weight, 80 seeds):

| std | center / upper / lower | center : outer |
|---|---|---|
| 0.025 | 80 / 0 / 0 | ∞ |
| 0.10 | 74 / 4 / 2 | ~25 |
| 0.20 | 63 / 8 / 9 | 7.4 |
| 0.40 | 49 / 17 / 14 | 3.2 |
| 0.80 | 35 / 26 / 19 | 1.6 |

Lowering `v` and `K`, and (slightly) `T`, all push further toward even at fixed
`std`. The 1:2:1 point is reachable several ways; two clean ones are below.

## Precise parameterizations that work

Both use the data-fixed geometry and warp/weight above; only the dynamical knobs
change. Ratios quoted as **top : center : bottom**; up/down differences of a few
counts are seed sampling, not a real asymmetry.

### Route 1 — constant noise, dt-robust, lands ~1:2:1

```
K = 4.5,  T = 0.2,  v = 0.3,  std = 0.4,  noise_exp = 0,  dt = 0.1
```

Converged census ≈ **1.1 : 2.1 : 0.9** and stable under dt refinement (see below).
Tracks are visibly noisier but both bifurcations and the three-way split are
clearly present. This is the safe choice when a defensible ratio with minimal dt
fuss is the priority.

### Route 2 — gated noise, cleanest-looking double bifurcation

```
noise_exp ≈ 1,  std ≈ 0.6,  v ≈ 0.5,  K = 4.5,  T = 0.2,  dt ≈ 0.025
```

The gated law (`σ·(1−R)^p·cos(Θ/2)`) explores hard while undecided (low R, on the
midline) and quiets on commitment (R→1), so committed tracks are tight and the
double bifurcation reads cleanly — the textbook picture: tight bundle → one split
into three streams → low-wiggle homing.

Trade-off (as originally written, with the **coupled** default `R_exp = 1/q`):
gating shuts the noise off right at the second bifurcation (R≈0.5), so it ran
**more center-heavy** and **dt-sensitive**. **This is superseded by the decoupled
analysis** — see [gated_pq_analysis.md](gated_pq_analysis.md). The drift exponent
`p = R_exp` and the gate exponent `q = noise_exp` are now independent knobs (the
model default is `R_exp = 1`, not `1/q`), and the earlier "keep `p≈1`" advice was
wrong: **raising `p` toward/above `q` (e.g. `q=2, p=3`) cancels the steep-gate
center bias *and* makes the ratio dt-robust** (`SNR(0.5) ∝ 2^(q−p)`). The clean
recipe is `q≈1.5–2` (kills circling — capture 1.0, tortuosity ~1.05), raise `σ`
to de-skew (the gate keeps it clean where constant noise would circle), and `p≈2–3`
to fix the bias and converge. The two example scripts use this (`q=2, p=3, σ≈2.75`).

Use Route 2 (gated, decoupled) when clean, loop-free tracks matter for matching
the empirical heatmaps; use Route 1 (constant) when you specifically need constant
noise and can accept a center-heavier, slightly looser result.

## How each knob moves the center fraction

All of these **leave the bifurcation x-locations untouched** (those are set by
geometry + warp/weight). They only rebalance the basins.

| knob | center share | mechanism / notes |
|---|---|---|
| **std** ↑ | ↓ (toward even) | most direct lever; too high smears the bifurcations together |
| **v** ↓ | ↓ | slower walker accumulates more heading diffusion per unit distance through the bifurcation zone; does **not** move bifurcation x |
| **K** ↓ | ↓ | K=2 overshoots to ~1:1.4:1; K=8 worsens the skew |
| **T** ↓ | ↓ slightly | sharper Ising gate, marginally larger spread |
| **noise_exp** (`q`, gate) ↑ | mild ↑ on its own | `q≥1` is the **circling fix** (noise→0 on commitment → no loops, capture 1.0); steeper gate alone nudges center up — offset it with `p`/`σ`. See [gated_pq_analysis.md](gated_pq_analysis.md). |
| **R_exp** (`p`, drift) ↑ | ↓ (fine-tune) + dt-robust | decoupled from `q` (default now `1`); raising `p` toward/above `q` cancels the steep-gate center bias and stabilizes the ratio vs `dt`. Effect ~0.05–0.10 in center fraction — a fine-tune, not a big lever. |
| **a_warp/b_warp, a_weight/b_weight** | — (moves *locations*, not skew) | use these only to place the first/second bifurcations at the data-observed x |

Why warp/weight do not appear as skew knobs: they set *where* the cascade above
happens, not the relative basin sizes at the second bifurcation. The skew is the
initial-condition asymmetry (facing the center target) and is rebalanced only by
the dynamical/noise knobs.

## How `dt` must respond when you move these parameters

The Euler–Maruyama heading update is

```
θ ← θ + K·R^{R_exp}·sin(Θ/2)·dt  +  σ_eff·cos(Θ/2)·√dt·Z
```

- **The drift step is `K·R·sin(Θ/2)·dt`, so the quantity that must stay small is
  `K·dt`** — keep the per-step turn ≲0.1–0.2 rad. At `K=4.5, dt=0.1` that is ≲0.2
  rad (borderline-OK). **If you raise `K`, lower `dt` proportionally**
  (`K·dt ≈ const`): e.g. `K=9` ⇒ `dt≈0.05`.
- **`v` does not affect heading accuracy.** Lowering `v` shrinks the *spatial* step
  `v·dt` (which only helps target detection — `target_tol=v·dt` — and trajectory
  resolution). Heading accuracy is governed by `K·dt` and the `σ·√dt` diffusion
  increment, both independent of `v`. So changing `v` for the ratio does not
  require changing `dt`.
- **The diffusion term is already the correct `√dt` Wiener increment** (per-unit-
  time angular variance independent of `dt`); raising `std` does not by itself
  demand a smaller `dt`. The drift, above, is the binding constraint.
- **Convergence is the real test.** A correctly-resolved ratio should not change as
  `dt` shrinks. Route 1 is converged (center ratio ~2.1–2.4 across dt 0.1→0.025).
  The gated law was dt-sensitive *only under the old coupled `R_exp=1/q`* (boosted
  `R^{1/q}` drift inflates the per-step turn); **decoupling `R_exp` upward (p≈2–3)
  both de-skews and converges** — `K·R^p·dt` shrinks with `p` (R<1), so the Euler
  error drops. The example scripts (`q=2, p=3`) are dt-robust (center fraction
  unchanged 0.05→0.025). See [gated_pq_analysis.md](gated_pq_analysis.md) Result 4.
  If a ratio moves with `dt`, you are under-resolved — shrink `dt` or raise `p`.

dt-convergence data (120 seeds, total integration time held fixed):

| config | dt=0.1 | dt=0.05 | dt=0.025 |
|---|---|---|---|
| const K4.5 std0.4 v0.3 | 1.14 : 2.14 : 0.86 | 1.10 : 2.14 : 0.90 | 1.05 : 2.36 : 0.95 |
| gated K4.5 std0.6 p1 v0.5 | 1.22 : 2.90 : 0.78 | 1.02 : 1.81 : 0.98 | 0.94 : 2.90 : 1.06 |

## Worked examples: `three_target_fly.py` and `three_target_locust.py`

> **Note:** the centre-skew analysis below predates the GODM-data match. The shipped
> scripts were since retuned (lower T, higher K, the foveal weight `a_weight` as the
> split lever, and the **locust corrected to 35°**), and the data turned out to be
> *outer*-biased for the locust — see [three_target_findings.md](three_target_findings.md).

Two runnable scripts in this folder apply the above to the experimental layouts:
three targets 40° apart, **fly** at radius 5 with **target radius 0.5** (diameter
1), **locust** at radius 3 with **target radius 0.1** (diameter 0.2). The target
sizes differ (from the paper's supplementary; the main figures misleadingly draw
them equal), which **breaks the radius scale-invariance** — so the two scripts use
*different* warps, each re-tuned at its own target size:

| | target r | a_warp | b_warp | first bif | second bif |
|---|---|---|---|---|---|
| fly | 0.5 | 0.47π | 0.92π (~±14° blind spot) | x≈1.5 | x≈3.5 |
| locust | 0.1 | 0.46π | 0.90π (~±18° blind spot) | x≈0.9 | x≈2.0 |

Both push the **first** bifurcation outward to the data-observed x with a wide
neural plateau (`a_warp≈0.46–0.47π`) and a near-panoramic weight window
(`a_weight/b_weight = 0.40π/0.80π`); `b_warp` sets the second-bifurcation location
(and doubles as the rear blind spot). The late first bifurcation means a single
strong central attractor, so the split needs real de-skewing noise — which under
**constant** noise makes walkers overshoot and **loop around the far side of the
targets**, a pathology the empirical heatmaps never show.

The scripts therefore use the **decoupled gated law** (see
[gated_pq_analysis.md](gated_pq_analysis.md)):

```
K = 2.0,  T = 0.2,  noise_exp (q) = 2,  R_exp (p) = 3,  dt = 0.05
fly:    v = 0.3,  std (σ) = 2.5,  target_tol = 0.2
locust: v = 0.2,  std (σ) = 2.0,  target_tol = 0.2
```

- `q = 2` (steep gate): noise → 0 on commitment, so the final approach is noise-free
  — this is what removes the *circling/orbiting* the constant law produced at a
  de-skewing σ.
- `p = 3` (drift exponent, decoupled up from the old `1/q`): cancels the steep
  gate's center bias and makes the ratio **dt-robust** (center fraction unchanged
  0.05→0.025).
- `σ` is set to **match the empirical heatmap density**, not a fixed ratio (the
  1:2:1 figure was only a guess); lower σ → more center-dominant, higher → more even.
  The locust uses a tighter σ to match its tighter empirical triangle.
- **Low `v` (with a small arrival radius `target_tol`) is the loop fix on the final
  approach.** The turn radius is ~`v/(K·R)`; setting `v` so that radius approaches
  the target size makes an off-axis approach curve *into* the target instead of
  swinging past it. The fly's large r=0.5 targets tolerate `v=0.3`; the locust's
  tiny r=0.1 targets need `v=0.2` (turn radius ~0.1 ≈ the target). A *large* arrival
  radius is the wrong fix for tiny targets — it leaves tracks stopping short in a
  visible crescent.

This yields tight, three-way branching that matches `godm_heatmap_fly3.png` /
`godm_heatmap_locust3.png` with no circling. (A residual ~2% of locust walkers
still make a single curved fly-by of the tiny center target — negligible in the
occupancy density, and distinct from the constant-noise orbits that were the
original problem.) As before, the bifurcation *locations* are untouched by
`K`/`v`/`σ`/`p`/`q`, so the warp tuning and the noise tuning are independent.

## Diagnostic snippets

**Midline cascade** — where the bifurcations sit and where the 0° center branch is
reborn (push that rebirth as far out as the data allows; watch the `n_stable`
sequence 2 → 5 → 3):

```python
import numpy as np
for x in np.linspace(0.3, 4.5, 22):
    angles, stab = neur_model.sc_equilib(focal_loc=np.array([x, 0.0]),
                                          stability_criterion='reduced')
    angles = np.degrees(np.asarray(angles)); stab = np.asarray(stab, bool)
    print(f"x={x:.2f}  n_stable={stab.sum()}  dirs={np.round(np.sort(angles[stab]),1)}")
```

**Endpoint census** — tune the ratio against this (flip `noise_exp`/`R_exp`/`v`/
`K`/`std`; needs `target_locs` and `model` in scope). `noise_exp` (q) and `R_exp`
(p) are independent, matching the model default `R_exp=1`; for the gated regime use
e.g. `noise_exp=2, R_exp=3`:

```python
import numpy as np
def census(nm, pm, targets, std, noise_exp, v, R_exp=1.0, dt=0.05,
           walk_std=np.pi/2, nseed=150):
    c = {-1:0, 0:0, 1:0, 2:0}
    for s in range(nseed):
        nm.rng = np.random.default_rng(s)
        pm.focal_loc = np.array([0.0,0.0]); pm.focal_angle = 0.0; nm.gamma = complex(1e-5)
        for _ in range(int(80/dt)):
            if np.any(targets.get_dist_to_targets(pm.focal_loc) < v*dt): break
            neur,_ = pm.get_neural_signals()
            if neur.size == 0:
                nm.gamma = 0j; dtheta = 0.0; sig = walk_std
            else:
                dtheta = nm.dtheta_dt(); R = abs(nm.gamma)
                sig = std*np.clip(1-R, 0, 1)**noise_exp
                if R > 0:
                    sx = dtheta/(nm.K*R)
                    if noise_exp != 0: sig *= np.sqrt(max(0,1-sx*sx))   # cos(Θ/2)
                    if R_exp != 1:     dtheta = nm.K*R**R_exp*sx        # R^p drift
            ang = pm.focal_angle + dtheta*dt + (sig*nm.rng.normal()*np.sqrt(dt) if sig>0 else 0)
            pm.focal_loc = pm.focal_loc + v*dt*np.array([np.cos(ang), np.sin(ang)])
            pm.focal_angle = model.convert_angles(ang)
        d = np.hypot(*(target_locs.T - pm.focal_loc[:,None])); i = d.argmin()
        c[i if d[i] < 0.5 else -1] += 1
    return c   # c[0]=center, c[1]=upper, c[2]=lower, c[-1]=none
```
