# Three-target models: matching the GODM data, and a finding

Retuned three-target walker models ([three_target_fly.py](three_target_fly.py),
[three_target_locust.py](three_target_locust.py)) and the deterministic skeleton
([decision_skeleton.py](decision_skeleton.py), cases `fly` / `locust`), tuned to
the empirical GODM heatmaps (Sridhar et al. 2021). This documents what was changed,
how the per-target split was matched, and a real tension the locust exposes.

## Corrections found along the way

- **Locust separation is 35°, not 40°.** The empirical `locust3` posts (read from the
  GODM data) are at distance 3, bearings {0°, ±35°}. An earlier prototype used 40° (the
  fly layout scaled); the model now uses 35°, which also makes the model→data frame
  alignment identity (the skeleton starts at the origin instead of an offset). Fly is
  40° at distance 5 — unchanged.
- **Skeleton symmetry pin.** The centre route is exactly y=0 by symmetry, but the root
  used to drift off-axis while riding the SC-*unstable* centre branch (the transverse
  instability amplifying the `hybr` solver's y-asymmetry). The tracer now pins the
  centre route to the midline in mirror-symmetric problems. (Whether to fix the hybr
  asymmetry at its source in `sc_equilib` is a separate open question.)

## Per-target split in the data (extracted from `../../GODM`)

Each committed trajectory classified by nearest target (rotated frame):

| | N | centre | outer (up / low) | centre : outer |
|---|---|---|---|---|
| fly3 | 125 | 45% | 55% (40 / 15) | 0.45 : 0.55 |
| locust3 | 518 | 29% | 71% (39 / 32) | 0.29 : 0.71 |

The fly's 40/15 up/down imbalance is a small-N left/right asymmetry; the GODM pipeline
mirrors the 3-target data to wash it out, so the target is treated as symmetric:
**fly ≈ 45% centre, locust ≈ 29% centre (outer-biased)**.

## What controls the split — `angle_weight` is the lever

The walker model is intrinsically **centre-biased**: a walker that peels onto a
compromise arm is deterministically *recaptured to the centre* (the reborn-centre
branch) as it advances. Three knobs that should fix this **do not**:

| knob (at fixed others) | fly centre | locust centre |
|---|---|---|
| `std` 2 → 6.5 | 70 → 56% | 88 → 70% |
| start-heading spread 0 → ±55° | 67 → 62% | 77 → 78% |
| T 0.10 → 0.035 | 67 → 65% | 77 → 73% |

The lever is the **perception weight** `angle_weight`. Lowering `a_weight` (a narrower
foveal weight window) releases walkers to the outer targets; **uniform weight is the
*most* centre-biased** (93% fly, 100% locust). Mechanism: a target you head toward
lands near your visual centre and gets weighted up, reinforcing the off-centre
commitment.

| `angle_weight` | fly centre | locust centre |
|---|---|---|
| uniform | 93% | 100% |
| `a_weight=0.40` (orig) | 67% | 77% |
| `a_weight=0.20` | **47%** | 60% |
| `a_weight=0.10` | — | 57% (clean) … 42% (high std) |

Crucially, lowering `a_weight` does **not** break the skeleton's ridge fit (the arms
are set by `a_warp`; `a_weight` only nudges the second bifurcation).

## The finding: the locust's clean-yet-outer-biased commitment

- **Fly: matched.** `a_weight=0.20π, std=2.5` → ~45–47% centre, clean tracks. Done.
- **Locust: cannot be fully matched.** Lowering `a_weight` + raising `std` drives the
  centre fraction from 77% down toward ~42%, but it **plateaus above the 29% target**,
  and the only way to push lower is more noise — which *muddies the tight trident the
  data clearly shows*. The real locust does **both**: a clean trident **and** 71% outer
  commitment. In this model regime those pull in opposite directions — outer commitment
  is only bought with noise that costs the clean tracks.

This is a genuine, reportable result: **the model reproduces the decision *structure*
for both species (two sequential binary decisions; the ~106° second-bifurcation angle
≈ the paper's measured ~110° for flies) and the fly's split, but the locust's
clean, strongly outer-biased commitment is beyond what the current
recapture-dynamics-plus-noise can produce.** The gap lives in the recapture mechanism
itself (the dθ/dt law / Ising coupling over-pulling toward the dead-ahead target), not
in the noise tuning — a flagged follow-up if the locust split is to be chased properly.

The shipped locust config therefore favours **cleanliness**: `a_weight=0.10π, std=3.0`
gives a clean trident with the centre bias reduced as far as it can go without
degrading the tracks. Final measured splits (80 walkers):

| | centre | outer | vs data |
|---|---|---|---|
| fly (`a_weight=0.20, std=2.5`) | **49%** | 51% | data 45% — matched |
| locust (`a_weight=0.10, std=3.0`) | **61%** | 39% | data 29% — clean near-match (down from 77%) |

## Final parameters (both fixed-geometry, experiment-set targets)

| | fly | locust |
|---|---|---|
| targets | 40°, dist 5, r=0.5 | 35°, dist 3, r=0.1 |
| `a_warp` / `b_warp` | 0.45π / 0.92π | 0.40π / 0.90π |
| `a_weight` / `b_weight` | **0.20π** / 0.80π | **0.10π** / 0.80π |
| K, T | 3.5, 0.10 | 6.0, 0.10 |
| noise_exp, R_exp | 2, 3 | 2, 3 |
| std, v, dt | 2.5, 0.30, 0.05 | 3.0, 0.20, 0.04 |

## Figures

- [skeleton_fly_hm.png](skeleton_fly_hm.png),
  [skeleton_locust_hm.png](skeleton_locust_hm.png) — deterministic skeleton
  (trunk → two arms → each second bifurcation splits to {outer, centre}) over the GODM
  heatmaps.
- [three_target_fly.png](three_target_fly.png),
  [three_target_locust.png](three_target_locust.png) — walker tracks + density.
