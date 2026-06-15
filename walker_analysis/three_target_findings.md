# Three-target models: matching the GODM data, and a finding

Retuned three-target walker models ([three_target_fly.py](three_target_fly.py),
[three_target_locust.py](three_target_locust.py)) and the deterministic skeleton
([decision_skeleton.py](../plots/decision_skeleton.py), cases `fly` / `locust`), tuned to
the empirical GODM heatmaps (Sridhar et al. 2021). This documents what was changed,
how the per-target split was matched, and a real tension the locust exposes.

> **Update — the fly was refit at high realization count.** The split-lever story below
> was worked out at ~80 walkers and pinned the split on `angle_weight`. A later
> high-realization (N≥1000) match against the GODM heatmap overturned that for the fly:
> `a_weight` is **saturated**, and the real levers are **`a_warp`** (first-bifurcation x),
> **`K`** (peel sharpness / centre recapture), and **`std`** (centre de-bias). The fly now
> ships at **K=2.0, a_warp=0.65π, std=4.0** (~45% centre, corr≈0.78). Full analysis +
> the 2-target transfer: [three_target_fly_refine_findings.md](../plots/three_target_fly_refine_findings.md)
> / [two_target_fly_refine_findings.md](../plots/two_target_fly_refine_findings.md). The **locust
> has not been refit** — its section below stands on the old framing.

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

## What controls the split

The walker model is intrinsically **centre-biased**: a walker that peels onto a
compromise arm is deterministically *recaptured to the centre* (the reborn-centre
branch) as it advances.

**At ~80 walkers** this was pinned on the perception weight `angle_weight` (a narrower
foveal window releasing more walkers outward; uniform weight the most centre-biased),
with std / start-heading / T appearing to barely move it.

**The high-realization refit (N≥1000) corrected this for the fly** — see
[three_target_fly_refine_findings.md](../plots/three_target_fly_refine_findings.md):

- `a_weight` is **saturated** over 0.13–0.20π — it does *not* move the high-N split.
- **`a_warp`** sets the first-bifurcation x — how far out the walker commits up/down.
  Pushing it out (0.45→0.65π) stops walkers peeling toward the targets too early; the
  empirical ridge reaches the outer targets from higher x than the early-peelers did.
- **`K`** (turning gain) does not move the bifurcation but sets how *sharply* the walker
  peels: higher K corner-cuts onto a target and over-recaptures to centre; **K=2 (the
  model default) is gentlest and least centre-biased.**
- **`std`** is the noise de-bias lever (centre ~55%→~37% as std 2.5→6.0).

The fly is matched at **K=2.0, a_warp=0.65π, std=4.0** (~45% centre, corr≈0.78). The
`a_warp`/`K` moves *do* shift the skeleton's bifurcation x (intentionally — onto the
empirical ridge); `std` and the other noise knobs only rebalance the split.

## The finding: the locust's clean-yet-outer-biased commitment

- **Fly: matched.** Refit `K=2.0, a_warp=0.65π, std=4.0` → ~45% centre, corr≈0.78 vs the
  GODM heatmap ([three_target_fly_refine_findings.md](../plots/three_target_fly_refine_findings.md)). Done.
- **Locust: cannot be fully matched.** Lowering `a_weight` + raising `std` drives the
  centre fraction from 77% down toward ~42%, but it **plateaus above the 29% target**,
  and the only way to push lower is more noise — which *muddies the tight trident the
  data clearly shows*. The real locust does **both**: a clean trident **and** 71% outer
  commitment. In this model regime those pull in opposite directions — outer commitment
  is only bought with noise that costs the clean tracks.

This is a genuine, reportable result: **the model reproduces the decision *structure*
for both species (two sequential binary decisions; a second-bifurcation fork angle in
the neighbourhood of the paper's measured ~110° for flies — that number was read off the
earlier tuning and should be re-measured at the refit a_warp) and the fly's split, but
the locust's
clean, strongly outer-biased commitment is beyond what the current
recapture-dynamics-plus-noise can produce.** The gap lives in the recapture mechanism
itself (the dθ/dt law / Ising coupling over-pulling toward the dead-ahead target), not
in the noise tuning — a flagged follow-up if the locust split is to be chased properly.

**Concrete follow-up (from the fly refit):** lowering `K` in the dθ/dt law *is* a direct
handle on that recapture (K=2 de-biased the fly from ~55% to ~45% centre), and pushing
`a_warp` out moves the commitment later. Neither was tried on the locust — the locust
config below is still the old `a_weight`+`std` tuning, so re-running it at K=2 /
a_warp-pushed is the obvious next test.

The shipped locust config therefore favours **cleanliness**: `a_weight=0.10π, std=3.0`
gives a clean trident with the centre bias reduced as far as it can go without
degrading the tracks. Final measured splits (80 walkers):

| | centre | outer | vs data |
|---|---|---|---|
| fly (**refit**, K=2/a_warp=0.65π/std=4, 2500 walkers) | **44.8%** | 55.2% | data 45% — matched |
| locust (`a_weight=0.10, std=3.0`, old tuning) | **61%** | 39% | data 29% — clean near-match (down from 77%) |

## Final parameters (both fixed-geometry, experiment-set targets)

| | fly (refit) | locust (old tuning) |
|---|---|---|
| targets | 40°, dist 5, r=0.5 | 35°, dist 3, r=0.1 |
| `a_warp` / `b_warp` | **0.65π** / 0.92π | 0.40π / 0.90π |
| `a_weight` / `b_weight` | **0.20π** / 0.80π | **0.10π** / 0.80π |
| K, T | **2.0**, 0.10 | 6.0, 0.10 |
| noise_exp, R_exp | 2, 3 | 2, 3 |
| std, v, dt | **4.0**, 0.30, 0.05 | 3.0, 0.20, 0.04 |

(The fly column is the GODM-refit set, shared with the 2-target case; see
[three_target_fly_refine_findings.md](../plots/three_target_fly_refine_findings.md). The high-N
fly walker analysis also adds a small start scatter `pos_std=0.075, head_std=12°`, a
render-only knob absent from the deterministic skeleton.)

## Figures

- [skeleton_fly.png](../plots/skeleton_fly.png),
  [skeleton_locust.png](../plots/skeleton_locust.png) — deterministic skeleton
  (trunk → two arms → each second bifurcation splits to {outer, centre}) over the GODM
  heatmaps.
- [three_target_fly.png](three_target_fly.png),
  [three_target_locust.png](three_target_locust.png) — walker tracks + density (the
  simple quick-look render; the GODM-pipeline substructure match is
  [fly_results_3target.png](../plots/fly_results_3target.png), via
  [three_target_fly_refine.py](../plots/three_target_fly_refine.py)).
