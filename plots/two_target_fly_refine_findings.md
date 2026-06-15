# Fly two-target: does the 3-target parameterization transfer? (GODM fly2)

Companion to [three_target_fly_refine_findings.md](three_target_fly_refine_findings.md).
[two_target_fly_refine.py](two_target_fly_refine.py) runs the identical fly
parameterization (imported from the 3-target module — single source of truth) on the
2-target geometry (GODM `fly2`: 60° separation, distance 5 → targets at (4.330, ±2.5),
r=0.5) and renders it through the same GODM max-projection pipeline matched to `fly2`
(extent (0, 4.33, −2.5, 2.5), blur 201, **mirror=False** — fly2 is not y-mirrored).
The hypothesis: same species, same setup, two targets instead of three → no re-tuning.

## Result: the shared parameterization transfers (after the bifurcation refit)

The first attempt used the original refine knobs (K=3.5, a_warp=0.45π) and transferred
only *partially*: right topology (single trunk → one bifurcation → two branches, clean
symmetric split) but a weak occupancy match (corr(all)=**0.572**), because the walker
peeled off **too early** and fanned wider than the empirical ridge.

That same early-peeling was visible in the 3-target outer branches too, and refitting
**K 3.5→2.0 and a_warp 0.45→0.65π** (detailed in
[three_target_fly_refine_findings.md](three_target_fly_refine_findings.md)) fixes both
cases with a *single shared* parameterization — no 2-target-specific tuning.

**2500 realizations, shared refit params** (K=2.0, T=0.10, σ=4.0, warp 0.65π/0.92π,
weight 0.20π/0.80π, q=2/p=3, v=0.30, dt=0.05, start jitter 0.075/12°):

| config | split | corr(all) | corr(support) |
|---|---|---|---|
| 2-target, original (K=3.5, a_warp=0.45π) | 49.2 / 50.8 | 0.572 | 0.508 |
| **2-target, refit (K=2.0, a_warp=0.65π)** | 49.5 / 50.5 up/down | **0.690** | **0.646** |
| 3-target, refit (same knobs), for reference | 44.8% centre | 0.781 | 0.760 |

## Why the refit works — bifurcation x + peel sharpness

The deterministic midline cascade (`sc_equilib` along y=0) put the model's 2-target
bifurcation — where the straight-ahead 0° branch dies and the walker commits up/down —
at **x≈1.8** under the old a_warp=0.45π, while the empirical GODM fly2 ridge keeps a
**long narrow trunk** and splits near **x≈2.7**. Raising a_warp to 0.65π pushes the
model's split out to **x≈2.2**, and lowering K to 2.0 gentles the peel so walkers ride
the trunk longer and arrive along the ridge instead of cutting the corner. Both moves
shift the walker density right, onto the empirical trunk — corr 0.572→0.690.

The 2-target still splits a touch earlier than the flies (x≈2.2 vs 2.7) — its trunk is
not quite as long as the empirical — so its fit (0.69) stays a bit below the 3-target's
(0.78). Pushing a_warp higher than 0.65π would lengthen it further but starts to flatten
the 3-target gain; 0.65π is the shared sweet spot.

## Conclusion

The hypothesis holds: **the same fly parameterization fits both the 2- and 3-target
GODM heatmaps**, once the first bifurcation is placed correctly (a_warp) with a gentle
turning gain (K=2.0). The fix was not a 2-target hack — it improved the 3-target match
too. Residual: the 2-target empirical trunk is slightly longer than the model produces.

## Reproduce

```
NR_REPS=2500 python plots/two_target_fly_refine.py   # writes two_target_fly_refine.npz + .png
python plots/fly_results.py 2                        # -> fly_results_2target.png (300 dpi, >=4in)
```
