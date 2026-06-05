# Foveal `angle_weight` decision (2026-06): keep uniform, treat delta as singular

**Decision: do _not_ adopt a foveal (vonMises) `angle_weight`. Keep uniform
weight (`angle_weight=None`, the current default).** The `1/N` "decision
paralysis on commitment" pathology is an artifact of the *exact delta* target,
not of physical (finite-radius) targets, and a foveal weight buys a marginal
commitment-signal lift at the cost of the far-target bistability "ears"
(documented in [README.md](README.md), "Why the ears exist").

This re-ran the analysis under the **current** model (`dθ/dt = K·R·sin(arg γ/2)`,
`K=2`, decoupled warp/weight; the README still uses the pre-decouple
`neural_weight`/`weight_angle_only` vocabulary — map via the CLAUDE.md Old→new
table). Setups: cutoff warp `a=0,b=π`, `angle_weight` uniform vs vonMises;
2- and 4-circle targets at `(4.33, ±2.5)` / `(4.33, ±{0.75,2.25})`.

## Why uniform is fine — the four findings

1. **Finite targets are not pathological.** Committed `R` (observer just outside
   the target surface, facing it) under uniform weight:

   | target radius r | committed R |
   |---|---|
   | **0.0 (delta)** | **0.250 = 1/N** |
   | 0.001 | 0.951 |
   | 0.01 | 0.938 |
   | 0.1 | 0.864 |
   | 0.5 | 0.790 |

   Any finite radius gives `R ≈ 0.79–0.95` near contact — the noise gate
   `(1−R)^p` closes fine. Only the zero-extent delta collapses to `1/N`.

2. **The delta is a *singular* limit, not the limit of small circles.** There is
   a discontinuity at `r=0`: `R` jumps `0.250 → 0.951` between `r=0` and
   `r=0.001`. Mechanism: with uniform weight `ρ ∝ angular extent`; the close
   target's extent dominates the far ones by a ratio `≈ distance/(r+margin)`
   that is *independent of r* as `r→0`, so `R` stays high. But a *point* has no
   extent, so uniform weight gives every delta the same mass `1/N`. Extent-
   proportional vs equal-mass: that is the jump.

3. **Approach-phase `R` is U-shaped and the dip is multi-stable.** Along the
   approach to a target, `R` is high far out (centroid compromise, ~0.74), dips
   to ~0.43 in the **multi-stable decision zone** (2–4 coexisting stable
   consensus directions), and rises to ~0.86 at contact (one target dominates by
   extent). So the noise gate is most open exactly where the decision is
   genuinely ambiguous and closes on arrival — the desired behavior, for free,
   under uniform weight. See `walker_analysis/dip_vs_bifurcation.png`.

4. **Foveal lift is marginal; the ears are the cost.** A vonMises `angle_weight`
   (`k=0.5`) lifts committed `R` by only **~0.03–0.05** across the whole
   approach (e.g. `r=0.1`: 0.864→0.905; delta floor 0.25→0.31), while the
   far-target bistability "ears" grow monotonically with concentration
   (excess-over-uniform 2-stable area ≈ 0.56 at `k=0.5`; ear-area table in
   README.md). Small benefit, real structural cost.

## What this means
The explore↔commit dynamics are handled by the **noise model** (the `(1−R)^p`
gate, the `cos(Θ/2)` modulation, and the `R^{R_exp}` drift exponent — see
CLAUDE.md "Euler-Maruyama heading step"), not by the perception weighting.
Uniform weight already gives faithful committed `R` for any physical target.
The delta's `1/N` is recorded as a **singular idealization** (zero extent +
uniform weight) that real targets do not share; no foveal weight is adopted.
If a future need for foveal target-weighting arises on perceptual grounds, the
ceiling on concentration is the ears onset (README.md).
