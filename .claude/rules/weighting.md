---
paths:
  - "weighting_analysis/**"
---

# Weighting vs warping — the "ears"

Auto-loads under `weighting_analysis/`. Full write-up: [weighting_analysis/README.md](../../weighting_analysis/README.md).

One-line takeaway: now that warp and weight are decoupled, **warping alone reproduces the bifurcation structure of full weighting except for two "ears"** of extra far-target bistability at off-axis observer positions behind two circle targets — present under a non-uniform `angle_weight`, absent under uniform (`angle_weight=None`, the default). The ears are therefore now opt-in.

The README also covers the delta-target threshold shift, a delta+ANGLE Hopf follow-up (Hopf-unstable foci but no limit cycle), and the cutoff blind-spot trap.

**Vocabulary note:** the README still uses the pre-decouple `neural_weight`/`weight_angle_only` vocabulary. Map it via the Old→new table in [.claude/rules/perception-and-solver.md](perception-and-solver.md) (`neural_weight=W, neural_angle='integral'` → `neural_angle_dist=W, angle_weight='neural_angle_dist'`; `weight_angle_only=True` → `angle_weight=None`).
