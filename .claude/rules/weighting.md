---
paths:
  - "weighting_analysis/**"
---

# Weighting vs warping — the "ears"

Auto-loads under `weighting_analysis/`. Full write-up: [weighting_analysis/README.md](../../weighting_analysis/README.md).

One-line takeaway: now that warp and weight are decoupled, **warping alone reproduces the bifurcation structure of full weighting except for two "ears"** of extra far-target bistability at off-axis observer positions behind two circle targets — present under a non-uniform `angle_weight`, absent under uniform (`angle_weight=None`, the default). The ears are therefore now opt-in.

The README also covers the delta-target threshold shift, a **retracted** delta+ANGLE Hopf follow-up (it used the removed `'coupled'` criterion; its retraction section carries the triage rule — `eig(A)` and `sign(det J)` are invariant under the reduction error, so only complex-pair/Hopf classifications anywhere in the folder are affected), and the cutoff blind-spot trap (**resolved** — its "torque death" step describes the pre-half-angle `sin(ego)` law; the README section now carries a status banner).

**Vocabulary note:** the README's tables and prose still say **FULL** / **ANGLE-only**; read those as *weight tied to the warp* / *uniform weight*. It now carries its own Old→new mapping table near the top (see also [.claude/rules/perception-and-solver.md](perception-and-solver.md)).

**Figures are reproducible again.** `ears_figure.png` / `ear_diagnostic.png` were originally built by throwaway scripts that were never committed; [weighting_analysis/ears_figure.py](../../weighting_analysis/ears_figure.py) now rebuilds both (cached in `_cache_ears.npz`). They were regenerated 2026-08 under the current model — `stability_criterion='reduced'` (was `'coupled'`) and after the wrapping-extent fix, which mattered here because the ears live close to a target under **uniform** weight, the one weighting whose support reaches the rear. Pre-regeneration copies are kept as `*_old.png`.

**Anti-foveal follow-up:** whether a *centre-dip* weight can bias the observer outward (the locust question) is [weighting_analysis/outward_bias.md](../../weighting_analysis/outward_bias.md), script `outward_bias.py`. Answer: **no**, twice over — (1) an egocentric weight suppresses whatever is dead ahead, so it penalizes centre and outer commitments alike and kills the fragile outer branches first (they annihilate entirely for `m ≤ 0.5`); (2) spreading ρ lowers `R`, and the gated walker drift `K·R^{R_exp}` collapses with it, so the ensemble narrows instead of fanning out. Walker census 100% centre vs the data's 29%. Rule of thumb it establishes: **concentrate the weight, don't spread it** — the same direction the foveal commitment-signal argument in [TODO.md](../../TODO.md) wants.

**The `lin_dip` / `lin_ring` families are NOT in the model.** They were added to `PerceptionModel` for that analysis and removed when it came back negative; [weighting_analysis/anti_foveal.py](../../weighting_analysis/anti_foveal.py) preserves them verbatim and re-registers them onto `PerceptionModel` at import so `outward_bias.py` still reproduces. Their numerics tests moved with them (`anti_foveal_selftest.py`, 250 checks — deliberately *not* named `test_*` so `pytest tests/` does not sweep up a run that monkeypatches the model). If they are ever readopted, move the blocks back onto the class rather than making the model depend on that shim.
