# Engineering TODO (Neural Ring Model)

Open engineering items for the active model, moved out of `CLAUDE.md` (which now links here). For the forward-looking research-direction prose (JWB/AJB notes, repulsion/avoidance ideas, scale-up plans), see [TODO_Jan_2026.md](TODO_Jan_2026.md).

- **Figures affected by the γ transport (2026-08-20): cosmetic only — no conclusion moves.**
  Measured old-vs-new with matched seeds, **through each script's own `build_model`/`run_walkers`**
  (hand-transcribing the configs got the weight family/params wrong and produced spurious
  'significant' shifts — always import the builder).

  | figure | statistic old → new | significance | tracks moved | same outcome |
  |---|---|---|---|---|
  | `fly_bifurcation.png` | basin wheels | **byte-identical render** | — | — |
  | `three_target_fly_refine.py` | centre 47.4% → 45.2% | 1.2 s.e. (N=800) | 99.5% | 94.6% |
  | `two_target_fly_refine.py` | 51.6/48.4 → 50.9/49.1 | 0.4 s.e. (N=800) | 87.2% | 97.8% |
  | `fly_geom.py` (3-target) | 64/20/16 → 58/18/24 | ≤1 s.e. (N=50) | 98.0% | 74.0% |
  | `oblique_walker.py` row 1 | 40/60 → 40/60 | none | 43% | 100% |
  | `oblique_walker.py` row 2 | 100/0 → 100/0 | none | 33% | 100% |
  | all deterministic figures | — | bit-identical | — | — |


  Basins spot-checked directly against `build_two`/`build_three` over 126 points: **0 of 25** multistable two-target points and **1 of 34** three-target points change. The single mover, (4.30, 0), sits 0.2 outside the centre target's surface — inside the `basin_target_margin=0.15` exclusion zone, so no wheel is ever drawn there. That is why the render comes out byte-identical.
  So the `STD=4.0` calibration, the fly-vs-locust centre comparison and the basin figure all
  stand unchanged. Individual trajectories shift in every walker ensemble, so re-rendering the
  four walker figures keeps images in sync with the code but corrects nothing. `.npz` caches and
  the `*_findings.md` numbers do **not** need revising.
  **Not yet measured:** `walker_analysis/*.py` and `weighting_analysis/outward_bias.py`.
- **DONE 2026-08-20 — γ is transported through each heading step.** `_simulate_one_walk`, `_basin_destination` and `plot_dtheta_dt`'s swept mode now apply `γ *= e^{−i·turn}` after advancing the heading (`turn` = the full change, `dtheta*dt + noise`). Rationale is in `plot_walkers`' "Carrying gamma between steps" and `.claude/rules/walker-dynamics.md`; the investigation is §11–§12 of [theory/iem_nbm_fold.md](theory/iem_nbm_fold.md). Applied unconditionally — not gated on `angle_distortion_nu` or on the warp — so one convention holds everywhere. Deterministic output verified bit-identical; walker/basin paths change only where the γ landscape is multistable.
- **DONE 2026-08-20 — `IsingExtModel` folded into NBM and deleted** (−1279 lines). NBM gained `angle_distortion_nu` (a validating property), `nu_cosine`, `plot_nu_cosine` and `plot_dtheta_dt`, with the ν-gated `_discrim_A` fallback and `sc_equilib` probe. `tests/test_broad_validation.py` (the cross-model comparison) was retired with it, and the IEM entries were dropped from `test_half_angle_torque.py` / `test_reduced_criterion.py`; `tests/test_angle_distortion_nu.py` keeps the intrinsic ν tests. Equivalence measurements, caveats and the walker carry-frame investigation: [theory/iem_nbm_fold.md](theory/iem_nbm_fold.md).


- **Residual heading-noise floor (idea, deferred).** Generalize the gated noise `σ·(1−R)^p` to `floor + (σ−floor)·(1−R)^p`, so a small noise persists even at full commitment (`R→1`) instead of vanishing. Appealing for several reasons but kept simple for now; revisit before merging this branch or before publication. Distinct from the two-scale `std` default (which is one scale, two regimes).

- **Commitment characterization in multistable basins (research; tool: `NBM.basin_arcs_at_focal_loc`, comparing neutral R≈0.15 vs committed R≈1 seed via its `R_seed` arg).** With `arg(γ)=0` fixed, the seed R is the only knob that selects between coexisting γ-branches at a fold, so comparing an uncommitted (R≈0.15) vs committed (R≈1) seed map isolates the *commitment-sensitive wedge* and which destination "requires commitment" (i.e. is only reached from the high-R warm-start). Initial 2-target finding at (4.0,1.5): commitment is required to hold the *farther / weaker-perception* target; an uncommitted walker defaults to the closer/stronger one; the wedge is thin (~4% of headings); the two SC-eq R's are nearly equal, so *perception strength* — not equilibrium R — is the discriminator there ([theory/basins_of_attraction.md](theory/basins_of_attraction.md) §9). Open questions: (a) does this generalize where the **γ-fold spans a wide angular range**; (b) the **consensus-equilibrium-vs-single-target** case, especially the **three-target (locust)** geometry — does the center target pull walkers off the second decision point, and would a lower-R (less-committed) γ warm-start change that outcome?

- **Cache perception signals in `sc_equilib`/`gamma_equilib` — DONE (2026-08-18).** `PerceptionModel.signal_cache()` is a block-scoped memo of `get_neural_signals` keyed on the *exact* `(focal_angle, focal_loc)`; `NBM.gamma_equilib` and `NBM.sc_equilib` open one for the duration of a call, so nothing outside them is affected. `gamma_equilib` holds the heading fixed across all 50 hybr multistarts (one perception state, ~9×); `sc_equilib` cannot hoist — θ is a solve variable — but the same θ recurs at each scan node's three probe radii, across the polish iterates, and again in the residual check and stability test (~65% of recomputes removed, count grid 1.6–2.2×). Verified bit-identical on 1465 recorded solver outputs across five perception setups × three stability criteria; regression tests in [tests/test_signal_cache.py](tests/test_signal_cache.py). **The exact key is load-bearing:** perception is the only θ dependence in `dgamma_dt`, so a tolerance key that merged the Jacobians' θ±h probes with the base point would zero the θ column, give `det(J)=0`, and report every equilibrium unstable.
