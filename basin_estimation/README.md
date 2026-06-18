# Basin-of-attraction estimator — vetting

> **Status (2026-06):** all 11 vetting steps complete — see
> [findings.md](findings.md) §12. Public entry point
> `compute_basins_at_focal_loc` in
> [basin_via_theta.py](basin_via_theta.py) (vetted, runs; not yet wired
> into `decision_model.py`). The plan below is preserved as the record of
> how the estimator was vetted.

## Migration into decision_model — basin-wheel overlay (2026-06-18)

The neutral-seed basin **wheel overlay** prototyped here (`basin_mesh.py` /
`basin_mesh_fly.py`) has been ported into NBM as an option on the bifurcation
diagram: `NeuralBandModel.plot_bifurcation_diagram(overlay_basins=True)` draws
the count colormap (dimmed, default α=0.9 via `basin_bg_alpha`) with a basin
wheel at each region.
Ported methods now live in `decision_model.py` (NBM):

- `basin_arcs_at_focal_loc` — per-location heading→stable-direction basin
  partition (neutral-seed slaved flow + destination-flip bisection). **Public.**
- `_basin_destination` — one slaved-flow trajectory's destination index.
- `_basin_wheel_placement` — deepest-interior + low-count min-area filter +
  richest-first min-separation merge (the `region` placement). Modular (static).
- `_render_basin_wheels` — the wheel glyph (θ-basin annulus + arrows,
  categorical rank colors, rank legend). Contained (static).
- `_overlay_basin_wheels` / `_basin_arcs_worker` — orchestration + pool wrapper.

Also folded in (general speedup): `run_dgamma_dt` now fetches
`get_neural_signals` **once per call** (the heading is fixed for the whole
integration) via a new `signals=` hook on `dgamma_dt` — exact (~1e-14) and
~9× faster, benefiting every caller including walkers.

**This directory is slated for retirement** (can proceed on a separate
branch). Cleanup checklist:

- `basin_arcs.py`, `basin_mesh.py`, `basin_mesh_fly.py` — superseded by the
  NBM methods; remove or keep only as throwaway demos.
- `theta_scan.py`, `basin_via_theta.py` — the prototype's model wiring and the
  OLD scan-based `compute_basins_at_focal_loc` (a *different*, γ-branch-extent
  estimator — **not** what the overlay uses); decide keep-as-record vs remove.
- **Keep** `findings.md` + `free_energy_derivation.md` as the vetting record.
- **Not migrated** (prototype-only, intentionally left behind): the scan-based
  `compute_basins_at_focal_loc`, the commitment/two-branch probes, target/R_sc
  descriptors, grid placement (`adaptive_cells`), and the single-panel figure
  scaffolding.

Standalone exploratory work to validate a basin-of-attraction estimator
for stable self-consistent equilibria in NBM, before any changes to
`decision_model.py`. Mirrors the structure of `VM_bifurcations/`:
diagnostic scripts here, design and findings in markdown, no
modifications to the main model code until conclusions justify them.

Running mathematical findings and results: see [findings.md](findings.md).

## Design decisions in scope

- **Model:** NBM only.
- **γ-continuation:** warm-start LSODA seeded from previous θ's γ_eq.
- **F_γ:** re-derive as part of vetting (not lifted from
  `Hamiltonian ideas.tex`).
- **Noise model:** γ-Langevin (primary) + θ-Gaussian (calibration).
- **Discontinuity catalog:** the "may not be exhaustive — revisit if a
  fourth type surfaces" loose end is resolved. Vetting settled on four
  basin-boundary types: smooth saddle, γ-fold, perception-collapse, and
  (under the half-angle torque law) branch-cut at the facing-away
  heading. The branch-cut was the fourth type. Detail: findings.md
  §7.5, §0.

## Calibration setup — primary

Setup VM-k055 (from `../VM_bifurcations/VERDICT.md`):

```python
target_locs = np.array([[4.33,  2.5],
                        [4.33, -2.5]])
targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)
percep = model.PerceptionModel(targets,
    focal_loc=(0, 0), focal_angle=0,
    neural_angle_dist='vonmises', angle_weight='neural_angle_dist',
    a_warp=0.55)
nbm = model.NeuralBandModel(percep)   # T=0.2, K=2
```

Test points within this setup (final list will be confirmed by reading
the bifurcation diagram):

| label | (x, y) | what we expect |
|---|---|---|
| `1stable_far` | something like (0.5, 0) or (5.5, 0) | one stable SC eq; trivial basin |
| `2stable_sym` | TBD from bifurcation diagram | two stable SC eqs, y-symmetric basins by symmetry |
| `2stable_asym` | something like (4.0, 1.5) | two stable SC eqs, asymmetric basins (user's close+far intuition) |
| `3stable` | (1.5, 0) | three stable SC eqs (per coupled criterion) |
| `hopf_island` | (2.1, 2.45) | zero stable SC eqs, stable limit cycle; graceful failure expected |

## Calibration setup — discontinuity stress test

Setup BlindSpot (from `../weighting_analysis/`):

- Delta targets (`geom_name=None`), `neural_angle_dist='cutoff'`,
  `angle_weight='neural_angle_dist'`, `a_warp=0.0`, `b_warp=π/2`
  (as implemented in `detect_discontinuities.py`).
- **Why `b_warp=π/2` and not the originally-planned `π`:** the smooth
  cutoff weight has compact support on `|ego| < b_warp`, so at `π/2`
  any target more than 90° off the heading gets `ρ = 0`. The entire
  rear hemisphere is then a perception blind spot: as the observer
  rotates, there is an extended θ-range over which *every* `ρ_j = 0`,
  so `γ_eq → 0` and `R → 0` — a genuine **perception-collapse** zone.
  That is exactly the static discontinuity the Step-7 detector is built
  to catch (findings.md §7.4). At the planned `b_warp=π` the window
  still spans the full circle, so no collapse zone forms.
- Purpose: exercise the perception-discontinuity detector at the
  perception-collapse boundary.

## Vetting steps (in order)

Each step is a small standalone script (or markdown derivation). Pass
criterion stated for each. We do not move to step N+1 until step N
passes — or the failure is understood and the plan is revised.

### Step 1 — Derive F_γ analytically
- **Deliverable:** `free_energy_derivation.md`
- **Content:** closed-form expression for the Lyapunov function
  F(γ; θ, focal_loc) such that dγ/dt = −∂F/∂γ̄ (or equivalent) for
  the NBM cosine kernel with arbitrary weighting and warping.
- **Sanity tests deferred to step 2.**

### Step 2 — Validate F_γ numerically
- **Deliverable:** `check_free_energy.py`
- **Tests at multiple random (γ, θ, focal_loc) triples:**
  1. Numerical ∇F via finite differences matches the analytical
     expression to ~1e-8.
  2. At γ-equilibria from `gamma_equilib(focal_angle=θ)`, ∇F = 0 to
     ~1e-6.
  3. Hessian of F at γ-equilibria labelled stable by `_discrim_A` is
     positive definite; at unstable, has a negative eigenvalue.
- **Pass:** all three.

### Step 3 — γ-Langevin
- **Deliverable:** `stoch_dgamma.py`
- **What:** a stochastic version of `run_dgamma_dt`:
  dγ = (deterministic dgamma_dt) dt + sqrt(2D) dW, with D set by T
  via the fluctuation–dissipation relation we derive from F.
- **Tests at a fixed θ with focal_loc held constant:**
  1. Run long simulation starting near γ_eq; histogram γ samples.
  2. Predicted stationary distribution: P(γ) ∝ exp(-F(γ; θ)/T_eff)
     for some T_eff we work out from the fluctuation–dissipation
     relation.
  3. Empirical histogram matches predicted up to MC noise.
- **Pass:** measured γ-fluctuation variance matches the F-Hessian
  prediction to within ~10% at moderate sample sizes.

### Step 4 — θ-scan with warm-start γ-continuation
- **Deliverable:** `theta_scan.py`
- **Function:** for given (focal_loc, init_θ, init_γ), sweep θ around
  S¹ in both directions; at each sample compute γ_eq(θ) via warm-start
  LSODA. Return arrays of θ, γ_eq, R, ego_angle, f(θ).
- **Tests on `1stable_far`:**
  1. f(θ) has exactly one stable zero and one unstable zero.
  2. Stable zero agrees with `sc_equilib` output to <1e-3 rad.
- **Pass:** both.

### Step 5 — V(θ), saddle detection, basin features
- **Deliverable:** `basin_via_theta.py`
- **What:** consume a θ-scan; integrate V(θ); locate stable/saddle
  zeros via brentq refinement; for each stable θₛ output Δθ⁻, Δθ⁺,
  ΔV⁻, ΔV⁺.
- **Tests on `2stable_sym`:**
  1. Detected saddles match `sc_equilib`'s unstable equilibria.
  2. Cross-check vs Method B (bisection between known stable eqs):
     saddle θ agrees to ~1e-3 rad.
  3. V(θ) curvature at θₛ matches local Jacobian eigenvalue from
     `_discrim_coupled` along the heading direction.
  4. Δθ⁻ ≈ Δθ⁺ by y-symmetry (point is on the y=0 line, so basins of
     symmetric stable eqs should mirror).
- **Pass:** all four.

### Step 6 — γ-saddle finding and ΔF_γ
- **Deliverable:** `basin_via_gamma.py`
- **What:** at each stable SC eq (θₛ, γₛ), use
  `gamma_equilib(focal_angle=θₛ)` to enumerate γ-equilibria at fixed
  θ. Identify γ-saddle(s) bounding the γ-basin of γₛ. Evaluate ΔF_γ
  using the F derived in step 1.
- **Tests on `2stable_sym` and `3stable`:**
  1. F-evaluation at γₛ and γ-saddle gives the same value (up to
     additive constant) as integrating −∇F along a path from γₛ to
     γ-saddle.
  2. γ-Hessian eigenvalue at γₛ matches V''(0) of F along the
     escape direction.
- **Pass:** both.

### Step 7 — Discontinuity detection
- **Deliverable:** `detect_discontinuities.py`
- **What:** during a θ-scan, threshold |Δγ_eq| and |Δf| between
  consecutive samples; classify each large jump as γ-fold or
  perception-discontinuity.
- **Tests on calibration setups:**
  1. On smooth calibration points (the VM-k055 ones above), zero
     spurious detections at default thresholds.
  2. Construct a known-fold point (sweep `focal_loc` along a known
     fold line in (x,y) if we can find one from `sc_equilib`'s output
     — or, failing that, a synthetic check via a γ-bifurcation under
     parameter change): fold detector fires at the right θ.
  3. On Setup BlindSpot at a configured walker pose: perception
     discontinuity detector fires at the blind-spot boundary.
- **Pass:** detector behaviour matches expectation on all three.
- **Note:** if step 7 turns up a discontinuity type beyond our three,
  document it and revise the catalog.

### Step 8 — Monte Carlo ground truth
- **Deliverable:** `mc_escape.py`
- **What:** at calibration points, run two MC ensembles:
  (a) γ-Langevin only (`std=0`, T from the model). For each stable SC
      eq, start γ at γₛ, focal_loc fixed, focal_angle = θₛ, evolve
      until θ leaves a θ-neighborhood; record escape time and
      destination basin.
  (b) θ-Gaussian noise only (`std>0`, no γ-Langevin). Same setup.
- **Predictions to check:**
  1. (a) escape rates vs Kramers from min(ΔF_γ/T, ΔV(θ)/T_eff).
  2. (b) escape rates vs exp(-2ΔV(θ)/σ²).
  3. Ratio (b)/(a) at matched parameters → empirical T_eff = T·g(geometry).
     Should be roughly constant if heading-noise is a faithful proxy.
- **Pass:**
  1. log-rate predictions match log-empirical to within factor of 2–3
     (Kramers is exponential, prefactors are approximate).
  2. Relative ordering of robustness across stable eqs is correctly
     predicted in all cases tested.

### Step 9 — Asymmetric basin test
- **Deliverable:** part of `basin_via_theta.py` test output
- **What:** at `2stable_asym` (close+far targets), verify the
  estimator finds Δθ near the far-target stable eq much smaller than
  Δθ near the close-target stable eq.
- **Pass:** qualitative result matches user's prior expectation.

### Step 10 — Hopf island graceful failure
- **Deliverable:** part of integration test
- **What:** at `hopf_island` (2.1, 2.45), `sc_equilib` returns zero
  stable eqs. The basin estimator should return an empty basin list
  with a sentinel/metadata note ("no stable SC eq; suspect
  oscillatory attractor"). It should not crash, hang, or silently
  return garbage.
- **Pass:** sentinel returned cleanly.

### Step 11 — Performance benchmark
- **Deliverable:** `bench_per_point.py`
- **What:** time the full basin-estimate pipeline at each calibration
  point. Extrapolate to a typical bifurcation grid.
- **Pass criterion:** total bifurcation diagram cost with basin
  estimate < ~10× cost without. If not, propose subgrid sampling or
  other mitigations before implementation.

## After vetting

All steps passed. What was built: the standalone public wrapper
`compute_basins_at_focal_loc(focal_loc, *, scan_single_stable=False)` in
[basin_via_theta.py](basin_via_theta.py), returning
`{'basins', 'stable_count', 'unstable_count', 'sentinel'}` with per-cell
basin-dict shapes by cell class (findings.md §10.2). Per-cell rendering
rules for the two-panel bifurcation+basin plot are spec'd in findings.md
§10.4 / §11.

Still to do (not yet done): wire the wrapper into `decision_model.py` as
a `NeuralBandModel` method, and build the two-panel plot (bifurcation
raster + per-cell basin glyphs). Both prerequisite modeling changes
(sin(Θ*/2) dθ/dt with K=2; warp/weight decouple) are complete and the
Steps 5–9 calibration points were re-vetted as invariant (findings.md
§0).

## Loose ends carried forward

- Discontinuity catalog completeness — RESOLVED (Step 7 + the
  half-angle law): four basin-boundary types, the fourth being the
  branch-cut. See the "Design decisions in scope" bullet and
  findings.md §7.5 / §0.
- Joint-system gradient question (whether a global Lyapunov function
  exists for the coupled (γ, θ) system, or only local quadratic forms
  near each stable SC eq). Not blocking; revisit if it becomes
  load-bearing for the F_γ-vs-V(θ) reconciliation.
- Hopf-near-but-not-in case: cells adjacent to Hopf curves have
  fragile Kramers estimates. Document as caveat once we've seen it.
