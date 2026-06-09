# Reduced vs coupled stability criterion

Companion analysis for the `'reduced'` stability criterion on `NeuralBandModel`
/ `IsingExtModel` (see the "Stability criterion" section of the top-level
`CLAUDE.md`). `'reduced'` is the default.

- **`reduced`** — timescale-separated test, consistent with γ slaved to
  equilibrium (the dynamics `plot_walkers` integrates). Stable iff the fast γ
  block `A` is Hurwitz **and** the slow Schur complement `λ_slow = d − c·A⁻¹·b`
  is negative; the sign is read off as `sign(det J)` (block-determinant
  identity, see `block_determinant_identity.md`) to stay well-conditioned at
  γ-folds.
- **`coupled`** — full 3×3 eigenvalue test; stability of the non-separated
  continuous ODE, which additionally sees coupled γ–θ Hopf / limit-cycle
  instabilities the slaved walker cannot exhibit.

All quantitative results below are for the vonmises k=0.55 two-target setup
(targets at (4.33, ±2.5), r=0.5) under the half-angle dθ/dt law with K=2.

## Scripts

- `compare_reduced_vs_coupled.py` — scans observer (x,y), counts stable SC
  equilibria under both criteria, writes side-by-side PNGs + a disagreement
  summary. Two configs: vonmises k=0.55 full-weighting (island setup) and
  smooth cutoff a=0, b=π, uniform weight (standard non-Hopf setup).
- `island_anatomy.py` — per-equilibrium classification of the upper island
  (saddle / slow-unstable / Hopf / stable), the eigenvalue table, the Hopf-cell
  detector, the near-γ-fold conditioning report, and the 4-panel figure
  (`island_anatomy.png`).
- `reduced_vs_full_dynamics.py` — full coupled ODE vs the slaved walker, side by
  side at a 0-stable point (`reduced_vs_full.png`).
- `reduced_dynamics_anatomy.py` — the reduced (slaved) dynamics in detail
  (`reduced_dynamics_anatomy.png`): the multivalued slaved slow flow, the
  γ-branches, the idealized relaxation cycle (period by branch integration),
  and an actual warm-started Euler slaved walker that reproduces it.
- `cycle_birth_death.py` — transverse trace establishing the full-system cycle
  is a supercritical Hopf (focus Re crosses 0, amplitude → 0, finite period).

## Criterion comparison

- **Standard cutoff (a=0, b=π, uniform weight): `reduced ≡ coupled`** — 0/1813
  cells disagree. With no Hopf and away from γ-folds the two criteria are
  identical.
- **vonmises k=0.55 island setup:** agreement on 99.8–99.9% of cells. The
  disagreements are of two well-understood kinds:
  1. **Near-saddle-node / separation-breakdown skin** (`reduced = coupled − 1`):
     a thin, y-symmetric band on the 1-/2-stable boundary where a γ-block
     eigenvalue passes through 0 (e.g. `eig(A) = [−0.998, +0.017]`). There the
     "fast" mode is not fast, so the separation is invalid: `reduced` gates on
     `A` Hurwitz and calls the eq unstable, while `coupled` finds the heading
     coupling stabilizes the marginal mode. 2–4 cells per grid.
  2. **Hopf island** (`reduced = coupled + 1`): `coupled` flags an oscillatory
     instability the slaved 1-D θ-flow cannot realize. `reduced` removes it.

The reduced↔coupled difference is confined to exactly the two regimes where the
γ-fast/θ-slow separation breaks down.

## Island anatomy (`island_anatomy.png`)

In the 0-stable band the lone SC equilibrium is an **unstable saddle-focus**.
`eig(A)` of the 2×2 γ-block (e.g. `[−1, +0.16]`) alone looks like a saddle, but
that positive γ-mode is not a full-3D eigenvalue: coupling θ turns it into a
**complex pair with positive real part** (e.g. `0.083 ± 0.258j` at (2.1, 2.45)),
plus one stable real direction ≈ −1. So there is a genuine Hopf.

Under the half-angle law the Hopf is a **near-degenerate knife-edge**: the only
Hopf-unstable focus (A Hurwitz + complex pair, Re > 0) sits at one cell
≈(2.467, ±2.633) with Re ≈ +0.0005, at the tip of the saddle arc — the codim-2
fold/Bautin organizing center, exactly where `eig(A) → 0`. It is weak enough to
slip between coarse grid points; `island_anatomy.py` lands on it.

**γ-fold ⇒ Schur blow-up, handled by the det(J) form.** At that same fold cell
`min|eig(A)| ≈ 8e-4` and the literal Schur complement `|λ_slow| ≈ 308`, while
`det(J) ≈ −0.25` stays bounded. The criterion's `sign(det J)` slow test gives
the correct, well-conditioned verdict. The two difficulties are anti-correlated:
near a fold `|λ_slow|` is *large* (verdict far from threshold, sign robust); a
*marginal* slow eigenvalue only occurs where `A` is well-conditioned.

## Dynamics in the 0-stable band: full vs reduced

Both systems oscillate at a 0-stable point, by different mechanisms.

**Full coupled system → smooth Hopf limit cycle** (`reduced_vs_full.png`). The
unstable saddle-focus spirals out to a stable limit cycle: heading oscillates
~9° p-p, period ~14, at high coherence R ≈ 0.75 (head-bobbing). It is a
supercritical Hopf — tracing transversally across the band (`cycle_birth_death.py`),
focus Re crosses 0 at both edges with amplitude → 0 and finite period (Re
−0.011 → +0.020 → −0.021, amp 0 → 2.4° → 0). The cycle lives inside a closed
Hopf loop; the fold tip is the degenerate Bautin point. This is the VERDICT.md
head-bobbing, and it survives the half-angle torque-law change.

**Reduced (slaved) system → γ-bistability relaxation oscillation**
(`reduced_dynamics_anatomy.png`). Where the symmetric SC equilibrium is
γ-unstable the fast γ-subsystem is **bistable**: over a heading window
θ ∈ [−0.100, −0.059] (~2.35° wide) two stable γ-branches (Θ_neur > 0 and < 0)
flank the unstable symmetric branch (Θ_neur = 0), so the slaved slow flow
`dθ/dt = g(h(θ))` is **multivalued**.

- There is **no stable rest point**: dθ/dt = 0 requires Θ_neur = 0, which is
  only the unstable symmetric branch; the stable branches always have
  Θ_neur ≠ 0, so the slaved walker never stops turning.
- Each stable branch drives θ toward the fold where that branch is destroyed —
  upper (dθ/dt > 0) toward the top fold, lower (dθ/dt < 0) toward the bottom —
  with **finite** dθ/dt at the fold (~0.02–0.03, not → 0: a relaxation
  oscillation, not a SNIC/bottleneck). At each fold γ jumps to the other branch.
- Result: a fast, small **relaxation oscillation**, θ shuttling across the
  window. Confirmed two independent ways — idealized branch integration
  (2.35° p-p, period 1.27) and a warm-started Euler slaved walker (2.43° p-p,
  period 1.33); the θ-range matches the window edges exactly.

The full Hopf cycle and the reduced relaxation oscillation are different objects:
different mechanism (spiral-out vs fold/hysteresis), ~3–4× different amplitude,
~10× different period. The slaved 1-D θ-flow can oscillate this way only because
γ is bistable here; where γ is monostable (including every multistable cell whose
count the criterion sets) the slaved flow is a clean 1-D flow with no oscillation.

None of this changes the stability criterion: it tests the symmetric SC
equilibrium, which is correctly unstable in this band. "0 stable equilibria"
means there is no fixed heading to commit to — not that the walker is quiescent.
