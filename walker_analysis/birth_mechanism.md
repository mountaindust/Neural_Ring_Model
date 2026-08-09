# Why the center and outer stable directions are born at different x (y=0 cut)

A mechanism study prompted by the upper-left (y=0) panels of the branch diagrams:
on the midline the reborn-center stable direction appears well *before* the
outer-target stable directions. This explains why and what controls the separation.

**Reproducing the analysis and figures.** The engine is
[skeleton_birth_analysis.py](skeleton_birth_analysis.py) — its `make_model`,
`landmarks`, and `ninegon` helpers compute every birth/death x-location and the
parameter sweeps in the tables below (run it directly for the landmark smoke test; it
prints the x-locations, it does not itself write PNGs). The figures are assembled from
those outputs:
- `birth_mechanism.png` (the summary) and `birth_mechanism_knobs.png` (the β / gap /
  `a_weight` sweeps) — composed from `skeleton_birth_analysis.py`'s `landmarks` sweeps.
- The 9-gon (x, θ)+(x, R) branch diagram is **not kept as a file** — it is hard to
  read and nothing here depends on it. Rebuild it on demand with
  [../plots/decision_skeleton.py](../plots/decision_skeleton.py)'s
  `plot_branch_diagram` on the `skeleton_birth_analysis.ninegon()` geometry.

## TL;DR

The two "births" are **different bifurcations governed by different physics**:

- **Center reborn = a pitchfork.** The straight-ahead (θ=0) equilibrium *always
  exists* by symmetry; only its **transverse stability** flips. It is governed by a
  geometric/warp condition — it re-stabilizes once the outer targets swing past
  **neural ±90° (broadside)**. Robust: nearly independent of the weight and only
  weakly β-dependent.
- **Outer born = a saddle-node.** A genuinely *new* off-axis equilibrium, locked onto
  a *single* outer target, is created. This is a **mean-field Ising ordering
  transition** that requires strong commitment (cold neural ring — large β —
  large outer-target
  weight ρ, well-resolved separation). It is **marginal at the default** (it flickers
  in/out at the solver's resolution) and **vanishes entirely** for β ≲ 13.6, narrow
  separation, large warp plateau, etc.

They are separated — center first — because **re-symmetrizing the balanced
all-targets consensus is "easy", but committing to ONE outer target (a symmetry
break) needs more order**, so the observer must get closer (higher coherence R)
before the outer lock can exist.

## The midline cascade (mechanism-study operating point)

> Operating point for the numbers in this doc: 3 targets, 40°, radius 5, r=0.5, at the
> `skeleton_birth_analysis.make_model` defaults `a_warp=0.47π, a_weight=0.40π, β=15`
> (**not** the shipped fly tuning, which is `a_warp=0.65π, a_weight=0.20π, β=30` and so
> shifts the absolute x-locations outward). The *mechanism* — pitchfork vs saddle-node,
> center reborn first — is tuning-invariant; the specific x-values below are at this
> operating point.
>
> **The tabulated x-locations need a re-run to be current.** They were computed
> when the coupling carried a factor of the *visible* target count (effective
> `β = N_visible/T`), so the rear blind spot dropping the far competing target at
> the outer-locked heading both removed a competitor **and** cooled the ring from
> `3/T` to `2/T`. Under a scene-independent `β` only the first happens, which
> should move the outer saddle-node **earlier** in x. The mechanism narrative
> (pitchfork vs saddle-node, centre first, β as the dominant knob) is unaffected;
> the numbers are the part that shifts.

| x | event | type |
|---|---|---|
| ~0.91 | compromise arms (±22°) born, center still stable | saddle-node |
| ~1.57 | center (0°) **destabilizes** | pitchfork |
| ~2.73 | center **reborn** stable | pitchfork |
| ~3.05 | outer-target branches (±74°) born (marginal, flickering) | saddle-node |

## Mechanism 1 — the center pitchfork: the neural-90° "broadside"

The θ=0 equilibrium exists at every x (the symmetric consensus); what changes is its
transverse stiffness. Measured by the slaved-flow slope Θ′(0) = d(arg γ)/d(heading) at
θ=0 (stable iff < 0):

```
x:      0.5   1.0   1.4   1.55 | 1.70   2.0   2.4 | 2.73  2.9   3.2
Θ'(0): -1.44 -1.44 -1.44 -1.43 |+62.7 +64.4 +65.0 |-1.05 -1.42 -1.44
center: stable .............. | UNSTABLE ........ | stable ..........
```

The instability is a huge, near-critical **positive** susceptibility. Its cause is
geometric: as the observer advances along the midline, the outer targets swing to
ever-larger egocentric (hence neural) angle. The outer-target **neural** angle is:

```
x:            1.0   1.55(destab)  2.0   2.73(reborn)  3.0
neural(deg):  70    78.6          86.8  102.3         108.7
```

The center is unstable for **outer neural ∈ (~79°, ~102°)** — i.e. **while the outer
targets straddle neural ±90°**.

Here ν is a target's **neural angle** (its egocentric bearing relative to the
heading/consensus, after the warp). Each target enters the consensus
γ = Σ ρ_k e^{iν_k} σ_k with an **Ising magnetization** σ_k = σ(2βR cos ν_k)
(σ = logistic): straight-ahead (cos ν≈1) ⇒ fully "on" (σ→1), broadside (ν=90°,
cos ν=0) ⇒ half-on (σ=½), behind (cos ν<0) ⇒ suppressed (σ→0).

Linearizing the symmetric state's transverse response (tilt the heading by δθ; both
outer neural angles shift by the same u = W′·δθ), the net first-order torque on the
consensus angle Θ is the sum of two competing terms:

```
destabilizing (magnetization feedback):  + 2*beta*R  · sigma'(z0) · sin^2(nu0)
restoring     (geometric):               -            sigma(z0) · cos(nu0)
                                          z0 = 2*beta*R*cos(nu0)
```

The symmetric state goes unstable when the first beats the second. The destabilizing
term **peaks sharply at broadside (ν0 = 90°)** for two reasons: the logistic is
steepest there (σ′ is maximal at z0 ∝ cos ν0 = 0) **and** sin²ν0 is maximal there —
i.e. tilting toward one outer target makes it "more on" and the other "more off"
fastest when they are broadside (a runaway *differential magnetization*). The
restoring term simultaneously passes through zero at 90° (cos ν0 = 0). Below ~90° the
geometric restoring wins (stable); above ~90° the outer targets fall behind and σ→0
suppresses them (stable again). Hence the unstable window **straddles neural 90°**
(and Θ′(0) spikes to +63 right there). Once the outer targets pass into the rear
hemisphere the center regains stiffness → **reborn**.

(The driver is the *magnetization* σ(…cos ν…) being most sensitive to ν at broadside,
not the bare alignment factor cos ν — though d cos ν/dν = −sin ν is also maximal at
90°, which is incidental.)

**This is set by GEOMETRY × WARP** (at what x do the swinging outer targets reach
neural 90°), so `a_warp`/`b_warp` shift it but the **weight does not** (see sweeps).

## Mechanism 2 — the outer saddle-node: an Ising single-target lock

The outer branch is a *new* equilibrium with the observer's heading pointed at one
outer target. At its birth (x≈3.05) the configuration is: the outer target dead-ahead
(neural ≈ 0, dominating the magnetization, cos ≈ 1), the competing center target at
large neural angle (≈ −110°, cos < 0) where the Ising logistic σ(2βR cos ν) ≈ 0
**suppresses** it, and the far outer target dropped out of the visual field (the rear
blind spot from `b_weight < π`). So the outer lock is essentially a *single-target*
mean-field magnetization: it exists only once ρ_outer · β · R clears the ordering
threshold. That needs the observer close enough that R is high (~0.6 here) — which is
**later** than the center pitchfork.

## What moves each birth — parameter sweeps

`reborn` = center pitchfork; `outer` = outer saddle-node; `gap = outer − reborn`.
`nan` = the outer branch never appears in range.

| sweep | center reborn | outer born | takeaway |
|---|---|---|---|
| **β** 30→7.5 | 2.55 → 2.85 (robust) | 2.67 → **vanishes for β≲13.6** | the outer is an Ising transition: β-fragile; center is not |
| **separation** 20→60° | 4.05 → 1.17 | **absent ≤30°** → 1.23 | targets must *resolve* for an outer lock; gap shrinks as they spread |
| **distance D** 3→7 (r fixed) | 1.63 → 3.83 | 1.83 → 4.91 | gap **grows** (fixed r ⇒ smaller angular targets ⇒ weaker ρ ⇒ outer pushed out) |
| **a_warp** 0.35→0.65π | 2.39 → 3.19 | 2.61 → vanishes ≥0.55π | warp sets the neural-90° crossing → moves both |
| **b_warp** 0.75→1.0π | 2.23 → 2.95 | 2.41 → vanishes at π | same |
| **a_weight** 0.0→0.5π | 2.57 → 2.73 (weak) | 2.81 → vanishes at 0.5π | weight moves the outer strongly (ρ_outer); the center only weakly (through R) |
| **b_weight** 0.55→0.9π | 2.73 (pinned) | non-monotonic, earliest at 0.80π | blind-spot edge; the default is already near-optimal |

These weight rows show the two births are largely independent: the weight window
changes the outer weight ρ_outer and so moves / kills the outer saddle-node, while the
center pitchfork barely moves — `b_weight` leaves it *exactly* put, `a_weight` shifts
it only weakly. (In the instability condition `2βRσ′sin²ν vs σcos ν` the explicit ρ
factors cancel, so the weight enters the center only through R.) `b_weight` (the
visual-field / blind-spot edge) is non-monotonic and already near its optimum at
0.80π: there the rear blind spot drops the far competing target at the outer-locked
heading, which is exactly what lets the outer lock; widening or narrowing it
pushes the outer *later*.

## Generality — different kernels

The structure is **not** specific to `lin_cutoff`:

| warp kernel | reborn | outer | note |
|---|---|---|---|
| lin_cutoff (default) | 2.73 | 3.05 | — |
| cutoff | 2.73 | 3.05 | identical (analytic sibling) |
| symmetric_beta (α=2) | 2.47 | 2.69 | same structure |
| vonmises (k=1) | 1.71 | 1.79 | same structure |
| symmetric_beta (α≥10), vonmises (k≥2) | — | — | **center never destabilizes** |

The last row is a positive control for the broadside mechanism: a warp concentrated
enough that the outer targets are *always* at large neural angle never lets them cross
neural ±90°, so the center pitchfork (and its interlude) simply does not occur.

## The 9-gon ring (9 targets 40° apart at radius 5)

Same center pitchfork embedded in a dense multi-target cascade: the 0° direction is
robust except for an unstable interlude (x ≈ 1.8–2.5) and is reborn by x ≈ 2.5 —
exactly the forward-three-target pitchfork — while the many side/rear targets each
contribute their own branches that are born and die as the observer moves.

These 9-gon x-locations **are** current (midline rescan at Δx = 0.05 under the
scene-independent β; the interlude moved only from 1.75–2.60 to 1.80–2.50). The
9-gon is the geometry most exposed to the change — with 9 targets the old
`N_visible/T` coupling swung between β ≈ 5 and 45 depending on where the observer
looked — so the fact that this cascade barely moves is itself the point: the centre
pitchfork is geometric, not thermal.

## Why "you can't get to the outer tracks easily" (the practical point)

The outer-target stable direction is the **marginal** state. At the default
(β=15, 40°, r=5, D=5) the outer saddle-node sits right at threshold — born late
(x≈3.05) and only briefly (it flickers). To make the outer tracks **robust and
early**: raise β, widen the separation, raise ρ_outer (larger targets / wider weight
window), or shrink D. To **suppress** them: lower β, narrow the separation, or widen
the warp plateau. The center track, by contrast, is always available — it is the easy,
symmetric, globally-coherent state.

## Pulling the outer birth up to the pitchfork — without moving the targets

The targets (positions, sizes) are fixed by the experiment, so the legal knobs are
**β, the warp, and the weight window**. Figure
[birth_mechanism_knobs.png](birth_mechanism_knobs.png) (extended-β + gap + a_weight).

- **β is the clean, dominant knob.** The gap (outer − reborn) shrinks
  **roughly linearly with 1/β**, and the outer **backs right up against the pitchfork**:

  | β | center reborn | outer born | gap |
  |---|---|---|---|
  | 120 | 2.35 | 2.39 | **+0.04** |
  | 60  | 2.43 | 2.49 | +0.06 |
  | 30  | 2.55 | 2.67 | +0.12 |
  | 15 (default) | 2.73 | 3.05 | +0.32 |

  Raising β brings **both** births in *and together*, so all three stable directions
  become available at nearly the same observer x. Mechanism: the outer is an Ising
  ordering transition with threshold ∝ 1/β, so colder → commitment is cheaper → the
  outer lock appears at lower R (smaller x), sliding toward the nearly-β-independent
  pitchfork.

- **The gap does not go negative.** Across β and combined high-β + warp/weight sweeps
  the smallest gap reached was ≈ +0.02 (e.g. β=120, a_weight=0.2π → reborn 2.33,
  outer 2.35). The outer-target lock appears **at or just after** the center
  re-stabilizes, never before — the pitchfork acts as a floor the outer backs up to.

- **Lowering `a_warp` shifts the whole cascade earlier in x** (both births) while
  keeping the gap small — e.g. β=60, a_warp=0.40π → reborn 2.21, outer 2.25
  (gap 0.04), vs outer 3.05 at the default. Use this if you want the outer tracks
  earlier in *absolute* x rather than just relative to the pitchfork.

- **`a_weight` and `b_weight` are weak / already-optimal.** Lowering `a_weight` pulls
  the outer in only modestly and at high β mostly moves the *center* (counterproductive
  for the gap); `b_weight = 0.80π` is already near the outer-earliest optimum.

**Caveat (modeling):** β and the warp also set the *first* bifurcation and the overall
walker behavior, which the shipped fly tuning fixes to match the heatmap (the GODM
refit: `a_warp=0.65π, β=30` — see
[../plots/three_target_fly_refine_findings.md](../plots/three_target_fly_refine_findings.md)).
Raising β or lowering `a_warp` to pull the outer in shifts the **whole** cascade, so the heatmap
match would need re-checking — these are not free relative to that fit.
