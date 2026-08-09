# Can an anti-foveal (centre-dip) weight bias the observer outward?

**Date:** 2026-08-06 · **Script:** [outward_bias.py](outward_bias.py)
· **Geometry:** empirical `locust3` (3 circle targets, `r=0.1`, distance 3,
bearings {0°, ±35°}), shipped locust warp `lin_cutoff a_warp=0.50π, b_warp=0.90π`,
`K=6, β=30`, `stability_criterion='reduced'`.

## The question

The fly and locust three-target datasets split very differently:

| | N | centre | outer |
|---|---|---|---|
| fly3 | 125 | **45%** | 55% |
| locust3 | 518 | **29%** | 71% |

The fly's ≈1:2:1 is the ratio a two-stage fair binary cascade would produce, and
the model matches it
([../plots/three_target_fly_refine_findings.md](../plots/three_target_fly_refine_findings.md)).
Worth stressing that the *ratio* is cascade-like while the bifurcation structure
is not: the first bifurcation's two compromise arms collapse back onto the centre
(the reborn-centre pitchfork), and the outer targets become stable separately, in
isolated folds — see the branch diagrams below and
[../walker_analysis/birth_mechanism.md](../walker_analysis/birth_mechanism.md).

The locust's strong outer bias the model does *not* match: it is intrinsically
**centre-biased**, precisely because of that structure — along the midline the
centre-target branch is **reborn stable before the outer-target branches are
born at all**, so a walker riding a compromise arm is recaptured to the centre on
the way out
([../walker_analysis/three_target_findings.md](../walker_analysis/three_target_findings.md)).
The locust data's outer bias was known to the original authors but reported in
the supplementary material, and their framework required an external, added
mechanism to bias walkers outward.

The idea tested here is a mechanism internal to *our* perception model. Keep the
foveal neural **density** that sets the warp, but give the **weight** — the
density integrated over each target's visible arc to set ρ — a **dip** in the
middle instead of a bump, so that whatever sits dead ahead is under-weighted and
the observer is pushed off it.

## What was built to test it

> **These families are not part of the model.** They were added to
> `PerceptionModel` to run this analysis and **removed again** once the result
> came back negative — the model should not carry perception machinery it does
> not use. [anti_foveal.py](anti_foveal.py) preserves them verbatim and
> re-registers them onto `PerceptionModel` at import, which is the only reason
> [outward_bias.py](outward_bias.py) still runs. Numerics self-test (250
> checks): `python weighting_analysis/anti_foveal_selftest.py`.

Two anti-foveal weight families, both piecewise-linear with closed-form integral
and inverse (no spline), everywhere positive, and both degenerating to uniform
as their floor `m → 1`:

- **`lin_dip(θ; m, b)`** — `m` at θ = 0, ramping up to 1 at `|θ| = b`, then flat
  1 out to ±π. The sign-flipped sibling of `lin_cutoff` and the *minimal*
  perturbation of the model's uniform default: `w ≡ 1` everywhere except inside
  the frontal hole.
- **`lin_ring(θ; m, p)`** — `m` at θ = 0, up to 1 at `|θ| = p`, then back down to
  0 at ±π. Same frontal dip, but it also sheds weight toward the rear.

Together with the two incumbents this is a 2×2 design that separates the two
things an anti-foveal shape can do:

| | full-weight periphery | rear falloff |
|---|---|---|
| **no frontal dip** | `angle_weight=None` (uniform — the model default) | `lin_cutoff` (foveal — the shipped locust weight) |
| **frontal dip** | **`lin_dip`** | **`lin_ring`** |

The warp is held fixed across all four; only the weight changes.

## Headline: the dip does bias the observer off the centre — and off the outer targets just as hard

**The answer to the motivating question is no.** A centre dip does not produce an
outward bias, and the reason is structural rather than a matter of tuning:

> A weight is a function of **egocentric** angle. A dip at ego = 0 suppresses
> **whatever the observer is currently facing** — it has no way to know whether
> that is the centre target or an outer one. So it penalises *every*
> single-target commitment equally. And because the outer-target commitments are
> the fragile ones (they are born in marginal Ising saddle-nodes — see
> [../walker_analysis/birth_mechanism.md](../walker_analysis/birth_mechanism.md)),
> they are what the dip destroys first.

The net effect on the bifurcation structure is therefore *less* commitment
everywhere, and a *stronger* centre bias in the walker, not a weaker one. The
walker census is unambiguous — centre fraction against the locust's 29%:

| uniform | foveal `lin_cutoff` (shipped) | `lin_dip` m=0.25 | `lin_ring` m=0.25 |
|---|---|---|---|
| 98.8% | 64.8% | **100%** | **100%** |

### The mechanism, measured

`outward_bias_mechanism.png`. Observer at (2.0, 0), inside the
second-bifurcation zone, ρ split under each weight at the two competing
candidate headings:

| weight | facing CENTRE → ρ = [centre, up, low] | facing UPPER OUTER → ρ = [centre, up, low] |
|---|---|---|
| uniform | [**0.471**, 0.264, 0.264] | [0.471, **0.264**, 0.264] |
| foveal `lin_cutoff` | [**0.620**, 0.190, 0.190] | [0.494, **0.506**, 0.000] |
| `lin_dip` m=0.25 | [**0.218**, 0.391, 0.391] | [0.553, **0.093**, 0.354] |
| `lin_ring` m=0.25 | [**0.218**, 0.391, 0.391] | [0.724, **0.122**, 0.153] |

(Bold = the target being faced. The foveal row's third entry is exactly zero in
the right-hand column because its `b_weight = 0.80π` support genuinely excludes
the third target, which sits at ego ≈ −150° there.)

Read the two columns together:

- The dip does exactly what it was supposed to do in the left column: the centre
  target drops from a commanding 0.47 (uniform) to a minority 0.22, so the
  straight-ahead consensus can no longer hold itself.
- But in the right column it does the *same thing* to the outer commitment:
  0.264 → 0.093. The faced outer target is now the weakest of the three.

Geometrically the two configurations are near-mirror images in the frontal
hemisphere: facing the centre puts the rivals at ego ≈ ±75°; facing an outer
puts its main rival (the centre) at ego ≈ −75°. Any function of `|ego|` treats
them the same. Only the *third* target distinguishes them (ego ≈ −150° when
committed outward, so behind), which is why the shipped foveal weight — whose
support ends at 0.80π = 144° — helps the outer commitment: it deletes that
rival. That is also why `lin_ring` is slightly less destructive than `lin_dip`
(0.122 vs 0.093), but the effect is second-order next to the direct suppression
of the faced target.

### The bifurcation cascade

`outward_bias_cascade.png` — midline (x, θ) branch diagrams under all four
weights, with the branch-presence ribbon underneath.

Two things change as the dip deepens, and they pull in opposite directions:

1. **The centre-unstable window widens** — the centre destabilises earlier and is
   reborn later. Fraction of the midline with no stable centre branch goes from
   0.14 (uniform) to 0.36 at `m = 0.1`. *This part works.*
2. **The outer-target branches are pushed out and then annihilated entirely.**
   Under `lin_dip` they are gone for `m ≤ 0.5`; under `lin_ring` for `m ≤ 0.4`.
   At those depths there is nothing at the ends of the trident to commit to.

`outward_bias_events.png` sweeps the depth. Event x-locations (locust geometry,
midline):

| weight | arms born | centre destabilises | centre **reborn** | outer born | gap | no-centre frac |
|---|---|---|---|---|---|---|
| uniform | 0.79 | 1.39 | 1.79 | 1.97 | +0.18 | 0.14 |
| foveal `lin_cutoff` (shipped) | 0.77 | 1.39 | 1.74 | 1.79 | **+0.05** | 0.13 |
| `lin_dip` m=0.95 | 0.79 | 1.39 | 1.79 | 1.99 | +0.19 | 0.14 |
| `lin_dip` m=0.80 | 0.79 | 1.37 | 1.81 | 2.04 | +0.23 | 0.16 |
| `lin_dip` m=0.60 | 0.81 | 1.35 | 1.85 | 2.22 | +0.37 | 0.17 |
| `lin_dip` m=0.50 | 0.81 | 1.34 | 1.86 | **never** | — | 0.19 |
| `lin_dip` m=0.25 | 0.82 | 1.28 | 2.00 | **never** | — | 0.26 |
| `lin_dip` m=0.10 | 0.86 | 1.21 | 2.23 | **never** | — | 0.36 |
| `lin_ring` m=0.60 | 0.81 | 1.35 | 1.85 | 2.06 | +0.21 | 0.17 |
| `lin_ring` m=0.50 | 0.81 | 1.34 | 1.86 | 2.16 | +0.30 | 0.19 |
| `lin_ring` m=0.40 | 0.82 | 1.32 | 1.90 | **never** | — | 0.21 |

`gap = x_outer_born − x_centre_reborn`. **This is the number that matters**: a
walker on a compromise arm meets whichever event comes first. An outward bias
needs `gap < 0` (the outer option available before the centre is back).
Every anti-foveal setting makes the gap *larger*, monotonically, until the outer
branch stops existing. The shipped foveal weight has the smallest gap of
anything tested (+0.05) — which is, retrospectively, exactly why it was chosen
for the locust.

Note that the last column moves the *right* way throughout (a wider stretch of
midline with no stable centre option, 0.14 → 0.36). That is the one thing the
dip does deliver — and the walker section below shows it does not cash out.

The dip **width** barely matters (`width control` in the script's console
output): at `m = 0.4`, sweeping `b` over {0.15π … π} moves the centre rebirth
only within a few hundredths and never resurrects the outer branch. Depth `m` is
the whole lever.

### Stable-equilibrium rasters

`outward_bias_rasters.png` — the (x, y) stable-count diagrams over
`[0, 3.4] × [−2.4, 2.4]` (`max_count=5`; this geometry genuinely reaches 5 in
the five-branch window, so clipping at 3 would hide the point). Area in square
units by count:

| weight | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| uniform | 8.45 | 5.72 | 2.27 | 0.04 | 0.01 |
| foveal `lin_cutoff` | 8.09 | 6.20 | 2.14 | 0.02 | 0.02 |
| `lin_dip` m=0.25 | **10.04** | 5.04 | 1.40 | **0** | **0** |
| `lin_ring` m=0.25 | 9.31 | 5.65 | 1.53 | **0** | **0** |

Two things to read off. The 1-stable region **grows** under the dip (8.45 →
10.04 sq units, +19%) at the expense of every multistable region — the dip
removes commitments, it does not redistribute them. And the 4- and 5-stable
cells vanish **entirely**: the five-branch window (centre + two arms + two
outer) that both incumbents have simply does not exist once the outer branches
are gone.

The loss concentrates where predicted. Within 0.9 of an outer target the
multistable fraction drops from 54.2% (uniform) / 56.1% (foveal) to 44.3%
(dip) / 48.0% (ring) — the region an outward-biased walker would have to
commit in.

### Walkers

`outward_bias_walkers.png` — 400 locust walkers per weight at the shipped
noise knobs (`noise_exp=2, R_exp=3, std=3.0, v=0.2, dt=0.04`), scored by nearest
target at termination:

| weight | centre / upper / lower | centre % | vs data (29%) |
|---|---|---|---|
| uniform | 395 / 3 / 2 | **98.8%** | far too centred |
| foveal `lin_cutoff` (shipped) | 259 / 75 / 66 | **64.8%** | the documented near-match |
| `lin_dip` m=0.25 | 400 / 0 / 0 | **100%** | worst possible |
| `lin_ring` m=0.25 | 400 / 0 / 0 | **100%** | worst possible |

No walker was lost in any run. **The anti-foveal weights are not merely
unhelpful, they are the worst of the four**: not one walker in 800 reaches an
outer target.

The tracks are the surprise, and they sharpen the finding. One might expect the
widened centre-unstable window (0.14 → 0.26 of the midline at m=0.25) to at
least fan the ensemble out. It does the **opposite** — mean per-walker
excursion `max |y|`:

| uniform | foveal | `lin_dip` | `lin_ring` |
|---|---|---|---|
| 0.387 | **0.745** | 0.253 | 0.257 |

Under the dip the walkers travel in a *tighter* spindle straight down the
midline into the centre target. The reason is coherence: splitting ρ more evenly
across the three targets is exactly what lowers `R = |γ|`, and on the midline
the dip depresses it throughout —

| stable-branch R on the midline | x=0.5 | x=1.0 | x=1.5 | x=2.0 |
|---|---|---|---|---|
| uniform | 0.68 | 0.55–0.57 | 0.51 | 0.26–0.47 |
| foveal | 0.71 | 0.62–0.64 | 0.60 | 0.51–0.62 |
| `lin_dip` m=0.25 | 0.61 | 0.43–0.45 | 0.39 | 0.21–0.37 |

— and the shipped locust walker law drives on `K·R^{R_exp}` with `R_exp = 3`.
Dropping R from ~0.51 to ~0.39 at x=1.5 cuts the pursuit drift by a factor of
~2.2, so the walker barely steers at all: with the centre target dead ahead and
nothing else able to hold a commitment, it simply flies straight in. (The gated
noise `σ(1−R)²` does rise as R falls, but only by ~1.2× over the same drop —
nowhere near enough to compensate.)

So the deterministic "wider centre-unstable window" and an actually wider walker
ensemble are **not the same thing**, and the dip delivers only the first. Note
the foveal weight, which *raises* R, is the one that fans the ensemble out and
produces the trident.

This is the crux: **peeling off the midline and committing outward are two
different things, and the dip buys neither — it buys only the deterministic
absence of a centre branch, which the walker never gets to use.**

## Why this is worth recording as a negative result

The failure is not "we did not find the right shape". It is a statement about
what an egocentric weight can express:

- **Egocentric weighting is heading-relative; "outer" is a configuration
  property.** The weight sees one number per target — its egocentric angle — and
  cannot ask "is this target flanked by others on both sides?" (which is what
  makes the centre target the centre target). The same limitation was noted much
  earlier for the *foveal* weight, in a different guise:
  `angle_weight` "is a fixed *egocentric* window with no notion of 'the pair
  under consideration'"
  ([../walker_analysis/three_target_analysis.md](../walker_analysis/three_target_analysis.md)).
  A dip inherits that limitation exactly.
- **The asymmetry that does exist runs the wrong way.** Committing outward is
  the *harder* commitment (fewer targets in front, a marginal saddle-node);
  committing to the centre is protected by the centre target's growing angular
  extent as the observer approaches. A perturbation that weakens all commitments
  uniformly therefore breaks the outer ones first — the dip is a strictly
  centre-biasing move despite pointing "outward" in shape.
- **The one lever that genuinely helps the outer branch is the rear cutoff, and
  the shipped foveal weight already uses it.** `b_weight = 0.80π` deletes the
  third target at an outer commitment. That is a *configuration*-sensitive
  effect obtained by accident of geometry, and it is already tuned to its
  optimum (`birth_mechanism.md` finds `b_weight` non-monotonic with its earliest
  outer birth at 0.80π).
- **Concentrating the weight raises `R`; spreading it lowers `R`.** Anything
  anti-foveal necessarily spreads ρ, and under the shipped gated walker law
  (`K·R^3` drift) low coherence means a walker that hardly steers. So an
  anti-foveal weight is *doubly* penalised: it removes the outer equilibria
  deterministically, and it removes the walker's ability to swing onto them
  anyway. A foveal weight does the reverse on both counts. This is worth keeping
  in mind for the still-open commitment-signal question in
  [../TODO.md](../TODO.md) — the R-lift argument for a foveal `angle_weight` and
  the outward-bias argument against an anti-foveal one are the same argument.

## What this leaves open

The locust gap is therefore still where
[../walker_analysis/three_target_findings.md](../walker_analysis/three_target_findings.md)
put it — **in the recapture mechanism, not in the perception weighting** — and
this analysis narrows it further: no reweighting of the frontal visual field can
close it, because the required bias is not a function of egocentric angle.
Directions that remain open, in rough order of how well they fit the model:

1. **The untried locust refit.** `three_target_findings.md` flags that the
   locust was never re-run at the fly's refit knobs (`K=2`, pushed-out
   `a_warp`), which de-biased the fly from ~55% to ~45% centre. That is a
   direct handle on the recapture and costs nothing to test.
2. **Configuration-sensitive weighting** — a weight on the *neural* angle after
   consensus formation, or an Ising coupling that saturates with the number of
   aligned neighbours, would be able to express "the flanked target". This is a
   change to the coupling, not to `PerceptionModel`.
3. **Species-different `b_warp` / blind spot.** Locusts have far wider visual
   fields than flies; `b_warp` is already the second-bifurcation knob, and the
   rear cutoff is the one thing shown here to help outer commitment.

The two families were **removed from the model** once this was settled — a
negative result does not earn permanent machinery — but they are kept intact in
[anti_foveal.py](anti_foveal.py). They are the only anti-foveal shapes ever
written for this model, `lin_dip` is a clean minimal perturbation of uniform
weight, and having them means the next "what if the front were under-weighted?"
starts from a measurement rather than from scratch.

## Reproduce

```
python weighting_analysis/outward_bias.py all
python weighting_analysis/outward_bias.py cascade events --regenerate
python weighting_analysis/anti_foveal_selftest.py     # the families' numerics
```

Each stage caches its numeric result in `_cache_outward_<stage>.npz` behind a
JSON fingerprint of every input that affects it, so re-running to adjust a plot
recomputes nothing. Worker count from [../parallel_config.py](../parallel_config.py).
