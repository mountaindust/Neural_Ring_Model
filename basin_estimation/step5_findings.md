# Step 5 findings: multistability, γ-folds, and what bounds a basin

A self-contained writeup of what Step 5 turned up about the slow-manifold
basin structure of the NBM with VM-k055 two circles. Aimed at a reader
who knows basic stability analysis (fixed points, Jacobian eigenvalues,
stable/unstable classifications) but is new to:

- folds (saddle-node bifurcations on a slow manifold),
- the Poincaré-Hopf theorem on the circle,
- effective potentials V(θ),
- basin-attribution bisection vs. sign-change bisection.

We introduce each idea as it comes up.

## 1. What we wanted to compute

The full state space of the model is (γ_re, γ_im, θ) ∈ ℝ² × S¹. A
**stable self-consistent equilibrium** is a fixed point of the coupled
deterministic dynamics where the observer is facing its consensus
direction. The set of states (γ, θ) that flow to a particular stable
under the deterministic dynamics is the **basin of attraction** of that
stable.

For visualization, we project this 3D basin onto the heading axis θ.
That gives us a 1D "basin in θ": the range of headings that, paired
with the appropriate γ-relaxation, end up converging to a particular
stable.

## 2. The slow manifold and the reduced θ-dynamics

The walker's γ relaxes much faster than θ moves (γ-time-scale ~1;
θ-time-scale set by K, also ~1, but with strong drift only when γ is
off the consensus direction). The **slow manifold** is the surface in
state space where γ has reached its quasi-steady value γ_eq(θ) for the
instantaneous θ:

$$γ_{eq}(θ) = \text{(stable γ-fixed-point of }d γ /d t\text{ at this θ)}$$

The **reduced 1D ODE** on the slow manifold describes how θ evolves
while γ tracks along:

$$\dot θ = f(θ) := K \cdot R(θ) \cdot \sin\!\bigl(\text{ego\_angle}(θ)\bigr),$$

where R(θ) = |γ_eq(θ)| and ego_angle(θ) is the inverse neural-mapping
of arg(γ_eq(θ)). This is just dθ/dt, but evaluated *on* the slow
manifold.

## 3. The effective potential V(θ)

A 1D ODE on a smooth manifold is automatically a **gradient system**:
there always exists a scalar function V such that

$$\dot θ = -\frac{dV}{dθ}.$$

Constructing it is just integration:

$$V(θ) = -\int^{θ}\! f(s)\, ds + \text{const}.$$

Why this matters: V is exactly the *effective potential* the walker is
descending. Equilibria of dθ/dt = f(θ) are critical points of V:

- **stable** equilibrium (f' < 0) ↔ **local minimum** of V
- **unstable** equilibrium (f' > 0) ↔ **local maximum** of V

Sketch — V(θ) for a single stable + single saddle on the circle:

```
V(θ)
 |               
 |        _____________
 |       /             \
 |      /               \
 |     /                 \
 |____/                   \____
 +----+---------+---------+----  θ
    saddle    stable   saddle
    -π          0        +π
                    (same point — S¹)
```

The walker would roll downhill toward θ=0, sit at the bottom, and only
escape if noise pushed it over the V-barrier to either side.

In a 2D autonomous ODE you'd typically *not* have a gradient structure
(only those with a Lyapunov function are gradient). But on the slow
manifold, the reduction to 1D in θ gives us one for free. This is one
of the reasons the slow-manifold projection is so useful.

## 4. The Poincaré-Hopf constraint on a circle

The reduced dynamics live on the circle S¹. There's a topological law
that constrains how many stable and unstable equilibria can coexist:

**Poincaré-Hopf theorem (1D circle case).** For a smooth vector field
on S¹ with only isolated zeros,

$$\sum_\text{zeros}\!\text{(index)} \;=\; χ(S^1) \;=\; 0,$$

where each zero has index +1 (unstable, f' > 0) or −1 (stable, f' < 0)
and χ is the Euler characteristic. Setting indices,

$$(\#\text{unstable}) - (\#\text{stable}) = 0,$$

i.e. **stable and unstable equilibria must be equal in number** on S¹.

Why intuitively: between two stable equilibria, f must cross zero with
positive slope (an unstable in between), so they alternate around the
loop.

Sketch — three stable + three unstable, alternating:

```
        stable
          ●
    -π ··········· +π
         /  \
   unstable    unstable
       \      /
      stable ● ● stable
              ╲╱
              unstable
            (at 0 say)
```

(Cartoon: imagine walking CCW around the circle starting at the leftmost
stable; you'd encounter unstable, stable, unstable, stable, unstable,
back to start.)

## 5. The simple case — (0.5, 0)

The observer at (0.5, 0) is close to the symmetry line between the two
targets but well in front of them. The model has exactly one stable SC
equilibrium at θ_sc = 0 (heading straight at the targets), and by
Poincaré-Hopf, exactly one unstable somewhere — physically, at θ = ±π
(facing directly away).

Our truncated θ-scans from θ_sc = 0 do exactly what you'd expect:

- CCW scan reaches the saddle at θ = +π and terminates.
- CW scan reaches the saddle at θ = −π (= +π) and terminates.
- V(θ) has a single well at θ=0 and a single barrier at ±π.

Half-widths Δθ⁻ = Δθ⁺ = π. Basin is "everything except the saddle."

But: **sc_equilib reports only the stable, not the saddle** at ±π. The
saddle is missing from its output. We'll explain why in §8.

## 6. The harder case — (1.2, 0), 3 stable equilibria

Now sc_equilib reports 3 stable SC eqs (at −0.66, 0, +0.66) and 2
unstable (at ±0.367). Poincaré-Hopf says there must be 3 unstable, so
one is missing from sc_equilib.

When we ran truncated θ-scans from each of the 3 stables — going CCW
and CW from each — we hit things that aren't saddles. Specifically: in
several scan directions, γ_eq(θ) **catastrophically jumps** between
consecutive θ samples (|Δγ| ~ 0.5 in a single step of dθ ~ 0.03). These
jumps are not numerical glitches — they're real **γ-folds**.

## 7. γ-folds: what they are

At any fixed θ, the model can have multiple γ-equilibria of the γ-only
dynamics. Each is a critical point of F̂(γ; θ) (see
[free_energy_derivation.md](free_energy_derivation.md)):

- A γ-local-minimum (γ-stable) — call it γ_eq.
- A γ-saddle (γ-unstable) — call it γ_sad.

As θ varies smoothly, both γ_eq and γ_sad trace out smooth paths in
the complex γ-plane. A **γ-branch** is one of these paths. The slow
manifold is built from γ-branches (specifically, the γ-stable ones).

A **fold** (saddle-node bifurcation) is the θ value where γ_eq and
γ_sad on a given branch *merge and disappear* — past the fold, neither
exists.

Sketch — γ_eq(θ) (R, say) along the slow manifold across a fold:

```
R              ╱╲                    ╱╲
 │            ╱   ╲                  ╱   ╲
 │           ╱     ╲                ╱     ╲
 │          ╱       ╲              ╱       ╲
 │ ────────●         ●─catastrophic●         ●────────
 │       branch A    │     jump    │       branch B
 │                fold-θ−        fold-θ+
 +──────────────────┴──────────────┴────────────────  θ
```

For θ in between the two folds, no smooth γ-branch exists at all (or
the branch shifts to a completely different shape). Our scan starts on
branch A; when we reach fold-θ−, γ_eq(θ) on branch A has vanished, so
the warm-started LSODA from the last branch-A γ finds itself out of
equilibrium and relaxes to whatever γ-equilibrium *is* nearby —
typically a γ_eq on branch B. The result: the scan's recorded γ_eq
jumps from a branch-A value to a branch-B value in a single dθ step.

This is what we detect as a "γ-fold event": |Δγ| between consecutive
scan steps is much larger than typical (we use both a relative
threshold — 8× the median step — and an absolute threshold of 0.4).

**The fold is itself a basin boundary** in (θ, γ) space. The walker
sitting on branch A at θ = fold-θ−, if perturbed slightly past the
fold, no longer has branch-A's γ to relax to — its γ catastrophically
jumps to branch B. Once on branch B, the θ-dynamics may carry it to a
completely different stable.

## 8. Why sc_equilib misses some saddles

sc_equilib finds SC equilibria: states where heading equals consensus.
In coordinates, this means γ ∈ ℝ_{>0} (γ on the positive real axis in
neural space). The code's `sc_equilib` enforces `0.01 < R_eq < 1.0`.

Two reasons saddles can be missed:

**(a) Saddles on the wrong γ-axis.** Some unstable equilibria of the
*coupled* (γ, θ) system have γ on the *negative* real axis — γ =
−R + 0j with R > 0 — corresponding to the observer facing directly
away from the consensus. These are physically meaningful equilibria,
but `sc_equilib`'s `R_eq > 0` filter excludes them. The saddle at ±π
at the 1-stable point (0.5, 0) is exactly this case.

**(b) Saddles on a different γ-branch.** At a multistable point, the
SC saddles (the ±0.367 ones at (1.2, 0)) live on the *central* γ=R+0j
branch — the same branch as the central stable. The *side* stables
(±0.66) sit on **different γ-branches**. When we scan from a side
stable with warm-start γ-continuation, we trace its own branch — which
never passes through γ=R+0j at θ = ±0.367. So the side-branch scan
never encounters those saddles; on its branch, what bounds the basin
is a γ-fold instead.

This is genuinely the right answer for the *walker's* perspective: a
walker starting at (+0.66, γ_side_branch) and getting perturbed
toward the center doesn't pass through γ = R+0j at θ = +0.367; it
follows the side-branch γ until the fold catches it.

## 9. The basin picture at (1.2, 0)

Putting it together:

```
   −π                              0                              +π
    ●═══════════●─────────✗════════●════════✗─────────●═══════════●
   sad?    left-stable   left-fold center right-fold right-stable  sad?
                                                                      
    └────  left basin  ────┘└──center basin──┘└─── right basin ───┘
```

Each basin is an arc bounded by either a **saddle** (●) at the ends or
a **γ-fold** (✗) in between. The central stable's basin is
saddle-bounded going CCW (to +π) and to a fold going CW; the side
stables' basins are saddle-bounded going *outward* (to ±π) and
fold-bounded going *inward* (toward each other). Our truncated scans
correctly detect both kinds of boundary.

The numerical issue we ran into in T4 was that for the *central
stable*, going CCW and going CW both can wind around all the way to
±π. The dynamics' sensitivity to which γ-branch the warm-start picks
up after the first wrap can split CCW and CW onto different effective
branches, giving an apparent asymmetry. This is a quirk of how we
chose to scan, not a feature of the underlying physics — restricting
attention to the fold-side (the inner boundaries) gives perfectly
y-symmetric results.

## 10. Two notions of "Method B" — saddle-finding via bisection

The planning notes mentioned "Method B" as bisection between two known
stable equilibria. Step 5 turned up that there are actually **two
distinct things** that could be called bisection here, and they answer
different questions.

### 10a. Sign-change bisection

Given two stable θs and a γ-branch (chosen via warm-start), bisect on
the sign of f(θ) along the branch:

- Probe a midpoint θ_m.
- Use warm-start γ-continuation to compute f(θ_m) on the chosen branch.
- Compare to the sign of f at the brackets.
- Continue bisecting toward where f changes sign on the branch.

This finds the **first zero of f on the chosen branch** between the
brackets. If that branch contains a saddle, you'll find it. If it
contains a fold (no zero of f), the bisection has nothing to converge
to and will misbehave.

### 10b. Basin-attribution bisection

Given two stable θs, bisect on which basin a midpoint lands in:

- Probe a midpoint θ_m with appropriate initial γ.
- **Run the deterministic dynamics from (θ_m, γ_init) until convergence.**
- See which stable the trajectory lands at.
- Bisect toward the change-over.

This finds the **dynamical boundary between the two basins** in θ —
whether it's a saddle, a fold, or any other discontinuity. The
boundary is the θ at which an arbitrarily small displacement flips the
endpoint.

Sketch of basin-attribution bisection in θ between two stables θ_L and θ_R:

```
                                     Bisection iter
 θ_L ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━● θ_R
       [trajectory lands at θ_L]       [trajectory lands at θ_R]
                       ↓ probe midpoint, evolve
                     │
                     ●           lands at θ_R, so bracket is now (θ_L, midpoint)
                     ↓ probe new midpoint
                ●                  lands at θ_L, so bracket is (this, prev)
                     ↓ ... converges to the basin boundary
```

In our (1.2, 0) example with the right and central stables, the
basin-attribution bisection would find the fold at θ ≈ +0.479,
because a walker starting just left of the fold (with appropriate γ)
ends up at the central stable, and starting just right ends up at the
right stable.

The sign-change bisection I implemented in Step 5, by contrast,
finds θ ≈ +0.331 — close to the *central γ-branch's* SC saddle at
+0.367. It's the right answer for "where does f change sign on the
central γ-branch" but not for "where does the walker's basin change."
Different question.

### Implication for Step 7

Step 7 (proper basin boundary detection across γ-branches) will need to
implement basin-attribution bisection, not just sign-change. It's
harder — each probe requires integrating the full coupled dynamics to
convergence — but it's the right ground truth for what the walker
actually experiences.

## 11. What this means going forward

- **F̂ derivation (Step 1–2) is solid.** Nothing in this writeup
  changes anything there.
- **γ-Langevin (Step 3) is solid.** The fast-relaxation regime we
  validated is exactly the regime relevant for the slow manifold.
- **The slow-manifold θ-scan (Step 4) is solid.** It does what it's
  supposed to: track one γ-branch.
- **The basin extraction (Step 5) is solid for fold-bounded
  scenarios** but the saddle-side at multistable points is
  numerically delicate — we need basin-attribution bisection (Step
  7) for full robustness.
- **Step 6 (γ-saddle finding at fixed θ and ΔF_γ evaluation) is
  next.** That work is at fixed θ, so γ-fold issues across θ don't
  apply — we're looking at the γ-Hessian landscape at a single SC
  equilibrium.
- **Step 7 will need basin-attribution bisection** as the proper way
  to find the θ-basin boundaries across multiple γ-branches.
- **Step 8 (MC ground truth) will validate the basin sizes** by
  running noise simulations and seeing where walkers actually end
  up — this is the empirical version of basin-attribution bisection.

## 12. Summary in one paragraph

In this model, the slow manifold projecting the (γ, θ) dynamics down to
1D heading dynamics is not a global object — it's a *union of γ-branches*
glued at γ-folds. A basin of attraction in θ around a stable SC
equilibrium is an arc on one γ-branch, bounded by either a saddle (a
smooth zero of f on the same branch) or a γ-fold (a discontinuity where
the branch terminates and γ jumps). The effective potential V(θ) = −∫f
is meaningful and smooth *within* a single γ-branch but has
discontinuities at the folds. The Poincaré-Hopf theorem (#stable =
#unstable on S¹) holds within each branch but not across the branch
union. sc_equilib's saddle list is necessarily incomplete because it
restricts to γ = R + 0j and to one γ-branch at a time. Two different
bisection methods can converge to different θ values depending on
whether you're tracking sign changes on one branch or actual basin
attribution under the full coupled dynamics; the latter is what the
walker experiences.
