# Free energy F̂(γ) for NBM — derivation

**Goal.** Derive a Lyapunov function F̂(γ; θ, focal_loc) for the
deterministic NBM γ-dynamics, in a form usable for (a) computing
γ-basin barrier heights ΔF_γ, and (b) calibrating γ-Langevin noise
amplitude in Step 3.

**Independence note.** This derivation uses the user's Hamiltonian
(Eq. `eq:H_orig` in the writeup shared in session) and the Glauber
route to the γ-ODE (which matches `decision_model.py:2225`). It does
**not** refer to any free-energy expression in
`Hamiltonian ideas.tex` or other prior notes. Step 2 will validate the
result numerically.

**Convention pinned in planning session:** F̂ is a real function on
(γ_re, γ_im) ∈ ℝ², defined such that the deterministic γ-dynamics is
exact gradient flow:

$$\dot γ_x = -\partial_{γ_x} F̂(γ), \qquad \dot γ_y = -\partial_{γ_y} F̂(γ).$$

## 1. Notation (matching the writeup and the code)

- `k` = number of visible targets (`neur_angles.size` in the code).
- `N` = total number of spins. Drops out in the large-N limit but
  resurfaces in Step 3 as the γ-noise amplitude.
- `ρⱼ` = relative group size, normalized by `Σⱼ ρⱼ = 1` (matches
  `PerceptionModel._get_target_signals`'s `rho`).
- `θ̂ⱼ` = neural angle of target j (egocentric angle through the
  warping; `neural_angles` in the code).
- `v̂ⱼ = (cos θ̂ⱼ, sin θ̂ⱼ)` = unit vector at neural angle θ̂ⱼ.
- `γ = (γ_x, γ_y) ∈ ℝ²` ≡ `γ_x + i γ_y ∈ ℂ`. R = |γ|, Θ̂ = arg(γ).
- `T` = Ising temperature (`self.T` in the code).
- Define `uⱼ(γ) := 2k · v̂ⱼ · γ / T = 2k(γ_x cos θ̂ⱼ + γ_y sin θ̂ⱼ)/T`.
  Sigmoid σ(u) := 1/(1+e^(−u)).

The γ-ODE from the user's writeup (with the missing sum restored):

$$\dot γ = \sum_{j=1}^k ρⱼ\, e^{i θ̂ⱼ}\, σ(uⱼ(γ)) \;-\; γ.$$

In (γ_x, γ_y):

$$\dot γ_x = \underbrace{\sum_{j=1}^k ρⱼ\cos θ̂ⱼ\, σ(uⱼ(γ))}_{Φ_x(γ)} - γ_x, \qquad
  \dot γ_y = \underbrace{\sum_{j=1}^k ρⱼ\sin θ̂ⱼ\, σ(uⱼ(γ))}_{Φ_y(γ)} - γ_y.$$

θ (the observer heading) and `focal_loc` enter only through the
perception-determined quantities θ̂ⱼ and ρⱼ. F̂ depends on these
implicitly.

## 2. The γ-flow is gradient (curl-free Jacobian)

If `Φ` has a symmetric Jacobian, then `-(Φ − γ)` is a gradient of some
scalar field.

$$\partial_{γ_y} Φ_x = \sum_j ρⱼ \cos θ̂ⱼ \cdot σ'(uⱼ) \cdot \tfrac{2k \sin θ̂ⱼ}{T}
                    = \frac{2k}{T}\sum_j ρⱼ\, \sin θ̂ⱼ \cos θ̂ⱼ\, σ(uⱼ)(1-σ(uⱼ)).$$

$$\partial_{γ_x} Φ_y = \sum_j ρⱼ \sin θ̂ⱼ \cdot σ'(uⱼ) \cdot \tfrac{2k \cos θ̂ⱼ}{T}
                    = \frac{2k}{T}\sum_j ρⱼ\, \sin θ̂ⱼ \cos θ̂ⱼ\, σ(uⱼ)(1-σ(uⱼ)).$$

Equal — so `Φ` is curl-free, and `dγ/dt = -∇F̂(γ)` for some F̂.

## 3. Integrate to find F̂

We need an antiderivative satisfying:

$$\partial_{γ_x} F̂ = γ_x - Φ_x(γ).$$

The `γ_x` piece integrates to `½ γ_x²`. For `Φ_x`, change variables in
the integral over `γ_x`:

$$\int \frac{ρⱼ \cos θ̂ⱼ}{1+e^{-uⱼ(γ)}} dγ_x
   = ρⱼ \cos θ̂ⱼ \cdot \frac{T}{2k\cos θ̂ⱼ} \cdot \ln(1+e^{uⱼ})
   = \frac{T ρⱼ}{2k} \ln(1+e^{uⱼ(γ)}).$$

Repeating for the `γ_y` partial gives the same antiderivative
(consistency follows from step 2). The candidate F̂:

$$\boxed{\;F̂(γ) \;=\; \tfrac{1}{2}|γ|^2 \;-\; \frac{T}{2k}\sum_{j=1}^k ρⱼ\, \ln\!\bigl(1 + e^{uⱼ(γ)}\bigr), \qquad uⱼ(γ) = \tfrac{2k}{T}\, v̂ⱼ\cdot γ\;}$$

**Verification:**

$$\partial_{γ_x} F̂ = γ_x - \frac{T}{2k}\sum_j ρⱼ \cdot \frac{e^{uⱼ}}{1+e^{uⱼ}} \cdot \frac{2k \cos θ̂ⱼ}{T}
                  = γ_x - \sum_j ρⱼ \cos θ̂ⱼ\, σ(uⱼ)
                  = γ_x - Φ_x. \quad\checkmark$$

So `dγ/dt = -∇F̂(γ)` exactly. F̂ is the Lyapunov function for the
deterministic γ-flow.

## 4. Equivalent log-cosh form (symmetric / standard mean-field)

Using `ln(1+e^u) = ln(2 cosh(u/2)) + u/2`:

$$F̂(γ) = \tfrac{1}{2}|γ|^2 - \frac{T}{2k}\sum_j ρⱼ \ln\!\bigl(2\cosh(uⱼ/2)\bigr) - \tfrac{1}{2}\,μ \cdot γ$$

where `μ := Σⱼ ρⱼ v̂ⱼ` is the perceptual centroid. The two forms differ
by `-½ μ·γ + const`, which is a linear (gauge) shift — same F̂ up to
constant, but the log-cosh form is more symmetric and matches the
"standard" mean-field expression seen in textbooks. We will use the
**log-(1+exp)** form everywhere in code because it's the direct
antiderivative.

## 5. Cross-check against mean-field free energy F = ⟨H⟩ − T·S

Independent route (validation of section 3, not used downstream).

Mean-field ⟨H⟩ at fixed `{nⱼ}` with the large-N rewriting from the
writeup (dropping the `k·Σ nⱼ` term that's `O(N⁰)` vs. the leading
`O(N¹)` term):

$$⟨H⟩/N \approx -k\, R^2 = -k|γ|^2.$$

Entropy of independent Bernoulli spins with within-group probabilities
`qⱼ = nⱼ/ρⱼ`:

$$S/N = -\sum_j ρⱼ\bigl[ qⱼ \ln qⱼ + (1-qⱼ)\ln(1-qⱼ)\bigr].$$

Constrained minimum over `{nⱼ}` at fixed γ gives
`qⱼ*(γ) = σ(uⱼ(γ))`. Substituting back:

$$(F_{\text{mf}})_{\text{per spin}}\big|_{n_j^*(γ)} \;=\; -k|γ|^2 \;-\; T\sum_j ρⱼ \ln(1+e^{-uⱼ}) \;-\; T\!\sum_j ρⱼ(1-qⱼ^*) uⱼ.$$

The last term: `T(1-qⱼ*)uⱼ = T·(1-σ(uⱼ))·uⱼ`. Using
`Σⱼ (ρⱼ - nⱼ*) v̂ⱼ · γ = μ·γ - |γ|²` (since `Σⱼ nⱼ* v̂ⱼ = γ` at the
constrained minimum):

$$T\sum_j ρⱼ(1-qⱼ^*) uⱼ = 2k(μ·γ - |γ|^2).$$

So:

$$(F_{\text{mf}})_{\text{per spin}}\big|_{n_j^*} = k|γ|^2 - 2k μ·γ - T\sum_j ρⱼ \ln(1+e^{-uⱼ}).$$

Compare to `2k · F̂(γ)` using the log-(1+exp) → log-(1+exp(−u)) − u
identity `ln(1+eᵘ) = ln(1+e^(−u)) + u`:

$$2k\, F̂(γ) = k|γ|^2 - T\sum_j ρⱼ[\ln(1+e^{-uⱼ}) + uⱼ]
            = k|γ|^2 - T\sum_j ρⱼ \ln(1+e^{-uⱼ}) - 2k μ·γ.$$

Equal. ✓

So `F̂(γ) = (F_mf per spin at constrained minimum) / (2k)`, up to an
additive constant. The `1/(2k)` is normalization that makes
`dγ/dt = -∇F̂` exact; the physical free energy per spin is `2k · F̂`.

This also makes contact with the user's identity `R² ≈ -H/(kN)` from
the writeup: at the constrained minimum and large N, the
energy-per-spin part of `2k · F̂` is `k|γ|² = k R²`, matching `-H/N` =
`k R²` from the writeup's eqn after the line "as N gets large".

## 6. Gradient and Hessian (for use in Steps 2, 5, 6)

Gradient:

$$\nabla F̂(γ) = γ - \sum_j ρⱼ\, σ(uⱼ(γ))\, v̂ⱼ.$$

Hessian:

$$H_{F̂}(γ) = I_2 - \frac{2k}{T}\sum_j ρⱼ\, σ(uⱼ)(1-σ(uⱼ))\, v̂ⱼ v̂ⱼ^T.$$

At any γ-equilibrium γ* (where `∇F̂(γ*) = 0`), the eigenvalues of
`H_{F̂}(γ*)` classify stability:
- Both positive: γ* is a γ-local-minimum (stable γ-eq).
- One negative: γ* is a γ-saddle (basin boundary in γ at fixed θ).
- Both negative: γ-local-maximum (unlikely in practice; would indicate
  a γ-repeller).

Note the connection to the writeup's stability scalar
`A = (k/(2T))·Σⱼ ρⱼ sech²(k γ⃗·v̂ⱼ/T)·(v̂ⱼ·n̂*)²`:
using `σ(u)(1-σ(u)) = ¼ sech²(u/2)` and `u/2 = k γ⃗·v̂ⱼ/T`,

$$(H_{F̂})_{nn} = 1 - \frac{2k}{T}\sum_j ρⱼ \cdot \tfrac{1}{4}\operatorname{sech}^2(k γ⃗·v̂ⱼ/T) \cdot (v̂ⱼ·n̂^*)^2
              = 1 - A.$$

So the writeup's `A < 1` stability criterion is exactly the statement
that the Hessian eigenvalue along the n̂* (perpendicular to γ*)
direction is positive. The eigenvalue along γ̂* is always positive at
self-consistent equilibria for any reasonable rule. Consistency
check. ✓

## 7. Effective noise temperature for γ-Langevin (preview of Step 3)

For Step 3 we will add Gaussian noise to `dγ/dt` so it becomes:

$$dγ = -\nabla F̂(γ)\, dt + \sqrt{2 D}\, dW$$

with `dW` a standard 2D Wiener increment. The Boltzmann distribution
of γ in the stationary state is

$$P(γ) \propto \exp\!\bigl(-F̂(γ)/D\bigr).$$

To match the underlying spin model (large-N expansion of binomial
fluctuations in {nⱼ}, projected onto γ), we want:

$$P(γ) \propto \exp\!\bigl(-N \cdot 2k\, F̂(γ)/T\bigr) \quad \Rightarrow \quad
  D = \frac{T}{2kN}.$$

So:

$$\boxed{\;D_\gamma = \frac{T}{2 k N}\;}$$

`N` is the effective spin count. For Step 3 we will take it as a
tunable parameter; in the `N → ∞` limit, `D → 0` and γ-dynamics is
deterministic (recovering the mean-field model). For Monte Carlo
vetting we pick a moderate `N` so that escape events occur on a
tractable timescale.

The derivation of `D = T/(2kN)` from the binomial fluctuations is
standard mean-field stat mech; I'll write it out properly in Step 3
where it's load-bearing. For now it's a stated result to be verified
numerically (Step 3's stationary-distribution test).

## 8. What stays implicit in F̂

- θ-dependence of θ̂ⱼ (through the warping mapping
  `PerceptionModel.get_neural_angle`).
- (θ, focal_loc)-dependence of ρⱼ (through perception with weighting,
  blocking, and visibility).
- Discontinuities in F̂(γ) as a function of (θ, focal_loc): when a
  target enters/leaves visibility, one term of the sum
  appears/disappears, so F̂ has a discontinuity in θ at that event.
  This is the "occlusion discontinuity" we plan to detect in Step 7.
  Inside any region where the visible-target set is constant, F̂ is
  C^∞ in (γ, θ, focal_loc) — folds happen, but they are smooth
  features of F̂, not jumps.

In code, `F̂(γ; θ, focal_loc)` will be computed by calling
`PerceptionModel.get_neural_signals(focal_angle=θ, focal_loc=...)` to
get `(neural_angles, rho)`, then evaluating the closed-form expression
above. No new perception code needed.

## 9. Summary — what Step 2 will verify

1. `np.gradient(F̂)` (finite-difference) matches the analytical
   gradient in section 6 to ~1e-8 at random (γ, θ, focal_loc) points.
2. At γ-equilibria from `gamma_equilib(focal_angle=θ)`, `∇F̂(γ*) ≈ 0`
   to ~1e-6.
3. Hessian eigenvalues at γ-equilibria match the stability labels
   from `_discrim_A` (and from `_discrim_coupled` after we project
   out the θ-direction).
4. The identity `2k·F̂(γ) = (F_mf per spin at constrained minimum) +
   const` from section 5 holds numerically: sample {n_j(γ)} from
   `n_j*(γ)`, compute mean-field F directly and compare.

Once these pass, F̂ is trusted for use in Step 6 (ΔF_γ evaluation) and
Step 3 (calibrating γ-Langevin noise via the `D_γ = T/(2kN)` relation).
