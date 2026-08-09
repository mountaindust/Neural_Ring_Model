# Free energy F̂(γ) for NBM — derivation

**Goal.** Derive a Lyapunov function F̂(γ; θ, focal_loc) for the
deterministic NBM γ-dynamics, in a form usable for (a) computing
γ-basin barrier heights ΔF_γ, and (b) calibrating the γ-Langevin noise
amplitude.

**Independence note.** This derivation uses the user's Hamiltonian
(Eq. `eq:H_orig` in the writeup shared in session) and the Glauber
route to the γ-ODE (which matches `decision_model.py`'s `dgamma_dt`). It
does **not** refer to any free-energy expression in
`Hamiltonian ideas.tex` or other prior notes.

**Status — validated.** Every numerical check in §9 passed: the
analytical gradient matches finite differences (~1e-8) and equals
dγ/dt; ∇F̂≈0 at γ-equilibria; the Hessian matches the full-block
stability criteria (`_discrim_reduced`/`_discrim_coupled` — but see
§6.1 for why the scalar `A < 1` alone is necessary-not-sufficient); and
`2β·F̂` equals the mean-field free energy per spin (measured in units of
`k_B·temp`). The barrier height ΔF_γ and the noise calibration
`D=1/(2βN)` built on this
are summarized in [basins_of_attraction.md](basins_of_attraction.md).

**Convention pinned in planning session:** F̂ is a real function on
(γ_re, γ_im) ∈ ℝ², defined such that the deterministic γ-dynamics is
exact gradient flow:

$$\dot γ_x = -\partial_{γ_x} F̂(γ), \qquad \dot γ_y = -\partial_{γ_y} F̂(γ).$$

## 1. Notation (matching the writeup and the code)

- `k` = number of visible targets (`neur_angles.size` in the code). It
  indexes the sum only — it does **not** set the coupling strength.
- `N` = total number of spins. Drops out in the large-N limit but
  resurfaces as the γ-noise amplitude `D=1/(2βN)` (§7).
- `ρⱼ` = relative group size, normalized by `Σⱼ ρⱼ = 1` (matches
  `PerceptionModel._get_target_signals`'s `rho`).
- `θ̂ⱼ` = neural angle of target j (egocentric angle through the
  warping; `neural_angles` in the code).
- `v̂ⱼ = (cos θ̂ⱼ, sin θ̂ⱼ)` = unit vector at neural angle θ̂ⱼ.
- `γ = (γ_x, γ_y) ∈ ℝ²` ≡ `γ_x + i γ_y ∈ ℂ`. R = |γ|, Θ̂ = arg(γ).
- `β` = neural Boltzmann factor `β = 𝓔/(k_B·temp)` (`self.beta` in the
  code), with `𝓔` the energy scale of the Hamiltonian. Energies below are
  measured in units of `𝓔`, so the temperature is `1/β`.
- Define `uⱼ(γ) := 2β · v̂ⱼ · γ = 2β(γ_x cos θ̂ⱼ + γ_y sin θ̂ⱼ)`.
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

$$\partial_{γ_y} Φ_x = \sum_j ρⱼ \cos θ̂ⱼ \cdot σ'(uⱼ) \cdot 2β \sin θ̂ⱼ
                    = 2β\sum_j ρⱼ\, \sin θ̂ⱼ \cos θ̂ⱼ\, σ(uⱼ)(1-σ(uⱼ)).$$

$$\partial_{γ_x} Φ_y = \sum_j ρⱼ \sin θ̂ⱼ \cdot σ'(uⱼ) \cdot 2β \cos θ̂ⱼ
                    = 2β\sum_j ρⱼ\, \sin θ̂ⱼ \cos θ̂ⱼ\, σ(uⱼ)(1-σ(uⱼ)).$$

Equal — so `Φ` is curl-free, and `dγ/dt = -∇F̂(γ)` for some F̂.

## 3. Integrate to find F̂

We need an antiderivative satisfying:

$$\partial_{γ_x} F̂ = γ_x - Φ_x(γ).$$

The `γ_x` piece integrates to `½ γ_x²`. For `Φ_x`, change variables in
the integral over `γ_x`:

$$\int \frac{ρⱼ \cos θ̂ⱼ}{1+e^{-uⱼ(γ)}} dγ_x
   = ρⱼ \cos θ̂ⱼ \cdot \frac{1}{2β\cos θ̂ⱼ} \cdot \ln(1+e^{uⱼ})
   = \frac{ρⱼ}{2β} \ln(1+e^{uⱼ(γ)}).$$

Repeating for the `γ_y` partial gives the same antiderivative
(consistency follows from step 2). The candidate F̂:

$$\boxed{\;F̂(γ) \;=\; \tfrac{1}{2}|γ|^2 \;-\; \frac{1}{2β}\sum_{j=1}^k ρⱼ\, \ln\!\bigl(1 + e^{uⱼ(γ)}\bigr), \qquad uⱼ(γ) = 2β\, v̂ⱼ\cdot γ\;}$$

**Verification:**

$$\partial_{γ_x} F̂ = γ_x - \frac{1}{2β}\sum_j ρⱼ \cdot \frac{e^{uⱼ}}{1+e^{uⱼ}} \cdot 2β \cos θ̂ⱼ
                  = γ_x - \sum_j ρⱼ \cos θ̂ⱼ\, σ(uⱼ)
                  = γ_x - Φ_x. \quad\checkmark$$

So `dγ/dt = -∇F̂(γ)` exactly. F̂ is the Lyapunov function for the
deterministic γ-flow.

## 4. Equivalent log-cosh form (symmetric / standard mean-field)

Using `ln(1+e^u) = ln(2 cosh(u/2)) + u/2`:

$$F̂(γ) = \tfrac{1}{2}|γ|^2 - \frac{1}{2β}\sum_j ρⱼ \ln\!\bigl(2\cosh(uⱼ/2)\bigr) - \tfrac{1}{2}\,μ \cdot γ$$

where `μ := Σⱼ ρⱼ v̂ⱼ` is the perceptual centroid. The two forms differ
by `-½ μ·γ + const`, which is a linear (gauge) shift — same F̂ up to
constant, but the log-cosh form is more symmetric and matches the
"standard" mean-field expression seen in textbooks. We will use the
**log-(1+exp)** form everywhere in code because it's the direct
antiderivative.

## 5. Cross-check against mean-field free energy F = ⟨H⟩ − (1/β)·S

Energies are in units of `𝓔`, so the temperature is `1/β` and the
equations below are the old energy-unit ones divided by `k_B·temp`.

Independent route (validation of section 3, not used downstream).

Mean-field ⟨H⟩ at fixed `{nⱼ}` with the large-N rewriting from the
writeup (dropping the `𝓔·Σ nⱼ` term that's `O(N⁰)` vs. the leading
`O(N¹)` term):

$$⟨H⟩/(N𝓔) \approx -R^2 = -|γ|^2.$$

Entropy of independent Bernoulli spins with within-group probabilities
`qⱼ = nⱼ/ρⱼ`:

$$S/N = -\sum_j ρⱼ\bigl[ qⱼ \ln qⱼ + (1-qⱼ)\ln(1-qⱼ)\bigr].$$

Constrained minimum over `{nⱼ}` at fixed γ gives
`qⱼ*(γ) = σ(uⱼ(γ))`. Substituting back:

$$β\,(F_{\text{mf}})_{\text{per spin}}\big|_{n_j^*(γ)} \;=\; -β|γ|^2 \;-\; \sum_j ρⱼ \ln(1+e^{-uⱼ}) \;-\; \sum_j ρⱼ(1-qⱼ^*) uⱼ.$$

The last term: `(1-qⱼ*)uⱼ = (1-σ(uⱼ))·uⱼ`. Using
`Σⱼ (ρⱼ - nⱼ*) v̂ⱼ · γ = μ·γ - |γ|²` (since `Σⱼ nⱼ* v̂ⱼ = γ` at the
constrained minimum):

$$\sum_j ρⱼ(1-qⱼ^*) uⱼ = 2β(μ·γ - |γ|^2).$$

So:

$$β\,(F_{\text{mf}})_{\text{per spin}}\big|_{n_j^*} = β|γ|^2 - 2β μ·γ - \sum_j ρⱼ \ln(1+e^{-uⱼ}).$$

Compare to `2β · F̂(γ)` using the log-(1+exp) → log-(1+exp(−u)) − u
identity `ln(1+eᵘ) = ln(1+e^(−u)) + u`:

$$2β\, F̂(γ) = β|γ|^2 - \sum_j ρⱼ[\ln(1+e^{-uⱼ}) + uⱼ]
            = β|γ|^2 - \sum_j ρⱼ \ln(1+e^{-uⱼ}) - 2β μ·γ.$$

Equal. ✓

So `F̂(γ) = (F_mf per spin at constrained minimum) / (2𝓔)`, up to an
additive constant. The `1/(2𝓔)` is normalization that makes
`dγ/dt = -∇F̂` exact; the free energy per spin in units of `k_B·temp` is
`2β · F̂`.

This also makes contact with the writeup's identity, which with the `k`
prefactor dropped from the Hamiltonian in favour of the energy scale `𝓔`
reads `R² ≈ -H/(𝓔N)`: at the constrained minimum and large N, the
energy-per-spin part of `2 · F̂` (in units of `𝓔`) is `|γ|² = R²`,
matching `-H/(N𝓔) = R²`.

## 6. Gradient and Hessian (used for ΔF_γ and the stability checks)

Gradient:

$$\nabla F̂(γ) = γ - \sum_j ρⱼ\, σ(uⱼ(γ))\, v̂ⱼ.$$

Hessian:

$$H_{F̂}(γ) = I_2 - 2β\sum_j ρⱼ\, σ(uⱼ)(1-σ(uⱼ))\, v̂ⱼ v̂ⱼ^T.$$

At any γ-equilibrium γ* (where `∇F̂(γ*) = 0`), the eigenvalues of
`H_{F̂}(γ*)` classify stability:
- Both positive: γ* is a γ-local-minimum (stable γ-eq).
- One negative: γ* is a γ-saddle (basin boundary in γ at fixed θ).
- Both negative: γ-local-maximum (unlikely in practice; would indicate
  a γ-repeller).

Note the connection to the writeup's stability scalar
`A = (β/2)·Σⱼ ρⱼ sech²(β γ⃗·v̂ⱼ)·(v̂ⱼ·n̂*)²`:
using `σ(u)(1-σ(u)) = ¼ sech²(u/2)` and `u/2 = β γ⃗·v̂ⱼ`,

$$(H_{F̂})_{nn} = 1 - 2β\sum_j ρⱼ \cdot \tfrac{1}{4}\operatorname{sech}^2(β γ⃗·v̂ⱼ) \cdot (v̂ⱼ·n̂^*)^2
              = 1 - A.$$

So the writeup's `A < 1` criterion is exactly the statement that the
Hessian's **tangential diagonal entry** `(H_{F̂})_{nn} = 1 − A` — the
curvature along n̂* (perpendicular to γ*) — is positive. But
`(H_{F̂})_{nn}` is a *directional curvature*, not an eigenvalue, unless
the R–Θ off-diagonal `(H_{F̂})_{γ̂ n̂}` vanishes. **`A < 1` is therefore
necessary but not sufficient for γ-stability.** The earlier version of
this note claimed sufficiency ("the eigenvalue along γ̂* is always
positive at self-consistent equilibria for any reasonable rule") — that
is **false off the mirror-symmetry axis and at low-R disordered
states**, where the missing radial and off-diagonal conditions bite.
See the correction in §6.1.

## 6.1 Correction (2026-07-07): `A < 1` is necessary but not sufficient

**What was wrong.** The §6 consistency check equated the scalar `A < 1`
with full γ-stability. It is only one of the *two* independent
conditions a symmetric 2×2 Hessian must satisfy to be positive definite.
`A < 1` is necessary but not sufficient; a genuine γ-saddle can satisfy
`A < 1`. Verified against the numeric fast block below.

**The Hessian is the fast γ-block.** At a self-consistent equilibrium
γ = R + 0j (so Θ̂ = arg γ = 0), the x-axis is radial (the R direction)
and the y-axis is tangential (the arg γ / n̂* direction). Writing
`w_k := (β/2)·ρ_k·sech²(β R cos θ̂_k) ≥ 0`, the Cartesian Hessian is

$$H_{F̂}(R{+}0j) = \begin{bmatrix} 1 - \sum_k w_k \cos^2 θ̂_k & -\sum_k w_k \cos θ̂_k \sin θ̂_k \\[2pt] -\sum_k w_k \cos θ̂_k \sin θ̂_k & 1 - \sum_k w_k \sin^2 θ̂_k \end{bmatrix} \equiv \begin{bmatrix} H_{xx} & H_{xy} \\ H_{xy} & H_{yy} \end{bmatrix}.$$

`H_yy = 1 − A` (the tangential curvature, `_discrim_A`); `H_xx = 1 − g′(R)`
is the radial curvature (`g` = the self-consistency map `g(R) = Σ_k ρ_k
cos θ̂_k σ(u_k)`); `H_xy` is the R–Θ coupling. Because dγ/dt = −∇F̂ is
gradient flow, `H_{F̂}` equals **−(the 2×2 fast block A** in
[`_coupled_jacobian`](../decision_model.py)**)**: numerically confirmed
`max|H_{F̂} − (−A_block)| = 2.5e-9` over 3777 SC equilibria. So this is
literally the fast block, and "γ-stable" ⇔ `A_block` Hurwitz ⇔
`H_{F̂} ≻ 0`.

**The positive-definiteness criterion.** For a symmetric 2×2, complete
the square around the `H_yy` pivot:

$$q(x,y) = x^\top H_{F̂}\, x = H_{yy}\Bigl(y + \tfrac{H_{xy}}{H_{yy}}x\Bigr)^2 + \frac{\det H_{F̂}}{H_{yy}}\,x^2.$$

A sum of two squares is positive in every nonzero direction **iff both
coefficients are positive**, giving

$$H_{F̂} \succ 0 \iff H_{yy} > 0 \ \text{ and } \ \det H_{F̂} > 0 \iff (A < 1) \ \text{ and } \ \det H_{F̂} > 0.$$

`_discrim_A` checks only the first factor. `det H_{F̂} > 0` is the
missing condition; given `H_yy > 0` it subsumes both the radial entry
(`det > 0 ⇒ H_xx H_yy > H_xy² ≥ 0 ⇒ H_xx > 0`) and the off-diagonal.
Necessity of `A < 1` is exact: `H_{F̂} ≻ 0 ⇒` all principal minors > 0
`⇒ H_yy > 0`; empirically the converse over-count `H_{F̂} ≻ 0` while
`A ≥ 1` **never** occurred (0 / 5582).

**Why the original spot-check (§9.3) passed anyway.** The off-diagonal
`H_xy = −Σ_k w_k cos θ̂_k sin θ̂_k` vanishes iff the visible targets are
mirror-symmetric about the consensus direction (each `+θ̂` paired with
`−θ̂` at equal ρ makes the `cos·sin` sum odd → 0). On such **on-axis**
equilibria the eigenvectors are exactly radial/tangential, `H_yy` *is*
an eigenvalue, and `A < 1` (plus a stable radial mode) does certify
stability. The §9.3 check sampled such symmetric configs, so it matched.
The equivalence breaks off-axis, where `H_xy ≠ 0` tilts the eigenvectors.

**Two failure modes** (both invisible to `A < 1`), from cutoff / power /
vonmises sweeps — 148 / 5582 SC equilibria had `A < 1` while `H_{F̂}`
was indefinite:

1. *Low-R radial fold* (R ≲ 0.1): the near-disordered state has
   `H_xx < 0` (classic mean-field pitchfork — the R≈0 blob grows into a
   consensus). Pure radial; orthogonal to `A`. Sits just above the
   `R < 0.01` filter.
2. *Committed-R determinant flip* (R ≈ 0.31–0.48): **both** diagonals
   positive but `det H_{F̂} < 0` — a genuine γ-saddle whose unstable
   mode is a tilted R–Θ blend. Witness (power warp c=0.5, β=4 — the
   two-target equivalent of the T=0.5 this was found at,
   focal_loc=(0.30, 0.71), θ=−0.607, R=0.338):
   `H_xx=+0.730, H_yy=+0.009 (A=0.991<1), H_xy=+0.321, det=−0.097`,
   eigenvalues `{−0.114, +0.853}`, unstable eigenvector 69° off the
   radial axis. Here `_discrim_A = True` (wrong) but
   `_discrim_reduced = False` (right).

**Implications for the code.**
- `_discrim_reduced` (default) and `_discrim_coupled` use the **full**
  2×2 block (`all(eig(A_block)) < 0`), i.e. the complete `H_{F̂} ≻ 0`
  test — both diagonal conditions *and* the determinant. They are
  unaffected: they correctly reject all 148 over-counts.
- `_discrim_A` / `_discrim_A_nu` (the `'discrim_a'` comparison
  criterion) are **incomplete on the fast layer**: they test only
  `H_yy > 0`, over-counting stable equilibria on the two modes above.
  This is *independent* of the documented slow-mode over-count (the
  `λ_slow > 0` heading-tracking instability, e.g. 3-vs-5 at (1.5,0)).
  `'discrim_a'` is thus doubly incomplete. The minimal complete
  fast-block test is `A < 1` **and** `det H_{F̂} > 0`.

## 7. Effective noise temperature for γ-Langevin

Adding Gaussian noise to `dγ/dt` gives:

$$dγ = -\nabla F̂(γ)\, dt + \sqrt{2 D}\, dW$$

with `dW` a standard 2D Wiener increment. The Boltzmann distribution
of γ in the stationary state is

$$P(γ) \propto \exp\!\bigl(-F̂(γ)/D\bigr).$$

To match the underlying spin model (large-N expansion of binomial
fluctuations in {nⱼ}, projected onto γ), we want:

$$P(γ) \propto \exp\!\bigl(-2βN\, F̂(γ)\bigr) \quad \Rightarrow \quad
  D = \frac{1}{2βN}.$$

So:

$$\boxed{\;D_\gamma = \frac{1}{2 β N}\;}$$

`N` is the effective spin count, a tunable parameter; in the `N → ∞`
limit, `D → 0` and γ-dynamics is deterministic (recovering the
mean-field model). For Monte Carlo vetting a moderate `N` was picked so
that escape events occur on a tractable timescale.

The derivation of `D = 1/(2βN)` from the binomial fluctuations is
standard mean-field stat mech (van Kampen system-size expansion; see
[theory_background.md](theory_background.md) §IV.4). It was confirmed
numerically by the stationary-distribution and 1/N-scaling tests
(empirical γ-variance scales as 1/N to ~2%).

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

## 9. Summary — numerical checks (all passed)

1. `np.gradient(F̂)` (finite-difference) matches the analytical
   gradient in section 6 to ~1e-8 at random (γ, θ, focal_loc) points,
   and equals `dγ/dt`.
2. At γ-equilibria from `gamma_equilib(focal_angle=θ)`, `∇F̂(γ*) ≈ 0`
   to ~1e-6.
3. Hessian eigenvalues at γ-equilibria match the *full 2×2*
   positive-definiteness label (and `_discrim_coupled` after projecting
   out the θ-direction). **Caveat (see §6.1):** they match `_discrim_A`'s
   `A < 1` scalar only where the R–Θ off-diagonal is ≈0 (mirror-symmetric
   / on-axis configs, which this spot-check happened to sample); `A < 1`
   is necessary but not sufficient in general.
4. The identity `2β·F̂(γ) = β·(F_mf per spin at constrained minimum) +
   const` from section 5 holds numerically.

With these passed, F̂ is trusted for ΔF_γ evaluation and for calibrating
γ-Langevin noise via the `D_γ = 1/(2βN)` relation — both summarized in
[basins_of_attraction.md](basins_of_attraction.md).
