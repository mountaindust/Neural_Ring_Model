# The block-determinant identity (and how the reduced criterion uses it)

Reference note for the `'reduced'` stability criterion in `decision_model.py`
(`NBM._discrim_reduced` / `IEM._discrim_reduced`). Explains why the slow
stability test is evaluated as `sign(det J)` rather than by inverting `A`.

## The identity

For a block matrix with `A` square and invertible and `d` a scalar,

$$
J=\begin{pmatrix} A & b \\ c & d\end{pmatrix},
\qquad
\det(J) = \det(A)\,\underbrace{\left(d - c\,A^{-1}b\right)}_{\lambda_{\text{slow}}}.
$$

The quantity `d - c A^{-1} b` is the **Schur complement** of `A` in `J`
(written `J/A`). In our 3×3 coupled Jacobian on `(γ_re, γ_im, θ)`:

- `A` = the 2×2 γ block `∂(dγ/dt)/∂γ`,
- `b` = `∂(dγ/dt)/∂θ` (2×1 column),
- `c` = `∂(dθ/dt)/∂γ` (1×2 row),
- `d` = `∂(dθ/dt)/∂θ` (scalar; `= 0` for NBM, `= −K·R/2` for IEM at an SC eq).

So `λ_slow = d − c A^{-1} b` is a single real number.

## Proof (block LU factorization)

Factor the coupling out of `J`:

$$
\begin{pmatrix} A & b \\ c & d\end{pmatrix}
=
\begin{pmatrix} I & 0 \\ c A^{-1} & 1\end{pmatrix}
\begin{pmatrix} A & b \\ 0 & d - cA^{-1}b\end{pmatrix}.
$$

Check by multiplying out the bottom row of the product:

- bottom-left: `cA^{-1}·A + 1·0 = c` ✓
- bottom-right: `cA^{-1}·b + 1·(d − cA^{-1}b) = d` ✓

(top row is trivially `A`, `b`). The determinant of a product is the product
of determinants:

- left factor: lower-triangular with unit diagonal blocks ⟹ `det = 1·1 = 1`,
- right factor: upper block-triangular ⟹ `det = det(A)·(d − cA^{-1}b)`.

Therefore `det(J) = det(A)·(d − cA^{-1}b) = det(A)·λ_slow`. ∎

## Why λ_slow is the slow eigenvalue

In the timescale-separated (slaved) reduction, γ is fast and tracks its
equilibrium branch `γ = h(θ)`, leaving the 1-D slow flow `dθ/dt = g(h(θ))`.
Its linearization is the total derivative

$$
\lambda_{\text{slow}}
= \frac{d}{d\theta}\,g(h(\theta))
= \underbrace{d}_{\partial_\theta(d\theta/dt)}
+ \; c\,\underbrace{h'(\theta)}_{=\,-A^{-1}b}
= d - cA^{-1}b.
$$

`h'(θ) = −A^{-1}b` comes from differentiating the branch condition
`F(h(θ), θ) = 0` in θ. Reading it physically: perturb θ → the γ-equilibrium
shifts by `h'(θ)` → that shift feeds back into `dθ/dt` through `c`. The direct
term `d` plus this indirect γ-mediated term is the net slow restoring rate.

`λ_slow` is also the leading (λ → 0) term of the **exact** eigenvalue equation
`λ = d − c (A − λI)^{-1} b`: setting λ = 0 inside the resolvent is precisely the
adiabatic elimination of γ.

## How the criterion uses it

`A` is symmetric (the γ-dynamics is a gradient flow, `A = −Hess F̂`), so its
eigenvalues are real. Once the **Hurwitz gate** passes (both `eig(A) < −tol`,
i.e. the γ-branch is attracting), the two eigenvalues are negative reals and

$$
\det(A) = \mu_1 \mu_2 > 0.
$$

With a positive prefactor, `det(J)` and `λ_slow` share a sign:

$$
\boxed{\;\lambda_{\text{slow}} < 0 \iff \det(J) < 0\;}
$$

So the slow-mode stability test is just `sign(det J)` — **no `A^{-1}`**. The
criterion is therefore:

1. `A` Hurwitz (fast/γ layer attracting), **and**
2. `det(J) < 0` (slow/θ layer stable).

### Why `det(J)` instead of computing `λ_slow` directly

Near a **γ-fold** (saddle-node of the fixed-heading γ-equilibrium), one
eigenvalue of `A` → 0, so `A^{-1}` — and hence `λ_slow = d − cA^{-1}b` itself —
**diverges** (`|λ_slow| → ∞`). The sign is still unambiguous, but the literal
Schur expression is numerically ill-conditioned. `det(J) = det(A)·λ_slow` stays
**bounded** (det(A) → 0 cancels the blow-up) and keeps the correct sign.

Measured at the island fold tip ≈(2.467, 2.633): `min|eig(A)| ≈ 8e-4`,
`|λ_slow| ≈ 308`, while `det(J) ≈ −0.25`. Verdict-identical to the Schur form
across the island grid (0/538 mismatches); unit-tested in
`tests/test_reduced_criterion.py`
(`test_schur_block_determinant_identity_*`, `test_reduced_robust_near_gamma_fold`).

A useful structural note: the ill-conditioned and the delicate-decision regimes
are **anti-correlated**. Near a fold `|λ_slow|` is large (the slow mode is far
from marginal, sign robust); a *marginal* slow eigenvalue (`λ_slow ≈ 0`, where
sign is delicate) only occurs where `A` is well-conditioned. You never face
both difficulties at once.
