# Theory background — Lyapunov functions, free energy, Langevin & Kramers

Self-contained study notes intended to bring a future Claude Code
session (or the user starting from scratch) up to speed on the theory
behind the basin-of-attraction work in this project. Written for someone
with a strong applied-math background (math ecology in this case),
comfortable with ODEs, linear algebra, probability, and PDEs from
population biology, but not previously exposed to statistical mechanics,
gradient-flow theory, or Kramers escape-rate analysis.

The aim is not to be encyclopedic — it's to give you the tools to
follow the derivation in
[free_energy_derivation.md](free_energy_derivation.md)
step by step, understand why each step is justified, and be able to
recognize the same patterns when they appear in other models. Each
section ends with concrete pointers for going deeper.

## Project context

You are working on a mathematical model of collective decision-making in
a neural ring, based on Ising-type mean-field dynamics. The state of the
neural population is summarized by an order parameter γ ∈ ℂ (a
"coherence value" whose magnitude measures how strongly the neurons
agree and whose phase encodes the agreed-on direction). The deterministic
dynamics is

$$\dot γ = \sum_j ρ_j e^{i \hat θ_j}\, σ(u_j(γ)) - γ, \qquad u_j(γ) = 2β\, v̂_j \cdot γ,$$

with `σ(u) = 1/(1+e^(−u))`, derived as the mean-field N → ∞ limit of
Glauber dynamics on N Ising spins.

The model has a self-consistent equilibrium picture: when the observer's
heading θ equals the consensus direction γ encodes, both stabilize. The
project asks: for each stable self-consistent equilibrium, how robust is
it to noise? That requires understanding:

1. The mean-field free energy F̂(γ) that γ-dynamics descends — a
   *Lyapunov function* in dynamical-systems language, the *free
   energy* in stat-mech language. Same object, different vocabularies.
2. How noise is added to the deterministic dynamics — *Langevin
   equations*.
3. How basin escape rates depend on the geometry of F̂ — *Kramers
   formula*.

These three threads — Lyapunov, free energy, Kramers — are what this
document covers. The full derivation lives in
[free_energy_derivation.md](free_energy_derivation.md) (same folder);
this file is the "why" behind the math there, and the findings built on
it are catalogued in [basins_of_attraction.md](basins_of_attraction.md).

---

## Part I — Lyapunov functions and gradient flows

### I.1 What a Lyapunov function is

Given a deterministic dynamical system `ẋ = f(x)` on some state space
(say ℝⁿ), a **Lyapunov function** at an equilibrium `x*` (so `f(x*) = 0`)
is a real-valued function V(x) such that:

1. V(x*) is a strict local minimum.
2. Along trajectories of the dynamics, V is non-increasing:
   `dV/dt = ∇V(x) · f(x) ≤ 0` in a neighborhood of x*.

If both hold (with strict decrease for x ≠ x*), then x* is *locally
asymptotically stable*: every nearby trajectory converges to it. This is
**Lyapunov's direct method**.

The intuition is that V is an "energy" that the system gives up over
time. Trajectories slide downhill on the V landscape until they reach
the bottom of a well.

**Why this is powerful in math ecology:** you don't have to solve the
ODE to prove stability. You just have to find a V. For some systems —
especially gradient systems and Hamiltonian-like systems — V is
constructable from the equations directly.

### I.2 Examples that should feel familiar

**Logistic growth.** `ẋ = r x (1 − x/K)` for population x > 0, carrying
capacity K. Take `V(x) = (x − K)²/2`. Then
`dV/dt = (x − K) · r x (1 − x/K) = -r x (x − K)²/K`, which is ≤ 0 for
x > 0 and 0 only at x = K. So K is asymptotically stable. (Of course
we knew that — but this is the simplest non-trivial Lyapunov function.)

**Lotka-Volterra predator-prey** (classic, with no carrying capacity):

$$\dot x = α x − β x y, \qquad \dot y = δ x y − γ y.$$

The function

$$V(x, y) = δ x − γ \ln x + β y − α \ln y$$

satisfies `dV/dt = 0` along trajectories — it's a *conserved quantity*,
not a strict Lyapunov function. That's why LV predator-prey has closed
orbits rather than asymptotic stability. The level sets of V *are* the
orbits.

(In the language of physics: LV without carrying capacity is
*Hamiltonian*, with V playing the role of the Hamiltonian. Adding a
carrying capacity term breaks the conservation and you get
asymptotic stability — V becomes a strict Lyapunov function.)

**Competitive LV with two species** has a Lyapunov function in certain
parameter regimes; see Hofbauer & Sigmund, *Evolutionary Games and
Population Dynamics*, sec. 4.

**Quasi-potential in stochastic ecology.** When you add noise to a
deterministic ecological model, you can sometimes construct a function
W(x) — the *quasi-potential* — such that the stationary distribution
under small noise looks like P(x) ∝ exp(−W(x)/ε). W generalizes the
Lyapunov function to the stochastic case. This is exactly the role F̂
plays in our project. See Freidlin & Wentzell, *Random Perturbations of
Dynamical Systems*, ch. 4 — or for an ecology-flavored treatment,
Allen, *An Introduction to Stochastic Processes with Applications to
Biology*, ch. 9.

### I.3 Gradient systems specifically

A gradient system is one of the form `ẋ = -∇V(x)` for some scalar V.
For gradient systems, **V is automatically a Lyapunov function** at
every isolated equilibrium:

$$\frac{dV}{dt} = ∇V · ẋ = ∇V · (-∇V) = -|∇V|^2 ≤ 0,$$

with equality only at equilibria. So V decreases monotonically along
trajectories. There are no limit cycles, no chaos — only convergence to
local minima of V.

Equilibria of `ẋ = -∇V` are critical points of V (`∇V = 0`). Stability
is determined by the Hessian:

- All Hessian eigenvalues > 0 → local minimum → stable.
- Some Hessian eigenvalues < 0 → saddle → unstable, but the unstable
  directions are exactly the eigenvectors with negative eigenvalues.
- All eigenvalues < 0 → local maximum → unstable in all directions.

### I.4 When is a dynamical system gradient?

Not all dynamical systems are gradient. For `ẋ = f(x)` to be gradient
with respect to *some* V, the Jacobian of f must be symmetric:
`∂f_i/∂x_j = ∂f_j/∂x_i`. This is the **integrability condition**,
sometimes called the *curl-free condition* because in ℝ³ it says the
vector field f has zero curl.

**Why does curl-free imply gradient?** This is the Poincaré lemma. In a
simply-connected region (no holes), every closed differential 1-form is
exact. In coordinates: if you have a vector field f whose
"curl" vanishes, then the line integral of f around any closed loop is
zero, so the line integral from a fixed basepoint to a variable
endpoint depends only on the endpoint, not the path. That defines V(x)
as `V(x) = -∫_basepoint^x f · dℓ`.

In ℝ² this is concretely: if `∂f_y/∂x = ∂f_x/∂y`, then there exists V
with `f = -∇V`, found by integrating either component along any path.
This is exactly section 2 of the F̂ derivation: we checked
`∂Φ_x/∂γ_y = ∂Φ_y/∂γ_x` algebraically, then constructed F̂ by
integrating Φ_x with respect to γ_x.

(In ℝⁿ the condition generalizes to "all `∂f_i/∂x_j − ∂f_j/∂x_i = 0`",
i.e. the curl-2-form vanishes. We didn't need n > 2 here.)

### I.5 The math behind sections 2 and 3 of the F̂ derivation

This connects what we just covered to the project. From the γ-ODE we
defined `Φ_x(γ)` and `Φ_y(γ)` so that `dγ/dt = Φ(γ) − γ`. To show this
is gradient flow, we computed `∂Φ_x/∂γ_y` and `∂Φ_y/∂γ_x`, showed they
agreed, concluded that the field `γ − Φ(γ)` is curl-free, and used the
Poincaré lemma to construct F̂ by integration.

That F̂ then satisfies `dγ/dt = -∇F̂`. So by I.3, F̂ is a Lyapunov
function for the γ-dynamics: every trajectory descends F̂ until it
reaches a critical point.

**Exercise to internalize this:** take the 1D logistic example
`ẋ = r x (1 − x/K)` and check that this is a gradient system. Find V.
Then take a 2D non-gradient example like LV predator-prey, verify the
integrability condition fails, conclude that no Lyapunov function of
gradient form exists. (A conserved Hamiltonian still works — but that's
a different structure.)

### I.6 Where to go deeper

- Strogatz, *Nonlinear Dynamics and Chaos*, ch. 7 (limit cycles vs.
  gradient systems, intuitive).
- Khalil, *Nonlinear Systems*, ch. 4 (Lyapunov's direct method,
  rigorous).
- Hofbauer & Sigmund, *Evolutionary Games and Population Dynamics*,
  for ecology-flavored Lyapunov examples.
- Smale's "On gradient dynamical systems" (1961) — the foundational
  paper, short and readable.

---

## Part II — Free energy in statistical mechanics

This is the section where math ecology background diverges most from
stat-mech background. Take it slowly.

### II.1 The Boltzmann distribution

Consider a system with energy function H(σ) where σ is a configuration
(e.g., for our project, σ = (σ_1, ..., σ_N), each `σ_j ∈ {0, 1}`).
At temperature T in thermal equilibrium, the probability of finding the
system in configuration σ is

$$P(σ) = \frac{1}{Z} e^{-H(σ)/T}, \qquad Z = \sum_σ e^{-H(σ)/T}.$$

This is the **Boltzmann distribution**. Z is the **partition function**.

This is the fundamental object of equilibrium statistical mechanics. It
can be derived from a few axioms (maximum entropy at fixed energy, plus
energy conservation in contact with a bath) but for our purposes treat
it as a definition. T is "temperature" — higher T means more uniform
distribution over configurations, lower T means concentration on low-H
configurations.

Note the parallel to ecology: the Boltzmann distribution is the
ecological "Hutchinson niche overlap"-style weighting, or the
"environmental filtering" probability of observing a community at
configuration σ given energy H(σ). The temperature is the noise level.

### II.2 Free energy as a thermodynamic potential

The **free energy** is defined as

$$F = -T \ln Z.$$

This is *the* central object of equilibrium stat mech. Reasons:

(a) Differentiating −F/T with respect to model parameters gives
expectation values:
`⟨A⟩ = -∂(F/T) / ∂λ` where λ is the parameter that couples to A in H.

(b) The free energy of a composite system is *additive* over independent
subsystems (because Z factorizes), so it's the right thing to differentiate
to find equilibrium between coupled subsystems.

(c) For a system in contact with a heat bath at temperature T, F is the
quantity that's minimized at equilibrium. Intuition: F = ⟨H⟩ − T·S
(energy minus T times entropy). Low F means either low average energy
(favored at low T) or high entropy (favored at high T). Equilibrium
balances both.

(d) Differences in F between configurations control transition rates —
this becomes Kramers' formula in Part IV.

### II.3 The role of an order parameter

For our model, σ ∈ {0,1}^N is intractable directly (2^N configurations).
But many properties depend only on a low-dimensional summary — an
*order parameter* — like γ = (1/N) Σ_j σ_j e^{i θ_j} in our case.

The **projected free energy** in γ is

$$F(γ) = -T \ln \sum_σ e^{-H(σ)/T} \mathbf{1}[γ(σ) = γ].$$

(The indicator restricts the sum to configurations consistent with the
given γ.) This F(γ) has the property that the marginal distribution of
γ is `P(γ) ∝ exp(-F(γ)/T)`. So F(γ) describes the energy landscape
*as seen by the order parameter*.

In the **large-N (thermodynamic) limit**, the marginal distribution
becomes sharply peaked around the γ that minimizes F(γ), and γ becomes
deterministic — this is mean-field theory.

### II.4 Mean field via the variational principle

Computing F(γ) exactly requires the full sum over σ, which we can't do.
The mean-field approximation:

(a) Choose a tractable trial distribution Q(σ) parameterized by some
variational parameters {m_j} (e.g., independent spins with
within-group probabilities q_j).

(b) Compute the variational free energy
`F_var = ⟨H⟩_Q − T·S(Q)` where ⟨⟩_Q is the expectation under Q and S is
its Shannon entropy.

(c) Minimize F_var over {m_j}. By Gibbs's inequality (a form of Jensen's
inequality), `F_var ≥ F_true` always, and equality holds when Q is
exactly the Boltzmann distribution at the given m_j.

This is the **Bogoliubov variational principle**. The minimization
gives self-consistency equations that fix m_j.

For our model: take Q to be N independent Bernoulli spins with
within-group probabilities q_j = n_j/ρ_j. Compute ⟨H⟩_Q (gives the
−𝓔N R² term I used in section 5 of the derivation), compute S(Q) (gives
the binomial entropy), minimize over {q_j} with γ held fixed.

### II.5 The math behind section 5 of the F̂ derivation

This is the calculation I did. Let me re-walk it more slowly.

The Hamiltonian (from the user's writeup) is

$$H = -\frac{\mathcal{E}}{N}\sum_{j \neq l} σ_j σ_l \cos(\hat θ_{g(j)} - \hat θ_{g(l)})$$

The prefactor is a fixed energy scale $\mathcal{E}$, **not** the target
count `k`. The `1/N` normalizes the energy per neuron; there is no
physical reason for the per-neuron energy to also grow with how many
targets happen to be in view. The Glauber temperature then enters only
through $\beta = \mathcal{E}/(k_B\,\mathrm{temp})$.

where the sum is over individual spin pairs (not groups), and g(j) is
the group label of spin j.

Mean-field approximation: assume all spins are independent, with the
probability of spin j being on equal to `q_{g(j)} = n_{g(j)}/ρ_{g(j)}`
where g(j) is its group. Then for j ≠ l,
`⟨σ_j σ_l⟩ ≈ q_{g(j)} q_{g(l)}`.

The number of ordered spin pairs in groups (a, b) with a ≠ b is
`N·ρ_a · N·ρ_b = N² ρ_a ρ_b` (to leading order in N). Within group a,
the number is `N·ρ_a · (N·ρ_a − 1) ≈ N² ρ_a²`.

So:
`⟨H⟩ ≈ -k/N · Σ_{a,b} N² ρ_a ρ_b · q_a q_b · cos(θ̂_a − θ̂_b)`
`    = -kN · Σ_{a,b} (ρ_a q_a)(ρ_b q_b) cos(θ̂_a − θ̂_b)`
`    = -kN · Σ_{a,b} n_a n_b cos(θ̂_a − θ̂_b)`
`    = -kN · |γ|²` (since `Σ_a n_a e^{iθ̂_a} = γ`).

For the entropy: each spin in group a is independent Bernoulli with
parameter q_a. The entropy per spin is `-q_a ln q_a − (1-q_a)ln(1-q_a)`.
Total entropy: `S = N · Σ_a ρ_a · [-q_a ln q_a − (1-q_a)ln(1-q_a)]`.

So:

$$β\,F_{\text{mf}}/(N\mathcal{E}) = -β|γ|^2 + \sum_a ρ_a [q_a \ln q_a + (1-q_a) \ln(1-q_a)].$$

To project onto γ: minimize over {q_a} with the constraint
`Σ_a ρ_a q_a e^{iθ̂_a} = γ`. The Lagrangian has multipliers for the
real and imaginary parts of the constraint. Setting derivatives to
zero gives `q_a* = σ(u_a)` where `u_a = 2β v̂_a · γ` (as in the
derivation), and the projected F is what comes out after substitution.

The key algebraic miracle of section 5 was that, after the substitution
and a small amount of identity manipulation,
`2β · F̂(γ) = β F_mf/(N𝓔)` at
the constrained minimum, up to a constant. That's why the F̂ derived
by gradient-flow integration in section 3 matches the F̂ derived by
mean-field projection in section 5 — they're the same object via two
routes.

### II.6 Where to go deeper

- Chandler, *Introduction to Modern Statistical Mechanics*, chs. 1–3
  for the basics, ch. 5 for mean field. Highly recommended — short,
  precise, pedagogical.
- Goldenfeld, *Lectures on Phase Transitions and the Renormalization
  Group*, chs. 2–3 for stat mech, ch. 5 for mean field with critical
  exponents. More advanced.
- For the connection to information theory and ecology: Jaynes,
  "Information theory and statistical mechanics" (1957), short and
  influential.
- For the variational principle in detail: Mezard, Parisi, Virasoro,
  *Spin Glass Theory and Beyond*, sec. 2.1.
- For the projection onto order parameters as a Legendre transform:
  Goldenfeld sec. 2.6.

---

## Part III — Wirtinger calculus (brief)

When γ ∈ ℂ and you want to take a "derivative" of a real-valued
function F(γ), there are two equivalent formalisms.

### III.1 (γ_re, γ_im) split

Treat γ as a real 2-vector (γ_re, γ_im). Compute ∂F/∂γ_re and
∂F/∂γ_im as usual partial derivatives. This is what we've been doing
throughout the project.

### III.2 Wirtinger derivatives

Treat γ and γ̄ (complex conjugate) as formally independent variables
and define

$$\frac{∂}{∂γ} = \tfrac{1}{2}\!\left(\frac{∂}{∂γ_{\text{re}}} - i\,\frac{∂}{∂γ_{\text{im}}}\right), \qquad
\frac{∂}{∂γ̄} = \tfrac{1}{2}\!\left(\frac{∂}{∂γ_{\text{re}}} + i\,\frac{∂}{∂γ_{\text{im}}}\right).$$

These satisfy `∂γ/∂γ = 1, ∂γ̄/∂γ = 0, ∂γ/∂γ̄ = 0, ∂γ̄/∂γ̄ = 1` — they
treat γ and γ̄ as independent. For a holomorphic (complex-analytic)
function f(γ), `∂f/∂γ̄ = 0` (this *is* the Cauchy-Riemann condition).
For a real-valued function F(γ, γ̄) = F(γ_re, γ_im), neither derivative
is zero in general.

The point: for a real-valued F on ℂ, the gradient-flow equation in
complex form is

$$\dot γ = -2\, \overline{\partial F/\partial γ̄}.$$

The factor of 2 and the conjugate are bookkeeping. The dynamics is the
same as the (γ_re, γ_im) split.

When to use Wirtinger: when you're doing a lot of complex analysis and
the conjugation structure is clearer that way (e.g., in signal
processing or in superconductivity). For us, the (γ_re, γ_im) split is
simpler and matches the codebase, so we stick with it.

### III.3 Where to go deeper

- Remmert, *Theory of Complex Functions*, sec. 1.3 — proper treatment.
- Anything on adaptive filtering or array signal processing will use
  Wirtinger heavily; Adali & Schreier, *Complex-Valued Signal
  Processing*, ch. 2.

---

## Part IV — Langevin dynamics and Kramers escape rate

This is the most directly relevant section for the basin-of-attraction
work.

### IV.1 From deterministic flow to Langevin

Add Gaussian white noise to a deterministic dynamical system:

$$\dot x = f(x) + \sqrt{2D}\, ξ(t), \qquad ⟨ξ(t)⟩ = 0, \quad ⟨ξ(t)ξ(t')⟩ = δ(t-t').$$

This is a **Langevin equation**, or equivalently an SDE
`dx = f(x) dt + √(2D) dW` where W is a Wiener process. D is the
**diffusion coefficient**.

When `f(x) = -∇V(x)` (gradient flow), the Langevin equation is
*overdamped Brownian motion in a potential V*. The stationary
probability distribution is

$$P_{ss}(x) = Z^{-1} e^{-V(x)/D}.$$

This is **Boltzmann's distribution with effective temperature D**. So
in a gradient Langevin system, the diffusion coefficient plays the
role of temperature.

(Why? You can derive this by solving the corresponding Fokker-Planck
equation `∂_t P = ∇·(P∇V) + D ∇²P` for the stationary state.
Equivalent: detailed balance with the Boltzmann form.)

**For our project:** `dγ = -∇F̂(γ) dt + √(2D) dW` has stationary
distribution `P_ss(γ) ∝ exp(-F̂(γ)/D)`. The choice `D = 1/(2βN)` makes
this match the underlying spin model's γ-marginal in the large-N
expansion — that's what section 7 of the derivation calibrates.

### IV.2 Mean first passage time and Kramers' formula

In a Langevin system with a stable equilibrium at `x_min` and a saddle
at `x_saddle` separating it from another basin, the noise occasionally
drives x over the saddle. The **mean first passage time** (MFPT) for
escape is the average time to first reach the saddle starting at
`x_min`. For small noise (D much less than the barrier height
ΔV = V(x_saddle) − V(x_min)), the MFPT is given by **Kramers' formula**:

$$τ_{\text{escape}} = \frac{2π}{\sqrt{|V''(x_{\text{saddle}})|\, V''(x_{\text{min}})}}\, \exp\!\left(\frac{ΔV}{D}\right).$$

(This is the 1D version. There's a higher-dimensional generalization
involving the Hessian determinants at the minimum and saddle.)

The exponential dependence on ΔV/D is the dominant feature. The
prefactor (the "attempt frequency") depends on the curvatures of V at
the minimum and saddle — how stiff the well is and how sharp the
barrier is.

**Connection to ecology:** Kramers' formula is structurally identical
to the formula for *mean time to extinction* in stochastic population
models near a stable equilibrium. The "barrier height" ΔV becomes the
quasi-potential difference between the stable population and the
extinction boundary; the prefactor depends on the local geometry. See
Allen, ch. 9, or Kessler & Shnerb, "Extinction rates for fluctuation-
induced metastabilities" (2007), for the ecology version.

### IV.3 Multi-dimensional Kramers and the saddle approximation

For x ∈ ℝⁿ, the formula generalizes:

$$τ_{\text{escape}} = \frac{2π}{|λ_-|} \cdot \sqrt{\frac{|\det H_{\text{saddle}}|}{\det H_{\text{min}}}}\, \exp\!\left(\frac{ΔV}{D}\right)$$

where `H_min` and `H_saddle` are the Hessians of V at the minimum and
saddle (all eigenvalues of `H_min` are positive; `H_saddle` has exactly
one negative eigenvalue λ_-). The mode of escape is along the unstable
direction of the saddle.

For our project: each stable SC equilibrium can have multiple basin
boundaries (a θ-saddle on the slow manifold, and γ-saddles in
γ-space). Each gives a contribution to the total escape rate. The
total rate is approximately the sum of individual Kramers rates over
all boundaries, with the smallest barrier dominating (because of the
exponential).

### IV.4 The math behind section 7 of the F̂ derivation

Section 7 says the γ-Langevin noise amplitude for matching the
underlying spin model is `D = 1/(2βN)`. Where does this come from?

Heuristic derivation. The spin model has fluctuations in n_j of order
`√(N q_j(1-q_j)/N²) = √(q_j(1-q_j)/N)` per individual spin variance;
collected across N spins, the variance of n_j (the fraction on)
is `q_j(1-q_j) ρ_j / N`. Sums of such fluctuations across groups
projected onto γ give variance of order `1/N`. Match this to the
Langevin stationary variance `D` (from `P ∝ exp(-F̂/D)` and a Gaussian
expansion around the minimum: variance = D / F̂''). Equating gives
`D ~ 1/(βN)` up to the geometric factor. The factor of 2 comes from
the careful accounting of 2β vs β in the F̂ normalization.

A proper derivation goes through the system-size expansion of the
master equation à la van Kampen — see *Stochastic Processes in Physics
and Chemistry*, ch. 10. The upshot: in mean-field models, finite-N
fluctuations give an effective Langevin description with
`D ~ 1/(β N · constant)` where the constant depends on the normalization
of the free energy. For us, the constant is `2`.

This will be checked numerically in Step 3 by histogramming γ
trajectories in equilibrium and comparing to the predicted Gaussian
variance from F̂''.

### IV.5 The user's project specifics

Two distinct escape mechanisms for the basin work:

**(a) θ-saddle escape on the slow manifold.** γ stays near γ_eq(θ) and
θ wanders along the slow manifold under a 1D effective Langevin
equation `θ̇ = f(θ) + √(2D_θ) dW`. The 1D effective potential V(θ)
satisfies `V'(θ) = -f(θ)`. Kramers gives `τ ~ exp(ΔV/D_θ)`.

**(b) γ-saddle escape at fixed θ.** γ leaves its current well in
γ-space, escaping over a γ-saddle. Kramers gives `τ ~ exp(ΔF̂/D)`,
where D = 1/(2βN) (the γ-Langevin coefficient).

These are competing escape mechanisms. The total escape rate is
approximately their sum, and the smaller barrier (larger rate)
dominates exponentially. Which one wins depends on the geometry at
each (x, y) grid point. The basin estimator we're building will
compute both barriers, report both, and (perhaps) use the dominant
one as the noise-robustness scalar.

### IV.6 Where to go deeper

- Gardiner, *Handbook of Stochastic Methods*, chs. 3–4 (general
  Langevin and Fokker-Planck), ch. 5 (escape rates and Kramers). The
  canonical reference. Comprehensive and surprisingly readable.
- Risken, *The Fokker-Planck Equation*, ch. 4 (escape problems).
  More PDE-focused than Gardiner.
- Hänggi, Talkner, Borkovec, "Reaction-rate theory: fifty years after
  Kramers" (1990), Rev. Mod. Phys. 62. The review that consolidated
  Kramers theory; long but highly readable in pieces.
- Freidlin & Wentzell, *Random Perturbations of Dynamical Systems*,
  chs. 3–4 (large-deviation theory for escape problems; the
  rigorous foundation of Kramers).
- For an ecology-flavored treatment of escape rates: Allen,
  *An Introduction to Stochastic Processes with Applications to
  Biology*, ch. 9. Or Ovaskainen & Meerson, "Stochastic models of
  population extinction" (2010), Trends in Ecology & Evolution.

---

## Cross-cutting concepts (small but important)

### Detailed balance

A stochastic process satisfies detailed balance if for all states a, b:
`P(a) · W(a→b) = P(b) · W(b→a)` where W are transition rates and P is
the stationary distribution. Detailed balance is what makes a Markov
process equivalent to equilibrium thermodynamics. The Glauber dynamics
in our project satisfy detailed balance with respect to the
Boltzmann distribution at temperature T.

In our project, the *deterministic* γ-flow is dissipative (loses
energy F̂) and so isn't time-reversible. The *stochastic* γ-flow under
Langevin noise (in equilibrium) does satisfy detailed balance and has
the Boltzmann form. The deterministic flow is what you get by taking
the noise to zero; the noisy flow is the "real" dynamics for a finite
system.

### Adiabatic limit

When a dynamical system has a separation of timescales — fast variables
that relax quickly to a slow manifold parameterized by slow variables —
you can approximate by saying the fast variables are always at their
equilibrium given the slow variables. This is called the *adiabatic*
or *quasi-static* limit (different from "adiabatic" in thermodynamics,
confusingly).

In our project, γ is fast and θ is slow. The slow manifold is γ_eq(θ).
The adiabatic limit gives a reduced 1D dynamics for θ on the slow
manifold. The full project's `plot_walkers` code uses this limit
implicitly: each step calls `run_dgamma_dt` to steady state at the
current θ, then updates θ.

### The fluctuation-dissipation theorem

A general relation: the noise amplitude in a Langevin equation is
linked to the dissipation rate by the temperature. For our project,
`D = 1/(2βN)` is a special case. The general statement is that linear
response coefficients (how the system responds to small perturbations)
and equilibrium fluctuations (how it spontaneously varies) are
proportional, with constant of proportionality involving T. Gardiner
ch. 1.5 has a clear treatment.

---

## Questions you might want to ask in a follow-up session

These are seed questions designed to be productive starting points for
working through the material. Each can branch into a multi-turn
conversation.

**Lyapunov / gradient flow:**
- "Walk me through the Poincaré lemma in ℝ², carefully, with an
  example where it fails because the domain isn't simply-connected."
- "Why isn't predator-prey Lotka-Volterra a gradient system? What
  would it take to make it one?"
- "Give me a 2D example where the dynamics is *almost* gradient
  but has a small non-gradient (curl) component. How does the
  long-time behavior differ from the purely gradient case?"

**Free energy / stat mech:**
- "Derive the Boltzmann distribution from first principles (maxent or
  canonical ensemble — pick one and walk me through it)."
- "Walk me through the mean-field approximation for a simple Ising
  model (1D or 2D nearest-neighbor) with no external field. What do
  we lose vs. the exact treatment?"
- "What is a Legendre transform and how does it relate the variational
  free energy F_var to the projected F(γ)?"
- "How does the Bogoliubov inequality work? Prove it."

**Langevin / Kramers:**
- "Derive the stationary distribution of `dx = -V'(x) dt + √(2D) dW`
  from the Fokker-Planck equation, step by step."
- "Derive the 1D Kramers formula via the mean first passage time
  approach. Where does the prefactor come from?"
- "How does Kramers' formula generalize to multi-dimensional systems?
  What's the difference between the n-dim formula and just summing 1D
  formulas for each escape direction?"
- "How does van Kampen's system-size expansion give the diffusion
  coefficient in a Langevin approximation to a master equation?"

**Specific to this project:**
- "In the basin-estimation derivation, F̂(γ) has a `(1/2)|γ|²` term
  and a log-sum-exp term. Walk me through what the two terms represent
  physically — the kinetic-like part vs. the entropic part."
- "Why is the spin-Hamiltonian-derived dγ/dt automatically gradient?
  Is this a general feature of mean-field Glauber dynamics?"
- "In what regimes does the θ-saddle escape dominate over γ-saddle
  escape, and vice versa?"

---

## File pointers for the future-session context load

- This file: `theory/theory_background.md`
- Project guide (architecture, conventions, gotchas): `CLAUDE.md`
- Basin findings catalogue (results, boundary kinds, robustness):
  [basins_of_attraction.md](basins_of_attraction.md)
- F̂ derivation (the one this file supports):
  [free_energy_derivation.md](free_energy_derivation.md)
- The user's writeup of model derivation: not in repo — was the basis
  for `free_energy_derivation.md`. (The original step-by-step vetting
  log lived in `basin_estimation/`, retired 2026-06; recover from git
  history if needed.)
- Code: `decision_model.py` (NeuralBandModel class, especially
  `dgamma_dt`, `sc_equilib`, `_discrim_reduced`).

A fresh session can load `CLAUDE.md`, `theory/theory_background.md`, and
[basins_of_attraction.md](basins_of_attraction.md) to recover full
context with a single read of three files.
