<p align="center">
  <img src="assets/logo-frameless.gif" alt="Animated MemoryKernelFitting.jl logo" width="240">
</p>

# MemoryKernelFitting.jl

[![CI](https://github.com/Louhokseson/MemoryKernelFitting.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Louhokseson/MemoryKernelFitting.jl/actions/workflows/CI.yml)

Fitting **underdamped Prony-series memory kernels** to configuration-resolved spectral
data, so that a quasi-Markovian generalised Langevin equation (QGLE) reproduces a
memory-dependent, configuration-dependent electronic friction.

The input is a real-valued spectral density $\Lambda(x,\omega)$ on a configuration grid
$x$ and a frequency window $\omega \in [0, \omega_{\max})$. The output is the set of
blocks $\widetilde{\boldsymbol\Gamma}(x)$, $\widetilde{\boldsymbol\Sigma}(x)$ of an
extended-variable SDE whose memory kernel matches that data — and which is guaranteed
to sample the right Gibbs measure.

## The model

$$
\Lambda_\theta(x,\omega) \;=\; a_0(x) \;+\; \sum_{k=1}^{K_R} a_k(x)\,\frac{\lambda_k}{\lambda_k^2+\omega^2} \;+\; \sum_{j=1}^{K_C} A_j(x)\,L(\omega;\alpha_j,\beta_j)
$$

$$
L(\omega;\alpha,\beta) \;=\; \frac{\alpha}{2}\left[\frac{1}{\alpha^2+(\omega-\beta)^2} + \frac{1}{\alpha^2+(\omega+\beta)^2}\right], \qquad a_0,\,a_k,\,A_j \ge 0
$$

with time-domain counterpart — the object the SDE actually realises —

$$
K(x,\tau) \;=\; 2a_0(x)\,\delta(\tau) \;+\; \sum_k a_k(x)\,e^{-\lambda_k\tau} \;+\; \sum_j A_j(x)\,e^{-\alpha_j\tau}\cos(\beta_j\tau).
$$

**The underdamped pairs are mandatory, not a refinement.** $\Lambda$ is not monotone in
$\omega$: it carries a resonance at $\omega_{\text{pk}} \approx 1.02$–$1.15\,|h(x)|$ that
sweeps from $0$ up to $9.4$ eV as the level position $h(x)$ crosses the Fermi energy. A
sum of zero-centred Lorentzians is strictly decreasing and cannot produce a peak at
$\omega > 0$ — only complex-conjugate pole pairs can.

The mirror in $L$ is likewise not a modelling choice. $\boldsymbol\Gamma_{2,2}$ is real,
so its complex eigenvalues $\alpha \pm i\beta$ come in conjugate pairs, and each pair
gives Lorentzians at $\omega = \pm\beta$ of half-width $\alpha$. A pair is only a
*resonance* when $\beta > \alpha/\sqrt3$; below that it merges into a monotone,
zero-centred shape — which is exactly the behaviour wanted at the crossing, where
$\beta \to 0$ while $\alpha$ stays finite. No special case needed.

## Why the fit is FDT-safe by construction

Set $m = K_R + 2K_C$ and keep the real rotation blocks rather than diagonalising:

```math
\boldsymbol\Gamma_{2,2} = \text{blockdiag}\big(\lambda_1,\dots,\lambda_{K_R},\ R(\alpha_1,\beta_1),\dots\big), \qquad R(\alpha,\beta) = \begin{pmatrix}\alpha & \beta\\ -\beta & \alpha\end{pmatrix},
```

$$
\widetilde{\boldsymbol\Gamma}_{1,1}(x) = a_0(x), \qquad \boldsymbol g(x) = \big(\sqrt{a_1},\dots,\ \sqrt{A_1},0,\ \dots\big)^T, \qquad \widetilde{\boldsymbol\Gamma}_{2,1} = \boldsymbol g,\quad \widetilde{\boldsymbol\Gamma}_{1,2} = -\boldsymbol g^T.
$$

Two consequences make the fit well posed rather than merely convenient:

- **Amplitudes are squares.** The fluctuation–dissipation theorem forces
  $\widetilde{\boldsymbol\Gamma}_{1,2}\boldsymbol Q = -\widetilde{\boldsymbol\Gamma}_{2,1}^T$,
  so one fits a *single* function $\boldsymbol g(x)$, not two independent blocks, and
  non-negativity of $a_k, A_j$ is structural. A negative fitted amplitude is a hard
  signal that no Gibbs-consistent QGLE reproduces the data.
- **$\boldsymbol Q = \boldsymbol I$ survives.** $\boldsymbol g^T e^{-\tau R}\boldsymbol g = |\boldsymbol g|^2 e^{-\alpha\tau}\cos\beta\tau$
  gives one non-negative amplitude per $2\times2$ block, and $R + R^T = 2\alpha\boldsymbol I$,
  so $\widetilde{\boldsymbol\Sigma}_2 = \sqrt{2\alpha}\,\boldsymbol I_2$. The requirement is
  $\boldsymbol\Gamma_{2,2}\boldsymbol Q + \boldsymbol Q\boldsymbol\Gamma_{2,2}^T \succ 0$,
  not symmetry — so complex eigenvalues, $\boldsymbol Q = \boldsymbol I$ and positive
  amplitudes hold *simultaneously*. This is also what makes the integrator's O-step
  well defined and $O(1)$ per block.

## How it is fitted

The model is **linear in the amplitudes and nonlinear only in the poles**, so the
problem separates (variable projection):

- **inner** — one convex non-negative least-squares problem per configuration, coupled
  across $x$ only by a roughness penalty; solved exactly.
- **outer** — a low-dimensional nonconvex search over the poles $(\lambda,\alpha,\beta)$.

Three features of the data dictate the objective rather than being tuning choices:
residuals must be weighted **relatively** ($\Lambda$ spans $\sim10^{-12}$ to $10^4$),
the configuration grid needs **quadrature weights** (83% of its columns cover 6.7% of the
range), and the amplitudes need a **roughness penalty** in $x$ because near-degenerate
poles otherwise share weight arbitrarily.

**Rates are shared across all $x$; only amplitudes vary.** That is the exact regime — the
two-time path-ordered kernel collapses to a genuine Prony series in the lag with no
error. The alternative, letting the poles track $x$, is an adiabatic approximation that
fails hardest exactly where the physics is interesting: at the crossing,
$\tau_{\text{mem}}/\tau_{\text{conf}} \approx 34$ even at 300 K. The price is a larger
fixed pole ladder ($K_C \sim 29$, so $m \approx 58$) instead of 2–3 tracking poles.
With the ladder fixed a priori, the outer loop disappears and the whole fit becomes a
single convex NNLS with a global optimum.

## Status

Early development. The model evaluation (`spectral_density`, `memory_kernel`, and the two
basis functions) is in place and tested as a cosine-transform pair. Still to come: the
weighted NNLS inner solve, the pole ladder and outer loop, the amplitude representation
$g_k(x)$, and the readout to $\widetilde{\boldsymbol\Gamma}(x)$.

## Installation

Not registered.

```julia
julia> using Pkg
julia> Pkg.develop(url="https://github.com/Louhokseson/MemoryKernelFitting.jl")
```

## Usage

```julia
using MemoryKernelFitting

a₀ = 0.0                                # white part; expected ≈ 0 for this data
a, λ = [0.75], [10.0]                   # overdamped modes
A, α, β = [1.5, 0.4], [0.5, 1.2], [3.0, 7.0]   # underdamped pairs

spectral_density(3.0, a₀, a, λ, A, α, β)   # Λ(ω)
memory_kernel(0.2, a, λ, A, α, β)          # K(τ), τ > 0, excluding 2a₀δ(τ)

# Markovian anchor — the zero-frequency friction a standard MDEF run would use
spectral_density(0.0, a₀, a, λ, A, α, β)
```

## Documentation

The theory, its provenance in the literature, and the optimisation problem in full are
in [docs/PronySeries/](docs/PronySeries/):

| file | contents |
|---|---|
| `Prony series (corrected).md` | **canonical for the theory** — QGLE, the two-time kernel, the FDT pairing, why there is no one-time configuration-dependent kernel |
| `Prony_fit_optimisation_problem.md` | the model above as a stated optimisation problem, calibrated against the real data; the underdamped series and its integrator in the appendix |
| `HANDOVER.md` | index and summary |

Principal references: Sachs, *PhD thesis* (Edinburgh, 2017), Eq. (3.7) and §4.7.2;
Leimkuhler & Sachs (2019), *Ergodic Properties of Quasi-Markovian GLEs with
Configuration Dependent Noise*; Stella, Lorenz & Kantorovich, *Phys. Rev. B* **89**,
134303 (2014) — direct prior art for the underdamped ansatz and for fitting in the
frequency domain; Lei, Baker & Li, *PNAS* **113**, 14183 (2016).

## Development

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

## License

This project is licensed under the [MIT License](./LICENSE). See the full text for details.

---

<div align="center">

<br/>

[**Xuexun Lu (Hokseon)**](https://louhokseson.github.io)<br>
*PhD Candidate, The Maurer Computational Surface Science Group*<br>
The University of Warwick, UK

[![Email](https://img.shields.io/badge/Email-louhokseson%40gmail.com-0054AD?logo=gmail&logoColor=white)](mailto:louhokseson@gmail.com)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0004--4916--5970-00A8E0?logo=orcid&logoColor=white)](https://orcid.org/0009-0004-4916-5970)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-CITED%202-006b3c?logo=g%20scholar&logoColor=white)](https://scholar.google.com/citations?user=233SExsAAAAJ&hl=en)

<br/>

</div>

<div align="center" style="font-size: 0.85em; color: #666; margin-top: 1em;">
Copyright © 2026 Xuexun Lu. All rights reserved.
</div>
