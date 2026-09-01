"""
    MemoryKernelFitting.Objective

Part 2 of the package: the **objective**. The boxed functional of
`docs/PronySeries/Prony_fit_optimisation_problem.md` §3,

```math
J = \\sum_i v_i \\sum_{j \\in ℬ_i}
    \\left[\\frac{Λ_{p,a_i}(x_i, ω_j) - Λ_{ij}}{σ_{ij}}\\right]^2
  + μ \\sum_k \\sum_i \\frac{(a_k(x_{i+1}) - a_k(x_i))^2}{x_{i+1} - x_i}
```

with `σ_ij = max(|Λ_ij|, ε Λ_i^pk)`.

**Index convention, used throughout this submodule and worth fixing firmly in mind:**

| index | runs over | count | grid |
|---|---|---|---|
| `i` | **configurations** `x` | 254 | Å, **graded** — `dx` spans a factor of 290 |
| `j` | **frequencies** `ω` | 561 retained | Hartree, from a **uniform** grid, `dω = 1e-3` eV |

So `vᵢ` is a quadrature weight on the **`x`** axis, never on `ω`. The frequency grid is
uniform and needs no such weight; the configuration grid is the one that is graded, and
`Λ` is stored `[N_ω, N_x]` — frequency down a column, configuration across.

Note also that the inner sum over `j` is **unweighted** while the retained band is
log-spaced, so it approximates an integral in `ln ω` rather than in `ω`. That falls out
of how [`subsample_omega`](@ref) samples rather than from a decision, and is recorded as
an open item in `docs/DEVLOG.md`.

Every piece is there for a measured reason, recorded in §3 — relative weighting because
`Λ` spans ~16 decades, the floor because the far tail decays as `ω⁻³` and is outside the
reach of any Lorentzian sum, the quadrature weights because the `x` grid is graded, and
the roughness penalty because near-degenerate poles otherwise exchange amplitude
arbitrarily.

The objective is **separable**: linear in the amplitudes, nonlinear only in the poles.
Keeping the roughness term quadratic in `a` is what preserves that, and hence what
lets [`MemoryKernelFitting.Fitting`](@ref) solve the inner problem as one convex NNLS.
"""
module Objective

using ..SpectralData: LambdaSpectrum, trapezoid_weights, subsample_omega
using ..PronySeriesFriction: PoleSet, lorentzian, mirrored_lorentzian,
                             amplitudes, mode_counts, n_configurations,
                             warp_coordinate

using ..MemoryKernelFitting: AmplitudeModel

export residual_scales, design_matrix, objective_value, roughness_penalty,
       markovian_anchor_row, configuration_weights

include("residuals.jl")

end # module
