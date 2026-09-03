"""
    MemoryKernelFitting.PronySeriesFriction

Part 1 of the package: the **model**. The boxed equation of
`docs/PronySeries/Prony_fit_optimisation_problem.md` §1, evaluated at a single
configuration `x` and — via [`AmplitudeModel`](@ref) — as a function of `x`.

```math
Λ_θ(x, ω) = a₀(x) + \\sum_k a_k(x) \\frac{λ_k}{λ_k² + ω²} + \\sum_j A_j(x) L(ω; α_j, β_j)
```

The poles `(λ, α, β)` are shared across `x` (regime A of the handover note); the
configuration dependence lives entirely in the amplitudes. All quantities are in
atomic units — see [`MemoryKernelFitting.Units`](@ref).
"""
module PronySeriesFriction

using ..MemoryKernelFitting: AmplitudeModel
# The continuous readout of `continuous.jl` needs the analytic `h`, `Δ` and `Λ(x,0)`
# behind the data. That is the one place this module looks at a specific system.
using ..SpectralData: ErpenbeckThossModel, level_position, hybridisation_width,
                      markovian_friction

export lorentzian, mirrored_lorentzian
export spectral_density, memory_kernel
export PoleSet, auxiliary_dimension, log_ladder, split_ladder, uniform_ladder, pair_budget
export identifiability_box, is_identifiable
export TrackingPoles, poleset, n_tracking, tracking_seed, tracking_report
export TiedLaw, tied_seed, expand, N_TIED
export GridAmplitudes, PolynomialAmplitudes, ChebyshevAmplitudes, warp_coordinate
export fit_tag, BasisAmplitudes
export n_configurations, mode_counts, amplitudes, parameters, parameters!
export QGLEBlocks, qgle_blocks, gamma_matrix, exp_gamma22, qgle_coupling!
export ContinuousAmplitudes, amplitudes!

include("basis.jl")
include("model.jl")
include("poles.jl")
include("tracking_poles.jl")
include("amplitudes.jl")
include("readout.jl")
include("continuous.jl")

end # module
