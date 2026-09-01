"""
    MemoryKernelFitting

Fitting underdamped Prony-series memory kernels to configuration-resolved spectral
data, for quasi-Markovian generalised Langevin equations with configuration-dependent
electronic friction.

The package is organised as the three parts of the problem:

| submodule | part | contents |
|---|---|---|
| [`SpectralData`](@ref) | 0 | load `Λ(x,ω)`, convert to atomic units, build grid weights |
| [`PronySeriesFriction`](@ref) | 1 | the model `Λ_θ(x,ω)`, its poles, its amplitudes `a_k(x)` |
| `Objective` | 2 | the weighted residual `J` of the optimisation note §3 |
| `Fitting` | 3 | inner NNLS, outer pole search, two-stage compression |

The model is the boxed equation of
`docs/PronySeries/Prony_fit_optimisation_problem.md` §1: a white part, a set of
zero-centred Lorentzians, and a set of *mirrored* Lorentzian pairs. The pairs are
mandatory — the data has a resonance at nonzero `ω`, and a sum of zero-centred
Lorentzians is strictly decreasing.

Frequency domain ([`spectral_density`](@ref)):

```math
Λ(ω) = a₀ + \\sum_k a_k \\frac{λ_k}{λ_k² + ω²} + \\sum_j A_j L(ω; α_j, β_j)
```

Time domain ([`memory_kernel`](@ref)):

```math
K(τ) = 2a₀ δ(τ) + \\sum_k a_k e^{-λ_k τ} + \\sum_j A_j e^{-α_j τ} \\cos(β_j τ)
```

The two are a one-sided cosine-transform pair. All amplitudes are non-negative — they
are squares of the fitted `g(q)` — and all rates are positive, so that `Γ₂,₂` is stable.

**Units.** Atomic units everywhere inside the package; the eV in the HDF5 files are
converted once, at load. See [`Units`](@ref).
"""
module MemoryKernelFitting

"""
    AmplitudeModel

Supertype for representations of the configuration dependence of the amplitudes,
`x ↦ (a₀(x), a(x), A(x))`. This is the only place `x` enters the model: in regime (A)
the poles are shared across configurations and the amplitudes carry all of it.

Concrete subtypes live in [`PronySeriesFriction`](@ref). The two-stage procedure of
`Prony_fit_optimisation_problem.md` §6 uses `GridAmplitudes` for the convex stage and
a basis-expansion subtype for the compression stage; a neural representation is a
further subtype and needs no change here.
"""
abstract type AmplitudeModel end
Base.broadcastable(m::AmplitudeModel) = Ref(m)

export AmplitudeModel

include("Units/Units.jl")
using .Units
export eV_to_hartree, hartree_to_eV, boltzmann_energy, atomic_time_to_fs
export BOHR_IN_ANGSTROM, ELECTRON_MASSES_PER_U

include("SpectralData/SpectralData.jl")
using .SpectralData
export LambdaSpectrum, load_lambda_spectrum, trapezoid_weights, subsample_omega
export ErpenbeckThossModel, level_position, hybridisation_width, coupling
export dlevel_position, dhybridisation_width, diabatic_potentials
export fermi_crossing, morse_zero_point

include("PronySeriesFriction/PronySeriesFriction.jl")
using .PronySeriesFriction
export spectral_density, memory_kernel, lorentzian, mirrored_lorentzian
export PoleSet, auxiliary_dimension, log_ladder, split_ladder, uniform_ladder, pair_budget
export identifiability_box, is_identifiable
export TrackingPoles, poleset, n_tracking, tracking_seed, tracking_report
export TiedLaw, tied_seed, expand, N_TIED
export GridAmplitudes, PolynomialAmplitudes, ChebyshevAmplitudes, warp_coordinate
export fit_tag, BasisAmplitudes
export n_configurations, mode_counts, amplitudes, parameters, parameters!
export QGLEBlocks, qgle_blocks, gamma_matrix, exp_gamma22

include("Objective/Objective.jl")
using .Objective
export residual_scales, design_matrix, objective_value, roughness_penalty,
       markovian_anchor_row, configuration_weights

include("Fitting/Fitting.jl")
using .Fitting
export fit_amplitudes, fit_column, relative_errors
export fit_tracking, tracking_solve, tracking_ladders, tracking_errors, fit_tracking_basis
export basis_matrix, shape_envelope, fit_basis_report

end # module
