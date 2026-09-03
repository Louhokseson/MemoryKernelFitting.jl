# The bridge: a fitted Prony series as an NQCDynamics `MemoryModel`.
#
#   include(joinpath(@__DIR__, "prony_memory.jl"))
#
# This file is deliberately *not* in `src/`. NQCDynamics pulls in OrdinaryDiffEq,
# StochasticDiffEq and the rest of the SciML stack; MemoryKernelFitting is a fitting
# package and has no business depending on an integrator. Everything reusable — the
# continuous readout, the envelope, `qgle_coupling!` — lives in the package and is
# tested there. What is left here is the adapter.
#
# ----------------------------------------------------------------------------------
# Why not `ConstantProny` or `ConfigurationDependentProny`
#
# NQCDynamics builds the coupling block as `Σ(x)·cᵀ`, where `Σ(x)` is the scalar
# friction from `friction!` and `c` is a fixed vector. That is rank-1 in `x`: every
# mode scales with one number, so the *shape* of the kernel cannot move. Our fit is the
# opposite case — the resonance sweeps from 0 to 9.4 eV across the Fermi crossing, which
# is exactly why `A_j(x)` are fitted independently — so no choice of constant `c`
# represents it. `ConfigurationDependentProny` is the intended escape hatch but does not
# run: `I_atoms`/`I_dim`/`memory_model` are unbound in its `GLE_friction!`, `_Λ`'s return
# value is discarded, `m.Λ .= kron(…, m.Λ)` is a dimension error, and `Γ₂₂` is never
# filled for that branch, so the auxiliary variables would never decay.
#
# `GLE_MDEF{a,T,MM}` is parameterised by the memory model, so dispatching on `MM` makes
# the method below *ours* and not piracy. It writes `Γ` directly and never touches the
# `Σ(x)·cᵀ` path, which also means `_Λ`'s all-2×2 block layout does not apply: the
# `K_R = 3` real modes stay 1×1 and `m` is 43, not the 46 a pair-padded encoding needs.
#
# ----------------------------------------------------------------------------------
# The two conversions
#
# 1. **Length.** NQCDynamics positions are in bohr; the fit is in Å (`docs/design/0001`).
# 2. **Mass.** The package writes the QGLE with unweighted momenta,
#        ṗ = F - a₀ M⁻¹p + gᵀs,   ṡ = -g M⁻¹p - Γ₂₂ s + β^{-1/2} Σ₂ Ẇ,
#    while NQCDynamics integrates `z = (p/√m, s)`. In that variable the drift is
#        Γ_mw = [ a₀/m   -gᵀ/√m ;   g/√m   Γ₂₂ ],
#    so the coupling carries `1/√m` and the white part `1/m`. Both are applied below and
#    nowhere else. `docs/design/0001` is explicit that this is the one place a silent
#    factor produces a stable, plausible SDE that samples the wrong temperature.
#

using MemoryKernelFitting
using NQCDynamics
using NQCDynamics: NQCModels
using NQCDynamics.DynamicsMethods.ClassicalMethods: MemoryModel, GLE_MDEF
# `import`, not `using`: the methods below extend theirs rather than shadowing them.
import NQCDynamics.DynamicsMethods.ClassicalMethods: GLE_friction!
import NQCDynamics.NQCModels.FrictionModels: friction!
using NQCDynamics: AbstractSimulation
using NQCDynamics: Ensembles
using Distributed: Distributed
using Random: Xoshiro, randn
using HDF5
using Unitful, UnitfulAtomic
using LinearAlgebra: I

"""
    PronyMemory(fit_file; kT, Γ_eV)

A stage-2 Chebyshev Prony fit, in the form NQCDynamics' `GLE_MDEF` integrates.

Regime (A): the poles are shared across configurations, so `Γ₂₂` and the noise are built
once and only the coupling vector `g(x)` moves. That is what makes the realised kernel
*exactly* the fitted one — the two-time path-ordered kernel collapses with no error when
`Γ₂₂` does not depend on the path.

# Fields
- `c` — the coupling buffer `g(x)`, rewritten every step. Named `c` because
  `GLE_MDEF`'s constructor reads `length(memory_model.c)` to size the auxiliary block;
  it is `Γ̃₂,₁`, not the `c` of `ConstantProny`.
- `Γ₂₂` — `blockdiag(λ₁,…,λ_{K_R}, R(α₁,β₁),…)`, configuration-independent.
- `model` — the [`ContinuousAmplitudes`](@ref) readout, evaluated at `x` in Å.
- `a`, `A` — amplitude scratch, so the per-step path allocates nothing.
"""
struct PronyMemory{T<:Real,M} <: MemoryModel
    c::Vector{T}
    Γ₂₂::Matrix{T}
    model::M
    a::Vector{T}
    A::Vector{T}
    ladder::PoleSet{T}
end

function PronyMemory(fit_file::AbstractString; kT, Γ_eV)
    isfile(fit_file) || error("fit file not found: $fit_file")
    C, ladder, K_R, K_C = h5open(fit_file, "r") do f
        (read(f["C_cheb"]),
         PoleSet(read(f["lambda_Ha"]), read(f["alpha_Ha"]), read(f["beta_Ha"])),
         read_attribute(f, "K_R"), read_attribute(f, "K_C"))
    end
    length(ladder.λ) == K_R || error(
        "ladder has $(length(ladder.λ)) real rates but the file says K_R = $K_R")
    length(ladder.α) == K_C || error(
        "ladder has $(length(ladder.α)) pairs but the file says K_C = $K_C")

    et = ErpenbeckThossModel(; Γ_eV)
    model = ContinuousAmplitudes(C, et, kT; K_R, K_C)
    # Zero amplitudes: only the `Γ₂₂` block of the result is wanted, and it does not
    # depend on them. Building it through `qgle_blocks` rather than by hand keeps the
    # rotation-block convention in one place.
    Γ₂₂ = qgle_blocks(ladder, 0.0, zeros(K_R), zeros(K_C)).Γ₂₂
    m = auxiliary_dimension(ladder)
    return PronyMemory(zeros(m), Γ₂₂, model, zeros(K_R), zeros(K_C), ladder)
end

MemoryKernelFitting.auxiliary_dimension(mm::PronyMemory) = length(mm.c)

function Base.show(io::IO, mm::PronyMemory)
    K_R, K_C = mode_counts(mm.model)
    print(io, "PronyMemory(K_R = ", K_R, ", K_C = ", K_C,
          ", m = ", auxiliary_dimension(mm), ")")
end

"""
    prony_simulation(atoms, model, mm::PronyMemory; kwargs...)

`Simulation{GLE_MDEF}` with the constant blocks of `mm` installed.

`GLE_MDEF`'s constructor only fills `Λ` and the `(2,2)` block of `Γ` for `ConstantProny`;
for any other memory model they are left as zeros, which would leave the auxiliary
variables undamped. Filling them here — once — is the whole reason this wrapper exists.
"""
function prony_simulation(atoms, model, mm::PronyMemory; kwargs...)
    sim = Simulation{GLE_MDEF}(atoms, model, mm; kwargs...)
    method = sim.method
    d = length(method.mass_vector)
    d == 1 || error("PronyMemory is a 1-D fit; got $d physical degrees of freedom")

    method.Λ .= mm.Γ₂₂
    @views method.Γ[(d + 1):end, (d + 1):end] .= mm.Γ₂₂
    return sim
end

"""
    GLE_friction!(gtmp, r, sim, t)

Rebuild the configuration-dependent part of `Γ` at the current position.

In regime (A) that is the coupling block alone:

    Γ = [ a₀/m      -gᵀ/√m ]
        [ g/√m       Γ₂₂    ]

with `Γ₂₂` constant and installed by `prony_simulation`. `gtmp` — the friction buffer
the integrator passes in — is deliberately untouched: the Markovian friction plays no
role here, the memory kernel replaces it entirely.
"""
function GLE_friction!(gtmp, r, sim::AbstractSimulation{<:GLE_MDEF{a,T,MM}}, t) where
        {a,T,MM<:PronyMemory}
    method = sim.method
    mm = method.memory_model
    d = length(method.mass_vector)

    # bohr → Å. The fit's configuration coordinate is in Å throughout the package.
    x = r[1] * BOHR_IN_ANGSTROM
    a₀ = qgle_coupling!(mm.c, mm.a, mm.A, mm.model, x)

    inv_sqrt_m = inv(method.m_sqrt[1])
    @views method.Γ[1, (d + 1):end] .= .-mm.c .* inv_sqrt_m
    @views method.Γ[(d + 1):end, 1] .= mm.c .* inv_sqrt_m
    method.Γ[1, 1] = a₀ / method.mass_vector[1]
    return nothing
end

# ----------------------------------------------------------------------------------
# The two surfaces
# ----------------------------------------------------------------------------------

"""
    GroundAdiabat(model)
    erpenbeck_thoss_ground(Γ_eV) → GroundAdiabat

The lower adiabat of the bare 2×2 ErpenbeckThoss Hamiltonian — the smallest eigenvalue of
`Hermitian([V₁₁ V₁₂; V₁₂ V₂₂])` — with no bath discretisation, in closed form:

```math
E_-(x) = \\frac{V_{11}+V_{22}}{2} - R, \\qquad
R = \\sqrt{\\Big(\\frac{V_{11}-V_{22}}{2}\\Big)^2 + V_{12}^2},
```
```math
E_-'(x) = \\frac{V_{11}'+V_{22}'}{2}
        - \\frac{1}{R}\\left[\\frac{V_{11}-V_{22}}{2}\\cdot\\frac{V_{11}'-V_{22}'}{2}
        + V_{12}V_{12}'\\right].
```

Verified against `MemoryKernelFitting.ErpenbeckThossModel`: `h = V₂₂ - V₁₁` and
`Δ = πV₁₂²` agree to all printed digits, so this is the same object the spectra were
computed for.

# Why not `AdiabaticStateSelector`
`AdiabaticStateSelector` is the library's way to do this and it does not work on the bare
model. Its `derivative!` loops `for I in eachindex(output, D)`, which assumes
`NQCModels.derivative(quantum_model, R)` returns an array of Hermitians shaped like `R`;
`ErpenbeckThoss` returns a single `Hermitian` 2×2, so the call raises a
`DimensionMismatch` (1×1 against 2×2). It only works wrapped in `WideBandBath`, which is
the bath this surface is meant to avoid. The closed form above is also exact and cheaper
than an eigendecomposition per step.

!!! warning "This surface is dissociative at 300 K"
    Past the crossing `V₂₂` falls below `V₁₁`, so the adiabat is essentially
    `min(V₁₁, V₂₂)`: a 6.9 meV barrier out of the neutral well at 1.92 Å against
    `k_BT = 25.9` meV, and an outer minimum at 3.16 Å only ~82 meV deep against the
    `V∞ = -1.5` eV asymptote. Use [`erpenbeck_thoss_neutral`](@ref) when a stationary
    distribution is the point.
"""
struct GroundAdiabat{M} <: NQCModels.ClassicalModels.ClassicalModel
    model::M
end

NQCModels.ndofs(::GroundAdiabat) = 1

# `R` is the gap term; it vanishes only where both the diabatic gap and the coupling do,
# which the ErpenbeckThoss coupling (bounded below by V̄ₖq > 0) never allows.
@inline function _adiabat_parts(m::GroundAdiabat, r::AbstractMatrix)
    V = NQCModels.potential(m.model, r)
    D = NQCModels.derivative(m.model, r)
    half_gap = (V[1, 1] - V[2, 2]) / 2
    Rgap = sqrt(half_gap^2 + V[1, 2]^2)
    return V, D, half_gap, Rgap
end

function NQCModels.potential(m::GroundAdiabat, r::AbstractMatrix)
    V, _, _, Rgap = _adiabat_parts(m, r)
    return (V[1, 1] + V[2, 2]) / 2 - Rgap
end

NQCModels.potential!(m::GroundAdiabat, V, r::AbstractMatrix) =
    (V .= NQCModels.potential(m, r))

function NQCModels.derivative(m::GroundAdiabat, r::AbstractMatrix)
    V, D, half_gap, Rgap = _adiabat_parts(m, r)
    dhalf_gap = (D[1, 1] - D[2, 2]) / 2
    return (D[1, 1] + D[2, 2]) / 2 - (half_gap * dhalf_gap + V[1, 2] * D[1, 2]) / Rgap
end

NQCModels.derivative!(m::GroundAdiabat, D::AbstractMatrix, r::AbstractMatrix) =
    (D .= NQCModels.derivative(m, r))

erpenbeck_thoss_ground(Γ_eV) =
    GroundAdiabat(NQCModels.QuantumModels.ErpenbeckThoss(; Γ = austrip(Γ_eV * u"eV")))

"""
    erpenbeck_thoss_neutral(Γ_eV) → Morse

The neutral diabat `V₁₁ = Morse + c`, `Dₑ = 3.52` eV, `x₀ = 1.78` Å, `ω₀ = 0.298` eV.

Bound, so a Gibbs measure exists and can be sampled — the surface to validate the FDT
machinery on before trusting the dissociative adiabat. The offset `c` is dropped: it is
a constant and shifts no force.
"""
erpenbeck_thoss_neutral(Γ_eV) =
    NQCModels.QuantumModels.ErpenbeckThoss(; Γ = austrip(Γ_eV * u"eV")).morse


"""
    MarkovianFriction(et, kT) → ElectronicFrictionProvider

`Λ(x, 0)` as an ordinary NQCDynamics friction provider, for the MDEF reference run.

This is the friction a standard MDEF simulation would use: the `ω → 0` limit of the same
spectral density the Prony series fits over all `ω`. Pairing it with the same PES makes
the GLE-vs-MDEF comparison a clean statement about memory, with nothing else changed.

Returns `Λ` in `mₑ/aut`, the units NQCModels' providers are expected to use; `friction!`
in `mdef.jl` divides by `√(mᵢmⱼ)` afterwards.
"""
struct MarkovianFriction{T,M} <: NQCModels.FrictionModels.ElectronicFrictionProvider
    ndofs::Int
    et::M
    kT::T
end

MarkovianFriction(et::ErpenbeckThossModel, kT) = MarkovianFriction(1, et, kT)

function friction!(model::MarkovianFriction, F::AbstractMatrix, R::AbstractMatrix)
    F[1, 1] = markovian_friction(R[1] * BOHR_IN_ANGSTROM, model.et, model.kT)
    return F
end

# ----------------------------------------------------------------------------------
# Running an ensemble
#
# NQCDynamics' `run_dynamics` takes `trajectories` and an `ensemble_algorithm`, and
# forwards both straight to `SciMLBase.solve`. Two facts make that usable here, and both
# are easy to get wrong:
#
# 1. **It is already thread-safe.** `EnsembleProblem`'s `safetycopy` defaults to
#    `prob_func !== DEFAULT_PROB_FUNC`, and NQCDynamics always installs its own
#    `prob_func` (`Selection`), so SciMLBase `deepcopy`s the whole problem — including
#    `p === sim` — once per trajectory. Nothing is shared: not `method.s`, not
#    `method.Γ`, not the `PronyMemory` scratch, not `sim.cache`.
#
# 2. **The deepcopy is also the trap.** It replicates *one* seeded `s`, so a naive
#    `trajectories = N` gives every trajectory the same bath initial condition: the
#    `s(0)` transient becomes common-mode instead of averaging out, and the ensemble
#    variance is understated.
#
# `SeededIC` closes (2). `OrderedSelection` — reached by passing `selection` — calls
# `sample_distribution(prob.p, distribution, j)` with the *already copied* simulation and
# the trajectory index `j`, which is the only per-trajectory hook `run_dynamics` exposes.
# Keying the stream on `j` rather than drawing from a shared RNG also makes the seeds
# independent of thread count, worker count and scheduling order.
# ----------------------------------------------------------------------------------

"""
    OutputAuxiliaryMeanSquare(sol, i) → ⟨s²⟩ of the final state

`run_dynamics` integrates a *copy* of the simulation, so `sim.method.s` is unreachable
from the caller afterwards, and `s` is not in the state vector `u` either. An output
function is the one place the integrated copy appears, as `sol.prob.p`.
"""
OutputAuxiliaryMeanSquare(sol, i) =
    sum(abs2, sol.prob.p.method.s) / length(sol.prob.p.method.s)

"""
    seed_auxiliaries!(sim, rng, kT)

Draw the auxiliary variables from their stationary law `N(0, k_BT·I)`.

A no-op for methods that have none — MDEF and Classical carry no state outside `u`, so
the same [`SeededIC`](@ref) serves all three branches of the comparison.
"""
seed_auxiliaries!(::AbstractSimulation, rng, kT) = nothing

function seed_auxiliaries!(sim::AbstractSimulation{<:GLE_MDEF}, rng, kT)
    s = sim.method.s
    s .= sqrt(kT) .* randn(rng, length(s))
    return nothing
end

"""
    SeededIC(u0, kT, seed)

A fixed `(x, p)` initial condition whose auxiliary bath is redrawn per trajectory.

`u0` is handed to every trajectory unchanged — this is a beam experiment, not a thermal
ensemble — while `s` is seeded from a stream keyed on the trajectory index, so the run is
reproducible under `EnsembleSerial`, `EnsembleThreads` and `EnsembleDistributed` alike.
Pass it as the `distribution` argument together with `selection = 1:trajectories`, which
is what selects the index-carrying `OrderedSelection` path.

!!! note "Reproducible in distribution, not bitwise"
    Only `s(0)` is keyed on the index. The SDE noise comes from task-local RNGs, so
    trajectory `k` is not bitwise identical across different worker counts.
"""
struct SeededIC{U,T}
    u0::U
    kT::T
    seed::UInt64
end

SeededIC(u0, kT, seed::Integer) = SeededIC(u0, kT, UInt64(seed))

function Ensembles.sample_distribution(sim::AbstractSimulation, d::SeededIC, i)
    seed_auxiliaries!(sim, Xoshiro(hash((d.seed, i))), d.kT)
    return copy(d.u0)
end

# `run_dynamics` also calls the two-argument form once, to build the template problem
# that is then deep-copied per trajectory. Nothing needs seeding there.
Ensembles.sample_distribution(::AbstractSimulation, d::SeededIC) = copy(d.u0)

"""
    ensemble_algorithm(; allow_threads = false) → SciMLBase.EnsembleAlgorithm

`EnsembleDistributed` if workers were added, otherwise `EnsembleSerial`.

!!! warning "`EnsembleThreads` is unsafe on the GLE path — do not enable it"
    `GLE_BAOAB` writes a module-level global on every step:
    `NQCDynamics/src/DynamicsMethods/IntegrationAlgorithms/mdef_baoab.jl:237` defines
    `global const energy_log = (KE=Float64[], PE=Float64[], AUX=Float64[])`, which
    `initialize!` (`:203–209`) empties and `perform_step!` (`:243–245`) pushes to. Under
    `EnsembleThreads` every trajectory `push!`es the same three vectors concurrently, so
    the resize races: `BoundsError` inside `unsafe_copyto!` on a good day, a segfault on
    a bad one, both non-deterministic. SciMLBase's per-trajectory `deepcopy` does not
    help — the global belongs to the module, not to the problem.

    Only `GLE_BAOAB` touches it, which is why the MDEF and Classical branches thread
    cleanly and the GLE one does not. Processes each get their own copy of the global, so
    `EnsembleDistributed` is unaffected — and is what this returns. `energy_log` is never
    read anywhere in NQCDynamics; deleting those six lines upstream would make
    `allow_threads = true` safe.
"""
function ensemble_algorithm(; allow_threads = false)
    Distributed.nworkers() > 1 && return Ensembles.SciMLBase.EnsembleDistributed()
    allow_threads && Threads.nthreads() > 1 &&
        return Ensembles.SciMLBase.EnsembleThreads()
    return Ensembles.SciMLBase.EnsembleSerial()
end
