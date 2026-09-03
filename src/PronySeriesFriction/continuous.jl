#
# The stage-2 fit as a function of `x`, not as a table indexed by node.
#
# `AmplitudeModel` and its subtypes answer `amplitudes(m, i)` — the `i`-th stored
# configuration. That is the right interface for fitting, where the nodes *are* the
# data. It is the wrong one for a QGLE integrator, which evaluates at a continuously
# moving `q(t)` and never lands on a node. This file closes that gap for the
# `ChebyshevAmplitudes` representation, which is the only one §6 recommends for
# production.
#
# Nothing here is an approximation of the fit. Both ingredients of
#
#     g_c(x) = √(Λ(x,0)) · ĝ_c(t(x)),   t(x) = (2/π) arctan(h(x) / (Δ(x) + k_BT))
#
# are available in closed form — `t` from `warp_coordinate` and the analytic `h`, `Δ`
# of `ErpenbeckThossModel`, and the envelope from `markovian_friction` — so the
# continuous model and the fitted table agree to quadrature accuracy at every node.
#

"""
    ContinuousAmplitudes(C, warp, Λ₀; K_R, K_C)
    ContinuousAmplitudes(C, m::ErpenbeckThossModel, kT; K_R, K_C, kwargs...)

The [`ChebyshevAmplitudes`](@ref) fit evaluated at **arbitrary** `x`, rather than at the
stored nodes.

```math
a_0(x),\\ a_k(x),\\ A_j(x) \\;=\\; g_c(x)^2, \\qquad
g_c(x) = \\sqrt{Λ(x,0)}\\;\\sum_{l=0}^{d} C_{l+1,c}\\,T_l\\big(t(x)\\big).
```

# Fields
- `C :: Matrix` — the fitted coefficients, `(degree+1) × (1 + K_R + K_C)`, in exactly
  the layout `ChebyshevAmplitudes` uses: column 1 the white part `a₀`, columns
  `2:1+K_R` the real modes, the rest the pairs. Read it straight off `C_cheb` in a
  `prony_stage2Cheby_*.h5`.
- `warp :: Function` — `x ↦ t ∈ [-1, 1]`. Note this is the **rescaled** coordinate
  `t = (2/π)s`, the one `_clenshaw` wants, not `s = ` [`warp_coordinate`](@ref) itself.
- `Λ₀ :: Function` — `x ↦ Λ(x, 0)`, the zero-frequency friction. Not its square root:
  the envelope of §6 is `√(Λ(x,0))` and the square root is taken here, so that what
  the caller supplies is the physical quantity [`markovian_friction`](@ref) returns.

# The convenience constructor
`ContinuousAmplitudes(C, m::ErpenbeckThossModel, kT; K_R, K_C)` builds both closures
from the analytic model. Extra keywords (`halfwidth`, `nodes`) are forwarded to
[`markovian_friction`](@ref). This is the constructor to use; the general one exists so
that a different system, or a cheaper envelope, needs no change here.

# Units
`x` in Å, `kT` and `Λ₀` in atomic units — the [`LambdaSpectrum`](@ref) convention of
`docs/design/0001`. The amplitudes come out in the units of the fit: `a₀` in `mₑ/aut`,
`a_k` and `A_j` in `mₑ/aut × Hartree`.

!!! warning "Outside the fitted range this extrapolates a polynomial"
    The `Γ = 0.1` fit was made on `x ∈ [0.8, 5.0]` Å. `t(x)` stays bounded in `[-1, 1]`
    by construction, so the Chebyshev sum cannot blow up — but `ĝ_c` is only *fitted*
    where the nodes are, and the graded grid leaves `[0.8, 1.0]` covered by three nodes
    that straddle the inner Fermi crossing at `0.88054` Å (see [`fermi_crossing`](@ref)).
    Amplitudes returned there are an extrapolation, not a fit. On the ground adiabat
    that region sits above `50` eV and is unreachable at `300` K; check before assuming
    the same on another surface.

See also [`amplitudes`](@ref), [`amplitudes!`](@ref), [`qgle_blocks`](@ref).
"""
struct ContinuousAmplitudes{T<:Real,W,E}
    C::Matrix{T}
    warp::W
    Λ₀::E
    K_R::Int
    K_C::Int

    function ContinuousAmplitudes{T,W,E}(C, warp, Λ₀, K_R, K_C) where {T<:Real,W,E}
        K_R ≥ 0 && K_C ≥ 0 || throw(ArgumentError("mode counts must be non-negative"))
        size(C, 2) == 1 + K_R + K_C || throw(DimensionMismatch(
            "C needs 1 + K_R + K_C = $(1 + K_R + K_C) columns, got $(size(C, 2))"))
        size(C, 1) ≥ 1 || throw(ArgumentError("C needs at least one coefficient row"))
        return new{T,W,E}(C, warp, Λ₀, K_R, K_C)
    end
end

function ContinuousAmplitudes(C::AbstractMatrix, warp, Λ₀; K_R::Integer, K_C::Integer)
    T = float(eltype(C))
    return ContinuousAmplitudes{T,typeof(warp),typeof(Λ₀)}(
        collect(T, C), warp, Λ₀, Int(K_R), Int(K_C))
end

function ContinuousAmplitudes(C::AbstractMatrix, m::ErpenbeckThossModel, kT;
                              K_R::Integer, K_C::Integer, kwargs...)
    kT > 0 || throw(ArgumentError("the thermal energy must be positive, got $kT"))
    # `(2/π)s`, matching what `ChebyshevAmplitudes` stores in its `t` field: the
    # coefficients `C` were fitted against that argument, not against `s`.
    warp = x -> (2 / π) * warp_coordinate(level_position(x, m),
                                          hybridisation_width(x, m), kT)
    Λ₀ = x -> markovian_friction(x, m, kT; kwargs...)
    return ContinuousAmplitudes(C, warp, Λ₀; K_R, K_C)
end

mode_counts(m::ContinuousAmplitudes) = (m.K_R, m.K_C)

"""
    auxiliary_dimension(m::ContinuousAmplitudes) → K_R + 2K_C

The size of the auxiliary block the fit implies, matching
[`auxiliary_dimension(::PoleSet)`](@ref) for the ladder it was fitted against.
"""
auxiliary_dimension(m::ContinuousAmplitudes) = m.K_R + 2 * m.K_C

function Base.show(io::IO, m::ContinuousAmplitudes)
    print(io, "ContinuousAmplitudes(degree ", size(m.C, 1) - 1,
          ", K_R = ", m.K_R, ", K_C = ", m.K_C, ", m = ", auxiliary_dimension(m), ")")
end

"""
    amplitudes(m::ContinuousAmplitudes, x::Real) → (a₀, a, A)

The amplitudes at configuration `x` (in Å): the white part, the `K_R` real-mode
amplitudes, and the `K_C` pair amplitudes, all non-negative by construction because
they are squares.

Allocates two vectors. Inside an integrator use [`amplitudes!`](@ref).
"""
function amplitudes(m::ContinuousAmplitudes, x::Real)
    a = Vector{float(eltype(m.C))}(undef, m.K_R)
    A = Vector{float(eltype(m.C))}(undef, m.K_C)
    a₀ = amplitudes!(a, A, m, x)
    return a₀, a, A
end

"""
    amplitudes!(a, A, m::ContinuousAmplitudes, x::Real) → a₀

Fill `a` (length `K_R`) and `A` (length `K_C`) with the amplitudes at `x`, and return
the white part `a₀`. Allocation-free apart from whatever `m.Λ₀` does.

The envelope and the warp are each evaluated **once** per call and shared across all
`1 + K_R + K_C` Clenshaw sums — `Λ₀` is a quadrature and is by far the most expensive
thing here, so calling `amplitudes` in a loop over modes would be the one mistake worth
avoiding.
"""
function amplitudes!(a::AbstractVector, A::AbstractVector,
                     m::ContinuousAmplitudes, x::Real)
    length(a) == m.K_R || throw(DimensionMismatch(
        "expected $(m.K_R) real-mode amplitudes, got $(length(a))"))
    length(A) == m.K_C || throw(DimensionMismatch(
        "expected $(m.K_C) pair amplitudes, got $(length(A))"))

    Λ0 = m.Λ₀(x)
    Λ0 ≥ 0 || throw(DomainError(Λ0,
        "the zero-frequency friction Λ(x,0) must be non-negative; got $Λ0 at x = $x"))
    e = sqrt(Λ0)
    t = m.warp(x)

    a₀ = abs2(e * _clenshaw(view(m.C, :, 1), t))
    for k in 1:m.K_R
        a[k] = abs2(e * _clenshaw(view(m.C, :, 1 + k), t))
    end
    for j in 1:m.K_C
        A[j] = abs2(e * _clenshaw(view(m.C, :, 1 + m.K_R + j), t))
    end
    return a₀
end

"""
    spectral_density(ω, ps::PoleSet, m::ContinuousAmplitudes, x::Real) → Λ_θ(x, ω)

The fitted spectrum at an arbitrary configuration — the continuous counterpart of
[`spectral_density(ω, ps, m::AmplitudeModel, i)`](@ref).

Use it to check the fit against the data at a configuration the grid does not contain,
and to confirm the Markovian anchor: `spectral_density(0, ps, m, x)` should return
`m.Λ₀(x)` to the accuracy §4's anchor bought.
"""
function spectral_density(ω, ps::PoleSet, m::ContinuousAmplitudes, x::Real)
    a₀, a, A = amplitudes(m, x)
    return spectral_density(ω, a₀, a, ps.λ, A, ps.α, ps.β)
end

"""
    memory_kernel(τ, ps::PoleSet, m::ContinuousAmplitudes, x::Real) → K(x, τ)

The fitted kernel at an arbitrary configuration, excluding the `2a₀δ(τ)` white part —
the continuous counterpart of [`memory_kernel(τ, ps, m::AmplitudeModel, i)`](@ref).
"""
function memory_kernel(τ, ps::PoleSet, m::ContinuousAmplitudes, x::Real)
    _, a, A = amplitudes(m, x)
    return memory_kernel(τ, a, ps.λ, A, ps.α, ps.β)
end

"""
    qgle_blocks(ps::PoleSet, m::ContinuousAmplitudes, x::Real) → QGLEBlocks

The extended-variable blocks at an arbitrary configuration. Equivalent to
`qgle_blocks(ps, amplitudes(m, x)...)`, and the entry point a QGLE integrator wants
when it can afford to allocate; the in-place path is [`qgle_coupling!`](@ref).
"""
qgle_blocks(ps::PoleSet, m::ContinuousAmplitudes, x::Real) =
    qgle_blocks(ps, amplitudes(m, x)...)

"""
    qgle_coupling!(g, a, A, m::ContinuousAmplitudes, x::Real) → a₀

Rebuild the coupling vector `g` at configuration `x`, using `a` and `A` as scratch, and
return the white part `a₀`. This is the per-step call of a regime-(A) QGLE integrator:
`Γ₂,₂` and `Σ̃₂` are configuration-independent and were built once, so `g` is the whole
update. Allocation-free apart from `m.Λ₀`.
"""
function qgle_coupling!(g::AbstractVector, a::AbstractVector, A::AbstractVector,
                        m::ContinuousAmplitudes, x::Real)
    a₀ = amplitudes!(a, A, m, x)
    qgle_coupling!(g, a, A)
    return a₀
end
