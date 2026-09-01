#
# The configuration dependence, `x ↦ (a₀(x), a(x), A(x))`.
#
# `docs/design/0002-amplitude-representation.md` is the authority here. Two decisions
# from it shape every type below:
#
#   1. **Which variable is fitted depends on the stage.** Stage 2 fits
#      `g_k(x) = √(a_k(x))`: squaring makes non-negativity structural, sign changes of
#      `g_k` harmless, and halves the dynamic range (`a_k` spans 10–17 decades, `g_k`
#      5–9). Thesis §4.7.2 form. **Stage 1 fits `a_k` directly** — it is what the model
#      is linear in, and substituting `a = g²` there would make the inner problem
#      quartic and destroy the convexity the stage exists for. See the note on
#      `GridAmplitudes` below.
#   2. **Two stages.** Stage 1 fits amplitudes per grid node — convex, a single NNLS,
#      global optimum ([`GridAmplitudes`](@ref)). Only then, having *seen* `g_k(x)`, is
#      a basis for it chosen (stage 2). Choosing the basis first inverts the dependency
#      and turns a convex problem into a guess.
#
# ## The interface
#
# Every `AmplitudeModel` supports exactly five things, and the NNLS solver of the next
# milestone should be written against these and nothing else:
#
# | call | meaning |
# |---|---|
# | `n_configurations(m)` | number of grid nodes `N_x` |
# | `mode_counts(m)` | `(K_R, K_C)`; must agree with the [`PoleSet`](@ref) |
# | `amplitudes(m, i)` | `(a₀, a, A)` at node `i`; `a` has `K_R` entries, `A` has `K_C` |
# | `parameters(m)` | the free parameters as a fresh flat `Vector` |
# | `parameters!(m, θ)` | write a flat vector back; returns `m` |
#
# Configurations are addressed by **node index**, not by `x`. That is forced by
# `GridAmplitudes`, which has no value between nodes; and it is what the objective
# needs, since it sums over the fixed grid. The smooth subtypes precompute their
# evaluation coordinate per node at construction, so the graded `x` grid costs nothing
# in the inner loop.
#
# What `parameters` *means* differs by subtype, deliberately: for `GridAmplitudes` it is
# the amplitudes themselves (NNLS's decision variable, `a ≥ 0`), for the basis subtypes
# it is the coefficient matrix of `ĝ`. Both are "the free parameters of this model",
# which is all the optimiser needs to know.
#

"""
    n_configurations(m::AmplitudeModel) → N_x

Number of configurations (grid nodes) the model is defined at.
"""
function n_configurations end

"""
    mode_counts(m::AmplitudeModel) → (K_R, K_C)

Number of real modes and underdamped pairs the model carries amplitudes for. Must equal
`(length(ps.λ), length(ps.α))` of the [`PoleSet`](@ref) it is used with.
"""
function mode_counts end

"""
    amplitudes(m::AmplitudeModel, i) → (a₀, a, A)

The amplitude vector of `Prony_fit_optimisation_problem.md` §2 at node `i`: the white
part `a₀ ≥ 0`, the `K_R` real-mode amplitudes `a ≥ 0`, and the `K_C` pair amplitudes
`A ≥ 0`. Non-negativity holds by construction in every subtype.
"""
function amplitudes end

"""
    parameters(m::AmplitudeModel) → Vector

The model's free parameters as a fresh flat vector, for the optimiser.
"""
function parameters end

"""
    parameters!(m::AmplitudeModel, θ) → m

Inverse of [`parameters`](@ref): write a flat parameter vector back into `m`.
"""
function parameters! end

"""
    warp_coordinate(h, Δ, kT) → atan(h / (Δ + kT))

The monotone warp of `Prony_fit_optimisation_problem.md` §6, obstacle 2.

`Λ(x, 0)` rises two decades over the last `0.025` Å into the Fermi crossing `x₀` —
pole-like in `x`, hopeless for any global polynomial. But that feature is *entirely* the
pullback of a smooth function of `h` through a steep `h(x)` (`|dh/dx| = 8.4` eV/Å
there). In `h` it is a peak of width `Δ + k_BT`: measured HWHM `0.077` eV against
`Δ + k_BT = 0.076` eV at `Γ = 0.1`, and `0.599` against `0.523` eV at `Γ = 1.0`.

Dividing by that width and taking `arctan` therefore maps the crossing to an `O(1)`
feature, strictly monotone in `h`, on the bounded interval `(-π/2, π/2)`. On the
`Γ = 0.1` grid `h ∈ [-90, 8.6]` eV and `Δ + k_BT ≈ 0.076` eV, so `s` covers essentially
the whole interval.

All three arguments in Hartree.
"""
warp_coordinate(h, Δ, kT) = atan(h / (Δ + kT))

# --------------------------------------------------------------------------------
# Stage 1 — the grid
# --------------------------------------------------------------------------------

function _check_nonnegative(v, what)
    all(≥(0), v) || throw(ArgumentError("$what must be non-negative (FDT / Bochner)"))
    return nothing
end

"""
    GridAmplitudes(a₀, a, A)
    GridAmplitudes(N_x, K_R, K_C)

Stage 1 of `docs/design/0002-amplitude-representation.md`: one independent amplitude
vector per grid node, with no smoothness assumed between nodes.

  - `a₀` — `N_x` white parts
  - `a` — `K_R × N_x`, one column per node
  - `A` — `K_C × N_x`, one column per node

With the poles fixed the model is *linear* in these, so stage 1 is a single convex NNLS
with a global optimum and no initialisation risk (`Prony_fit_optimisation_problem.md`
§5). This is the subtype the inner solver fills in; the second constructor allocates a
zeroed model of the right shape for it to write into.

Node-major storage (`x` along columns) makes [`amplitudes`](@ref) a pair of contiguous
views, and matches the block-per-column structure the roughness penalty `μ` of §3
couples.

Unlike the basis subtypes this one stores `a`, not `g = √a`, because NNLS's decision
variable *is* `a`. Non-negativity is therefore checked rather than structural — both
constructors and [`parameters!`](@ref) reject a negative entry.
"""
struct GridAmplitudes{T<:Real} <: AmplitudeModel
    a₀::Vector{T}
    a::Matrix{T}
    A::Matrix{T}

    function GridAmplitudes{T}(a₀, a, A) where {T<:Real}
        length(a₀) == size(a, 2) == size(A, 2) || throw(DimensionMismatch(
            "a₀, a and A must cover the same nodes, got $(length(a₀)), $(size(a, 2)), $(size(A, 2))"))
        _check_nonnegative(a₀, "white amplitudes a₀")
        _check_nonnegative(a, "real-mode amplitudes a")
        _check_nonnegative(A, "pair amplitudes A")
        return new{T}(a₀, a, A)
    end
end

GridAmplitudes(a₀::AbstractVector, a::AbstractMatrix, A::AbstractMatrix) =
    GridAmplitudes{promote_type(float(eltype(a₀)), float(eltype(a)), float(eltype(A)))}(a₀, a, A)

GridAmplitudes(N_x::Integer, K_R::Integer, K_C::Integer) =
    GridAmplitudes(zeros(N_x), zeros(K_R, N_x), zeros(K_C, N_x))

n_configurations(m::GridAmplitudes) = length(m.a₀)
mode_counts(m::GridAmplitudes) = (size(m.a, 1), size(m.A, 1))
amplitudes(m::GridAmplitudes, i::Integer) =
    (m.a₀[i], view(m.a, :, i), view(m.A, :, i))

# Layout: one contiguous block (a₀, a, A) per node — the NNLS decision vector.
function parameters(m::GridAmplitudes{T}) where {T}
    K_R, K_C = mode_counts(m)
    w = 1 + K_R + K_C
    θ = Vector{T}(undef, n_configurations(m) * w)
    for i in 1:n_configurations(m)
        o = (i - 1) * w
        θ[o + 1] = m.a₀[i]
        copyto!(θ, o + 2, view(m.a, :, i))
        copyto!(θ, o + 2 + K_R, view(m.A, :, i))
    end
    return θ
end

function parameters!(m::GridAmplitudes, θ)
    K_R, K_C = mode_counts(m)
    w = 1 + K_R + K_C
    length(θ) == n_configurations(m) * w || throw(DimensionMismatch(
        "expected $(n_configurations(m) * w) parameters, got $(length(θ))"))
    _check_nonnegative(θ, "grid amplitudes")
    for i in 1:n_configurations(m)
        o = (i - 1) * w
        m.a₀[i] = θ[o + 1]
        copyto!(view(m.a, :, i), view(θ, (o + 2):(o + 1 + K_R)))
        copyto!(view(m.A, :, i), view(θ, (o + 2 + K_R):(o + w)))
    end
    return m
end

# --------------------------------------------------------------------------------
# Stage 2 — basis expansions of the shape function ĝ
# --------------------------------------------------------------------------------

# Both basis subtypes represent the *shape function* ĝ of §6 obstacle 1, with the known
# envelope √(Λ(x,0)) divided out:
#
#     g_c(x) = √(Λ(x,0)) · ĝ_c(u(x)),      a_c(x) = g_c(x)² ≥ 0
#
# and differ only in (u, basis): raw `x` with monomials, or the warped `s(x)` with
# Chebyshev. `C` is (degree+1) × (1 + K_R + K_C); column 1 is the white part, columns
# 2:1+K_R the real modes, the rest the pairs.

function _horner(c::AbstractVector, u)
    v = zero(promote_type(eltype(c), typeof(u)))
    for l in lastindex(c):-1:firstindex(c)
        v = muladd(v, u, c[l])
    end
    return v
end

# Clenshaw recurrence for Σ_l c[l+1] T_l(u); stable where the power basis is not.
function _clenshaw(c::AbstractVector, u)
    z = zero(promote_type(eltype(c), typeof(u)))
    isempty(c) && return z
    length(c) == 1 && return z + c[1]
    b1 = b2 = z
    for l in lastindex(c):-1:(firstindex(c) + 1)
        b1, b2 = muladd(2u, b1, c[l] - b2), b1
    end
    return muladd(u, b1, c[1] - b2)
end

"""
    PolynomialAmplitudes(C, x, envelope; K_R, K_C)

Monomials in **raw `x`**, with the envelope factored out: `g_c(x) = envelope · ĝ_c(x)`
and `ĝ_c(x) = Σ_l C[l, c] x^{l-1}`.

`x` is one raw configuration coordinate per node (Å — `x` is the one quantity *not*
converted to atomic units, see `docs/design/0001`), `envelope[i] = √(Λ(x_i, 0))`, and
`C` is `(degree+1) × (1 + K_R + K_C)`: column 1 is `a₀`, the next `K_R` the real modes,
the last `K_C` the pairs.

!!! note "This is the control experiment, and it is expected to fail"
    `docs/design/0002-amplitude-representation.md` keeps this basis as a first-class,
    tested subtype precisely because §6 predicts it will ring: degree `≳ 10` in raw `x`
    is Runge-unstable, and the graded grid (`dx` varies by 290× — 83% of the columns
    cover 6.7% of the range) makes it worse. Implementing it properly makes the ringing
    a *measured* result rather than an asserted one, which is what the thesis argument
    needs. For production use [`ChebyshevAmplitudes`](@ref).

Amplitudes are `g²`, so non-negativity is structural and `C` is unconstrained.
"""
struct PolynomialAmplitudes{T<:Real} <: AmplitudeModel
    C::Matrix{T}
    x::Vector{T}
    envelope::Vector{T}
    K_R::Int
    K_C::Int

    function PolynomialAmplitudes{T}(C, x, envelope, K_R, K_C) where {T<:Real}
        _check_basis(C, x, envelope, K_R, K_C)
        return new{T}(C, x, envelope, K_R, K_C)
    end
end

"""
    ChebyshevAmplitudes(C, s, envelope; K_R, K_C)

Chebyshev expansion of the shape function in the **warped** coordinate: with
`t = 2s/π ∈ [-1, 1]`,

```math
g_c(x) = \\sqrt{Λ(x,0)}\\;\\sum_{l=0}^{d} C_{l+1,c}\\,T_l(t(x)),
\\qquad s(x) = \\arctan\\!\\big(h(x)/(Δ(x) + k_BT)\\big).
```

`s` is one [`warp_coordinate`](@ref) value per node, `envelope[i] = √(Λ(x_i, 0))`, and
`C` has the same layout as in [`PolynomialAmplitudes`](@ref).

This is the recommended representation of §6. All three obstacles a polynomial in raw
`x` faces are removed: the envelope takes out the 5–9 decades of range, the warp takes
out the crossing, and Chebyshev in a bounded variable takes out the conditioning. What
is left is an `O(1)` shape function on `[-1, 1]`, on which a *low-degree* polynomial is
ample — so the original "polynomial in `x`" instinct survives, acting on `ĝ_c(s)`
instead of on `a_c(x)`.

`s` is rescaled by `π/2` rather than by its own observed range: `arctan` is bounded by
`±π/2` by construction, which is a fixed map needing no stored state, and on the
`Γ = 0.1` grid the nodes already cover `s ∈ [-1.570, 1.562]`, so the two choices differ
by well under a percent.

Amplitudes are `g²`, so non-negativity is structural and `C` is unconstrained.
"""
struct ChebyshevAmplitudes{T<:Real} <: AmplitudeModel
    C::Matrix{T}
    t::Vector{T}
    envelope::Vector{T}
    K_R::Int
    K_C::Int

    function ChebyshevAmplitudes{T}(C, s, envelope, K_R, K_C) where {T<:Real}
        _check_basis(C, s, envelope, K_R, K_C)
        all(σ -> abs(σ) ≤ π / 2, s) || throw(ArgumentError(
            "warped coordinates must lie in [-π/2, π/2]; pass warp_coordinate.(h, Δ, kT), not raw x"))
        return new{T}(C, (2 / π) .* s, envelope, K_R, K_C)
    end
end

function _check_basis(C, u, envelope, K_R, K_C)
    K_R ≥ 0 && K_C ≥ 0 || throw(ArgumentError("mode counts must be non-negative"))
    size(C, 2) == 1 + K_R + K_C || throw(DimensionMismatch(
        "C needs 1 + K_R + K_C = $(1 + K_R + K_C) columns, got $(size(C, 2))"))
    length(u) == length(envelope) || throw(DimensionMismatch(
        "node coordinates and envelope must have equal length"))
    _check_nonnegative(envelope, "the envelope √(Λ(x,0))")
    return nothing
end

const BasisAmplitudes = Union{PolynomialAmplitudes,ChebyshevAmplitudes}

for M in (:PolynomialAmplitudes, :ChebyshevAmplitudes)
    @eval $M(C::AbstractMatrix, u::AbstractVector, envelope::AbstractVector;
             K_R::Integer, K_C::Integer) =
        $M{promote_type(float(eltype(C)), float(eltype(u)), float(eltype(envelope)))}(
            C, u, envelope, K_R, K_C)
end

_shape(m::PolynomialAmplitudes, col, i) = _horner(view(m.C, :, col), m.x[i])
_shape(m::ChebyshevAmplitudes, col, i) = _clenshaw(view(m.C, :, col), m.t[i])

n_configurations(m::BasisAmplitudes) = length(m.envelope)
mode_counts(m::BasisAmplitudes) = (m.K_R, m.K_C)

function amplitudes(m::BasisAmplitudes, i::Integer)
    e = m.envelope[i]
    a₀ = abs2(e * _shape(m, 1, i))
    a = [abs2(e * _shape(m, 1 + k, i)) for k in 1:m.K_R]
    A = [abs2(e * _shape(m, 1 + m.K_R + j, i)) for j in 1:m.K_C]
    return a₀, a, A
end

parameters(m::BasisAmplitudes) = vec(copy(m.C))

function parameters!(m::BasisAmplitudes, θ)
    length(θ) == length(m.C) || throw(DimensionMismatch(
        "expected $(length(m.C)) coefficients, got $(length(θ))"))
    copyto!(m.C, θ)
    return m
end

# --------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------

"""
    fit_tag(::Type{<:AmplitudeModel}) → String
    fit_tag(m::AmplitudeModel) → String

Short identifier for the representation a fit used, for filenames and file attributes:
`"stage1Grid"`, `"stage2Poly"`, `"stage2Cheby"`.

Two fits of the same ladder under different representations are different models and
must not collide on disk. Deriving the tag from the type rather than writing a literal
into each script keeps the name from drifting away from what was actually fitted — the
type-domain method exists so a *reader* can build the expected filename before it has a
model to inspect.
"""
function fit_tag end

fit_tag(::Type{<:GridAmplitudes})       = "stage1Grid"
fit_tag(::Type{<:PolynomialAmplitudes}) = "stage2Poly"
fit_tag(::Type{<:ChebyshevAmplitudes})  = "stage2Cheby"
fit_tag(m::AmplitudeModel) = fit_tag(typeof(m))

# --------------------------------------------------------------------------------

"""
    spectral_density(ω, ps::PoleSet, m::AmplitudeModel, i) → Λ

Evaluate the model of `Prony_fit_optimisation_problem.md` §1 at frequency `ω` and node
`i`: rates from `ps`, amplitudes from `m`. This is the residual's forward map.
"""
function spectral_density(ω, ps::PoleSet, m::AmplitudeModel, i::Integer)
    a₀, a, A = amplitudes(m, i)
    return spectral_density(ω, ps, a₀, a, A)
end

"""
    memory_kernel(τ, ps::PoleSet, m::AmplitudeModel, i) → K

Time-domain counterpart of `spectral_density(ω, ::PoleSet, ::AmplitudeModel, i)`, less
the singular `2a₀(x_i)δ(τ)` part.
"""
function memory_kernel(τ, ps::PoleSet, m::AmplitudeModel, i::Integer)
    _, a, A = amplitudes(m, i)
    return memory_kernel(τ, ps, a, A)
end
