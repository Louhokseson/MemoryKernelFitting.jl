# The objective of `Prony_fit_optimisation_problem.md` §3. Three things are assembled
# here and then held fixed for the whole fit — the residual scales `σ`, the quadrature
# weights `v` ([`trapezoid_weights`](@ref)) and the retained band `ℬ`
# ([`subsample_omega`](@ref)) — leaving the objective a function of the poles and the
# amplitudes alone, which is what variable projection needs.
#
# Index convention: `i` runs over configurations `x` (254, graded grid), `j` over
# retained frequencies `ω` (561, from a uniform grid). `vᵢ` is therefore a weight on the
# `x` axis — the `ω` grid needs none. `Λ` is `[N_ω, N_x]`: frequency down a column,
# configuration across.
#
# `ℬ` is taken to be **shared** across configurations rather than per-column as the
# `ℬᵢ` of §3 permits. That is what makes [`design_matrix`](@ref) a single matrix, built
# once and reused for all 254 columns, and it is the only form the block-banded inner
# NNLS of §5 can be assembled in.

# --------------------------------------------------------------------------------
# Residual scales
# --------------------------------------------------------------------------------

"""
    configuration_weights(s::LambdaSpectrum; θ = 0.5) → v

Configuration quadrature weights `vᵢ` for §3, blending the plain `x` measure with the
warped measure of [`warp_coordinate`](@ref), `σ(x) = atan(h/(Δ + k_BT))`:

```math
v = (1-θ) \\frac{v^{x}}{\\sum_i v^{x}_i} + θ \\frac{v^{σ}}{\\sum_i v^{σ}_i},
```

both trapezoidal, each normalised to sum to 1, so `sum(v) == 1` for every `θ`. `θ = 0`
is [`trapezoid_weights`](@ref) up to that normalisation; `θ = 1` is the pure warp.

**Why a blend, and not either end.** §3 sanctions "trapezoidal weights, optionally times
a physical importance weight". On this grid neither pure measure is usable — measured at
`Γ = 0.1` eV over the 254 retained columns:

| scheme | `|x-x₀| < 0.15` Å | `1.0 < x < 1.8` Å | `x > 3` Å |
|---|---|---|---|
| share of columns | 84% | 2.8% | 7.9% |
| `θ = 0` (trapezoid in `x`) | 8.0% | 16.7% | **46.4%** |
| `θ = 1` (trapezoid in `σ`) | 48.5% | **0.35%** | 0.16% |
| `θ = 0.5` | 28.2% | 8.5% | 23.3% |

Pure `x` hands 46% of the objective to twenty flat tail columns and 8% to the resonance
region that is 84% of the grid. Pure warp inverts that and starves `1.0 < x < 1.8` Å to a
third of a percent — the band carrying `|h|ₘₐₓ = 8.65` eV, hence `β_max`, hence the
timestep. Weighting it to zero does not *test* whether the top rungs are needed; it drops
them by fiat.

The warp is the right corrective because the feature it resolves is genuinely narrow in
`x`: `|dh/dx| ≈ 8.4` eV/Å at `x₀` against a width `Δ + k_BT ≈ 0.076` eV makes the
crossing `≈ 0.009` Å across, on a grid whose coarsest cell is 0.1 Å. In `σ` it is about
one radian.

**This weight never touches the amplitudes.** At `μ = 0` the columns decouple, so
[`fit_amplitudes`](@ref) solves each independently and `v` cancels from the inner problem
entirely. `vᵢ` enters only where columns are *summed*: [`objective_value`](@ref), any
ranking of one ladder against another, and — once it exists — the outer pole search. It
is a **ladder-selection instrument, not a fitting one**; changing `θ` cannot change a fit
at fixed ladder, only which ladder you would pick.
"""
function configuration_weights(s::LambdaSpectrum; θ = 0.5)
    0 ≤ θ ≤ 1 || throw(ArgumentError("θ must lie in [0, 1], got $θ"))
    vx = trapezoid_weights(s.x)
    vs = abs.(trapezoid_weights(warp_coordinate.(s.h, s.Δ, s.kT)))
    return (1 - θ) .* (vx ./ sum(vx)) .+ θ .* (vs ./ sum(vs))
end

"""
    residual_scales(Λ; ε = 1e-3) → σ

The residual scales `σᵢⱼ` of `Prony_fit_optimisation_problem.md` §3,

```math
σ_{ij} = \\max\\big(|Λ_{ij}|,\\ ε\\,Λ_i^{\\text{pk}}\\big),
\\qquad Λ_i^{\\text{pk}} = \\max_j Λ_{ij},
```

for `Λ` given as `[N_ω, N_x]`, one column per configuration, or as a single column.
The result has the shape of the input.

**Why relative.** `Λ` spans some 16 decades over the grid — `1.4e3` a.u. at the Fermi
crossing against `9.5e-12` far from it (`docs/design/0001`) — so dividing by
`σ ∝ |Λ|` is what stops an absolute least squares from seeing the resonance column
and nothing else.

**Why the floor.** The far tail decays as `ω⁻³`, outside the `ω⁻²` reach of any
Lorentzian sum, so a purely relative fit would spend its freedom where the model
provably cannot follow. Below `ε` of the column peak the floor caps a residual at
`(Λ/εΛ^pk)² < 1`. It also absorbs the negative entries that appear at `Γ ≥ 0.2` (§0
fact 4) with no special case, since only `|Λ|` enters.

The peak is taken **per column**, and over the columns *as passed* — i.e. over the
retained band `ℬ` when the caller has already subsampled. Both matter: a global peak
would floor every column outside the resonance into irrelevance, and a peak taken
over discarded frequencies would not be a scale of anything being fitted.

`ε` must be positive: it is the only thing keeping `σ` off zero where `Λ` crosses it.
A column whose peak is not positive is rejected rather than silently given a
non-positive floor.
"""
function residual_scales(Λ::AbstractMatrix; ε = 1e-3)
    ε > 0 || throw(ArgumentError("the floor fraction ε must be positive, got $ε"))
    σ = similar(Λ, float(eltype(Λ)))
    for i in axes(Λ, 2)
        col = view(Λ, :, i)
        pk = maximum(col)
        pk > 0 || throw(ArgumentError(
            "column $i has non-positive peak Λ^pk = $pk, so the floor εΛ^pk is not a scale"))
        floor_i = ε * pk
        for j in axes(Λ, 1)
            σ[j, i] = max(abs(col[j]), floor_i)
        end
    end
    return σ
end

residual_scales(Λ::AbstractVector; ε = 1e-3) =
    vec(residual_scales(reshape(Λ, :, 1); ε))

# --------------------------------------------------------------------------------
# The linear map from amplitudes to model values
# --------------------------------------------------------------------------------

"""
    design_matrix(ps::PoleSet, ω) → M

The model of `Prony_fit_optimisation_problem.md` §1 as a matrix: `M` is
`[length(ω), 1 + K_R + K_C]` with

```math
M_{j1} = 1,\\qquad
M_{j,1+k} = \\frac{λ_k}{λ_k^2 + ω_j^2},\\qquad
M_{j,1+K_R+l} = L(ω_j; α_l, β_l),
```

so that `M * [a₀; a; A]` is `Λ_θ(x, ω)` at every retained frequency. The column order
is the `(a₀, a, A)` of [`amplitudes`](@ref) and of `GridAmplitudes`'s flat parameter
layout, so an amplitude vector drops in without reshuffling.

This is the *whole* content of "the objective is separable" (§5): with the poles fixed
the model is this one matrix, identical for every configuration, and the fit is linear
in what remains. `M` is built once per pole set and reused across all `N_x` columns —
by [`objective_value`](@ref) here, and by the inner NNLS of §5 in `Fitting`.

`M` therefore carries the forward map of
[`spectral_density`](@ref MemoryKernelFitting.spectral_density) in a second form. That
the two agree to machine precision is the load-bearing test of this file: everything
downstream *fits* through `M` but is *read out* through the poles and amplitudes, so a
discrepancy would produce an SDE that does not realise what was fitted.
"""
function design_matrix(ps::PoleSet, ω::AbstractVector)
    K_R, K_C = length(ps.λ), length(ps.α)
    T = float(promote_type(eltype(ω), eltype(ps.β)))
    M = Matrix{T}(undef, length(ω), 1 + K_R + K_C)
    fill!(view(M, :, 1), one(T))
    for k in 1:K_R
        view(M, :, 1 + k) .= lorentzian.(ω, ps.λ[k])
    end
    for l in 1:K_C
        view(M, :, 1 + K_R + l) .= mirrored_lorentzian.(ω, ps.α[l], ps.β[l])
    end
    return M
end

"""
    markovian_anchor_row(ps::PoleSet) → c

The coefficients of the Markovian anchor of `Prony_fit_optimisation_problem.md` §4,

```math
a_0(x_i) + \\sum_k \\frac{a_k(x_i)}{λ_k}
         + \\sum_l A_l(x_i)\\,\\frac{α_l}{α_l^2 + β_l^2} = Λ(x_i, 0),
```

as a vector `c` with `c ⋅ [a₀; a; A] = Λ_θ(x_i, 0)`. Being linear in the amplitudes it
drops into the inner NNLS as an equality row, or as a heavily weighted residual.

`Λ(x, 0)` is the zero-frequency friction a standard MDEF simulation would use — the
single most physically important number in the fit, and the one quantity for which
this package must reduce to established practice.

Implemented as `vec(design_matrix(ps, [0]))`, which is not a shortcut but the statement
that the anchor *is* the model at `ω = 0`: `lorentzian(0, λ) = 1/λ` and
`mirrored_lorentzian(0, α, β) = α/(α² + β²)` are exactly the coefficients above. There
is deliberately no second copy of the formula to drift out of step.

!!! note "The mirror is a factor of two here"
    `mirrored_lorentzian(0, α, β)` is *twice* the `α/2(α² + β²)` a single un-mirrored
    peak would give: at `ω = 0` the two terms of `L` coincide exactly
    (`docs/design/0003`). Dropping the mirror as "outside the data" would halve every
    pair's contribution to precisely this number.
"""
markovian_anchor_row(ps::PoleSet) =
    vec(design_matrix(ps, [zero(eltype(ps.β))]))

# The amplitudes of every node as one `[1 + K_R + K_C, N_x]` matrix, in the column
# order of `design_matrix`, so that `M * Θ` is the model at every (ω, x). Generic over
# `AmplitudeModel`: the basis subtypes evaluate their expansion, `GridAmplitudes` hands
# back views.
function _amplitude_columns(m::AmplitudeModel)
    N_x = n_configurations(m)
    N_x ≥ 1 || throw(ArgumentError("the amplitude model carries no configurations"))
    K_R, K_C = mode_counts(m)
    a₀, a, A = amplitudes(m, 1)
    T = float(promote_type(typeof(a₀), eltype(a), eltype(A)))
    Θ = Matrix{T}(undef, 1 + K_R + K_C, N_x)
    for i in 1:N_x
        a₀, a, A = amplitudes(m, i)
        Θ[1, i] = a₀
        copyto!(view(Θ, 2:(1 + K_R), i), a)
        copyto!(view(Θ, (2 + K_R):(1 + K_R + K_C), i), A)
    end
    return Θ
end

# --------------------------------------------------------------------------------
# The roughness penalty
# --------------------------------------------------------------------------------

"""
    roughness_penalty(m::AmplitudeModel, x; μ = 1) → P

The roughness term of `Prony_fit_optimisation_problem.md` §3,

```math
P = μ \\sum_c \\sum_i \\frac{\\big(a_c(x_{i+1}) - a_c(x_i)\\big)^2}{x_{i+1} - x_i},
```

summed over **every** amplitude component `c` — the white part `a₀` included — and over
the `N_x - 1` intervals of the configuration grid `x` (Å, graded, strictly ascending).
A model with a single node has no intervals and no penalty.

**What it is for.** Neighbouring rungs of the fixed ladder overlap by construction
(`docs/design/0003`: consecutive `β` sit about one half-width apart), so they are
near-degenerate, and per-column independent fits distribute weight between them
arbitrarily. `Λ(x, ·)` comes out smooth and `a_c(x)` comes out jagged. The penalty is
what makes the amplitude *functions* well posed, hence what gives stage 2 of
`docs/design/0002` a curve worth compressing. `a₀` is included for the same reason and
not merely for symmetry: §4 records that a mode with `λ ≫ ω_max` is flat over the
window and indistinguishable from `a₀`, which is exactly such a degeneracy.

**Why `1/Δx`.** It makes the sum a Riemann approximation to `∫ (da_c/dx)² dx`, so what
is penalised is the *slope*, not the increment. On this grid that is not cosmetic: `dx`
spans a factor of 290, and with an undivided increment the 214 columns packed into
0.15 Å around the crossing could wander freely while the coarse tail was pinned.

**Why quadratic in `a`.** Written on `a_c` the penalty stays quadratic, so the inner
problem remains one convex NNLS — now block-tridiagonal, coupling neighbouring columns
instead of leaving them independent. §3 notes that penalising `g_c = √a_c` is better
scaled, and `docs/design/0002` explains why it must nevertheless wait for stage 2: it
is not quadratic in `a`, and stage 1's convexity is the whole reason stage 1 exists.

`Fitting` folds the identical quantity into the NNLS as appended rows
`√(μ/Δxᵢ)·(θᵢ₊₁ - θᵢ)` against a zero right-hand side; `P` is the squared 2-norm of
those rows, which is asserted in the tests so that the two cannot drift.

!!! warning "μ has no natural units"
    `a₀` carries the units of `Λ`, while `a_k` and `A_l` carry `Λ × frequency`
    (`docs/design/0001`), so this sum adds squares of dimensionally different
    quantities and no single `μ` is simultaneously right for both. That is the note's
    own form and is kept deliberately — §3's remark that `g = √a` is "better scaled" is
    the same observation. Choose `μ` by what it does to the fit, not by argument, and
    re-examine it if the white part and the modes come out decades apart.
"""
function roughness_penalty(m::AmplitudeModel, x::AbstractVector; μ = 1)
    N_x = n_configurations(m)
    length(x) == N_x || throw(DimensionMismatch(
        "the grid has $(length(x)) nodes but the amplitude model has $N_x"))
    Θ = _amplitude_columns(m)
    P = zero(float(promote_type(eltype(Θ), eltype(x))))
    for i in 1:(N_x - 1)
        Δx = x[i + 1] - x[i]
        Δx > 0 || throw(ArgumentError(
            "the configuration grid must be strictly ascending, got x[$i] = $(x[i]), " *
            "x[$(i + 1)] = $(x[i + 1])"))
        c = zero(P)
        for r in axes(Θ, 1)
            d = Θ[r, i + 1] - Θ[r, i]
            c += d * d
        end
        P += c / Δx
    end
    return μ * P
end

# --------------------------------------------------------------------------------
# The objective
# --------------------------------------------------------------------------------

"""
    objective_value(ps, m, ω, Λ, v, σ; μ = 0, x = nothing) → J
    objective_value(ps, m, s::LambdaSpectrum; idx, ε = 1e-3, μ = 0) → J

The objective of `Prony_fit_optimisation_problem.md` §3, at poles `ps::PoleSet` and
amplitudes `m::AmplitudeModel`:

```math
J = \\sum_{i=1}^{N_x} v_i \\sum_{j \\in ℬ}
    \\left[\\frac{Λ_θ(x_i, ω_j) - Λ_{ij}}{σ_{ij}}\\right]^2
  + μ \\sum_c \\sum_i \\frac{\\big(a_c(x_{i+1}) - a_c(x_i)\\big)^2}{x_{i+1} - x_i}.
```

`ω` is the retained band ([`subsample_omega`](@ref)), `Λ` the data on that band as
`[length(ω), N_x]`, `v` the quadrature weights ([`trapezoid_weights`](@ref)) and `σ`
the scales ([`residual_scales`](@ref)). The roughness term is off by default; passing
`μ ≠ 0` requires the grid `x`, since a penalty on slopes has to know the spacing.

The weights are a required argument rather than a defaulted one because
`docs/design/0001` calls them non-negotiable: 214 of the 257 columns lie within 0.15 Å
of the crossing, so `vᵢ ≡ 1` makes `J` the resonance alone and the remaining 93% of the
range contributes nothing.

**Reading the number.** With the zero model every residual whose floor is inactive is
exactly `-1`, so

```math
J_0 \\le |ℬ| \\sum_i v_i = |ℬ|\\,(x_N - x_1),
```

with equality when no floor is active anywhere. `J / (|ℬ| (x_N - x_1))` is therefore a
mean squared relative error: `1` for the zero model, `1e-4` for a fit that is uniformly
1% off. `J` itself carries the Å of the weights.

The `LambdaSpectrum` method is the convenience form — it subsamples, weights and scales
the file itself, defaulting `idx = subsample_omega(s.ω)`. It rebuilds `σ` and `v` on
every call, so **do not use it inside a fitting loop**: assemble those once and call the
explicit method.
"""
function objective_value(ps::PoleSet, m::AmplitudeModel, ω::AbstractVector,
                         Λ::AbstractMatrix, v::AbstractVector, σ::AbstractMatrix;
                         μ = 0, x = nothing)
    N_x = n_configurations(m)
    mode_counts(m) == (length(ps.λ), length(ps.α)) || throw(DimensionMismatch(
        "the amplitude model carries $(mode_counts(m)) modes but the pole set has " *
        "$((length(ps.λ), length(ps.α)))"))
    size(Λ) == (length(ω), N_x) || throw(DimensionMismatch(
        "Λ is $(size(Λ)) but the band has $(length(ω)) frequencies and the model $N_x " *
        "configurations"))
    size(σ) == size(Λ) || throw(DimensionMismatch(
        "the residual scales are $(size(σ)) but Λ is $(size(Λ))"))
    length(v) == N_x || throw(DimensionMismatch(
        "there are $(length(v)) quadrature weights for $N_x configurations"))

    R = design_matrix(ps, ω) * _amplitude_columns(m)
    J = zero(float(promote_type(eltype(R), eltype(Λ), eltype(σ), eltype(v))))
    @inbounds for i in 1:N_x
        c = zero(J)
        for j in axes(Λ, 1)
            r = (R[j, i] - Λ[j, i]) / σ[j, i]
            c += r * r
        end
        J += v[i] * c
    end

    if !iszero(μ)
        x === nothing && throw(ArgumentError(
            "the roughness penalty is a penalty on slopes and needs the configuration " *
            "grid; pass x = ... alongside μ = $μ"))
        J += roughness_penalty(m, x; μ)
    end
    return J
end

function objective_value(ps::PoleSet, m::AmplitudeModel, s::LambdaSpectrum;
                         idx = subsample_omega(s.ω), ε = 1e-3, μ = 0)
    Λ = s.Λ[idx, :]
    return objective_value(ps, m, s.ω[idx], Λ, trapezoid_weights(s.x),
                           residual_scales(Λ; ε); μ, x = s.x)
end
