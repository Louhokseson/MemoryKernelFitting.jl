# Stage 1: the convex inner solve. One NNLS per configuration, since without the
# roughness penalty the columns of the objective are independent.
#
# Index convention: `i` runs over configurations `x`, `j` over retained frequencies `ω`.
# The weighted problem at configuration i is
#
#     min_{θ ≥ 0}  Σ_j [ (M θ - Λ_ij) / σ_ij ]²
#
# which is an ordinary NNLS on the row-scaled pair (M ./ σᵢ, Λᵢ ./ σᵢ). The quadrature
# weight vᵢ multiplies the whole column and therefore **cancels in the per-column
# argmin** — it changes the value of `objective_value`, not the amplitudes. It starts
# mattering the moment `μ` couples the columns.

"""
    fit_column(M, y, σ; anchor = nothing, anchor_weight = 0, alg = :nnls) → θ

Solve one configuration's non-negative least squares problem:

```math
θ = \\arg\\min_{θ ≥ 0} \\sum_j \\left[\\frac{(Mθ)_j - y_j}{σ_j}\\right]^2,
```

with `M` the [`design_matrix`](@ref) shared by every configuration, `y` the data column
and `σ` its [`residual_scales`](@ref). Returns the amplitude vector `θ = (a₀, a, A)` in
the design matrix's column order, all entries `≥ 0`.

Convex, so this is a global optimum — the whole reason `docs/design/0003` fixes the
ladder instead of searching for poles.

**The Markovian anchor.** Passing `anchor = (row, target)` with `anchor_weight > 0`
appends the extra residual `w(row ⋅ θ - target)/σ₁`, i.e. the §4 anchor equation
enforced as a heavily weighted row rather than a hard equality. `row` is
[`markovian_anchor_row`](@ref) and `target` is `Λ(x_i, 0)`.

Note what this does and does not add. Since [`subsample_omega`](@ref) always retains
`j = 1`, the anchor is *already* one of the `|ℬ|` residuals — on the `Γ = 0.1` grid
`ω₁ = 1e-6` eV, where the anchor row and row 1 of `M` agree to about `1e-10` relative.
So `anchor_weight` does not introduce a new condition; it decides how much more than
`1/561` of the fit that one physically load-bearing number is worth. The added weight
is `w²`. Left at `0` by default because §4 calls the equality *recommended*, not
required, and its effect is something to measure rather than assume.

!!! warning "Weight is not free"
    A large `w` makes the anchor row dominate the normal equations and costs
    conditioning in every other direction. Measured on data the model represents
    exactly, the max relative error in the recovered amplitudes runs `2e-15` at
    `w = 0`, `2.5e-5` at `w = 1`, `9e-4` at `w = 10`, and then **saturates** at
    `1.4e-3` for `w ≳ 100`. It plateaus rather than diverging, so a large weight is
    safe — past the point where the anchor is an equality it simply stops buying
    anything, while the rest of the fit keeps that fixed 1e-3 of slack.
"""
function fit_column(M::AbstractMatrix, y::AbstractVector, σ::AbstractVector;
                    anchor = nothing, anchor_weight = 0, alg::Symbol = :nnls)
    length(y) == length(σ) == size(M, 1) || throw(DimensionMismatch(
        "M has $(size(M, 1)) rows, the data column $(length(y)) and the scales $(length(σ))"))
    A = M ./ σ
    b = y ./ σ
    if !iszero(anchor_weight)
        anchor === nothing && throw(ArgumentError(
            "anchor_weight = $anchor_weight was given without an anchor; pass " *
            "anchor = (markovian_anchor_row(ps), Λ_i1)"))
        row, target = anchor
        length(row) == size(M, 2) || throw(DimensionMismatch(
            "the anchor row has $(length(row)) entries for $(size(M, 2)) amplitudes"))
        w = anchor_weight / σ[1]
        A = vcat(A, reshape(w .* row, 1, :))
        b = vcat(b, w * target)
    end
    return vec(nonneg_lsq(A, b; alg))
end

"""
    fit_amplitudes(ps::PoleSet, ω, Λ; ε = 1e-3, anchor_weight = 0, alg = :nnls) → GridAmplitudes
    fit_amplitudes(ps::PoleSet, s::LambdaSpectrum; idx, ε, anchor_weight, alg) → GridAmplitudes

Stage 1 of `docs/design/0002-amplitude-representation.md`: fit the amplitudes at every
configuration with the poles held fixed.

`ω` is the retained band and `Λ` the data on it as `[length(ω), N_x]`. The scales are
built internally with [`residual_scales`](@ref), so the same `ε` reaches both the fit
and any later [`objective_value`](@ref) call — passing mismatched scales is the easiest
way to make a fit look better than it is.

Returns a [`GridAmplitudes`](@ref): `N_x` independent non-negative amplitude vectors,
one per node, which is the curve stage 2 compresses.

Each column is an independent convex problem ([`fit_column`](@ref)), so there is no
initialisation, no local minimum and no convergence criterion to tune. The design
matrix is built once and shared; only the row scaling `σᵢ` differs between columns,
which is precisely why this is a loop rather than one multi-right-hand-side solve.

`anchor_weight` is forwarded to [`fit_column`](@ref); the anchor target is taken from
`Λ[1, i]`, which is `Λ(x_i, ω₁)` — the zero-frequency friction, retained by
[`subsample_omega`](@ref) by construction.
"""
function fit_amplitudes(ps::PoleSet, ω::AbstractVector, Λ::AbstractMatrix;
                        ε = 1e-3, anchor_weight = 0, alg::Symbol = :nnls)
    size(Λ, 1) == length(ω) || throw(DimensionMismatch(
        "Λ has $(size(Λ, 1)) rows for $(length(ω)) frequencies"))
    K_R, K_C = length(ps.λ), length(ps.α)
    N_x = size(Λ, 2)
    M = design_matrix(ps, ω)
    σ = residual_scales(Λ; ε)
    row = iszero(anchor_weight) ? nothing : markovian_anchor_row(ps)

    Θ = Matrix{Float64}(undef, 1 + K_R + K_C, N_x)
    for i in 1:N_x
        anchor = row === nothing ? nothing : (row, Λ[1, i])
        Θ[:, i] = fit_column(M, view(Λ, :, i), view(σ, :, i);
                             anchor, anchor_weight, alg)
    end
    # `GridAmplitudes`'s flat layout is one contiguous (a₀, a, A) block per node, i.e.
    # exactly `vec` of this matrix — the same convention `design_matrix` uses.
    return parameters!(GridAmplitudes(N_x, K_R, K_C), vec(Θ))
end

function fit_amplitudes(ps::PoleSet, s::LambdaSpectrum;
                        idx = subsample_omega(s.ω), ε = 1e-3, anchor_weight = 0,
                        alg::Symbol = :nnls)
    return fit_amplitudes(ps, s.ω[idx], s.Λ[idx, :]; ε, anchor_weight, alg)
end

"""
    relative_errors(ps::PoleSet, m::AmplitudeModel, ω, Λ) → E

The pointwise relative error `E[j, i] = (Λ_θ(x_i, ω_j) - Λ_ij) / |Λ_ij|`, as
`[length(ω), N_x]`.

This is the quantity §5's benchmark is quoted in — a single mode gives 5–25% median
relative error, greedy pursuit plateaus at 2–10%, full-dictionary NNLS reaches 0.1–5% —
so it is what calibrates the achievable error against `K_C`, the explicit open item of
§7. Summarise it however the question needs; the median over `j` per column is the
usual reading.

Note this is *not* the objective's residual: it has no `σ` floor, so points in the far
`ω⁻³` tail — which no Lorentzian sum can reach and which the floor deliberately
discounts — appear here at full weight. A fit can therefore look considerably worse in
`relative_errors` than in [`objective_value`](@ref) without anything being wrong. Where
`Λ = 0` the ratio is undefined and comes back as `Inf`; `abs` in the denominator keeps
the negative entries of `Γ ≥ 0.2` (§0 fact 4) meaningful.
"""
function relative_errors(ps::PoleSet, m::AmplitudeModel, ω::AbstractVector,
                         Λ::AbstractMatrix)
    N_x = n_configurations(m)
    size(Λ) == (length(ω), N_x) || throw(DimensionMismatch(
        "Λ is $(size(Λ)) but the band has $(length(ω)) frequencies and the model $N_x " *
        "configurations"))
    E = Matrix{Float64}(undef, length(ω), N_x)
    for i in 1:N_x
        a₀, a, A = amplitudes(m, i)
        for (j, ωj) in enumerate(ω)
            E[j, i] = (spectral_density(ωj, ps, a₀, a, A) - Λ[j, i]) / abs(Λ[j, i])
        end
    end
    return E
end
