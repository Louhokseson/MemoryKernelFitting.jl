# Track B — the outer VARPRO search over a regime-(B) pole model.
#
# `Prony_fit_optimisation_problem.md` §5 states the separation this file exercises: the
# model is linear in the amplitudes and nonlinear only in the poles, so the poles can be
# searched over an objective in which the amplitudes have already been eliminated by the
# inner solve. Stage 1 never needed it — with the ladder fixed there is nothing to search
# — which is why §5 marks it *optional* and it has not existed until now.
#
# What makes it cheap is that the inner solve is unchanged. `fit_column` does not care
# where its design matrix came from, so a per-configuration ladder costs one rebuild of
# `M` per column and nothing else: still convex, still Lawson–Hanson, still a global
# optimum *given* the poles. The columns are also independent given the poles, so the
# inner solve is embarrassingly parallel — run with `julia -t auto`; the objective value
# is summed in index order and does not depend on the thread count. The nonconvexity is confined to `5K_C + 3K_R` outer
# coefficients — 13 for the two-pair, one-rate case.
#
# Nelder–Mead, not a gradient method. The reduced objective is only **piecewise** smooth:
# the NNLS active set changes discontinuously as a pole moves, so the derivative of
# `J(θ)` jumps whenever a mode enters or leaves the support. This is the standard
# obstruction to gradient VARPRO with bound constraints, and at these dimensions a
# derivative-free simplex is both simpler and safer than defending a gradient against it.
#
# Read `TrackingPoles`' own warning before reading a number out of here: §9.7 settles
# regime (B) negatively on this data, and this path exists to *measure* what it gives up,
# not to produce a model.

"""
    tracking_ladders(tp::TrackingPoles, h, Δ, kT) → Vector{PoleSet}

The ladder at every configuration — [`poleset`](@ref) mapped over the grid. `h` and `Δ`
are per-configuration (Hartree); `kT` is one scalar.

Returned for reporting rather than used by the solver, which rebuilds one ladder at a
time to avoid holding `N_x` of them. Feeding these to `is_identifiable` column by column
is how a tracking fit is checked against the §4 box, which a fixed ladder only has to
pass once.
"""
function tracking_ladders(tp::TrackingPoles, h::AbstractVector, Δ::AbstractVector, kT)
    length(h) == length(Δ) || throw(DimensionMismatch(
        "h has $(length(h)) entries and Δ has $(length(Δ))"))
    return [poleset(tp, h[i], Δ[i], kT) for i in eachindex(h)]
end

# In-place design-matrix fill. Not a second copy of the model: it calls the same
# `lorentzian` / `mirrored_lorentzian` that `design_matrix` calls, and only the loop is
# written twice, so the *formula* still lives in one place (`basis.jl`). It exists
# because this is the hot loop — one rebuild per column per objective evaluation, tens of
# millions of times over a sweep — and `design_matrix` allocates a fresh matrix each
# call. `test_tracking_poles.jl` asserts the two agree exactly.
function _fill_design!(M::AbstractMatrix, ps::PoleSet, ω::AbstractVector)
    K_R, K_C = length(ps.λ), length(ps.α)
    size(M) == (length(ω), 1 + K_R + K_C) || throw(DimensionMismatch(
        "design buffer is $(size(M)) but the ladder needs $((length(ω), 1 + K_R + K_C))"))
    fill!(view(M, :, 1), one(eltype(M)))
    @inbounds for k in 1:K_R
        @views M[:, 1 + k] .= lorentzian.(ω, ps.λ[k])
    end
    @inbounds for l in 1:K_C
        @views M[:, 1 + K_R + l] .= mirrored_lorentzian.(ω, ps.α[l], ps.β[l])
    end
    return M
end

"""
    tracking_solve(tp, ω, Λ, σ, h, Δ, kT, v; anchor_weight = 0, alg = :nnls)
        → (; J, model)

The **inner** half of the variable projection, at fixed poles: one NNLS per
configuration against that configuration's own ladder, and the resulting §3 objective

```math
J = \\sum_i v_i \\sum_{j∈ℬ} \\left[\\frac{Λ_θ(x_i,ω_j) - Λ_{ij}}{σ_{ij}}\\right]^2 .
```

`J` excludes the anchor term even when `anchor_weight > 0`, exactly as
[`objective_value`](@ref) does — §3's `J` has no anchor in it, so including one here
would make tracking fits incomparable with the fixed-ladder numbers they exist to be
compared against.

Returns the objective and the amplitudes as a [`GridAmplitudes`](@ref). Because
`mode_counts(tp)` is the same at every configuration, the amplitude table is rectangular
and the ordinary grid representation applies unchanged — only the *ladder* varies with
`x`, never the number of modes.
"""
function tracking_solve(tp::TrackingPoles, ω::AbstractVector, Λ::AbstractMatrix,
                        σ::AbstractMatrix, h::AbstractVector, Δ::AbstractVector, kT,
                        v::AbstractVector; anchor_weight = 0, alg::Symbol = :nnls)
    N_x = size(Λ, 2)
    K_R, K_C = mode_counts(tp)
    # One design buffer per thread. `:static` scheduling is what makes indexing them by
    # `threadid()` correct — under the default scheduler a task may migrate between
    # threads mid-iteration and two columns could then share a buffer.
    Ms = [Matrix{Float64}(undef, length(ω), 1 + K_R + K_C) for _ in 1:Threads.nthreads()]
    a₀ = Vector{Float64}(undef, N_x)
    a = Matrix{Float64}(undef, K_R, N_x)
    A = Matrix{Float64}(undef, K_C, N_x)
    # Per-column, then summed in index order: the total is bit-identical however many
    # threads run it, which a running `J += …` inside the loop would not be.
    Jcol = Vector{Float64}(undef, N_x)
    Threads.@threads :static for i in 1:N_x
        M = Ms[Threads.threadid()]
        @inbounds begin
            ps = poleset(tp, h[i], Δ[i], kT)
            _fill_design!(M, ps, ω)
            yi, σi = view(Λ, :, i), view(σ, :, i)
            θ = iszero(anchor_weight) ?
                fit_column(M, yi, σi; alg) :
                fit_column(M, yi, σi; anchor = (markovian_anchor_row(ps), Λ[1, i]),
                           anchor_weight, alg)
            a₀[i] = θ[1]
            copyto!(view(a, :, i), view(θ, 2:(1 + K_R)))
            copyto!(view(A, :, i), view(θ, (2 + K_R):(1 + K_R + K_C)))
            c = 0.0
            for j in axes(Λ, 1)
                r = 0.0
                for p in axes(M, 2)
                    r += M[j, p] * θ[p]
                end
                d = (r - Λ[j, i]) / σ[j, i]
                c += d * d
            end
            Jcol[i] = v[i] * c
        end
    end
    return (; J = sum(Jcol), model = GridAmplitudes(a₀, a, A))
end

"""
    fit_tracking(tp0, ω, Λ, h, Δ, kT, v; ε = 1e-3, anchor_weight = 0,
                 iterations = 4000, g_tol = 1e-10, verbose = false)
        → (; poles, model, J, converged, iterations, evaluations)
    fit_tracking(tp0, s::LambdaSpectrum; idx, ε, anchor_weight, θv, …)

Fit a [`TrackingPoles`](@ref) by variable projection: Nelder–Mead over the
`5K_C_track + 3K_R` pole coefficients, with [`tracking_solve`](@ref) eliminating the
amplitudes at every step.

`tp0` supplies both the starting coefficients and the *shape* of the problem — the mode
counts, the fixed tail and the rate floor are read from it and never varied.
[`tracking_seed`](@ref) builds one from §6's measured values.

`v` are the configuration weights. Unlike stage 1, where §3's `vᵢ` cancels from the
per-column argmin and only affects which ladder you would pick, **here it is a genuine
fitting weight**: the poles are shared through the coefficients, so every column pulls on
the same `θ` and their relative weight decides the answer. Use
[`configuration_weights`](@ref) rather than the plain trapezoid, for the reason §3 gives.

The returned `J` is the §3 objective at the optimum, comparable with
[`objective_value`](@ref) on a fixed ladder at the same `ε` and `v` — that comparison
being the entire purpose of this path.

!!! note "Convergence is not a global claim"
    The outer problem is nonconvex, so `converged` means Nelder–Mead's simplex stopped
    moving, not that the optimum is global. This is the trade regime (B) makes and stage
    1 does not: with the ladder fixed there is no outer problem and the NNLS answer *is*
    the global one. Report the value, not the certainty.
"""
function fit_tracking(tp0::TrackingPoles, ω::AbstractVector, Λ::AbstractMatrix,
                      h::AbstractVector, Δ::AbstractVector, kT, v::AbstractVector;
                      ε = 1e-3, anchor_weight = 0, iterations::Integer = 4000,
                      g_tol = 1e-10, alg::Symbol = :nnls, verbose::Bool = false)
    N_x = size(Λ, 2)
    size(Λ, 1) == length(ω) || throw(DimensionMismatch(
        "Λ is $(size(Λ)) but the band has $(length(ω)) frequencies"))
    length(h) == length(Δ) == length(v) == N_x || throw(DimensionMismatch(
        "h, Δ and v must each have $N_x entries, got " *
        "$(length(h)), $(length(Δ)), $(length(v))"))

    σ = residual_scales(Λ; ε)
    scratch = deepcopy(tp0)
    θ0 = parameters(tp0)

    nev = Ref(0)
    function objective(θ)
        nev[] += 1
        parameters!(scratch, θ)
        return tracking_solve(scratch, ω, Λ, σ, h, Δ, kT, v; anchor_weight, alg).J
    end

    res = optimize(objective, θ0, NelderMead(),
                   Optim.Options(; iterations, g_tol, show_trace = verbose))

    fitted = deepcopy(tp0)
    parameters!(fitted, Optim.minimizer(res))
    out = tracking_solve(fitted, ω, Λ, σ, h, Δ, kT, v; anchor_weight, alg)
    return (; poles = fitted, model = out.model, J = out.J,
            converged = Optim.converged(res), iterations = Optim.iterations(res),
            evaluations = nev[])
end

function fit_tracking(tp0::TrackingPoles, s::LambdaSpectrum;
                      idx = subsample_omega(s.ω), ε = 1e-3, anchor_weight = 0,
                      v = nothing, kwargs...)
    Λ = s.Λ[idx, :]
    w = v === nothing ? fill(1 / length(s.x), length(s.x)) : v
    return fit_tracking(tp0, s.ω[idx], Λ, s.h, s.Δ, s.kT, w;
                        ε, anchor_weight, kwargs...)
end

"""
    tracking_errors(tp, m::AmplitudeModel, ω, Λ, h, Δ, kT) → E

Signed relative errors `(Λ_θ - Λ)/|Λ|` on the retained band, as `[length(ω), N_x]` — the
per-column-ladder counterpart of [`relative_errors`](@ref), and the quantity every
comparison in `docs/design/0003` is stated in.

Signed rather than absolute for the reason that record gives: the far tail is a
systematic one-sided undershoot (`Λ` decays as `ω⁻³`, outside the `ω⁻²` reach of any
Lorentzian sum), and taking the modulus would hide that it is structural.
"""
function tracking_errors(tp::TrackingPoles, m::AmplitudeModel, ω::AbstractVector,
                         Λ::AbstractMatrix, h::AbstractVector, Δ::AbstractVector, kT)
    N_x = n_configurations(m)
    size(Λ) == (length(ω), N_x) || throw(DimensionMismatch(
        "Λ is $(size(Λ)) but the band has $(length(ω)) frequencies and the model $N_x " *
        "configurations"))
    E = Matrix{Float64}(undef, length(ω), N_x)
    for i in 1:N_x
        ps = poleset(tp, h[i], Δ[i], kT)
        a₀, a, A = amplitudes(m, i)
        for (j, ωj) in enumerate(ω)
            E[j, i] = (spectral_density(ωj, ps, a₀, a, A) - Λ[j, i]) / abs(Λ[j, i])
        end
    end
    return E
end

"""
    fit_tracking_basis(Model, tp, ω, Λ, u, envelope, h, Δ, kT; degree, kwargs...)
        → (model, report)

Stage 2 on top of track B: compress the amplitudes into a `Model` basis expansion
(`ChebyshevAmplitudes` or `PolynomialAmplitudes`) while the poles are the fitted
[`TrackingPoles`](@ref) `tp`, held fixed.

This is the fourth corner of the two-cost picture. Stage 2 walks along the *parameter*
axis at fixed `m`; `fit_tracking` walks along the *`m`* axis at fixed parameter count;
this does both. The two moves are independent, so nothing new is derived here — the same
`_fg!` and the same analytic `∂J/∂C = Bᵀ(4 v e² G ⊙ P)` serve, with `P` assembled per
column because the ladder now varies with `x`.

Arguments follow [`fit_amplitudes`](@ref)`(::Type{Model}, ps, …)` exactly, with the pole
set replaced by `tp` plus the three configuration features `(h, Δ, kT)` the pole law reads.
`init` should be the [`GridAmplitudes`](@ref) that `fit_tracking` returned for the *same*
`tp`, which is what makes this stage 2 rather than a fresh nonconvex fit — the same
ordering, and for the same reason, as `docs/design/0002` step 2.

Returns the model **and** the report, unlike `fit_amplitudes`, because stage 2 on the
fixed ladder is on record as hitting its iteration cap with the gradient still above
tolerance; a number from here is not worth reading without `converged` beside it.

There are fewer curves to compress than on the fixed ladder — `1 + K_R + K_C` of them,
8 against 24 for the production ladder — so at equal `degree` the coefficient count is
proportionally smaller.
"""
function fit_tracking_basis(Model, tp::TrackingPoles, ω::AbstractVector,
                            Λ::AbstractMatrix, u::AbstractVector,
                            envelope::AbstractVector, h::AbstractVector,
                            Δ::AbstractVector, kT; degree::Integer, kwargs...)
    N_x = size(Λ, 2)
    length(h) == length(Δ) == N_x || throw(DimensionMismatch(
        "h ($(length(h))) and Δ ($(length(Δ))) must both cover the $N_x columns of Λ"))
    K_R, K_C = mode_counts(tp)
    n_c = 1 + K_R + K_C
    Ms = Vector{Matrix{Float64}}(undef, N_x)
    cvecs = Vector{Vector{Float64}}(undef, N_x)
    for i in 1:N_x
        ps = poleset(tp, h[i], Δ[i], kT)
        Ms[i] = _fill_design!(Matrix{Float64}(undef, length(ω), n_c), ps, ω)
        cvecs[i] = markovian_anchor_row(ps)
    end
    return _fit_basis_core(Model, Ms, cvecs, K_R, K_C, ω, Λ, u, envelope; degree, kwargs...)
end

"""
    fit_tracking(tl0::TiedLaw, tail, rate_floor, ω, Λ, h, Δ, kT, v; kwargs...)
        → (; law, poles, model, J, converged, iterations, evaluations)

Fit the **tied** pole law: Nelder–Mead over its 12 numbers rather than over the
`5K_C + 3K_R` a free [`TrackingPoles`](@ref) carries, with [`expand`](@ref) materialising
the geometric families at every step and [`tracking_solve`](@ref) eliminating the
amplitudes as before.

`tl0` fixes `K_C` and `K_R`; `tail` and `rate_floor` are the parts of the pole model that
are not searched over. Keyword arguments are those of the free-coefficient method.

This is what makes `K_C ≈ 8` reachable: the outer dimension no longer grows with the mode
count, so the simplex stays in the range where it works while the ladder gets as dense as
the data asks for. Returns the fitted `law` alongside the expanded `poles`, since the
twelve numbers are the object worth reading — they are the pole law itself.
"""
function fit_tracking(tl0::TiedLaw, tail::PoleSet, rate_floor, ω::AbstractVector,
                      Λ::AbstractMatrix, h::AbstractVector, Δ::AbstractVector, kT,
                      v::AbstractVector; ε = 1e-3, anchor_weight = 0,
                      iterations::Integer = 4000, g_tol = 1e-10, alg::Symbol = :nnls,
                      verbose::Bool = false)
    N_x = size(Λ, 2)
    length(h) == length(Δ) == length(v) == N_x || throw(DimensionMismatch(
        "h, Δ and v must each have $N_x entries"))
    σ = residual_scales(Λ; ε)
    K_C, K_R = tl0.K_C, tl0.K_R
    nev = Ref(0)
    function objective(θ)
        nev[] += 1
        tp = expand(TiedLaw(θ, K_C, K_R), tail, rate_floor)
        return tracking_solve(tp, ω, Λ, σ, h, Δ, kT, v; anchor_weight, alg).J
    end
    res = optimize(objective, parameters(tl0), NelderMead(),
                   Optim.Options(; iterations, g_tol, show_trace = verbose))
    law = TiedLaw(Optim.minimizer(res), K_C, K_R)
    poles = expand(law, tail, rate_floor)
    out = tracking_solve(poles, ω, Λ, σ, h, Δ, kT, v; anchor_weight, alg)
    return (; law, poles, model = out.model, J = out.J,
            converged = Optim.converged(res), iterations = Optim.iterations(res),
            evaluations = nev[])
end
