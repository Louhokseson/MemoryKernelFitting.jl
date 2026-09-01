# Stage 2: the compression. `docs/design/0002-amplitude-representation.md` step 2 —
# having *seen* the amplitude curve stage 1 produced, fit a smooth representation to it
# and re-optimise the coefficients against the data.
#
# Two properties separate this file from `nnls.jl`, and both follow from the same
# substitution `a_c = g_c²`:
#
#   - **It is nonconvex.** `g` is linear in `C`, so `a` is quadratic in `C` and the
#     residual is quartic. There is an initialisation, there are local minima, and there
#     is a convergence criterion — none of which stage 1 has. The initialisation is not a
#     guess: it is the stage-1 solution, which is the entire reason the stages are
#     ordered this way.
#   - **The columns are coupled.** Every node shares the same `C`, so `v` no longer
#     cancels out of the inner problem as it does at `μ = 0` in stage 1. It is a genuine
#     fitting weight here, not only a ladder-selection instrument.
#
# The gradient is analytic. With `G = BC`, `a_{ic} = e_i² G_{ic}²` and
# `Λ_θ = M aᵀ`, the chain is three matrix products and no autodiff:
#
#     S = R ./ σ,  P = Sᵀ M,  ∂J/∂G = 4 v e² G ⊙ P,  ∂J/∂C = Bᵀ (∂J/∂G)
#
# which matters because a finite-difference gradient over `(d+1)(1 + K_R + K_C)`
# parameters would cost 264 objective evaluations per step.

"""
    basis_matrix(::Type{ChebyshevAmplitudes}, s, degree) → B
    basis_matrix(::Type{PolynomialAmplitudes}, x, degree) → B

The `[N_x, degree+1]` matrix of basis functions evaluated at the nodes, so that
`B * C` is the shape function `ĝ_c` of `docs/design/0002` at every node.

For [`ChebyshevAmplitudes`](@ref), `s` is the raw [`warp_coordinate`](@ref) — *not*
`t` — and the `t = 2s/π` rescaling is applied here exactly as the constructor applies
it, so the matrix and the model can never disagree about what coordinate they are in.
Built by the Chebyshev recurrence `T_{l+1} = 2t T_l - T_{l-1}`.

For [`PolynomialAmplitudes`](@ref), `x` is the raw configuration coordinate in Å and
`B` is the Vandermonde matrix. This is the control basis of `docs/design/0002`, kept so
that its predicted ringing is a measured result rather than an asserted one.
"""
function basis_matrix end

function basis_matrix(::Type{ChebyshevAmplitudes}, s::AbstractVector, degree::Integer)
    degree ≥ 0 || throw(ArgumentError("degree must be non-negative, got $degree"))
    all(σ -> abs(σ) ≤ π / 2 + 1e-12, s) || throw(ArgumentError(
        "warped coordinates must lie in [-π/2, π/2]; pass warp_coordinate.(h, Δ, kT), not raw x"))
    t = (2 / π) .* s                       # the constructor's own rescaling
    B = Matrix{Float64}(undef, length(t), degree + 1)
    @views fill!(B[:, 1], 1.0)
    degree ≥ 1 && (@views B[:, 2] .= t)
    for l in 2:degree
        @views @. B[:, l + 1] = 2 * t * B[:, l] - B[:, l - 1]
    end
    return B
end

function basis_matrix(::Type{PolynomialAmplitudes}, x::AbstractVector, degree::Integer)
    degree ≥ 0 || throw(ArgumentError("degree must be non-negative, got $degree"))
    B = Matrix{Float64}(undef, length(x), degree + 1)
    @views fill!(B[:, 1], 1.0)
    for l in 1:degree
        @views @. B[:, l + 1] = B[:, l] * x
    end
    return B
end

"""
    shape_envelope(Λ) → e

The envelope `e_i = √(Λ(x_i, 0))` of `docs/design/0002` §"the warp", read off the data
rather than fitted: `Λ` is the retained band as `[|ℬ|, N_x]` and row 1 is `ω ≈ 0`,
which [`subsample_omega`](@ref) always keeps.

Dividing it out is what leaves an `O(1)` shape function: `a_c` spans 10–17 decades over
this grid and `g_c = √a_c` still spans 5–9, so no basis expansion of `g_c` itself would
be well conditioned. It costs no parameters — it is data.
"""
function shape_envelope(Λ::AbstractMatrix)
    all(>(0), view(Λ, 1, :)) || throw(ArgumentError(
        "the envelope √(Λ(x,0)) needs a positive zero-frequency row; column " *
        "$(findfirst(≤(0), view(Λ, 1, :))) is not"))
    return sqrt.(view(Λ, 1, :))
end

# Sign recovery for the initialisation.
#
# `docs/design/0002` says sign changes of `g_c` are harmless, and for the *model* they
# are: only `g²` is observable. For the *initialisation* they are not, because stage 1
# hands back `a_c`, and `√a_c = |g_c|` has thrown the sign away. Where the underlying
# `g_c` crosses zero, `|g_c|` has a kink, and projecting a kink onto a low-degree
# polynomial lands the quartic in the wrong basin — measured on a synthetic truth with
# 1–2 crossings per column, the projection starts at `J = 12.3` and L-BFGS converges to
# `J = 0.099` when the truth is at `4e-30`.
#
# This is not a synthetic-only concern: stage 1 is an NNLS, so it produces *exact* zeros,
# and a column that touches zero and comes back is precisely the ambiguous case.
#
# A touchdown — a local minimum of `|g_c|` pressed against zero — is where a crossing
# plausibly hides. Flipping the sign after each one turns the kinked `|g_c|` back into a
# candidate smooth `g_c`. It is a heuristic and can be wrong, so it is not trusted: both
# candidates are scored on the actual objective and the better one starts the optimiser,
# which makes it safe (never worse than the all-positive projection) without making it
# sufficient — on the synthetic truth above neither candidate escapes the bad basin.
#
# **What this does and does not bound.** With a one-signed generating shape the fit
# recovers the truth to 5e-15, so the solver is exact where the initialisation can reach.
# Where the optimum genuinely needs `g_c` to change sign, the initialisation is the
# limiting factor, not the optimiser. On real data there is no true `C` to miss — the
# target is a non-negative curve `√a_c(x)` that a one-signed expansion represents fine —
# so this bites only if a fitted amplitude must touch zero and come back.
function _sign_pattern(y::AbstractVector)
    n = length(y)
    σ = ones(float(eltype(y)), n)
    ymax = maximum(y)
    ymax > 0 || return σ                            # a dead column: every sign is equivalent
    τ = 1e-2 * ymax
    cur = one(eltype(σ))
    @inbounds for i in 1:n
        σ[i] = cur
        if 1 < i < n && y[i] ≤ y[i - 1] && y[i] ≤ y[i + 1] && y[i] < τ
            cur = -cur                              # takes effect from i+1 on
        end
    end
    return σ
end

# Least-squares projection of the stage-1 curve onto the basis, as both sign candidates.
# This is the *initialisation*, not the answer: `docs/design/0002` step 2 fits the grid
# solution and then re-optimises against the data, because stage 1's columns were solved
# independently and nothing forced the curve it traced to be smooth.
function _initial_coefficients(B::AbstractMatrix, Θ::AbstractMatrix, e::AbstractVector)
    Y = permutedims(sqrt.(max.(Θ, 0))) ./ e        # [N_x, n_c], non-negative
    Ys = similar(Y)
    for c in axes(Y, 2)
        @views Ys[:, c] .= _sign_pattern(view(Y, :, c)) .* view(Y, :, c)
    end
    return B \ Y, B \ Ys
end

# The design matrix is shared across columns when the ladder is fixed, and per column when
# the poles track the configuration (`varpro.jl`). Only two lines of `_fg!` depend on which,
# so they dispatch instead of the whole gradient being written twice — the chain through
# `a = e²G²` and `G = BC` is identical either way, and there is exactly one copy of it.
# The `AbstractMatrix` methods are what the fixed ladder has always used, unchanged and
# still one BLAS-3 call.
_model_values(M::AbstractMatrix, a) = M * permutedims(a)
function _model_values(Ms::AbstractVector{<:AbstractMatrix}, a)
    Y = Matrix{eltype(a)}(undef, size(first(Ms), 1), size(a, 1))
    Threads.@threads for i in axes(a, 1)
        @inbounds mul!(view(Y, :, i), Ms[i], view(a, i, :))
    end
    return Y
end

_project(S, M::AbstractMatrix) = transpose(S) * M
function _project(S, Ms::AbstractVector{<:AbstractMatrix})
    P = Matrix{eltype(S)}(undef, size(S, 2), size(first(Ms), 2))
    # Each column writes its own slice of P and reads its own slice of S, so no buffer is
    # shared and the default scheduler is fine here — unlike `tracking_solve`, which needs
    # `:static` because it indexes a per-thread scratch matrix.
    Threads.@threads for i in axes(S, 2)
        @inbounds mul!(view(P, i, :), transpose(Ms[i]), view(S, :, i))
    end
    return P
end

@inline _col_anchor(c::AbstractVector{<:Real}, ::Integer) = c
@inline _col_anchor(cs::AbstractVector{<:AbstractVector}, i::Integer) = cs[i]

# J and ∇J at coefficients C. Writes the gradient into ∇ when it is not `nothing`.
# `M` is either one design matrix or one per column; `cvec` likewise one anchor row or one
# per column.
function _fg!(∇, C, M, B, Λ, σ, v, e, cvec, w)
    G  = B * C                                      # [N_x, n_c]  shape ĝ_c(u_i)
    e2 = e .^ 2
    a  = e2 .* (G .^ 2)                             # [N_x, n_c]  amplitudes, ≥ 0 by construction
    R  = (_model_values(M, a) .- Λ) ./ σ            # [|ℬ|, N_x]  scaled residual

    J = zero(eltype(R))
    @inbounds for i in axes(R, 2)
        c = zero(J)
        for j in axes(R, 1)
            c += R[j, i]^2
        end
        J += v[i] * c
    end

    P = _project(R ./ σ, M)                         # [N_x, n_c]
    Q = 2 .* v .* P                                 # ∂J/∂a

    # The §4 Markovian anchor, as the same weighted row stage 1 appends — but weighted by
    # vᵢ as well, since here the columns share C and vᵢ no longer cancels.
    if !iszero(w)
        @inbounds for i in axes(Λ, 2)
            ci = _col_anchor(cvec, i)
            wi = w / σ[1, i]
            ra = wi * (dot(ci, view(a, i, :)) - Λ[1, i])
            J += v[i] * ra^2
            @views Q[i, :] .+= (2 * v[i] * ra * wi) .* ci
        end
    end

    if ∇ !== nothing
        mul!(∇, transpose(B), Q .* (2 .* e2 .* G))  # chain through a = e²G², G = BC
    end
    return J
end

for (Model, coord) in ((:ChebyshevAmplitudes, "warp_coordinate.(s.h, s.Δ, s.kT)"),
                       (:PolynomialAmplitudes, "the raw x grid in Å"))
    @eval begin
        """
            fit_amplitudes(::Type{$($Model)}, ps, ω, Λ, u, envelope; degree, kwargs...) → $($Model)

        Stage 2 of `docs/design/0002-amplitude-representation.md`: fit the coefficients of
        the shape function `ĝ_c` so that `a_c(x) = (envelope · ĝ_c(u(x)))²` reproduces the
        data, with the pole ladder `ps` held fixed exactly as in stage 1.

        `u` is the evaluation coordinate — $($coord) — and `envelope` is
        [`shape_envelope`](@ref). `degree` is the polynomial degree, so the model carries
        `(degree+1)(1 + K_R + K_C)` free coefficients against stage 1's
        `N_x (1 + K_R + K_C)`.

        Unlike stage 1 this is **nonconvex** — the residual is quartic in the coefficients
        — so it needs a starting point, and takes one from `init`. Pass a
        [`GridAmplitudes`](@ref) (the usual case: the stage-1 fit) and its curve is
        projected onto the basis by least squares first; pass a coefficient matrix to
        resume; pass nothing and the projection is skipped, which is only sensible for
        testing. **`init` is not a convenience.** Choosing a basis before seeing the curve
        it has to represent is what `docs/design/0002` argues against, and starting the
        quartic from an arbitrary point is the concrete cost of doing so.

        Keyword arguments:

          - `degree` — required.
          - `v` — configuration quadrature weights. Required in substance: at `μ = 0`
            stage 1's columns decouple and `v` cancels, but here every node shares `C`, so
            `v` is a genuine fitting weight. Defaults to uniform only so the function is
            usable on a synthetic grid; on real data pass
            [`configuration_weights`](@ref).
          - `ε` — the [`residual_scales`](@ref) floor, matched to the stage-1 fit.
          - `anchor_weight` — the §4 Markovian anchor, as in [`fit_column`](@ref).
          - `iterations`, `g_tol`, `show_trace` — passed to Optim's L-BFGS. The default
            budget is large on purpose: the quartic is badly conditioned on real data and
            L-BFGS is still descending at 1000 iterations, where it reports `J/J₀ = 0.45`
            against the `0.23` it reaches by 20000. Always check `report.converged`; a
            non-converged fit is a **bound**, not a limit.

        Returns the fitted model. The optimiser's own report is available from
        [`fit_basis_report`](@ref) if the caller wants convergence detail.
        """
        function fit_amplitudes(::Type{$Model}, ps::PoleSet, ω::AbstractVector,
                                Λ::AbstractMatrix, u::AbstractVector,
                                envelope::AbstractVector; degree::Integer,
                                v = nothing, ε = 1e-3, anchor_weight = 0,
                                init = nothing, iterations::Integer = 20_000,
                                g_tol = 1e-10, show_trace::Bool = false)
            m, _ = _fit_basis($Model, ps, ω, Λ, u, envelope; degree, v, ε,
                              anchor_weight, init, iterations, g_tol, show_trace)
            return m
        end
    end
end

"""
    fit_basis_report(Model, ps, ω, Λ, u, envelope; degree, kwargs...) → (model, report)

As `fit_amplitudes(Model, …)`, but also returns the optimiser's report: a NamedTuple of
`(; converged, iterations, g_residual, J_initial, J_final, C_initial)`.

`J_initial` is the objective at the least-squares projection of the stage-1 curve, before
any re-optimisation. The gap between it and `J_final` is the measurement that says whether
step 2 of `docs/design/0002` — *re-optimise the coefficients* — earns its place, or whether
projecting the stage-1 curve would have been enough on its own.
"""
fit_basis_report(Model, ps::PoleSet, ω::AbstractVector, Λ::AbstractMatrix,
                 u::AbstractVector, envelope::AbstractVector; kwargs...) =
    _fit_basis(Model, ps, ω, Λ, u, envelope; kwargs...)

function _fit_basis(Model, ps::PoleSet, ω::AbstractVector, Λ::AbstractMatrix,
                    u::AbstractVector, envelope::AbstractVector; degree::Integer,
                    v = nothing, ε = 1e-3, anchor_weight = 0, init = nothing,
                    iterations::Integer = 20_000, g_tol = 1e-10, show_trace::Bool = false)
    size(Λ, 1) == length(ω) || throw(DimensionMismatch(
        "Λ has $(size(Λ, 1)) rows for $(length(ω)) frequencies"))
    N_x = size(Λ, 2)
    length(u) == length(envelope) == N_x || throw(DimensionMismatch(
        "u ($(length(u))) and envelope ($(length(envelope))) must both cover the " *
        "$N_x configurations of Λ"))
    return _fit_basis_core(Model, design_matrix(ps, ω), markovian_anchor_row(ps),
                           length(ps.λ), length(ps.α), ω, Λ, u, envelope;
                           degree, v, ε, anchor_weight, init, iterations, g_tol, show_trace)
end

# The stage-2 solve itself, over whatever design the pole model produced. `Ms`/`cvecs` are
# a single matrix and row for a fixed ladder, or one per column for tracking poles; nothing
# below distinguishes the two cases.
function _fit_basis_core(Model, Ms, cvec, K_R::Integer, K_C::Integer,
                         ω::AbstractVector, Λ::AbstractMatrix, u::AbstractVector,
                         envelope::AbstractVector; degree::Integer,
                         v = nothing, ε = 1e-3, anchor_weight = 0, init = nothing,
                         iterations::Integer = 20_000, g_tol = 1e-10,
                         show_trace::Bool = false)
    N_x = size(Λ, 2)
    n_c = 1 + K_R + K_C

    weights = v === nothing ? fill(1 / N_x, N_x) : v
    length(weights) == N_x || throw(DimensionMismatch(
        "there are $(length(weights)) quadrature weights for $N_x configurations"))

    M = Ms
    σ = residual_scales(Λ; ε)
    B = basis_matrix(Model, u, degree)

    C0 = if init === nothing
        zeros(degree + 1, n_c)
    elseif init isa AbstractMatrix
        size(init) == (degree + 1, n_c) || throw(DimensionMismatch(
            "init coefficients are $(size(init)), expected $((degree + 1, n_c))"))
        collect(float.(init))
    elseif init isa AmplitudeModel
        n_configurations(init) == N_x || throw(DimensionMismatch(
            "init model carries $(n_configurations(init)) configurations, Λ has $N_x"))
        mode_counts(init) == (K_R, K_C) || throw(DimensionMismatch(
            "init model carries $(mode_counts(init)) modes, the ladder has $((K_R, K_C))"))
        Cp, Cs = _initial_coefficients(B, _amplitude_columns(init), envelope)
        score(C) = _fg!(nothing, C, M, B, Λ, σ, weights, envelope, cvec, anchor_weight)
        score(Cs) < score(Cp) ? Cs : Cp            # the heuristic is scored, not trusted
    else
        throw(ArgumentError("init must be an AmplitudeModel, a coefficient matrix, or nothing"))
    end

    # Optim's two-function form rather than a fused fg!: `only_fg!` is gone in Optim v2,
    # and the extra forward pass per gradient is nothing next to the L-BFGS line search.
    f(C)     = _fg!(nothing, C, M, B, Λ, σ, weights, envelope, cvec, anchor_weight)
    g!(G, C) = (_fg!(G, C, M, B, Λ, σ, weights, envelope, cvec, anchor_weight); G)
    J0 = f(C0)

    res = optimize(f, g!, C0, LBFGS(),
                   Optim.Options(; iterations, g_tol, show_trace))
    C = Optim.minimizer(res)

    model = Model(C, u, envelope; K_R, K_C)
    report = (; converged   = Optim.converged(res),
                iterations  = Optim.iterations(res),
                g_residual  = Optim.g_residual(res),
                J_initial   = J0,
                J_final     = Optim.minimum(res),
                C_initial   = C0)
    return model, report
end
