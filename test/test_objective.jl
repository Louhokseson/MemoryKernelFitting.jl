# The real files are ~123 MB each and data/ is gitignored, so the tests that touch one
# are guarded on isfile, as in test_spectral_data.jl.
et_file = joinpath(@__DIR__, "..", "data", "sims", "friction", "ErpenbeckThoss",
                   "Lambda_spectrum_Gamma_eV=0.1_Nx=257_T_K=300.0_domega_eV=0.001.h5")

# Synthesise data from the model *through Part 1's forward map*, never through the
# design matrix under test — that independence is what makes the exact-fit tests below
# mean anything.
function synthesise(ps, m, ω)
    return [spectral_density(w, ps, amplitudes(m, i)...)
            for w in ω, i in 1:n_configurations(m)]
end

# The roughness penalty as `Fitting` will assemble it: appended NNLS rows
# √(μ/Δxᵢ)(θᵢ₊₁ - θᵢ) against a zero right-hand side.
function difference_rows(Θ, x; μ)
    rows = Matrix{Float64}[]
    for i in 1:(size(Θ, 2) - 1)
        push!(rows, sqrt(μ / (x[i + 1] - x[i])) .* (Θ[:, i + 1] .- Θ[:, i])[:, :])
    end
    return rows
end

@testset "Objective" begin
    ps = PoleSet([1.0, 10.0], [0.5, 1.2], [3.0, 7.0])     # K_R = 2, K_C = 2
    ω = [0.0, 0.05, 0.5, 3.0, 7.0, 20.0]
    a₀, a, A = 0.5, [2.0, 0.75], [1.5, 0.4]
    θ = vcat(a₀, a, A)

    @testset "design_matrix" begin
        M = design_matrix(ps, ω)
        @test size(M) == (length(ω), 1 + 2 + 2)
        @test all(==(1), M[:, 1])                          # the white part is constant

        # THE LOAD-BEARING TEST. Everything downstream fits through M but is read out
        # through the poles and amplitudes; if these two forms of the model disagree,
        # the SDE does not realise what was fitted.
        @test M * θ == [spectral_density(w, ps, a₀, a, A) for w in ω]

        # Column c is basis function c, in the (a₀, a, A) order of `amplitudes`.
        for (k, λ) in enumerate(ps.λ)
            @test M[:, 1 + k] == lorentzian.(ω, λ)
        end
        for (l, (α, β)) in enumerate(zip(ps.α, ps.β))
            @test M[:, 3 + l] == mirrored_lorentzian.(ω, α, β)
        end

        # Degenerate ladders: pairs only (the production case), and reals only.
        pairs_only = PoleSet(Float64[], [0.5], [3.0])
        @test size(design_matrix(pairs_only, ω)) == (length(ω), 2)
        reals_only = PoleSet([1.0], Float64[], Float64[])
        @test size(design_matrix(reals_only, ω)) == (length(ω), 2)
        @test design_matrix(reals_only, ω)[:, 2] == lorentzian.(ω, 1.0)

        # An integer grid promotes rather than failing — `markovian_anchor_row` leans
        # on this when it asks for ω = 0.
        @test design_matrix(ps, [0]) ≈ design_matrix(ps, [0.0])
        @test eltype(design_matrix(ps, [0])) == Float64
    end

    @testset "markovian_anchor_row" begin
        c = markovian_anchor_row(ps)
        @test length(c) == 1 + 2 + 2
        # The §4 equation, written out longhand exactly once — here.
        @test c ≈ vcat(1.0, 1 ./ ps.λ, ps.α ./ (ps.α .^ 2 .+ ps.β .^ 2))
        # It is the model at ω = 0, which is why there is no second formula in src.
        @test c == vec(design_matrix(ps, [0.0]))
        @test sum(c .* θ) ≈ spectral_density(0.0, ps, a₀, a, A)

        # design/0003: at ω = 0 the mirror is not a 1% correction but a factor of two.
        # Dropping it would halve every pair's contribution to the anchor.
        for (l, (α, β)) in enumerate(zip(ps.α, ps.β))
            single_peak = (α / 2) / (α^2 + β^2)
            @test c[3 + l] ≈ 2 * single_peak
        end
    end

    @testset "residual_scales" begin
        Λ = [1.0 1e-6; 0.5 1e-7; 1e-9 1e-12]               # two columns, 9 decades apart
        σ = residual_scales(Λ; ε = 1e-3)
        @test size(σ) == size(Λ)
        @test all(>(0), σ)

        # Relative above the floor, floored below it — and the floor is *per column*,
        # so the weak column is scaled by its own peak, not by the strong one.
        @test σ[1, 1] == 1.0 && σ[2, 1] == 0.5             # |Λ| > εΛ^pk
        @test σ[3, 1] == 1e-3 * 1.0                        # floored at εΛ^pk
        @test σ[1, 2] == 1e-6 && σ[2, 2] == 1e-7
        @test σ[3, 2] == 1e-3 * 1e-6                       # its own peak, 9 decades down
        @test σ[3, 2] != σ[3, 1]

        # Where the floor is inactive the zero model's residual is exactly ∓1 — the
        # fact the objective's normalisation rests on.
        @test (0 - Λ[1, 2]) / σ[1, 2] == -1.0
        @test abs((0 - Λ[3, 1]) / σ[3, 1]) < 1

        # Fact 4 (Γ ≥ 0.2 has negative entries) needs no special case: only |Λ| enters.
        @test residual_scales([1.0 -0.4]')[2] == 0.4
        @test residual_scales(reshape([1.0, -0.4], :, 1))[2] == 0.4

        # A single column may be passed as a vector, and agrees with the matrix form.
        col = [1.0, 0.5, 1e-9]
        @test residual_scales(col) == vec(residual_scales(reshape(col, :, 1)))
        @test residual_scales(col) isa Vector

        @test_throws ArgumentError residual_scales(Λ; ε = 0)
        @test_throws ArgumentError residual_scales(Λ; ε = -1e-3)
        @test_throws ArgumentError residual_scales([0.0 0.0; -1.0 -1.0])   # peak ≤ 0
    end

    @testset "roughness_penalty" begin
        x = [0.0, 0.5, 1.5]                                # deliberately non-uniform
        flat = GridAmplitudes(fill(0.3, 3), fill(2.0, 2, 3), fill(1.0, 2, 3))
        @test roughness_penalty(flat, x) == 0              # constant in x ⇒ no penalty
        @test roughness_penalty(flat, x; μ = 1e6) == 0

        bumpy = GridAmplitudes([0.3, 0.4, 0.3], fill(2.0, 2, 3), fill(1.0, 2, 3))
        P = roughness_penalty(bumpy, x)
        @test P ≈ 0.1^2 / 0.5 + 0.1^2 / 1.0                # longhand
        @test roughness_penalty(bumpy, x; μ = 7) ≈ 7P      # linear in μ
        @test P > 0

        # a₀ is penalised too: `bumpy` differs from `flat` only in the white part, and
        # §4's λ ≫ ω_max degeneracy is exactly between a₀ and a fast mode.
        @test roughness_penalty(bumpy, x) > roughness_penalty(flat, x)

        # The M3 bridge: P is the squared 2-norm of the NNLS rows √(μ/Δx)(θᵢ₊₁ - θᵢ).
        # If these ever disagree, the penalty the outer loop reports is not the one the
        # inner solve applied.
        jagged = GridAmplitudes([0.3, 0.9, 0.2], [1.0 2.0 0.5; 0.1 0.7 0.3],
                                [4.0 1.0 2.0; 0.0 0.5 0.25])
        Θ = reduce(hcat, [vcat(amplitudes(jagged, i)...) for i in 1:3])
        for μ in (1.0, 0.03)
            @test roughness_penalty(jagged, x; μ) ≈
                  sum(sum(abs2, r) for r in difference_rows(Θ, x; μ))
        end

        # 1/Δx penalises the slope, not the increment: the same jump costs 10× more
        # across a cell 10× finer. This is what keeps the 214 packed columns from
        # wandering freely while the coarse tail is pinned.
        step = GridAmplitudes([0.0, 1.0], zeros(0, 2), zeros(0, 2))
        @test roughness_penalty(step, [0.0, 1.0]) ≈ 1.0
        @test roughness_penalty(step, [0.0, 0.1]) ≈ 10.0

        # A single column — the §7 step-2 per-x diagnostic — has no intervals.
        @test roughness_penalty(GridAmplitudes(1, 2, 2), [1.0]) == 0

        @test_throws DimensionMismatch roughness_penalty(flat, [0.0, 1.0])
        @test_throws ArgumentError roughness_penalty(flat, [0.0, 1.0, 0.5])  # descending
        @test_throws ArgumentError roughness_penalty(flat, [0.0, 0.5, 0.5])  # repeated
    end

    @testset "objective_value" begin
        # A graded grid with the shape of the real one, as in test_spectral_data.jl.
        x = vcat(0.0:0.5:1.0, 1.45:0.01:1.55, 2.0:0.5:3.0)
        N_x = length(x)
        v = trapezoid_weights(x)
        ωb = exp.(range(log(0.01), log(30.0); length = 40))
        nB = length(ωb)

        # Column amplitudes spanning several decades, so that any accidental absolute
        # weighting shows up immediately.
        scale = 10.0 .^ range(-4, 2; length = N_x)
        truth = GridAmplitudes(0.5 .* scale, [2.0; 0.75] .* scale', [1.5; 0.4] .* scale')
        Y = synthesise(ps, truth, ωb)
        σ = residual_scales(Y; ε = 1e-12)                  # floor deliberately inactive
        @test all(σ .== abs.(Y))

        # An exact fit is exactly zero.
        @test objective_value(ps, truth, ωb, Y, v, σ) ≈ 0 atol = 1e-20

        # A uniform 1% error costs 1e-4 per point per unit weight — *exactly*, and
        # independently of the 6 decades between columns. This is the relative
        # weighting of §3 in closed form.
        off = GridAmplitudes(1.01 .* truth.a₀, 1.01 .* truth.a, 1.01 .* truth.A)
        @test objective_value(ps, off, ωb, Y, v, σ) ≈ 1e-4 * nB * sum(v) rtol = 1e-12

        # The zero model: every unfloored residual is -1, so J₀ = |ℬ| Σvᵢ exactly.
        zero_model = GridAmplitudes(N_x, 2, 2)
        @test objective_value(ps, zero_model, ωb, Y, v, σ) ≈ nB * sum(v) rtol = 1e-12

        # THE GRADING TRAP (design/0001), as a behavioural test. Perturb only the
        # densely sampled columns: with trapezoid weights they carry under 5% of the
        # objective, with vᵢ ≡ 1 they carry more than half of it, purely by outnumbering
        # the rest.
        dense = 1.45 .< x .< 1.55
        @test count(dense) > N_x / 2
        bump = 1.0 .+ 0.01 .* dense'
        patchy = GridAmplitudes(vec(bump) .* truth.a₀, bump .* truth.a, bump .* truth.A)
        J_all = objective_value(ps, off, ωb, Y, v, σ)
        @test objective_value(ps, patchy, ωb, Y, v, σ) ≈ 1e-4 * nB * sum(v[dense]) rtol = 1e-12
        @test objective_value(ps, patchy, ωb, Y, v, σ) < 0.05 * J_all
        ones_v = ones(N_x)
        @test objective_value(ps, patchy, ωb, Y, ones_v, σ) >
              0.5 * objective_value(ps, off, ωb, Y, ones_v, σ)

        # σ ≡ 1 recovers plain absolute least squares, so the scales are genuinely the
        # only thing making the fit relative.
        R = [spectral_density(w, ps, amplitudes(off, i)...) for w in ωb, i in 1:N_x]
        @test objective_value(ps, off, ωb, Y, v, ones(size(Y))) ≈
              sum(v[i] * sum(abs2, R[:, i] .- Y[:, i]) for i in 1:N_x)

        # The roughness term is additive and off by default.
        @test objective_value(ps, truth, ωb, Y, v, σ; μ = 0.5, x) ≈
              objective_value(ps, truth, ωb, Y, v, σ) + roughness_penalty(truth, x; μ = 0.5)
        @test objective_value(ps, truth, ωb, Y, v, σ) ==
              objective_value(ps, truth, ωb, Y, v, σ; μ = 0)

        # A slope penalty without a grid is a silent no-op waiting to happen.
        @test_throws ArgumentError objective_value(ps, truth, ωb, Y, v, σ; μ = 0.5)

        @test_throws DimensionMismatch objective_value(ps, truth, ωb[1:5], Y, v, σ)
        @test_throws DimensionMismatch objective_value(ps, truth, ωb, Y, v[1:3], σ)
        @test_throws DimensionMismatch objective_value(ps, truth, ωb, Y, v, σ[1:5, :])
        # A model whose mode counts do not match the ladder is caught, not silently
        # broadcast into the wrong columns.
        @test_throws DimensionMismatch objective_value(ps, GridAmplitudes(N_x, 1, 2),
                                                       ωb, Y, v, σ)

        # A single column is a legal problem — it is the §7 step-2 diagnostic.
        one_col = GridAmplitudes([0.5], [2.0; 0.75][:, :], [1.5; 0.4][:, :])
        Y1 = synthesise(ps, one_col, ωb)
        @test objective_value(ps, one_col, ωb, Y1, [1.0], residual_scales(Y1)) ≈ 0 atol = 1e-20
    end

    if isfile(et_file)
        @testset "on the Γ = 0.1 spectrum" begin
            s = load_lambda_spectrum(et_file)
            idx = subsample_omega(s.ω)
            Y = s.Λ[idx, :]
            v = trapezoid_weights(s.x)
            σ = residual_scales(Y)

            @test length(idx) == 561
            @test size(σ) == (561, 254)
            @test all(>(0), σ)
            @test sum(v) ≈ 4.2

            # 30% of the retained points sit below 1e-3 of their column peak — the ω⁻³
            # tail the floor exists to stop from dominating. Some columns are floored
            # nowhere at all, so the floor is not quietly flattening the whole fit.
            floored = σ .> abs.(Y)
            @test count(floored) / length(σ) ≈ 0.302 rtol = 1e-2
            @test minimum(sum(floored; dims = 1)) == 0

            # The production ladder of design/0003, with M1's correction: pair_budget
            # counts *intervals*, and a ladder spanning them needs one more rung, so
            # the ladder is 30 pairs and m = 60 rather than 29 and 58.
            @test pair_budget(1.8e-3, 0.345) == 29
            K_C = pair_budget(1.8e-3, 0.345) + 1
            ladder = log_ladder(1.8e-3, 0.345, K_C)
            @test auxiliary_dimension(ladder) == 60
            @test length(markovian_anchor_row(ladder)) == 1 + K_C

            # The zero model, against the bound of the docstring. J₀ ≤ |ℬ| Σvᵢ holds
            # exactly, and lands 5% under it because ~30% of the points are floored.
            zero_model = GridAmplitudes(length(s.x), 0, K_C)
            J₀ = objective_value(ladder, zero_model, s.ω[idx], Y, v, σ)
            bound = length(idx) * sum(v)
            @test J₀ ≈ 2227.15 rtol = 1e-4
            @test J₀ < bound
            @test J₀ / bound ≈ 0.945 rtol = 1e-3

            # What the relative weighting buys, measured: the column peaks span 11
            # decades, yet every column contributes an O(1) normalised error to the
            # zero model. Under absolute weighting the ratio would span 11 decades too.
            per_column = vec(sum((Y ./ σ) .^ 2; dims = 1)) ./ length(idx)
            peaks = vec(maximum(Y; dims = 1))
            @test log10(maximum(peaks) / minimum(peaks)) > 10
            @test all(r -> 0.4 < r ≤ 1.0, per_column)

            # The convenience method assembles exactly the same pieces.
            @test objective_value(ladder, zero_model, s) == J₀
            @test objective_value(ladder, zero_model, s; idx) == J₀

            @testset "configuration_weights" begin
                vx = trapezoid_weights(s.x)

                # Normalised for every θ, and θ = 0 is the trapezoid rule up to that scale.
                for θ in 0:0.125:1
                    v = configuration_weights(s; θ)
                    @test sum(v) ≈ 1
                    @test all(>(0), v)
                    @test length(v) == length(s.x)
                end
                @test configuration_weights(s; θ = 0.0) ≈ vx ./ sum(vx)

                # Affine in θ, since it is a blend of two fixed normalised measures.
                v0, v1 = configuration_weights(s; θ = 0.0), configuration_weights(s; θ = 1.0)
                @test configuration_weights(s; θ = 0.25) ≈ 0.75 .* v0 .+ 0.25 .* v1

                @test_throws ArgumentError configuration_weights(s; θ = -0.1)
                @test_throws ArgumentError configuration_weights(s; θ = 1.5)

                # The point of the warp: it moves weight onto the crossing, which the plain
                # x measure starves because the grid is graded towards it. The crossing
                # column carries 0.012% of the trapezoid weight and 0.89% of the warp's.
                i₀ = argmin(abs.(s.x .- s.x_resonance))
                @test v0[i₀] < 2e-4        # 0.0119% of the total
                @test v1[i₀] > 50 * v0[i₀]
                near = abs.(s.x .- s.x_resonance) .< 0.15
                @test sum(v0[near]) < 0.1          # 8.3% under the plain x measure
                @test sum(v1[near]) > 0.4          # 48.6% under the warp

                # …but the pure warp abandons the |h| peak, which is why the default blends.
                peak = 1.0 .< s.x .< 1.8
                @test sum(v1[peak]) < 0.01         # 0.35% — β_max lives here
                @test sum(configuration_weights(s; θ = 0.5)[peak]) > 0.05

                # It is a ladder-selection instrument: v never enters the decoupled inner
                # problem, so θ cannot change a fit at fixed ladder.
                fit = fit_amplitudes(ladder, s.ω[idx], s.Λ[idx, :]; ε = 1e-3)
                σ_ = residual_scales(s.Λ[idx, :]; ε = 1e-3)
                J_a = objective_value(ladder, fit, s.ω[idx], s.Λ[idx, :],
                                      configuration_weights(s; θ = 0.0), σ_)
                J_b = objective_value(ladder, fit, s.ω[idx], s.Λ[idx, :],
                                      configuration_weights(s; θ = 0.5), σ_)
                @test J_a != J_b                   # the ranking number does change
                @test parameters(fit) == parameters(
                    fit_amplitudes(ladder, s.ω[idx], s.Λ[idx, :]; ε = 1e-3))
            end
        end
    else
        @info "Skipping Objective tests on real data: $(et_file) not found " *
              "(data/ is gitignored; each file is ~123 MB)."
    end
end
