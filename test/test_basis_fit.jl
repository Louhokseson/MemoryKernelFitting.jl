# Stage 2 — the nonconvex compression. Three things are load-bearing here and nothing
# else in the suite covers them: the basis matrices are what they claim to be, the
# analytic gradient matches the objective it claims to differentiate, and the solver
# recovers data the model generated.

using Random
using LinearAlgebra: norm, ⋅

const _F = MemoryKernelFitting.Fitting
const _O = MemoryKernelFitting.Objective

@testset "stage 2 — basis fitting" begin

    @testset "basis_matrix" begin
        # Chebyshev against the closed forms, in t = 2s/π — the rescaling the model's
        # own constructor applies. A drift between the two would be invisible in the fit
        # and fatal in the readout, since the coefficients would mean a different curve.
        s = [0.0, 0.4, -1.2, π / 2]
        B = basis_matrix(ChebyshevAmplitudes, s, 4)
        t = (2 / π) .* s
        @test size(B) == (4, 5)
        @test B[:, 1] ≈ ones(4)
        @test B[:, 2] ≈ t
        @test B[:, 3] ≈ @. 2t^2 - 1
        @test B[:, 4] ≈ @. 4t^3 - 3t
        @test B[:, 5] ≈ @. 8t^4 - 8t^2 + 1
        @test all(abs.(B) .≤ 1 + 1e-12)              # |T_l| ≤ 1 on [-1, 1]

        # The matrix and the model must agree on what ĝ is, coefficient for coefficient.
        C = randn(5, 1 + 2 + 3)
        env = fill(1.0, 4)
        m = ChebyshevAmplitudes(C, s, env; K_R = 2, K_C = 3)
        for i in 1:4, (c, a) in enumerate(vcat(amplitudes(m, i)...))
            @test a ≈ (B[i, :] ⋅ C[:, c])^2
        end

        x = [0.8, 1.5, 3.0]
        V = basis_matrix(PolynomialAmplitudes, x, 3)
        @test V ≈ [x .^ 0 x .^ 1 x .^ 2 x .^ 3]

        @test_throws ArgumentError basis_matrix(ChebyshevAmplitudes, s, -1)
        @test_throws ArgumentError basis_matrix(ChebyshevAmplitudes, [2.0], 3)  # outside ±π/2
    end

    @testset "shape_envelope" begin
        Λ = [4.0 9.0; 1.0 2.0]
        @test shape_envelope(Λ) ≈ [2.0, 3.0]
        @test_throws ArgumentError shape_envelope([0.0 1.0; 1.0 1.0])
    end

    @testset "analytic gradient" begin
        # A finite-difference check is the only thing standing between a sign slip in the
        # chain rule and a fit that converges confidently to the wrong place.
        Random.seed!(11)
        K_R, K_C, d, N_x = 2, 3, 4, 25
        ps = split_ladder([(0.01, 0.5, K_C)], K_R, 1e-3, 0.05)
        ω = collect(range(1e-6, 2.0; length = 60))
        sw = collect(range(-1.4, 1.4; length = N_x))
        env = sqrt.(exp.(range(1.0, -2.0; length = N_x)))
        C = 0.3 .* randn(d + 1, 1 + K_R + K_C)
        truth = ChebyshevAmplitudes(C, sw, env; K_R, K_C)
        Λ = hcat([[spectral_density(w, ps, truth, i) for w in ω] for i in 1:N_x]...)
        σ = residual_scales(Λ; ε = 1e-3)
        v = fill(1 / N_x, N_x)
        M = design_matrix(ps, ω)
        B = basis_matrix(ChebyshevAmplitudes, sw, d)
        cvec = markovian_anchor_row(ps)

        C0 = C .+ 0.05 .* randn(size(C))
        for w in (0.0, 50.0)                          # with and without the anchor term
            g = similar(C0)
            _F._fg!(g, C0, M, B, Λ, σ, v, env, cvec, w)
            gfd = similar(C0)
            h = 1e-6
            for k in eachindex(C0)
                Cp = copy(C0); Cp[k] += h
                Cm = copy(C0); Cm[k] -= h
                gfd[k] = (_F._fg!(nothing, Cp, M, B, Λ, σ, v, env, cvec, w) -
                          _F._fg!(nothing, Cm, M, B, Λ, σ, v, env, cvec, w)) / 2h
            end
            @test norm(g - gfd) / norm(gfd) < 1e-7
        end
    end

    @testset "recovery of the generating model" begin
        # The counterpart of stage 1's recovery test. A one-signed shape function, because
        # that is what an initialisation built from √a can represent: `√a = |g|` throws the
        # sign away, and where the truth crosses zero the projection starts in the wrong
        # basin. See the sign-recovery note in basis_fit.jl.
        Random.seed!(3)
        K_R, K_C, d, N_x = 2, 3, 4, 40
        ps = split_ladder([(0.01, 0.5, K_C)], K_R, 1e-3, 0.05)
        ω = collect(range(1e-6, 2.0; length = 90))
        sw = collect(range(-1.4, 1.4; length = N_x))
        env = sqrt.(exp.(range(2.0, -3.0; length = N_x)))
        C = 0.15 .* randn(d + 1, 1 + K_R + K_C)
        C[1, :] .= 1.0                                 # keeps ĝ_c away from zero
        truth = ChebyshevAmplitudes(C, sw, env; K_R, K_C)
        @test all(>(0), basis_matrix(ChebyshevAmplitudes, sw, d) * C)   # one-signed indeed

        Λ = hcat([[spectral_density(w, ps, truth, i) for w in ω] for i in 1:N_x]...)
        grid = fit_amplitudes(ps, ω, Λ; ε = 1e-3)
        v = fill(1 / N_x, N_x)
        m, r = fit_basis_report(ChebyshevAmplitudes, ps, ω, Λ, sw, env;
                                degree = d, v = v, ε = 1e-3, init = grid)
        @test maximum(abs.(relative_errors(ps, m, ω, Λ))) < 1e-10
        @test mode_counts(m) == (K_R, K_C)
        @test n_configurations(m) == N_x
        @test fit_tag(m) == "stage2Cheby"

        # The envelope is data, not a parameter: it must not be counted as freedom.
        @test length(parameters(m)) == (d + 1) * (1 + K_R + K_C)

        # Starting from the answer stays at the answer — the objective has no drift that
        # would pull a correct fit away from itself.
        m2, r2 = fit_basis_report(ChebyshevAmplitudes, ps, ω, Λ, sw, env;
                                  degree = d, v = v, ε = 1e-3, init = C)
        @test r2.J_final ≤ r2.J_initial
        @test maximum(abs.(relative_errors(ps, m2, ω, Λ))) < 1e-10
    end

    @testset "argument checking" begin
        K_R, K_C, N_x = 1, 2, 8
        ps = split_ladder([(0.01, 0.5, K_C)], K_R, 1e-3, 0.05)
        ω = collect(range(1e-6, 1.0; length = 12))
        sw = collect(range(-1.0, 1.0; length = N_x))
        env = ones(N_x)
        Λ = fill(1.0, length(ω), N_x)
        @test_throws DimensionMismatch fit_amplitudes(ChebyshevAmplitudes, ps, ω[1:5], Λ,
                                                      sw, env; degree = 2)
        @test_throws DimensionMismatch fit_amplitudes(ChebyshevAmplitudes, ps, ω, Λ,
                                                      sw[1:4], env; degree = 2)
        @test_throws DimensionMismatch fit_amplitudes(ChebyshevAmplitudes, ps, ω, Λ,
                                                      sw, env; degree = 2, v = ones(3))
        @test_throws DimensionMismatch fit_amplitudes(ChebyshevAmplitudes, ps, ω, Λ,
                                                      sw, env; degree = 2,
                                                      init = zeros(9, 1 + K_R + K_C))
        @test_throws ArgumentError fit_amplitudes(ChebyshevAmplitudes, ps, ω, Λ,
                                                  sw, env; degree = 2, init = :nope)
    end

    @testset "sign recovery" begin
        # A curve that touches zero and comes back is the ambiguous case: stage 1 is an
        # NNLS, so exact zeros are the norm, not an edge case.
        y = [1.0, 0.5, 0.004, 0.6, 1.2]
        @test _F._sign_pattern(y) == [1.0, 1.0, 1.0, -1.0, -1.0]
        @test _F._sign_pattern([1.0, 0.9, 0.8]) == ones(3)      # no touchdown, no flip
        @test _F._sign_pattern(zeros(4)) == ones(4)             # a dead column
    end
end
