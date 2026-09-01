# Track B — the regime-(B) tracking pole model and the VARPRO search over it.
#
# Two things carry the weight here. The `_fill_design!` equality is a no-drift assertion:
# `varpro.jl` rebuilds the design matrix in place for speed, and the moment that loop
# disagrees with `design_matrix` the fit would be optimising a different model from the
# one the readout uses. And the degenerate-`h` case ties the whole new path back to the
# old one — with `h ≡ 0` and `Δ` constant every column's ladder is the same, so
# `tracking_solve` must reproduce `fit_amplitudes` on that single `PoleSet` exactly.

using MemoryKernelFitting.Fitting: tracking_solve, _fill_design!

@testset "TrackingPoles" begin
    kT = boltzmann_energy(300.0)
    tail = PoleSet(Float64[], [0.09, 0.44], [0.441, 2.205])     # the [12, 60] eV band
    floor_ = 1.84e-4                                            # 5·dω at Γ = 0.1

    @testset "construction and shape" begin
        tp = tracking_seed(2, 1, tail, floor_)
        @test n_tracking(tp) == (1, 2, 2)
        @test mode_counts(tp) == (1, 4)
        @test auxiliary_dimension(tp) == 1 + 2 * 4
        @test length(parameters(tp)) == 5 * 2 + 3 * 1

        # The seed is §6's measured values, not a generic guess.
        r = tracking_report(tp)
        @test all(≈(0.2), r.u)
        @test all(≈(1.0), r.z)
        # c and d now span the bands measured off the fitted fixed ladder, not a guess.
        @test collect(extrema(r.c)) ≈ [0.51, 6.26]
        @test collect(extrema(hartree_to_eV.(r.d))) ≈ [0.049, 0.401] rtol = 1e-3
        # `w` is spread, not repeated. Identical coefficients are an exactly degenerate
        # start, and at the crossing `w` is the *only* thing that separates the modes —
        # `β_j = c_j·0 = 0` there for every pair, whatever `c` is.
        @test allunique(r.w) && collect(extrema(r.w)) ≈ [1 / 3, 3.0]

        @test_throws DimensionMismatch TrackingPoles([1.0], [0.01], [0.2, 0.2], [1.0],
                                                     [1.0], Float64[], Float64[],
                                                     Float64[], tail, floor_)
        @test_throws DimensionMismatch TrackingPoles([1.0], [0.01], [0.2], [1.0], [1.0],
                                                     [0.0], [1.0], Float64[],
                                                     tail, floor_)
        # Real rates in the fixed tail are rejected: they exist for the crossing, which
        # is exactly the feature that tracks.
        @test_throws ArgumentError TrackingPoles([1.0], [0.01], [0.2], [1.0], [1.0],
                                                 Float64[], Float64[], Float64[],
                                                 PoleSet([0.01], [0.09], [0.441]), floor_)
        @test_throws ArgumentError tracking_seed(1, 0, tail, -1.0)
        @test_throws DimensionMismatch parameters!(tp, zeros(3))
    end

    @testset "poleset" begin
        tp = tracking_seed(2, 1, tail, floor_)
        Δx = 0.05 / 27.211386245988                              # Δ = 0.05 eV in Ha

        # At the Fermi crossing h = 0, so β collapses to zero and the width falls back on
        # w·Δ + z·k_BT. That is the whole content of the K_R prediction: the pair becomes
        # the broad zero-centred shape the fixed ladder needed real rates to build.
        # One pair: the width at the crossing is exactly Δ + k_BT, the measured
        # half-width there — the whole content of the K_R prediction.
        ps1 = poleset(tracking_seed(1, 0, tail, floor_), 0.0, Δx, kT)
        @test ps1.α[1] ≈ Δx + kT
        # β no longer collapses to zero at the crossing: that degeneracy is what the
        # offset `d` exists to break, and it lands in the band the fixed ladder's live
        # pairs occupy there.
        @test 0 < hartree_to_eV(ps1.β[1]) < 0.5
        @test hartree_to_eV(Δx + kT) ≈ 0.0759 atol = 1e-3        # the measured 0.077 eV

        # Several pairs: the widths must *differ* there and bracket Δ + k_BT, or the
        # ladder is degenerate at the one column that carries the dataset's maximum.
        ps0 = poleset(tp, 0.0, Δx, kT)
        @test all(>(0), ps0.β[1:2]) && allunique(ps0.β[1:2])
        @test allunique(ps0.α[1:2])
        @test minimum(ps0.α[1:2]) < Δx + kT < maximum(ps0.α[1:2])

        # Ordering: tracking pairs first, then the tail, unchanged.
        @test ps0.α[3:4] == tail.α
        @test ps0.β[3:4] == tail.β

        # Only |h| enters, and β is linear in it.
        h = 0.3
        @test poleset(tp, h, Δx, kT).β ≈ poleset(tp, -h, Δx, kT).β
        # β is affine in |h|, not linear — equal steps in |h| give equal steps in β, and
        # the intercept is the offset itself.
        β(x) = poleset(tp, x, Δx, kT).β[1]
        @test β(2h) - β(h) ≈ β(h) - β(0.0)
        @test poleset(tp, 0.0, Δx, kT).β[1:2] ≈ abs.(tp.d)

        # The §4 floor binds where the linear form would go below it, and every rate the
        # model can produce is a legal PoleSet rate (α > 0) for any coefficients.
        flat = TrackingPoles([1.0], [0.0], [0.0], [0.0], [0.0], [-5.0], [-5.0], [-5.0],
                             tail, floor_)
        psf = poleset(flat, 0.4, Δx, kT)
        @test psf.α[1] == floor_
        @test psf.λ[1] == floor_

        # Being an ordinary PoleSet, the §4 box check applies column by column.
        @test is_identifiable(poleset(tp, 0.2, Δx, kT), floor_ / 5, 2.205)
    end

    @testset "parameters round-trip" begin
        tp = tracking_seed(3, 2, tail, floor_)
        θ = parameters(tp)
        scrambled = collect(range(0.11, 0.99; length = length(θ)))
        parameters!(tp, scrambled)
        @test parameters(tp) == scrambled
        parameters!(tp, θ)
        @test parameters(tp) == θ
    end

    # The no-drift assertion promised in varpro.jl.
    @testset "_fill_design! agrees with design_matrix" begin
        tp = tracking_seed(2, 2, tail, floor_)
        ω = exp.(range(log(1e-4), log(2.0); length = 90))
        for h in (0.0, 0.05, 0.3, -0.3)
            ps = poleset(tp, h, 0.002, kT)
            M = Matrix{Float64}(undef, length(ω), 1 + sum(mode_counts(tp)))
            _fill_design!(M, ps, ω)
            @test M == design_matrix(ps, ω)
        end
        @test_throws DimensionMismatch _fill_design!(zeros(3, 3),
                                                     poleset(tp, 0.1, 0.002, kT), ω)
    end

    # ------------------------------------------------------------------------------
    # Synthetic ground truth: a spectrum generated by the model itself.
    # ------------------------------------------------------------------------------
    # Δ must vary across the grid or (w, z) are not separately identifiable — they enter
    # only as w·Δ + z·k_BT, one constant, when Δ is constant. On the real data Δ(x) does
    # vary (1.4e-4 to 0.05 eV), so making it vary here is fidelity, not convenience.
    N_x = 14
    h_true = collect(range(-0.18, 0.30; length = N_x))          # crosses zero
    Δ_true = collect(range(0.0008, 0.0020; length = N_x))
    # ω_max must sit above the highest tail pole. §4: a mode flat over the whole fitted
    # band is indistinguishable from a₀, so with ω_max = 2 Ha the β = 2.2 rung and the
    # white part trade amplitude freely and the *amplitudes* stop being identifiable even
    # though the *fit* stays exact. test_fitting.jl picks well-separated poles for the
    # same reason. Extending the band is the fix; loosening the tolerance would have hidden
    # a real property of the model.
    ω_t = exp.(range(log(1e-5), log(8.0); length = 120))
    tp_true = TrackingPoles([1.10], [0.004], [0.20], [1.00], [1.50],
                            Float64[], Float64[], Float64[], tail, floor_)

    A_true = reshape(collect(range(0.5, 2.0; length = N_x)), 1, N_x)
    A_tail = fill(0.05, length(tail.α), N_x)        # one amplitude per tail rung
    truth = GridAmplitudes(fill(0.0, N_x), zeros(0, N_x), vcat(A_true, A_tail))

    Λ_t = Matrix{Float64}(undef, length(ω_t), N_x)
    for i in 1:N_x
        ps = poleset(tp_true, h_true[i], Δ_true[i], kT)
        a₀, a, A = amplitudes(truth, i)
        Λ_t[:, i] = [spectral_density(w, ps, a₀, a, A) for w in ω_t]
    end
    σ_t = residual_scales(Λ_t; ε = 1e-3)
    v_t = fill(1 / N_x, N_x)

    @testset "inner solve is exact at the true poles" begin
        out = tracking_solve(tp_true, ω_t, Λ_t, σ_t, h_true, Δ_true, kT, v_t)
        got = reduce(hcat, [vcat(amplitudes(out.model, i)...) for i in 1:N_x])
        want = reduce(hcat, [vcat(amplitudes(truth, i)...) for i in 1:N_x])
        # Whole-matrix `≈`, as test_fitting.jl states its own recovery: an entrywise
        # relative error divides by the exactly-zero true a₀ and reports the 6e-17 NNLS
        # returns there as a 6e-5 failure.
        @test got ≈ want rtol = 1e-10
        @test out.J < 1e-20
        # Convexity of the inner problem is not a claim about the outer search: at fixed
        # poles this is the same NNLS stage 1 solves, and it is exact for the same reason.
        @test all(≥(0), got)
    end

    @testset "errors vanish on the truth" begin
        E = tracking_errors(tp_true, truth, ω_t, Λ_t, h_true, Δ_true, kT)
        @test size(E) == (length(ω_t), N_x)
        @test maximum(abs, E) < 1e-12
    end

    @testset "outer search recovers perturbed coefficients" begin
        start = deepcopy(tp_true)
        parameters!(start, parameters(tp_true) .* [1.15, 1.20, 0.80, 1.25, 0.75])
        J_start = tracking_solve(start, ω_t, Λ_t, σ_t, h_true, Δ_true, kT, v_t).J

        fit = fit_tracking(start, ω_t, Λ_t, h_true, Δ_true, kT, v_t; iterations = 3000)
        @test fit.J < 1e-4 * J_start
        got, want = parameters(fit.poles), parameters(tp_true)
        @test maximum(abs.(got .- want) ./ abs.(want)) < 0.05
        @test fit.evaluations > 0
        @test mode_counts(fit.model) == mode_counts(tp_true)
    end

    # The stage-2 gradient is now shared between the fixed ladder and the tracking one,
    # so the two paths must agree wherever the ladders coincide. `u` is deliberately not
    # the warp here: with h ≡ 0 the warp is constant and the basis matrix would be rank 1,
    # which would make the test pass for the wrong reason.
    @testset "tracking + basis reduces to stage 2" begin
        tp = tracking_seed(1, 1, tail, floor_)
        Δc = fill(0.0015, N_x)
        h0 = zeros(N_x)
        ps = poleset(tp, 0.0, Δc[1], kT)
        u = collect(range(-1.0, 1.0; length = N_x))

        M = design_matrix(ps, ω_t)
        Λb = Matrix{Float64}(undef, length(ω_t), N_x)
        for i in 1:N_x
            Λb[:, i] = M * [0.01 + 0.002i, 0.3, 0.9, 0.05, 0.02]
        end
        env = shape_envelope(Λb)

        m1, r1 = fit_basis_report(ChebyshevAmplitudes, ps, ω_t, Λb, u, env;
                                  degree = 4, iterations = 200)
        m2, r2 = fit_tracking_basis(ChebyshevAmplitudes, tp, ω_t, Λb, u, env,
                                    h0, Δc, kT; degree = 4, iterations = 200)
        @test parameters(m1) ≈ parameters(m2)
        @test r1.J_final ≈ r2.J_final
        @test mode_counts(m2) == mode_counts(tp)
    end

    # ------------------------------------------------------------------------------
    # The tie back to stage 1.
    # ------------------------------------------------------------------------------
    @testset "h ≡ 0 reduces to a fixed ladder" begin
        # Every column then carries the same ladder, so the tracking path must reproduce
        # the fixed-ladder one it is meant to be compared against — exactly, not nearly.
        tp = tracking_seed(1, 1, tail, floor_)
        Δc = fill(0.0015, N_x)
        h0 = zeros(N_x)
        ps = poleset(tp, 0.0, Δc[1], kT)

        # Deliberately *not* exactly representable: a deterministic 2 % ripple makes J
        # meaningfully non-zero, so that comparing the two routes to it compares the
        # objective rather than the rounding noise of two sums that are both 1e-29.
        Λ0 = Matrix{Float64}(undef, length(ω_t), N_x)
        M = design_matrix(ps, ω_t)
        for i in 1:N_x
            Λ0[:, i] = (M * [0.02 * i, 0.4, 1.1, 0.06, 0.03]) .*
                       (1 .+ 0.02 .* sin.(3 .* ω_t .+ i))     # a₀, a, A×3
        end
        σ0 = residual_scales(Λ0; ε = 1e-3)

        want = fit_amplitudes(ps, ω_t, Λ0; ε = 1e-3)
        out = tracking_solve(tp, ω_t, Λ0, σ0, h0, Δc, kT, v_t)
        got_amp = reduce(hcat, [vcat(amplitudes(out.model, i)...) for i in 1:N_x])
        want_amp = reduce(hcat, [vcat(amplitudes(want, i)...) for i in 1:N_x])
        @test got_amp ≈ want_amp
        @test out.J ≈ objective_value(ps, want, ω_t, Λ0, v_t, σ0)

        # …and the two error routines agree there too.
        @test tracking_errors(tp, want, ω_t, Λ0, h0, Δc, kT) ≈
              relative_errors(ps, want, ω_t, Λ0)
    end
end
