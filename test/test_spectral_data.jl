# The real files are ~123 MB each and data/ is gitignored, so every test that touches
# one is guarded on isfile. The pure functions below are tested unconditionally.
et_file = joinpath(@__DIR__, "..", "data", "sims", "friction", "ErpenbeckThoss",
                   "Lambda_spectrum_Gamma_eV=0.1_Nx=257_T_K=300.0_domega_eV=0.001.h5")

@testset "SpectralData" begin
    # A graded grid with the shape of the real one: a coarse baseline refined by a
    # patch around the resonance. dx spans a factor of 50 here, 290 in the data.
    graded = vcat(0.0:0.5:1.0, 1.45:0.01:1.55, 2.0:0.5:3.0)

    @testset "trapezoid_weights" begin
        @test trapezoid_weights([0.0, 2.0]) == [1.0, 1.0]
        @test trapezoid_weights(range(0.0, 1.0; length = 5)) ==
              [0.125, 0.25, 0.25, 0.25, 0.125]
        @test_throws ArgumentError trapezoid_weights([1.0])

        v = trapezoid_weights(graded)
        @test length(v) == length(graded)
        @test all(>(0), v)
        # Telescoping: the weights integrate the constant 1 over [x₁, x_N] exactly.
        @test sum(v) ≈ graded[end] - graded[1]
        @test sum(v) ≈ 3.0
        # vᵢ = ½(x_{i+1} - x_{i-1}) interior, one-sided at the ends.
        @test v[1] ≈ (graded[2] - graded[1]) / 2
        @test v[end] ≈ (graded[end] - graded[end - 1]) / 2
        for i in 2:(length(graded) - 1)
            @test v[i] ≈ (graded[i + 1] - graded[i - 1]) / 2
        end

        # The trap of design doc 0001: most of the points cover almost none of the
        # range, so vᵢ ≡ 1 would make the objective the refined patch alone.
        dense = 1.45 .< graded .< 1.55
        @test count(dense) > length(graded) / 2   # a majority of the points...
        @test sum(v[dense]) < sum(v) / 20         # ...carrying < 5% of the weight
    end

    @testset "subsample_omega" begin
        ω = collect(0.0:0.001:6.0)                # uniform, as the real grid is
        idx = subsample_omega(ω; n = 200)

        @test eltype(idx) <: Integer
        @test idx[1] == 1                         # the Markovian anchor, always kept
        @test idx[end] == length(ω)
        @test all(>(0), diff(idx))                # strictly increasing ⇒ unique
        @test length(idx) ≤ 200
        @test 100 ≤ length(idx)                   # collapse at the bottom is modest
        @test all(i -> 1 ≤ i ≤ length(ω), idx)

        # Logarithmic, not uniform: the index step grows from 1 at the bottom to
        # hundreds at the top, and successive retained frequencies sit in a near
        # constant ratio once the grid is fine enough to resolve the spacing.
        gaps = diff(idx)
        @test gaps[1] == 1
        @test gaps[end] > 50
        top = ω[idx][ω[idx] .≥ 1.0]
        ratios = top[2:end] ./ top[1:(end - 1)]
        @test all(r -> 1.0 < r < 1.1, ratios)
        @test maximum(ratios) - minimum(ratios) < 0.01
        # Half the retained points lie in the bottom 2% of the window; a uniform
        # subsample would put 2% of them there.
        @test count(<(0.1), ω[idx]) > length(idx) / 4

        # Asking for more points than the grid has just returns the whole grid.
        @test subsample_omega(collect(1.0:10.0); n = 100) == 1:10
        # ω₁ = 0 is fine: the log spacing spans [ω₂, ω_max] and index 1 is appended.
        @test first(subsample_omega(ω; n = 3)) == 1

        @test_throws ArgumentError subsample_omega(ω; n = 2)
        @test_throws ArgumentError subsample_omega([1.0])
    end

    if isfile(et_file)
        @testset "load_lambda_spectrum (Γ = 0.1 eV)" begin
            s = load_lambda_spectrum(et_file)

            # Domain restriction of the optimisation note §4: x ≥ 0.8 Å.
            @test length(s.x) == 254
            @test size(s.Λ) == (60_000, 254)
            @test length(s.h) == length(s.Δ) == 254
            @test s.x[1] == 0.8
            @test s.x[end] == 5.0
            @test issorted(s.x)                       # graded but strictly ascending
            @test length(load_lambda_spectrum(et_file; xmin = 0.0).x) == 257

            # x stays in Å (design doc 0001); everything spectral is in Hartree.
            @test s.ω[end] ≈ 59.999001 / 27.211386245988
            @test s.ω[1] ≈ eV_to_hartree(1e-6)
            @test length(s.ω) == 60_000
            @test s.Γ ≈ 0.1 / 27.211386245988
            @test s.kT ≈ 9.500e-4 rtol = 1e-3
            @test s.T_K == 300.0
            @test s.x_resonance ≈ 2.02034 rtol = 1e-5
            @test hartree_to_eV(minimum(s.h)) ≈ -7.6517 rtol = 1e-4
            @test hartree_to_eV(maximum(s.h)) ≈ 8.6462 rtol = 1e-4   # |h| ≪ ω_max
            @test hartree_to_eV(maximum(s.Δ)) ≈ 0.04999 rtol = 1e-3

            # Data facts of design doc 0001, recorded so later work need not re-check.
            @test extrema(s.Λ)[1] ≈ 9.47e-12 rtol = 1e-2
            @test extrema(s.Λ)[2] ≈ 1.398e3 rtol = 1e-3
            @test all(≥(0), s.Λ)                      # negatives appear only at Γ ≥ 0.2

            # The grading, measured: 84% of the columns carry 8% of the weight (the
            # 6.7% of design doc 0001, plus the half-cells reaching outwards into the
            # coarse baseline). With vᵢ ≡ 1 they would carry 84%.
            v = trapezoid_weights(s.x)
            @test sum(v) ≈ s.x[end] - s.x[1]
            near = abs.(s.x .- s.x_resonance) .< 0.15
            @test count(near) > 0.8 * length(s.x)
            @test sum(v[near]) < 0.09 * sum(v)

            idx = subsample_omega(s.ω)
            @test 500 ≤ length(idx) ≤ 1000
            @test idx[1] == 1
            @test size(s.Λ[idx, :]) == (length(idx), 254)
        end
    else
        @info "Skipping load_lambda_spectrum tests: $(et_file) not found " *
              "(data/ is gitignored; each file is ~123 MB)."
    end
end
