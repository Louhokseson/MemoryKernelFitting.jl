# The analytic model is checkable without the data — limits, derivatives and the
# crossing are all properties of the formulas — so only the regression against the
# stored columns is guarded on `isfile`. That guard is the one test that matters most:
# it is what certifies the parameters, not the algebra.

using HDF5

et_dir = joinpath(@__DIR__, "..", "data", "sims", "friction", "ErpenbeckThoss")
et_files = isdir(et_dir) ?
    sort(filter(f -> startswith(basename(f), "Lambda_spectrum_"),
                readdir(et_dir; join = true))) :
    String[]

@testset verbose = true "ErpenbeckThossModel" begin
    m = ErpenbeckThossModel(; Γ_eV = 0.1)

    @testset "construction" begin
        @test m.Γ ≈ eV_to_hartree(0.1)
        @test m.Dₑ ≈ eV_to_hartree(3.52)
        @test m.x₀ == m.x₀′ == 1.78            # x₀′ defaults to x₀
        @test m.a == 1.7361                    # Å⁻¹, not converted: x stays in Å
        @test m.V∞ ≈ eV_to_hartree(-1.5)

        @test_throws ArgumentError ErpenbeckThossModel(; Γ_eV = -0.1)
        @test_throws ArgumentError ErpenbeckThossModel(; Γ_eV = 0.1, Dₑ_eV = -1.0)
        @test_throws ArgumentError ErpenbeckThossModel(; Γ_eV = 0.1, a = 0.0)
        @test_throws ArgumentError ErpenbeckThossModel(; Γ_eV = 0.1, a′ = -1.0)
        @test_throws ArgumentError ErpenbeckThossModel(; Γ_eV = 0.1, ã = 0.0)
        @test_throws ArgumentError ErpenbeckThossModel(; Γ_eV = 0.1, q = 1.5)
        @test_throws ArgumentError ErpenbeckThossModel(; Γ_eV = 0.1, q = -0.01)
    end

    @testset "morse_zero_point" begin
        # Eq. 36/40 at n = 0, the value the file data was generated with.
        E₀ = morse_zero_point(m.Dₑ, m.a, ELECTRON_MASSES_PER_U)
        @test hartree_to_eV(E₀) ≈ 0.147336 rtol = 1e-5
        @test m.c ≈ -E₀                        # c = -E₀ is the default offset

        # A heavier adsorbate has a lower zero point: E₀ ∝ ω₀ ∝ 1/√m to leading order.
        @test morse_zero_point(m.Dₑ, m.a, 2 * ELECTRON_MASSES_PER_U) < E₀
    end

    @testset "level position" begin
        # h = U₁ - U₀, with both surfaces evaluated at the same x.
        for x in (0.8, 1.5, 2.0203445, 3.0, 5.0)
            U₀, U₁ = diabatic_potentials(x, m)
            @test level_position(x, m) ≈ U₁ - U₀
        end

        # The four facts §0 quotes, in eV, from the analytic form alone.
        @test hartree_to_eV(level_position(0.8, m)) ≈ -7.6517 rtol = 1e-4
        @test hartree_to_eV(level_position(1.2, m)) ≈ 8.6462 rtol = 1e-4
        @test hartree_to_eV(level_position(5.0, m)) ≈ -4.8551 rtol = 1e-4
        # §0 quotes the peak as "8.65 eV at x = 1.2 Å", but 1.2 is a *grid node*: the
        # stationary point of the analytic h is at 1.18774, where h = 8.6535 eV.
        x_pk = 1.1877367256323428
        @test dlevel_position(x_pk, m) ≈ 0 atol = 1e-12
        @test hartree_to_eV(level_position(x_pk, m)) ≈ 8.6535 rtol = 1e-4
        grid = 0.8:0.0001:5.0
        @test grid[argmax(level_position.(grid, Ref(m)))] ≈ x_pk atol = 1e-4
        @test level_position(x_pk, m) > level_position(1.2, m)
    end

    @testset "hybridisation width" begin
        @test hybridisation_width(2.0, m) ≈ π * coupling(2.0, m)^2

        # The switch is centred at x̃ = 3.5 Å between Γ/2 and (Γ/2)q².
        @test hybridisation_width(-10.0, m) ≈ m.Γ / 2 rtol = 1e-12
        @test hybridisation_width(50.0, m) ≈ (m.Γ / 2) * m.q^2 rtol = 1e-12
        # Halfway on the *coupling* switch, hence a quarter-ish on Δ — check the
        # coupling itself, which is where the tanh actually is.
        @test coupling(m.x̃, m) ≈ sqrt(m.Γ / 2π) * ((1 - m.q) / 2 + m.q)

        # Monotone decreasing everywhere.
        Δs = hybridisation_width.(0.5:0.01:6.0, Ref(m))
        @test all(diff(Δs) .< 0)

        # Still on the plateau at the crossing — the reason Δ carries no information
        # in the fit (the `TiedLaw` collinearity).
        @test hybridisation_width(2.0203445, m) ≈ m.Γ / 2 rtol = 1e-2
        @test hartree_to_eV(hybridisation_width(2.0203445, m)) ≈ 0.04975 rtol = 1e-3
        # …and well off it beyond the switch.
        @test hybridisation_width(5.0, m) < 0.01 * (m.Γ / 2)
    end

    @testset "analytic derivatives" begin
        # Central differences on a step small enough that the O(δ²) error is below the
        # tolerance and large enough that cancellation is not.
        δ = 1e-6
        central(f, x) = (f(x + δ, m) - f(x - δ, m)) / 2δ
        for x in (0.8, 1.2, 2.0203445, 3.0, 3.5, 5.0)
            @test dlevel_position(x, m) ≈ central(level_position, x) rtol = 1e-6
            @test dhybridisation_width(x, m) ≈ central(hybridisation_width, x) rtol = 1e-6
        end

        # The number regime (B)'s legality turns on (§9.7).
        @test hartree_to_eV(dlevel_position(2.0203445032675518, m)) ≈ -8.3897 rtol = 1e-4
        # Δ drifts by ~2 % per Å where the friction peak lives — small next to h's
        # 8.39 eV/Å, but not zero; the earlier claim of "negligible" was too strong.
        @test dhybridisation_width(2.02, m) / hybridisation_width(2.02, m) ≈ -0.02035 rtol = 1e-3
        @test abs(dhybridisation_width(2.02, m)) < 0.03 * abs(dhybridisation_width(3.5, m))
    end

    @testset "fermi_crossing" begin
        x₀ = fermi_crossing(m)
        @test x₀ ≈ 2.0203445032675518 atol = 1e-9

        # Converged to the last representable Float64: no root-finder can do better
        # than |h| <= |h'| * eps(x0), so state that bound rather than a bare 1e-16 —
        # the bare threshold had only 3.8x of margin and is a libm change away from
        # failing.
        @test abs(level_position(x₀, m)) ≤ 2 * abs(dlevel_position(x₀, m)) * eps(x₀)

        # Bracket-independent to within a couple of ULP. Not `==`: bisection from a
        # different bracket visits different midpoints and may land on the adjacent
        # float, so bitwise equality would be asserting the search path, not the root.
        @test fermi_crossing(m; bracket = (1.9, 2.5)) ≈ x₀ atol = 4 * eps(x₀)

        @test_throws ArgumentError fermi_crossing(m; bracket = (3.0, 1.5))
        @test_throws ArgumentError fermi_crossing(m; bracket = (2.5, 3.0))  # no sign change

        # The second, inner crossing: real, but the graded grid barely samples it.
        x_inner = fermi_crossing(m; bracket = (0.8, 1.0))
        @test x_inner ≈ 0.8805357513456046 atol = 1e-9
        @test abs(level_position(x_inner, m)) ≤
              2 * abs(dlevel_position(x_inner, m)) * eps(x_inner)
        @test x_inner < x₀
    end

    @testset "c shifts h rigidly and moves the crossing" begin
        # Not a re-parametrisation: overriding c gives a different model.
        shifted = ErpenbeckThossModel(; Γ_eV = 0.1, c_eV = 0.0)
        d = level_position(1.5, shifted) - level_position(1.5, m)
        @test d ≈ m.c                    # h = U₁ - Morse - c, so Δh = c_old - c_new
        for x in (1.0, 2.0, 3.0, 4.0)
            @test level_position(x, shifted) - level_position(x, m) ≈ d
        end
        @test !isapprox(fermi_crossing(shifted), fermi_crossing(m); atol = 1e-6)
        # Δ does not depend on c at all.
        @test hybridisation_width(2.0, shifted) == hybridisation_width(2.0, m)
    end

    @testset "Γ scales Δ and leaves h alone" begin
        m2 = ErpenbeckThossModel(; Γ_eV = 1.0)
        @test level_position(2.0, m2) == level_position(2.0, m)
        @test fermi_crossing(m2) == fermi_crossing(m)
        # Δ ∝ Γ at every x, so Δ(x)/Γ is one universal function.
        for x in (1.0, 2.0, 3.5, 5.0)
            @test hybridisation_width(x, m2) / m2.Γ ≈ hybridisation_width(x, m) / m.Γ
        end
    end

    if isempty(et_files)
        @info "ErpenbeckThoss data not found — skipping the regression against the " *
              "stored h_eV and Delta_eV" dir = et_dir
    else
        @testset "reproduces the stored columns" begin
            # Only h_eV, Delta_eV and x_Ang are read: Lambda_au is 123 MB per file and
            # is not needed to certify the model's parameters.
            for f in et_files
                h5open(f, "r") do fid
                    x = read(fid["x_Ang"])
                    h_eV = read(fid["h_eV"])
                    Δ_eV = read(fid["Delta_eV"])
                    Γ_eV = read_attribute(fid, "Gamma_eV")
                    mf = ErpenbeckThossModel(; Γ_eV)

                    @test maximum(abs, hartree_to_eV.(level_position.(x, Ref(mf))) .- h_eV) < 1e-11
                    @test hartree_to_eV.(hybridisation_width.(x, Ref(mf))) ≈ Δ_eV rtol = 1e-12

                    # x_resonance is stored per file; recover it from h alone.
                    @test fermi_crossing(mf) ≈
                          read_attribute(fid, "x_resonance_Ang") atol = 1e-9
                end
            end
            @test length(et_files) == 7          # all seven Γ, not a lucky one
        end

        @testset "ErpenbeckThossModel(::LambdaSpectrum)" begin
            f = joinpath(et_dir,
                "Lambda_spectrum_Gamma_eV=0.1_Nx=257_T_K=300.0_domega_eV=0.001.h5")
            s = load_lambda_spectrum(f)
            ms = ErpenbeckThossModel(s)
            @test ms.Γ ≈ s.Γ
            # The loaded columns are already in Hartree, so this compares like with like.
            @test level_position.(s.x, Ref(ms)) ≈ s.h atol = 1e-13
            @test hybridisation_width.(s.x, Ref(ms)) ≈ s.Δ rtol = 1e-12
            @test fermi_crossing(ms) ≈ s.x_resonance atol = 1e-9

            # The analytic derivative against a central difference on the real grid.
            i = argmin(abs.(s.h))
            @test dlevel_position(s.x[i], ms) ≈
                  (s.h[i+1] - s.h[i-1]) / (s.x[i+1] - s.x[i-1]) rtol = 1e-5
        end
    end
end
