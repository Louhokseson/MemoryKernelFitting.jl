# The continuous readout: `markovian_friction`, `ContinuousAmplitudes`, `qgle_coupling!`.
#
# The fit files live in the gitignored data/, so the tests that compare against a real
# fit are guarded on isfile. Everything structural is tested unconditionally.

using HDF5

fit_file = joinpath(@__DIR__, "..", "data", "fits",
    "prony_stage2Cheby_Gamma_eV=0.1_m=43_K_C=20_K_R=3_deg=22_eps=0.001_anchor=1000.0_thetav=0.5.h5")

@testset "Continuous readout" begin

    et = ErpenbeckThossModel(; Γ_eV = 0.1)
    kT = boltzmann_energy(300.0)

    @testset "markovian_friction — validation" begin
        @test_throws ArgumentError markovian_friction(2.0, et, 0.0)
        @test_throws ArgumentError markovian_friction(2.0, et, -1e-3)
        @test_throws ArgumentError markovian_friction(2.0, et, kT; halfwidth = 0.0)
        # Simpson needs an even number of intervals.
        @test_throws ArgumentError markovian_friction(2.0, et, kT; nodes = 400)
        @test_throws ArgumentError markovian_friction(2.0, et, kT; nodes = 1)
    end

    @testset "markovian_friction — positivity and shape" begin
        xs = [1.5, 1.9, 2.0203445, 2.1, 2.5, 3.0, 4.0]
        Λ = [markovian_friction(x, et, kT) for x in xs]
        @test all(>(0), Λ)
        # The friction peaks at the Fermi crossing, by orders of magnitude. This is the
        # feature the whole fit exists to represent, so it is worth asserting.
        @test argmax(Λ) == findfirst(≈(2.0203445), xs)
        @test maximum(Λ) > 1e4 * minimum(Λ)
    end

    @testset "markovian_friction — quadrature is converged" begin
        # The defaults claim ~1e-7 relative against a far finer rule. Check it where the
        # integrand is hardest: on and around the crossing.
        for x in (1.9243, 2.0203445, 2.0943, 2.5)
            ref = markovian_friction(x, et, kT; halfwidth = 80kT, nodes = 20001)
            @test markovian_friction(x, et, kT) ≈ ref rtol = 1e-6
        end
    end

    @testset "markovian_friction — the kT → 0 limit is analytic" begin
        # As -∂f/∂ϵ → δ(ϵ), the integral collapses to its integrand at the Fermi level:
        #     Λ → π (∂h - h ∂Γ/Γ)² A(0)²,   A(0) = (Γ/2) / (π (h² + (Γ/2)²)).
        # Independent of the quadrature, so this catches a wrong prefactor or a dropped
        # ∂Γ term, which the node comparison below could not distinguish from a bad fit.
        for x in (1.5, 2.3, 3.0)
            h = level_position(x, et)
            Γ = 2 * hybridisation_width(x, et)
            ∂h = dlevel_position(x, et) * BOHR_IN_ANGSTROM
            ∂Γ = 2 * dhybridisation_width(x, et) * BOHR_IN_ANGSTROM
            A0 = (Γ / 2) / (π * (h^2 + (Γ / 2)^2))
            limit = π * (∂h - h * ∂Γ / Γ)^2 * A0^2

            cold = Γ / 2000                      # ≪ every other energy in the problem
            @test markovian_friction(x, et, cold; nodes = 2001) ≈ limit rtol = 1e-3
        end
    end

    @testset "ContinuousAmplitudes — construction" begin
        C = randn(5, 1 + 2 + 3)
        m = ContinuousAmplitudes(C, identity, x -> 1.0; K_R = 2, K_C = 3)
        @test mode_counts(m) == (2, 3)
        @test auxiliary_dimension(m) == 2 + 2 * 3
        @test occursin("ContinuousAmplitudes", sprint(show, m))

        @test_throws DimensionMismatch ContinuousAmplitudes(
            C, identity, x -> 1.0; K_R = 2, K_C = 4)
        @test_throws ArgumentError ContinuousAmplitudes(
            C, identity, x -> 1.0; K_R = -1, K_C = 3)
        @test_throws ArgumentError ContinuousAmplitudes(C, et, 0.0; K_R = 2, K_C = 3)
    end

    @testset "ContinuousAmplitudes — evaluation is g² with the envelope factored in" begin
        # Degree 0, so every Clenshaw sum is just its single coefficient: the amplitudes
        # are then exactly Λ₀(x) · C[1, c]², which can be written down by hand.
        C = reshape([2.0, -3.0, 0.5, 1.5], 1, 4)      # K_R = 1, K_C = 2
        Λ0 = 4.0
        m = ContinuousAmplitudes(C, x -> 0.0, x -> Λ0; K_R = 1, K_C = 2)
        a₀, a, A = amplitudes(m, 1.234)
        @test a₀ ≈ Λ0 * 2.0^2
        @test a ≈ [Λ0 * (-3.0)^2]                     # the sign is squared away
        @test A ≈ [Λ0 * 0.5^2, Λ0 * 1.5^2]
        @test all(≥(0), vcat(a₀, a, A))

        # The in-place path must agree with the allocating one, exactly.
        ain, Ain = zeros(1), zeros(2)
        @test amplitudes!(ain, Ain, m, 1.234) == a₀
        @test ain == a && Ain == A
        @test_throws DimensionMismatch amplitudes!(zeros(2), Ain, m, 1.234)
        @test_throws DimensionMismatch amplitudes!(ain, zeros(3), m, 1.234)
        @test_throws DomainError amplitudes(
            ContinuousAmplitudes(C, x -> 0.0, x -> -1.0; K_R = 1, K_C = 2), 1.0)
    end

    @testset "qgle_coupling! — layout" begin
        a, A = [4.0, 9.0], [16.0, 25.0, 36.0]
        g = fill(NaN, length(a) + 2length(A))
        @test qgle_coupling!(g, a, A) === g
        @test g == [2.0, 3.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0]
        # Only s₁ of each pair couples — the structural zeros of the readout.
        @test all(iszero, g[(length(a) + 2):2:end])

        @test_throws DimensionMismatch qgle_coupling!(zeros(7), a, A)
        @test_throws DomainError qgle_coupling!(g, [-1.0, 9.0], A)
        @test_throws DomainError qgle_coupling!(g, a, [16.0, -1.0, 36.0])

        # Every element is rewritten, so a dirty buffer cannot leak into the result.
        fill!(g, 12345.0)
        qgle_coupling!(g, a, A)
        @test g == [2.0, 3.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0]
    end

    @testset "qgle_coupling! agrees with qgle_blocks" begin
        ps = PoleSet([0.1, 0.2], [0.3, 0.4], [1.0, 2.0])          # K_R = 2, K_C = 2
        C = randn(4, 1 + 2 + 2)
        m = ContinuousAmplitudes(C, x -> tanh(x), x -> 1.0 + x^2; K_R = 2, K_C = 2)
        x = 0.7
        b = qgle_blocks(ps, m, x)
        a, A = zeros(2), zeros(2)
        g = zeros(auxiliary_dimension(m))
        a₀ = qgle_coupling!(g, a, A, m, x)
        @test g == b.g
        @test a₀ == b.Γ₁₁
        @test length(g) == auxiliary_dimension(ps)
    end

    if isfile(fit_file)
        C, x_nodes, envelope, t_nodes, ladder, K_R, K_C = h5open(fit_file, "r") do f
            (read(f["C_cheb"]), read(f["x_Ang"]), read(f["envelope"]), read(f["t_cheb"]),
             PoleSet(read(f["lambda_Ha"]), read(f["alpha_Ha"]), read(f["beta_Ha"])),
             read_attribute(f, "K_R"), read_attribute(f, "K_C"))
        end

        @testset "markovian_friction reproduces the stored envelope" begin
            # `envelope[i] = √(Λ(xᵢ,0))` was read off the data the fit was made against.
            # Agreement here is a statement about the analytic model, its derivatives and
            # the Å → bohr conversion all at once — the one place `docs/design/0001`
            # warns a silent factor would be invisible.
            for i in eachindex(x_nodes)
                @test markovian_friction(x_nodes[i], et, kT) ≈ envelope[i]^2 rtol = 1e-6
            end
        end

        @testset "the continuous model reproduces the fit at every node" begin
            cheb = ChebyshevAmplitudes(C, (π / 2) .* t_nodes, envelope; K_R, K_C)
            cont = ContinuousAmplitudes(C, et, kT; K_R, K_C)
            @test auxiliary_dimension(cont) == auxiliary_dimension(ladder)

            for i in eachindex(x_nodes)
                a₀c, ac, Ac = amplitudes(cheb, i)
                a₀n, an, An = amplitudes(cont, x_nodes[i])
                @test a₀n ≈ a₀c rtol = 1e-6
                @test an ≈ ac rtol = 1e-6
                @test An ≈ Ac rtol = 1e-6
            end
        end

        @testset "the Markovian anchor holds off the grid" begin
            # §4 weights Λ_θ(x,0) = Λ(x,0) into the fit; if the continuous readout is
            # assembled correctly it must still hold between the nodes, where nothing
            # was fitted. Loose tolerance: this is the *fit's* accuracy, not ours.
            cont = ContinuousAmplitudes(C, et, kT; K_R, K_C)
            for x in (1.55, 1.9243, 2.0203445, 2.0943, 2.37, 3.16, 4.4)
                @test spectral_density(0.0, ladder, cont, x) ≈ cont.Λ₀(x) rtol = 5e-3
            end
        end

        @testset "the kernel is the fitted one, and starts at Λ(x,0)-ish" begin
            cont = ContinuousAmplitudes(C, et, kT; K_R, K_C)
            for x in (1.9, 2.0203445, 2.6)
                a₀, a, A = amplitudes(cont, x)
                @test memory_kernel(0.0, ladder, cont, x) ≈ sum(a) + sum(A)
                # A decaying kernel: no growing mode can survive a positive ladder.
                @test abs(memory_kernel(5000.0, ladder, cont, x)) <
                      abs(memory_kernel(0.0, ladder, cont, x))
            end
        end
    else
        @info "Skipping the fitted-model tests: $(fit_file) not found " *
              "(data/ is gitignored)."
    end
end
