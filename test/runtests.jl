using MemoryKernelFitting
using Test

# Cosine-transform pair check by trapezoid quadrature. The left-endpoint rule's O(Δτ)
# bias is larger than Λ itself at high ω, where the Lorentzians have decayed.
function cosine_transform(f, ω; Δτ = 1e-4, τmax = 80.0)
    τs = 0.0:Δτ:τmax
    g(τ) = f(τ) * cos(ω * τ)
    return Δτ * (sum(g, τs) - (g(first(τs)) + g(last(τs))) / 2)
end

@testset "MemoryKernelFitting.jl" begin
    a₀ = 0.5
    a, λ = [2.0, 0.75], [1.0, 10.0]        # overdamped modes
    A, α, β = [1.5, 0.4], [0.5, 1.2], [3.0, 7.0]  # underdamped pairs

    @testset "basis functions" begin
        @test lorentzian(0.0, 2.0) == 1 / 2.0
        @test mirrored_lorentzian(0.0, 0.5, 3.0) ≈ 0.5 / (0.5^2 + 3.0^2)
        # L is even in ω and in β
        @test mirrored_lorentzian(2.0, 0.5, 3.0) ≈ mirrored_lorentzian(-2.0, 0.5, 3.0)
        @test mirrored_lorentzian(2.0, 0.5, 3.0) ≈ mirrored_lorentzian(2.0, 0.5, -3.0)
        # β → 0 collapses the pair onto the zero-centred Lorentzian
        @test mirrored_lorentzian(1.7, 0.8, 0.0) ≈ lorentzian(1.7, 0.8)
    end

    @testset "resonance requires β > α/√3" begin
        ωs = 0.0:0.005:15.0
        # Underdamped: peak sits away from ω = 0. This is why the pairs are mandatory.
        above = [mirrored_lorentzian(ω, 1.0, 5.0) for ω in ωs]
        @test ωs[argmax(above)] > 0.0
        # Overdamped-collapsed: monotone, peak at ω = 0
        below = [mirrored_lorentzian(ω, 1.0, 0.2) for ω in ωs]
        @test argmax(below) == 1
        @test all(≤(0.0), diff(below))
        # A sum of zero-centred Lorentzians can never produce a resonance
        overdamped = [spectral_density(ω, a₀, a, λ) for ω in ωs]
        @test all(≤(0.0), diff(overdamped))
    end

    @testset "spectral_density" begin
        # Markovian anchor: Λ(0) = a₀ + Σ aₖ/λₖ + Σ Aⱼ αⱼ/(αⱼ² + βⱼ²)
        @test spectral_density(0.0, a₀, a, λ, A, α, β) ≈
              a₀ + sum(a ./ λ) + sum(@. A * α / (α^2 + β^2))
        # Λ(ω) → a₀ as ω → ∞
        @test spectral_density(1e8, a₀, a, λ, A, α, β) ≈ a₀ atol = 1e-8
        # No modes at all ⇒ pure white part
        @test spectral_density(3.0, a₀, Float64[], Float64[]) == a₀
        # Omitting the pairs must agree with passing empty pairs
        @test spectral_density(2.0, a₀, a, λ) ==
              spectral_density(2.0, a₀, a, λ, Float64[], Float64[], Float64[])
        @test_throws DimensionMismatch spectral_density(1.0, a₀, a, λ[1:1], A, α, β)
        @test_throws DimensionMismatch spectral_density(1.0, a₀, a, λ, A, α, β[1:1])
    end

    @testset "memory_kernel" begin
        @test memory_kernel(0.0, a, λ, A, α, β) ≈ sum(a) + sum(A)
        @test memory_kernel(1e3, a, λ, A, α, β) ≈ 0.0 atol = 1e-12
        # cos(βτ) makes the kernel change sign — a resonance in ω, ringing in τ
        @test memory_kernel(π / β[1], [0.0], [1.0], A[1:1], α[1:1], β[1:1]) < 0
        @test memory_kernel(2.0, a, λ) ==
              memory_kernel(2.0, a, λ, Float64[], Float64[], Float64[])
        @test_throws DimensionMismatch memory_kernel(1.0, a, λ[1:1], A, α, β)
    end

    @testset "kernel/spectrum consistency" begin
        # Λ(ω) - a₀ = ∫₀^∞ K(τ) cos(ωτ) dτ, checked by quadrature.
        K(τ) = memory_kernel(τ, a, λ, A, α, β)
        for ω in (0.0, 0.5, 3.0, 7.0)
            @test cosine_transform(K, ω) ≈
                  spectral_density(ω, a₀, a, λ, A, α, β) - a₀ rtol = 1e-3
        end
    end
end
