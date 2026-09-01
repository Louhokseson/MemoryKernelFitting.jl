using MemoryKernelFitting.PronySeriesFriction

# Cosine-transform pair check by trapezoid quadrature. The left-endpoint rule's O(Δτ)
# bias is larger than Λ itself at high ω, where the Lorentzians have decayed.
function cosine_transform(f, ω; Δτ = 1e-4, τmax = 80.0)
    τs = 0.0:Δτ:τmax
    g(τ) = f(τ) * cos(ω * τ)
    return Δτ * (sum(g, τs) - (g(first(τs)) + g(last(τs))) / 2)
end

@testset "Prony model" begin
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

# LinearAlgebra is not a declared dependency, so the linear algebra the readout tests
# need is spelled out: `*` and `exp` on a Matrix are available from the sysimage, `dot`
# and `opnorm` are not.
dot_(u, v) = sum(u .* v)

# Largest singular value by power iteration on MᵀM. Converges from below, so it is a
# lower bound on ‖M‖₂ — enough to catch a sign error in the Γ̃₁,₂ = -Γ̃₂,₁ᵀ pairing,
# which would push the norm well above 1.
function opnorm2(M; iters = 4000)
    Mᵀ = permutedims(M)
    v = fill(1 / sqrt(size(M, 2)), size(M, 2))
    for _ in 1:iters
        w = Mᵀ * (M * v)
        n = sqrt(sum(abs2, w))
        n == 0 && return 0.0
        v = w ./ n
    end
    return sqrt(sum(abs2, M * v))
end

@testset "PoleSet" begin
    @testset "validation" begin
        @test auxiliary_dimension(PoleSet([1.0, 2.0], [0.3], [1.0])) == 2 + 2 * 1
        @test auxiliary_dimension(PoleSet(Float64[], [0.3, 0.4], [1.0, 2.0])) == 4
        # Γ₂,₂ is stable only for λ > 0, α > 0; β = 0 is legal (Appendix A.0(ii)).
        @test_throws ArgumentError PoleSet([0.0], [0.3], [1.0])
        @test_throws ArgumentError PoleSet([-1.0], [0.3], [1.0])
        @test_throws ArgumentError PoleSet(Float64[], [0.0], [1.0])
        @test_throws ArgumentError PoleSet(Float64[], [0.3], [-1.0])
        @test PoleSet(Float64[], [0.3], [0.0]) isa PoleSet
        @test_throws DimensionMismatch PoleSet(Float64[], [0.3, 0.4], [1.0])
    end

    @testset "log ladder" begin
        β_min, β_max, K_C = 1.8e-3, 0.345, 29     # [0.05, 9.4] eV in Hartree
        ps = log_ladder(β_min, β_max, K_C)
        @test length(ps.β) == K_C
        @test isempty(ps.λ)
        @test auxiliary_dimension(ps) == 2K_C
        # spans exactly the requested range, log-spaced
        @test first(ps.β) ≈ β_min
        @test last(ps.β) ≈ β_max
        @test issorted(ps.β)
        r = ps.β[2:end] ./ ps.β[1:end-1]
        @test all(≈(first(r)), r)
        # Real rates are opt-in and log-spaced over their own span, which is *not*
        # the pair span: the crossing needs rates an order of magnitude below β_min.
        withR = log_ladder(β_min, β_max, K_C; K_R = 12, λ_min = 1.8e-4)
        @test length(withR.λ) == 12
        @test withR.α == ps.α && withR.β == ps.β        # pairs unchanged
        @test auxiliary_dimension(withR) == 12 + 2K_C
        @test issorted(withR.λ)
        @test first(withR.λ) ≈ 1.8e-4
        @test last(withR.λ) ≈ β_max                     # λ_max defaults to β_max
        rr = withR.λ[2:end] ./ withR.λ[1:end-1]
        @test all(≈(first(rr)), rr)                     # log-spaced
        @test log_ladder(β_min, β_max, K_C; K_R = 1, λ_min = 1.8e-4).λ ≈ [1.8e-4]
        # λ_min has no default on purpose — defaulting it to β_min would silently
        # reproduce the measured failure at the Fermi crossing.
        @test_throws ArgumentError log_ladder(β_min, β_max, K_C; K_R = 12)
        @test_throws ArgumentError log_ladder(β_min, β_max, K_C; K_R = -1, λ_min = 1e-4)
        @test_throws ArgumentError log_ladder(β_min, β_max, K_C; K_R = 2, λ_min = 0.0)
        @test_throws ArgumentError log_ladder(β_min, β_max, K_C; K_R = 2, λ_min = 1.0, λ_max = 0.5)

        # α/β is the constant that the log spacing exists to keep constant
        @test all(≈(0.2), ps.α ./ ps.β)
        @test all(≈(0.5), log_ladder(β_min, β_max, K_C; α_ratio = 0.5).α ./ ps.β)
        @test_throws ArgumentError log_ladder(β_min, β_max, 1)
        @test_throws ArgumentError log_ladder(β_max, β_min, 8)
        @test_throws ArgumentError log_ladder(0.0, β_max, 8)
    end

    @testset "split ladder" begin
        # 16 rungs over the resonance band + 4 over the tail band, the Γ = 0.1 default.
        ps = split_ladder([(1.8e-3, 0.345, 16), (0.441, 2.205, 4)], 3, 1.84e-4, 1.84e-3)
        @test length(ps.β) == 20
        @test length(ps.λ) == 3
        @test auxiliary_dimension(ps) == 43
        @test issorted(ps.β)
        @test allunique(ps.β)
        @test ps.β[1] ≈ 1.8e-3
        @test ps.β[16] ≈ 0.345
        @test ps.β[17] ≈ 0.441
        @test ps.β[20] ≈ 2.205
        @test all(≈(0.2), ps.α ./ ps.β)

        # Log spacing holds *within* each segment, not across the join.
        lo = ps.β[1:16]
        @test all(≈(lo[2] / lo[1]), lo[2:end] ./ lo[1:(end - 1)])
        hi = ps.β[17:20]
        @test all(≈(hi[2] / hi[1]), hi[2:end] ./ hi[1:(end - 1)])
        # The two segments are separate progressions, not one: the ladder as a whole is
        # not geometric, and the high segment steps at a different rate from the low.
        @test !all(≈(lo[2] / lo[1]), ps.β[2:end] ./ ps.β[1:(end - 1)])
        @test !(hi[2] / hi[1] ≈ lo[2] / lo[1])

        # One segment reproduces log_ladder exactly.
        @test split_ladder([(1.8e-3, 0.345, 29)], 0, 1.0, 1.0).β ≈
              log_ladder(1.8e-3, 0.345, 29).β

        # Real rates: same convention as log_ladder, and K_R = 0 means none.
        @test split_ladder([(1.8e-3, 0.345, 8)], 0, 1.0, 1.0).λ == Float64[]
        @test split_ladder([(1.8e-3, 0.345, 8)], 1, 1.84e-4, 1e-2).λ ≈ [1.84e-4]

        # Overlapping or descending segments are rejected, not silently merged: duplicate
        # rungs would make the NNLS design matrix rank deficient.
        @test_throws ArgumentError split_ladder(
            [(1.8e-3, 0.345, 8), (0.2, 2.205, 4)], 0, 1.0, 1.0)
        @test_throws ArgumentError split_ladder(
            [(0.441, 2.205, 4), (1.8e-3, 0.345, 8)], 0, 1.0, 1.0)
        @test_throws ArgumentError split_ladder([], 0, 1.0, 1.0)
        @test_throws ArgumentError split_ladder([(1.8e-3, 0.345, 0)], 0, 1.0, 1.0)
        @test_throws ArgumentError split_ladder([(0.0, 0.345, 8)], 0, 1.0, 1.0)
        @test_throws ArgumentError split_ladder([(1.8e-3, 0.345, 8)], 0, 1.0, 1.0;
                                                α_ratio = 0.0)
        @test_throws ArgumentError split_ladder([(1.8e-3, 0.345, 8)], 3, 1.0, 0.5)
    end

    @testset "uniform ladder" begin
        β_max, K_C = 0.345, 12
        ps = uniform_ladder(β_max, K_C)
        @test length(ps.β) == K_C
        @test auxiliary_dimension(ps) == 2K_C
        @test first(ps.β) == 0.0            # the degenerate k = 0 rung of PRB 89, 134303
        @test last(ps.β) ≈ β_max
        d = diff(ps.β)
        @test all(≈(first(d)), d)           # uniform in β, not in ln β
        @test all(==(first(ps.α)), ps.α)    # equal widths, their Eq. (85)
        @test first(ps.α) ≈ β_max / K_C
        @test all(≈(0.01), uniform_ladder(β_max, K_C; α = 0.01).α)
        @test_throws ArgumentError uniform_ladder(β_max, 1)
        @test_throws ArgumentError uniform_ladder(-1.0, 8)
    end

    @testset "pair budget" begin
        # §7: K_C ~ ln(β_max/β_min) / ln(1 + α/β) ≈ 29 over [0.05, 9.4] eV at α = 0.2β
        @test pair_budget(1.8e-3, 0.345) == 29
        # a ratio, so the budget cannot depend on the unit it is computed in
        @test pair_budget(eV_to_hartree(0.05), eV_to_hartree(9.4)) ==
              pair_budget(0.05, 9.4) == 29
        # the budget is the number of ×(1 + α_ratio) steps that cross the span, and it
        # is the smallest such number
        β_min, β_max = 1.8e-3, 0.345
        K_C = pair_budget(β_min, β_max)
        @test (β_max / β_min)^(1 / K_C) ≤ 1.2
        @test (β_max / β_min)^(1 / (K_C - 1)) > 1.2
        # a K_C-rung ladder spans K_C - 1 intervals, so budget + 1 rungs realises it
        r = log_ladder(β_min, β_max, K_C + 1).β
        @test r[2] / r[1] ≤ 1.2
    end

    @testset "identifiability box" begin
        dω, ω_max = eV_to_hartree(1e-3), eV_to_hartree(60.0)
        @test identifiability_box(dω, ω_max) == (5dω, ω_max)
        # 0001 quotes the box in Hartree to two significant figures
        @test identifiability_box(dω, ω_max)[1] ≈ 1.8e-4 rtol = 0.03
        @test identifiability_box(dω, ω_max)[2] ≈ 2.205 rtol = 1e-3
        @test is_identifiable(log_ladder(1.8e-3, 0.345, 29), dω, ω_max)
        # a rate below 5dω is narrower than the ω grid resolves
        @test !is_identifiable(PoleSet([1e-5], [0.3], [1.0]), dω, ω_max)
        # λ ≫ ω_max is a flat Lorentzian — fold it into a₀ rather than fitting it
        @test !is_identifiable(PoleSet([1e3], [0.3], [1.0]), dω, ω_max)
        # β = 0 is inside the box: §4 permits collapse onto a zero-centred shape
        @test is_identifiable(PoleSet(Float64[], [0.3], [0.0]), dω, ω_max)
        @test !is_identifiable(PoleSet(Float64[], [0.3], [10ω_max]), dω, ω_max)
    end
end

@testset "QGLE readout" begin
    ps = PoleSet([1.0, 10.0], [0.5, 1.2, 0.05], [3.0, 7.0, 0.0])
    a₀, a, A = 0.5, [2.0, 0.75], [1.5, 0.4, 0.9]
    b = qgle_blocks(ps, a₀, a, A)
    m = auxiliary_dimension(ps)

    @testset "block structure" begin
        @test size(b.Γ₂₂) == (m, m)
        @test length(b.g) == length(b.Σ₂) == m
        @test b.Γ₁₁ == a₀
        # real modes: 1×1 blocks, √a in g, √(2λ) noise
        @test b.Γ₂₂[1, 1] == 1.0 && b.Γ₂₂[2, 2] == 10.0
        @test b.g[1:2] ≈ sqrt.(a)
        @test b.Σ₂[1:2] ≈ sqrt.(2 .* ps.λ)
        # pairs: R(α,β) = [α β; -β α], √A in the *first* slot only
        for j in 1:3
            p = 2 + 2j - 1
            @test b.Γ₂₂[p:p+1, p:p+1] == [ps.α[j] ps.β[j]; -ps.β[j] ps.α[j]]
            @test b.g[p] ≈ sqrt(A[j])
            @test b.g[p+1] == 0.0        # only s₁ couples — Stella Eqs. (31)/(35)
            @test b.Σ₂[p] == b.Σ₂[p+1] ≈ sqrt(2 * ps.α[j])
        end
        @test qgle_blocks(ps, GridAmplitudes([a₀], reshape(a, :, 1), reshape(A, :, 1)), 1).g == b.g
        @test_throws DimensionMismatch qgle_blocks(ps, a₀, a[1:1], A)
        @test_throws DimensionMismatch qgle_blocks(ps, a₀, a, A[1:1])
        @test_throws ArgumentError qgle_blocks(ps, -1.0, a, A)
    end

    @testset "closed-form exponential" begin
        # Appendix A.1(ii): e^{-ΔtR} = e^{-αΔt}[cos(βΔt) I - sin(βΔt) J], verified
        # against a general matrix exponential.
        for Δt in (0.01, 0.3, 2.0)
            @test exp_gamma22(ps, Δt) ≈ exp(-Δt .* b.Γ₂₂)
        end
        @test exp_gamma22(ps, 0.0) ≈ exp(zero(b.Γ₂₂))
    end

    @testset "gᵀ e^{-τR} g = |g|² e^{-ατ} cos(βτ)" begin
        α₁, β₁, A₁ = 0.5, 3.0, 1.5
        one_pair = PoleSet(Float64[], [α₁], [β₁])
        bp = qgle_blocks(one_pair, 0.0, Float64[], [A₁])
        for τ in (0.0, 0.37, 1.4, π / β₁, 9.0)
            q = dot_(bp.g, exp_gamma22(one_pair, τ) * bp.g)
            @test q ≈ sum(abs2, bp.g) * exp(-α₁ * τ) * cos(β₁ * τ)
            # ... and that is exactly the single-pair Prony mode that was fitted
            @test q ≈ memory_kernel(τ, one_pair, Float64[], [A₁])
        end
        # the sine dies because gᵀJg = 0 for *any* g, not just the structural one
        for g in ([1.0, 2.0], [-0.3, 0.8], [0.0, 1.7])
            τ = 0.9
            @test dot_(g, exp_gamma22(one_pair, τ) * g) ≈
                  sum(abs2, g) * exp(-α₁ * τ) * cos(β₁ * τ)
        end
        # whole ladder at once: the readout reproduces the fitted kernel
        for τ in (0.0, 0.2, 1.1, 5.0)
            @test dot_(b.g, exp_gamma22(ps, τ) * b.g) ≈ memory_kernel(τ, ps, a, A)
        end
    end

    @testset "FDT" begin
        Γ = gamma_matrix(b)
        @test size(Γ) == (m + 1, m + 1)
        @test Γ[1, 1] == a₀
        @test Γ[2:end, 1] == b.g
        @test Γ[1, 2:end] == -b.g          # Γ̃₁,₂ = -Γ̃₂,₁ᵀ
        @test Γ[2:end, 2:end] == b.Γ₂₂
        # Appendix A.1(iii): the antisymmetric arms cancel and what is left is diagonal
        S = Γ .+ permutedims(Γ)
        expected = [2a₀; 2 .* ps.λ; vcat(([2αⱼ, 2αⱼ] for αⱼ in ps.α)...)]
        @test S ≈ [i == j ? expected[i] : 0.0 for i in 1:m+1, j in 1:m+1]
        @test all(≥(0), expected)          # ⪰ 0, hence S_Δt of Eq. (4.7) exists
        # ⇒ both propagators are contractions
        for Δt in (0.05, 0.5, 3.0)
            E = exp_gamma22(ps, Δt)
            G = permutedims(E) * E
            # block-diagonal with equal singular values per block: EᵀE is diagonal
            @test G ≈ [i == j ? G[i, i] : 0.0 for i in 1:m, j in 1:m]
            @test all(≤(1), [G[i, i] for i in 1:m])
            @test maximum(G[i, i] for i in 1:m) ≈ exp(-2Δt * minimum([ps.λ; ps.α]))
            @test opnorm2(exp(-Δt .* Γ)) ≤ 1 + 1e-10
        end
    end
end

@testset "amplitude models" begin
    ps = PoleSet([1.0, 10.0], [0.5, 1.2], [3.0, 7.0])
    K_R, K_C = 2, 2
    N = 6
    xs = collect(range(0.8, 5.0; length = N))

    @testset "warp_coordinate" begin
        Δ, kT = eV_to_hartree(0.05), boltzmann_energy(300.0)
        hs = range(eV_to_hartree(-90.0), eV_to_hartree(8.65); length = 400)
        s = [warp_coordinate(h, Δ, kT) for h in hs]
        @test all(>(0), diff(s))                 # strictly monotone in h
        @test all(σ -> abs(σ) < π / 2, s)        # bounded
        @test warp_coordinate(0.0, Δ, kT) == 0.0 # the Fermi crossing sits at s = 0
        # the crossing is an O(1) feature: half the peak width maps to ~π/4
        @test warp_coordinate(Δ + kT, Δ, kT) ≈ π / 4
        # the full h range essentially fills the interval: s ∈ [-1.570, 1.562]
        @test extrema(s)[1] < -π / 2 + 1e-3
        @test extrema(s)[2] > π / 2 - 1e-2
        @test_throws ArgumentError ChebyshevAmplitudes(zeros(2, 1), [2.0], [1.0]; K_R = 0, K_C = 0)
    end

    @testset "GridAmplitudes" begin
        g = GridAmplitudes(N, K_R, K_C)
        @test n_configurations(g) == N
        @test mode_counts(g) == (K_R, K_C)
        @test length(parameters(g)) == N * (1 + K_R + K_C)
        θ = collect(1.0:(N*(1+K_R+K_C)))
        @test parameters(parameters!(g, θ)) == θ
        # block-per-node layout: node 2 is entries 6..10
        a₀₂, a₂, A₂ = amplitudes(g, 2)
        @test a₀₂ == 6.0 && a₂ == [7.0, 8.0] && A₂ == [9.0, 10.0]
        @test spectral_density(1.3, ps, g, 2) == spectral_density(1.3, ps, a₀₂, a₂, A₂)
        @test memory_kernel(1.3, ps, g, 2) == memory_kernel(1.3, ps, a₂, A₂)
        # non-negativity is the FDT/Bochner constraint — rejected, not silently clamped
        @test_throws ArgumentError GridAmplitudes([-1.0], zeros(0, 1), zeros(0, 1))
        @test_throws ArgumentError GridAmplitudes([1.0], -ones(1, 1), zeros(0, 1))
        @test_throws ArgumentError parameters!(g, -θ)
        @test_throws DimensionMismatch parameters!(g, θ[1:end-1])
        @test_throws DimensionMismatch GridAmplitudes([1.0, 2.0], zeros(0, 1), zeros(0, 1))
    end

    @testset "PolynomialAmplitudes round-trips a polynomial exactly" begin
        C = zeros(4, 1 + K_R + K_C)
        c = [0.7, 0.5, -0.2, 0.03]           # ĝ(x) = Σ c_l x^{l-1}, positive on [0.8, 5]
        C[:, 1] .= c
        env = sqrt.(exp.(-xs))               # a stand-in for √(Λ(x,0))
        pm = PolynomialAmplitudes(C, xs, env; K_R = K_R, K_C = K_C)
        @test n_configurations(pm) == N
        @test mode_counts(pm) == (K_R, K_C)
        ĝ(x) = c[1] + c[2] * x + c[3] * x^2 + c[4] * x^3
        @test all(>(0), ĝ.(xs))
        for i in 1:N
            a₀ᵢ, aᵢ, Aᵢ = amplitudes(pm, i)
            @test sqrt(a₀ᵢ) ≈ env[i] * ĝ(xs[i])   # g = √(Λ(x,0)) · ĝ, §6 obstacle 1
            @test aᵢ == zeros(K_R) && Aᵢ == zeros(K_C)
        end
        @test length(parameters(pm)) == 4 * (1 + K_R + K_C)
        θ = randn(length(parameters(pm)))
        @test parameters(parameters!(pm, θ)) == θ
        # squares: non-negative for *any* coefficients, including the random ones above
        @test all(i -> all(≥(0), vcat(amplitudes(pm, i)...)), 1:N)
        @test_throws DimensionMismatch PolynomialAmplitudes(C, xs, env; K_R = 1, K_C = 1)
        @test_throws DimensionMismatch PolynomialAmplitudes(C, xs, env[1:end-1]; K_R, K_C)
        @test_throws ArgumentError PolynomialAmplitudes(C, xs, -env; K_R, K_C)
    end

    @testset "ChebyshevAmplitudes round-trips a smooth function spectrally" begin
        # nodes in the warped coordinate, i.e. what warp_coordinate produces
        s = collect(range(-1.4, 1.5; length = N))
        t(σ) = 2σ / π
        f(u) = exp(u)                        # positive, entire ⇒ spectral convergence
        # Chebyshev–Gauss quadrature for the coefficients of f on [-1, 1]
        function chebcoeffs(f, d; M = 4d + 16)
            nodes = [cos((k - 0.5) * π / M) for k in 1:M]
            c = [(2 / M) * sum(f(u) * cos(l * acos(u)) for u in nodes) for l in 0:d]
            c[1] /= 2
            return c
        end
        # exact: a known Chebyshev combination is reproduced to roundoff
        cex = [0.4, 0.25, -0.1]
        C = zeros(3, 1 + K_R + K_C)
        C[:, 2] .= cex
        ch = ChebyshevAmplitudes(C, s, ones(N); K_R = K_R, K_C = K_C)
        for i in 1:N
            u = t(s[i])
            @test sqrt(amplitudes(ch, i)[2][1]) ≈
                  cex[1] + cex[2] * u + cex[3] * (2u^2 - 1)
        end
        # approximate: geometric convergence, machine precision by degree 16
        for (d, tol) in ((4, 1e-3), (8, 1e-7), (16, 1e-14))
            C = zeros(d + 1, 1 + K_R + K_C)
            C[:, 1] .= chebcoeffs(f, d)
            ch = ChebyshevAmplitudes(C, s, ones(N); K_R = K_R, K_C = K_C)
            err = maximum(abs(sqrt(amplitudes(ch, i)[1]) - f(t(s[i]))) for i in 1:N)
            @test err ≤ tol
        end
        # envelope and interface
        env = sqrt.(exp.(-xs))
        C = zeros(5, 1 + K_R + K_C)
        ch = ChebyshevAmplitudes(C, s, env; K_R = K_R, K_C = K_C)
        @test n_configurations(ch) == N
        @test mode_counts(ch) == (K_R, K_C)
        θ = randn(5 * (1 + K_R + K_C))
        @test parameters(parameters!(ch, θ)) == θ
        @test all(i -> all(≥(0), vcat(amplitudes(ch, i)...)), 1:N)
        @test spectral_density(0.4, ps, ch, 3) ==
              spectral_density(0.4, ps, amplitudes(ch, 3)...)
        @test memory_kernel(0.4, ps, ch, 3) ==
              memory_kernel(0.4, ps, amplitudes(ch, 3)[2:3]...)
        @test_throws DimensionMismatch parameters!(ch, θ[1:end-1])
    end
end

@testset "fit_tag" begin
    # Filenames must distinguish representations: the same ladder fitted as a grid and
    # as a Chebyshev expansion are different models and must not collide on disk.
    @test fit_tag(GridAmplitudes) == "stage1Grid"
    @test fit_tag(PolynomialAmplitudes) == "stage2Poly"
    @test fit_tag(ChebyshevAmplitudes) == "stage2Cheby"
    @test allunique(fit_tag.((GridAmplitudes, PolynomialAmplitudes, ChebyshevAmplitudes)))

    # The instance method must agree with the type method — that is the whole point of
    # deriving the tag rather than writing a literal into each script.
    K_R, K_C = 2, 3
    xs = range(0.8, 3.0; length = 6)
    C = zeros(4, 1 + K_R + K_C)
    env = ones(length(xs))
    for m in (GridAmplitudes(length(xs), K_R, K_C),
              PolynomialAmplitudes(C, collect(xs), env; K_R = K_R, K_C = K_C),
              ChebyshevAmplitudes(C, warp_coordinate.(collect(xs) .- 2.0, 0.05, 0.001),
                                  env; K_R = K_R, K_C = K_C))
        @test fit_tag(m) == fit_tag(typeof(m))
    end
end
