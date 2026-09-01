@inline function _check(a, λ, A, α, β)
    length(a) == length(λ) ||
        throw(DimensionMismatch("real-mode amplitudes and rates must have equal length"))
    length(A) == length(α) == length(β) ||
        throw(DimensionMismatch("pair amplitudes, decay rates and frequencies must have equal length"))
    return nothing
end

"""
    spectral_density(ω, a₀, a, λ[, A, α, β]) → Λ

Evaluate the fit model at frequency `ω`:

```math
Λ(ω) = a₀ + \\sum_{k=1}^{K_R} a_k \\frac{λ_k}{λ_k² + ω²}
           + \\sum_{j=1}^{K_C} A_j L(ω; α_j, β_j).
```

`a₀ ≥ 0` is the white (instantaneous) friction `Γ̃₁,₁`, `a[k] ≥ 0` and `λ[k] > 0` the
overdamped modes, and `A[j] ≥ 0`, `α[j] > 0`, `β[j] ≥ 0` the underdamped pairs. Omitting
the pair arguments gives the overdamped-only model, which cannot represent a resonance.

At each configuration `x` this is what is fitted to `Λ(x, ·)`; it is linear in the
amplitudes and nonlinear only in the poles, which is what makes variable projection
applicable.

The zero-frequency value is the Markovian friction a standard MDEF simulation would use:
`Λ(0) = a₀ + Σ aₖ/λₖ + Σ Aⱼ αⱼ/(αⱼ² + βⱼ²)`.
"""
function spectral_density(ω, a₀, a, λ, A, α, β)
    _check(a, λ, A, α, β)
    Λ = float(a₀)
    for (aₖ, λₖ) in zip(a, λ)
        Λ += aₖ * lorentzian(ω, λₖ)
    end
    for (Aⱼ, αⱼ, βⱼ) in zip(A, α, β)
        Λ += Aⱼ * mirrored_lorentzian(ω, αⱼ, βⱼ)
    end
    return Λ
end

spectral_density(ω, a₀, a, λ) =
    spectral_density(ω, a₀, a, λ, similar(a, 0), similar(a, 0), similar(a, 0))

"""
    memory_kernel(τ, a, λ[, A, α, β]) → K

Evaluate the underdamped Prony series

```math
K(τ) = \\sum_k a_k e^{-λ_k τ} + \\sum_j A_j e^{-α_j τ} \\cos(β_j τ)
```

at lag `τ > 0`. This is the regular part of the kernel; the singular `2a₀ δ(τ)`
contribution is excluded, since it enters the dynamics as the instantaneous friction
`Γ̃₁,₁` rather than through the convolution.

`K` is the cosine transform of [`spectral_density`](@ref) less its white part, and
`K(0⁺) = Σ aₖ + Σ Aⱼ`.
"""
function memory_kernel(τ, a, λ, A, α, β)
    _check(a, λ, A, α, β)
    K = zero(float(τ))
    for (aₖ, λₖ) in zip(a, λ)
        K += aₖ * exp(-λₖ * τ)
    end
    for (Aⱼ, αⱼ, βⱼ) in zip(A, α, β)
        K += Aⱼ * exp(-αⱼ * τ) * cos(βⱼ * τ)
    end
    return K
end

memory_kernel(τ, a, λ) =
    memory_kernel(τ, a, λ, similar(a, 0), similar(a, 0), similar(a, 0))
