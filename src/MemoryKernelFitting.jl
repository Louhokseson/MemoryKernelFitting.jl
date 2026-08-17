"""
    MemoryKernelFitting

Fitting underdamped Prony-series memory kernels to configuration-resolved spectral
data, for quasi-Markovian generalised Langevin equations with configuration-dependent
electronic friction.

The model is the boxed equation of `docs/PronySeries/Prony_fit_optimisation_problem.md`
§1: a white part, a set of zero-centred Lorentzians, and a set of *mirrored* Lorentzian
pairs. The pairs are mandatory — the data has a resonance at nonzero `ω`, and a sum of
zero-centred Lorentzians is strictly decreasing.

Frequency domain ([`spectral_density`](@ref)):

```math
Λ(ω) = a₀ + \\sum_k a_k \\frac{λ_k}{λ_k² + ω²} + \\sum_j A_j L(ω; α_j, β_j)
```

Time domain ([`memory_kernel`](@ref)):

```math
K(τ) = 2a₀ δ(τ) + \\sum_k a_k e^{-λ_k τ} + \\sum_j A_j e^{-α_j τ} \\cos(β_j τ)
```

The two are a one-sided cosine-transform pair. All amplitudes are non-negative — they
are squares of the fitted `g(q)` — and all rates are positive, so that `Γ₂,₂` is stable.
"""
module MemoryKernelFitting

export spectral_density, memory_kernel, lorentzian, mirrored_lorentzian

"""
    lorentzian(ω, λ) → λ / (λ² + ω²)

The zero-centred (overdamped) Lorentzian: the one-sided cosine transform of
`exp(-λτ)`. Strictly decreasing in `|ω|`, and flat over the whole window once
`λ ≫ ω_max` — at which point it is indistinguishable from the white part `a₀` and
should be folded into it rather than fitted.

Corresponds to a `1×1` block `λ` of `Γ₂,₂`.
"""
lorentzian(ω, λ) = λ / (λ^2 + ω^2)

"""
    mirrored_lorentzian(ω, α, β) → L(ω; α, β)

The mirrored (underdamped) Lorentzian pair

```math
L(ω; α, β) = \\frac{α}{2}\\left[\\frac{1}{α² + (ω-β)²} + \\frac{1}{α² + (ω+β)²}\\right],
```

the one-sided cosine transform of `exp(-ατ) cos(βτ)`: peaks of half-width `α` at
`ω = ±β`. The mirror is not a modelling choice — it is realness of `Γ₂,₂`, whose
complex eigenvalues `α ± iβ` must come in conjugate pairs.

Corresponds to a `2×2` block `R(α, β) = [α β; -β α]` of `Γ₂,₂`.

The shape is only a resonance when `β > α/√3`; below that the pair merges into a
monotone, zero-centred shape. That is the desired behaviour at a Fermi crossing,
where `β → 0` while `α` stays finite.
"""
mirrored_lorentzian(ω, α, β) =
    (α / 2) * (1 / (α^2 + (ω - β)^2) + 1 / (α^2 + (ω + β)^2))

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

end # module
