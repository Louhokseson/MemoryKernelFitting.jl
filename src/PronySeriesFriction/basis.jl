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

!!! warning "Both terms are needed even though the data only covers ω ≥ 0"
    The second term is not a separate basis function that may be dropped because the
    fit never evaluates negative `ω`. Both terms fall out of the *same* integral
    `∫₀^∞ exp(-ατ)cos(βτ)cos(ωτ) dτ` at positive `ω`. Keeping only the peak at `+β`
    would fit a function that is not the transform of any real Prony mode, and the
    readout to `Γ₂,₂` would then be inconsistent with the fitted spectrum. The size
    of the effect on `ω ≥ 0` is `α²/4β²` (≈ 1% at `α = 0.2β`) near the peak, but a
    factor of two at `ω = 0`, where the two terms are exactly equal — and `Λ(x, 0)`
    is the Markovian anchor of `Prony_fit_optimisation_problem.md` §4.
"""
mirrored_lorentzian(ω, α, β) =
    (α / 2) * (1 / (α^2 + (ω - β)^2) + 1 / (α^2 + (ω + β)^2))
