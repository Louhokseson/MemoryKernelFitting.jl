#
# Readout: fitted model → the extended-variable blocks the QGLE integrates.
#
# `Prony_fit_optimisation_problem.md` §1. This is the step where a wrong factor would be
# invisible — it produces a stable, plausible SDE that samples the wrong temperature —
# so every block below is tied back to the equation it comes from.
#

"""
    QGLEBlocks(Γ₁₁, g, Γ₂₂, Σ₂)

The blocks of the quasi-Markovian GLE realising a fitted spectrum at one configuration
(`Prony_fit_optimisation_problem.md` §1), with `n = 1` physical degree of freedom and
`m = ` [`auxiliary_dimension`](@ref) extended variables.

| field | symbol | size |
|---|---|---|
| `Γ₁₁` | `Γ̃₁,₁ = a₀` | scalar |
| `g` | `Γ̃₂,₁ = g`, `Γ̃₁,₂ = -gᵀ` | `m` |
| `Γ₂₂` | `blockdiag(λ₁,…,λ_{K_R}, R(α₁,β₁),…)` | `m × m` |
| `Σ₂` | diagonal of `Σ̃₂` | `m` |

Build with [`qgle_blocks`](@ref); assemble the full `(1+m) × (1+m)` drift with
[`gamma_matrix`](@ref).
"""
struct QGLEBlocks{T<:Real}
    Γ₁₁::T
    g::Vector{T}
    Γ₂₂::Matrix{T}
    Σ₂::Vector{T}
end

"""
    qgle_blocks(ps::PoleSet, a₀, a, A) → QGLEBlocks
    qgle_blocks(ps::PoleSet, m::AmplitudeModel, i) → QGLEBlocks

Build the extended-variable blocks of `Prony_fit_optimisation_problem.md` §1 from a
pole set and one configuration's amplitudes:

```math
Γ₂,₂ = \\operatorname{blockdiag}\\big(λ_1,…,λ_{K_R},\\ R(α_1,β_1),…,R(α_{K_C},β_{K_C})\\big),
\\qquad R(α,β) = \\begin{pmatrix} α & β \\\\ -β & α\\end{pmatrix},
```

```math
\\widetilde{Γ}_{1,1} = a_0, \\qquad
g = \\big(\\sqrt{a_1},…,\\sqrt{a_{K_R}},\\ \\sqrt{A_1},0,\\ …,\\ \\sqrt{A_{K_C}},0\\big)^T,
\\qquad \\widetilde{Γ}_{2,1} = g,\\ \\ \\widetilde{Γ}_{1,2} = -g^T .
```

Three things this encodes that are easy to get wrong:

  - **The zero in the second slot of each pair.** Only the first auxiliary component of
    a pair couples to the physical variable. This is not an approximation: it is what
    makes `gᵀ e^{-τR(α,β)} g = |g|² e^{-ατ} cos(βτ)`, one non-negative amplitude per
    `2×2` block, and it agrees with Stella, Lorenz & Kantorovich Eqs. (31)/(35), where
    the force involves `s^{(k)}_1` only (Appendix A.3, agreement 1).
  - **The square root.** Amplitudes enter the drift as `√a`, which is why §6 fits
    `g = √a` and not `a` — see [`AmplitudeModel`](@ref).
  - **`Σ̃₂ = √(2α) I₂` per pair, `√(2λ)` per real mode**, with `Q = I`. The FDT `(2,2)`
    block needs `Γ₂,₂ + Γ₂,₂ᵀ = 2 diag(rates)`, which the rotation blocks satisfy
    because their antisymmetric part cancels — so the noise is *isotropic* within each
    block and no Cholesky is needed (Appendix A.1(iii)). The refinement of Appendix B
    applies: `Q = I` survives here because we never diagonalise the `2×2` blocks; the
    requirement is `Γ₂,₂Q + QΓ₂,₂ᵀ ≻ 0`, not symmetry of `Γ₂,₂`.
"""
function qgle_blocks(ps::PoleSet, a₀, a, A)
    K_R, K_C = length(ps.λ), length(ps.α)
    length(a) == K_R || throw(DimensionMismatch(
        "expected $K_R real-mode amplitudes, got $(length(a))"))
    length(A) == K_C || throw(DimensionMismatch(
        "expected $K_C pair amplitudes, got $(length(A))"))
    a₀ ≥ 0 || throw(ArgumentError("the white amplitude a₀ must be non-negative"))

    T = float(promote_type(eltype(ps.λ), eltype(a), eltype(A), typeof(a₀)))
    m = K_R + 2K_C
    Γ₂₂ = zeros(T, m, m)
    g = zeros(T, m)
    Σ₂ = zeros(T, m)

    for k in 1:K_R
        Γ₂₂[k, k] = ps.λ[k]
        g[k] = sqrt(a[k])
        Σ₂[k] = sqrt(2 * ps.λ[k])
    end
    for j in 1:K_C
        p = K_R + 2j - 1                      # first row/column of pair j's 2×2 block
        αⱼ, βⱼ = ps.α[j], ps.β[j]
        Γ₂₂[p, p] = αⱼ
        Γ₂₂[p, p+1] = βⱼ
        Γ₂₂[p+1, p] = -βⱼ
        Γ₂₂[p+1, p+1] = αⱼ
        g[p] = sqrt(A[j])                     # g[p+1] stays 0 — only s₁ couples
        Σ₂[p] = Σ₂[p+1] = sqrt(2αⱼ)
    end
    return QGLEBlocks{T}(a₀, g, Γ₂₂, Σ₂)
end

qgle_blocks(ps::PoleSet, m::AmplitudeModel, i::Integer) =
    qgle_blocks(ps, amplitudes(m, i)...)

"""
    gamma_matrix(b::QGLEBlocks) → Γ̃

Assemble the full `(1+m) × (1+m)` drift

```math
\\widetilde{Γ} = \\begin{pmatrix} a_0 & -g^T \\\\ g & Γ_{2,2}\\end{pmatrix}.
```

The antisymmetric off-diagonal pairing `Γ̃₁,₂ = -Γ̃₂,₁ᵀ` is what makes

```math
\\widetilde{Γ} + \\widetilde{Γ}^T = \\operatorname{diag}(2a_0,\\ 2λ_1,…,\\ 2α_1, 2α_1,\\ …) \\succeq 0,
```

hence `‖e^{-Δt Γ̃}‖₂ ≤ 1` and the existence of the noise factor `S_Δt` of thesis
Eq. (4.7) — Appendix A.1(iii). The FDT pairing is therefore not only a modelling
constraint; it is exactly what makes the exact O-step well defined.

Only needed for analysis and for the constant-coefficient gle-BAOAB validation runs.
Production AR-2 never forms this matrix: it touches `Γ₂,₂` (block-diagonal, closed-form
exponential) and applies `g` as a linear kick, `O(m)` per step (Appendix A.2).
"""
function gamma_matrix(b::QGLEBlocks{T}) where {T}
    m = length(b.g)
    Γ = zeros(T, m + 1, m + 1)
    Γ[1, 1] = b.Γ₁₁
    for i in 1:m
        Γ[1, 1+i] = -b.g[i]
        Γ[1+i, 1] = b.g[i]
        for j in 1:m
            Γ[1+i, 1+j] = b.Γ₂₂[i, j]
        end
    end
    return Γ
end

"""
    exp_gamma22(ps::PoleSet, Δt) → e^{-Δt Γ₂,₂}

The propagator of the auxiliary variables, in closed form — Appendix A.1(ii). Block by
block, with `R = αI + βJ` and `J = [0 1; -1 0]`, `J² = -I`, so that `J` acts as the
imaginary unit and the exponential series collapses to a damped rotation:

```math
e^{-Δt λ_k}, \\qquad
e^{-Δt R(α,β)} = e^{-αΔt}\\big[\\cos(βΔt)\\,I - \\sin(βΔt)\\,J\\big]
= e^{-αΔt}\\begin{pmatrix}\\cos βΔt & -\\sin βΔt \\\\ \\sin βΔt & \\cos βΔt\\end{pmatrix}.
```

Complex eigenvalues never enter the arithmetic — a *real* matrix is exponentiated — and
no `expm` call is needed at runtime. Both singular values of a pair block equal
`e^{-αΔt} < 1`, so it is a strict contraction and the O-step noise is isotropic,
`S_Δt = √(1 - e^{-2αΔt}) I₂`.

This is the whole reason AR-2 is the production integrator at `m ≈ 58`: `Γ₂,₂` is
block-diagonal with `1×1` and `2×2` blocks, so the step costs `O(m)` rather than the
`O(m³)` of a general `expm` (Appendix A.2).
"""
function exp_gamma22(ps::PoleSet, Δt)
    K_R, K_C = length(ps.λ), length(ps.α)
    T = float(promote_type(eltype(ps.λ), typeof(Δt)))
    m = K_R + 2K_C
    E = zeros(T, m, m)
    for k in 1:K_R
        E[k, k] = exp(-ps.λ[k] * Δt)
    end
    for j in 1:K_C
        p = K_R + 2j - 1
        d = exp(-ps.α[j] * Δt)
        c, s = cos(ps.β[j] * Δt), sin(ps.β[j] * Δt)
        E[p, p] = d * c
        E[p, p+1] = -d * s
        E[p+1, p] = d * s
        E[p+1, p+1] = d * c
    end
    return E
end
