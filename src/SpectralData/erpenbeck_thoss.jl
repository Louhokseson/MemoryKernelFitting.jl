"""
    LambdaSpectrum

The contents of one `Lambda_spectrum_*.h5` file — the training set `𝒟_Γ` of
`Prony_fit_optimisation_problem.md` §0 — in the units the package works in.

One file per hybridisation strength `Γ`, and each file is a **separate model, hence a
separate fit**; never pool them.

| field | unit | contents |
|---|---|---|
| `x` | Å | configuration grid, **graded**, strictly ascending |
| `ω` | Hartree | frequency grid, uniform, from `omega_eV` |
| `Λ` | atomic (`mₑ/aut`) | `Λ(xᵢ, ωⱼ)` as `[N_ω, N_x]`, unchanged from file |

Throughout the package `i` indexes **configurations** and `j` indexes **frequencies**, so
`Λ` runs frequency down a column and configuration across.
| `h` | Hartree | level position `h(x)`, one per column |
| `Δ` | Hartree | hybridisation width `Δ(x)`, one per column |
| `Γ` | Hartree | hybridisation strength, the `Gamma_eV` attribute |
| `kT` | Hartree | thermal energy, from the `T_K` attribute |
| `T_K` | K | temperature, as stored |
| `x_resonance` | Å | Fermi crossing `h(x₀) = 0`, as stored |

`x` stays in Ångström on purpose: it is a configuration coordinate, never combined with
`ω` in any formula, and keeping it in Å preserves the comparison to every figure in
`docs/PronySeries/`. See `docs/design/0001-units-and-data-conventions.md`.

The `x` grid is graded by a factor of ~290 in `dx`, so **any sum over columns needs the
quadrature weights of [`trapezoid_weights`](@ref)**, and the `ω` grid is finer than the
kernel by orders of magnitude, so residuals are evaluated on the subset returned by
[`subsample_omega`](@ref).
"""
struct LambdaSpectrum
    x::Vector{Float64}
    ω::Vector{Float64}
    Λ::Matrix{Float64}
    h::Vector{Float64}
    Δ::Vector{Float64}
    Γ::Float64
    kT::Float64
    T_K::Float64
    x_resonance::Float64
end

"""
    load_lambda_spectrum(path; xmin = 0.8) → LambdaSpectrum

Read a `Lambda_spectrum_*.h5` file and convert it to atomic units.

The file stores `Lambda_au` in atomic units but `omega_eV`, `h_eV`, `Delta_eV` and the
`Gamma_eV` attribute in electronvolts. This is the one place the eV → Hartree factor
appears (`docs/design/0001-units-and-data-conventions.md`); nothing downstream sees an
eV. The `T_K` attribute becomes `kT` in Hartree via [`boltzmann_energy`](@ref).

**Columns with `x < xmin` are dropped**, `xmin = 0.8` Å by default, which keeps 254 of
the 257 columns. `Prony_fit_optimisation_problem.md` §4: below 0.8 Å the level position
plunges — `h(0.7) = -23` eV, `h(0.5) = -90` eV — so the resonance at `ω_pk ≈ |h|` leaves
the 60 eV window and fitting those columns is unconstrained extrapolation. For
`x ≥ 0.8` Å, `|h| ≤ 8.65` eV, comfortably inside the window.

Only the retained columns are read off disk; the full `[60000, 257]` matrix is 123 MB.
Assumes `x_Ang` is strictly ascending, as the file's `x_grid_note` attribute promises.
"""
function load_lambda_spectrum(path; xmin = 0.8)
    h5open(path, "r") do f
        x = read(f["x_Ang"])
        keep = searchsortedfirst(x, xmin):length(x)
        T_K = read_attribute(f, "T_K")
        return LambdaSpectrum(
            x[keep],
            eV_to_hartree.(read(f["omega_eV"])),
            f["Lambda_au"][:, keep],
            eV_to_hartree.(read(f["h_eV"])[keep]),
            eV_to_hartree.(read(f["Delta_eV"])[keep]),
            eV_to_hartree(read_attribute(f, "Gamma_eV")),
            boltzmann_energy(T_K),
            T_K,
            read_attribute(f, "x_resonance_Ang"),
        )
    end
end

"""
    trapezoid_weights(x) → v

Trapezoidal quadrature weights on a non-uniform grid,

```math
v_i = \\tfrac12 (x_{i+1} - x_{i-1}),
```

with the one-sided endpoints `v₁ = ½(x₂ - x₁)` and `v_N = ½(x_N - x_{N-1})`. They
telescope, so `sum(v) == x[end] - x[1]` exactly up to rounding.

These are the `vᵢ` of `Prony_fit_optimisation_problem.md` §3. **The subscript is the
configuration index `i`, not the frequency index `j`** — a point worth stating because
the two grids behave oppositely here: `ω` is uniform (`dω = 1e-3` eV exactly, so it needs
no quadrature weight at all), while `x` is graded by a factor of 290 and cannot be summed
without one. They are
**non-negotiable**: the configuration grid is graded towards the resonance, with 214 of
the 257 columns inside 0.15 Å of `x₀` — 83% of the columns covering 6.7% of the range.
With `vᵢ ≡ 1` the objective is the resonance alone and the rest of the potential energy
surface contributes nothing.
"""
function trapezoid_weights(x::AbstractVector)
    N = length(x)
    N ≥ 2 || throw(ArgumentError("need at least two grid points, got $N"))
    v = similar(x, float(eltype(x)))
    for i in 2:(N - 1)
        v[i] = (x[i + 1] - x[i - 1]) / 2
    end
    v[1] = (x[2] - x[1]) / 2
    v[N] = (x[N] - x[N - 1]) / 2
    return v
end

# Index of the entry of the ascending grid `ω` closest to `t`.
function _nearest_index(ω, t)
    j = searchsortedfirst(ω, t)
    j > length(ω) && return length(ω)
    j == 1 && return 1
    return t - ω[j - 1] < ω[j] - t ? j - 1 : j
end

"""
    subsample_omega(ω; n = 800) → indices

Indices of a logarithmically spaced subset of the frequency grid — the retained band
`ℬᵢ` of `Prony_fit_optimisation_problem.md` §3.

Indices rather than values, so that `Λ` can be sliced with the same object:
`s.Λ[subsample_omega(s.ω), :]`.

All 60000 frequencies per column are redundant: the model is smooth on the scale of the
narrowest pole, `α ≳ 0.05` eV, which is 50 grid steps of `dω = 10⁻³` eV. Logarithmic
spacing is what resolves both the `ω → 0` Markovian limit and the `ω⁻²` tail. **Index 1
is always retained** — `ω₁ ≈ 0` carries the Markovian anchor `Λ(x, 0)` of §4, the single
most physically important number in the fit.

The result is index 1 plus `n - 1` logarithmically spaced points, so `n ≥ 3`. It is
strictly increasing and unique, and has *at most* `n` entries: below
`ω ≈ 0.04` eV the log spacing is finer than the grid itself and neighbouring targets
collapse onto the same point, so the real grid yields 561 indices for the default
`n = 800`, comfortably inside the 500–1000 per column that §3 asks for.
"""
function subsample_omega(ω::AbstractVector; n::Integer = 800)
    length(ω) ≥ 2 || throw(ArgumentError("need at least two frequencies"))
    n ≥ 3 || throw(ArgumentError("need at least three subsample points, got $n"))
    # Span [ω₂, ω_max]: ω₁ is the anchor, appended below, and is 0 on some grids.
    targets = exp.(range(log(ω[2]), log(ω[end]); length = n - 1))
    idx = [_nearest_index(ω, t) for t in targets]
    push!(idx, 1)
    return unique!(sort!(idx))
end
