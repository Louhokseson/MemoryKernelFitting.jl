# The analytic model behind the data. Everything in `erpenbeck_thoss.jl` reads `h` and
# `Δ` off disk at 257 stored nodes; this file evaluates them anywhere, and differentiates
# them. See the `ErpenbeckThossModel` docstring for why that matters.

"""
    ErpenbeckThossModel(; Γ_eV, kwargs...)
    ErpenbeckThossModel(s::LambdaSpectrum)

The analytic level position `h(x)` and hybridisation width `Δ(x)` behind every
`Lambda_spectrum_*.h5` file — a single molecular orbital on a Morse-bound adsorbate
coupled to a wide-band metal, J. Chem. Phys. **158**, 064101 (2023),
[doi:10.1063/5.0137137](https://doi.org/10.1063/5.0137137).

```math
U_0(x) = D_e\\big(e^{-a(x-x_0)} - 1\\big)^2 + c, \\qquad
U_1(x) = D_1 e^{-2a'(x-x_0')} - D_2 e^{-a'(x-x_0')} + V_∞,
```
```math
h(x) = U_1(x) - U_0(x), \\qquad
V(x) = \\sqrt{Γ/2π}\\left[\\tfrac{1-q}{2}\\Big(1 - \\tanh\\tfrac{x-\\tilde{x}}{\\tilde{a}}\\Big) + q\\right],
\\qquad Δ(x) = π V(x)^2 .
```

`h` is the four-exponential combination `Prony_fit_optimisation_problem.md` §6 names as
the natural variable for the rates, and `Δ` the wide-band hybridisation width — the
energy-independent limit of `Δ(ε) = π Σ_k |V_k|² δ(ε-ε_k)`, with the `x` dependence
carried entirely by the coupling `V(x)`.

# Why this exists when the data already carries `h` and `Δ`
Three things the stored columns cannot do:

  - **Evaluate off-grid.** A regime-(B) pole law is `β_l(q) = |c_l||h(q)| + |d_l|`, and a
    QGLE integrator evaluates it at a continuously moving `q(t)`, not at 257 nodes. This
    is what takes [`TrackingPoles`](@ref) from *fittable* to *runnable*.
  - **Differentiate exactly.** The adiabatic condition of §6 is
    `τ_mem/τ_conf = ħ|dh/dx||v|/α²`, so the legality of regime (B) is a statement about
    `dh/dx`. [`dlevel_position`](@ref) makes it a curve rather than the two quoted
    numbers (`15.2` thermal, `133` at 1 eV at the crossing).
  - **Locate the crossing.** [`fermi_crossing`](@ref) solves `h(x₀) = 0` to machine
    precision instead of taking the nearest stored node.

# Units
Energies in **Hartree**, lengths in **Å** — the [`LambdaSpectrum`](@ref) convention of
`docs/design/0001`, so the values returned here drop straight into `poleset`,
`warp_coordinate` and the rest with no conversion. The keyword constructor takes the
paper's numbers in **eV and Å** and converts once, exactly as
[`load_lambda_spectrum`](@ref) does for the file.

!!! warning "The derivatives carry a mixed unit"
    [`dlevel_position`](@ref) and [`dhybridisation_width`](@ref) return **Hartree per
    Å**, because that is what `x` in Å and energies in Hartree implies. Atomic-unit
    formulas such as `ħ|dh/dx||v|/α²` need `dh/dx` per *bohr*: multiply by
    [`BOHR_IN_ANGSTROM`](@ref). This is the one place in the package where a length and
    an energy meet, which is why `docs/design/0001` could leave `x` in Å everywhere else.

# Parameters
Defaults are the reference's, in its units. `x₀′` defaults to `x₀` and `c` to `-E₀`, the
Morse zero-point of [`morse_zero_point`](@ref) — the file data was generated with both
defaults, so overriding either gives a *different model*, not a re-parametrisation: `c`
shifts `h` rigidly and therefore moves `x₀`.

| keyword | default | | keyword | default |
|---|---|---|---|---|
| `Dₑ_eV` | `3.52` | | `q` | `0.05` |
| `x₀` | `1.78` Å | | `ã` | `0.5` Å |
| `a` | `1.7361` Å⁻¹ | | `x̃` | `3.5` Å |
| `D₁_eV` | `4.52` | | `m_u` | `1.0` |
| `D₂_eV` | `0.79` | | `x₀′` | `x₀` |
| `a′` | `1.379` Å⁻¹ | | `c_eV` | `-morse_zero_point` |
| `V∞_eV` | `-1.5` | | | |

`Γ_eV` has no default: it is the one parameter that varies between files, and picking
one silently would be picking a dataset. `ErpenbeckThossModel(s::LambdaSpectrum)` reads
it from the loaded spectrum, which is the constructor to prefer.

`test/test_erpenbeck_thoss_model.jl` asserts this reproduces the stored `h_eV` and
`Delta_eV` on all seven `Γ` files.
"""
struct ErpenbeckThossModel{T<:Real}
    Γ::T          # Hartree
    Dₑ::T         # Hartree
    x₀::T         # Å
    a::T          # Å⁻¹
    D₁::T         # Hartree
    D₂::T         # Hartree
    x₀′::T        # Å
    a′::T         # Å⁻¹
    V∞::T         # Hartree
    c::T          # Hartree — the U₀ offset
    q::T          # dimensionless
    ã::T          # Å
    x̃::T          # Å

    function ErpenbeckThossModel{T}(Γ, Dₑ, x₀, a, D₁, D₂, x₀′, a′, V∞, c, q, ã, x̃) where {T<:Real}
        Γ > 0 || throw(ArgumentError("the hybridisation strength must be positive, got $Γ"))
        Dₑ > 0 || throw(ArgumentError("the Morse well depth must be positive, got $Dₑ"))
        a > 0 && a′ > 0 || throw(ArgumentError(
            "the range parameters a, a′ must be positive, got $a and $a′"))
        ã > 0 || throw(ArgumentError("the switching width ã must be positive, got $ã"))
        # q is the floor of the coupling switch: Δ(∞) = (Γ/2)q². Outside [0, 1] the
        # switch is no longer monotone between Γ/2 and its floor, which is the one
        # structural property `hybridisation_width` documents.
        0 ≤ q ≤ 1 || throw(ArgumentError("the coupling floor q must lie in [0, 1], got $q"))
        return new{T}(Γ, Dₑ, x₀, a, D₁, D₂, x₀′, a′, V∞, c, q, ã, x̃)
    end
end

"""
    morse_zero_point(Dₑ, a, m) → E₀ in Hartree

Ground-state energy of the Morse well `Dₑ(e^{-a(x-x₀)}-1)²` measured from its minimum,

```math
E_n = ω_0\\left[(n+\\tfrac12) - \\tfrac{1}{2λ}(n+\\tfrac12)^2\\right], \\qquad
ω_0 = \\sqrt{2D_e a^2/m}, \\quad λ = \\sqrt{2mD_e}/a,
```

at `n = 0` (Eqs. 36 and 40 of the reference). `Dₑ` in Hartree, `a` in **Å⁻¹**, `m` in
electron masses; the conversion of `a` to inverse bohr happens here.

This is the sole reason a mass appears anywhere in this package. It enters the model only
through the offset `c = -E₀`, which places `U₀`'s zero at the vibrational ground state
rather than at the well bottom — a rigid shift of `h`, but one that moves the Fermi
crossing, so it is not cosmetic. At the reference parameters `E₀ = 0.147336` eV.
"""
function morse_zero_point(Dₑ, a, m)
    # Checked here, not only in the constructor: this runs first, and without it a
    # negative well depth reaches `sqrt` and raises DomainError instead.
    Dₑ > 0 || throw(ArgumentError("the Morse well depth must be positive, got $Dₑ"))
    a > 0 || throw(ArgumentError("the range parameter a must be positive, got $a"))
    m > 0 || throw(ArgumentError("the mass must be positive, got $m"))
    a_bohr = a * BOHR_IN_ANGSTROM
    ω₀ = sqrt(2 * Dₑ * a_bohr^2 / m)
    λ = sqrt(2 * m * Dₑ) / a_bohr
    return ω₀ * (1 / 2 - (1 / 2)^2 / (2λ))
end

function ErpenbeckThossModel(; Γ_eV, Dₑ_eV = 3.52, x₀ = 1.78, a = 1.7361, D₁_eV = 4.52,
                             D₂_eV = 0.79, x₀′ = nothing, a′ = 1.379, V∞_eV = -1.5,
                             m_u = 1.0, q = 0.05, ã = 0.5, x̃ = 3.5, c_eV = nothing)
    Dₑ = eV_to_hartree(float(Dₑ_eV))
    c = c_eV === nothing ?
        -morse_zero_point(Dₑ, float(a), float(m_u) * ELECTRON_MASSES_PER_U) :
        eV_to_hartree(float(c_eV))
    vals = promote(eV_to_hartree(float(Γ_eV)), Dₑ, float(x₀), float(a),
                   eV_to_hartree(float(D₁_eV)), eV_to_hartree(float(D₂_eV)),
                   x₀′ === nothing ? float(x₀) : float(x₀′), float(a′),
                   eV_to_hartree(float(V∞_eV)), c, float(q), float(ã), float(x̃))
    return ErpenbeckThossModel{eltype(vals)}(vals...)
end

"""
    ErpenbeckThossModel(s::LambdaSpectrum) → ErpenbeckThossModel

The model matching a loaded spectrum, taking `Γ` from the file. Every other parameter is
the reference default, which is what generated the file — the assertion being the point
of `test/test_erpenbeck_thoss_model.jl`.
"""
ErpenbeckThossModel(s::LambdaSpectrum) = ErpenbeckThossModel(; Γ_eV = hartree_to_eV(s.Γ))

"""
    diabatic_potentials(x, m::ErpenbeckThossModel) → (U₀, U₁)

Both diabatic surfaces at `x` (Å), in Hartree: `U₀` the Morse well with the orbital
empty, `U₁` the surface with it occupied. Returned together because only their
*difference* is ever used — see [`level_position`](@ref) — and computing them apart
invites the two being evaluated at different `x`.
"""
function diabatic_potentials(x, m::ErpenbeckThossModel)
    e₀ = exp(-m.a * (x - m.x₀))
    U₀ = m.Dₑ * (e₀ - 1)^2 + m.c
    e₁ = exp(-m.a′ * (x - m.x₀′))
    U₁ = m.D₁ * e₁^2 - m.D₂ * e₁ + m.V∞
    return (U₀, U₁)
end

"""
    level_position(x, m::ErpenbeckThossModel) → h(x) in Hartree

`h(x) = U₁(x) - U₀(x)`, the molecular level relative to the Fermi energy, at `x` in Å.

Negative below the Fermi level, positive above, and zero at the crossing
[`fermi_crossing`](@ref) returns. `|h|` is the energy of the electron–hole excitation the
friction spectrum resonates at (`ω_pk = 1.02`–`1.15 |h|`, §0 fact 1), which is why the
regime-(B) pole law reads `β_l(x) = |c_l||h(x)| + |d_l|`.

Agrees with the stored `h_eV` to `2.1e-13` eV on all 257 nodes of all seven files.
"""
level_position(x, m::ErpenbeckThossModel) =
    ((U₀, U₁) = diabatic_potentials(x, m); U₁ - U₀)

"""
    dlevel_position(x, m::ErpenbeckThossModel) → dh/dx in **Hartree per Å**

Analytic derivative of [`level_position`](@ref) — see the type's unit warning before
combining it with anything in atomic units.

This is the quantity regime (B)'s legality turns on: `τ_mem/τ_conf = ħ|dh/dx||v|/α²`, and
`|dh/dx| = 8.39` eV/Å at the Fermi crossing is what makes the ratio `15.2` at thermal
velocities there (§9.7). Matches a central difference on the stored grid to `1e-6` eV/Å.
"""
function dlevel_position(x, m::ErpenbeckThossModel)
    e₀ = exp(-m.a * (x - m.x₀))
    dU₀ = 2 * m.Dₑ * (e₀ - 1) * (-m.a) * e₀
    e₁ = exp(-m.a′ * (x - m.x₀′))
    dU₁ = -2 * m.a′ * m.D₁ * e₁^2 + m.a′ * m.D₂ * e₁
    return dU₁ - dU₀
end

"""
    coupling(x, m::ErpenbeckThossModel) → V(x) in Hartree

The molecule–metal coupling `V(x) = √(Γ/2π)[(1-q)/2 (1 - tanh((x-x̃)/ã)) + q]`, a
monotone switch from `√(Γ/2π)` at the surface to `q√(Γ/2π)` far away.
[`hybridisation_width`](@ref) is `πV²`; this is exposed because the *square* hides the
switch's simple shape.
"""
coupling(x, m::ErpenbeckThossModel) =
    sqrt(m.Γ / 2π) * ((1 - m.q) / 2 * (1 - tanh((x - m.x̃) / m.ã)) + m.q)

"""
    hybridisation_width(x, m::ErpenbeckThossModel) → Δ(x) in Hartree

`Δ(x) = πV(x)²`, the wide-band level broadening: an electron in the orbital escapes to
the metal in `ħ/Δ`.

# The shape, and why `Δ` carries almost no information in the fit
The `tanh` gives a switch **centred at `x̃ = 3.5` Å with width `ã = 0.5` Å**, between

```math
Δ(-∞) = Γ/2 \\qquad\\text{and}\\qquad Δ(+∞) = (Γ/2)\\,q^2 = Γ/800 .
```

The Fermi crossing sits at `x₀ = 2.02` Å — three switch widths to the *left* — so `Δ` is
still on its plateau across the entire region the friction peak occupies: measured
`0.04975` Ha-equivalent-eV against `Γ/2 = 0.05` at `Γ = 0.1`, and `Δ` only falls below
half its plateau beyond `x ≈ 3.5` Å.

Two consequences already on record. It is why `α_l = u_l|h| + w_lΔ + z_l k_BT` floors at
`Δ + k_BT ≈ 0.076` eV at the crossing (measured HWHM `0.077`); and it is why `w_lΔ` and
`z_l k_BT` are nearly collinear there — both are constants in `x` over that region, with
ratio `≈ 2` — which is the pathology [`TiedLaw`](@ref) removes by sharing `z`.
"""
hybridisation_width(x, m::ErpenbeckThossModel) = π * coupling(x, m)^2

"""
    dhybridisation_width(x, m::ErpenbeckThossModel) → dΔ/dx in **Hartree per Å**

Analytic derivative of [`hybridisation_width`](@ref); see the type's unit warning.

Small over the resonance region for the reason that docstring gives, but *measurably* so
rather than zero: `dlnΔ/dx = -0.0203` per Å at the crossing, i.e. `Δ` drifts by about 2 %
per Å there, against `dh/dx = -8.39` eV/Å for the level position. It is the ratio of
those two that makes `h` the variable the pole law tracks.
"""
function dhybridisation_width(x, m::ErpenbeckThossModel)
    V̄ = sqrt(m.Γ / 2π)
    V = V̄ * ((1 - m.q) / 2 * (1 - tanh((x - m.x̃) / m.ã)) + m.q)
    dV = -V̄ * (1 - m.q) / 2 * sech((x - m.x̃) / m.ã)^2 / m.ã
    return 2π * V * dV
end

"""
    fermi_crossing(m::ErpenbeckThossModel; bracket = (1.5, 3.0)) → x₀ in Å

Solve `h(x₀) = 0` by bisection on `bracket`, to the last representable `Float64`.

Recovers the files' `x_resonance_Ang = 2.0203445032676` to `1.4e-11` Å. Use it rather
than the nearest stored node whenever the crossing itself is the object of interest — the
`x` grid is graded, so "nearest node" is a different distance in different places.

!!! note "There is a second crossing"
    `h` also changes sign at `x = 0.88054` Å, on the steep inner branch where `h` plunges
    to `-90` eV. The default bracket excludes it deliberately: the graded grid puts only
    **two** nodes in `[0.8, 0.95]` and `Λ(0.9, 0)` is `0.28` against `1398` at `x₀`, so
    the inner crossing is all but invisible in the training set. Pass a bracket
    containing it to locate it anyway — worth doing before densifying that region.
"""
function fermi_crossing(m::ErpenbeckThossModel; bracket = (1.5, 3.0))
    lo, hi = float(bracket[1]), float(bracket[2])
    lo < hi || throw(ArgumentError("bracket must be increasing, got $bracket"))
    flo, fhi = level_position(lo, m), level_position(hi, m)
    signbit(flo) == signbit(fhi) && throw(ArgumentError(
        "h does not change sign on $bracket: h($lo) = $flo, h($hi) = $fhi"))
    # Bisection, not Newton: `h` is a difference of exponentials with a second sign
    # change on the grid, and a bracket that is never left is worth more here than
    # quadratic convergence on a call made once.
    while true
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && return mid
        fm = level_position(mid, m)
        iszero(fm) && return mid
        if signbit(fm) == signbit(flo)
            lo, flo = mid, fm
        else
            hi = mid
        end
    end
end

"""
    markovian_friction(x, m::ErpenbeckThossModel, kT; halfwidth = 60kT, nodes = 401)

The zero-frequency friction `Λ(x, 0)` of the wide-band model, in **mₑ/aut** — the same
quantity the `Lambda_spectrum_*.h5` files store in row 1, evaluated analytically at any
`x` in Å.

```math
Λ(x,0) = -π\\int \\mathrm{d}ϵ \\;
  \\Big(\\partial_x h + (ϵ-h)\\frac{\\partial_x Γ}{Γ}\\Big)^2 A(ϵ)^2 \\,
  \\frac{\\partial f}{\\partial ϵ},
\\qquad
A(ϵ) = \\frac{1}{π}\\frac{Γ/2}{(ϵ-h)^2 + (Γ/2)^2},
\\qquad Γ = 2Δ,
```

with the Fermi level at `ϵ = 0`, which is where [`level_position`](@ref) already measures
`h`. This is the wide-band single-level friction; it is the closed form
`NQCCalculators.WideBandExact` evaluates, and the two agree to eight significant figures.

# Why this exists
[`ChebyshevAmplitudes`](@ref) factors the fitted amplitudes as
`g_c(x) = √(Λ(x,0)) · ĝ_c(t(x))`, so **the envelope is needed wherever the amplitudes
are**. Stored on the 254 graded nodes it is a table; a QGLE integrator asks for it at a
continuously moving `x`. Interpolating the table costs `10%` (log-linear) to `4.5%`
(log-cubic) at the midpoints of the coarse intervals beyond the crossing — an error of
the same size as the fit's own, added for nothing, when the generating formula is three
lines long. Verified: `1.3e-5 %` worst deviation across all 254 stored nodes.

The warp `t(x)` of [`warp_coordinate`](@ref) is analytic already, so with this the
stage-2 fit becomes a genuine function of `x` with no interpolation anywhere.

# Units
`x` in Å; `kT` and the returned value in atomic units. The derivatives
[`dlevel_position`](@ref) and [`dhybridisation_width`](@ref) return Hartree **per Å** and
are converted to per bohr here — this is the mixed unit their docstrings warn about, and
it is squared in the integrand, so getting it wrong rescales `Λ` by `0.28`.

# Quadrature
Composite Simpson on `[-halfwidth, halfwidth]` with `nodes` points. Both factors are
localised at the Fermi level — `-∂f/∂ϵ` has width `k_BT` and `A²` width `Δ ≈ 2k_BT` — and
`-∂f/∂ϵ` cuts off as `e^{-β|ϵ|}`, so `60 k_BT` is a `e^{-60}` truncation. The defaults
converge to `9e-6 %` against `nodes = 20001`; they are not tuned to a particular `x`.
"""
function markovian_friction(x, m::ErpenbeckThossModel, kT;
                            halfwidth = 60kT, nodes::Integer = 401)
    kT > 0 || throw(ArgumentError("the thermal energy must be positive, got $kT"))
    halfwidth > 0 || throw(ArgumentError(
        "the integration half-width must be positive, got $halfwidth"))
    # Simpson needs an even number of intervals, i.e. an odd number of points.
    nodes ≥ 3 && isodd(nodes) || throw(ArgumentError(
        "Simpson needs an odd number of nodes ≥ 3, got $nodes"))

    h = level_position(x, m)
    Γ = 2 * hybridisation_width(x, m)                      # wide-band width, 2Δ
    ∂h = dlevel_position(x, m) * BOHR_IN_ANGSTROM           # Ha/Å → Ha/bohr
    ∂Γ = 2 * dhybridisation_width(x, m) * BOHR_IN_ANGSTROM
    β = 1 / kT

    # -∂f/∂ϵ at μ = 0, written as β/(4cosh²(βϵ/2)) rather than through `exp`: the
    # equivalent e^{βϵ}/(1+e^{βϵ})² overflows to 0/Inf for |ϵ| ≳ 700kT, and the caller
    # chooses the half-width.
    function integrand(ϵ)
        A = (Γ / 2) / (π * ((ϵ - h)^2 + (Γ / 2)^2))
        return π * (∂h + (ϵ - h) * ∂Γ / Γ)^2 * A^2 * β / (4 * cosh(β * ϵ / 2)^2)
    end

    n = nodes - 1
    dϵ = 2 * halfwidth / n
    s = integrand(-halfwidth) + integrand(halfwidth)
    for i in 1:(n - 1)
        s += (isodd(i) ? 4 : 2) * integrand(-halfwidth + i * dϵ)
    end
    return s * dϵ / 3
end
