"""
    TrackingPoles(c, u, w, z, ru, rw, rz, tail, rate_floor)

Poles that **move with the configuration** — regime (B) of
`Prony_fit_optimisation_problem.md` §0, in the parametrisation §6 prescribes:

```math
β_j(x) = |c_j|\\,|h(x)| + |d_j|, \\qquad
α_j(x) = \\max\\big(u_j|h(x)| + w_jΔ(x) + z_j k_BT,\\ \\text{floor}\\big),
```

The offset `d_j` is **not** in §6, which writes `β_j = c_j|h|` alone. It is here because that
form is degenerate at the Fermi crossing: `h(x₀) = 0` sends every pair to `β = 0` at once,
where a mirrored Lorentzian is `L(ω;α,0) = lorentzian(ω,α)` — the same shape a real rate
makes. Measured on the `Γ = 0.1` fit, the design matrix then goes near-collinear and the
inner NNLS keeps **one mode out of eleven** on the column carrying the dataset's global
maximum. The fixed ladder spends seven pairs at `β = 0.05`–`0.40` eV on that column. An
offset is the smallest change that lets the tracking law reach them.

and the same form without a frequency for the real rates,
`λ_k(x) = max(ru_k|h| + rw_kΔ + rz_k k_BT, floor)`.

Evaluating it at one configuration gives an ordinary [`PoleSet`](@ref) — see
[`poleset`](@ref) — so everything downstream (`design_matrix`, `fit_column`,
`qgle_blocks`, `is_identifiable`) works unchanged. What changes is that the ladder is
rebuilt per column instead of once.

!!! warning "Diagnostic only on this data — not a production model"
    §9.7 settles regime (B) negatively for the Erpenbeck–Thoss spectra, on two counts
    that this type does nothing to fix:

      - **The adiabatic condition fails at the Fermi crossing.** With the width floored
        at `Δ + k_BT` the ratio `τ_mem/τ_conf = ħ|dh/dx||v|/α²` is `15.2` at thermal
        velocities and `133` at 1 eV for `Γ = 0.1` — and the crossing carries the
        friction peak. The SDE would then not realise the fitted kernel exactly, which
        is the one thing regime (A) buys (§9.5).
      - **It does not save auxiliary variables here.** A hybrid in `x` saves nothing
        (§9.7 Obstruction 2): every legal split leaves both `|h|ₘₐₓ` and `x₀` in the
        regime-(A) region, so the fixed ladder still spans the whole `β` range.

    It exists for the measurements §7 step 2 and the note's open items ask for and that
    nothing else in this package can make: how much relative error is *achievable* at a
    given `K_C` (the fixed-ladder numbers are otherwise uncalibrated), whether the real
    rates `K_R` are needed once `α` is free of the `α = 0.2β` tie, and whether the
    amplitude table is smooth when the poles track. Compare `PolynomialAmplitudes`,
    which is likewise kept so that a predicted failure is a measured one.

# Fields
- `c, d, u, w, z :: Vector{T}` — the `K_C` tracking pairs' coefficients, one entry each.
- `ru, rw, rz :: Vector{T}` — the `K_R` tracking real rates' coefficients.
- `tail :: PoleSet{T}` — **fixed**, configuration-independent pairs, appended after the
  tracking ones. Carries no real rates.
- `rate_floor :: T` — the `5·dω` identifiability floor of §4, in Hartree.

# The tail, and why it is fixed
`|h| ≤ 8.65` eV over the whole grid, so nothing above ~9 eV is a resonance and there is
nothing there for a pole to track; that band exists only to set the `ω⁻²` coefficient of
the far tail (`fit_gamma01.jl`, the split-ladder rationale). Leaving it out would not be
a cheaper model but a different comparison — §3's band decomposition puts **56.2 %** of
the objective above 10 eV — so the tail rungs stay fixed and only the resonance modes
track. Note this is a split over *modes*, not over `x`: §9.7's Obstruction 2 is about an
`x` split and does not apply. It also does not make regime (B) legal — the tracking
modes are the ones carrying the crossing.

# Sign and positivity
Coefficients are unconstrained reals. `β` takes `|c_j|` (`L` is even in `β` anyway) and
both rate families are floored at `rate_floor`, so every parameter vector the optimiser
can propose yields a valid `PoleSet` and the outer search needs no box. The floor is not
a guard: §4 says a rate below `5·dω` is narrower than the frequency grid resolves and is
therefore not determined by the data, so clamping there discards nothing that was ever
identifiable.
"""
struct TrackingPoles{T<:Real}
    c::Vector{T}
    d::Vector{T}
    u::Vector{T}
    w::Vector{T}
    z::Vector{T}
    ru::Vector{T}
    rw::Vector{T}
    rz::Vector{T}
    tail::PoleSet{T}
    rate_floor::T

    function TrackingPoles{T}(c, d, u, w, z, ru, rw, rz, tail, rate_floor) where {T<:Real}
        length(c) == length(d) == length(u) == length(w) == length(z) ||
            throw(DimensionMismatch(
                "pair coefficients c, d, u, w, z must have equal length, got " *
                "$(length(c)), $(length(d)), $(length(u)), $(length(w)), $(length(z))"))
        length(ru) == length(rw) == length(rz) || throw(DimensionMismatch(
            "real-rate coefficients ru, rw, rz must have equal length, got " *
            "$(length(ru)), $(length(rw)), $(length(rz))"))
        isempty(tail.λ) || throw(ArgumentError(
            "the fixed tail carries pairs only, got $(length(tail.λ)) real rates; real " *
            "rates exist to build the broad zero-centred peak at the crossing, which is " *
            "exactly the feature that tracks, so they belong in (ru, rw, rz)"))
        rate_floor > 0 || throw(ArgumentError(
            "the identifiability floor must be positive, got $rate_floor"))
        return new{T}(c, d, u, w, z, ru, rw, rz, tail, rate_floor)
    end
end

function TrackingPoles(c::AbstractVector, d::AbstractVector, u::AbstractVector,
                       w::AbstractVector, z::AbstractVector, ru::AbstractVector,
                       rw::AbstractVector, rz::AbstractVector, tail::PoleSet, rate_floor)
    T = promote_type(float(eltype(c)), float(eltype(d)), float(eltype(u)),
                     float(eltype(w)), float(eltype(z)),
                     float(eltype(ru)), float(eltype(rw)), float(eltype(rz)),
                     float(eltype(tail.α)), typeof(float(rate_floor)))
    conv(v) = collect(T, v)
    # Convert, do not sanitise: stripping `tail.λ` here would silently discard real
    # rates instead of letting the inner constructor reject them.
    tailT = PoleSet(conv(tail.λ), conv(tail.α), conv(tail.β))
    return TrackingPoles{T}(conv(c), conv(d), conv(u), conv(w), conv(z),
                            conv(ru), conv(rw), conv(rz), tailT, T(rate_floor))
end

"""
    n_tracking(tp) → (K_R, K_C_track, K_C_tail)

The three mode counts, kept separate because only the first two move with `x`.
[`mode_counts`](@ref) folds the last two together, since `design_matrix` does.
"""
n_tracking(tp::TrackingPoles) = (length(tp.ru), length(tp.c), length(tp.tail.α))

"""
    mode_counts(tp) → (K_R, K_C)

`(K_R, K_C_track + K_C_tail)` — the shape of the amplitude vector `design_matrix`
expects, so a `TrackingPoles` and the [`GridAmplitudes`](@ref) fitted against it agree
on `mode_counts` exactly as a [`PoleSet`](@ref) would.
"""
function mode_counts(tp::TrackingPoles)
    K_R, K_track, K_tail = n_tracking(tp)
    return (K_R, K_track + K_tail)
end

"""
    auxiliary_dimension(tp) → m = K_R + 2(K_C_track + K_C_tail)

Same accounting as [`auxiliary_dimension`](@ref)`(::PoleSet)` — the SDE carries one
`Γ₂,₂` whatever the poles depend on, which is the whole content of §9.7's Obstruction 2.
"""
auxiliary_dimension(tp::TrackingPoles) =
    ((K_R, K_C) = mode_counts(tp); K_R + 2 * K_C)

"""
    poleset(tp::TrackingPoles, h, Δ, kT) → PoleSet

The ladder at one configuration, from that configuration's level position `h`,
hybridisation width `Δ` and thermal energy `kT` — all in Hartree, as
`docs/design/0001` requires. `h` may have either sign; only `|h|` enters.

The pair block is ordered **tracking first, then tail**, matching the column order of
[`design_matrix`](@ref) and hence the `(a₀, a, A)` layout of the amplitudes, so one
amplitude vector reads the same against every column's ladder.

Being an ordinary `PoleSet`, the result can be handed to `is_identifiable`,
`qgle_blocks`, `spectral_density` and the rest unchanged — the point of returning one
rather than a bespoke type.
"""
function poleset(tp::TrackingPoles{T}, h, Δ, kT) where {T}
    ah = abs(float(h))
    λ = similar(tp.ru)
    @inbounds for k in eachindex(tp.ru)
        λ[k] = max(tp.ru[k] * ah + tp.rw[k] * Δ + tp.rz[k] * kT, tp.rate_floor)
    end
    K_track = length(tp.c)
    α = Vector{T}(undef, K_track + length(tp.tail.α))
    β = similar(α)
    @inbounds for j in 1:K_track
        α[j] = max(tp.u[j] * ah + tp.w[j] * Δ + tp.z[j] * kT, tp.rate_floor)
        β[j] = abs(tp.c[j]) * ah + abs(tp.d[j])
    end
    @inbounds for (j, t) in enumerate(eachindex(tp.tail.α))
        α[K_track + j] = tp.tail.α[t]
        β[K_track + j] = tp.tail.β[t]
    end
    return PoleSet(λ, α, β)
end

"""
    parameters(tp::TrackingPoles) → θ

The coefficients as one flat vector, in the order
`(c, d, u, w, z, ru, rw, rz)` — what the outer search of [`fit_tracking`](@ref) optimises.
Length `5K_C_track + 3K_R`, which is the whole cost of regime (B)'s pole model: **12
numbers** for the three-pair, no-real-rate case, against a ladder's worth of fixed rungs.

The fixed tail and the rate floor are not parameters and are not included.
"""
function parameters(tp::TrackingPoles{T}) where {T}
    return vcat(tp.c, tp.d, tp.u, tp.w, tp.z, tp.ru, tp.rw, tp.rz)::Vector{T}
end

"""
    parameters!(tp::TrackingPoles, θ) → tp

Write a flat coefficient vector back into `tp`, in place, in the order
[`parameters`](@ref) produces.

In place because the outer search evaluates the objective thousands of times and each
evaluation would otherwise allocate a new pole model; the scratch object is read
immediately and never retained, so no caller ever observes a shared mutation.
"""
function parameters!(tp::TrackingPoles, θ::AbstractVector)
    K_R, K_track, _ = n_tracking(tp)
    length(θ) == 5 * K_track + 3 * K_R || throw(DimensionMismatch(
        "expected $(5 * K_track + 3 * K_R) coefficients for K_C_track = $K_track, " *
        "K_R = $K_R, got $(length(θ))"))
    o = 0
    for v in (tp.c, tp.d, tp.u, tp.w, tp.z, tp.ru, tp.rw, tp.rz)
        copyto!(v, 1, θ, o + 1, length(v))
        o += length(v)
    end
    return tp
end

"""
    tracking_seed(K_C, K_R, tail, rate_floor; u = 0.2, w = 1, z = 1,
                  ru = 0, rw = 1, rz = 1, c = nothing) → TrackingPoles

The starting point of the outer search, set from what the data has already been measured
to do rather than from a generic guess — the reason §6 calls this ansatz "a polynomial in
**physical features**" and expects it to be well conditioned:

  - `c_j ≈ 1` because `ω_pk/|h| = 1.02`–`1.15` across the whole grid (§6);
  - `u = 0.2` from the measured FWHM `≈ 0.35`–`0.43 ω_pk` (§5), the same ratio
    [`log_ladder`](@ref) ties its rungs at;
  - `w = z = 1`, so that at the crossing (`h = 0`) the width falls back on
    `Δ + k_BT ≈ 0.077` eV — the measured half-width there, `0.077` against `0.076`
    predicted at `Γ = 0.1` (§9.7);
  - `ru = 0, rw = rz = 1`, i.e. real rates that sit at `Δ + k_BT` everywhere and do not
    follow `|h|` at all, since §"the crossing" is the only thing they were ever for.

Three of the defaults are **spread** rather than repeated, because identical coefficients
across modes are an exactly degenerate starting point: the design matrix gets repeated
columns, the inner NNLS is free to split their amplitude arbitrarily, and the outer
simplex starts on a degenerate face.

The two spreads that matter are read off the **fitted fixed ladder**, not guessed:

  - `c` geometrically over `[0.51, 6.26]`. Since `β_l/|h| = c_l` is constant in `x`, the
    right band is the one the production ladder's *live* rungs occupy in that ratio —
    measured at `Γ = 0.1` over the resonance band as 10th–90th percentile `0.51`–`6.26`,
    median `1.96`. An earlier default used §6's `ω_pk/|h| = 1.02`–`1.15`, which is the
    ratio of the **peak** rather than of every rung that carries amplitude, and spread the
    modes over a factor of 1.5 where the data asks for 12.
  - `d` geometrically over `[1.80e-3, 1.474e-2]` Ha = `[0.049, 0.401]` eV — where the fixed ladder's seven live pairs
    sit on the crossing column, which is the band the tracking law cannot otherwise reach.
  - `w` over `[1/3, 3]`, so the widths at the crossing bracket `Δ + k_BT`.

**Why the ladder cannot be pruned instead.** All 20 of its pairs carry amplitude in at
least 10 % of columns — none is dead weight. What tracking replaces is not unused rungs
but *fixed positions*: 20 of them, against `K_C` that move.

**`d` and `w` are the two that matter at `x₀`, and `c` cannot substitute for either.** At
the crossing `h = 0`, so `β_j = c_j·0 + d_j`: the `c` spread does nothing there and only
`d` places a pair off zero at all.
"""
function tracking_seed(K_C::Integer, K_R::Integer, tail::PoleSet, rate_floor;
                       u = 0.2, w = 1.0, z = 1.0, ru = 0.0, rw = 1.0, rz = 1.0,
                       c = nothing, d = nothing, c_band = (0.51, 6.26),
                       d_band = (1.80e-3, 1.474e-2))   # Ha — [0.049, 0.401] eV
    K_C ≥ 0 && K_R ≥ 0 || throw(ArgumentError("mode counts must be non-negative"))
    # Geometric over a band; one mode sits at the band's geometric mean rather than at an
    # arbitrary end of it.
    logspread(lo, hi, n) = n ≤ 0 ? Float64[] :
        n == 1 ? [sqrt(lo * hi)] : exp.(range(log(lo), log(hi); length = n))
    spread(v, n, halfwidth) =
        n ≤ 1 ? fill(float(v), n) : float(v) .* exp.(range(-halfwidth, halfwidth; length = n))
    cs = c === nothing ? logspread(c_band..., K_C) : collect(float.(c))
    ds = d === nothing ? logspread(d_band..., K_C) : collect(float.(d))
    length(cs) == K_C || throw(DimensionMismatch(
        "c has $(length(cs)) entries but K_C = $K_C"))
    length(ds) == K_C || throw(DimensionMismatch(
        "d has $(length(ds)) entries but K_C = $K_C"))
    return TrackingPoles(cs, ds, fill(float(u), K_C), spread(w, K_C, log(3)),
                         fill(float(z), K_C),
                         fill(float(ru), K_R), spread(rw, K_R, log(3)),
                         fill(float(rz), K_R), tail, rate_floor)
end

"""
    tracking_report(tp) → NamedTuple

The fitted coefficients grouped as §6 writes them, for printing next to the values
`tracking_seed` started from. `c` is reported as `|c|`, which is what
[`poleset`](@ref) uses.

Reading it is the point of the diagnostic: `c ≈ 1.02`–`1.15` confirms §6's ansatz on this
data rather than assuming it, and `w ≈ z ≈ 1` turns "the width floors at `Δ + k_BT`" from
a prediction into a measurement.
"""
tracking_report(tp::TrackingPoles) =
    (; c = abs.(tp.c), d = abs.(tp.d), u = copy(tp.u), w = copy(tp.w), z = copy(tp.z),
       ru = copy(tp.ru), rw = copy(tp.rw), rz = copy(tp.rz),
       K_R = length(tp.ru), K_C_track = length(tp.c), K_C_tail = length(tp.tail.α),
       m = auxiliary_dimension(tp))

# ==================================================================================================
# TiedLaw — the same pole law with the per-mode freedom removed
# ==================================================================================================

"""
    TiedLaw(c₀, cρ, d₀, dρ, u, w₀, wρ, z, ru, rw₀, rwρ, rz, K_C, K_R)

The [`TrackingPoles`](@ref) law with its per-mode coefficients replaced by **geometric
families**, so the number of free numbers is 12 whatever `K_C` and `K_R` are:

```math
c_l = c_0\\,c_ρ^{\\,l-1}, \\qquad d_l = d_0\\,d_ρ^{\\,l-1}, \\qquad w_l = w_0\\,w_ρ^{\\,l-1},
```

with `u`, `z`, `ru` and `rz` shared outright and `rw_k = rw_0 rw_ρ^{\\,k-1}`.

**Why the tie exists.** A free `TrackingPoles` costs `5K_C + 3K_R` outer parameters — 43 at
`K_C = 8, K_R = 1` — and Nelder–Mead stagnates well below that. The fixed ladder has the
same shape of answer: [`log_ladder`](@ref) is a span and a count, not 20 free rungs, and
`α_j = α_{ratio}β_j` ties the widths for exactly this reason. A law that needs one free
number per mode was never the claim §6 makes; it writes *one* law.

It also removes a measured pathology. With free per-mode coefficients the fit reaches its
optimum through large cancellations — the largest `|coefficient|` in the `K_C = 3, K_R = 3`
fit is `1518` against `0.51` in the one-pair model — because `Δ ∈ [1.4e-4, 0.05]` eV and the
constant `k_BT = 0.0259` eV are nearly collinear, so `wΔ` and `z k_BT` trade off freely.
Sharing `z` across modes takes that freedom away.

Expand it with [`expand`](@ref); seed it with [`tied_seed`](@ref).
"""
struct TiedLaw{T<:Real}
    c₀::T;  cρ::T
    d₀::T;  dρ::T
    u::T
    w₀::T;  wρ::T
    z::T
    ru::T;  rw₀::T; rwρ::T; rz::T
    K_C::Int
    K_R::Int
end

"Number of free numbers in a `TiedLaw` — 12, independent of the mode counts."
const N_TIED = 12

"""
    TiedLaw(θ::AbstractVector, K_C, K_R)

Rebuild from the flat vector [`parameters`](@ref) produces. `TiedLaw` is immutable and only
12 scalars wide, so the outer search reconstructs rather than mutating — there is no shared
object for an optimiser to alias.
"""
function TiedLaw(θ::AbstractVector, K_C::Integer, K_R::Integer)
    length(θ) == N_TIED || throw(DimensionMismatch(
        "a TiedLaw has $N_TIED numbers, got $(length(θ))"))
    return TiedLaw(θ..., Int(K_C), Int(K_R))
end

parameters(tl::TiedLaw{T}) where {T} =
    T[tl.c₀, tl.cρ, tl.d₀, tl.dρ, tl.u, tl.w₀, tl.wρ, tl.z, tl.ru, tl.rw₀, tl.rwρ, tl.rz]

"""
    expand(tl::TiedLaw, tail, rate_floor) → TrackingPoles

Materialise the geometric families into the per-mode vectors [`TrackingPoles`](@ref) holds,
so everything downstream is unchanged — the tie is a reparametrisation of the search, not a
different model.

The ratios enter as `|ρ|`: `ρ^{l-1}` with a negative `ρ` would alternate the sign mode by
mode, which is a ladder in name only. `poleset` already takes `|c|` and `|d|`, so the
magnitudes are what the model ever sees.
"""
function expand(tl::TiedLaw, tail::PoleSet, rate_floor)
    geo(x₀, ρ, n) = n ≤ 0 ? Float64[] : [float(x₀) * abs(float(ρ))^(l - 1) for l in 1:n]
    K_C, K_R = tl.K_C, tl.K_R
    return TrackingPoles(geo(tl.c₀, tl.cρ, K_C), geo(tl.d₀, tl.dρ, K_C),
                         fill(float(tl.u), K_C), geo(tl.w₀, tl.wρ, K_C),
                         fill(float(tl.z), K_C),
                         fill(float(tl.ru), K_R), geo(tl.rw₀, tl.rwρ, K_R),
                         fill(float(tl.rz), K_R), tail, rate_floor)
end

"""
    tied_seed(K_C, K_R; c_band, d_band, w_band, u, z, ru, rz, rw_band) → TiedLaw

The [`tracking_seed`](@ref) bands, expressed as ratios. `c` spans the measured `[0.51,
6.26]` of the production ladder's live `β/|h|`, `d` the `[0.049, 0.401]` eV its live pairs
occupy on the crossing column, and `w` the `[1/3, 3]` that brackets `Δ + k_BT`.

A one-mode family sits at the band's geometric mean and its ratio is 1, so `K_C = 1`
reduces to a single mode rather than to an end of the band.
"""
function tied_seed(K_C::Integer, K_R::Integer;
                   c_band = (0.51, 6.26), d_band = (1.80e-3, 1.474e-2),
                   w_band = (1 / 3, 3.0), rw_band = (1 / 3, 3.0),
                   u = 0.2, z = 1.0, ru = 0.0, rz = 1.0)
    K_C ≥ 1 || throw(ArgumentError("a tied law needs at least one pair, got K_C = $K_C"))
    K_R ≥ 0 || throw(ArgumentError("K_R must be non-negative, got $K_R"))
    base(band, n) = n ≤ 1 ? (sqrt(band[1] * band[2]), 1.0) :
                            (float(band[1]), (band[2] / band[1])^(1 / (n - 1)))
    c₀, cρ = base(c_band, K_C)
    d₀, dρ = base(d_band, K_C)
    w₀, wρ = base(w_band, K_C)
    rw₀, rwρ = K_R == 0 ? (1.0, 1.0) : base(rw_band, K_R)
    return TiedLaw(promote(c₀, cρ, d₀, dρ, float(u), w₀, wρ, float(z),
                           float(ru), rw₀, rwρ, float(rz))..., Int(K_C), Int(K_R))
end
