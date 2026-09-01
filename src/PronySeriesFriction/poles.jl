"""
    PoleSet(λ, α, β)

The pole ladder, shared across all configurations `x` — regime (A) of
`Prony_fit_optimisation_problem.md` §2 and `docs/design/0003-poles-and-the-mirror.md`.

  - `λ[k] > 0` — `K_R` real rates, each a `1×1` block of `Γ₂,₂` and a zero-centred
    Lorentzian.
  - `α[j] > 0`, `β[j] ≥ 0` — `K_C` underdamped pairs, each a `2×2` block
    `R(α, β) = [α β; -β α]` and a *mirrored* Lorentzian at `ω = ±β`.

Everything is in **Hartree** (`docs/design/0001-units-and-data-conventions.md`); the eV
of the HDF5 files are converted once, at load.

Positivity is not cosmetic: `Γ₂,₂` is stable — every eigenvalue has positive real part,
which is what thesis Prop. 4.4.1 assumes — exactly when `λ > 0` and `α > 0`. `β` may be
zero, in which case the pair degenerates into two copies of a real pole (Appendix
A.0(ii)) and its Lorentzian collapses onto `ω = 0`. That is the wanted behaviour at the
Fermi crossing, where `β ≈ |h| → 0`.

The auxiliary dimension is [`auxiliary_dimension`](@ref), `m = K_R + 2K_C`.

Build one with [`log_ladder`](@ref) or [`uniform_ladder`](@ref); use this constructor
directly to combine a pair ladder with real poles, `PoleSet(λ, ladder.α, ladder.β)`.
"""
struct PoleSet{T<:Real}
    λ::Vector{T}
    α::Vector{T}
    β::Vector{T}

    function PoleSet{T}(λ, α, β) where {T<:Real}
        length(α) == length(β) ||
            throw(DimensionMismatch("pair rates α and frequencies β must have equal length"))
        all(>(0), λ) ||
            throw(ArgumentError("real rates λ must be strictly positive (Γ₂,₂ stability)"))
        all(>(0), α) ||
            throw(ArgumentError("pair decay rates α must be strictly positive (Γ₂,₂ stability)"))
        all(≥(0), β) ||
            throw(ArgumentError("pair frequencies β must be non-negative (L is even in β)"))
        return new{T}(λ, α, β)
    end
end

function PoleSet(λ::AbstractVector, α::AbstractVector, β::AbstractVector)
    T = promote_type(float(eltype(λ)), float(eltype(α)), float(eltype(β)))
    return PoleSet{T}(λ, α, β)
end

"""
    auxiliary_dimension(ps) → m = K_R + 2K_C

Size of the auxiliary block `Γ₂,₂`, i.e. the number of extended variables `s` the
QGLE carries (`Prony_fit_optimisation_problem.md` §1). Each real pole costs one, each
underdamped pair costs two.

This is the number the fixed-ladder decision of `0003` is paid in: `K_C ≈ 29` pairs
means `m ≈ 58`, against `m ≈ 4`–`6` if the (inadmissible) pole-tracking regime (B) were
available.
"""
auxiliary_dimension(ps::PoleSet) = length(ps.λ) + 2 * length(ps.α)

"""
    log_ladder(β_min, β_max, K_C; α_ratio = 0.2, K_R = 0, λ_min, λ_max = β_max) → PoleSet

A log-spaced ladder of `K_C` underdamped pairs with `β_j` geometrically spaced over
`[β_min, β_max]` and `α_j = α_ratio · β_j`, optionally with `K_R` real (overdamped)
rates log-spaced over `[λ_min, λ_max]`.

!!! warning "K_R = 0 fails at the Fermi crossing"
    An earlier version of this docstring claimed no real poles were needed, on the
    grounds that small-`β` rungs cover the zero-centred shapes (Appendix A.0(ii)). That
    is **measured false**, because `α_j = α_ratio · β_j` ties the width to the
    frequency: as `β → 0` the width `α → 0` too, giving an arbitrarily *narrow* spike at
    the origin rather than the broad zero-centred Lorentzian the data has there.

    At the crossing the resonance collapses onto `ω = 0` with a half-width of
    `Δ + k_BT ≈ 0.077` eV, and that column carries the global maximum of the whole
    dataset. Fitted with pairs alone it comes out at **95.5%** median relative error
    below 1 eV; adding real poles brings it to **8.1%**. See
    `docs/design/0003-poles-and-the-mirror.md`.

`λ_min` is required whenever `K_R > 0` and has no default, because the useful value is
*not* `β_min`: the real rates have to reach an order of magnitude further down, to the
identifiability floor `5·dω` of [`identifiability_box`](@ref). Starting them at `β_min`
instead leaves the crossing column at 31%. `λ_max` barely matters — extending it from
`β_max` to `ω_max` changes nothing measurable.

This is the seeding of `Prony_fit_optimisation_problem.md` §5: the data gives
`β ≈ |h(x)|` and `α ≈ 0.2|h(x)|` (measured FWHM `≈ 0.35`–`0.43 ω_pk`), and since regime
(A) needs `x`-independent poles the moving resonance is synthesised by shifting weight
between fixed rungs. The physically motivated span is `β ∈ [0.05, 9.4]` eV
`= [1.8e-3, 0.345]` Ha; [`pair_budget`](@ref) turns that span into a rung count.

Constant `α_j/β_j` is the point of the log spacing: the relative width, hence the
fractional overlap of neighbouring rungs, is the same everywhere, so resolution is
uniform in `ln β`. Contrast [`uniform_ladder`](@ref), which is uniform in `β`. `0003`
records that both are worth measuring and that the answer is not known a priori.
"""
function log_ladder(β_min, β_max, K_C; α_ratio = 0.2, K_R::Integer = 0,
                    λ_min = nothing, λ_max = β_max)
    K_C ≥ 2 || throw(ArgumentError("a ladder needs at least two rungs, got K_C = $K_C"))
    0 < β_min < β_max ||
        throw(ArgumentError("need 0 < β_min < β_max, got ($β_min, $β_max)"))
    α_ratio > 0 || throw(ArgumentError("α_ratio must be positive, got $α_ratio"))
    K_R ≥ 0 || throw(ArgumentError("K_R must be non-negative, got $K_R"))
    β = exp.(range(log(β_min), log(β_max); length = K_C))

    λ = Float64[]
    if K_R > 0
        λ_min === nothing && throw(ArgumentError(
            "λ_min is required when K_R > 0 and deliberately has no default: the useful " *
            "value is the identifiability floor 5·dω, an order of magnitude below " *
            "β_min = $β_min. Defaulting it to β_min would silently reproduce the " *
            "measured failure at the Fermi crossing."))
        0 < λ_min ≤ λ_max ||
            throw(ArgumentError("need 0 < λ_min ≤ λ_max, got ($λ_min, $λ_max)"))
        λ = K_R == 1 ? [float(λ_min)] :
            exp.(range(log(λ_min), log(λ_max); length = K_R))
    end
    return PoleSet(λ, α_ratio .* β, β)
end

"""
    split_ladder(segments, K_R, λ_min, λ_max; α_ratio = 0.2) → PoleSet

A ladder whose pairs are log-spaced **within each of several disjoint β segments**,
rather than across one span. `segments` is a vector of `(β_lo, β_hi, n)` triples, all in
Hartree, each contributing `n` rungs log-spaced over `[β_lo, β_hi]`; `α_j = α_ratio·β_j`
throughout, as in [`log_ladder`](@ref).

The reason this exists is that the Erpenbeck–Thoss spectrum asks two unrelated things of
the ladder, in two disjoint places:

  - `β ≲ 9.4` eV — resolve the resonance, which moves with `x` as `ω_pk ≈ |h(x)|`. This
    needs rungs dense enough to overlap, since a moving peak is synthesised by shifting
    weight between fixed ones.
  - `β ∈ [12, 60]` eV — set the `ω⁻²` coefficient of the far tail. Nothing is resolved
    here; the data has no resonance above `|h| = 8.65` eV. A handful of rungs suffices,
    and they need not tile.

Spending a single log-spaced ladder on both wastes rungs: stretching `β_max` from 9.4 to
60 eV at fixed `K_C = 20` moves six rungs out of the resonance band to buy tail coverage
that four would have bought, and the crossing column pays for it. Measured at `Γ = 0.1`,
`m = 52`: one span gives `J/J₀ = 0.132` with the crossing at 9.6%, while `16 + 4` split
gives `0.135` with the crossing at 8.5% *and* a better sub-eV band.

`K_R` real rates are log-spaced over `[λ_min, λ_max]` exactly as in [`log_ladder`](@ref),
and both are required — there is no useful default for either. **`λ_max` belongs near the
crossing width `Δ + k_BT ≈ 0.077` eV, not at `β_max`.** The real rates exist only to build
the broad zero-centred peak at the Fermi crossing; rates above that width are dead weight.
Measured on this dataset, 9 of 12 rates log-spaced to 9.39 eV carry amplitude in *zero*
columns, and cutting to `K_R = 3` over `[0.005, 0.05]` eV removes 9 auxiliary variables
while improving every band.

Segments must be ascending and disjoint; overlapping them is rejected rather than
silently producing duplicate rungs, which would make the NNLS design matrix rank
deficient.
"""
function split_ladder(segments, K_R::Integer, λ_min, λ_max; α_ratio = 0.2)
    isempty(segments) && throw(ArgumentError("need at least one β segment"))
    α_ratio > 0 || throw(ArgumentError("α_ratio must be positive, got $α_ratio"))
    K_R ≥ 0 || throw(ArgumentError("K_R must be non-negative, got $K_R"))

    β = Float64[]
    prev_hi = 0.0
    for (lo, hi, n) in segments
        n ≥ 1 || throw(ArgumentError("each segment needs at least one rung, got n = $n"))
        0 < lo ≤ hi || throw(ArgumentError("need 0 < β_lo ≤ β_hi, got ($lo, $hi)"))
        lo > prev_hi || throw(ArgumentError(
            "β segments must be ascending and disjoint; segment starting at $lo " *
            "overlaps the previous one ending at $prev_hi"))
        n == 1 && hi != lo && throw(ArgumentError(
            "a one-rung segment must be a point, got ($lo, $hi, 1)"))
        append!(β, n == 1 ? [float(lo)] : exp.(range(log(lo), log(hi); length = n)))
        prev_hi = hi
    end

    λ = Float64[]
    if K_R > 0
        0 < λ_min ≤ λ_max ||
            throw(ArgumentError("need 0 < λ_min ≤ λ_max, got ($λ_min, $λ_max)"))
        λ = K_R == 1 ? [float(λ_min)] :
            exp.(range(log(λ_min), log(λ_max); length = K_R))
    end
    return PoleSet(λ, α_ratio .* β, β)
end

"""
    uniform_ladder(β_max, K_C; α = β_max / K_C) → PoleSet

A uniform ladder of `K_C` pairs, `β_j` equally spaced over `[0, β_max]` with a common
width `α` — the Stella, Lorenz & Kantorovich choice (*PRB* **89**, 134303, Eqs. 84–85:
`ω_k = k ω_D/K`, equal `τ_k`, equal weights). Their stated reason for preferring it to
regression is that it *converges as `K → ∞`*, which is the same argument `0003` makes
for fixing the ladder at all.

The `j = 1` rung has `β = 0`. That is deliberate and matches their degenerate `k = 0`
pair; the block then has a repeated real eigenvalue `α` and behaves as two copies of a
real pole (Appendix A.0(ii)).

The default `α = β_max/K_C` is the rung spacing, so neighbouring Lorentzians cross near
their half-maxima. Unlike [`log_ladder`](@ref) the width does *not* track `β`, so the
low rungs are relatively broad and the high rungs relatively sharp — the whole
difference between the two ladders, and the thing to measure.
"""
function uniform_ladder(β_max, K_C; α = β_max / K_C)
    K_C ≥ 2 || throw(ArgumentError("a ladder needs at least two rungs, got K_C = $K_C"))
    β_max > 0 || throw(ArgumentError("need β_max > 0, got $β_max"))
    β = collect(range(0.0, float(β_max); length = K_C))
    return PoleSet(Float64[], fill(float(α), K_C), β)
end

"""
    pair_budget(β_min, β_max; α_ratio = 0.2) → K_C

How many log-spaced rungs it takes to tile `[β_min, β_max]` at constant relative width
`α/β = α_ratio` (`Prony_fit_optimisation_problem.md` §7):

```math
K_C \\sim \\frac{\\ln(β_{\\max}/β_{\\min})}{\\ln(1 + α/β)}
```

Consecutive rungs sit a factor `1 + α_ratio` apart, i.e. roughly one half-width, so the
ladder is neither gapped nor redundant. For the useful span `β ∈ [1.8e-3, 0.345]` Ha
(`[0.05, 9.4]` eV) at `α_ratio = 0.2` this returns **29**, hence `m ≈ 58` — the cost of
regime (A) and, per §7, the main open decision of the whole scheme.
"""
pair_budget(β_min, β_max; α_ratio = 0.2) =
    ceil(Int, log(β_max / β_min) / log1p(α_ratio))

"""
    identifiability_box(dω, ω_max; c = 5) → (lo, hi)

The rate window `[c·dω, ω_max]` of `Prony_fit_optimisation_problem.md` §4, in Hartree.
Rates outside it are not determined by the data:

  - below `c·dω` the Lorentzian is narrower than the frequency grid resolves;
  - above `ω_max` it is flat over the whole fitted band, hence indistinguishable from
    the white part — **fold such a mode into `a₀` rather than fitting it**.

For the `Γ = 0.1` file (`ω ∈ [1e-6, 60]` eV) this is `[1.8e-4, 2.205]` Ha. In practice
the useful region is `~6×` tighter, since all observed structure lies in
`β ∈ [0, 9.4]` eV.
"""
identifiability_box(dω, ω_max; c = 5) = (c * dω, ω_max)

"""
    is_identifiable(ps, dω, ω_max; c = 5) → Bool

Whether every rate in `ps` lies in the [`identifiability_box`](@ref): `λ_k` and `α_j` in
`[c·dω, ω_max]`, `β_j` in `[0, ω_max]` (`β` may be zero — §4 permits the collapse onto a
zero-centred shape).

A check, not a constraint: a ladder that fails this is still a valid `PoleSet` and will
still fit, but the offending rungs are fitting noise or duplicating `a₀`.
"""
function is_identifiable(ps::PoleSet, dω, ω_max; c = 5)
    lo, hi = identifiability_box(dω, ω_max; c)
    inbox(r) = lo ≤ r ≤ hi
    return all(inbox, ps.λ) && all(inbox, ps.α) && all(b -> 0 ≤ b ≤ ω_max, ps.β)
end

"""
    spectral_density(ω, ps::PoleSet, a₀, a, A) → Λ

The model of `Prony_fit_optimisation_problem.md` §1 with the rates supplied by a
[`PoleSet`](@ref) and the amplitudes of one configuration supplied separately — the
split that regime (A) forces. `a` must have `K_R` entries and `A` must have `K_C`.
"""
spectral_density(ω, ps::PoleSet, a₀, a, A) =
    spectral_density(ω, a₀, a, ps.λ, A, ps.α, ps.β)

"""
    memory_kernel(τ, ps::PoleSet, a, A) → K

Time-domain counterpart of `spectral_density(ω, ::PoleSet, a₀, a, A)`, less the
singular `2a₀δ(τ)` part.
"""
memory_kernel(τ, ps::PoleSet, a, A) = memory_kernel(τ, a, ps.λ, A, ps.α, ps.β)
