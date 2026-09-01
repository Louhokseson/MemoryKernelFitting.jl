# Stage-1 Prony fit of the Erpenbeck–Thoss spectral density at Γ = 0.1 eV.
#
#   julia scripts/fitting/fit_gamma01.jl
#
# Γ = 0.1 first, per `Prony_fit_optimisation_problem.md` §7 step 1: no negative entries,
# the resonance sits well inside the 60 eV window, and Δ ≈ 0.05 eV is comfortably above
# k_BT = 0.0259 eV. Each Γ is a separate model and hence a separate fit; never pool them.
#
# This is a *measurement* script, not a demo. §7 step 3 says to choose ε, μ and the basis
# only after a first fit exists, and the three sweeps below are what that choice needs:
# where in ω the error actually lives, what the §4 Markovian anchor costs and buys, and
# how the error responds to the rung budget K_C.
#
# The algorithm itself lives in `src/Fitting/` and is tested; this file only chooses the
# settings, reports, and saves.

using DrWatson
@quickactivate "MemoryKernelFitting"

using MemoryKernelFitting
using HDF5
using Printf

# Statistics is not a dependency of the package; this is the only summary it needs.
function median_(v)
    s = sort(vec(collect(v)))
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

# ------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------

const GAMMA_EV   = 0.1

# Ladder composition. Two corrections stand behind this, both measured on this dataset.
#
# design/0003: pairs alone leave the Fermi crossing — the tallest column in the set — at
# 95% relative error, because α = 0.2β ties width to frequency and so cannot make a
# *broad* zero-centred peak. Real rates reaching to the identifiability floor fix it.
#
# The split, and K_R = 3. The spectrum asks two unrelated things of the ladder. Below
# 9.4 eV it must resolve a resonance that moves with x; above 10 eV it must only set the
# ω⁻² coefficient of the tail, since |h| ≤ 8.65 eV means there is no resonance up there.
# One log span spends rungs as if both jobs were the same: stretching β_max to 60 eV at
# K_C = 20 moves six rungs out of the resonance band to buy what four would have bought.
# Splitting 16 + 4 keeps the resonance band dense and still fixes the far-column tail,
# which decays as ω⁻² (the near-crossing columns decay as ω⁻³ and are out of reach of any
# Lorentzian sum — that is what the ε floor exists to discount, not a budget problem).
#
# K_R = 3 rather than 12 because 9 of the 12 rates were dead: on the previous ladder,
# λ₅…λ₁₂ carried amplitude in *zero* columns. Real rates exist to build the broad
# zero-centred peak at the crossing, whose half-width is Δ + k_BT ≈ 0.077 eV, so λ_max
# belongs there and not at β_max. That alone is 9 auxiliary variables.
#
# Net: m = 52 → 43, better on every band, J/J₀ 0.193 → 0.135.
const K_C_LO     = 16            # pairs over the resonance band
const K_C_HI     = 4             # pairs over the tail band
const K_R        = 3             # real (overdamped) rates
const BETA_LO    = (1.8e-3, 0.345)      # Ha — [0.049, 9.39] eV, spanning max ω_pk = 8.89
const BETA_HI    = (0.441, 2.20496)     # Ha — [12.0, 60.0] eV, the tail band
const LAMBDA_LO  = 1.84e-4       # Ha — 5·dω, the identifiability floor of §4
const LAMBDA_HI  = 1.84e-3       # Ha — 0.05 eV, just under the crossing width Δ + k_BT
const EPSILON    = 1e-3          # §3 residual floor
const ANCHOR_W   = 1e3           # §4 Markovian anchor, as a weighted row
const BANDS      = [(0.0, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, 60.0)]
const DATAFILE   = datadir("sims", "friction", "ErpenbeckThoss",
    "Lambda_spectrum_Gamma_eV=$(GAMMA_EV)_Nx=257_T_K=300.0_domega_eV=0.001.h5")

isfile(DATAFILE) || error("spectrum file not found: $DATAFILE")

# ------------------------------------------------------------------------------------
# Load
# ------------------------------------------------------------------------------------

s   = load_lambda_spectrum(DATAFILE)          # drops x < 0.8 Å, per §4
idx = subsample_omega(s.ω)                    # retained band ℬ, index 1 always kept
ω   = s.ω[idx]
Λ   = s.Λ[idx, :]
v   = trapezoid_weights(s.x)                  # non-negotiable on this graded grid
σ   = residual_scales(Λ; ε = EPSILON)
# Index convention: rows of Λ (and σ) are frequencies j, columns are configurations i.
# v is indexed by i alone — the ω grid is uniform and needs no weight, the x grid is
# graded by 290× and cannot be integrated without one.
ωeV = hartree_to_eV.(ω)
low = ωeV .<= 1.0                             # the band the resonance lives in

ladder = split_ladder([(BETA_LO..., K_C_LO), (BETA_HI..., K_C_HI)],
                      K_R, LAMBDA_LO, LAMBDA_HI)
zero_model = GridAmplitudes(length(s.x), K_R, K_C_LO + K_C_HI)
J₀ = objective_value(ladder, zero_model, ω, Λ, v, σ)

@printf("Γ = %.2f eV | %d configurations (x ∈ [%.2f, %.2f] Å) | %d of %d frequencies retained\n",
        hartree_to_eV(s.Γ), length(s.x), first(s.x), last(s.x), length(idx), length(s.ω))
@printf("ladder: %d + %d pairs + %d real rates, m = %d\n",
        K_C_LO, K_C_HI, K_R, auxiliary_dimension(ladder))
@printf("        β ∈ [%.3f, %.2f] ∪ [%.1f, %.1f] eV, λ ∈ [%.4f, %.3f] eV\n",
        hartree_to_eV(BETA_LO[1]), hartree_to_eV(BETA_LO[2]),
        hartree_to_eV(BETA_HI[1]), hartree_to_eV(BETA_HI[2]),
        hartree_to_eV(LAMBDA_LO), hartree_to_eV(LAMBDA_HI))
@printf("zero-model objective J₀ = %.1f (bound |ℬ|Σvᵢ = %.1f)\n\n",
        J₀, length(idx) * sum(v))

# ------------------------------------------------------------------------------------
# The fit
# ------------------------------------------------------------------------------------

t_fit = @elapsed fit = fit_amplitudes(ladder, ω, Λ; ε = EPSILON, anchor_weight = ANCHOR_W)
J = objective_value(ladder, fit, ω, Λ, v, σ)
E_signed = relative_errors(ladder, fit, ω, Λ)
E = abs.(E_signed)

@printf("fit: %d independent NNLS solves in %.2f s (%.1f ms/column)\n",
        length(s.x), t_fit, 1000 * t_fit / length(s.x))
@printf("J = %.1f, J/J₀ = %.4f\n\n", J, J / J₀)

# ------------------------------------------------------------------------------------
# 1. Where the error lives in ω
# ------------------------------------------------------------------------------------
#
# The headline number depends entirely on which band you quote it over, so quote all of
# them. Λ decays as ω⁻³ in the far tail and no Lorentzian sum follows past ω⁻², so the
# error there is structural — it is what the §3 floor exists to discount, not a defect.

band_count  = [count(lo .< ωeV .<= hi) for (lo, hi) in BANDS]
band_median = [median_(E[lo .< ωeV .<= hi, :]) for (lo, hi) in BANDS]

println("median |relative error| by frequency band")
println("  band [eV]        n     median")
for (b, (lo, hi)) in enumerate(BANDS)
    @printf("  (%4.1f, %4.1f]  %5d  %9.2f%%\n", lo, hi, band_count[b], 100 * band_median[b])
end
@printf("  %-14s %5d  %9.2f%%\n\n", "whole band", length(ω), 100 * median_(E))

# ------------------------------------------------------------------------------------
# 1b. Where the error lives in x
# ------------------------------------------------------------------------------------
#
# The section above medians over all 254 columns, which is exactly how the crossing
# column hid: it is one column in 254, so a 95% failure there moved the median by
# nothing. Report it on its own.

i_res = argmin(abs.(s.x .- s.x_resonance))
per_column = [median_(E[low, i]) for i in axes(E, 2)]
window = 1.5 .<= s.x .<= 3.0

println("median |relative error| below 1 eV, by configuration")
@printf("  at the crossing x₀ = %.4f Å  %8.2f%%   (Λ(x₀,0) = %.1f, the dataset maximum)\n",
        s.x[i_res], 100 * per_column[i_res], Λ[1, i_res])
@printf("  worst single column          %8.2f%%   (at x = %.3f Å)\n",
        100 * maximum(per_column), s.x[argmax(per_column)])
@printf("  window x ∈ [1.5, 3.0] Å      %8.2f%%   (%d columns, %.0f%% of the weight)\n",
        100 * median_(E[low, window]), count(window), 100 * sum(v[window]) / sum(v))
@printf("  all configurations           %8.2f%%\n\n", 100 * median_(E[low, :]))

# ------------------------------------------------------------------------------------
# 2. The Markovian anchor (§4) — what weighting it costs and buys
# ------------------------------------------------------------------------------------
#
# Λ(x,0) is the zero-frequency friction a standard MDEF run would use, and it is already
# one of the |ℬ| residuals since subsample_omega always keeps j = 1. The question is
# whether it is worth more than 1/|ℬ| of the fit. §4 recommends the equality; this is
# the measurement behind taking that up.

c = markovian_anchor_row(ladder)
anchor_error(m) = [abs(sum(c .* vcat(amplitudes(m, i)...)) - Λ[1, i]) / Λ[1, i]
                   for i in axes(Λ, 2)]

sweep_w      = [0.0, 1.0, 10.0, 100.0, 1000.0]
sweep_w_med  = similar(sweep_w)
sweep_w_max  = similar(sweep_w)
sweep_w_low  = similar(sweep_w)
for (k, w) in enumerate(sweep_w)
    m = fit_amplitudes(ladder, ω, Λ; ε = EPSILON, anchor_weight = w)
    a = anchor_error(m)
    e = abs.(relative_errors(ladder, m, ω, Λ))
    sweep_w_med[k], sweep_w_max[k] = median_(a), maximum(a)
    sweep_w_low[k] = median_(e[low, :])
end

println("Markovian anchor vs anchor_weight")
println("  weight    anchor median    anchor max    median |rel err| (ω ≤ 1 eV)")
for (k, w) in enumerate(sweep_w)
    @printf("  %6d  %14.4f%%  %11.2f%%  %25.2f%%\n",
            w, 100 * sweep_w_med[k], 100 * sweep_w_max[k], 100 * sweep_w_low[k])
end
println()

# ------------------------------------------------------------------------------------
# 3. Error vs rung budget K_C — the open item of §7
# ------------------------------------------------------------------------------------
#
# §5 expects a full-dictionary NNLS with 10–40 atoms to reach 0.1–5%. If the error
# plateaus then the limit is the model class, not the budget, and buying more rungs — two
# auxiliary variables each — is wasted.
#
# The three axes are not independent, so sweep the whole composition rather than one at a
# time. `(K_C_lo, K_C_hi, K_R)`, with the last two rows the ladders this file used to
# run, kept so the comparison that motivated the split stays in the record.
#
# The far tail is reported separately, over x ≥ 3 Å only. Those columns decay as ω⁻² and
# the model can follow them; the near-crossing columns decay as ω⁻³ and it cannot, so a
# tail number pooled over all x measures the model class rather than the ladder. The far
# columns are not a minority interest — they carry half the quadrature weight.
far   = s.x .>= 3.0
tail  = 10.0 .< ωeV .<= 60.0
compo = [(16, 4, 3), (16, 3, 3), (15, 4, 3), (14, 4, 3), (13, 3, 3),
         (16, 4, 12), (20, 0, 12), (20, 0, 0)]
sweep_KC     = [c[1] for c in compo]
sweep_KH     = [c[2] for c in compo]
sweep_KR     = [c[3] for c in compo]
sweep_KC_m    = zeros(Int, length(compo))
sweep_KC_low  = zeros(length(compo))
sweep_KC_res  = zeros(length(compo))
sweep_KC_tail = zeros(length(compo))
sweep_KC_J    = zeros(length(compo))
for (k, (KC, KH, KR)) in enumerate(compo)
    segs = KH == 0 ? [(BETA_LO..., KC)] : [(BETA_LO..., KC), (BETA_HI..., KH)]
    lad = split_ladder(segs, KR, LAMBDA_LO, KR > 3 ? BETA_LO[2] : LAMBDA_HI)
    m = fit_amplitudes(lad, ω, Λ; ε = EPSILON, anchor_weight = ANCHOR_W)
    e = relative_errors(lad, m, ω, Λ)
    sweep_KC_m[k]    = auxiliary_dimension(lad)
    sweep_KC_low[k]  = median_(abs.(e[low, :]))
    sweep_KC_res[k]  = median_(abs.(e[low, i_res]))
    sweep_KC_tail[k] = median_(e[tail, far])          # signed: the sign is the finding
    sweep_KC_J[k]    = objective_value(lad, m, ω, Λ, v, σ) / J₀
end

println("error vs ladder composition")
println("  K_C_lo  K_C_hi   K_R     m     all-x      at x₀   tail x≥3      J/J₀")
for k in eachindex(compo)
    @printf("  %6d  %6d  %4d  %4d  %8.2f%%  %8.2f%%  %8.1f%%  %8.4f\n",
            sweep_KC[k], sweep_KH[k], sweep_KR[k], sweep_KC_m[k],
            100 * sweep_KC_low[k], 100 * sweep_KC_res[k],
            100 * sweep_KC_tail[k], sweep_KC_J[k])
end
println()

# ------------------------------------------------------------------------------------
# Save
# ------------------------------------------------------------------------------------
#
# Amplitudes in the (a₀, a, A) column order of `design_matrix`, alongside the ladder
# that produced them — the pair is meaningless split up. Atomic units throughout
# (design/0001); x stays in Å.

# The whole ladder belongs in the name, not just the counts: two fits with the same
# (K_C, K_R) but different segment spans are different models, and without the spans the
# second silently overwrites the first. `m` leads because it is the number the rest of
# the scheme is paid in — the size of Γ₂,₂ and of the auxiliary state the SDE carries.
# `sort = false` keeps that reading order; DrWatson would otherwise alphabetise it away.
eV3(x) = round(hartree_to_eV(x); sigdigits = 3)
fit_id = (Gamma_eV = GAMMA_EV,
          m        = auxiliary_dimension(ladder),
          KC       = "$(K_C_LO)+$(K_C_HI)",
          K_R      = K_R,
          blo_eV   = "$(eV3(BETA_LO[1]))-$(eV3(BETA_LO[2]))",
          bhi_eV   = "$(eV3(BETA_HI[1]))-$(eV3(BETA_HI[2]))",
          lam_eV   = "$(eV3(LAMBDA_LO))-$(eV3(LAMBDA_HI))",
          eps      = EPSILON,
          anchor   = ANCHOR_W)

# The representation is part of the model's identity, not a detail: the same ladder
# fitted as a grid and as a Chebyshev expansion are different fits and must not collide.
# `fit_tag` derives it from the type actually fitted, so the name cannot drift.
outfile = datadir("fits", savename("prony_$(fit_tag(fit))", fit_id, "h5"; sort = false))
mkpath(dirname(outfile))
h5open(outfile, "w") do f
    f["a0_au"]      = fit.a₀
    f["a_au"]       = fit.a
    f["A_au"]       = fit.A
    f["x_Ang"]      = s.x
    f["lambda_Ha"]  = ladder.λ
    f["alpha_Ha"]   = ladder.α
    f["beta_Ha"]    = ladder.β
    f["omega_Ha"]   = ω
    attrs(f)["Gamma_eV"]      = GAMMA_EV
    attrs(f)["K_C_lo"]        = K_C_LO
    attrs(f)["K_C_hi"]        = K_C_HI
    attrs(f)["K_C"]           = K_C_LO + K_C_HI
    attrs(f)["K_R"]           = K_R
    attrs(f)["beta_lo_eV"]    = [eV3(BETA_LO[1]), eV3(BETA_LO[2])]
    attrs(f)["beta_hi_eV"]    = [eV3(BETA_HI[1]), eV3(BETA_HI[2])]
    attrs(f)["lambda_eV"]     = [eV3(LAMBDA_LO), eV3(LAMBDA_HI)]
    attrs(f)["m_auxiliary"]   = auxiliary_dimension(ladder)
    attrs(f)["epsilon"]       = EPSILON
    attrs(f)["anchor_weight"] = ANCHOR_W
    attrs(f)["mu"]            = 0.0
    attrs(f)["J"]             = J
    attrs(f)["J_over_J0"]     = J / J₀
    attrs(f)["units"]         = "amplitudes and ω in atomic units (Ha); x in Å"
    attrs(f)["column_order"]  = "amplitude vector per node is (a₀, a, A), as design_matrix"
    attrs(f)["amplitude_model"] = string(nameof(typeof(fit)))
    attrs(f)["fit_tag"]       = fit_tag(fit)
    attrs(f)["stage"]         = "stage 1 (grid amplitudes, μ = 0, decoupled per column)"
    attrs(f)["git"]           = gitdescribe(projectdir())

    # Error diagnostics travel with the fit. A set of amplitudes without a record of
    # how well it fits, and where, is not a result — and the error here is strongly
    # ω-dependent, so no single headline number stands in for it.
    R    = Λ .+ E_signed .* abs.(Λ)                    # model values on the band
    Jcol = [v[i] * sum(((R[:, i] .- Λ[:, i]) ./ σ[:, i]) .^ 2) for i in axes(Λ, 2)]
    f["errors/relative"]         = E_signed            # signed, [|ℬ|, N_x]; abs for summaries
    f["errors/median_by_x"]      = [median_(abs.(E_signed[:, i])) for i in axes(E_signed, 2)]
    f["errors/median_by_omega"]  = [median_(abs.(E_signed[j, :])) for j in axes(E_signed, 1)]
    f["errors/band_lo_eV"]       = first.(BANDS)
    f["errors/band_hi_eV"]       = last.(BANDS)
    f["errors/band_count"]       = band_count
    f["errors/band_median"]      = band_median
    f["errors/anchor"]           = anchor_error(fit)
    f["errors/objective_by_x"]   = Jcol
    attrs(f)["errors_note"] = "relative errors carry no σ floor, unlike the objective: " *
        "the ω⁻³ tail the floor discounts appears here at full weight"

    # The two sweeps, so the findings outlive the terminal scrollback.
    f["sweeps/anchor_weight"]            = sweep_w
    f["sweeps/anchor_median"]            = sweep_w_med
    f["sweeps/anchor_max"]               = sweep_w_max
    f["sweeps/anchor_lowband_median"]    = sweep_w_low
    f["sweeps/K_C_lo"]                   = sweep_KC
    f["sweeps/K_C_hi"]                   = sweep_KH
    f["sweeps/K_R"]                      = sweep_KR
    f["sweeps/K_C_tail_far_median"]      = sweep_KC_tail
    f["sweeps/K_C_resonance_median"]     = sweep_KC_res
    f["errors/median_by_x_lowband"]      = per_column
    attrs(f)["x_resonance_Ang"]          = s.x_resonance
    attrs(f)["i_resonance"]              = i_res
    f["sweeps/K_C_m"]                    = sweep_KC_m
    f["sweeps/K_C_lowband_median"]       = sweep_KC_low
    f["sweeps/K_C_J_over_J0"]            = sweep_KC_J
end
println("saved: ", outfile)
