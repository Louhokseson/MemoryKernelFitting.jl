# Track B — the regime-(B) VARPRO diagnostic, at Γ = 0.1 eV.
#
#   julia -t auto --project=. scripts/fitting/fit_trackB_gamma01.jl
#
# **Run it threaded.** Given the poles the configurations are independent, so both hot
# loops — the inner NNLS of `tracking_solve` and the per-column products of the stage-2
# gradient — are parallel over the 254 columns. Measured on 10 cores: 18.7 → 5.0 ms per
# objective evaluation, a 3.7× speedup, and the objective is summed in index order so the
# value is **bit-identical** at any thread count. Sub-linear because each column's NNLS is
# only 561 × 11; more cores than columns buys nothing.
#
# This is **not** a production fit and its output is not a candidate model. §9.7 settles
# regime (B) negatively on this data — with the width floored at Δ + k_BT the adiabatic
# ratio τ_mem/τ_conf is 15.2 thermal and 133 hot at the crossing, and the crossing is the
# region of interest — so a tracking fit that scores better here would still not give an
# SDE that realises what was fitted (§9.5). `TrackingPoles` carries that warning too.
#
# It is run because three things the note asks for cannot be measured any other way, and
# the outer VARPRO search of §5 has never existed until now:
#
#   1. **The §7 step 2 regime diagnostic.** "Fit each x independently with free poles,
#      plot λ, α, β vs x." This is the constrained version of it: instead of 254
#      independent nonlinear fits it fits one smooth pole *law* in the physical features,
#      which is §6's own ansatz and far better determined. Reading the fitted `c` against
#      the measured ω_pk/|h| = 1.02–1.15, and `w, z` against the Δ + k_BT floor, turns
#      two assumptions into measurements.
#
#   2. **The open item "calibrate the achievable relative error vs K_C".** §5's greedy
#      numbers are a lower bound on quality, not the answer, and the fixed-ladder results
#      (1.41 % all-x, 8.2 % at x₀) have nothing to be read against. A tracking fit with
#      K_C ~ 1–3 is the closest thing to an upper bound on what any ladder could reach.
#
#   3. **Whether K_R is needed at all once α is free.** The real rates exist only because
#      `log_ladder` ties α_j = 0.2 β_j, so β → 0 at the crossing drags the width down with
#      it and no broad zero-centred peak can be made (`docs/design/0003`, the 2026-08-18
#      correction). §6's α(x) = u|h| + wΔ + z·k_BT has no such tie — at h = 0 it falls
#      back on Δ + k_BT ≈ 0.077 eV, which is the measured half-width there. So track B
#      should need K_R = 0. That is a falsifiable prediction and this sweep tests it.
#
# The tail stays fixed. |h| ≤ 8.65 eV, so nothing above ~9 eV is a resonance and there is
# nothing there to track; that band exists to set the ω⁻² coefficient of the far tail, and
# §3's band decomposition puts 56.2 % of the objective above 10 eV. Dropping it would not
# make a cheaper model, it would make an incomparable one. This is a split over *modes*,
# not over x — §9.7's Obstruction 2 concerns an x split and does not apply — and it does
# not make regime (B) legal either, since the tracking modes are the ones carrying x₀.

using DrWatson
@quickactivate "MemoryKernelFitting"

using LinearAlgebra: BLAS
# One BLAS thread. The parallelism that pays here is over columns, and the per-column
# matrices (561 × ~11) are far too small for BLAS threading to help; leaving both layers
# threaded oversubscribes the machine and is measurably slower.
BLAS.set_num_threads(1)

using MemoryKernelFitting
using MemoryKernelFitting.Fitting: tracking_solve
using HDF5
using Printf

function median_(v)
    s = sort(vec(collect(v)))
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

# ------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------

const GAMMA_EV  = 0.1
const EPSILON   = 1e-3           # §3 residual floor
const ANCHOR_W  = 1e3            # §4 Markovian anchor, as a weighted row
const THETA_V   = 0.5            # configuration_weights(θ) — see below
# 20000, not 4000. Checked directly: at 4000 the K_C=3 rows hit the cap, and re-running
# them at 20000 changes the reported errors by ≤ 0.01 pp (K_C=3 K_R=1: 51.66 % at x₀
# either way, converging at 8693 evaluations). The budget buys a table without caveats,
# not a different answer.
const ITERS     = 20_000

# The production ladder of `fit_gamma01.jl`, reproduced here as the regime-(A) baseline.
const BETA_LO   = (1.8e-3, 0.345)       # Ha — [0.049, 9.39] eV, the resonance band
const BETA_HI   = (0.441, 2.20496)      # Ha — [12.0, 60.0] eV, the tail band
const K_C_LO    = 16
const K_C_HI    = 4
const K_R_A     = 3
const LAMBDA_LO = 1.84e-4               # Ha — 5·dω, the §4 identifiability floor
const LAMBDA_HI = 1.84e-3               # Ha — 0.0501 eV, what the production fit uses

# Track B sweep. K_C here counts **tracking** pairs only; the four tail rungs are on top.
#
# Two parametrisations. The **free** law carries `5K_C + 3K_R` coefficients and is the more
# capable of the two — the tied family is a strict subset of it — but Nelder–Mead stagnates
# above ~20 dimensions, so it stops at K_C = 3. The **tied** law is 12 numbers whatever the
# mode count, which is what makes K_C = 8 reachable at all. K_C = 3 is run both ways, so the
# table reads off directly what the tie costs where both are available.
#
# K_R = 3 is dropped: the earlier sweep measured 0 / 1 / 3 as identical at x₀ (50.4 / 51.7 /
# 49.8 %) with K_R = 0 the best on all-x, so a third row buys nothing.
const SWEEP_K_C = (1, 2, 3)
const SWEEP_K_R = (0, 1)
const TIED_K_C  = (3, 5, 8)
const TIED_K_R  = (0, 1)

# Stage 2 on top of the best track-B fit. Degrees match `fit_stage2_gamma01.jl`'s upper
# range so the coefficient counts can be read against its 552; ITERS_B matches its 20000,
# which that script's own sweep shows is the difference between a converged number and one
# that merely stopped early.
const TRACKB_DEGREES = (10, 16, 22)
const ITERS_B        = 20_000

const DATAFILE = datadir("sims", "friction", "ErpenbeckThoss",
    "Lambda_spectrum_Gamma_eV=$(GAMMA_EV)_Nx=257_T_K=300.0_domega_eV=0.001.h5")
isfile(DATAFILE) || error("spectrum file not found: $DATAFILE")

# ------------------------------------------------------------------------------------
# Load
# ------------------------------------------------------------------------------------

s   = load_lambda_spectrum(DATAFILE)
idx = subsample_omega(s.ω)
ω   = s.ω[idx]
Λ   = s.Λ[idx, :]
σ   = residual_scales(Λ; ε = EPSILON)

# θ = 0.5 throughout, for both the fits and the objectives they are ranked by. Stage 1's
# columns decouple so `v` cancels from its argmin and only picks ladders; **track B's do
# not** — every column pulls on the same pole coefficients, so `v` is a genuine fitting
# weight here. DEVLOG's stage-2 table sets the precedent: everything that couples columns
# is quoted at θ = 0.5, and the stage-1 file's own J/J₀ = 0.135 (at θ = 0) is not
# comparable with it. Recomputed here at θ = 0.5, every row of the table below is.
v = configuration_weights(s; θ = THETA_V)

ωeV   = hartree_to_eV.(ω)
low   = ωeV .<= 1.0                       # the band the resonance lives in
i_res = argmin(abs.(s.x .- s.x_resonance))
win   = 1.5 .<= s.x .<= 3.0

# The zero model has R ≡ 0, so J₀ = Σᵢ vᵢ Σⱼ (Λᵢⱼ/σᵢⱼ)² does not depend on the ladder and
# one value normalises every row.
J₀ = objective_value(split_ladder([(BETA_LO..., K_C_LO)], 0, LAMBDA_LO, LAMBDA_HI),
                     GridAmplitudes(length(s.x), 0, K_C_LO), ω, Λ, v, σ)

# The width the crossing column actually has, from the file rather than from Γ/2.
Δ_x0_eV = hartree_to_eV(s.Δ[i_res] + s.kT)

@printf("Γ = %.2f eV | %d configurations | %d of %d frequencies retained | %d thread%s\n",
        hartree_to_eV(s.Γ), length(s.x), length(idx), length(s.ω),
        Threads.nthreads(), Threads.nthreads() == 1 ? " (pass -t auto)" : "s")
@printf("x₀ = %.4f Å (column %d) | Δ(x₀) + k_BT = %.4f eV | J₀ = %.1f at θ = %.1f\n\n",
        s.x[i_res], i_res, Δ_x0_eV, J₀, THETA_V)

# The four fixed tail rungs, lifted from the production ladder so that track B and the
# baseline share them exactly rather than approximately.
_full = split_ladder([(BETA_LO..., K_C_LO), (BETA_HI..., K_C_HI)], K_R_A,
                     LAMBDA_LO, LAMBDA_HI)
tail = PoleSet(Float64[], _full.α[(K_C_LO + 1):end], _full.β[(K_C_LO + 1):end])

# ------------------------------------------------------------------------------------
# Metrics — the columns of `docs/design/0003`'s tables, so the rows can be read together
# ------------------------------------------------------------------------------------

score(label, m_aux, E, J; note = "") =
    (; label, m = m_aux, allx = median_(E[low, :]), at_x0 = median_(E[low, i_res]),
       window = median_(E[low, win]), JJ0 = J / J₀, note)

# Fraction of exact zeros and how often the live set moves — the statistic DEVLOG reports
# for stage 1 (65.3 % zeros, live set changing across 118 of 253 gaps) and the reason it
# concludes the amplitude table is not compressible. If tracking poles make the table
# smooth, that is the M5/stage-2 finding this diagnostic is worth most for.
function support_stats(m)
    Θ = reduce(hcat, [vcat(amplitudes(m, i)...) for i in 1:n_configurations(m)])
    live = Θ .> 0
    changes = count(i -> live[:, i] != live[:, i + 1], 1:(size(live, 2) - 1))
    return (; zeros = count(iszero, Θ) / length(Θ), changes,
            gaps = size(live, 2) - 1, live_median = median_(sum(live; dims = 1)))
end

rows = Any[]
supports = Dict{String,Any}()

# ------------------------------------------------------------------------------------
# Regime (A) baselines
# ------------------------------------------------------------------------------------
#
# Two of them. The first is the production fit, recomputed here so its numbers sit in the
# same weighting as track B's. The second tests a discrepancy `fit_gamma01.jl`'s own
# comment points at: it says λ_max "belongs" at the crossing half-width Δ + k_BT ≈ 0.077
# eV "and not at β_max", but the constant it sets is 0.0501 eV, below it. A sum of
# zero-centred Lorentzians has half-width bounded by its largest λ, so rates capped at
# 0.05 eV cannot on their own build a 0.076 eV-wide peak. Whether that costs anything is
# measurable in one extra fit, and it fixes which baseline track B is compared against.

for (lab, λhi) in (("regime A · production", LAMBDA_HI),
                   ("regime A · λ_max = Δ+k_BT", eV_to_hartree(Δ_x0_eV)),
                   ("regime A · λ_max = 0.10 eV", eV_to_hartree(0.10)))
    lad = split_ladder([(BETA_LO..., K_C_LO), (BETA_HI..., K_C_HI)], K_R_A,
                       LAMBDA_LO, λhi)
    t = @elapsed fit = fit_amplitudes(lad, ω, Λ; ε = EPSILON, anchor_weight = ANCHOR_W)
    E = abs.(relative_errors(lad, fit, ω, Λ))
    J = objective_value(lad, fit, ω, Λ, v, σ)
    push!(rows, score(lab, auxiliary_dimension(lad), E, J;
                      note = @sprintf("λ ≤ %.4f eV", hartree_to_eV(λhi))))
    supports[lab] = support_stats(fit)
    @printf("  %-28s m = %2d   %.1f s\n", lab, auxiliary_dimension(lad), t)
end

# ------------------------------------------------------------------------------------
# Track B sweep
# ------------------------------------------------------------------------------------

println()
best = nothing
minimal = nothing
for K_R in SWEEP_K_R, K_C in SWEEP_K_C
    seed = tracking_seed(K_C, K_R, tail, LAMBDA_LO)
    t = @elapsed fit = fit_tracking(seed, ω, Λ, s.h, s.Δ, s.kT, v;
                                    ε = EPSILON, anchor_weight = ANCHOR_W,
                                    iterations = ITERS)
    E = abs.(tracking_errors(fit.poles, fit.model, ω, Λ, s.h, s.Δ, s.kT))
    # "3+4" not "3": the `m` column counts every pair, tracking and tail alike, so a
    # label naming only the tracking ones cannot be reconciled with it by eye. Same
    # convention as `fit_gamma01.jl`'s KC = "16+4".
    lab = @sprintf("track B · K_C=%d+%d K_R=%d", K_C, length(tail.α), K_R)
    push!(rows, score(lab, auxiliary_dimension(fit.poles), E, fit.J;
                      note = fit.converged ? "" : "not converged"))
    supports[lab] = support_stats(fit.model)
    @printf("  %-28s m = %2d   %5.0f s  %5d evals  %s\n", lab,
            auxiliary_dimension(fit.poles), t, fit.evaluations,
            fit.converged ? "converged" : "HIT ITERATION CAP")
    if best === nothing || fit.J < best.fit.J
        global best = (; lab, fit, E, K_C, K_R)
    end
    if K_C == 1 && K_R == 0
        global minimal = (; lab, fit)
    end
end

# ------------------------------------------------------------------------------------
# The tied law — the same pole law with the per-mode freedom removed
# ------------------------------------------------------------------------------------
#
# `K_C = 8` is chosen against a measurement, not a budget: the production ladder carries a
# **median of 8 live pairs per column** (min 3, max 12). It is not that the other rungs are
# waste — every one of the 20 is live in ≥10 % of columns — but that at any single x only
# about eight shapes are doing the work. A law whose poles move needs to carry the eight,
# not the twenty.

println()
for K_R in TIED_K_R, K_C in TIED_K_C
    seed = tied_seed(K_C, K_R)
    t = @elapsed fit = fit_tracking(seed, tail, LAMBDA_LO, ω, Λ, s.h, s.Δ, s.kT, v;
                                    ε = EPSILON, anchor_weight = ANCHOR_W,
                                    iterations = ITERS)
    E = abs.(tracking_errors(fit.poles, fit.model, ω, Λ, s.h, s.Δ, s.kT))
    lab = @sprintf("tied · K_C=%d+%d K_R=%d", K_C, length(tail.α), K_R)
    push!(rows, score(lab, auxiliary_dimension(fit.poles), E, fit.J;
                      note = fit.converged ? "12 par" : "12 par, not converged"))
    supports[lab] = support_stats(fit.model)
    @printf("  %-28s m = %2d   %5.0f s  %6d evals  %s\n", lab,
            auxiliary_dimension(fit.poles), t, fit.evaluations,
            fit.converged ? "converged" : "HIT ITERATION CAP")
    if fit.J < best.fit.J
        global best = (; lab, fit, E, K_C, K_R)
    end
end

# ------------------------------------------------------------------------------------
# The fourth corner — track B poles *and* a compressed amplitude basis
# ------------------------------------------------------------------------------------
#
# Stage 2 walks along the parameter axis at fixed m; the sweep above walks along the m
# axis at fixed parameter count. They are independent, so both can be taken. Initialised
# from the track-B grid fit at the same poles, which is what makes this stage 2 rather
# than a fresh nonconvex fit — the ordering `docs/design/0002` step 2 insists on.
#
# There are `1 + K_R + K_C` curves to compress here against 24 on the production ladder,
# so at equal degree the coefficient count is proportionally smaller. Whether they are
# *easier* to compress is the M5-relevant question: DEVLOG records that stage 1's table is
# not compressible because its support moves, and track B's support moves less.

println()
warp = warp_coordinate.(s.h, s.Δ, s.kT)
env  = shape_envelope(Λ)
n_c  = 1 + sum(mode_counts(best.fit.poles))
@printf("  best by J: %s\n", best.lab)

for d in TRACKB_DEGREES
    t = @elapsed m2, r2 = fit_tracking_basis(ChebyshevAmplitudes, best.fit.poles, ω, Λ,
                                             warp, env, s.h, s.Δ, s.kT;
                                             degree = d, v = v, ε = EPSILON,
                                             anchor_weight = ANCHOR_W,
                                             init = best.fit.model, iterations = ITERS_B)
    E = abs.(tracking_errors(best.fit.poles, m2, ω, Λ, s.h, s.Δ, s.kT))
    J = r2.J_final
    lab = @sprintf("  └ + Chebyshev deg %d", d)
    push!(rows, score(lab, auxiliary_dimension(best.fit.poles), E, J;
                      note = r2.converged ? @sprintf("%d coef", (d + 1) * n_c) :
                             @sprintf("%d coef, not converged", (d + 1) * n_c)))
    @printf("  %-28s %4d coef  %5.0f s  %6d iters  %s\n", lab, (d + 1) * n_c, t,
            r2.iterations, r2.converged ? "converged" : "HIT ITERATION CAP")
end

# ------------------------------------------------------------------------------------
# The table
# ------------------------------------------------------------------------------------

println("\nmedian |relative error| below 1 eV")
println("  " * "-"^88)
@printf("  %-28s %4s %9s %9s %9s %9s  %s\n",
        "", "m", "all-x", "at x₀", "window", "J/J₀", "")
println("  " * "-"^88)
for r in rows
    @printf("  %-28s %4d %8.2f%% %8.2f%% %8.2f%% %9.4f  %s\n",
            r.label, r.m, 100r.allx, 100r.at_x0, 100r.window, r.JJ0, r.note)
end
println("  " * "-"^88)
println("  all rows at configuration_weights(θ = $THETA_V), ε = $EPSILON, anchor = $ANCHOR_W")
@printf("  K_C is written tracking+tail; m = K_R + 2(K_C_track + %d), the %d tail rungs\n",
        length(tail.α), length(tail.α))
@printf("  costing %d of every m below and shared with the regime-A rows.\n",
        2 * length(tail.α))

# ------------------------------------------------------------------------------------
# What the fitted pole law says — §6's ansatz, measured rather than assumed
# ------------------------------------------------------------------------------------

# Read off the *minimal* model, not the best-scoring one. With one tracking pair and no
# real rates there are four parameters and `c` is exactly the fitted ω_pk/|h|; the larger
# models reach the same fit through coefficients that cancel against each other (see the
# conditioning note below), which is a fact about the ansatz, not a reading of the data.
r = tracking_report(minimal.fit.poles)
println("\nfitted pole law of $(minimal.lab) — the minimal tracking model")
@printf("  β_j(x) = c_j|h|            c = %s\n", string(round.(r.c; digits = 4)))
@printf("      §6 measures ω_pk/|h| = 1.02–1.15 across the grid\n")
@printf("  α_j(x) = u|h| + wΔ + z·k_BT\n")
@printf("      u = %s   (log_ladder ties α/β = 0.2)\n", string(round.(r.u; digits = 4)))
@printf("      w = %s   z = %s   (§9.7 floors the width at Δ + k_BT)\n",
        string(round.(r.w; digits = 4)), string(round.(r.z; digits = 4)))
if r.K_R > 0
    @printf("  λ_k(x) = ru|h| + rwΔ + rz·k_BT\n")
    @printf("      ru = %s  rw = %s  rz = %s\n", string(round.(r.ru; digits = 4)),
            string(round.(r.rw; digits = 4)), string(round.(r.rz; digits = 4)))
end
ps0 = poleset(minimal.fit.poles, 0.0, s.Δ[i_res], s.kT)
@printf("  at x₀: β = %s eV, α = %s eV  (measured half-width there: %.4f eV)\n",
        string(round.(hartree_to_eV.(ps0.β[1:r.K_C_track]); sigdigits = 3)),
        string(round.(hartree_to_eV.(ps0.α[1:r.K_C_track]); sigdigits = 3)), Δ_x0_eV)

# Every column's ladder has to pass the §4 box, not just one. Checked on the *tracking*
# modes only: the four tail rungs are shared with the baseline, and the top one sits at
# β = 60.0 eV against a grid whose last frequency is 59.999 eV, so it reads as marginally
# outside for both models and would swamp the count with a property of neither.
dω = s.ω[2] - s.ω[1]
n_tr = r.K_C_track
bad = count(eachindex(s.x)) do i
    ps = poleset(minimal.fit.poles, s.h[i], s.Δ[i], s.kT)
    !is_identifiable(PoleSet(ps.λ, ps.α[1:n_tr], ps.β[1:n_tr]), dω, s.ω[end])
end
@printf("  §4 identifiability box: %d of %d columns outside (tracking modes)\n",
        bad, length(s.x))

# §6's three features are not independent enough to be fitted separately. k_BT is a
# single constant over the whole grid, so `z·k_BT` is an intercept, and Δ spans only
# [1.4e-4, 0.05] eV against k_BT = 0.0259 eV — so `wΔ` and `z·k_BT` are nearly collinear
# and the larger models reach their optimum through large cancelling coefficients. Report
# the spread so the effect is visible rather than inferred from odd-looking numbers.
big = maximum(abs, vcat(parameters(best.fit.poles)))
@printf("  conditioning: largest |coefficient| in %s is %.1f (minimal model: %.2f)\n",
        best.lab, big, maximum(abs, parameters(minimal.fit.poles)))

# ------------------------------------------------------------------------------------
# Is the amplitude table compressible? — the M5 question
# ------------------------------------------------------------------------------------

println("\namplitude support (DEVLOG reports 65.3 % zeros and 118/253 moving gaps for stage 1)")
@printf("  %-28s %9s %9s %14s\n", "", "zeros", "live/col", "support moves")
for lab in ("regime A · production", best.lab)
    st = supports[lab]
    @printf("  %-28s %8.1f%% %9.1f %8d / %-4d\n",
            lab, 100 * st.zeros, st.live_median, st.changes, st.gaps)
end

# ------------------------------------------------------------------------------------
# Save
# ------------------------------------------------------------------------------------

out = datadir("fits", savename("prony_trackB",
    (Gamma_eV = GAMMA_EV, m = auxiliary_dimension(best.fit.poles),
     KCtrack = best.K_C, KCtail = length(tail.α), K_R = best.K_R,
     eps = EPSILON, anchor = ANCHOR_W, thetav = THETA_V), "h5"; sort = false))
mkpath(dirname(out))
h5open(out, "w") do f
    # `r` reports the *minimal* model — the saved fit is `best`, so its own law has to be
    # re-read here or the file would mix two different models under one name.
    rb = tracking_report(best.fit.poles)
    f["coefficients"] = parameters(best.fit.poles)
    f["c"], f["u"], f["w"], f["z"] = rb.c, rb.u, rb.w, rb.z
    f["ru"], f["rw"], f["rz"] = rb.ru, rb.rw, rb.rz
    f["tail_alpha_Ha"], f["tail_beta_Ha"] = tail.α, tail.β
    f["a0_au"] = best.fit.model.a₀
    f["a_au"], f["A_au"] = best.fit.model.a, best.fit.model.A
    f["omega_Ha"], f["x_Ang"] = ω, s.x
    f["h_Ha"], f["Delta_Ha"] = s.h, s.Δ
    f["errors/relative"] = tracking_errors(best.fit.poles, best.fit.model, ω, Λ,
                                           s.h, s.Δ, s.kT)
    attrs(f)["J"], attrs(f)["J_over_J0"] = best.fit.J, best.fit.J / J₀
    attrs(f)["m_auxiliary"] = auxiliary_dimension(best.fit.poles)
    attrs(f)["kT_Ha"] = s.kT
    attrs(f)["converged"] = best.fit.converged
    attrs(f)["regime"] = "B — diagnostic only, see §9.7 and TrackingPoles"
end
@printf("\nsaved %s\n", basename(out))
