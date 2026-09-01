# Stage-2 Prony fit of the Erpenbeck–Thoss spectral density at Γ = 0.1 eV.
#
#   julia --project=. scripts/fitting/fit_stage2_gamma01.jl
#
# Compression. `scripts/fitting/fit_gamma01.jl` produced 254 independent amplitude
# vectors — a table, not a function, so there is no Γ̃₂,₁(q) to hand an integrator. This
# script replaces the table with a Chebyshev expansion of the shape function in the
# warped coordinate, which *is* a function and can be evaluated anywhere on the range.
#
# `docs/design/0002-amplitude-representation.md` fixes the order: stage 1 first, then the
# basis, because you cannot choose a basis for ĝ_c(x) before you have seen it. This file
# reads the stage-1 fit and uses it as the initialiser — that is what "having seen it"
# means operationally.
#
# The measurement, not just the fit: the degree sweep is the point. §6 predicts a *low*
# degree is ample once the envelope and the warp are divided out, and the polynomial
# control at the end is what `docs/design/0002` has been promising as a measured result.

using DrWatson
@quickactivate "MemoryKernelFitting"

using MemoryKernelFitting
using HDF5
using Printf

median_(v) = (t = sort(vec(collect(v))); n = length(t);
              isodd(n) ? t[(n + 1) ÷ 2] : (t[n ÷ 2] + t[n ÷ 2 + 1]) / 2)

# ------------------------------------------------------------------------------------
# Settings — the ladder must match the stage-1 fit exactly, or the initialiser is a
# different model's answer. They are read back from the file rather than retyped.
# ------------------------------------------------------------------------------------

const GAMMA_EV  = 0.1
const EPSILON   = 1e-3          # matched to stage 1; a different ε rescales the objective
const ANCHOR_W  = 1e3
const THETA_V   = 0.5           # configuration weighting, §3 / configuration_weights
const DEGREES   = [6, 10, 14, 18, 22]
const DEGREE    = 22            # the one that is saved; chosen from the sweep below
const ITERS     = 20_000        # see the convergence note in the report below
const BANDS     = [(0.0, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, 60.0)]

const DATAFILE  = datadir("sims", "friction", "ErpenbeckThoss",
    "Lambda_spectrum_Gamma_eV=$(GAMMA_EV)_Nx=257_T_K=300.0_domega_eV=0.001.h5")

# The stage-1 fit to compress. Whichever grid fit is newest for this Γ — the ladder,
# ε and anchor all travel with it, so nothing here needs to agree with it by hand.
stage1_candidates = filter(readdir(datadir("fits"); join = true)) do f
    endswith(f, ".h5") && occursin("prony_stage1Grid", basename(f)) &&
        occursin("Gamma_eV=$(GAMMA_EV)_", basename(f))
end
isempty(stage1_candidates) && error("""
    no stage-1 grid fit found for Γ = $GAMMA_EV eV in $(datadir("fits")).
    Run scripts/fitting/fit_gamma01.jl first — stage 2 is a compression of stage 1's
    answer, not an independent fit.""")
const STAGE1FILE = argmax(mtime, stage1_candidates)

isfile(DATAFILE) || error("spectrum file not found: $DATAFILE")

# ------------------------------------------------------------------------------------
# Load
# ------------------------------------------------------------------------------------

s   = load_lambda_spectrum(DATAFILE)              # drops x < 0.8 Å, as stage 1 did
idx = subsample_omega(s.ω)
ω   = s.ω[idx]
Λ   = s.Λ[idx, :]
σ   = residual_scales(Λ; ε = EPSILON)
v   = configuration_weights(s; θ = THETA_V)

grid1, ladder, meta = h5open(STAGE1FILE, "r") do f
    lad = PoleSet(read(f["lambda_Ha"]), read(f["alpha_Ha"]), read(f["beta_Ha"]))
    (GridAmplitudes(read(f["a0_au"]), read(f["a_au"]), read(f["A_au"])), lad,
     (; x = read(f["x_Ang"]), ω = read(f["omega_Ha"]),
        J_over_J0 = read_attribute(f, "J_over_J0")))
end

meta.x ≈ s.x || error("the stage-1 fit was made on a different configuration grid")
meta.ω ≈ ω   || error("the stage-1 fit used a different retained band ℬ")

K_R, K_C = mode_counts(grid1)
n_c = 1 + K_R + K_C

# The two ingredients §6 factors out before any expansion happens. Both are data, not
# parameters: the envelope is read off Λ(x,0), the warp off the analytic h(x) and Δ(x).
env  = shape_envelope(Λ)
warp = warp_coordinate.(s.h, s.Δ, s.kT)

zero_model = GridAmplitudes(length(s.x), K_R, K_C)
J₀ = objective_value(ladder, zero_model, ω, Λ, v, σ)
J1 = objective_value(ladder, grid1, ω, Λ, v, σ)
E1 = abs.(relative_errors(ladder, grid1, ω, Λ))

ωeV = hartree_to_eV.(ω)
low = ωeV .<= 1.0
i_res = argmin(abs.(s.x .- s.x_resonance))

@printf("Γ = %.2f eV | %d configurations | %d of %d frequencies retained\n",
        hartree_to_eV(s.Γ), length(s.x), length(idx), length(s.ω))
@printf("ladder: %d pairs + %d real rates, m = %d (from %s)\n",
        K_C, K_R, auxiliary_dimension(ladder), basename(STAGE1FILE))
@printf("weights: configuration_weights(θ = %.1f); ε = %g; anchor = %g\n", THETA_V, EPSILON, ANCHOR_W)
@printf("warp s(x) ∈ [%.3f, %.3f] rad; envelope √Λ(x,0) ∈ [%.3g, %.3g]\n",
        minimum(warp), maximum(warp), minimum(env), maximum(env))
# Every J on this page is recomputed here under configuration_weights(θ = THETA_V),
# including stage 1's. `fit_gamma01.jl` used trapezoid_weights (θ = 0) and its file says
# so, hence a different number for the same fit — %.4f here against %.4f there. Only the
# numbers on this page compare with each other; §3's weighting is not a detail of
# reporting, it changes which columns the objective is about.
@printf("\nstage 1 (the table being compressed): %d free numbers, J/J₀ = %.4f\n",
        length(s.x) * n_c, J1 / J₀)
@printf("  (its own file records J/J₀ = %.4f — that is trapezoid_weights, θ = 0; " *
        "every J here is θ = %.1f)\n", meta.J_over_J0, THETA_V)

# ------------------------------------------------------------------------------------
# 1. Degree sweep — the measurement §6 asks for
# ------------------------------------------------------------------------------------
#
# "A low-degree polynomial is fine" is the claim the envelope and the warp are supposed
# to buy. This is what tests it. `J_proj` is the objective at the least-squares
# projection of the stage-1 curve, before any re-optimisation, so the gap to `J` says
# whether step 2's "re-optimise the coefficients" earns its place or whether projecting
# would have been enough.

println("\ndegree sweep — Chebyshev in the warped coordinate")
println("  deg   params   J_proj/J₀    J/J₀   ratio to stage 1   med|rel| ω≤1eV   at x₀   iters")
sweep = NamedTuple[]
for d in DEGREES
    m, r = fit_basis_report(ChebyshevAmplitudes, ladder, ω, Λ, warp, env;
                            degree = d, v = v, ε = EPSILON,
                            anchor_weight = ANCHOR_W, init = grid1, iterations = ITERS)
    J = objective_value(ladder, m, ω, Λ, v, σ)
    e = abs.(relative_errors(ladder, m, ω, Λ))
    push!(sweep, (; d, np = (d + 1) * n_c, Jproj = r.J_initial, J,
                    low = median_(e[low, :]), res = median_(e[low, i_res]),
                    iters = r.iterations, conv = r.converged))
    @printf("  %3d   %6d   %9.4f  %8.4f   %14.2f×   %13.2f%%  %6.2f%%  %5d%s\n",
            d, (d + 1) * n_c, r.J_initial / J₀, J / J₀, J / J1,
            100 * median_(e[low, :]), 100 * median_(e[low, i_res]),
            r.iterations, r.converged ? "" : " *")
end
any(!s.conv for s in sweep) && println("  * did not meet the gradient tolerance")

# ------------------------------------------------------------------------------------
# 2. The saved fit
# ------------------------------------------------------------------------------------

t_fit = @elapsed fit, report = fit_basis_report(
    ChebyshevAmplitudes, ladder, ω, Λ, warp, env;
    degree = DEGREE, v = v, ε = EPSILON, anchor_weight = ANCHOR_W, init = grid1,
    iterations = ITERS)

J = objective_value(ladder, fit, ω, Λ, v, σ)
E_signed = relative_errors(ladder, fit, ω, Λ)
E = abs.(E_signed)
C = reshape(parameters(fit), DEGREE + 1, n_c)

@printf("\nsaved fit: degree %d, %d coefficients, %.2f s, %d L-BFGS iterations (converged: %s)\n",
        DEGREE, length(C), t_fit, report.iterations, report.converged)
@printf("compression: %d → %d free parameters (%.1f×)\n",
        length(s.x) * n_c, length(C), length(s.x) * n_c / length(C))
@printf("J/J₀ = %.4f   (stage 1: %.4f — stage 2 is %.2f× its objective)\n",
        J / J₀, J1 / J₀, J / J1)

band_count  = [count(lo .< ωeV .<= hi) for (lo, hi) in BANDS]
band_median = [median_(E[lo .< ωeV .<= hi, :]) for (lo, hi) in BANDS]
band_med1   = [median_(E1[lo .< ωeV .<= hi, :]) for (lo, hi) in BANDS]

println("\nmedian |relative error| by frequency band")
println("  band [eV]        n    stage 2    stage 1")
for (b, (lo, hi)) in enumerate(BANDS)
    @printf("  (%4.1f, %4.1f]  %5d  %8.2f%%  %8.2f%%\n",
            lo, hi, band_count[b], 100 * band_median[b], 100 * band_med1[b])
end
@printf("  %-14s %5d  %8.2f%%  %8.2f%%\n\n", "whole band", length(ω),
        100 * median_(E), 100 * median_(E1))

per_column = [median_(E[low, i]) for i in axes(E, 2)]
window = 1.5 .<= s.x .<= 3.0
println("median |relative error| below 1 eV, by configuration")
@printf("  at the crossing x₀ = %.4f Å  %8.2f%%   (stage 1: %.2f%%)\n",
        s.x[i_res], 100 * per_column[i_res], 100 * median_(E1[low, i_res]))
@printf("  worst single column          %8.2f%%   (at x = %.3f Å)\n",
        100 * maximum(per_column), s.x[argmax(per_column)])
@printf("  window x ∈ [1.5, 3.0] Å      %8.2f%%   (%d columns)\n",
        100 * median_(E[low, window]), count(window))
@printf("  all configurations           %8.2f%%   (stage 1: %.2f%%)\n\n",
        100 * median_(E[low, :]), 100 * median_(E1[low, :]))

# ------------------------------------------------------------------------------------
# 3. The polynomial control — design/0002's open promise
# ------------------------------------------------------------------------------------
#
# §6 predicts monomials in raw x will ring: degree ≳ 10 is Runge-unstable and dx varies
# by 290× across this grid. `PolynomialAmplitudes` exists so that prediction is measured
# rather than asserted. Same solver, same initialiser, same everything — only (u, basis)
# differs, which is the whole content of the claim.

println("control: monomials in raw x, same solver and initialiser")
println("  deg    J/J₀   ratio to Chebyshev   med|rel| ω≤1eV   max|C| coefficient")
control = NamedTuple[]
for d in DEGREES[1:3]
    local mp, rp
    try
        mp, rp = fit_basis_report(PolynomialAmplitudes, ladder, ω, Λ, s.x, env;
                                  degree = d, v = v, ε = EPSILON,
                                  anchor_weight = ANCHOR_W, init = grid1, iterations = ITERS)
    catch err
        @printf("  %3d   failed: %s\n", d, sprint(showerror, err))
        continue
    end
    Jp = objective_value(ladder, mp, ω, Λ, v, σ)
    ep = abs.(relative_errors(ladder, mp, ω, Λ))
    Jc = sweep[findfirst(t -> t.d == d, sweep)].J
    push!(control, (; d, J = Jp, low = median_(ep[low, :]),
                      cmax = maximum(abs, parameters(mp))))
    @printf("  %3d  %7.4f   %16.2f×   %13.2f%%   %.3g\n",
            d, Jp / J₀, Jp / Jc, 100 * median_(ep[low, :]), maximum(abs, parameters(mp)))
end

# ------------------------------------------------------------------------------------
# Save
# ------------------------------------------------------------------------------------
#
# Both representations of the same fit travel together: the coefficients C (the thing
# that makes Γ̃₂,₁(q) evaluable anywhere) and the amplitudes evaluated at the nodes (what
# every reader already understands, and what the plot scripts consume). Storing only C
# would make every downstream file re-implement the basis.

Θ = hcat([collect(vcat(amplitudes(fit, i)...)) for i in 1:length(s.x)]...)

eV3(x) = round(hartree_to_eV(x); sigdigits = 3)
fit_id = (Gamma_eV = GAMMA_EV,
          m        = auxiliary_dimension(ladder),
          K_C      = K_C,
          K_R      = K_R,
          deg      = DEGREE,
          eps      = EPSILON,
          anchor   = ANCHOR_W,
          thetav   = THETA_V)

outfile = datadir("fits", savename("prony_$(fit_tag(fit))", fit_id, "h5"; sort = false))
mkpath(dirname(outfile))
h5open(outfile, "w") do f
    # Node-evaluated amplitudes, in the (a₀, a, A) order of design_matrix — the same
    # layout the stage-1 file uses, so readers need no special case.
    f["a0_au"]     = Θ[1, :]
    f["a_au"]      = Θ[2:(1 + K_R), :]
    f["A_au"]      = Θ[(2 + K_R):n_c, :]
    # The representation itself.
    f["C_cheb"]    = C
    f["s_warp"]    = warp
    f["t_cheb"]    = (2 / π) .* warp
    f["envelope"]  = env
    f["C_initial"] = report.C_initial
    f["x_Ang"]     = s.x
    f["lambda_Ha"] = ladder.λ
    f["alpha_Ha"]  = ladder.α
    f["beta_Ha"]   = ladder.β
    f["omega_Ha"]  = ω
    f["v_weights"] = v

    attrs(f)["Gamma_eV"]        = GAMMA_EV
    attrs(f)["amplitude_model"] = string(nameof(typeof(fit)))
    attrs(f)["fit_tag"]         = fit_tag(fit)
    attrs(f)["stage"]           = "stage 2 (Chebyshev in the warped coordinate, " *
                                  "degree $DEGREE, initialised from stage 1)"
    attrs(f)["degree"]          = DEGREE
    attrs(f)["K_C"]             = K_C
    attrs(f)["K_R"]             = K_R
    attrs(f)["m_auxiliary"]     = auxiliary_dimension(ladder)
    attrs(f)["epsilon"]         = EPSILON
    attrs(f)["anchor_weight"]   = ANCHOR_W
    attrs(f)["theta_v"]         = THETA_V
    attrs(f)["J_weighting"]     = "configuration_weights(θ = $THETA_V); the stage-1 file " *
                                  "used trapezoid_weights (θ = 0), so its own J_over_J0 " *
                                  "is not comparable — J_stage1 here is recomputed at θ = $THETA_V"
    attrs(f)["mu"]              = 0.0
    attrs(f)["J"]               = J
    attrs(f)["J_over_J0"]       = J / J₀
    attrs(f)["J_stage1"]        = J1
    attrs(f)["J_over_stage1"]   = J / J1
    attrs(f)["n_parameters"]    = length(C)
    attrs(f)["n_parameters_stage1"] = length(s.x) * n_c
    attrs(f)["converged"]       = report.converged
    attrs(f)["iterations"]      = report.iterations
    attrs(f)["stage1_file"]     = basename(STAGE1FILE)
    attrs(f)["units"]           = "amplitudes and ω in atomic units (Ha); x in Å; warp in rad"
    attrs(f)["column_order"]    = "amplitude vector per node is (a₀, a, A), as design_matrix"
    attrs(f)["basis_note"]      = "a_c(x) = (envelope[i] * Σ_l C[l+1,c] T_l(t[i]))^2, " *
                                  "t = 2 s/π, s = atan(h/(Δ+kT))"
    attrs(f)["git"]             = gitdescribe(projectdir())

    f["errors/relative"]        = E_signed
    f["errors/median_by_x"]     = [median_(abs.(E_signed[:, i])) for i in axes(E_signed, 2)]
    f["errors/median_by_omega"] = [median_(abs.(E_signed[j, :])) for j in axes(E_signed, 1)]
    f["errors/median_by_x_lowband"] = per_column
    f["errors/band_lo_eV"]      = first.(BANDS)
    f["errors/band_hi_eV"]      = last.(BANDS)
    f["errors/band_count"]      = band_count
    f["errors/band_median"]     = band_median
    f["errors/band_median_stage1"] = band_med1
    c_anchor = markovian_anchor_row(ladder)
    f["errors/anchor"] = [abs(sum(c_anchor .* view(Θ, :, i)) - Λ[1, i]) / Λ[1, i]
                          for i in axes(Λ, 2)]
    attrs(f)["x_resonance_Ang"] = s.x_resonance
    attrs(f)["i_resonance"]     = i_res

    f["sweeps/degree"]          = [t.d for t in sweep]
    f["sweeps/n_parameters"]    = [t.np for t in sweep]
    f["sweeps/J_projection"]    = [t.Jproj for t in sweep]
    f["sweeps/J"]               = [t.J for t in sweep]
    f["sweeps/J_over_J0"]       = [t.J / J₀ for t in sweep]
    f["sweeps/lowband_median"]  = [t.low for t in sweep]
    f["sweeps/resonance_median"] = [t.res for t in sweep]
    if !isempty(control)
        f["control/degree"]     = [t.d for t in control]
        f["control/J_over_J0"]  = [t.J / J₀ for t in control]
        f["control/lowband_median"] = [t.low for t in control]
        f["control/max_abs_coefficient"] = [t.cmax for t in control]
        attrs(f)["control_note"] = "PolynomialAmplitudes in raw x — the design/0002 " *
                                   "control experiment, same solver and initialiser"
    end
end
println("\nsaved: ", outfile)
