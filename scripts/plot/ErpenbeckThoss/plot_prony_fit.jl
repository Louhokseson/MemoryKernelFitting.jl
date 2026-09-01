# A Prony fit against the Erpenbeck–Thoss spectral density, at Γ = 0.1 eV.
#
# Reads either stage: set AMPLITUDES below. Both stages write the same node-evaluated
# (a₀, a, A) layout, so every panel here is stage-agnostic — but a stage-2 file also
# carries its coefficients, and when it does the model plotted is the actual
# ChebyshevAmplitudes object, not a grid re-reading of its values at the nodes.
#
#   julia scripts/plot/ErpenbeckThoss/plot_prony_fit.jl
#
# Reads the data and the saved fit; fits nothing itself. The `scripts/fitting/` scripts are
# the producers, this is the reader — so what is drawn here is exactly the artefact that goes
# on to build Γ₂,₂, not a re-derivation that could silently drift from it.
#
# Two figures:
#
#   spectrum_and_fit_*.pdf  Λ(x,ω) data vs model over the five decades of ω the fit
#                           spans, the signed error, the Markovian anchor Λ(x,0), and
#                           where in (x, ω) the error actually lives.
#   prony_modes_*.pdf       the same fit taken apart: the individual Lorentzians that
#                           sum to it, and the amplitudes on the fixed pole ladder.
#
# Encoding. Colour carries the configuration x and nothing else, as five distinguishable
# hues (Okabe–Ito) rather than steps of one ramp — with only five curves, telling them
# apart matters more than reading their order off the colour, and the legend keeps the
# order anyway. Marker shape repeats that identity so no reading depends on hue alone.
# Data vs model is separate again: markers are data, solid lines are the fit. The two
# heatmaps show a magnitude, not an identity, so those keep a sequential ramp (batlow).

using DrWatson
@quickactivate "MemoryKernelFitting"

using MemoryKernelFitting
using CairoMakie
using ColorSchemes
using HDF5
using Printf

CairoMakie.activate!(type = "pdf")

# ------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------

# These name the fit to plot and must match the settings `scripts/fitting/fit_gamma01.jl`
# ran with — the whole ladder, not just the counts, since two fits differing only in a
# segment span are different models and get different filenames. A mismatch is a
# missing-file error below, listing what is actually on disk, not a silently wrong figure.
const GAMMA_EV  = 0.1
const K_C_LO    = 16
const K_C_HI    = 4
const K_R       = 3
const BETA_LO   = (1.8e-3, 0.345)       # Ha — [0.049, 9.39] eV
const BETA_HI   = (0.441, 2.20496)      # Ha — [12.0, 60.0] eV
const LAMBDA_LO = 1.84e-4               # Ha
const LAMBDA_HI = 1.84e-3               # Ha
const EPSILON   = 1e-3
const ANCHOR_W  = 1e3
const M_AUX     = K_R + 2 * (K_C_LO + K_C_HI)

const DATAFILE = datadir("sims", "friction", "ErpenbeckThoss",
    "Lambda_spectrum_Gamma_eV=$(GAMMA_EV)_Nx=257_T_K=300.0_domega_eV=0.001.h5")

eV3(x) = round(hartree_to_eV(x); sigdigits = 3)
fit_id = (Gamma_eV = GAMMA_EV,
          m        = M_AUX,
          KC       = "$(K_C_LO)+$(K_C_HI)",
          K_R      = K_R,
          blo_eV   = "$(eV3(BETA_LO[1]))-$(eV3(BETA_LO[2]))",
          bhi_eV   = "$(eV3(BETA_HI[1]))-$(eV3(BETA_HI[2]))",
          lam_eV   = "$(eV3(LAMBDA_LO))-$(eV3(LAMBDA_HI))",
          eps      = EPSILON,
          anchor   = ANCHOR_W)
# `fit_tag` on the *type* — the reader must name the file before it has a model to
# inspect. Flip this constant to switch stages; the id below follows it, because a
# stage-2 fit is identified by its degree and weighting, not by nothing at all.
const AMPLITUDES = ChebyshevAmplitudes
const DEGREE     = 22           # stage 2 only
const THETA_V    = 0.5          # stage 2 only — configuration_weights(θ)

const stage2_id = (Gamma_eV = GAMMA_EV, m = M_AUX, K_C = K_C_LO + K_C_HI, K_R = K_R,
                   deg = DEGREE, eps = EPSILON, anchor = ANCHOR_W, thetav = THETA_V)
const FITFILE = datadir("fits", savename("prony_$(fit_tag(AMPLITUDES))",
    AMPLITUDES === GridAmplitudes ? fit_id : stage2_id, "h5"; sort = false))

# Figures carry the ladder's identity too, so a figure can never be mistaken for one made
# from a different fit. Shorter than the .h5 name — that file holds the full provenance.
const FIGTAG = fit_tag(AMPLITUDES) * "_" *
    savename(AMPLITUDES === GridAmplitudes ?
             (Gamma_eV = GAMMA_EV, m = M_AUX, KC = fit_id.KC, K_R = K_R) :
             (Gamma_eV = GAMMA_EV, m = M_AUX, KC = fit_id.KC, K_R = K_R, deg = DEGREE);
             sort = false)

# Configurations to draw as curves. Chosen to walk the resonance in: the peak sits at
# ω_pk ≈ |h(x)|, which sweeps down to zero at the Fermi crossing x₀ = 2.02 Å and back up
# beyond it. x₀ itself is substituted for the nearest node below, so the crossing column
# — the tallest in the dataset, and the one the K_R real rates exist for — is always
# among them.
# 1.9 and 2.1 bracket the crossing closely — both sit inside the ±0.15 Å refinement
# patch, so they land on real nodes rather than the 0.1 Å baseline.
const X_SHOWN_ANG = [1.0, 1.9, 2.0203, 2.3, 5.0]

# Line/marker sizes are the final on-page sizes: the figures are authored at the width
# they are included at.
const FIG_W       = 900
const LINE_WIDTH  = 2.0
const MARKER_SIZE = 7
const N_MARKERS   = 45          # data markers per curve, log-spaced

# ------------------------------------------------------------------------------------
# Load
# ------------------------------------------------------------------------------------

isfile(DATAFILE) || error("spectrum file not found: $DATAFILE")
isfile(FITFILE)  || error("""
    fit file not found: $(basename(FITFILE))
    run scripts/fitting/fit_gamma01.jl with these settings, or point the block above at
    one of the fits already in $(datadir("fits")):
      $(join(filter(endswith(".h5"), readdir(datadir("fits"))), "\n      "))""")

s = load_lambda_spectrum(DATAFILE)              # drops x < 0.8 Å, as the fit did

fitdata = h5open(FITFILE, "r") do f
    (; a₀ = read(f["a0_au"]), a = read(f["a_au"]), A = read(f["A_au"]),
       λ = read(f["lambda_Ha"]), α = read(f["alpha_Ha"]), β = read(f["beta_Ha"]),
       ω = read(f["omega_Ha"]), x = read(f["x_Ang"]),
       E = read(f["errors/relative"]),
       anchor_err = read(f["errors/anchor"]),
       J_over_J0 = read_attribute(f, "J_over_J0"),
       m_aux = read_attribute(f, "m_auxiliary"),
       # Present only in a stage-2 file. Its absence is what makes a file stage 1.
       basis = haskey(f, "C_cheb") ?
               (; C = read(f["C_cheb"]), s = read(f["s_warp"]),
                  env = read(f["envelope"]), degree = read_attribute(f, "degree")) :
               nothing)
end

# The fit is only meaningful against the grid it was fitted on. A silent mismatch here —
# a different xmin, a regenerated spectrum — would draw a model against data it never saw.
fitdata.x ≈ s.x || error("fit was made on a different configuration grid than $DATAFILE")

ladder = PoleSet(fitdata.λ, fitdata.α, fitdata.β)
# Build the representation the fit actually is. Both give identical values *at the
# nodes* — every panel here samples nodes — but reconstructing the Chebyshev object
# means the figure is drawn from the coefficients that go on to build Γ̃₂,₁(q), not from
# a table of its values that could drift from them.
model = if fitdata.basis === nothing
    GridAmplitudes(fitdata.a₀, fitdata.a, fitdata.A)
else
    ChebyshevAmplitudes(fitdata.basis.C, fitdata.basis.s, fitdata.basis.env;
                        K_R = size(fitdata.a, 1), K_C = size(fitdata.A, 1))
end

# A stage-2 file must reproduce its own saved node values from its own coefficients.
if fitdata.basis !== nothing
    Θsaved = vcat(reshape(fitdata.a₀, 1, :), fitdata.a, fitdata.A)
    Θrebuilt = hcat([collect(vcat(amplitudes(model, i)...)) for i in axes(Θsaved, 2)]...)
    isapprox(Θsaved, Θrebuilt; rtol = 1e-8) ||
        error("the saved coefficients do not reproduce the saved amplitudes")
end

ωband_eV = hartree_to_eV.(fitdata.ω)            # the 561 retained frequencies ℬ
ω_eV     = hartree_to_eV.(s.ω)                  # the full 60000-point grid

# Model curves are drawn on their own dense log grid, not on ℬ: the model is a continuous
# function and drawing it through the fitted points only would hide any structure it puts
# between them.
const ω_model_eV = exp10.(range(log10(first(ω_eV)), log10(last(ω_eV)); length = 1200))
const ω_model_Ha = eV_to_hartree.(ω_model_eV)

# Node indices of the requested configurations.
node_of(xq) = argmin(abs.(s.x .- xq))
const NODES  = node_of.(X_SHOWN_ANG)
const I_RES  = node_of(s.x_resonance)

# One colour per configuration, in fixed slot order — Okabe–Ito, the colour-vision-safe
# qualitative set. Validated over *all ten pairs*, not just neighbouring slots, because
# these five curves cross each other and any two may be compared: worst normal-vision
# separation ΔE 15.6 (floor 15), worst CVD ΔE 7.6 (deutan, pink↔green).
#
# That 7.6 sits in the 6–8 band, which is only admissible with a second, non-colour
# channel — hence `XMARKERS`. Marker shape rather than line style, because every fit
# curve is drawn solid: a dashed line reads as a run of samples, and "solid line = model,
# markers = data" is the one distinction these panels cannot afford to blur. The shapes
# are not decoration — they are what makes the figure readable in deuteranopia and in
# greyscale print, so every panel that separates configurations carries them, panel (b)
# included. The slots are also ordered against the data: #E69F00 is the faintest on
# white, and it goes to x = 5.00 Å, the curve sitting twelve decades below the rest with
# nothing to confuse it with.
const XCOLORS  = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
const XMARKERS = [:circle, :rect, :utriangle, :diamond, :star5]

length(NODES) == length(XCOLORS) || error(
    "X_SHOWN_ANG has $(length(NODES)) entries but the palette has $(length(XCOLORS)) " *
    "slots; extending it means re-running the skill's palette validator, not cycling")

label_of(i) = i == I_RES ?
    @sprintf("x₀ = %.2f Å (crossing)", s.x[i]) :
    @sprintf("x = %.2f Å", s.x[i])

# Log-spaced marker positions into the full ω grid, so the markers read as sampled data
# rather than a second line.
const MARK_IDX = unique(round.(Int, exp10.(range(log10(2), log10(length(ω_eV));
                                                 length = N_MARKERS))))
const BAND_MARK_IDX = round.(Int, range(1, length(ωband_eV); length = 26))

model_curve(i) = [spectral_density(w, ladder, model, i) for w in ω_model_Ha]

# Explicit cell edges. Makie's default is to extend half a cell past the outermost
# coordinate, which on a log axis sends the bottom edge through zero — ω₁ = 10⁻⁶ eV is
# the anchor frequency, and half a cell below it is negative. Both grids are also
# non-uniform (x is graded, ω is log-spaced), so the edges have to be built anyway.
lin_edges(v) = vcat(v[1] - (v[2] - v[1]) / 2,
                    (v[1:(end - 1)] .+ v[2:end]) ./ 2,
                    v[end] + (v[end] - v[end - 1]) / 2)
geo_edges(v) = vcat(v[1] * sqrt(v[1] / v[2]),
                    sqrt.(v[1:(end - 1)] .* v[2:end]),
                    v[end] * sqrt(v[end] / v[end - 1]))

# ------------------------------------------------------------------------------------
# Figure 1 — the fit
# ------------------------------------------------------------------------------------

fig1 = Figure(size = (FIG_W, 700))

# (a) Λ(x, ω): data and model over the whole window ------------------------------------

ax_a = Axis(fig1[1, 1];
    xscale = log10, yscale = log10,
    xlabel = "ω [eV]", ylabel = "Λ(x, ω)  [a.u.]",
    title = "(a)  spectral density: data (markers), Prony fit (lines)",
    titlealign = :left, titlefont = :regular,
    xminorticksvisible = true, xminorticks = IntervalsBetween(9))

for (k, i) in enumerate(NODES)
    lines!(ax_a, ω_model_eV, model_curve(i);
           color = XCOLORS[k], linewidth = LINE_WIDTH, label = label_of(i))
    # A white ring keeps the data markers legible where they sit on their own fit line.
    # Same label as the line above, so `merge` folds the two into one legend entry that
    # keys the colour and the shape together.
    scatter!(ax_a, ω_eV[MARK_IDX], s.Λ[MARK_IDX, i];
             color = XCOLORS[k], marker = XMARKERS[k], markersize = MARKER_SIZE,
             strokewidth = 0.6, strokecolor = :white, label = label_of(i))
end
xlims!(ax_a, 1e-3, 80)

# (b) signed relative error, on the band the objective saw -----------------------------
#
# Signed, not absolute: the far tail is a systematic *under*shoot — Λ decays as ω⁻³ and no
# Lorentzian sum follows past ω⁻² — and taking the modulus would hide that it is one-sided
# and therefore structural rather than noise.

ax_b = Axis(fig1[1, 2];
    xscale = log10,
    xlabel = "ω [eV]", ylabel = L"100\,(\Lambda_\theta - \Lambda)\,/\,|\Lambda|",
    title = "(b)  signed relative error on the retained band ℬ  [%]",
    titlealign = :left, titlefont = :regular,
    xminorticksvisible = true, xminorticks = IntervalsBetween(9))

hlines!(ax_b, [0.0]; color = (:black, 0.35), linewidth = 1)
for (k, i) in enumerate(NODES)
    lines!(ax_b, ωband_eV, 100 .* fitdata.E[:, i];
           color = XCOLORS[k], linewidth = LINE_WIDTH)
    # ℬ is itself log-spaced, so evenly spaced indices are evenly spaced in log ω. These
    # mark identity, not samples — the curve is the quantity.
    scatter!(ax_b, ωband_eV[BAND_MARK_IDX], 100 .* fitdata.E[BAND_MARK_IDX, i];
             color = XCOLORS[k], marker = XMARKERS[k], markersize = MARKER_SIZE - 1,
             strokewidth = 0.6, strokecolor = :white)
end
xlims!(ax_b, 1e-3, 80)
ylims!(ax_b, -112, 112)
# The far tail is a one-sided undershoot, not scatter: Λ decays as ω⁻³ and no sum of
# Lorentzians follows past ω⁻². Signed rather than absolute error is what makes that
# visible, and it is why the ε floor of §3 exists.

# (c) the Markovian anchor Λ(x, 0) -----------------------------------------------------
#
# The one number a standard MDEF run would use, so the one the fit must reproduce
# everywhere and not merely on average. All 254 columns, not just the five drawn above.

anchor_row   = markovian_anchor_row(ladder)
anchor_model = [sum(anchor_row .* vcat(amplitudes(model, i)...)) for i in axes(s.Λ, 2)]

ax_c = Axis(fig1[2, 1];
    yscale = log10,
    xlabel = "x [Å]", ylabel = "Λ(x, 0)  [a.u.]",
    title = "(c)  Markovian anchor, all $(length(s.x)) configurations",
    titlealign = :left, titlefont = :regular)

vlines!(ax_c, [s.x_resonance]; color = (:black, 0.3), linestyle = :dash, linewidth = 1)
scatter!(ax_c, s.x, s.Λ[1, :];
         color = (:black, 0.45), markersize = 5, strokewidth = 0, label = "data")
lines!(ax_c, s.x, anchor_model;
       color = (:black, 0.8), linewidth = LINE_WIDTH, label = "fit")
for (k, i) in enumerate(NODES)
    scatter!(ax_c, [s.x[i]], [s.Λ[1, i]];
             color = XCOLORS[k], marker = XMARKERS[k], markersize = 12,
             strokewidth = 1.2, strokecolor = :white)
end
text!(ax_c, s.x_resonance, minimum(s.Λ[1, :]);
      text = " x₀", align = (:left, :bottom), fontsize = 12, color = (:black, 0.6))
axislegend(ax_c; position = :rt, framevisible = false, labelsize = 12)

# (d) where the error lives in (x, ω) --------------------------------------------------

ax_d = Axis(fig1[2, 2];
    yscale = log10,
    xlabel = "x [Å]", ylabel = "ω [eV]",
    title = "(d)  where the error lives in (x, ω)",
    titlealign = :left, titlefont = :regular)

hm = heatmap!(ax_d, lin_edges(s.x), geo_edges(ωband_eV),
              permutedims(log10.(abs.(fitdata.E) .+ 1e-8));
              colormap = ColorSchemes.batlow, colorrange = (-3, 1))
vlines!(ax_d, [s.x_resonance]; color = (:white, 0.6), linestyle = :dash, linewidth = 1)
ylims!(ax_d, 1e-3, 80)          # matches (a) and (b); ω → 0 has panel (c) to itself
Colorbar(fig1[2, 3], hm;
         label = L"\log_{10}\,|\Lambda_\theta - \Lambda|\,/\,|\Lambda|",
         ticks = -3:1:1, width = 12)

Legend(fig1[0, 1:3], ax_a; orientation = :horizontal, framevisible = false,
       labelsize = 12, nbanks = 1, merge = true)

Label(fig1[3, 1:3],
      @sprintf("Erpenbeck–Thoss, Γ = %.1f eV, T = %.0f K   |   split ladder: %d pairs in [%.2f, %.1f] eV + %d in [%.0f, %.0f] eV, %d real rates → m = %d   |   J/J₀ = %.3f",
               hartree_to_eV(s.Γ), s.T_K,
               K_C_LO, hartree_to_eV(BETA_LO[1]), hartree_to_eV(BETA_LO[2]),
               K_C_HI, hartree_to_eV(BETA_HI[1]), hartree_to_eV(BETA_HI[2]),
               K_R, fitdata.m_aux, fitdata.J_over_J0);
      fontsize = 12, color = :gray30, halign = :left)

colgap!(fig1.layout, 18)
rowgap!(fig1.layout, 10)

out1 = plotsdir("ErpenbeckThoss", "spectrum_and_fit_$(FIGTAG).pdf")
mkpath(dirname(out1))
save(out1, fig1)
save(replace(out1, ".pdf" => ".png"), fig1; px_per_unit = 2)
@info "saved" out1

# ------------------------------------------------------------------------------------
# Figure 2 — the series taken apart
# ------------------------------------------------------------------------------------

fig2 = Figure(size = (FIG_W, 400))

# (a) the individual Lorentzians at the crossing column --------------------------------
#
# x₀ is the interesting column: β → 0 there, so the resonance has collapsed onto ω = 0 and
# the shape is carried by the *real* rates. Drawing the modes shows that directly — the
# thin lines under the peak are zero-centred, not a pair at ±β.

_, a, A = amplitudes(model, I_RES)   # a₀ is the white part, a flat line, and NNLS drops it
n_active = count(>(0), a) + count(>(0), A)

ax2a = Axis(fig2[1, 1];
    xscale = log10, yscale = log10,
    xlabel = "ω [eV]", ylabel = "contribution to Λ  [a.u.]",
    # Stage 1 is an NNLS and *drops* modes — exact zeros are its normal output. Stage 2
    # parametrises a = g², which is zero only where g is, so it keeps every rung and the
    # count stops being the interesting number. Say which mechanism produced the figure.
    title = @sprintf("(a)  the %d modes %s at x₀ = %.2f Å, of %d offered",
                     n_active,
                     fitdata.basis === nothing ? "NNLS keeps" : "the degree-$DEGREE fit carries",
                     s.x[I_RES], K_R + K_C_LO + K_C_HI),
    titlealign = :left, titlefont = :regular,
    xminorticksvisible = true, xminorticks = IntervalsBetween(9))

for (k, λₖ) in enumerate(ladder.λ)
    a[k] > 0 || continue
    lines!(ax2a, ω_model_eV, a[k] .* lorentzian.(ω_model_Ha, λₖ);
           color = (XCOLORS[1], 0.75), linewidth = 0.9)
end
for (j, (αⱼ, βⱼ)) in enumerate(zip(ladder.α, ladder.β))
    A[j] > 0 || continue
    lines!(ax2a, ω_model_eV, A[j] .* mirrored_lorentzian.(ω_model_Ha, αⱼ, βⱼ);
           color = (XCOLORS[2], 0.75), linewidth = 0.9)
end
lines!(ax2a, ω_model_eV, model_curve(I_RES);
       color = :black, linewidth = LINE_WIDTH,
       label = L"\Lambda_\theta,~\mathrm{their~sum}")
scatter!(ax2a, ω_eV[MARK_IDX], s.Λ[MARK_IDX, I_RES];
         color = (:black, 0.5), markersize = MARKER_SIZE, strokewidth = 0, label = "data")
xlims!(ax2a, 1e-3, 80)
ylims!(ax2a, 1e-2, 1e4)
axislegend(ax2a; position = :lb, framevisible = false, labelsize = 11)

# Colour key for the two mode families, in text rather than one legend entry per mode.
# Top right: the only corner clear of both the decaying tail and the legend.
text!(ax2a, 0.97, 0.97; text = "zero-centred, real rates λₖ", space = :relative,
      align = (:right, :top), fontsize = 11, color = XCOLORS[1])
text!(ax2a, 0.97, 0.90; text = "mirrored pairs (αⱼ, βⱼ)", space = :relative,
      align = (:right, :top), fontsize = 11, color = XCOLORS[2])

# (b) where the weight sits on the ladder, across every configuration ------------------
#
# Regime (A): the poles never move, so the whole x dependence is in the amplitudes. The
# resonance does move — the data peaks at ω_pk ≈ |h(x)|, overlaid — and this panel is how
# a fixed ladder follows it: weight walks down the rungs into the crossing and back up
# again. Normalised per column because the magnitudes span 15 decades in x (panel (c) of
# the other figure carries that); the question here is *which* rung, not how much.
#
# The blank horizontal band near 10 eV is the gap in the split ladder, not missing data.
# Nothing is above |h| = 8.65 eV to resolve; the four rungs above it only set the ω⁻²
# coefficient of the far tail, and the panel shows them earning that keep on x ≥ 3 Å.

β_eV   = hartree_to_eV.(ladder.β)
A_norm = fitdata.A ./ maximum(fitdata.A; dims = 1)  # [K_C_LO+K_C_HI, N_x], col max = 1

ax2b = Axis(fig2[1, 2];
    yscale = log10,
    xlabel = "x [Å]", ylabel = "rung frequency βⱼ [eV]",
    title = "(b)  pair amplitudes Aⱼ(x), normalised per configuration",
    titlealign = :left, titlefont = :regular,
    yticks = ([0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 60.0],
              ["0.05", "0.1", "0.5", "1", "5", "10", "60"]))

hm2 = heatmap!(ax2b, lin_edges(s.x), geo_edges(β_eV), permutedims(log10.(A_norm .+ 1e-12));
               colormap = ColorSchemes.batlow, colorrange = (-6, 0))
lines!(ax2b, s.x, abs.(hartree_to_eV.(s.h));
       color = (:white, 0.85), linewidth = 1.6, label = "|h(x)|, where the data peaks")
vlines!(ax2b, [s.x_resonance]; color = (:white, 0.55), linestyle = :dash, linewidth = 1)
ylims!(ax2b, extrema(geo_edges(β_eV))...)
# Bottom-left, the one corner of this panel that is reliably dark, so white reads.
axislegend(ax2b; position = :lb, framevisible = false, labelsize = 11,
           labelcolor = :white)
Colorbar(fig2[1, 3], hm2; label = L"\log_{10}\,A_j/\max_j A_j", ticks = -6:2:0, width = 12)

colgap!(fig2.layout, 18)
rowgap!(fig2.layout, 6)

out2 = plotsdir("ErpenbeckThoss", "prony_modes_$(FIGTAG).pdf")
save(out2, fig2)
save(replace(out2, ".pdf" => ".png"), fig2; px_per_unit = 2)
@info "saved" out2

median_(v) = (w = sort(vec(collect(v))); n = length(w);
               isodd(n) ? w[(n + 1) ÷ 2] : (w[n ÷ 2] + w[n ÷ 2 + 1]) / 2)

@printf("\nΓ = %.1f eV | m = %d | J/J₀ = %.4f | anchor: median %.4f%%, max %.4f%%\n",
        hartree_to_eV(s.Γ), fitdata.m_aux, fitdata.J_over_J0,
        100 * median_(fitdata.anchor_err), 100 * maximum(fitdata.anchor_err))
