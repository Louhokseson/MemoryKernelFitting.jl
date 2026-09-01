# The two basis functions of the fit model, drawn on their own.
#
#   julia scripts/plot/ErpenbeckThoss/plot_lorentzian_basis.jl
#
# `lorentzian(ω, λ)` and `mirrored_lorentzian(ω, α, β)` are the whole frequency-domain
# vocabulary of `Prony_fit_optimisation_problem.md` §1 — every fitted Λ_θ is a
# non-negative combination of these two shapes and a constant. This script reads no data
# and no fit: it draws the functions themselves, so the figure is a statement about the
# model class rather than about one dataset.
#
# It exists because three facts about these shapes are load-bearing for the whole scheme
# and are currently only written down in prose:
#
#   1. The two families are one family. `mirrored_lorentzian(ω, α, 0) == lorentzian(ω, α)`
#      exactly, so the "overdamped" basis is the β → 0 edge of the "underdamped" one, not
#      a separate ansatz. Panels (a) and (b) draw that edge as the β/α = 0 curve.
#   2. A pair is only a *resonance* above β = α/√3 (`basis.jl`, README). Below it the
#      mirrored pair is monotone and zero-centred — which is exactly what the data does at
#      the Fermi crossing, where β ≈ |h(x)| → 0. Panel (c) is that threshold, as a curve.
#   3. The mirror term cannot be dropped (`docs/design/0003`). It is a ~1 % effect at the
#      peak and *exactly 50 %* at ω = 0 — and Λ(x, 0) is the Markovian anchor. Panel (d)
#      is the record's table, drawn.
#
# Everything is in units of α, so the figure is dimensionless and applies to every rung of
# every ladder. The one place the Erpenbeck–Thoss ladder enters is the β/α = 5 curve:
# `log_ladder` ties α_j = α_ratio·β_j at α_ratio = 0.2, so every production rung sits at
# β/α = 5 and panel (d) is drawn there. Hence this lives under `ErpenbeckThoss/` despite
# reading none of its data.
#
# Encoding. Colour carries β/α, which is an *order*, so it gets a sequential ramp — the
# opposite choice from `plot_prony_fit.jl`, which spends five qualitative hues on the
# configuration x because x is an identity. Lightness ordering survives CVD and greyscale
# on its own, so no marker-shape channel is needed here. Panel (d) separates three *terms*
# of one formula, which is an identity again, so it goes back to the Okabe–Ito hues.

using DrWatson
@quickactivate "MemoryKernelFitting"

using MemoryKernelFitting
using CairoMakie
using ColorSchemes
using Printf

CairoMakie.activate!(type = "pdf")

# ------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------

# All frequencies are in units of α, so α ≡ 1 and β is read as β/α. Nothing below depends
# on this value; it is here so the calls read as the package's own signatures rather than
# as a rescaled special case.
const ALPHA = 1.0

# The five curves of panels (a) and (b). Chosen as the four qualitatively distinct regimes
# plus the one the production ladder actually uses:
#   0        — the degenerate pair, identically the overdamped Lorentzian
#   1/√3     — the resonance onset, where ω = 0 stops being the maximum
#   1.5, 3   — a resonance forming, then well separated from the mirror
#   5        — `log_ladder(...; α_ratio = 0.2)`, i.e. every rung of the fitted ladder
const BETA_OVER_ALPHA = [0.0, 1 / sqrt(3), 1.5, 3.0, 5.0]
const ONSET           = 1 / sqrt(3)          # β/α below which the pair is monotone

# Panel (d) is drawn at the production ratio, where the mirror is at its least visible —
# the point being that it still carries half of Λ(0).
const BETA_D = 5.0

const FIG_W      = 900
const LINE_WIDTH = 2.0

# `α·L` and `α·lorentzian` are dimensionless, so the y axes of (b) and (d) are pure
# numbers; with ALPHA = 1 the multiplication is a no-op and is left implicit.
const OMEGA_MAX = 8.0        # ω/α for panels (a), (b)
const TAU_MAX   = 8.0        # α·τ for panel (a)

# ------------------------------------------------------------------------------------
# The shapes, through the package's own functions
# ------------------------------------------------------------------------------------

# Time domain via `memory_kernel` rather than a retyped `exp(-ατ)cos(βτ)`: this figure
# claims the two panels are a transform pair, and it can only claim that if it is drawing
# the same expressions the fit and the readout use. A single underdamped mode is the
# whole-series call with no real modes and unit amplitude.
kernel_mode(τ, α, β) =
    memory_kernel(τ, Float64[], Float64[], [1.0], [float(α)], [float(β)])

# The two halves of `mirrored_lorentzian`, split for panel (d). `main` is the peak at +β
# that survives if the mirror is (wrongly) dropped; `mirror` is the image at −β. Their sum
# is `mirrored_lorentzian`, which the verification block below asserts.
main_half(ω, α, β)   = (α / 2) / (α^2 + (ω - β)^2)
mirror_half(ω, α, β) = (α / 2) / (α^2 + (ω + β)^2)

# Closed-form peak location of L(·; α, β):
#
#   L'(ω) = 0  ⟺  ω = 0  or  ω² = 2β√(α²+β²) − (α²+β²),
#
# and the second root is real exactly when 4β² > α²+β², i.e. β > α/√3 — the resonance
# criterion `basis.jl` states. Returns 0 below the onset, where ω = 0 is the maximum.
# Derived here rather than found by a grid search so the threshold is exact; the search is
# kept below as the check on this expression.
function peak_frequency(α, β)
    c = α^2 + β^2
    v = 2 * β * sqrt(c) - c
    return v > 0 ? sqrt(v) : zero(float(v))
end

# ------------------------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------------------------

# Sequential ramp, trimmed at both ends: batlow's extremes are a near-black and a pale
# sand, neither of which reads as a line on white.
const RAMP = [get(ColorSchemes.batlow, t) for t in range(0.08, 0.82; length = length(BETA_OVER_ALPHA))]

# Okabe–Ito slots for panel (d)'s three terms — the same qualitative set
# `plot_prony_fit.jl` validated, used here for term identity rather than configuration.
const C_FULL   = "#000000"
const C_MAIN   = "#0072B2"
const C_MIRROR = "#D55E00"
const C_SHARE  = "#009E73"

# Plain Unicode rather than `L"..."`: `latexstring` is not exported by Makie (only the
# `L_str` macro is), and these labels carry prose, which needs `\text{}` — the one
# construct `plot_prony_fit.jl` never asks MathTeXEngine for.
ratio_label(r) =
    r == 0      ? "β/α = 0  (real pole)"   :
    r ≈ ONSET   ? "β/α = 1/√3  (onset)"    :
    r == BETA_D ? "β/α = 5  (ladder)"      :
                  @sprintf("β/α = %.1f", r)

fig = Figure(size = (FIG_W, 640))

# (a) time domain — the Prony modes ----------------------------------------------------
#
# The overdamped mode decays monotonically; the underdamped one rings, and the ringing is
# what a zero-centred Lorentzian sum cannot produce in ω. β/α = 0 lies exactly on the
# overdamped curve, which is the point: it is the same function.

ax_a = Axis(fig[1, 1];
    xlabel = L"\alpha\tau", ylabel = L"K(\tau)/A = e^{-\alpha\tau}\cos(\beta\tau)",
    title = "(a)  time domain: one Prony mode",
    titlealign = :left, titlefont = :regular)

hlines!(ax_a, [0.0]; color = (:black, 0.3), linewidth = 1)
for (k, r) in enumerate(BETA_OVER_ALPHA)
    τs = range(0, TAU_MAX; length = 800)
    lines!(ax_a, τs, kernel_mode.(τs, ALPHA, r * ALPHA);
           color = RAMP[k], linewidth = LINE_WIDTH, label = ratio_label(r))
end
# The envelope is the β/α = 0 curve, so it needs no separate line — say so instead of
# drawing a redundant dashed exp(-ατ) on top of it.
text!(ax_a, 0.97, 0.97;
      text = "β/α = 0 is the envelope\nof every other curve",
      space = :relative, align = (:right, :top), fontsize = 10.5, color = :gray35)
xlims!(ax_a, 0, TAU_MAX)
ylims!(ax_a, -0.62, 1.06)

# (b) frequency domain — the same modes, transformed -----------------------------------

ax_b = Axis(fig[1, 2];
    xlabel = L"\omega/\alpha", ylabel = L"\alpha\,L(\omega;\alpha,\beta)",
    title = "(b)  frequency domain: the one-sided cosine transform",
    titlealign = :left, titlefont = :regular)

for (k, r) in enumerate(BETA_OVER_ALPHA)
    ωs = range(0, OMEGA_MAX; length = 1200)
    lines!(ax_b, ωs, mirrored_lorentzian.(ωs, ALPHA, r * ALPHA);
           color = RAMP[k], linewidth = LINE_WIDTH)
    ω_pk = peak_frequency(ALPHA, r * ALPHA)
    ω_pk > 0 && scatter!(ax_b, [ω_pk], [mirrored_lorentzian(ω_pk, ALPHA, r * ALPHA)];
                         color = RAMP[k], markersize = 8,
                         strokewidth = 0.8, strokecolor = :white)
end
# `lorentzian(ω, α)` drawn over the β/α = 0 curve, dashed and in black: the two coincide
# to machine precision (asserted below), and overplotting is the only honest way to show
# a claim of exact equality.
let ωs = range(0, OMEGA_MAX; length = 1200)
    lines!(ax_b, ωs, lorentzian.(ωs, ALPHA);
           color = (:black, 0.8), linewidth = 1.2, linestyle = :dash,
           label = "lorentzian(ω, α)")
end
xlims!(ax_b, 0, OMEGA_MAX)
ylims!(ax_b, 0, 1.08)
axislegend(ax_b; position = :rt, framevisible = false, labelsize = 11, padding = (4, 4, 2, 2))

# (c) the resonance onset --------------------------------------------------------------
#
# Where the maximum of L sits, against β/α. It is pinned at ω = 0 until β = α/√3 and
# leaves the origin with infinite slope — so a pair whose β drifts below the onset does
# not "become a narrow peak near zero", it stops having a peak at all. That is the
# behaviour `docs/design/0003` relies on at the Fermi crossing, and also the reason the
# ladder's own α_j = 0.2 β_j tie fails there: it takes the *width* down with β, giving a
# narrow spike rather than the broad zero-centred shape the data has.

ax_c = Axis(fig[2, 1];
    xlabel = L"\beta/\alpha", ylabel = L"\omega_{pk}/\alpha",
    title = "(c)  the pair is a resonance only above β = α/√3",
    titlealign = :left, titlefont = :regular)

rs = range(0, 4; length = 1000)
band!(ax_c, range(0, ONSET; length = 2), [0.0, 0.0], [4.0, 4.0];
      color = (:gray, 0.13))
lines!(ax_c, rs, rs; color = (:black, 0.35), linewidth = 1, linestyle = :dot,
       label = "ωₚₖ = β  (mirror dropped)")
lines!(ax_c, rs, peak_frequency.(ALPHA, rs .* ALPHA);
       color = RAMP[2], linewidth = 2.4, label = "ωₚₖ of L")
vlines!(ax_c, [ONSET]; color = (:black, 0.45), linestyle = :dash, linewidth = 1)
text!(ax_c, ONSET / 2, 1.6; text = "monotone, zero-centred",
      rotation = π / 2, align = (:center, :center), fontsize = 10.5, color = :gray35)
text!(ax_c, ONSET, -0.28; text = L"\alpha/\sqrt{3}", align = (:center, :bottom),
      fontsize = 11, color = :gray25)
xlims!(ax_c, 0, 4)
ylims!(ax_c, -0.3, 4)
axislegend(ax_c; position = :lt, framevisible = false, labelsize = 11)

# (d) why the mirror term cannot be dropped --------------------------------------------
#
# `docs/design/0003`, as a picture. The two halves of L are equal at ω = 0 and separate as
# ω grows; at the peak the mirror is α²/4β² of the total (1 % at the production ratio),
# which is what makes "it is negligible" tempting. But Λ(x, 0) is the Markovian anchor —
# the one number this package must reproduce for standard MDEF — and there the mirror is
# exactly half of it.

ax_d = Axis(fig[2, 2];
    yscale = log10,
    xlabel = L"\omega/\alpha", ylabel = "α × contribution",
    title = @sprintf("(d)  the mirror at the production ratio β/α = %.0f", BETA_D),
    titlealign = :left, titlefont = :regular)

# Right-hand axis for the share. Transparent background, no x decorations: it exists only
# to carry a second scale over the same x.
ax_d2 = Axis(fig[2, 2];
    yaxisposition = :right, ylabel = "mirror share of L  [%]",
    ylabelcolor = C_SHARE, yticklabelcolor = C_SHARE,
    backgroundcolor = :transparent)
linkxaxes!(ax_d, ax_d2)
hidespines!(ax_d2)
hidexdecorations!(ax_d2)

let ωs = range(0, OMEGA_MAX; length = 1600)
    full = mirrored_lorentzian.(ωs, ALPHA, BETA_D)
    mn   = main_half.(ωs, ALPHA, BETA_D)
    mr   = mirror_half.(ωs, ALPHA, BETA_D)

    lines!(ax_d, ωs, full; color = C_FULL, linewidth = LINE_WIDTH,
           label = "L  =  their sum")
    lines!(ax_d, ωs, mn; color = C_MAIN, linewidth = 1.5,
           label = "peak at +β")
    lines!(ax_d, ωs, mr; color = C_MIRROR, linewidth = 1.5,
           label = "mirror at −β")
    lines!(ax_d2, ωs, 100 .* mr ./ full; color = C_SHARE, linewidth = 1.6,
           linestyle = :dashdot)
    # Keyed by the right axis' own colour rather than a fourth legend entry, since it is
    # the only series on that axis and a legend entry would not say which scale it reads.
    text!(ax_d, 0.03, 0.62; text = "mirror share →", space = :relative,
          align = (:left, :center), fontsize = 10, color = C_SHARE)
end

# The two numbers the design record quotes, marked where they are read off.
scatter!(ax_d, [0.0], [mirrored_lorentzian(0.0, ALPHA, BETA_D)];
         color = C_FULL, markersize = 8, strokewidth = 0.8, strokecolor = :white)
text!(ax_d, 0.97, 0.97;
      text = @sprintf("ω = 0:  the halves are equal — 50 %%\nω = β:  the mirror is α²/4β² = %.0f %%",
                      100 * ALPHA^2 / (4 * BETA_D^2)),
      space = :relative, align = (:right, :top), fontsize = 10, color = :gray25)

xlims!(ax_d, 0, OMEGA_MAX)
ylims!(ax_d, 1e-3, 2.0)
ylims!(ax_d2, 0, 62)
axislegend(ax_d; position = :lb, framevisible = false, labelsize = 11)

Legend(fig[0, 1:2], ax_a; orientation = :horizontal, framevisible = false,
       labelsize = 12, nbanks = 1)

Label(fig[3, 1:2],
      "Λ_θ = a₀ + Σₖ aₖ·lorentzian(ω, λₖ) + Σⱼ Aⱼ·mirrored_lorentzian(ω, αⱼ, βⱼ)   |   " *
      "units of α   |   the fitted ladder ties αⱼ = 0.2 βⱼ, so every rung is at β/α = 5";
      fontsize = 11, color = :gray30, halign = :left)

colgap!(fig.layout, 22)
rowgap!(fig.layout, 10)

out = plotsdir("ErpenbeckThoss", "lorentzian_basis.pdf")
mkpath(dirname(out))
save(out, fig)
save(replace(out, ".pdf" => ".png"), fig; px_per_unit = 2)
@info "saved" out

# ------------------------------------------------------------------------------------
# What the figure claims, checked
# ------------------------------------------------------------------------------------
#
# The panels above assert four things. A figure that asserts is worth less than one that
# has been made to prove itself, and all four are cheap, so they are checked here and the
# numbers printed under the figure's own settings. `test/test_prony_model.jl` owns the
# same transform check as a test; this is the figure certifying its own axes.

# Trapezoid quadrature of ∫₀^∞ f(τ)cos(ωτ)dτ, as in `test/test_prony_model.jl`.
function cosine_transform(f, ω; Δτ = 1e-4, τmax = 120.0)
    τs = 0.0:Δτ:τmax
    g(τ) = f(τ) * cos(ω * τ)
    return Δτ * (sum(g, τs) - (g(first(τs)) + g(last(τs))) / 2)
end

relerr(got, want) = abs(got - want) / max(abs(want), eps())

println("\n", "─"^78)
println("checks on the figure's own settings (α = $(ALPHA))")
println("─"^78)

# 1 — (a) and (b) are a transform pair.
worst_pair = maximum(
    relerr(cosine_transform(τ -> kernel_mode(τ, ALPHA, r * ALPHA), ω),
           mirrored_lorentzian(ω, ALPHA, r * ALPHA))
    for r in BETA_OVER_ALPHA, ω in (0.0, 0.5, 1.0, 2.0, 5.0, 8.0))
@printf("1. panels (a)↔(b) are a cosine-transform pair    max rel. err %.2e  (quadrature)\n",
        worst_pair)

# 2 — the β → 0 edge really is the overdamped basis function, exactly.
worst_edge = maximum(relerr(mirrored_lorentzian(ω, ALPHA, 0.0), lorentzian(ω, ALPHA))
                     for ω in range(0, OMEGA_MAX; length = 501))
@printf("2. mirrored_lorentzian(ω, α, 0) == lorentzian    max rel. err %.2e\n", worst_edge)

# 3 — the closed-form peak of panel (c), against a grid search of the same function.
function peak_by_search(α, β; n = 2_000_001)
    ωs = range(0, 2 * β + 6 * α; length = n)
    return ωs[argmax(mirrored_lorentzian.(ωs, α, β))]
end
worst_pk = maximum(
    abs(peak_by_search(ALPHA, r * ALPHA) - peak_frequency(ALPHA, r * ALPHA))
    for r in (0.2, 0.5, ONSET, 0.7, 1.0, 2.0, 3.0, 5.0))
@printf("3. closed-form ω_pk vs grid argmax               max abs. err %.2e α\n", worst_pk)

# 4 — the onset is where the grid search first leaves the origin. Bisected on the grid
#     search, not on the formula, so it is an independent measurement of the threshold.
function measured_onset(lo, hi; iters = 40)
    for _ in 1:iters
        mid = (lo + hi) / 2
        peak_by_search(ALPHA, mid * ALPHA; n = 200_001) > 1e-9 ? (hi = mid) : (lo = mid)
    end
    return (lo + hi) / 2
end
onset_measured = measured_onset(0.3, 0.9)
@printf("4. measured onset β/α = %.6f, 1/√3 = %.6f      Δ = %.1e\n",
        onset_measured, ONSET, abs(onset_measured - ONSET))

# 5 — panel (d)'s two annotated numbers.
share(ω) = mirror_half(ω, ALPHA, BETA_D) / mirrored_lorentzian(ω, ALPHA, BETA_D)
@printf("5. mirror share at ω = 0: %.4f %%   at ω = β: %.4f %%  (α²/4β² = %.4f %%)\n",
        100 * share(0.0), 100 * share(BETA_D), 100 * ALPHA^2 / (4 * BETA_D^2))
println("─"^78, "\n")
