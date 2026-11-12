# BCME against the three friction pictures: the energy a scattering molecule hands to the
# electrons, from Nils' broadened classical master equation and from our GLE runs.
#
#   julia --project=. scripts/plot/gle/plot_bcme_comparison.jl            # sweep + every energy
#   julia --project=. scripts/plot/gle/plot_bcme_comparison.jl 2.0 3.5    # sweep + just these
#
# Reads two kinds of artefact and runs no dynamics itself:
#
#   data/sims/gle/scatter_*.h5          `scripts/gle/run_gle.jl` — GLE, Markovian MDEF, Classical
#   data/sims/bcme/bcme_scatter_*.h5    `scripts/bcme/reduce_bcme.jl` — Nils' BCME runs, reduced
#
# The raw BCME files (data/BCME_calculations, 193 MB each, no metadata) are never opened
# here. `reduce_bcme.jl` reads them once, checks what can be checked about them — launch
# point and energy, the diabat the run starts on, Γ and the bandwidth in eV — and writes
# what a figure needs in the layout of our own artefacts; its header holds the evidence.
#
# **The two calculations do not run on the same surface.** Ours is the lower adiabat of the
# bare 2×2, whose avoided crossing is split by 2V₁₂ = 1.3 eV and whose barrier at the Fermi
# crossing is 1.18 eV above the launch level. BCME moves on the *broadened diabats*
# — U₂ and U₁ = U₂ − h with the wide-band corrections of width Γ = 0.1 eV — which cross at
# the same 2.02 Å but 1.67 eV above the launch level. Panel (a) draws both, from the
# `landscape/` group the reducer computes. The result is in the data before any friction
# is discussed: at 1.5 eV our molecule crosses and loses 300–380 meV, the BCME molecule
# turns around at 2.06 Å and loses nothing.
#
# **What "energy to the electrons" means for BCME.** The molecule's state is not stored, so
# its potential energy at t_end is not known — but every scattered molecule is at x ≥ 20
# bohr where both diabats are flat to 10⁻⁵ eV, so its *translational* energy is. The
# reducer writes `E_total − V∞ − ½mv_end²` with V₂₂ as the reference, 0.02 meV from the
# reference our artefacts use, so all four methods share one classical zero. It is defined
# for *scattered* trajectories; a trapped one (11–19 % above 2 eV, none in any friction
# run) is bound in the neutral Morse well 3.4 eV below the V₁₁ asymptote and cannot be
# assigned an asymptotic energy. Panel (c) therefore draws the cumulative distribution over
# *all* trajectories with the trapped ones at +∞, so the BCME curve plateaus at
# 1 − P_trap and the gap to 1 *is* the trapping probability.
#
# The BCME distributions are bimodal — more than half the scattered molecules lose nothing
# at all, the rest lose most of what they brought — so the sweep's mean ± SE is a poor
# summary of them on its own. Panel (b) of the sweep breaks the ensemble into no-hop,
# hop-and-leave, and trapped, which is what the mean is an average over.
#
# Encoding. Four methods, so colour carries identity: the three Okabe-Ito hues of
# `plot_energy_loss.jl` are kept unchanged and BCME takes the fourth, reddish purple. The
# four-colour palette passes the CVD checks (worst adjacent ΔE 15.8); purple sits at 2.98:1
# on white, which is why every panel direct-labels its curves and the legend carries the
# numbers. Each method also has its own dash pattern. One axis per panel.

using DrWatson
@quickactivate "MemoryKernelFitting"

using MemoryKernelFitting
using CairoMakie
using HDF5
using Printf
using Statistics

CairoMakie.activate!(type = "pdf")

# ------------------------------------------------------------------------------------
# Input
#
# Both artefact names are recomputed with the `savename` calls their producers use
# (`sort = false`: field order is part of the name) rather than spelled out.
# ------------------------------------------------------------------------------------

const GAMMA_EV = 0.1
const T_K = 300
const N_TRAJ = 400

scatter_params(Einc_eV) = (; Gamma_eV = GAMMA_EV, T_K, Einc_eV, N = N_TRAJ,
                             tfinal = 4000, dt = 0.5, m = 43)
scatter_file(p) = datadir("sims", "gle",
    savename("scatter_ErpenbeckThoss_ground", p, "h5"; sort = false))

bcme_params(Einc_eV) = (; Gamma_eV = GAMMA_EV, Einc_eV, N = N_TRAJ,
                          tmax_fs = 1000.0, dt_fs = 0.05, bandwidth_eV = 50)
bcme_file(p) = datadir("sims", "bcme",
    savename("bcme_scatter_ErpenbeckThoss", p, "h5"; sort = false))

# The full sweep; the command line narrows the per-energy figures, never the sweep.
const SWEEP_EV = collect(0.5:0.5:5.0)
const EINC_EV = isempty(ARGS) ? SWEEP_EV : parse.(Float64, ARGS)

# Fixed order, fixed hues. BCME first: it is the reference the other three are held to.
const SERIES = [
    (key = :bcme,      label = "BCME",             colour = "#CC79A7", dash = :dashdot),
    (key = :gle,       label = "MDEF (memory)",    colour = "#D55E00", dash = :solid),
    (key = :mdef,      label = "MDEF (Markovian)", colour = "#0072B2", dash = :dash),
    (key = :classical, label = "Classical",        colour = "#009E73", dash = :dot),
]
const GLE_GROUPS = Dict(:gle => "GLE_Prony_m_43_", :mdef => "MDEF_Markovian_x_0_",
                        :classical => "Classical_no_friction_")

# BCME outcomes, ordered no-hop → hop → trapped: one hue, three lightness steps. The
# light end is the darkest step that still reads as light and clears 2:1 on white.
const OUTCOME_RAMP = ["#D9A0C0", "#B9629A", "#7A3E66"]
const NO_LOSS_EV = 0.1               # a hop costs ≥ h at the hop; hopless reads 0 ± 1e-4

const HARTREE_TO_EV = hartree_to_eV(1.0)
const INK = (:black, 0.75)
const INK_MUTED = (:black, 0.45)

# ------------------------------------------------------------------------------------
# Loading
#
# Every series is reduced to the same NamedTuple whatever it came from:
#   t, x        the mean trajectory ⟨x(t)⟩ [aut, Å] over scattered trajectories
#   loss        energy to the electrons per scattered trajectory [eV]
#   n_total     ensemble size, n_trapped how many never left
#   x_turn      mean innermost position of the scattered trajectories [Å], with its SE
# ------------------------------------------------------------------------------------

function load_gle(file)
    h5open(file, "r") do f
        meta = (E_inc = read_attribute(f, "E_incident_eV"),
                N = read_attribute(f, "n_trajectories"),
                T = read_attribute(f, "T_K"),
                x0 = read_attribute(f, "x_start_Ang"),
                Γ = read_attribute(f, "Gamma_eV"),
                t_final = read_attribute(f, "t_final_aut"),
                E_total = read_attribute(f, "E_total_Ha"),
                V_asym = read_attribute(f, "V_asymptote_Ha"))
        series = Dict(k => begin
                g = f[name]
                loss = read(g["per_traj_loss"]) .* HARTREE_TO_EV
                (t = read(g["t_aut"]),
                 x = read(g["x_mean_bohr"]) .* BOHR_IN_ANGSTROM,
                 loss,
                 n_total = length(loss),
                 n_trapped = round(Int, read(g["stuck"]) * length(loss)),
                 x_turn = read(g["x_turn"]),
                 x_turn_se = read(g["x_turn_se"]))
            end for (k, name) in GLE_GROUPS)
        # The adiabat the three friction runs move on, relative to the launch level
        # V(x₀) = E_total − E_inc: the grid stops at 3 Å and does not reach x₀ itself.
        landscape = (x = read(f["landscape/x_Ang"]),
                     V = read(f["landscape/V_Ha"]) .* HARTREE_TO_EV .-
                         (meta.E_total * HARTREE_TO_EV - meta.E_inc))
        (series, landscape, meta)
    end
end

# The reduced BCME artefact: the series in the shape above, plus the surfaces it was
# integrated on, in eV relative to the launch level U₂(x₀) — the first grid point, since
# the reducer's grid starts at x₀.
function load_bcme(file, meta)
    h5open(file, "r") do f
        g = f["BCME"]
        for (k, v) in (("Gamma_eV", meta.Γ), ("E_incident_eV", meta.E_inc),
                       ("n_trajectories", meta.N))
            read_attribute(f, k) == v || error("$(basename(file)): $k = $(read_attribute(f, k)), GLE artefact has $v")
        end
        t = read(g["t_aut"])
        w = t .<= meta.t_final                        # the GLE window; BCME ran ten times longer
        loss = read(g["per_traj_loss"]) .* HARTREE_TO_EV
        sc = read(g["outcome"]) .== 1
        series = (t = t[w],
                  x = read(g["x_mean_bohr"])[w] .* BOHR_IN_ANGSTROM,
                  loss = loss[sc],
                  n_total = length(loss),
                  n_trapped = count(!, sc),
                  x_turn = read(g["x_turn"]),
                  x_turn_se = read(g["x_turn_se"]))
        l = f["landscape"]
        U2 = read(l["U2_Ha"])
        surfaces = (x = read(l["x_Ang"]),
                    U2 = (U2 .- U2[1]) .* HARTREE_TO_EV,
                    U1 = (read(l["U1_Ha"]) .- U2[1]) .* HARTREE_TO_EV,
                    V11 = (read(l["V11_Ha"]) .- U2[1]) .* HARTREE_TO_EV,
                    V22 = (read(l["V22_Ha"]) .- U2[1]) .* HARTREE_TO_EV,
                    x_cross = read_attribute(l, "x_cross_Ang"))
        (series, surfaces)
    end
end

# ------------------------------------------------------------------------------------
# Figure helpers
# ------------------------------------------------------------------------------------

const GRID_KW = (; xgridcolor = (:black, 0.06), ygridcolor = (:black, 0.06),
                 xgridwidth = 0.8, ygridwidth = 0.8,
                 topspinevisible = false, rightspinevisible = false,
                 spinewidth = 0.8)

# Push label heights apart bottom-up so no two are closer than one label height.
function spread!(ys, gap)
    ord = sortperm(ys)
    for k in 2:length(ord)
        ys[ord[k]] = max(ys[ord[k]], ys[ord[k-1]] + gap)
    end
    return ys
end

legend_elements() = [LineElement(color = s.colour, linestyle = s.dash, linewidth = 2)
                     for s in SERIES]

# One trajectory is the deterministic classical run, which run_gle.jl integrates once
# (older artefacts hold N identical copies). Its spread is exactly zero, not the NaN
# `std` gives for a single sample.
spread(v) = length(v) == 1 ? 0.0 : std(v)
mean_se(v) = (mean(v), spread(v) / sqrt(length(v)))

# "95 ± 10 meV" — the precision follows the size of the error. A hopless BCME ensemble
# reads −0.02 meV, which "%.1f" would print as "-0.0".
function pm_meV(v)
    μ, se = 1e3 .* mean_se(v)
    abs(μ) < 0.05 && (μ = 0.0)
    se < 0.5 ? @sprintf("%.1f ± %.1f meV", μ, se) : @sprintf("%.0f ± %.0f meV", μ, se)
end

# ------------------------------------------------------------------------------------
# Per-energy figure
# ------------------------------------------------------------------------------------

function comparison_figure(data, landscape, surfaces, meta)
    fig = Figure(size = (940, 640), fontsize = 12)

    # --- (a) the two surfaces ---------------------------------------------------------
    ax_V = Axis(fig[1:3, 1]; xlabel = "x  [Å]", ylabel = "V − V(x₀)  [eV]",
                title = "(a)  the surfaces the two calculations move on",
                subtitle = "thin grey: the bare diabats V₁₁ and V₂₂", subtitlesize = 10,
                subtitlecolor = INK_MUTED, titlealign = :left, titlefont = :bold, GRID_KW...)
    x_lo, x_hi = 1.3, 3.0
    xc = surfaces.x_cross
    vlines!(ax_V, [xc]; color = (:black, 0.35), linewidth = 1, linestyle = :dot)
    hlines!(ax_V, [0.0]; color = (:black, 0.35), linewidth = 1, linestyle = :dot)
    # The energy the molecule brings, above the launch level: what it can climb.
    hlines!(ax_V, [meta.E_inc]; color = (:black, 0.35), linewidth = 1, linestyle = :dot)
    text!(ax_V, x_hi - 0.03, meta.E_inc; text = @sprintf("E_inc = %.1f eV", meta.E_inc),
          align = (:right, :bottom), fontsize = 10, color = INK, offset = (0, 2))
    # Left of where the adiabat comes down through zero, so the curve does not strike
    # the text; every surface is well above the launch level at 2.1–2.5 Å.
    text!(ax_V, 2.12, 0.0; text = "launch level, x₀ = 5 Å",
          align = (:left, :bottom), fontsize = 10, color = INK, offset = (0, 2))

    lines!(ax_V, surfaces.x, surfaces.V11; color = (:black, 0.25), linewidth = 1)
    lines!(ax_V, surfaces.x, surfaces.V22; color = (:black, 0.25), linewidth = 1)
    bc = SERIES[1].colour
    lines!(ax_V, surfaces.x, surfaces.U2; color = bc, linewidth = 2)
    lines!(ax_V, surfaces.x, surfaces.U1; color = bc, linewidth = 2, linestyle = :dash)
    lines!(ax_V, landscape.x, landscape.V; color = :black, linewidth = 2)

    # Barrier heights: the adiabat's maximum beyond the neutral well, and U₂ at the
    # crossing where the BCME molecule must hop or turn back.
    inner = landscape.x .> 1.85
    ia = findfirst(inner) - 1 + argmax(landscape.V[inner])
    V_bar = landscape.V[ia]
    ib = argmin(abs.(surfaces.x .- xc))
    U_bar = surfaces.U2[ib]
    rangebars!(ax_V, [landscape.x[ia]], [0.0], [V_bar]; color = (:black, 0.75),
               linewidth = 1.2, whiskerwidth = 8)
    rangebars!(ax_V, [xc + 0.05], [0.0], [U_bar]; color = (bc, 0.9),
               linewidth = 1.2, whiskerwidth = 8)
    text!(ax_V, landscape.x[ia] - 0.05, 0.55 * V_bar;
          text = @sprintf("adiabat\n%.2f eV", V_bar), align = (:right, :center),
          fontsize = 10, color = INK)
    # Just under the top of its bar, in the wedge between U₂ falling away to the right
    # and U₁ climbing: the one place near the crossing no curve passes through.
    text!(ax_V, xc + 0.10, U_bar - 0.04;
          text = @sprintf("BCME U₂\n%.2f eV", U_bar), align = (:left, :top),
          fontsize = 10, color = INK)

    # Direct labels, each on its own curve where the curves are apart.
    j = argmin(abs.(landscape.x .- 2.98))
    text!(ax_V, 2.98, landscape.V[j]; text = "ground adiabat\n(GLE, MDEF, Classical)",
          align = (:right, :top), fontsize = 10, color = INK, font = :bold, offset = (0, -3))
    j = argmin(abs.(surfaces.x .- 2.35))
    text!(ax_V, 2.35, surfaces.U2[j]; text = "BCME U₂ (ionic)", align = (:left, :bottom),
          fontsize = 10, color = INK, font = :bold, offset = (4, 3))
    text!(ax_V, xc, -0.32; text = "Fermi crossing", align = (:center, :bottom),
          fontsize = 10, color = INK)
    y_top = max(meta.E_inc, U_bar) + 0.55
    # U₁ is labelled on its rising branch beyond the crossing, where nothing else is —
    # the well itself has V₁₁, U₁ and the adiabat within 0.2 eV of each other. Where the
    # branch leaves through the frame's top, the label sits just under the top edge:
    # to the right of the branch if a label's width fits before the frame, else to the
    # left, above the label height, where the branch has not yet climbed. When it never
    # reaches that height, the label rides the branch 0.4 Å in from the right edge,
    # clear of the E_inc label that lives at the edge.
    # The surface grid runs from x₀ = 5 Å inward, so the search is confined to the frame.
    # Two lines: at this axis width a one-line label is half an ångström wide and does
    # not fit beside the branch anywhere above 2.5 eV.
    let y_lab = y_top - 0.06 * (y_top + 0.4), lab = "BCME U₁\n(neutral)", w_lab = 0.33,
        outer = (surfaces.x .> xc) .& (surfaces.x .< x_hi)
        k = findlast(i -> outer[i] && surfaces.U1[i] > y_lab, eachindex(surfaces.x))
        if isnothing(k)
            j = argmin(abs.(surfaces.x .- (x_hi - 0.40)))
            text!(ax_V, x_hi - 0.40, surfaces.U1[j]; text = lab, align = (:right, :bottom),
                  fontsize = 10, color = INK, font = :bold, offset = (0, 4))
        elseif surfaces.x[k] + 0.03 + w_lab <= x_hi
            text!(ax_V, surfaces.x[k] + 0.03, y_lab; text = lab, align = (:left, :top),
                  fontsize = 10, color = INK, font = :bold)
        else
            text!(ax_V, surfaces.x[k] - 0.03, y_lab; text = lab, align = (:right, :bottom),
                  fontsize = 10, color = INK, font = :bold)
        end
    end

    xlims!(ax_V, x_lo, x_hi)
    ylims!(ax_V, -0.4, y_top)

    # --- (b) where the molecule goes --------------------------------------------------
    ax_x = Axis(fig[1, 2]; xlabel = "t  [aut]", ylabel = "⟨x(t)⟩  [Å]",
                title = "(b)  mean trajectory of the scattered molecules",
                titlealign = :left, titlefont = :bold, GRID_KW...)
    x_min = minimum(minimum(data[s.key].x) for s in SERIES)
    returns = Float64[]
    for s in SERIES
        d = data[s.key]
        itn = argmin(d.x)
        jr = findfirst(i -> d.x[i] > 0.97 * meta.x0, itn:length(d.x))
        isnothing(jr) || push!(returns, d.t[itn + jr - 1])
    end
    t_end = maximum(data[s.key].t[end] for s in SERIES)
    t_show = isempty(returns) ? t_end : min(t_end, 1.35 * maximum(returns))

    y_lo = min(x_min, xc) - 0.2
    hlines!(ax_x, [xc]; color = (:black, 0.35), linewidth = 1, linestyle = :dot)
    crossed = x_min < xc
    text!(ax_x, 0.985 * t_show, xc + (crossed ? -0.08 : 0.08);
          text = "Fermi crossing", fontsize = 10, color = INK,
          align = (:right, crossed ? :top : :bottom))
    for s in SERIES
        d = data[s.key]
        lines!(ax_x, d.t, d.x; color = s.colour, linewidth = 2, linestyle = s.dash)
    end
    # Direct labels on the outbound leg, at ascending fractions of this run's excursion.
    for (s, yl) in zip(SERIES, x_min .+ (0.42, 0.60, 0.78, 0.93) .* (meta.x0 - x_min))
        d = data[s.key]
        itn = argmin(d.x)
        jr = findfirst(i -> d.x[i] > yl, itn:length(d.x))
        isnothing(jr) && continue
        i = itn + jr - 1
        text!(ax_x, d.t[i] + 0.015 * t_show, d.x[i]; text = s.label, color = INK,
              fontsize = 10, align = (:left, :center), font = :bold)
    end
    xlims!(ax_x, 0, t_show)
    ylims!(ax_x, y_lo, meta.x0 + 0.2)

    # --- (c) the outcome: cumulative distribution of the loss --------------------------
    all_loss = reduce(vcat, data[s.key].loss for s in SERIES)
    in_meV = maximum(all_loss) - min(0.0, minimum(all_loss)) < 0.2
    scale = in_meV ? 1e3 : 1.0
    ax_c = Axis(fig[2:3, 2]; xlabel = "energy to the electrons  [$(in_meV ? "meV" : "eV")]",
                ylabel = "fraction of trajectories (cumulative)",
                title = "(c)  energy handed to the electrons, per trajectory",
                titlealign = :left, titlefont = :bold, GRID_KW...)
    lo = min(0.0, minimum(all_loss)) * scale
    hi = maximum(all_loss) * scale
    pad = 0.06 * max(hi - lo, 1e-9)
    x_left, x_right = lo - pad, hi + pad
    for s in SERIES
        d = data[s.key]
        N = d.n_total
        ls = sort(d.loss) .* scale
        n = length(ls)
        # Right-continuous ECDF, closed at both ends; trapped trajectories sit at +∞ so
        # the curve plateaus at n_scattered / N.
        xs = vcat(x_left, ls, x_right)
        ys = vcat(0.0, (1:n) ./ N, n / N)
        stairs!(ax_c, xs, ys; step = :post, color = s.colour, linewidth = 2,
                linestyle = s.dash)
    end
    # The plateau is the trapping probability; say so where the gap is.
    for s in SERIES
        d = data[s.key]
        d.n_trapped > 0 || continue
        p = d.n_trapped / d.n_total
        text!(ax_c, x_right, 1 - p; text = @sprintf("%.0f %% trapped", 100p),
              align = (:right, :top), fontsize = 10, color = INK, offset = (-3, -3))
    end
    labels = map(SERIES) do s
        d = data[s.key]
        tag = @sprintf("%s   ⟨ΔE⟩ = %s", s.label, pm_meV(d.loss))
        d.n_trapped > 0 ? tag * @sprintf("  (%d of %d trapped)", d.n_trapped, d.n_total) : tag
    end
    # Outside the axis: an ECDF has no empty corner — the classical step fills the left
    # edge and the Markovian rise the right — so an inset would always cover a curve.
    Legend(fig[4, 1:2], legend_elements(), labels; orientation = :horizontal, nbanks = 2,
           framevisible = false, labelsize = 10, tellwidth = false, tellheight = true,
           padding = (0, 0, 0, 0), rowgap = 2, colgap = 24)
    xlims!(ax_c, x_left, x_right)
    ylims!(ax_c, 0, 1.06)

    Label(fig[0, 1:2],
          @sprintf("%.1f eV incident on ErpenbeckThoss (Γ = %g eV), %d trajectories per method, surface at %.0f K",
                   meta.E_inc, meta.Γ, meta.N, meta.T);
          fontsize = 13, font = :bold, padding = (0, 0, 0, 6))
    colsize!(fig.layout, 1, Relative(0.44))
    rowgap!(fig.layout, 8)
    colgap!(fig.layout, 24)
    return fig
end

# ------------------------------------------------------------------------------------
# Sweep figure
# ------------------------------------------------------------------------------------

function sweep_figure(runs, meta)
    E = [r.E for r in runs]
    fig = Figure(size = (1200, 390), fontsize = 12)
    # Room to the right of the last point for the direct labels, ticks only where
    # there is data.
    x_lims = (E[1] - 0.3, E[end] + 2.4)
    xticks = ceil(Int, E[1]):floor(Int, E[end])

    # --- (a) the mean loss ------------------------------------------------------------
    ax_a = Axis(fig[1, 1]; xlabel = "incident energy  [eV]",
                ylabel = "⟨energy to the electrons⟩  [eV]", xticks,
                title = "(a)  mean loss of the scattered molecules", titlealign = :left,
                titlefont = :bold, GRID_KW...)
    ends = Float64[]
    for s in SERIES
        μ = [mean(r.data[s.key].loss) for r in runs]
        se = [last(mean_se(r.data[s.key].loss)) for r in runs]
        errorbars!(ax_a, E, μ, se; color = s.colour, whiskerwidth = 6, linewidth = 1)
        scatterlines!(ax_a, E, μ; color = s.colour, linestyle = s.dash, linewidth = 2,
                      markersize = 9, strokecolor = :white, strokewidth = 1.5)
        push!(ends, μ[end])
    end
    y_top = maximum(ends) * 1.08
    ys = spread!(copy(ends), 0.06 * y_top)
    for (s, y) in zip(SERIES, ys)
        text!(ax_a, E[end] + 0.15, y; text = s.label, align = (:left, :center),
              fontsize = 10, color = INK, font = :bold)
    end
    xlims!(ax_a, x_lims...)
    ylims!(ax_a, -0.05 * y_top, y_top)

    # --- (b) what the BCME ensemble is made of ----------------------------------------
    # Columns stacked to 1 leave no empty corner for an inset, so the legend sits beside
    # the panel in a nested layout of its own.
    gl_b = fig[1, 2] = GridLayout()
    ax_b = Axis(gl_b[1, 1]; xlabel = "incident energy  [eV]",
                ylabel = "fraction of trajectories", xticks,
                title = "(b)  BCME outcomes", titlealign = :left, titlefont = :bold,
                GRID_KW...)
    frac = map(runs) do r
        d = r.data[:bcme]
        n_quiet = count(<(NO_LOSS_EV), d.loss)
        (n_quiet, length(d.loss) - n_quiet, d.n_trapped) ./ d.n_total
    end
    xs = repeat(E, inner = 3)
    hs = reduce(vcat, collect.(frac))
    grp = repeat(1:3, outer = length(E))
    barplot!(ax_b, xs, hs; stack = grp, color = OUTCOME_RAMP[grp], width = 0.34,
             strokecolor = :white, strokewidth = 1.5)
    outcome_elements = [PolyElement(color = c) for c in OUTCOME_RAMP]
    Legend(gl_b[1, 2], outcome_elements,
           [@sprintf("scattered,\nlost < %.1f eV\n(no hop)", NO_LOSS_EV),
            @sprintf("scattered,\nlost ≥ %.1f eV", NO_LOSS_EV), "trapped"];
           framevisible = false, labelsize = 10, rowgap = 6, valign = :center,
           tellheight = false, padding = (0, 0, 0, 0))
    xlims!(ax_b, E[1] - 0.35, E[end] + 0.35)
    ylims!(ax_b, 0, 1.0)

    # --- (c) how far in the molecule gets ---------------------------------------------
    ax_c = Axis(fig[1, 3]; xlabel = "incident energy  [eV]", ylabel = "⟨x_turn⟩  [Å]",
                xticks, title = "(c)  turning point of the scattered molecules",
                titlealign = :left, titlefont = :bold, GRID_KW...)
    hlines!(ax_c, [meta.x_cross]; color = (:black, 0.35), linewidth = 1, linestyle = :dot)
    # Above the line in the middle of the sweep: only the lowest energies turn around
    # beyond the crossing, and the Markovian run hugs it from below.
    text!(ax_c, 0.5 * (E[1] + E[end]), meta.x_cross; text = "Fermi crossing",
          align = (:left, :bottom), fontsize = 10, color = INK, offset = (0, 2))
    ends = Float64[]
    for s in SERIES
        μ = [r.data[s.key].x_turn for r in runs]
        se = [r.data[s.key].x_turn_se for r in runs]
        errorbars!(ax_c, E, μ, se; color = s.colour, whiskerwidth = 6, linewidth = 1)
        scatterlines!(ax_c, E, μ; color = s.colour, linestyle = s.dash, linewidth = 2,
                      markersize = 9, strokecolor = :white, strokewidth = 1.5)
        push!(ends, μ[end])
    end
    y_lo = minimum(minimum(r.data[s.key].x_turn for r in runs) for s in SERIES) - 0.1
    y_hi = maximum(maximum(r.data[s.key].x_turn for r in runs) for s in SERIES) + 0.15
    ys = spread!(copy(ends), 0.06 * (y_hi - y_lo))
    for (s, y) in zip(SERIES, ys)
        text!(ax_c, E[end] + 0.15, y; text = s.label, align = (:left, :center),
              fontsize = 10, color = INK, font = :bold)
    end
    xlims!(ax_c, x_lims...)
    ylims!(ax_c, y_lo, y_hi)

    Legend(fig[2, 1:3], legend_elements(), [s.label for s in SERIES];
           orientation = :horizontal, framevisible = false, padding = (0, 0, 0, 0))
    Label(fig[0, 1:3],
          @sprintf("BCME against the friction pictures: ErpenbeckThoss, Γ = %g eV, %d trajectories per point, surface at %d K",
                   GAMMA_EV, N_TRAJ, T_K);
          fontsize = 13, font = :bold, padding = (0, 0, 0, 6))
    rowgap!(fig.layout, 6)
    colgap!(fig.layout, 26)
    return fig
end

# ------------------------------------------------------------------------------------

# Load every energy of the sweep once; the per-energy figures draw from the same runs.
runs = []
for E in SWEEP_EV
    gfile = scatter_file(scatter_params(E))
    bfile = bcme_file(bcme_params(E))
    missing_ = filter(!isfile, (gfile, bfile))
    if !isempty(missing_)
        @warn "skipping this energy — artefact not found (run scripts/gle/run_gle.jl / scripts/bcme/reduce_bcme.jl)" E missing_
        continue
    end
    series, landscape, meta = load_gle(gfile)
    series[:bcme], surfaces = load_bcme(bfile, meta)
    push!(runs, (; E, data = series, landscape, surfaces, meta))
end
isempty(runs) && error("nothing to plot: no energy has both a GLE and a BCME artefact")

# The BCME surfaces do not depend on the incident energy; every artefact carries the same.
const surfaces = runs[1].surfaces
@printf("BCME surfaces: crossing at %.4f Å, U₂ there %.4f eV above the launch level\n",
        surfaces.x_cross, surfaces.U2[argmin(abs.(surfaces.x .- surfaces.x_cross))])

println("="^100)
@printf("%-5s %-18s %10s %8s %8s %9s %8s\n", "Einc", "method", "loss[meV]", "se", "sd",
        "trapped", "x_turn")
println("="^100)
for r in runs, s in SERIES
    d = r.data[s.key]
    @printf("%-5.1f %-18s %10.2f %8.2f %8.2f %5d/%-3d %8.4f\n", r.E, s.label,
            1e3 * mean(d.loss), 1e3 * last(mean_se(d.loss)), 1e3 * spread(d.loss),
            d.n_trapped, d.n_total, d.x_turn)
end
println("="^100)

let fig = sweep_figure(runs, (; x_cross = surfaces.x_cross))
    outfile = plotsdir("gle", savename("bcme_vs_mdef_sweep",
        (; Gamma_eV = GAMMA_EV, N = N_TRAJ, T_K), "pdf"; sort = false))
    mkpath(dirname(outfile))
    save(outfile, fig)
    save(replace(outfile, ".pdf" => ".png"), fig; px_per_unit = 2)
    println("saved: ", outfile)
end

for r in runs
    r.E in EINC_EV || continue
    fig = comparison_figure(r.data, r.landscape, surfaces, r.meta)
    outfile = plotsdir("gle", savename("bcme_vs_mdef",
        (; Einc_eV = r.E, N = N_TRAJ, T_K), "pdf"; sort = false))
    save(outfile, fig)
    save(replace(outfile, ".pdf" => ".png"), fig; px_per_unit = 2)
    println("saved: ", outfile)
end
