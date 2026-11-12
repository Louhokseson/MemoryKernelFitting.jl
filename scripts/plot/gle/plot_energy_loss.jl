# Electronic energy loss in a scattering event: memory kernel vs Markovian friction.
#
#   julia scripts/plot/gle/plot_energy_loss.jl            # every energy in EINC_EV
#   julia scripts/plot/gle/plot_energy_loss.jl 2.0 3.5    # just these
#
# Reads the artefacts `scripts/gle/run_gle.jl` writes; runs no dynamics itself. The
# producer/reader split matters here for the same reason it does in
# `scripts/plot/ErpenbeckThoss/`: what is drawn is exactly the ensemble that was run, not a
# re-derivation that could drift from it.
#
# One figure per incident energy, four panels each:
#
#   (a) the friction landscape Λ(x,0) and the potential — why anything happens at all
#   (b) ⟨x(t)⟩ — where each method lets the molecule go
#   (c) ⟨E(t)⟩ — the energy loss itself, the headline
#   (d) the distribution of energy handed to the electrons, per trajectory
#
# The sweep spans 0.5–5 eV and the dynamics are not one figure stretched: at 0.5 eV the
# molecule turns around *before* the Fermi crossing and loses nothing, at 5 eV it is off
# the wall in 500 aut while MDEF crawls back for 2000. So every place-dependent number —
# axis limits, the time window, where the direct labels sit, whether the crossing is
# marked at all — is computed from the ensemble being drawn, never hardcoded to one
# energy's shape.
#
# Encoding. Three methods, so colour carries *identity* and gets three fixed Okabe-Ito
# hues assigned in a fixed order — never cycled, never reassigned if a series is dropped.
# Identity never rests on hue alone: each method also has its own dash pattern and is
# direct-labelled at the end of its curve, so the figure survives greyscale printing and
# any colour-vision deficiency. Okabe-Ito is the categorical palette already used across
# `scripts/plot/`, chosen for exactly that property.
#
# One axis per panel — no dual axes anywhere. Panels (a) and (b) both speak about position,
# but they are separate panels sharing an x-scale rather than two y-scales on one plot;
# `Λ` spans eight decades and gets a log axis of its own with `V` on a linked twin panel
# above, which is the small-multiple form, not a dual axis.
#
# Grid and axes are recessive: values are read off the curves and the direct labels, and
# the grid is only there to let the eye carry a level across the panel.

using DrWatson
@quickactivate "MemoryKernelFitting"

using MemoryKernelFitting
using CairoMakie
using HDF5
using Printf
using StatsBase: fit, Histogram
using Statistics

CairoMakie.activate!(type = "pdf")

# ------------------------------------------------------------------------------------
# Input
#
# The artefact names are never spelled out here: they are *recomputed* with the same
# `savename` call `run_gle.jl` uses to write them — same prefix, same keys, in the same
# order, with the same types. A NamedTuple rather than a Dict because the producer passes
# `sort = false`: field order is part of the filename, and a Dict would come back
# alphabetised into an order the producer never wrote. Change the grid in `run_gle.jl`
# or `run_gle.sh` and `scatter_params` / `EINC_EV` are the only things to touch.
# ------------------------------------------------------------------------------------

scatter_params(Einc_eV) = (; Gamma_eV = 0.1, T_K = 300, Einc_eV, N = 400,
                             tfinal = 4000, dt = 0.5, m = 43)

scatter_file(p) = datadir("sims", "gle",
    savename("scatter_ErpenbeckThoss_ground", p, "h5"; sort = false))

# The ten energies of the `run_gle.sh` sweep, unless the command line narrows it.
const EINC_EV = isempty(ARGS) ? collect(0.5:0.5:5.0) : parse.(Float64, ARGS)

# Fixed order, fixed hues. Okabe-Ito: vermillion, blue, bluish green.
const SERIES = [
    (key = "GLE_Prony_m_43_",      label = "MDEF (memory)",    colour = "#D55E00", dash = :solid),
    (key = "MDEF_Markovian_x_0_",  label = "MDEF (Markovian)", colour = "#0072B2", dash = :dash),
    (key = "Classical_no_friction_", label = "Classical",  colour = "#009E73", dash = :dot),
]

const HARTREE_TO_EV = hartree_to_eV(1.0)
const X_CROSSING = 2.0203445

# The classical run is deterministic, so its loss is a spike, not a distribution.
# run_gle.jl integrates it once; older artefacts hold N identical copies. `std` of a
# single sample is NaN, so the one-trajectory case is caught by length.
is_spike(v) = length(v) == 1 || std(v) < 1e-9

function load_run(infile)
    h5open(infile, "r") do f
        d = Dict(s.key => (
                t = read(f["$(s.key)/t_aut"]),
                x = read(f["$(s.key)/x_mean_bohr"]) .* BOHR_IN_ANGSTROM,
                E = read(f["$(s.key)/E_mean_Ha"]) .* HARTREE_TO_EV,
                E_sd = read(f["$(s.key)/E_sd_Ha"]) .* HARTREE_TO_EV,
                loss = read(f["$(s.key)/per_traj_loss"]) .* (1e3 * HARTREE_TO_EV),
            ) for s in SERIES)
        l = (x = read(f["landscape/x_Ang"]),
             Λ = read(f["landscape/Lambda0"]),
             V = read(f["landscape/V_Ha"]) .* HARTREE_TO_EV)
        m = (E_inc = read_attribute(f, "E_incident_eV"),
             N = read_attribute(f, "n_trajectories"),
             T = read_attribute(f, "T_K"),
             x0 = read_attribute(f, "x_start_Ang"),
             Γ = read_attribute(f, "Gamma_eV"),
             # The PES level at the launch point, recovered from the run's own numbers
             # (E_total = E_inc + V(x₀)) rather than re-evaluated — the landscape grid
             # stops at 3 Å and does not reach it.
             V_launch = read_attribute(f, "E_total_Ha") * HARTREE_TO_EV -
                        read_attribute(f, "E_incident_eV"))
        (d, l, m)
    end
end

# ------------------------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------------------------

const GRID_KW = (; xgridcolor = (:black, 0.06), ygridcolor = (:black, 0.06),
                 xgridwidth = 0.8, ygridwidth = 0.8,
                 topspinevisible = false, rightspinevisible = false,
                 spinewidth = 0.8)

function energy_loss_figure(data, landscape, meta)
    x_min = minimum(minimum(data[s.key].x) for s in SERIES)

    # The time window: show until the slowest method has climbed back out to the launch
    # region, plus a third for the divergence to read. A method that never returns (none
    # in the current sweep, but sticking at low Γ is possible) falls back to the full run.
    returns = Float64[]
    for s in SERIES
        d = data[s.key]
        itn = argmin(d.x)
        j = findfirst(i -> d.x[i] > 0.97 * meta.x0, itn:length(d.x))
        isnothing(j) || push!(returns, d.t[itn + j - 1])
    end
    t_end = maximum(data[s.key].t[end] for s in SERIES)
    t_show = isempty(returns) ? t_end : min(t_end, 1.35 * maximum(returns))

    fig = Figure(size = (900, 640), fontsize = 12)

    # --- (a) the landscape -----------------------------------------------------------
    ax_V = Axis(fig[1, 1]; ylabel = "PES  [eV]", xticklabelsvisible = false,
                title = "(a)  the potential energy surface and its friction", titlealign = :left,
                titlefont = :bold, GRID_KW...)
    ax_Λ = Axis(fig[2, 1]; xlabel = "x  [Å]", ylabel = "Λ(x, 0)  [mₑ/aut]",
                yscale = log10, GRID_KW...)
    rowsize!(fig.layout, 1, Relative(0.16))
    linkxaxes!(ax_V, ax_Λ)

    for ax in (ax_V, ax_Λ)
        vlines!(ax, [X_CROSSING]; color = (:black, 0.35), linewidth = 1, linestyle = :dot)
    end
    lines!(ax_V, landscape.x, landscape.V; color = :black, linewidth = 2)
    lines!(ax_Λ, landscape.x, landscape.Λ; color = :black, linewidth = 2)

    # The Fermi-crossing barrier: how far the PES climbs from the launch level at 5 Å to
    # the crossing. An incident energy below this never reaches the resonance at all —
    # which is the whole story of the low-energy figures — so the number belongs on the
    # landscape panel, drawn as a capped bar from the (dotted) launch level up to V(x_c).
    j = searchsortedfirst(landscape.x, X_CROSSING)
    w = (X_CROSSING - landscape.x[j-1]) / (landscape.x[j] - landscape.x[j-1])
    V_cross = (1 - w) * landscape.V[j-1] + w * landscape.V[j]
    hlines!(ax_V, [meta.V_launch]; color = (:black, 0.35), linewidth = 1, linestyle = :dot)
    rangebars!(ax_V, [X_CROSSING], [meta.V_launch], [V_cross];
               color = (:black, 0.75), linewidth = 1.2, whiskerwidth = 8)
    text!(ax_V, X_CROSSING + 0.06, V_cross + 0.35;
          text = @sprintf("Fermi-crossing barrier\n%.2f eV above x = %.0f Å", V_cross - meta.V_launch, meta.x0),
          align = (:left, :bottom), fontsize = 10, color = (:black, 0.75))
    text!(ax_Λ, X_CROSSING + 0.03, 3e2; text = "Fermi crossing\nΛ = 1399",
          align = (:left, :top), fontsize = 10, color = (:black, 0.75))
    xlims!(ax_Λ, 1.3, 3.0)

    # --- (b) where the molecule goes -------------------------------------------------
    ax_x = Axis(fig[1:2, 2]; xlabel = "t  [aut]", ylabel = "⟨x(t)⟩  [Å]",
                title = "(b)  trajectory", titlealign = :left, titlefont = :bold, GRID_KW...)
    # The lower limit always includes the crossing, even below 1.5 eV where the molecule
    # turns around above it — *that* is the story of those panels: no crossing, no loss.
    y_lo = min(x_min, X_CROSSING) - 0.2
    hlines!(ax_x, [X_CROSSING]; color = (:black, 0.35), linewidth = 1, linestyle = :dot)
    # Below the curves when they cross it, above when they turn around short of it —
    # in the second case "below" is the last 0.2 Å of the axis and the text clips.
    crossed = x_min < X_CROSSING
    text!(ax_x, 0.985 * t_show, X_CROSSING + (crossed ? -0.08 : 0.08);
          text = "Fermi crossing", fontsize = 10,
          color = (:black, 0.75), align = (:right, crossed ? :top : :bottom))
    for s in SERIES
        d = data[s.key]
        lines!(ax_x, d.t, d.x; color = s.colour, linewidth = 2, linestyle = s.dash,
               label = s.label)
    end
    xlims!(ax_x, 0, t_show)
    ylims!(ax_x, y_lo, meta.x0 + 0.2)

    # Direct labels on the *outbound* leg, where the three curves are furthest apart. At
    # the turning point GLE and Classical are within 0.02 Å of each other and any label
    # there collides; on the way out MDEF has fallen a clear 1 Å behind. The three
    # heights are fractions of this run's excursion, ascending so the labels spread out
    # along the climb (and simply stack along the merged curve when the methods agree).
    for (s, yl) in zip(SERIES, x_min .+ (0.50, 0.83, 0.92) .* (meta.x0 - x_min))
        d = data[s.key]
        itn = argmin(d.x)
        j = findfirst(i -> d.x[i] > yl, itn:length(d.x))
        isnothing(j) && continue
        i = itn + j - 1
        text!(ax_x, d.t[i] + 0.017 * t_show, d.x[i]; text = s.label, color = s.colour,
              fontsize = 10, align = (:left, :center), font = :bold)
    end

    # --- (c) the energy loss ---------------------------------------------------------
    ax_E = Axis(fig[3, 1]; xlabel = "t  [aut]", ylabel = "⟨E(t)⟩  [eV]",
                title = "(c)  total energy", titlealign = :left, titlefont = :bold, GRID_KW...)
    E_lo, E_hi = Inf, -Inf
    for s in SERIES
        d = data[s.key]
        w = d.t .<= t_show
        E_lo = min(E_lo, minimum((d.E .- d.E_sd)[w]))
        E_hi = max(E_hi, maximum((d.E .+ d.E_sd)[w]))
    end
    r = max(E_hi - E_lo, 1e-6)

    # When the frictionless molecule passes the resonance, inbound and outbound. Those
    # two instants are what panel (c) is about: the GLE loses energy in two steps, one
    # per crossing, while MDEF loses almost everything at the first. Below ~1.5 eV the
    # molecule turns around before the resonance and there is nothing to mark.
    let d = data[SERIES[3].key], itn = argmin(data[SERIES[3].key].x)
        if d.x[itn] < X_CROSSING
            t_in = d.t[findlast(i -> d.x[i] > X_CROSSING, 1:itn)]
            k = findfirst(i -> d.x[i] > X_CROSSING, itn:length(d.x))
            t_out = d.t[itn + k - 1]
            vlines!(ax_E, [t_in, t_out]; color = (:black, 0.3), linewidth = 1, linestyle = :dot)
            text!(ax_E, t_in - 0.01 * t_show, E_hi + 0.16r; text = "inbound\ncrossing",
                  fontsize = 9, color = (:black, 0.7), align = (:right, :top))
            text!(ax_E, t_out + 0.01 * t_show, E_hi + 0.16r; text = "outbound\ncrossing",
                  fontsize = 9, color = (:black, 0.7), align = (:left, :top))
        end
    end

    for s in SERIES
        d = data[s.key]
        band!(ax_E, d.t, d.E .- d.E_sd, d.E .+ d.E_sd; color = (s.colour, 0.15))
        lines!(ax_E, d.t, d.E; color = s.colour, linewidth = 2, linestyle = s.dash)
    end

    # Final ⟨E⟩, direct-labelled at each curve's end — the values are the point of the
    # panel (MDEF (Markovian) lands at V(x_c) ≈ −0.39 eV whatever the incident energy),
    # so they are printed rather than left to be read off the axis. Finishes closer than
    # a label height (everything, below the barrier) are nudged apart bottom-up; the
    # precision follows the axis range, which spans only meV when nothing crosses.
    let gap = 0.11r                                  # one label height, in data units
        ends = [(colour = s.colour,
                 E = data[s.key].E[findlast(<=(t_show), data[s.key].t)]) for s in SERIES]
        ys = [e.E for e in ends]
        ord = sortperm(ys)
        for k in 2:length(ord)
            ys[ord[k]] = max(ys[ord[k]], ys[ord[k-1]] + gap)
        end
        over = maximum(ys) - (E_hi + 0.05r)          # keep the ladder inside the frame
        over > 0 && (ys .-= over)
        for (e, y) in zip(ends, ys)
            # Bottom-aligned just above its own y so the label rides on the curve
            # rather than being struck through by it.
            text!(ax_E, 0.99 * t_show, y + 0.015r;
                  text = r < 0.1 ? @sprintf("%.4f eV", e.E) : @sprintf("%.2f eV", e.E),
                  color = e.colour, fontsize = 10, font = :bold, align = (:right, :bottom))
        end
    end
    xlims!(ax_E, 0, t_show)
    ylims!(ax_E, E_lo - 0.17r, E_hi + 0.19r)

    # --- (d) the outcome -------------------------------------------------------------
    ax_h = Axis(fig[3, 2]; xlabel = "energy to the electrons  [meV]", ylabel = "trajectories",
                title = "(d)  per-trajectory loss", titlealign = :left, titlefont = :bold,
                GRID_KW...)
    for s in SERIES
        d = data[s.key]
        # The classical run is deterministic: a spike at zero, drawn as a rule rather
        # than a one-bin histogram that would misread as a wide distribution.
        if is_spike(d.loss)
            vlines!(ax_h, [mean(d.loss)]; color = s.colour, linewidth = 2, linestyle = s.dash)
        else
            hist!(ax_h, d.loss; bins = 30, color = (s.colour, 0.55),
                  strokecolor = s.colour, strokewidth = 0.8)
        end
    end
    # Labels above each distribution rather than on the baseline, where they would sit
    # inside the bars and overlap the neighbouring series. GLE and MDEF get fixed
    # *separate* rows because below ~1.5 eV the two distributions — and their means —
    # coincide, and one shared row would print the labels on top of each other.
    let ymax = maximum(maximum(h.weights) for h in
            (fit(Histogram, data[s.key].loss; nbins = 30) for s in SERIES if !is_spike(data[s.key].loss)))
        ylims!(ax_h, 0, 1.52 * ymax)
        x_lo = min(0.0, minimum(minimum(data[s.key].loss) for s in SERIES))
        x_hi = maximum(maximum(data[s.key].loss) for s in SERIES)
        # Classical sits at exactly 0, inside the GLE distribution's x-range, so it is
        # dropped to its own row just right of its rule rather than fought into the top.
        for (s, yfrac, align) in zip(SERIES, (1.04, 1.22, 0.72),
                                     ((:center, :bottom), (:center, :bottom), (:left, :bottom)))
            d = data[s.key]
            x = is_spike(d.loss) ? x_lo + 0.04 * (x_hi - x_lo) : mean(d.loss)
            # A label centred on a distribution hugging the right axis edge clips;
            # once the mean is in the last sixth of the range, hang it to the left.
            x > x_hi - 0.17 * (x_hi - x_lo) && (align = (:right, :bottom))
            μ = mean(d.loss)
            abs(μ) < 0.5 && (μ = 0.0)   # "%.0f" of -0.4 prints "-0 meV"
            text!(ax_h, x, yfrac * ymax;
                  text = @sprintf("%s\n%.0f meV", s.label, μ),
                  color = s.colour, fontsize = 10, align = align, font = :bold)
        end
    end

    Legend(fig[4, 1:2], ax_x; orientation = :horizontal, framevisible = false,
           padding = (0, 0, 0, 0))

    Label(fig[0, 1:2],
          @sprintf("%.1f eV incident on the ErpenbeckThoss (Γ = %g eV) ground-state PES, %d trajectories, surface at %.0f K",
                   meta.E_inc, meta.Γ, meta.N, meta.T);
          fontsize = 13, font = :bold, padding = (0, 0, 0, 6))

    rowgap!(fig.layout, 6)
    colgap!(fig.layout, 22)
    return fig
end

# ------------------------------------------------------------------------------------

n_done = 0
for E in EINC_EV
    infile = scatter_file(scatter_params(E))
    if !isfile(infile)
        @warn "no artefact for this energy — run scripts/gle/run_gle.jl first" E infile
        continue
    end
    data, landscape, meta = load_run(infile)
    fig = energy_loss_figure(data, landscape, meta)

    outfile = plotsdir("gle", savename("energy_loss_scatter",
        (; Einc_eV = meta.E_inc, N = meta.N, T_K = Int(meta.T)), "pdf"; sort = false))
    mkpath(dirname(outfile))
    save(outfile, fig)
    save(replace(outfile, ".pdf" => ".png"), fig; px_per_unit = 2)
    println("saved: ", outfile)
    global n_done += 1
end
n_done == 0 && error("nothing plotted: no artefacts in $(datadir("sims", "gle")) matched")
