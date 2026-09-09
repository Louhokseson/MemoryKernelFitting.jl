# Reduce Nils' raw BCME trajectories to the per-ensemble artefacts the plots read.
#
#   julia --project=. scripts/bcme/reduce_bcme.jl                    # every file in data/BCME_calculations
#   julia --project=. scripts/bcme/reduce_bcme.jl a.h5 some/dir      # just these files / directories
#   julia --project=. scripts/bcme/reduce_bcme.jl --force            # redo artefacts that already exist
#   julia --project=. scripts/bcme/reduce_bcme.jl --outdir DIR       # write elsewhere than data/sims/bcme
#
# A raw file is 193 MB: 400 trajectories × 20 002 samples of position, velocity and time,
# plus an outcome flag, and nothing else — no attributes, no electronic state. Everything a
# figure needs from it fits in 200 kB, so this runs once, where the raw data lives, and the
# plots never open the raw files again. One file takes about a second; it is I/O bound, so
# run it serially on the node that holds the data and copy the artefacts home. Keep the raw
# trajectories archived: the mean trajectory, the turning point and the surface check
# below all needed the full x(t), and any question about *where* the hops happen will too.
#
# ----------------------------------------------------------------------------------
# What is known about the raw files, and how
#
# Established from the trajectories themselves (2026-09-09), since the files carry no
# metadata; the checks marked ✓ are re-done on every file below and warn if they fail.
#
#   - Launch is identical to `run_gle.jl`: x₀ = 9.4486 bohr = 5.0 Å, ½mv₀² equal to the
#     `incidence_energy` in eV to six figures with m = 1 u, and monoenergetic — all
#     trajectories share one (x₀, v₀). ✓
#   - The first two steps accelerate at the slope of V₂₂ (−0.0111 eV/Å; V₁₁ would give
#     −0.0455), so the molecule starts on the *ionic* diabat — the lower one at 5 Å and
#     the branch our ground adiabat starts on. ✓
#   - `gamma=0.1` is 0.1 eV and `bandwidth=50` is 50 eV. Integrating the BCME state-2
#     force (`cme.jl`, `acceleration!`) with those values reproduces the surface the
#     hopless 0.5, 1.0 and 1.5 eV runs conserve energy on to 0.004 meV rms; Γ = 0.1 Ha
#     misses by 1 eV at the turning point. The surface temperature cannot be read off the
#     data (Γ ≫ k_BT) and is *assumed* to be our 300 K — see `T_K_ASSUMED`.
#   - `dt=0.05` and `tmax=1000.0` are femtoseconds; the stored `Time` is already in aut
#     (2.067 aut per step, 41 341 aut in all). The last sample is duplicated, a `saveat`
#     artefact, and is dropped here. ✓
#   - `OutputOutcome` is 1 for a molecule that left and 0 for one that did not: the counts
#     equal `x_end < 20 bohr` in nine files of ten and differ by one trajectory in the
#     tenth (4.5 eV: a molecule sitting at 5.04 Å with 0.8 meV of kinetic energy).
#
# ----------------------------------------------------------------------------------
# What "energy to the electrons" means here
#
# The electronic state is not stored, so the potential energy at t_end is not known — but
# every scattered molecule is at x ≥ 20 bohr where both diabats are flat to 10⁻⁵ eV, so
# its *translational* energy is. The loss is
#
#     ΔE = E_total − V∞ − ½mv_end²,     E_total = ½mv₀² + V₂₂(x₀),
#
# with V₂₂ the diabat the run starts on and V∞ its asymptote (the model's `V∞`). It is
# written for every trajectory but is only meaningful for the scattered ones; a trapped
# molecule is bound in the neutral Morse well 3.4 eV below the V₁₁ asymptote and has no
# asymptotic energy. `loss`, `E_trans`, `x_turn` and the mean trajectory are therefore
# over the scattered trajectories, and `stuck` is the trapped fraction. The reference
# differs from `run_gle.jl`'s (ground adiabat at x₀ and at 40 Å) by 0.02 meV.
#
# ----------------------------------------------------------------------------------
# The artefact
#
# One HDF5 file per raw file, named like the GLE artefacts and with the same keys where
# the quantity exists for both, so a reader can treat BCME as a fourth method:
#
#   BCME/                      per trajectory: per_traj_loss [Ha], outcome, KE_end_Ha,
#                              v_end [bohr/aut], x_min_bohr, x_end_bohr, t_exit_aut (NaN
#                              if it never left), n_turning_points
#                              scalars: loss, loss_se, loss_sd, E_trans, E_trans_se,
#                              x_turn, x_turn_se [Å], x_end [Å], stuck, n_scattered
#                              mean trajectory, scattered only, every 5th sample (10 aut):
#                              t_aut, x_mean_bohr, x_sd_bohr, KE_mean_Ha; and
#                              x_mean_all_bohr over every trajectory
#   landscape/                 the surfaces BCME integrates on, from x₀ inward: x_Ang,
#                              U2_Ha, U1_Ha (broadened diabats), V11_Ha, V22_Ha (bare)
#   attributes                 Gamma_eV, E_incident_eV, bandwidth_eV, dt_fs, tmax_fs,
#                              n_trajectories, x_start_Ang, mass_u, E_total_Ha,
#                              V_asymptote_Ha, initial_state, T_K_assumed, source, git
#
# There is no `E_mean_Ha`: without the state the total energy along the path is not
# defined. `KE_mean_Ha` is what can be had.

using DrWatson
@quickactivate "MemoryKernelFitting"

using NQCDynamics                     # ErpenbeckThoss and the BCME broadening integrals
using MemoryKernelFitting: BOHR_IN_ANGSTROM, hartree_to_eV
using HDF5
using Printf
using Statistics
using Dates
using Unitful, UnitfulAtomic

const NQCModels = NQCDynamics.NQCModels
const SHM = NQCDynamics.DynamicsMethods.SurfaceHoppingMethods

# ------------------------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------------------------

let inputs = String[], force = false, outdir = datadir("sims", "bcme"), i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--force"
            force = true; i += 1
        elseif ARGS[i] == "--outdir" && i < length(ARGS)
            outdir = ARGS[i+1]; i += 2
        else
            push!(inputs, ARGS[i]); i += 1
        end
    end
    isempty(inputs) && push!(inputs, datadir("BCME_calculations"))
    global const INPUTS = inputs
    global const FORCE = force
    global const OUTDIR = outdir
end

const MASS = austrip(1.0u"u")         # `Atoms([austrip(1.0u"u")])` in both calculations
const T_K_ASSUMED = 300.0             # not recoverable from the data; ours
const STRIDE = 5                      # mean-trajectory samples: every 5th step ≈ 10 aut,
                                      # the spacing the GLE artefacts use
const X_INNER_ANG = 1.3               # inner end of the landscape grid, as in run_gle.jl
const HARTREE_TO_EV = hartree_to_eV(1.0)

# ------------------------------------------------------------------------------------
# The surfaces BCME integrates on
# ------------------------------------------------------------------------------------

"""
    bcme_surfaces(x_from_Ang, x_to_Ang; Γ_eV, T_K, bandwidth_eV) → NamedTuple

The broadened diabats, in Hartree, by integrating the state-2 force of `cme.jl`'s
`acceleration!` inward from `x_from`, anchored at `V₂₂(x_from)`.

The force is conservative in a fixed state — the mixed partials of the broadening
corrections in (h, Γ) agree — and `U₁ = U₂ − h` exactly, so one integration gives both
branches. Verified against the hopless runs to 0.004 meV rms (see the header).
"""
function bcme_surfaces(x_from_Ang, x_to_Ang; Γ_eV, T_K, bandwidth_eV, npts = 4001)
    et = NQCModels.QuantumModels.ErpenbeckThoss(; Γ = austrip(Γ_eV * u"eV"))
    β = 1 / austrip(T_K * u"K")
    W = austrip(bandwidth_eV * u"eV")
    μ = 0.0                                           # NQCModels.fermilevel default
    fermi(ϵ) = 1 / (1 + exp(β * (ϵ - μ)))

    function force2(x)
        V = NQCModels.potential(et, x)
        D = NQCModels.derivative(et, x)
        h = V[2, 2] - V[1, 1]
        Γ = 2π * V[2, 1]^2
        n = SHM.evaluate_broadening(h, μ, β, Γ, W)
        n2 = SHM.evaluate_force_broadening(h, μ, β, Γ, W)
        ∂h = D[2, 2] - D[1, 1]
        ∂Γ = 4π * V[2, 1] * D[2, 1]
        return -D[2, 2] + fermi(h) * ∂h - ∂h * n - ∂Γ / Γ * n2
    end

    xg = collect(range(austrip(x_from_Ang * u"Å"), austrip(x_to_Ang * u"Å"); length = npts))
    F = force2.(xg)
    V = [NQCModels.potential(et, x) for x in xg]
    U2 = zeros(npts)
    U2[1] = V[1][2, 2]
    for i in 2:npts
        U2[i] = U2[i-1] - 0.5 * (F[i] + F[i-1]) * (xg[i] - xg[i-1])
    end
    h = [v[2, 2] - v[1, 1] for v in V]
    # The diabatic crossing h = 0, which the broadened branches share since U₂ − U₁ = h.
    ic = findfirst(>(0), h)                           # inward: h is negative outside
    w = -h[ic-1] / (h[ic] - h[ic-1])
    x_cross = ((1 - w) * xg[ic-1] + w * xg[ic]) * BOHR_IN_ANGSTROM

    return (x_Ang = xg .* BOHR_IN_ANGSTROM, U2 = U2, U1 = U2 .- h,
            V11 = [v[1, 1] for v in V], V22 = [v[2, 2] for v in V], x_cross)
end

# Per Γ, not per file: the surfaces do not depend on the incident energy.
const SURFACES = Dict{Tuple{Float64,Float64},Any}()
surfaces_for(Γ_eV, W_eV) = get!(SURFACES, (Γ_eV, W_eV)) do
    bcme_surfaces(5.0, X_INNER_ANG; Γ_eV, T_K = T_K_ASSUMED, bandwidth_eV = W_eV)
end

# ------------------------------------------------------------------------------------
# One file
# ------------------------------------------------------------------------------------

artefact_name(p) = savename("bcme_scatter_ErpenbeckThoss",
    (; Gamma_eV = p["gamma"], Einc_eV = p["incidence_energy"], N = p["trajectories"],
       tmax_fs = p["tmax"], dt_fs = p["dt"], bandwidth_eV = p["bandwidth"]),
    "h5"; sort = false)

function reduce_file(path)
    _, p, _ = parse_savename(basename(path))
    for k in ("gamma", "incidence_energy", "bandwidth", "trajectories", "tmax", "dt")
        haskey(p, k) || error("$(basename(path)): no `$k` in the name")
    end
    get(p, "method", "BCME") == "BCME" || @warn "not a BCME file by its name" path
    outfile = joinpath(OUTDIR, artefact_name(p))
    if isfile(outfile) && !FORCE
        println("[skip] already done: ", basename(outfile))
        return nothing
    end

    Γ_eV = Float64(p["gamma"]); E_inc = Float64(p["incidence_energy"]); W_eV = Float64(p["bandwidth"])
    et = NQCModels.QuantumModels.ErpenbeckThoss(; Γ = austrip(Γ_eV * u"eV"))
    V∞ = et.V∞

    h5open(path, "r") do h
        names = sort(filter(k -> startswith(k, "trajectory_"), keys(h));
                     by = k -> parse(Int, k[length("trajectory_")+1:end]))
        N = length(names)
        N == p["trajectories"] ||
            @warn "trajectory count differs from the name" path N p["trajectories"]

        t = read(h[names[1]]["Time"])
        n_t = length(t)
        # The duplicated final sample: a `saveat` artefact, dropped so the grid is uniform.
        n_t > 2 && t[end] - t[end-1] < 0.5 * (t[2] - t[1]) && (n_t -= 1)
        t = t[1:n_t]
        keep = 1:STRIDE:n_t

        outcome = zeros(Int, N)
        x_min = zeros(N); x_end = zeros(N); v_end = zeros(N); KE_end = zeros(N)
        t_exit = fill(NaN, N); n_turn = zeros(Int, N)
        x_sum_sc = zeros(length(keep)); x2_sum_sc = zeros(length(keep))
        KE_sum_sc = zeros(length(keep)); x_sum_all = zeros(length(keep))
        x0 = NaN; v0 = NaN

        for (k, name) in enumerate(names)
            g = h[name]
            x = vec(read(g["OutputPosition"]))
            v = vec(read(g["OutputVelocity"]))
            length(x) >= n_t && length(v) >= n_t ||
                error("$(basename(path)) $name: $(length(x)) samples, expected $n_t")

            if k == 1
                x0, v0 = x[1], v[1]
                # Which diabat the run starts on, from the first step's acceleration.
                F_obs = MASS * (v[2] - v[1]) / (t[2] - t[1])
                D = NQCModels.derivative(et, x0)
                state0 = abs(F_obs + D[2, 2]) < abs(F_obs + D[1, 1]) ? 2 : 1
                state0 == 2 || @warn "launch is on the neutral diabat, not the ionic one the GLE starts on" path F_obs
            elseif x[1] != x0 || v[1] != v0
                @warn "not monoenergetic: trajectory starts differ" path name x[1] v[1]
            end

            outcome[k] = read(g["OutputOutcome"])
            x_min[k] = minimum(@view x[1:n_t])
            x_end[k] = x[n_t]
            v_end[k] = v[n_t]
            KE_end[k] = 0.5 * MASS * v[n_t]^2
            j = findlast(<(x0), @view x[1:n_t])
            (j === nothing || j == n_t) || (t_exit[k] = t[j])
            n_turn[k] = count(i -> v[i] * v[i+1] < 0, 1:n_t-1)

            xk = @view x[keep]
            x_sum_all .+= xk
            if outcome[k] == 1
                x_sum_sc .+= xk
                x2_sum_sc .+= xk .^ 2
                KE_sum_sc .+= 0.5 .* MASS .* (@view v[keep]) .^ 2
            end
        end

        E_total = 0.5 * MASS * v0^2 + NQCModels.potential(et, x0)[2, 2]
        loss = E_total - V∞ .- KE_end
        sc = outcome .== 1
        n_sc = count(sc)
        n_sc > 0 || @warn "no scattered trajectory in this file" path
        se(v) = length(v) > 1 ? std(v) / sqrt(length(v)) : NaN
        x_mean_sc = x_sum_sc ./ max(n_sc, 1)
        x_sd_sc = sqrt.(max.(x2_sum_sc ./ max(n_sc, 1) .- x_mean_sc .^ 2, 0.0))

        mkpath(OUTDIR)
        h5open(outfile, "w") do f
            g = create_group(f, "BCME")
            g["per_traj_loss"] = loss
            g["outcome"] = outcome
            g["KE_end_Ha"] = KE_end
            g["v_end"] = v_end
            g["x_min_bohr"] = x_min
            g["x_end_bohr"] = x_end
            g["t_exit_aut"] = t_exit
            g["n_turning_points"] = n_turn

            g["loss"] = n_sc > 0 ? mean(loss[sc]) : NaN
            g["loss_se"] = n_sc > 0 ? se(loss[sc]) : NaN
            g["loss_sd"] = n_sc > 1 ? std(loss[sc]) : NaN
            g["E_trans"] = n_sc > 0 ? mean(KE_end[sc]) : NaN
            g["E_trans_se"] = n_sc > 0 ? se(KE_end[sc]) : NaN
            g["x_turn"] = n_sc > 0 ? mean(x_min[sc]) * BOHR_IN_ANGSTROM : NaN
            g["x_turn_se"] = n_sc > 0 ? se(x_min[sc]) * BOHR_IN_ANGSTROM : NaN
            g["x_end"] = n_sc > 0 ? mean(x_end[sc]) * BOHR_IN_ANGSTROM : NaN
            g["stuck"] = 1 - n_sc / N
            g["n_scattered"] = n_sc

            g["t_aut"] = t[keep]
            g["x_mean_bohr"] = x_mean_sc
            g["x_sd_bohr"] = x_sd_sc
            g["KE_mean_Ha"] = KE_sum_sc ./ max(n_sc, 1)
            g["x_mean_all_bohr"] = x_sum_all ./ N

            s = surfaces_for(Γ_eV, W_eV)
            f["landscape/x_Ang"] = s.x_Ang
            f["landscape/U2_Ha"] = s.U2
            f["landscape/U1_Ha"] = s.U1
            f["landscape/V11_Ha"] = s.V11
            f["landscape/V22_Ha"] = s.V22
            attrs(f["landscape"])["x_cross_Ang"] = s.x_cross
            attrs(f["landscape"])["T_K"] = T_K_ASSUMED
            attrs(f["landscape"])["what"] =
                "broadened diabats BCME integrates on (U2 ionic, U1 neutral), by integrating the state-2 force from x0 inward; bare diabats alongside"

            attrs(f)["Gamma_eV"] = Γ_eV
            attrs(f)["E_incident_eV"] = E_inc
            attrs(f)["bandwidth_eV"] = W_eV
            attrs(f)["dt_fs"] = Float64(p["dt"])
            attrs(f)["tmax_fs"] = Float64(p["tmax"])
            attrs(f)["n_trajectories"] = N
            attrs(f)["x_start_Ang"] = x0 * BOHR_IN_ANGSTROM
            attrs(f)["mass_u"] = 1.0
            attrs(f)["E_total_Ha"] = E_total
            attrs(f)["V_asymptote_Ha"] = V∞
            attrs(f)["initial_state"] = 2
            attrs(f)["T_K_assumed"] = T_K_ASSUMED
            attrs(f)["outcome_meaning"] = "1 scattered, 0 trapped (Nils' OutputOutcome)"
            attrs(f)["units"] = "energies in Hartree, x in bohr unless _Ang, times in aut"
            attrs(f)["source"] = basename(path)
            attrs(f)["git"] = gitdescribe(projectdir())
            attrs(f)["reduced_on"] = string(now())
        end

        eV(x) = 1e3 * HARTREE_TO_EV * x
        @printf("Γ = %-5g E = %-4g  %3d/%3d scattered  ΔE = %8.2f ± %6.2f meV  x_turn = %.4f Å  → %s\n",
                Γ_eV, E_inc, n_sc, N, eV(mean(loss[sc])), eV(se(loss[sc])),
                mean(x_min[sc]) * BOHR_IN_ANGSTROM, basename(outfile))
    end
    return outfile
end

# ------------------------------------------------------------------------------------

files = String[]
for p in INPUTS
    if isdir(p)
        append!(files, filter(f -> endswith(f, ".h5"), readdir(p; join = true)))
    elseif isfile(p)
        push!(files, p)
    else
        @warn "not found, skipped" p
    end
end
sort!(files)
isempty(files) && error("nothing to reduce: no .h5 files under $(join(INPUTS, ", "))")

println("reducing $(length(files)) file$(length(files) == 1 ? "" : "s") → $OUTDIR")
for f in files
    reduce_file(f)
end
