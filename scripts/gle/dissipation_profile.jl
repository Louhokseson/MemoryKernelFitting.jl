# Where along the reaction path is the energy actually given to the electrons?
#
#   julia --project=. scripts/gle/dissipation_profile.jl
#
# `run_gle.jl` finds GLE and MDEF agreeing to `+1.1 ± 2.5` meV out of ~69 meV handed to
# the electrons — a null result. That is surprising, because the transit through the
# friction resonance is strongly non-Markovian: the fitted kernel at the crossing has a
# first moment `∫τK dτ / ∫K dτ = 1350` aut, while the molecule crosses the peak in 22-223
# aut depending on how wide you call it. Memory ought to matter there.
#
# The hypothesis this script was written to test was that **the peak is too narrow in `x`
# to contribute much** — `Λ` is 1399 at 2.0203 Å but 6.75 by 2.05 Å, enormous height and
# negligible width, and the friction impulse goes as `∫γ dx` rather than `γ_max`. That
# would have made the null result trivial: the region where memory matters contributes too
# little of the total to show up.
#
# **That hypothesis is wrong.** The peak contributes 95-100 % of the energy loss in both
# models. Essentially all the dissipation happens exactly where the dynamics is most
# non-Markovian.
#
# What the profile does show is that memory changes *where* the energy is taken, not how
# much. Paired against MDEF on identical initial conditions, the GLE takes statistically
# the same amount at the peak and the same total, but keeps draining on the descent
# (2.07-2.5 Å) where the Markovian model takes nothing — the auxiliary variables absorb at
# the crossing and release downstream, because the kernel has not decayed. The total is
# protected by a sum rule the two models share, `Λ(x,0) = ∫K dτ`; the *distribution along
# the path* is not, and that is where the memory shows up.
#
# Note on what `ΔE` means for the GLE. The physical energy `KE + V` also flows *reversibly*
# into the auxiliary variables and can come back, so a bin's `ΔE` is not all dissipation.
# Over the whole outward trip that distinction closes — the molecule ends up far away with
# `g → 0` and the auxiliary variables decoupled — so the *cumulative* profile is
# meaningful even where the differential one is not.

using DrWatson
@quickactivate "MemoryKernelFitting"

include(joinpath(@__DIR__, "prony_memory.jl"))

using HDF5
using Printf
using Random
using Statistics

const GAMMA_EV = 0.1
const FIT_FILE = datadir("fits",
    "prony_stage2Cheby_Gamma_eV=0.1_m=43_K_C=20_K_R=3_deg=22_eps=0.001_anchor=1000.0_thetav=0.5.h5")

const T_K = 300.0
const N_TRAJ = 400
const T_FINAL = 8_000.0
const MD_DT = 2.0
const X_WELL = (1.60, 1.92)
const SEED = 20260902

# The bins the question is about, in Å.
# The fine grid the saved profile is binned on. Fine enough to resolve the friction peak
# (Λ falls from 1399 to 6.75 between 2.020 and 2.050 Å) without emptying the bins.
const XGRID = range(1.55, 2.60; step = 0.005)

const REGIONS = [("well      x < 1.95", -Inf, 1.95),
                 ("approach  1.95-1.97", 1.95, 1.9703445),
                 ("PEAK      ±0.05 Å ", 1.9703445, 2.0703445),
                 ("descent   2.07-2.5", 2.0703445, 2.5),
                 ("far       x > 2.5 ", 2.5, Inf)]

isfile(FIT_FILE) || error("fit file not found: $FIT_FILE")

kT_fit = boltzmann_energy(300.0)
kT = boltzmann_energy(T_K)
atoms = Atoms([austrip(1.0u"u")])
mass = atoms.masses[1]
et = ErpenbeckThossModel(; Γ_eV = GAMMA_EV)
pes = erpenbeck_thoss_ground(GAMMA_EV)
V(x_bohr) = NQCModels.potential(pes, hcat(x_bohr))

function sample_well(rng, β, lo, hi)
    Umin = minimum(V, range(lo, hi; length = 401))
    while true
        x = lo + (hi - lo) * rand(rng)
        rand(rng) < exp(-β * (V(x) - Umin)) && return x
    end
end

"""
    profile(label, build) → (total, by_region)

Energy lost per region, averaged over the ensemble. `ΔE` between consecutive saved states
is attributed to the region containing the midpoint of the step.
"""
function profile(label, build)
    rng_ic = MersenneTwister(SEED)
    rng_method = MersenneTwister(SEED + 1)
    β = 1 / kT
    lo, hi = austrip(X_WELL[1] * u"Å"), austrip(X_WELL[2] * u"Å")

    by_region = zeros(N_TRAJ, length(REGIONS))
    by_bin = zeros(N_TRAJ, length(XGRID) - 1)
    total = zeros(N_TRAJ)

    for k in 1:N_TRAJ
        x0 = sample_well(rng_ic, β, lo, hi)
        v0 = sqrt(kT / mass) * randn(rng_ic)
        sim = build(rng_method)
        sol = run_dynamics(sim, (0.0, T_FINAL), DynamicsVariables(sim, hcat(v0), hcat(x0));
                           dt = MD_DT, output = (OutputPosition, OutputVelocity))
        r = [p[1] for p in sol[:OutputPosition]]
        v = [p[1] for p in sol[:OutputVelocity]]
        E = [0.5 * mass * v[i]^2 + V(r[i]) for i in eachindex(r)]

        total[k] = E[1] - E[end]
        for i in 1:(length(E) - 1)
            xm = (r[i] + r[i+1]) / 2 * BOHR_IN_ANGSTROM
            dE = E[i] - E[i+1]
            j = findfirst(reg -> reg[2] ≤ xm < reg[3], REGIONS)
            isnothing(j) || (by_region[k, j] += dE)
            # Same attribution on the fine grid. Binning by position and then
            # cumulating keeps "energy lost while the molecule was inside x" well
            # defined even for a path that doubles back before it leaves.
            b = searchsortedlast(XGRID, xm)
            1 ≤ b ≤ size(by_bin, 2) && (by_bin[k, b] += dE)
        end
    end
    return total, by_region, by_bin
end

eV(x) = auconvert(u"eV", x).val

println("Where the energy goes: $(N_TRAJ) trajectories, $(Int(T_FINAL)) aut, T = $(Int(T_K)) K")
println("crossing at 2.0203 Å;  Λ(2.0203,0) = 1399,  Λ(2.05,0) = 6.75,  Λ(2.2,0) = 0.003\n")

mm_probe = PronyMemory(FIT_FILE; kT = kT_fit, Γ_eV = GAMMA_EV)

results = Dict{String,Any}()
for (label, build) in (
    ("GLE  (Prony, m = 43)", rng -> begin
        mm = PronyMemory(FIT_FILE; kT = kT_fit, Γ_eV = GAMMA_EV)
        sim = prony_simulation(atoms, pes, mm; temperature = T_K * u"K")
        sim.method.s .= sqrt(kT) .* randn(rng, auxiliary_dimension(mm))
        sim
    end),
    ("MDEF (Markovian)    ", rng -> Simulation{MDEF}(atoms,
        CompositeFrictionModel(pes, MarkovianFriction(et, kT_fit)); temperature = T_K * u"K")))

    total, by_region, by_bin = profile(label, build)
    results[strip(label)] = (; total, by_region, by_bin)
    se(v) = std(v) / sqrt(length(v))
    @printf("%s   total ΔE = %.1f ± %.1f meV\n",
            label, 1e3 * eV(mean(total)), 1e3 * eV(se(total)))
    for (i, reg) in enumerate(REGIONS)
        col = @view by_region[:, i]
        @printf("    %-20s %+7.2f ± %.2f meV   %5.1f %%\n",
                reg[1], 1e3 * eV(mean(col)), 1e3 * eV(se(col)),
                100 * mean(col) / mean(total))
    end
    println()
end

# Paired comparison — the same x₀ and v₀ entered both ensembles, so differencing
# trajectory by trajectory removes the thermal spread that dominates each column.
let g = results["GLE  (Prony, m = 43)"], m = results["MDEF (Markovian)"]
    println("="^68)
    println("paired GLE - MDEF, same initial conditions")
    println("="^68)
    se(v) = std(v) / sqrt(length(v))
    for (i, reg) in enumerate(REGIONS)
        d = g.by_region[:, i] .- m.by_region[:, i]
        @printf("  %-20s %+7.2f ± %.2f meV   (%.1f σ)\n",
                reg[1], 1e3 * eV(mean(d)), 1e3 * eV(se(d)), abs(mean(d)) / se(d))
    end
    d = g.total .- m.total
    @printf("  %-20s %+7.2f ± %.2f meV   (%.1f σ)\n",
            "TOTAL", 1e3 * eV(mean(d)), 1e3 * eV(se(d)), abs(mean(d)) / se(d))
end

# ------------------------------------------------------------------------------------
# Save the fine profile. `scripts/plot/gle/plot_energy_loss.jl` is the reader; this is
# the producer, so what gets drawn is this artefact and not a re-derivation of it.
# ------------------------------------------------------------------------------------

centres = [(XGRID[i] + XGRID[i+1]) / 2 for i in 1:(length(XGRID) - 1)]
key(label) = replace(strip(label), r"[^A-Za-z0-9]+" => "_")

outfile = datadir("sims", "gle", savename("dissipation_profile",
    (; Gamma_eV = GAMMA_EV, T_K = Int(T_K), N = N_TRAJ,
       tfinal = Int(T_FINAL), dt = MD_DT), "h5"; sort = false))
mkpath(dirname(outfile))
h5open(outfile, "w") do f
    f["x_Ang"] = centres
    f["Lambda0"] = [mm_probe.model.Λ₀(x) for x in centres]
    for (label, r) in results
        g = create_group(f, key(label))
        # cumsum along x: the mean over trajectories of "lost so far by position x"
        cum = cumsum(r.by_bin; dims = 2)
        g["cumulative_mean"] = vec(mean(cum; dims = 1))
        g["cumulative_se"] = vec(std(cum; dims = 1)) ./ sqrt(size(cum, 1))
        g["per_bin_mean"] = vec(mean(r.by_bin; dims = 1))
        g["total_mean"] = mean(r.total)
        g["total_se"] = std(r.total) / sqrt(length(r.total))
    end
    # The paired difference, trajectory by trajectory, on the same grid.
    let gl = results["GLE  (Prony, m = 43)"], md = results["MDEF (Markovian)"]
        d = cumsum(gl.by_bin; dims = 2) .- cumsum(md.by_bin; dims = 2)
        f["paired_GLE_minus_MDEF/cumulative_mean"] = vec(mean(d; dims = 1))
        f["paired_GLE_minus_MDEF/cumulative_se"] = vec(std(d; dims = 1)) ./ sqrt(size(d, 1))
    end
    attrs(f)["Gamma_eV"] = GAMMA_EV
    attrs(f)["T_K"] = T_K
    attrs(f)["n_trajectories"] = N_TRAJ
    attrs(f)["t_final_aut"] = T_FINAL
    attrs(f)["dt_aut"] = MD_DT
    attrs(f)["x_crossing_Ang"] = 2.0203445
    attrs(f)["units"] = "energies in Hartree, x in Å"
    attrs(f)["surface"] = "lower adiabat of the bare 2×2 ErpenbeckThoss, no bath"
    attrs(f)["git"] = gitdescribe(projectdir())
end
println("saved: ", outfile)
