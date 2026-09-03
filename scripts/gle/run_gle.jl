# Scattering off the ErpenbeckThoss ground-state PES: how much energy do the electrons take?
#
#   julia --project=. scripts/gle/run_gle.jl
#
# A molecule is launched at `x = 5` Å with `2` eV of translational energy directed at the
# surface. It accelerates in, crosses the Fermi resonance, turns around on the repulsive
# wall near `1.4` Å, crosses the resonance again on the way out, and leaves. The observable
# is the translational energy it leaves with — which is what a molecular-beam experiment
# measures, and whose deficit is the energy handed to the electron-hole pairs.
#
# The surface is the lower adiabat of the bare 2×2 ErpenbeckThoss Hamiltonian, the smallest
# eigenvalue of `Hermitian([V₁₁ V₁₂; V₁₂ V₂₂])`, with no bath discretisation:
#
#     PES(5 Å) = -1.5089 eV     PES(∞)   = -1.5002 eV
#     2.0203 Å                  the Fermi crossing, where Λ(x,0) = 1399 peaks
#     ~1.4 Å                    turning point for 3 eV incident (E_tot = 1.4911 eV)
#
# **Why this is the experiment memory should show up in.** Two things follow from the 3 eV:
#
#   - The resonance is crossed *twice*, and the two crossings are separated by only a few
#     hundred aut — well inside the kernel's memory. The fitted kernel at the crossing has
#     a first moment `∫τK dτ / ∫K dτ = 1350` aut, so the auxiliary variables excited on the
#     way in are still excited on the way out. The outbound crossing therefore happens into
#     a bath the inbound crossing has already disturbed, which a Markovian model cannot
#     represent at all.
#   - Each crossing is impulsive. At `0.0087` bohr/aut the molecule clears a 0.05 Å-wide
#     peak in `10.8` aut against 1350 aut of memory — a ratio of 125, against ~6-60 for a
#     thermally desorbing molecule.
#
# That is the difference from a thermal-desorption setup, where the resonance is crossed
# once and slowly: there the totals agreed to `+0.3 ± 2.7` meV and memory only moved *where*
# the energy was taken (`dissipation_profile.jl`). Here the two crossings are coupled
# through the auxiliary variables, so the total itself is in play.
#
# Three simulations, identical but for the friction:
#
#   GLE        the fitted Prony memory kernel K(x,τ), m = 43 auxiliary variables
#   MDEF       the Markovian limit Λ(x,0) of the same spectral density
#   Classical  no friction — the deterministic baseline, and a check that the
#              integrator conserves energy on this surface
#
# Initial conditions are monoenergetic and identical across methods and trajectories: the
# beam is not thermally broadened, so every trajectory starts at the same `x₀` and `v₀` and
# the ensemble spread is *entirely* the noise. That makes the comparison as tight as it can
# be — the classical run has no noise and so no spread at all, which is why its standard
# error is exactly zero and not a bug.

using DrWatson
@quickactivate "MemoryKernelFitting"

using Distributed

# ------------------------------------------------------------------------------------
# Workers
#
# The trajectories are independent, so this parallelises perfectly. Two ways in, in
# the order the script prefers them:
#
#   sbatch / srun   julia --project=. -p $SLURM_CPUS_PER_TASK scripts/gle/run_gle.jl
#   one node        GLE_WORKERS=8 julia --project=. scripts/gle/run_gle.jl
#
# Workers, not threads. `-t` buys nothing here and `EnsembleThreads` is actively unsafe
# on this path: `GLE_BAOAB` pushes to a module-level global every step, so concurrent
# trajectories corrupt it. See `ensemble_algorithm` in `prony_memory.jl` for the detail.
#
# Workers are only spawned here if the session has none and one of the environment
# variables asks for them, so `-p` on the command line is respected untouched. Every
# worker needs the *types* — `PronyMemory`, `GroundAdiabat`, the `GLE_friction!` method,
# `SeededIC` — or the simulation cannot be deserialised on the far side, which is what
# the `@everywhere include` below is for. The fit file is read once, on this process.
# ------------------------------------------------------------------------------------

const REQUESTED_WORKERS = parse(Int, get(ENV, "GLE_WORKERS",
                                         get(ENV, "SLURM_CPUS_PER_TASK", "0")))
if nprocs() == 1 && REQUESTED_WORKERS > 1
    addprocs(REQUESTED_WORKERS; exeflags = "--project=$(dirname(Base.active_project()))")
end

@everywhere include($(joinpath(@__DIR__, "prony_memory.jl")))

using HDF5
using Printf
using Random
using Statistics

# ------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------

# Γ and the incident energy come from the command line when this is one point of a
# sweep, and fall back to the defaults when it is run by hand:
#
#   julia --project=. scripts/gle/run_gle.jl --gamma 0.05 --energy 2.5
#
# `--force` re-runs a combination whose artefact already exists; without it the run is
# skipped, which is what makes the whole sweep re-submittable.
#
# Same hand-rolled loop as `run_Markovian_CPA_et.jl` — an unrecognised argument is
# skipped rather than fatal, so extra flags from a batch script do no harm.
let _gamma = 0.1, _energy = 2.0, _force = false, i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--gamma" && i < length(ARGS)
            _gamma = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--energy" && i < length(ARGS)
            _energy = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--force"
            _force = true; i += 1
        else
            i += 1
        end
    end
    global const GAMMA_EV = _gamma
    global const E_INC_EV = _energy
    global const FORCE = _force
end

const T_K = 300.0            # surface temperature: sets the noise, not the beam
const X_START = 5.0          # Å — launch point, where Λ(x,0) ≈ 1e-11 and the coupling is dead
const N_TRAJ = 400
const T_FINAL = 4_000.0      # aut; the round trip 5 → 1.4 → 5 Å takes ~1400 aut
const MD_DT = 0.5            # aut. The peak transit is 10.8 aut at these speeds, so the
                             # step has to resolve it: dt = 2 would cross it in 5 steps.
const SEED = 20260902

"""
    find_fit(Γ_eV) → path or nothing

The stage-2 Chebyshev fit for this Γ, located by Γ rather than by a hardcoded filename.

The rest of the name — `m`, `K_C`, `K_R`, `deg` — records the *result* of fitting and is
not the same across Γ, so a sweep cannot spell it in advance. Matching on
`Gamma_eV=<Γ>_` is the same discovery `fit_stage2_gamma01.jl` does for its stage-1 input.
The trailing underscore matters: without it `Gamma_eV=0.1` also matches `0.125`.
"""
function find_fit(Γ_eV)
    dir = datadir("fits")
    isdir(dir) || return nothing
    hits = filter(readdir(dir)) do f
        startswith(f, "prony_stage2Cheby_") &&
            occursin("Gamma_eV=$(Γ_eV)_", f) && endswith(f, ".h5")
    end
    isempty(hits) && return nothing
    # More than one stage-2 fit for the same Γ — the saved `deg` has changed over time, so
    # the directory really does hold several. Newest wins, by mtime: the same rule
    # `fit_stage2_gamma01.jl` uses to pick its own stage-1 input. Sorting the names
    # instead would be lexicographic, and `deg=6` sorts after `deg=22`.
    return argmax(mtime, joinpath.(dir, hits))
end

const FIT_FILE = find_fit(GAMMA_EV)

# Skip, do not fail. In a sweep a missing fit means "this Γ is not ready yet", not that
# the job is broken — exiting 0 keeps `--dependency=afterok` and the array accounting
# clean, while the marker line is greppable across a few hundred task logs.
if FIT_FILE === nothing
    @printf("[skip] no stage-2 fit for Γ = %g eV in %s\n", GAMMA_EV, datadir("fits"))
    println("[skip] reason=missing-fit  gamma=", GAMMA_EV, "  energy=", E_INC_EV)
    exit(0)
end

kT_fit = boltzmann_energy(300.0)          # the fit is a 300 K object
kT = boltzmann_energy(T_K)
atoms = Atoms([austrip(1.0u"u")])
mass = atoms.masses[1]
et = ErpenbeckThossModel(; Γ_eV = GAMMA_EV)
pes = erpenbeck_thoss_ground(GAMMA_EV)
PES(x_bohr) = NQCModels.potential(pes, hcat(x_bohr))

const X0 = austrip(X_START * u"Å")
const E_INC = austrip(E_INC_EV * u"eV")
const V0 = -sqrt(2 * E_INC / mass)        # negative: incident, towards the surface
const E_TOTAL = 0.5 * mass * V0^2 + PES(X0)
const V_ASYMPTOTE = PES(austrip(40.0u"Å"))

# ------------------------------------------------------------------------------------
# One ensemble
# ------------------------------------------------------------------------------------

"""
    run_ensemble(label, build) → NamedTuple

`build()` returns `(sim, extra_outputs)`. The simulation is built **once**: SciMLBase
deep-copies the whole problem — `sim` included — for every trajectory, so there is no
shared mutable state to protect, and [`SeededIC`](@ref) redraws the auxiliary bath on
each copy from a stream keyed on the trajectory index.

Parallelism is whatever the session offers ([`ensemble_algorithm`](@ref)); the results
are identical in distribution under all three, and the `s(0)` seeds are identical
outright, so a run on `n` workers is comparable with a run on one.
"""
function run_ensemble(label, build)
    sim, outs = build()
    u0 = DynamicsVariables(sim, hcat(V0), hcat(X0))
    dist = SeededIC(u0, kT, SEED)

    timing = @timed run_dynamics(sim, (0.0, T_FINAL), dist;
                                 dt = MD_DT, trajectories = N_TRAJ,
                                 selection = 1:N_TRAJ,
                                 ensemble_algorithm = ENSEMBLE,
                                 output = (OutputPosition, OutputVelocity, outs...))
    sols = timing.value

    # Trajectory averages. The initial conditions are identical across trajectories, so
    # ⟨x(t)⟩ and ⟨E(t)⟩ are meaningful curves and not a smear over different starts.
    nsave = length(0:MD_DT:T_FINAL)
    xt = zeros(nsave); Et = zeros(nsave); xt2 = zeros(nsave); Et2 = zeros(nsave)

    E_final = zeros(N_TRAJ)
    x_turn = zeros(N_TRAJ)
    x_end = zeros(N_TRAJ)
    s_final = fill(NaN, N_TRAJ)
    stuck = falses(N_TRAJ)

    for k in 1:N_TRAJ
        sol = sols[k]
        r = [p[1] for p in sol[:OutputPosition]]
        v = [p[1] for p in sol[:OutputVelocity]]

        E = [0.5 * mass * v[i]^2 + PES(r[i]) for i in eachindex(r)]
        if length(r) == nsave
            xt .+= r; xt2 .+= r .^ 2
            Et .+= E; Et2 .+= E .^ 2
        end

        E_final[k] = E[end]
        x_turn[k] = minimum(r) * BOHR_IN_ANGSTROM
        x_end[k] = r[end] * BOHR_IN_ANGSTROM
        stuck[k] = E_final[k] ≤ V_ASYMPTOTE
        haskey(sol, :OutputAuxiliaryMeanSquare) &&
            (s_final[k] = sol[:OutputAuxiliaryMeanSquare])
    end

    loss = E_TOTAL .- E_final                        # energy handed to the electrons
    E_trans = E_final .- V_ASYMPTOTE                 # what the molecule leaves with
    se(v) = std(v) / sqrt(length(v))
    xt ./= N_TRAJ; Et ./= N_TRAJ
    x_sd = sqrt.(max.(xt2 ./ N_TRAJ .- xt .^ 2, 0.0))
    E_sd = sqrt.(max.(Et2 ./ N_TRAJ .- Et .^ 2, 0.0))

    return (; label, wall = timing.time, per_traj_loss = loss,
            t = collect(0:MD_DT:T_FINAL), xt, Et, x_sd, E_sd,
            loss = mean(loss), loss_se = se(loss), loss_sd = std(loss),
            E_trans = mean(E_trans), E_trans_se = se(E_trans),
            x_turn = mean(x_turn), x_turn_se = se(x_turn),
            x_end = mean(x_end), stuck = mean(stuck),
            s_final = all(isnan, s_final) ? NaN : mean(filter(!isnan, s_final)) / kT)
end

# ------------------------------------------------------------------------------------

mm_template = PronyMemory(FIT_FILE; kT = kT_fit, Γ_eV = GAMMA_EV)
m_aux = auxiliary_dimension(mm_template)
eV(x) = auconvert(u"eV", x).val

# The artefact path has to be known *before* the ensembles run, or the skip is useless —
# and it cannot be known before the fit is loaded, because `m` is a property of the fit.
# Hence: discover fit → load fit → name the artefact → decide whether to run.
outfile = datadir("sims", "gle", savename("scatter_ErpenbeckThoss_ground",
    (; Gamma_eV = GAMMA_EV, T_K = Int(T_K), Einc_eV = E_INC_EV, N = N_TRAJ,
       tfinal = Int(T_FINAL), dt = MD_DT, m = m_aux), "h5"; sort = false))

if isfile(outfile) && !FORCE
    println("[skip] reason=already-done  gamma=", GAMMA_EV, "  energy=", E_INC_EV)
    println("[skip] ", outfile)
    exit(0)
end

# Chosen once, so every method in the comparison runs the same way.
const ENSEMBLE = ensemble_algorithm()

println("Scattering off the ErpenbeckThoss ground-state PES (lower adiabat of the 2×2)")
println("  ", mm_template)
@printf("  %s   (%d worker%s, %d thread%s)\n",
        nameof(typeof(ENSEMBLE)), nworkers(), nworkers() == 1 ? "" : "s",
        Threads.nthreads(), Threads.nthreads() == 1 ? "" : "s")
@printf("  %.1f eV incident at %.1f Å, surface at %.0f K,  %d trajectories × %.0f aut, dt = %.2f\n",
        E_INC_EV, X_START, T_K, N_TRAJ, T_FINAL, MD_DT)
@printf("  PES(%.1f Å) = %+.4f eV,  PES(∞) = %+.4f eV,  E_total = %+.4f eV\n",
        X_START, eV(PES(X0)), eV(V_ASYMPTOTE), eV(E_TOTAL))
println("  crossing at 2.0203 Å: Λ = 1399, peak transit ≈ 10.8 aut, kernel memory ≈ 1350 aut\n")

results = [
    # The auxiliary variables start from their own stationary law, N(0, k_BT·I) — seeded
    # per trajectory by `SeededIC`. At 5 Å the coupling is ~1e-11, so they are simply
    # decoupled OU processes at the surface temperature; starting them at zero would be a
    # cold, unphysical bath, and starting them all from *one* draw would make that
    # transient common-mode instead of averaging it out.
    run_ensemble("GLE (Prony, m = $m_aux)", function ()
        mm = PronyMemory(FIT_FILE; kT = kT_fit, Γ_eV = GAMMA_EV)
        sim = prony_simulation(atoms, pes, mm; temperature = T_K * u"K")
        return sim, (OutputAuxiliaryMeanSquare,)
    end),

    run_ensemble("MDEF (Markovian Λ(x,0))", function ()
        model = CompositeFrictionModel(pes, MarkovianFriction(et, kT_fit))
        return Simulation{MDEF}(atoms, model; temperature = T_K * u"K"), ()
    end),

    run_ensemble("Classical (no friction)", function ()
        return Simulation{Classical}(atoms, pes; temperature = T_K * u"K"), ()
    end),
]

println("="^96)
@printf("%-24s %-10s %-24s %-22s %s\n", "", "stuck",
        "→ electrons [meV]", "E_trans [eV]", "turning point [Å]")
println("="^96)
for r in results
    @printf("%-24s %.3f      %+8.2f ± %.2f (σ %.1f)  %.4f ± %.4f      %.4f ± %.4f\n",
            r.label, r.stuck,
            1e3 * eV(r.loss), 1e3 * eV(r.loss_se), 1e3 * eV(r.loss_sd),
            eV(r.E_trans), eV(r.E_trans_se), r.x_turn, r.x_turn_se)
end
println("="^96)

let gl = results[1], md = results[2], cl = results[3]
    d = gl.per_traj_loss .- md.per_traj_loss
    sed = std(d) / sqrt(length(d))
    @printf("GLE - MDEF energy to electrons: %+.2f ± %.2f meV   (%.1f σ)\n",
            1e3 * eV(mean(d)), 1e3 * eV(sed), abs(mean(d)) / sed)
    @printf("as a fraction of the %.2g eV incident: GLE %.3f %%, MDEF %.3f %%\n",
            E_INC_EV, 100 * gl.loss / E_INC, 100 * md.loss / E_INC)
    @printf("classical energy drift (integrator check): %+.3f meV over %.0f aut\n",
            1e3 * eV(cl.loss), T_FINAL)
    isnan(gl.s_final) || @printf("GLE auxiliary variables ended thermal: ⟨s²⟩/kT = %.4f\n",
                                 gl.s_final)
end
@printf("\nwall: %s\n", join([@sprintf("%s %.0f s", r.label, r.wall) for r in results], ",  "))

# ------------------------------------------------------------------------------------

mkpath(dirname(outfile))
h5open(outfile, "w") do f
    for r in results
        g = create_group(f, replace(r.label, r"[^A-Za-z0-9]+" => "_"))
        for k in (:loss, :loss_se, :loss_sd, :E_trans, :E_trans_se,
                  :x_turn, :x_turn_se, :x_end, :stuck, :s_final, :wall)
            g[string(k)] = getfield(r, k)
        end
        g["per_traj_loss"] = r.per_traj_loss
        # Subsample the mean trajectory: 8001 points is more than any figure needs.
        keep = 1:20:length(r.t)
        g["t_aut"] = r.t[keep]
        g["x_mean_bohr"] = r.xt[keep]
        g["x_sd_bohr"] = r.x_sd[keep]
        g["E_mean_Ha"] = r.Et[keep]
        g["E_sd_Ha"] = r.E_sd[keep]
    end
    attrs(f)["Gamma_eV"] = GAMMA_EV
    attrs(f)["T_K"] = T_K
    attrs(f)["E_incident_eV"] = E_INC_EV
    attrs(f)["x_start_Ang"] = X_START
    attrs(f)["n_trajectories"] = N_TRAJ
    attrs(f)["t_final_aut"] = T_FINAL
    attrs(f)["dt_aut"] = MD_DT
    attrs(f)["seed"] = SEED
    attrs(f)["m_auxiliary"] = m_aux
    attrs(f)["surface"] = "lower adiabat of the bare 2×2 ErpenbeckThoss, no bath"
    attrs(f)["V_asymptote_Ha"] = V_ASYMPTOTE
    attrs(f)["E_total_Ha"] = E_TOTAL
    attrs(f)["fit_file"] = basename(FIT_FILE)
    attrs(f)["units"] = "energies in Hartree, x in Å, times in atomic time units"
    # The friction landscape the figure needs as context, on a grid of its own.
    xg = collect(range(1.3, 3.0; length = 400))
    f["landscape/x_Ang"] = xg
    f["landscape/Lambda0"] = [mm_template.model.Λ₀(x) for x in xg]
    f["landscape/V_Ha"] = [PES(austrip(x * u"Å")) for x in xg]
    attrs(f)["git"] = gitdescribe(projectdir())
end
println("saved: ", outfile)
