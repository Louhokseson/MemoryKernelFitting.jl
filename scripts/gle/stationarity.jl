# Does the QGLE preserve the Gibbs measure when x moves?
#
#   julia --project=. scripts/gle/stationarity.jl
#
# `validate_fdt.jl` tried to answer this by running one long trajectory and asking whether
# its time average is Boltzmann. That test cannot work here: the time-step scan showed
# `2⟨KE⟩/k_BT` scattering between 0.35 and 1.50 with no dependence on `Δt`, and differing
# by 2× between the two halves of a single run. The energy simply does not mix on any
# affordable trajectory — the particle spends nearly all its time away from the crossing,
# where `Λ ≈ 1e-3` gives a relaxation time `m/Λ ≈ 1.8e6` aut, and the runs were `1.2e6`.
#
# So do not rely on mixing. Sample the *exact* stationary distribution directly,
#
#     x ~ e^{-βU(x)},   p ~ N(0, m k_BT),   s ~ N(0, k_BT I),
#
# propagate an ensemble for a short time, and check the distribution comes back unchanged.
# If `ρ_Gibbs` is stationary for the SDE this holds however slowly the process mixes, and
# the error falls as `1/√N` in the number of trajectories rather than as the inverse of an
# unknown correlation time. This is the property `Prony_fit_optimisation_problem.md` §1 is
# built to deliver, stated as something measurable in minutes.
#
# Temperature. 1500 K, not 300 K. `U` at the Fermi crossing is `0.23` eV, which is `8.9
# k_BT` at 300 K — the crossing is never visited and the configuration dependence goes
# untested — but only `1.8 k_BT` at 1500 K, so the ensemble straddles it and `Λ` varies
# over decades within the sampled region. The well is still bound: `Dₑ/k_BT = 27`, so
# `e^{-βU}` is normalisable and the truncation below is harmless. (The Morse plateau makes
# `e^{-βU}` non-normalisable on the half line for any `T`; at 3000 K the neglected weight
# stops being negligible, which is a second reason the earlier hot runs were a bad test.)
#
# The fit itself stays a 300 K object throughout — `Λ(x,ω)` was computed at 300 K and the
# warp `t(x) = arctan(h/(Δ + k_BT))` is part of the fit. Only the thermostat moves. What is
# asserted is that the SDE preserves `e^{-βU}` for the kernel it was handed.

using DrWatson
@quickactivate "MemoryKernelFitting"

include(joinpath(@__DIR__, "prony_memory.jl"))

using LinearAlgebra
using Printf
using Random
using Statistics

const GAMMA_EV = 0.1
const FIT_FILE = datadir("fits",
    "prony_stage2Cheby_Gamma_eV=0.1_m=43_K_C=20_K_R=3_deg=22_eps=0.001_anchor=1000.0_thetav=0.5.h5")

const T_K = 1500.0
const N_TRAJ = 1000
const T_PROP = 5000.0        # aut ≈ 9 Morse periods
const MD_DT = 2.0
const X_BOX = (1.15, 3.0)    # Å — the box the initial ensemble is drawn in
const SEED = 20260902

isfile(FIT_FILE) || error("fit file not found: $FIT_FILE")

kT_fit = boltzmann_energy(300.0)
kT = boltzmann_energy(T_K)
atoms = Atoms([austrip(1.0u"u")])
mass = atoms.masses[1]
pes = erpenbeck_thoss_neutral(GAMMA_EV)

U(x_bohr) = NQCModels.potential(pes, hcat(x_bohr))

"""
    OutputAuxiliaryMeanSquare(sol, i) → ⟨s²⟩ of the final state

The auxiliary variables live on `sim.method.s`, not in the state vector `u`, so no
built-in output sees them — and `run_dynamics` integrates a *copy* of the simulation, so
reading `sim.method.s` afterwards returns the untouched original. An output function is
the one place the integrated copy is reachable, as `sol.prob.p`.
"""
OutputAuxiliaryMeanSquare(sol, i) =
    sum(abs2, sol.prob.p.method.s) / length(sol.prob.p.method.s)

"""
    sample_position(rng, β, box) → x in bohr

Draw from `e^{-βU}` on `box` by rejection against a uniform proposal. `U` is bounded
below by 0 on the Morse well, so `exp(-βU)` is its own acceptance probability.
"""
function sample_position(rng, β, lo, hi)
    while true
        x = lo + (hi - lo) * rand(rng)
        rand(rng) < exp(-β * U(x)) && return x
    end
end

# ------------------------------------------------------------------------------------

rng = MersenneTwister(SEED)
Random.seed!(SEED)
lo, hi = austrip(X_BOX[1] * u"Å"), austrip(X_BOX[2] * u"Å")
β = 1 / kT

mm = PronyMemory(FIT_FILE; kT = kT_fit, Γ_eV = GAMMA_EV)
m_aux = auxiliary_dimension(mm)

println("Stationarity of the Gibbs measure under the fitted QGLE")
println("  ", mm)
@printf("  T = %.0f K,  %d trajectories × %.0f aut at dt = %.1f  (%d steps total)\n",
        T_K, N_TRAJ, T_PROP, MD_DT, N_TRAJ * round(Int, T_PROP / MD_DT))
@printf("  initial ensemble drawn from e^{-βU} on x ∈ [%.2f, %.2f] Å\n\n",
        X_BOX...)

x0 = zeros(N_TRAJ); v0 = zeros(N_TRAJ)
x1 = zeros(N_TRAJ); v1 = zeros(N_TRAJ)
s1 = zeros(N_TRAJ)

timing = @timed for k in 1:N_TRAJ
    sim = prony_simulation(atoms, pes, mm; temperature = T_K * u"K")
    x0[k] = sample_position(rng, β, lo, hi)
    v0[k] = sqrt(kT / mass) * randn(rng)
    # `s` is not part of the state vector, so it is seeded on the method itself. The
    # stationary law is N(0, k_BT·Q) with Q = I — see the readout's Σ̃₂ = √(2α) I₂.
    sim.method.s .= sqrt(kT) .* randn(rng, m_aux)

    u0 = DynamicsVariables(sim, hcat(v0[k]), hcat(x0[k]))
    sol = run_dynamics(sim, (0.0, T_PROP), u0; dt = MD_DT,
                       output = (OutputPosition, OutputVelocity, OutputAuxiliaryMeanSquare))
    x1[k] = sol[:OutputPosition][end][1]
    v1[k] = sol[:OutputVelocity][end][1]
    s1[k] = sol[:OutputAuxiliaryMeanSquare]
end
println("  wall: ", round(timing.time, digits = 1), " s\n")

# ------------------------------------------------------------------------------------
# Comparison. Standard errors: Var(v²) = 2⟨v²⟩² for a Gaussian, so the relative standard
# error on ⟨v²⟩ is √(2/N) — 4.5 % at N = 1000. Nothing here is meaningful below that.
# ------------------------------------------------------------------------------------

se_var = sqrt(2 / N_TRAJ)
ratio(v) = mass * mean(v .^ 2) / kT

println("="^76)
@printf("%-26s %-14s %-14s %s\n", "", "t = 0", "t = $(Int(T_PROP))", "target")
println("="^76)
@printf("%-26s %-14.4f %-14.4f %s\n", "2⟨KE⟩/kT", ratio(v0), ratio(v1), "1")
@printf("%-26s %-14.4f %-14.4f %s\n", "⟨x⟩ [Å]",
        mean(x0) * BOHR_IN_ANGSTROM, mean(x1) * BOHR_IN_ANGSTROM, "unchanged")
@printf("%-26s %-14.4f %-14.4f %s\n", "std(x) [Å]",
        std(x0) * BOHR_IN_ANGSTROM, std(x1) * BOHR_IN_ANGSTROM, "unchanged")
@printf("%-26s %-14s %-14.4f %s\n", "⟨s²⟩/kT", "1 (by construction)",
        mean(s1) / kT, "1")
@printf("%-26s %-14.4f %-14.4f %s\n", "⟨U⟩/kT",
        mean(U, x0) / kT, mean(U, x1) / kT, "unchanged")
println("="^76)
@printf("relative standard error on the ⟨v²⟩ entries: %.1f %%\n", 100 * se_var)

# How much of the crossing did the ensemble actually see? Without this the numbers above
# could all be right and still say nothing about the configuration dependence.
xÅ = x1 .* BOHR_IN_ANGSTROM
Λ = mm.model.Λ₀.(xÅ)
@printf("\nensemble at t = %d:  x ∈ [%.3f, %.3f] Å,  Λ(x,0) ∈ [%.3g, %.3g]  (spans %.0f×)\n",
        Int(T_PROP), minimum(xÅ), maximum(xÅ), minimum(Λ), maximum(Λ),
        maximum(Λ) / minimum(Λ))
@printf("fraction with x > 2.0203 Å (past the crossing): %.3f\n", count(>(2.0203), xÅ) / N_TRAJ)

drift_KE = abs(ratio(v1) - ratio(v0))
drift_x = abs(mean(x1) - mean(x0)) / std(x0)
pass = drift_KE < 3 * se_var && drift_x < 0.1 && isapprox(mean(s1) / kT, 1; rtol = 0.1)
println("\n", pass ? "Gibbs measure preserved ✓" :
                     "distribution moved — investigate ✗")
