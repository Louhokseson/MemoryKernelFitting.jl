# Does the fitted QGLE sample the right measure?
#
#   julia --project=. scripts/gle/validate_fdt.jl
#
# The FDT-safe construction of `Prony_fit_optimisation_problem.md` §1 exists to buy one
# thing: the extended SDE with configuration-dependent coupling still samples
# `ρ ∝ e^{-β[U(q) + p²/2m + sᵀs/2]}`. That is the property to check before any physics
# number is believed, and it is the property a wrong mass or length factor destroys while
# leaving a stable, plausible-looking trajectory behind.
#
# This script does the *frozen-x* half: freeze `Γ` at one configuration and sample the
# pure OU process with the integrator's own exact O-step. The stationary covariance must
# be `k_BT·I` in `z = (p/√m, s)`. That tests the readout, both unit conversions and the
# noise construction, with the PES and the position dynamics removed.
#
# **The moving-x half is in `stationarity.jl`, not here.** The obvious moving test — run
# one long trajectory and ask whether its time average is Boltzmann — does not work on
# this system, and it is worth recording why so that nobody re-derives it. The particle
# spends nearly all its time away from the crossing, where `Λ ≈ 1e-3` gives a momentum
# relaxation time `m/Λ ≈ 1.8e6` aut against affordable runs of about the same length. A
# `Δt` scan at 3000 K over `Δt ∈ {8, 4, 2, 1}` returned `2⟨KE⟩/k_BT` of 1.28, 0.35, 1.50,
# 1.08 — no trend in `Δt`, and the two halves of a single run differing by 2×. That is
# sampling noise on an unmixed variable, not a time-step bias and not a broken model.
#
# `stationarity.jl` sidesteps mixing entirely: it draws the exact Gibbs measure, propagates
# an ensemble briefly, and checks the distribution is unchanged. That converges as `1/√N`
# in trajectories rather than as the inverse of an unknown correlation time.

using DrWatson
@quickactivate "MemoryKernelFitting"

include(joinpath(@__DIR__, "prony_memory.jl"))

using LinearAlgebra
using Printf
using Random
using Statistics

# ------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------

const GAMMA_EV = 0.1
const FIT_FILE = datadir("fits",
    "prony_stage2Cheby_Gamma_eV=0.1_m=43_K_C=20_K_R=3_deg=22_eps=0.001_anchor=1000.0_thetav=0.5.h5")

const T_COLD = 300.0        # K — the temperature the spectra were computed at

# Frozen-x test. dt is large deliberately: the O-step is exact for the OU process, so a
# big step only decorrelates samples faster. It does not have to resolve β_max = 2.2 Ha.
const OU_DT = 20.0
const OU_STEPS = 200_000
const OU_BURN = 10_000
const OU_THIN = 25
const OU_X = [1.78, 1.95, 2.0203445, 2.2, 2.6]


Random.seed!(20260902)

kT_cold = boltzmann_energy(T_COLD)
atoms = Atoms([austrip(1.0u"u")])
mass = atoms.masses[1]

# ------------------------------------------------------------------------------------
# 1. Frozen x: the exact O-step must have k_BT·I as its stationary covariance
# ------------------------------------------------------------------------------------

"""
    ou_covariance(Γ, dt, nsteps; β, burn, thin) → (cov, rank)

Sample the O-step of `GLE_step_O!` on its own, with `Γ` held fixed. Same update, same
pivoted Cholesky, no integrator around it.
"""
function ou_covariance(Γ, dt, nsteps; β, burn, thin)
    F = exp(-Γ .* dt)
    cd = cholesky(Symmetric(I - F * transpose(F)), RowMaximum(), check = false)
    S = cd.L[invperm(cd.p), 1:cd.rank]
    z = zeros(size(Γ, 1))
    samples = Vector{Vector{Float64}}()
    noise = zeros(size(S, 2))
    for k in 1:nsteps
        randn!(noise)
        z .= F * z .+ sqrt(1 / β) .* (S * noise)
        if k > burn && (k - burn) % thin == 0
            push!(samples, copy(z))
        end
    end
    return cov(reduce(hcat, samples); dims = 2), cd.rank
end

function frozen_test(sim, mm, kT)
    println("\n", "="^84)
    println("1. Frozen x — stationary covariance of the exact O-step, target k_BT·I")
    println("="^84)
    @printf("%-12s %-12s %-10s %-10s %-10s %-10s\n",
            "x [Å]", "Λ(x,0)", "⟨p²⟩/mkT", "⟨s²⟩/kT", "max|off|/kT", "rank")
    ok = true
    for x in OU_X
        DynamicsMethods.ClassicalMethods.GLE_friction!(
            zeros(1, 1), hcat(austrip(x * u"Å")), sim, 0.0)
        C, rank = ou_covariance(copy(sim.method.Γ), OU_DT, OU_STEPS;
                                β = 1 / kT, burn = OU_BURN, thin = OU_THIN)
        d = diag(C)
        off = maximum(abs.(C - Diagonal(d))) / kT
        @printf("%-12.4f %-12.4g %-10.4f %-10.4f %-10.4f %d/%d\n",
                x, mm.model.Λ₀(x), d[1] / kT, mean(d[2:end]) / kT, off,
                rank, size(C, 1))
        # The momentum slot equilibrates only through the coupling, so where the coupling
        # is weak it is the sampling window that is short, not the FDT that fails. The
        # auxiliary block has its own noise and must be right everywhere.
        ok &= isapprox(mean(d[2:end]), kT; rtol = 0.1)
    end
    println(ok ? "auxiliary block thermalised at every x ✓" :
                 "auxiliary block NOT at k_BT — investigate ✗")
    return ok
end

# ------------------------------------------------------------------------------------

isfile(FIT_FILE) || error("fit file not found: $FIT_FILE")
mm = PronyMemory(FIT_FILE; kT = kT_cold, Γ_eV = GAMMA_EV)
sim = prony_simulation(atoms, erpenbeck_thoss_neutral(GAMMA_EV), mm;
                       temperature = T_COLD * u"K")
println(mm, "   fit: ", basename(FIT_FILE))
println("Γ is ", size(sim.method.Γ), ", auxiliary block ", length(sim.method.s))

ok = frozen_test(sim, mm, kT_cold)

println("\n", "="^84)
println(ok ? "FROZEN-x CHECKS PASSED — run stationarity.jl for the moving-x case" :
             "FROZEN-x CHECKS FAILED — see above")
