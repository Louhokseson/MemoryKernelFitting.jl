# Does the §3 configuration weighting change which ladder you would choose?
#
#   julia --project=. scripts/fitting/reweight_gamma01.jl
#
# `configuration_weights(s; θ)` blends the plain x measure (θ=0, the trapezoid weights
# used until now) with the warped measure of §6.2 (θ=1). Because stage 1 decouples the
# columns at μ = 0, θ cannot change a fit at fixed ladder — only the J that ranks one
# ladder against another. This script measures that ranking under both weightings.

using DrWatson; @quickactivate "MemoryKernelFitting"
using MemoryKernelFitting, Printf

median_(v) = (s = sort(vec(collect(v))); n = length(s);
              isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1])/2)

const BETA_LO = (1.8e-3, 0.345); const BETA_HI = (0.441, 2.20496)
const LAM_LO  = 1.84e-4;         const LAM_HI  = 1.84e-3
const EPS = 1e-3; const ANCHOR = 1e3

s   = load_lambda_spectrum(datadir("sims","friction","ErpenbeckThoss",
        "Lambda_spectrum_Gamma_eV=0.1_Nx=257_T_K=300.0_domega_eV=0.001.h5"))
idx = subsample_omega(s.ω); ω = s.ω[idx]; Λ = s.Λ[idx,:]; σ = residual_scales(Λ; ε=EPS)
ωeV = hartree_to_eV.(ω); low = ωeV .<= 1.0; tail = ωeV .> 10.0
x0 = s.x_resonance; i0 = argmin(abs.(s.x .- x0))
near = abs.(s.x .- x0) .< 0.15; peak = 1.0 .< s.x .< 1.8; far = s.x .>= 3.0
v0 = configuration_weights(s; θ=0.0); vh = configuration_weights(s; θ=0.5)

# (K_C_lo, K_C_hi, K_R). First block: the compositions already on record in
# fit_gamma01.jl. Second block: reallocations toward the resonance band, which is what
# a crossing-weighted objective should reward if the diagnosis is right.
compo = [(16,4,3), (16,3,3), (15,4,3), (14,4,3), (13,3,3), (16,4,12), (20,0,12), (20,0,0),
         (18,2,3), (20,0,3), (18,4,3), (16,4,5), (16,4,2), (22,0,3), (20,2,3)]

zm(K_R,K_C) = GridAmplitudes(length(s.x), K_R, K_C)
rows = []
for (KC,KH,KR) in compo
    segs = KH == 0 ? [(BETA_LO..., KC)] : [(BETA_LO..., KC), (BETA_HI..., KH)]
    lad  = split_ladder(segs, KR, LAM_LO, KR > 3 ? BETA_LO[2] : LAM_HI)
    fit  = fit_amplitudes(lad, ω, Λ; ε=EPS, anchor_weight=ANCHOR)
    e    = relative_errors(lad, fit, ω, Λ); ae = abs.(e)
    z    = zm(KR, KC+KH)
    push!(rows, (KC=KC, KH=KH, KR=KR, m=auxiliary_dimension(lad),
        J0 = objective_value(lad,fit,ω,Λ,v0,σ)/objective_value(lad,z,ω,Λ,v0,σ),
        Jh = objective_value(lad,fit,ω,Λ,vh,σ)/objective_value(lad,z,ω,Λ,vh,σ),
        cross = median_(ae[low, i0]), nearm = median_(ae[low, near]),
        peakm = median_(ae[:, peak]),  tailm = median_(e[tail, far])))
end

println("ladder composition sweep, Γ = 0.1 eV, ε = 1e-3, anchor = 1e3, μ = 0\n")
println("  K_C_lo K_C_hi  K_R    m    J/J₀ θ=0   J/J₀ θ=.5   at x₀   |x-x₀|<.15   1<x<1.8   tail x≥3")
for r in rows
    @printf("  %6d %6d %4d %4d  %10.4f %11.4f %8.2f%% %10.2f%% %9.2f%% %9.1f%%\n",
        r.KC, r.KH, r.KR, r.m, r.J0, r.Jh, 100r.cross, 100r.nearm, 100r.peakm, 100r.tailm)
end

rank(f) = sortperm([f(r) for r in rows])
r0, rh = rank(r->r.J0), rank(r->r.Jh)
println("\n  best 6 by J (θ=0, current):    ", join(["($(rows[i].KC),$(rows[i].KH),$(rows[i].KR)) m=$(rows[i].m)" for i in r0[1:6]], "  "))
println("  best 6 by J (θ=0.5, warped):   ", join(["($(rows[i].KC),$(rows[i].KH),$(rows[i].KR)) m=$(rows[i].m)" for i in rh[1:6]], "  "))
