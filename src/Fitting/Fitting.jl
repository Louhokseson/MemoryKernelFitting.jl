"""
    MemoryKernelFitting.Fitting

Part 3 of the package: **solving** the objective of
[`MemoryKernelFitting.Objective`](@ref).

Stage 1 of `docs/design/0002-amplitude-representation.md` — the convex one. With the
pole ladder fixed (`docs/design/0003-poles-and-the-mirror.md`), the model is linear in
the amplitudes and the problem is a non-negative least squares: convex, globally
solvable by Lawson–Hanson, with no initialisation risk and no local minima. There is
no iterative outer optimiser on this path at all, and that is the point of fixing the
ladder.

Without the roughness penalty the configurations **decouple**, so stage 1 is `N_x`
independent NNLS problems of size `[|ℬ|, 1 + K_R + K_C]` rather than one large one.
`Prony_fit_optimisation_problem.md` §7 step 3 defers the choice of `μ` until after a
first fit exists, so this is the right thing to build first — and the resulting
amplitude curve is exactly what `docs/design/0002` argues you cannot choose the stage-2
basis without having seen.

What is deliberately **not** here yet:

  - the `μ`-coupled block-tridiagonal solve (§3), which couples all `N_x` columns into
    one system of `[N_x|ℬ| + (N_x-1)(1+K_R+K_C), N_x(1+K_R+K_C)]` — 150337 × 7874 on
    the `Γ = 0.1` grid, too large to hold densely;
  - the `μ`-coupled solve above is the only genuinely missing piece of stage 1.

The outer **VARPRO** pole search of §5 now exists, in `varpro.jl`, over the regime-(B)
[`TrackingPoles`](@ref) model of §6. It is *not* on the production path — §9.7 settles
regime (B) negatively for this data and `TrackingPoles` carries that verdict in its own
docstring — but §7 step 2 and the note's open items both ask for measurements only this
path can make, chiefly how much relative error is *achievable* at a given `K_C`. The
fixed-ladder numbers have no upper bound to be read against until it has been run.
"""
module Fitting

using NonNegLeastSquares: nonneg_lsq
using Optim: optimize, LBFGS, NelderMead, Optim
using LinearAlgebra: mul!, dot

using ..SpectralData: LambdaSpectrum, subsample_omega
using ..PronySeriesFriction: PoleSet, GridAmplitudes, PolynomialAmplitudes,
                             ChebyshevAmplitudes, BasisAmplitudes, spectral_density,
                             amplitudes, mode_counts, n_configurations, parameters,
                             parameters!, warp_coordinate,
                             TrackingPoles, poleset, lorentzian, mirrored_lorentzian,
                             TiedLaw, expand
using ..Objective: design_matrix, residual_scales, markovian_anchor_row,
                   _amplitude_columns

using ..MemoryKernelFitting: AmplitudeModel

export fit_amplitudes, fit_column, relative_errors
export basis_matrix, shape_envelope, fit_basis_report
export fit_tracking, tracking_solve, tracking_ladders, tracking_errors, fit_tracking_basis

include("nnls.jl")
include("basis_fit.jl")
include("varpro.jl")

end # module
