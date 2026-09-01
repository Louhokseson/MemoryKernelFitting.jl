"""
    MemoryKernelFitting.SpectralData

Part 0 of the package: getting `Λ(x, ω)` off disk and into atomic units, and
constructing the fixed weights the objective needs.

Reads the `Lambda_spectrum_*.h5` files under `data/sims/friction/ErpenbeckThoss/`.
One file per hybridisation strength `Γ`, and each file is a **separate model, hence a
separate fit** — never pool them.

See `docs/design/0001-units-and-data-conventions.md` for the conversion and the
graded-grid trap.
"""
module SpectralData

using HDF5

using ..Units: eV_to_hartree, hartree_to_eV, boltzmann_energy,
               BOHR_IN_ANGSTROM, ELECTRON_MASSES_PER_U

export LambdaSpectrum, load_lambda_spectrum, trapezoid_weights, subsample_omega
export ErpenbeckThossModel, level_position, hybridisation_width, coupling
export dlevel_position, dhybridisation_width, diabatic_potentials
export fermi_crossing, morse_zero_point

include("erpenbeck_thoss.jl")
# After the loader: `ErpenbeckThossModel(::LambdaSpectrum)` needs the type.
include("erpenbeck_thoss_model.jl")

end # module
