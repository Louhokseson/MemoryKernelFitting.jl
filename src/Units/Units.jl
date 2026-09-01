"""
    MemoryKernelFitting.Units

Unit constants and conversions.

**The package works in atomic units throughout.** The stored data does not: the
`Lambda_spectrum_*.h5` files carry `Lambda_au` in atomic units but `omega_eV`,
`h_eV` and `Delta_eV` in electronvolts. Conversion happens once, at load time, in
[`MemoryKernelFitting.SpectralData`](@ref); nothing downstream of the loader should
see an eV.

With `ħ = 1` an angular frequency in atomic units is numerically equal to an energy
in Hartree, so [`eV_to_hartree`](@ref) converts `ω`, `h` and `Δ` alike.

Consequences for the fit model, worth stating because they are easy to get wrong:

  - `λ_k`, `α_j`, `β_j` are in Hartree (equivalently, inverse atomic time units).
  - `τ` in `K(τ)` is in atomic time units, `1 aut = 0.0241888 fs`.
  - `a₀` has the units of `Λ` (atomic friction, `mₑ/aut`); `a_k` and `A_j` carry an
    extra inverse time, since they multiply a Lorentzian of dimension `1/ω`.
"""
module Units

export HARTREE_IN_EV, BOLTZMANN_HARTREE_PER_KELVIN, ATOMIC_TIME_IN_FS
export BOHR_IN_ANGSTROM, ELECTRON_MASSES_PER_U
export eV_to_hartree, hartree_to_eV, boltzmann_energy, atomic_time_to_fs

"Hartree energy in electronvolts (CODATA 2018)."
const HARTREE_IN_EV = 27.211386245988

"Boltzmann constant in Hartree per Kelvin (CODATA 2018)."
const BOLTZMANN_HARTREE_PER_KELVIN = 3.166811563e-6

"One atomic unit of time in femtoseconds."
const ATOMIC_TIME_IN_FS = 0.024188843265857

"""
Bohr radius in Ångström (CODATA 2018).

Needed only by [`MemoryKernelFitting.SpectralData.ErpenbeckThossModel`](@ref), whose
Morse range parameter `a` is quoted in `Å⁻¹` while its zero-point energy must be formed
in atomic units. Everywhere else `x` stays in Å and is never combined with an energy —
see `docs/design/0001-units-and-data-conventions.md`.
"""
const BOHR_IN_ANGSTROM = 0.529177210903

"""
Mass of one atomic mass unit in electron masses, `m_u / m_e` (CODATA 2018).

Same single caller as [`BOHR_IN_ANGSTROM`](@ref): the Morse zero-point `ω₀ = √(2Dₑa²/m)`
is the one place in the package where a mass appears at all.
"""
const ELECTRON_MASSES_PER_U = 1822.888486209

"""
    eV_to_hartree(E) → E / 27.211386245988

Convert an energy in eV to Hartree. Because `ħ = 1` in atomic units, this is also
the conversion for an angular frequency: `ω_au = ω_eV / HARTREE_IN_EV`.
"""
eV_to_hartree(E) = E / HARTREE_IN_EV

"""
    hartree_to_eV(E) → E * 27.211386245988

Inverse of [`eV_to_hartree`](@ref). Used for reporting and for comparing against the
figures in `docs/PronySeries/`, which are quoted in eV.
"""
hartree_to_eV(E) = E * HARTREE_IN_EV

"""
    boltzmann_energy(T_K) → k_B T in Hartree

Thermal energy at temperature `T_K` in Kelvin. At the `T_K = 300` of every file in
`data/sims/friction/ErpenbeckThoss/` this is `9.500e-4` Ha `= 0.02585` eV, which sets
the floor on the resonance width at the Fermi crossing (see
`Prony_fit_optimisation_problem.md` §5).
"""
boltzmann_energy(T_K) = BOLTZMANN_HARTREE_PER_KELVIN * T_K

"""
    atomic_time_to_fs(τ) → τ * 0.024188843265857

Convert a lag `τ` in atomic time units to femtoseconds. Only needed when comparing a
fitted memory time against the timestep discussion of
`Prony_fit_optimisation_problem.md` Appendix A.4, which is written in fs.
"""
atomic_time_to_fs(τ) = ATOMIC_TIME_IN_FS * τ

end # module
