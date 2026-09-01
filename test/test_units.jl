@testset "Units" begin
    hartree_in_eV = 27.211386245988   # CODATA 2018

    @testset "eV ↔ Hartree" begin
        @test eV_to_hartree(hartree_in_eV) ≈ 1.0
        @test hartree_to_eV(1.0) == hartree_in_eV
        @test eV_to_hartree(1.0) ≈ 3.6749322175655e-2 rtol = 1e-12
        @test eV_to_hartree(0.0) == 0.0
        for E in (1e-6, 0.05, 9.4, 60.0)
            @test hartree_to_eV(eV_to_hartree(E)) ≈ E
            @test eV_to_hartree(hartree_to_eV(E)) ≈ E
        end
        # Conversion is linear, so it applies to ω as well as to h and Δ (ħ = 1).
        @test eV_to_hartree(2.0) ≈ 2 * eV_to_hartree(1.0)

        # The identifiability box of the optimisation note §4, converted in design
        # doc 0001: [5 dω, ω_max] = [1.8e-4, 2.205] Ha, useful β range → [1.8e-3, 0.345].
        # The doc rounds the two 1.8s to two significant figures, hence the loose rtol.
        @test eV_to_hartree(60.0) ≈ 2.205 rtol = 1e-3
        @test eV_to_hartree(5e-3) ≈ 1.8e-4 rtol = 3e-2
        @test eV_to_hartree(0.05) ≈ 1.8e-3 rtol = 3e-2
        @test eV_to_hartree(9.4) ≈ 0.345 rtol = 1e-2
    end

    @testset "boltzmann_energy" begin
        # Every file in data/sims/friction/ErpenbeckThoss/ has T_K = 300.
        @test boltzmann_energy(300.0) ≈ 9.500e-4 rtol = 1e-3
        @test hartree_to_eV(boltzmann_energy(300.0)) ≈ 0.02585 rtol = 1e-3
        @test boltzmann_energy(0.0) == 0.0
        @test boltzmann_energy(600.0) ≈ 2 * boltzmann_energy(300.0)
    end

    @testset "atomic_time_to_fs" begin
        @test atomic_time_to_fs(1.0) ≈ 0.024188843265857
        @test atomic_time_to_fs(0.0) == 0.0
        # Appendix A.4's Δt ≲ 0.07 fs is ≲ 2.9 aut (design doc 0001).
        @test atomic_time_to_fs(2.9) ≈ 0.07 rtol = 1e-2
        # 1 fs is ~41.3 aut.
        @test atomic_time_to_fs(41.341373336493) ≈ 1.0 rtol = 1e-9
    end
end
