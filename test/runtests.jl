using MemoryKernelFitting
using Test

@testset "MemoryKernelFitting.jl" begin
    include("test_units.jl")
    include("test_spectral_data.jl")
    include("test_erpenbeck_thoss_model.jl")
    include("test_prony_model.jl")
    include("test_objective.jl")
    include("test_fitting.jl")
    include("test_basis_fit.jl")
    include("test_tracking_poles.jl")
end
