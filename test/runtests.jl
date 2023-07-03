using ReverseAD
using Test

@testset "ReverseAD.jl" begin

include("inspect.jl")

include("register.jl")
include("trace.jl")
include("broadcast.jl")
include("back.jl")

include("grad_check.jl")

include("ops.jl")
include("test_bmm.jl")
include("diff_types.jl")

end
