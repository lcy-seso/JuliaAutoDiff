module ReverseAD  # ReverseAD is used as a temporary package name.

using DataStructures
using MacroTools
using MacroTools: @forward

include("register.jl")
include("tape.jl")
include("inspect.jl")

include("ops/wrappers.jl")
include("ops/diffrules.jl")
include("ops/real.jl")
include("ops/array.jl")
include("ops/batched_gemm.jl")

include("broadcast.jl")

include("tracer.jl")

include("back.jl")
include("grad.jl")

include("grad_check.jl")

export @vjp, @diff_arrayop
export Derivative
export gradient

end # module
