@testset "Broadcast" begin

using ReverseAD: RealWrapper, ArrayWrapper, Tape, add_input!, add_constant!, val,
    broadcast_get_tape, BroadcastNode, BroadcastJoinNode, WrapperStyle
using ReverseAD.Inspection: extract_primitive_graph
using Base.Broadcast: broadcasted, materialize, BroadcastStyle

# Laws required by JPL document.
@testset "Laws of broadcast" begin
let t = Tape()
    a = add_input!(t, [3 4 5])
    lhs = collect(Broadcast.broadcastable(a)) 
    rhs = collect(a)
    # Ignore internal data like `:tape`.
    @test size(lhs) == size(rhs)
    @test ((l,r)->typeof(l) == typeof(r)).(lhs, rhs) |> all
    @test ((l,r)->val(l) == val(r)).(lhs, rhs) |> all
end
end # !testset "Laws of broadcast"

@test broadcast_get_tape(()) === nothing
@test broadcast_get_tape((1,)) === nothing
@test broadcast_get_tape((1,2,3)) === nothing
@test broadcast_get_tape(42) === nothing

function test_bc_node(node::BroadcastNode, op, result, parentidx::Vector{Int})
    @test node.operation === op
    @test node.result == result
    @test node.parentidx == parentidx
end

@testset "Code-inspect broadcast" begin
    let t = Tape()
        a_val = [3.0 4 5]'
        b_val = [5.0 6 7]
        a = add_input!(t, a_val)
        b = add_input!(t, b_val)
        c = a .* b
        @test c isa ArrayWrapper{Float64,2}
        @test length(t) == 3
        test_input_node(t[1], [3.0 4 5]')
        test_input_node(t[2], [5.0 6 7])
        test_bc_node(t[3], *, (a_val .* b_val), [1,2])
    end

    let t = Tape()
        a_val = [1.0 2.0 3.0]'
        b_val = [4.0 5.0]
        a = add_input!(t, a_val)
        b = add_input!(t, b_val)
        function f(x, y) 
            z1 = x * y
            z2 = sin(x)
            z3 = cos(y)
            z4 = z2 / z3
            min(z1, z4)
        end
        @test nothing !== extract_primitive_graph(
            f, RealWrapper{Float64}, RealWrapper{Float64}, primitives=[*,/,sin,cos,min])

        c = f.(a, b)
        @test c isa ArrayWrapper{Float64,2}
        @test length(t) == 7
        test_bc_node(t[3], *,   (a_val .* b_val), [1,2])
        test_bc_node(t[4], sin, sin.(a_val),      [1])
        test_bc_node(t[5], cos, cos.(b_val),      [2])
        test_bc_node(t[6], /,   (sin.(a_val) ./ cos.(b_val)), [4,5])
        test_bc_node(t[7], min, f.(a_val, b_val), [3,6])
    end

    let t = Tape()
        a = add_input!(t, 3)
        b = add_input!(t, 4)
        c = a .* b
        @test c isa RealWrapper
        test_bc_node(t[3], *, 12, [1,2])
    end

    @testset "Empty broadcast" begin
        t = Tape()
        empty_arr = Array{Int,3}(undef, 0, 0, 0)
        a = add_input!(t, empty_arr)
        b = broadcasted(WrapperStyle(), sin, a) |> materialize
        @test b isa ArrayWrapper{Float64} # Although empty, the element type is inferred by `sin(::Int)`.
        @test b.nodeidx == 2
        @test isempty(b)
        @test ndims(b) == 3

        @test length(t) == 2 # 1 input, 1 bc node

        bc_node = t[end]
        test_bc_node(bc_node, sin, empty_arr, [1])
        @test isempty(bc_node.result)
        @test ndims(bc_node.result) == 3
    end # !testset "Empty broadcast"

end

@testset "Trace and join scalars" begin
    t = Tape()
    a = add_input!(t, [3.0 4 5]')
    b = add_input!(t, [5.0 6 7])
    bc = broadcasted(WrapperStyle(), *, a, b; primitives=[]) # Deprecate code-inspect broadcast.
    @test similar(bc, RealWrapper{Float64}) isa Array{RealWrapper{Float64}, 2}
    @test similar(bc, RealWrapper{Real}) isa Array{RealWrapper{Real}, 2}
    @test similar(bc, Any) isa Array{Any, 2}
    
    let t = Tape()
        a = add_input!(t, [2 3 4])
        @test broadcast_get_tape(a) === t
        bc = broadcasted(WrapperStyle(), *, a, [4 5 6]; primitives=[])
        @test broadcast_get_tape(bc) === t
    end

    let t = Tape()
        a = add_input!(t, [3 4 5])
        f(x) = iseven(val(x)) ? sin(x) : cos(x)
        b = f.(a)
        @test b isa ArrayWrapper{Float64}
        @test b.nodeidx == 11
        @test val(b) == [cos(3) sin(4) cos(5)]

        @test length(t) == 11 # 1 input, 3 pairs of {coordinate,getindex,sin(elem)}, 1 join node.
        test_input_node(t[1], [3 4 5])

        for i in 0:2
            coord_node = t[1+i*3+1]
            getidx_node = t[1+i*3+2]
            op_node = t[1+i*3+3]
            test_const_node(coord_node, CartesianIndex(1, i+1))
            test_node(getidx_node, getindex, [3 4 5][i+1], [1, coord_node.id])

            op = iseven(i) ? cos : sin
            test_node(op_node, op, op([3 4 5][i+1]), [getidx_node.id])
        end

        join_node = t[end]
        @test join_node isa BroadcastJoinNode
        @test join_node.result == [cos(3) sin(4) cos(5)]
        @test join_node.input_indexes == [4 7 10]
    end

    @testset "Join scalar" begin
        t = Tape()
        multiply(x, y) = true ? x * y : missing 
        a = add_input!(t, 3)
        b = add_input!(t, 4)
        c = multiply.(a, b)
        @test c isa RealWrapper
        test_node(t[3], *, 12, [1,2]) # Not `BroadcastJoinNode`.
    end

    @testset "Empty broadcast" begin
        t = Tape()
        a = add_input!(t, Array{Int,3}(undef, 0, 0, 0))
        b = broadcasted(WrapperStyle(), sin, a; primitives=[]) |> materialize
        @test b isa ArrayWrapper{Float64} # Although empty, the element type is inferred by `sin(::Int)`.
        @test b.nodeidx == 2
        @test isempty(b)
        @test ndims(b) == 3

        @test length(t) == 2 # 1 input, 1 join node

        join_node = t[end]
        @test isempty(join_node.result)
        @test ndims(join_node.result) == 3
        @test isempty(join_node.input_indexes)
        @test ndims(join_node.input_indexes) == 3
    end # !testset "Empty broadcast"

end # !testset "Trace and join scalars"



#TODO(caox)
#=
```
struct SparseVecStyle <: Broadcast.AbstractArrayStyle{1} end
struct SparseMatStyle <: Broadcast.AbstractArrayStyle{2} end
Base.BroadcastStyle(::Type{<:SparseVector}) = SparseVecStyle()
Base.BroadcastStyle(::Type{<:SparseMatrixCSC}) = SparseMatStyle()
```
It's common that others derive AbstractArrayStyle. 
Make sure our derived Style works well with them.

Check `ArrayConflict` in `broadcast.jl`.
This is the standard fallback when two derived AbstractArrayStyles meet.
=#

#TODO
# test when (assuming) BB.combine_eltypes is a concrete type like RealWrapper{Int}

# test when (assuming) BB.combine_eltypes is not a concrete type,
# e.g. in case of ArrayWrapper{Real}.*ArrayWrapper{Real} the combined eltype will be Any 
# (may change in future version, but assume it in the UT)


end # !testset "Broadcast"