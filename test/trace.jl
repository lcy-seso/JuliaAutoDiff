# Utils for test tape nodes
#TODO can move them to dedicated file
#-----------------------------
function test_wrapper(wrapper::ReverseAD.Wrapper, val, tape, nodeidx)
    @test wrapper.val == val
    @test wrapper.tape === tape
    @test wrapper.nodeidx == nodeidx
end

test_input_node(node::ReverseAD.Input, result) = (@test node.result == result)

test_const_node(node::ReverseAD.Constant, result) = (@test node.result == result)

function test_node(node::ReverseAD.TapeNode, op, result, parentidx::Vector)
    @test node.operation === op
    @test node.result == result
    @test node.parentidx == parentidx
end

#-----------------------------

@testset "Trace" begin

using ReverseAD: add_input!, add_constant!, add_op!, Tape, trace_step, trace!
using ReverseAD: RealWrapper

get_inputs(tape::Tape) = collect(Iterators.flatten(tape.inputs))

@testset "Inputs" begin


let input_t = Tape()
    a = add_input!(input_t, 42)
    @test length(input_t) == 1
    @test get_inputs(input_t) == [1]
    @test a isa RealWrapper
    test_wrapper(a, 42, input_t, 1)
    test_input_node(input_t[1], 42)
end

let unary_t = Tape()
    a = add_input!(unary_t, 42)
    # Result in the node is controlled by caller
    b = add_op!(unary_t, 99, error, a.nodeidx)
    @test length(unary_t) == 2
    @test get_inputs(unary_t) == [1]
    @test b isa RealWrapper
    test_wrapper(b, 99, unary_t, 2)
    test_node(unary_t[2], error, 99, [1])
end

let binary_t = Tape()
    a = add_input!(binary_t, 2)
    b = add_input!(binary_t, 3)
    c = add_op!(binary_t, 6, *, a.nodeidx, b.nodeidx)
    @test length(binary_t) == 3
    @test get_inputs(binary_t) == [1,2]
    @test c isa RealWrapper
    test_wrapper(c, 6, binary_t, 3)
    test_node(binary_t[3], *, 6, [1,2])
end

let t = Tape()
    a = add_input!(t, 2)
    b = 3
    c = a*b
    @test length(t) == 3
    @test get_inputs(t) == [1]
    test_wrapper(c, 6, t, 3)
    test_const_node(t[2], 3)
    test_node(t[3], *, 6, [1,2])
end

let t = Tape()
    a = add_input!(t, 100)
    b = trace_step(log10, a)
    @test length(t) == 2
    @test get_inputs(t) == [1]
    @test b.tape === a.tape
    test_wrapper(b, 2, t, 2)
    test_node(t[2], log10, 2, [1])
end

let t = Tape()
    a = add_input!(t, 4)
    b = add_input!(t, 6)
    c = a*b
    @test length(t) == 3
    @test get_inputs(t) == [1,2]
    test_wrapper(c, 24, t, 3)
    test_node(t[3], *, 24, [1,2])
end

end # !testset "Inputs"

@testset "Constants" begin

let const_t = Tape()
    a_idx = add_constant!(const_t, 42)
    @test length(const_t) == 1
    @test get_inputs(const_t) == []
    @test a_idx == 1
    test_const_node(const_t[a_idx], 42)
end

let literal_t = Tape()
    a = add_input!(literal_t, 4)
    b = a*2
    @test length(literal_t) == 3
    @test get_inputs(literal_t) == [1]
    test_input_node(literal_t[1], 4)
    test_const_node(literal_t[2], 2)
    test_node(literal_t[3], *, 8, [1,2])
end

let literal_t2 = Tape()
    a = add_input!(literal_t2, 4)
    b = a*2
    c = b*2
    @test length(literal_t2) == 5
    @test get_inputs(literal_t2) == [1]
    test_input_node(literal_t2[1], 4)
    test_const_node(literal_t2[2], 2)
    test_node(literal_t2[3], *, 8, [1,2])
    test_const_node(literal_t2[4], 2)
    test_node(literal_t2[5], *, 16, [3,4])
end

let any_const_t = Tape()
    a_idx = add_constant!(any_const_t, "A string")
    b_idx = add_constant!(any_const_t, CartesianIndex(2,3))
    c_idx = add_constant!(any_const_t, nothing)
    @test length(any_const_t) == 3
    @test get_inputs(any_const_t) == []
    test_const_node(any_const_t[a_idx], "A string")
    test_const_node(any_const_t[b_idx], CartesianIndex(2,3))
    test_const_node(any_const_t[c_idx], nothing)
end

end # !testset "Constants"

@testset "trace!" begin

let simple_t = Tape()
    val = trace!(simple_t, *, 4, 5)
    @test val isa Int64
    @test length(simple_t) == 3
    @test get_inputs(simple_t) == [1,2]
    test_input_node(simple_t[1], 4)
    test_input_node(simple_t[2], 5)
    test_node(simple_t[3], *, 20, [1,2])
end

let complex_t = Tape()
    k(a,b,c) = (a*b+log10(c))*2
    val = trace!(complex_t, k, 3, 4, 100)
    @test val isa Float64
    @test length(complex_t) == 8
    @test get_inputs(complex_t) == [1,2,3]
    test_input_node(complex_t[1], 3)
    test_input_node(complex_t[2], 4)
    test_input_node(complex_t[3], 100)

    # TODO For 4-th and 5-th nodes, ensure the order of evaluation.
    # TODO We're assuming it's from-left-to-right.
    test_node(complex_t[4], *, 12, [1,2])
    test_node(complex_t[5], log10, 2, [3])

    test_node(complex_t[6], +, 14, [4,5])

    test_const_node(complex_t[7], 2)
    test_node(complex_t[8], *, 28, [6,7])
end

let any_t1 = Tape()
    a = add_input!(any_t1, [4 5 6; 7 8 9])
    b = trace_step(getindex, a, CartesianIndex(2,3))
    @test b isa RealWrapper{Int}
    @test length(any_t1) == 3
    test_const_node(any_t1[2], CartesianIndex(2,3))
    test_node(any_t1[3], getindex, 9, [1,2])
end

let any_t2 = Tape()
    f(a) = a[CartesianIndex(2,3)]
    val = trace!(any_t2, f, [4 5 6; 7 8 9])
    @test val isa Int
    @test length(any_t2) == 3
    test_const_node(any_t2[2], CartesianIndex(2,3))
    test_node(any_t2[3], getindex, 9, [1,2])
end

let any_t3 = Tape()
    f(x, y::String) = x * length(y)
    a = add_input!(any_t3, 2)
    b = trace_step(f, a, "String")
    @test b isa RealWrapper{Int}
    @test length(any_t3) == 3
    test_const_node(any_t3[2], "String")
    test_node(any_t3[3], f, 2*length("String"), [1,2]) # f is not differentiable, although.
end

end # !testset "trace!"

@testset "Primitive Dispatch" begin

let t = Tape()
    a = add_input!(t, 3)
    b = a^2
    @test length(t) == 3
    @test b isa RealWrapper
    @test get_inputs(t) == [1]
    test_wrapper(b, 9, t, 3)
    test_input_node(t[1], 3)
    test_const_node(t[2], 2)
    test_node(t[3], ^, 9, [1,2])
end

end # !testset "Primitive Dispatch"

end # !testset "Trace"
