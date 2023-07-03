@testset "Code Inspection" begin

#= Important note:

All sample cases below, if defined and inspected via `code_lowered()` in REPL, will result in
different CodeInfo (IR) than defined and inspected here.
Because we are defining them within `@testset`, which is a local scope, thus the generated IR may
unexpectedly use closure. Please pay attention to the difference.
=#

using ReverseAD.Inspection: Graph, GraphNode, Parameter, Const, Call, extract_primitive_graph,
    IRContext, maybe_ir_value, handle_expr_call

using Core: SSAValue, SlotNumber

TEST_PS = Set([sin, cos, exp, *, /])

@testset "Get IR value" begin
    irctx = IRContext(
        [42], # SSA values
        [36], # Slots
        [sin, cos, nothing], # primitives
        Graph(GraphNode[], 0)
    )
    @test maybe_ir_value(irctx, GlobalRef(Base, :π)) == Some(π)
    @test maybe_ir_value(irctx, GlobalRef(Base, :sin)) == Some(sin)
    @test maybe_ir_value(irctx, GlobalRef(Base, :tan)) == nothing
    @test maybe_ir_value(irctx, GlobalRef(Base, :nothing)) == Some(nothing)
    @test maybe_ir_value(irctx, SSAValue(1)) == Some(42)
    @test maybe_ir_value(irctx, SlotNumber(1)) == Some(36)
    @test maybe_ir_value(irctx, 3.14f0) == Some(Float32(3.14))

    # Unsupported literals in IR
    @test maybe_ir_value(irctx, "str") == nothing
    @test maybe_ir_value(irctx, nothing) == nothing
    @test maybe_ir_value(irctx, Parameter(42)::GraphNode) == nothing # Impossible literals.
end

function test_params(g::Graph, arity::Int)
    @test g.arity == arity
    @test all(x->x isa Parameter, g.nodes[1:arity])
end
test_const(constant::Const{T}, v::T) where T = @test constant.val == v
test_call(call::Call{F}, f::F, inputs::AbstractVector{Int}) where F = @test call.inputs == inputs

@testset "Primitive" begin
    let g = extract_primitive_graph(sin, Int; primitives=TEST_PS)
        test_params(g, 1)
        test_call(g.nodes[2], sin, 1:g.arity)
    end

    let g = extract_primitive_graph(/, Int, Int; primitives=TEST_PS)
        test_params(g, 2)
        test_call(g.nodes[3], /, [1,2])
    end
end

@testset "Function" begin
    let g = extract_primitive_graph(x->sin(cos(x)), Int; primitives=TEST_PS)
        test_params(g, 1)
        test_call(g.nodes[2], cos, 1:g.arity)
        test_call(g.nodes[3], sin, [2])
    end

    function comp(x, y)
        z1 = x * y
        z2 = y / x
        z1 * z2
    end
    let g = extract_primitive_graph(comp, Int, Int; primitives=TEST_PS)
        test_params(g, 2)
        test_call(g.nodes[3], *, [1,2])
        test_call(g.nodes[4], /, [2,1])
        test_call(g.nodes[5], *, [3,4])
    end
end

@testset "Closure" begin
    g = extract_primitive_graph(sin∘cos, Int; primitives=TEST_PS)
    test_params(g, 1)
    test_call(g.nodes[2], cos, [1])
    test_call(g.nodes[3], sin, [2])

    #= The input is a nested closure. It's built as:
    > closure1 = sin∘cos
    > closure2 = closure1∘exp
    Nested closures are not supported now. =#
    @test nothing === extract_primitive_graph(sin∘cos∘exp, Int; primitives=TEST_PS)
end

@testset "Invalid cases" begin
    # Control flow is not supported
    @test nothing === extract_primitive_graph(x->x>5 ? sin(x) : cos(x), Int; primitives=TEST_PS)

    let rec_f(x) = rec_f(x)
        @test nothing === extract_primitive_graph(rec_f, Int; primitives=TEST_PS)
    end
end

@testset "Consistent flow" begin
    io_prim(x) = x # Assuming it's doing IO.
    function cf(x)
        ignored = io_prim(x)
        return 42
    end    
    g = extract_primitive_graph(cf, Int; primitives=[io_prim])
    test_params(g, 1)
    test_call(g.nodes[2], io_prim, 1:g.arity)
    test_const(g.nodes[3], 42)

    function puref(x)
        y = 42
        y = sin(x)
        x = 36
        y
    end
    g = extract_primitive_graph(puref, Int; primitives=TEST_PS)
    test_params(g, 1)
    @test length(g.nodes) == 2
    test_call(g.nodes[2], sin, 1:g.arity)
end

end # !testset "Code Inspection"