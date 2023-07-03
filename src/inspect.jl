module Inspection

using Core: CodeInfo, GlobalRef, SSAValue, SlotNumber
using MacroTools: @capture

# A naive IR design.
# Only covering no-control-flow composition of primitives, and function calls that are potentially
# with side effects.
abstract type GraphNode end

struct Parameter <: GraphNode
    nodeidx::Int
end
struct Const{T} <: GraphNode    
    nodeidx::Int
    val::T
end
struct Call{F} <: GraphNode    
    nodeidx::Int
    func::F
    inputs::AbstractVector{Int}
end
nodeidx(n::GraphNode) = n.nodeidx

"""
    Graph

Represents the structure of a function. Within this function only pre-defined primitives can be
called and control flow cannot be used.
"""
struct Graph
    #TODO(caox) nodes::Vector{Union{GraphNode,Graph}}
    nodes::Vector{GraphNode}
    arity::Int
end

struct IRContext
    # Only GraphNodes, primitives in `ps` and Real (hard-coded literals in IR) can be contained.
    # Since we don't support control flow (loop), each position of SSA values is written at most
    # once.
    ssa_values::Vector
    slots::Vector
    ps

    graph::Graph
end

_is_gref(gref::GlobalRef, mod::Module, name::Symbol) = gref.mod == mod && gref.name == name 
_is_gref(::Vararg{Any,3}) = false

_is_self_slot(a::SlotNumber) = a.id == 1 # Slot #self# always has id 1.
_is_self_slot(::Any) = false

_is_callable(a) = !isempty(methods(a))

# In CodeInfo-level IR, only literal values can appear as constant.
# E.g. 42::Int64, 0x12::UInt8, 3.14::Float64, 3.14f0::Float32, true::Bool, "123"::String.
# Others, even primitive types, that have no literal syntax can only be represented by a SSAValue
# pointing to its constructor, e.g. Float16(3.14), 1//2, 1im.
#
# We only care about Real literals in CodeInfo-level IR for now.
_is_valid_const(a) = a isa Real

macro stop_if_null(cond)
    :($cond === nothing && return nothing) |> esc
end


"""
    maybe_ir_value(irctx::IRContext, v)::Union{Some{Union{GraphNode,_PrimitiveT,Real}}, Nothing}

Get the context of an IR element like GlobalRef, SSAValue or encoded literal values.
If the context of that element is not valid, returns `nothing`. Otherwise, that context is
wrapped in `Some`, like `Some(42)::Some{Int}`. 

This adds an extra flag in case that `Nothing` is made callable and be registered as a primitive.
"""
maybe_ir_value(irctx::IRContext, t::SSAValue) = Some(irctx.ssa_values[t.id])
maybe_ir_value(irctx::IRContext, t::SlotNumber) = Some(irctx.slots[t.id])
#=
The two implementations below are the start-points of all IR elements (SSAValue, SlotNumber) or
GraphNodes. All of them are "containers". 
The **initial values** within "containers" mentioned are certain functions or values of, say, Real.
And those functions or values must be originally from a GlobalRef or a literal constant
hard-coded in IR.

Those initial values are allowed to be, ordered by priority: registered primitives and Reals. The
preference is because that a concrete type like T<:Real can be extended to be callable, thus
registered as a primitive. E.g. `irctx.ps == [sin, cos, 3.14, "str", nothing]`.
=#
function maybe_ir_value(irctx::IRContext, gref::GlobalRef)
    v = getfield(gref.mod, gref.name)
    if v in irctx.ps || _is_valid_const(v) 
        return Some(v)
    end
    nothing
end
maybe_ir_value(irctx::IRContext, v) = _is_valid_const(v) ? Some(v) : nothing # Literals in IR

function _append_node!(g::Graph, ctor::T, args...) where T<:Type
    n = ctor(length(g.nodes)+1, args...)
    push!(g.nodes, n)
    return n
end

# Only when encountering terminators (calls, returns) in IR, do we ensure all related constants
# have been appended to the graph.
_ensure_node!(g::Graph, v::GraphNode) = v
_ensure_node!(g::Graph, v) = _append_node!(g, Const, v)

#TODO(caox) Support nested closure. 
#TODO(caox) We're reading external states before the computation gets realized. Those states may
# be changed by the side effects of the computation, then the value read here is inconsistent.
function handle_expr_call(irctx::IRContext, ir_op, ir_args)::Union{Some,Nothing}
    # Firstly check if it's `Core.getfield(#self#, :field)`.
    # `Core.getfield` is a built-in function thus unextendable.
    if _is_gref(ir_op, Core, :getfield) && _is_self_slot(ir_args[1])
        v = getfield(irctx.slots[1], (ir_args[2]::QuoteNode).value)
        return v in irctx.ps || _is_valid_const(v) ? Some(v) : nothing
    end
    
    # `Core._apply(callable, args...)` is a built-in.
    if _is_gref(ir_op, Core, :_apply)
        ir_op = ir_args[1]
        ir_args = ir_args[2:end]
    end

    # Element of `irctx.ps` takes priority. It may be a Real if its type is made callable.
    @stop_if_null maybe_func = maybe_ir_value(irctx, ir_op)
    func = something(maybe_func) # Retrieve from `Some`.
    func in irctx.ps || return nothing

    maybe_args = maybe_ir_value.(Ref(irctx), ir_args) 
    any(==(nothing), maybe_args) && return nothing
    args = something.(maybe_args)
    # Terminator: function call. Ensure all constants are appended as arguments.
    inputs = nodeidx.(_ensure_node!.(Ref(irctx.graph), args))

    Some(_append_node!(irctx.graph, Call, func, inputs))
end

function extract_primitive_graph(f, argtypes::Type{<:Real}...; primitives)
    #= Allowed expressions in CodeInfo:
    - `GlobalRef { mod::Module, name::Symbol }` Function identifier, e.g. `NaNMath.pow`.
    - `Expr { head::Symbol=:call, args=[op, ::SSAValue|SlotNumber, ...] }`
        Function invocation, e.g. `a - 1`, `sin(%4)`.
        `op` can be `GlobalRef, SSAValue, SlotNumber(ID of local variable)`, its value can only be
        a primitive or GlobalRefs: `Core.tuple`, `Base.float`...
    - `Expr { head=:(=)`, args=[::SlotNumber, ::GlobalRef|Constant] }`
    - `Expr { head=:return, args=[::SSAValue|SlotNumber|Constant] }`
    - `Expr { head=:call, args=[Core.getfield, SlotNumber(1), ::Symbol...]}`
        When `f` is a callable closure, `SlotNumber(1)`, denoted as `#self#`, is the closure
        itself. Such an expression is to get the variable from the closure. 
            E.g. `sin ∘ cos` (Note `∘` is a function, not a syntactic element.)

    In our expectation, the value of every line in `codeinfo.code` must come either from the
    function's inputs (SlotNumber), or from the result of calling another function with values of
    some lines (which are ultimately from the inputs).

    Expressions like control flow, global state accessing, are not allowed.
    =#

    _is_callable(f) || throw(ArgumentError("`f` is not callable."))
    #TODO(caox) Allow inputs to be abstract types. In this way `code_lowered` may return multiple
    # CodeInfo to cover all possible concrete subtypes (unless there are ambiguous dispatches).
    # Then make sure all CodeInfos are valid and peak suitable one after broadcast is performed.
    all(isconcretetype, argtypes) || throw(ArgumentError("`argtypes` must all be concrete."))

    arity = length(argtypes)
    params = Parameter.(1:arity)
    graph = Graph(GraphNode[params...], arity)
    if f in primitives
        _append_node!(graph, Call, f, 1:arity)
        return graph
    end

    codeinfos = code_lowered(f, argtypes)
    @assert length(codeinfos) == 1 "Unexpected ambiguous method resolution."
    codeinfo = codeinfos[1]

    # Unlike SSA values, slots can be overwritten. Slots include the callee itself, its parameters
    # and its local variables (the latter twos are placeholders, with no initial values but are
    # assigned later).
    slots = Vector{Any}(undef, length(codeinfo.slotnames))
    slots[1] = f
    slots[2:(1+arity)] = params

    irctx = IRContext(
        Vector{Any}(undef, length(codeinfo.code)),
        slots,
        primitives,
        graph)

    for (idx, ir_line) in enumerate(codeinfo.code)
        if ir_line isa Expr
            if @capture(ir_line, ir_a_ = ir_b_)
                # In JPL IR, only assignment to local variables or parameters will lead to an
                # explicit `=`-Expr. For other kinds of intermediate value, the Expr itself stands
                # for a `SSAValue`.
                @assert ir_a isa SlotNumber "Unexpected: left side of `=`-expr is not a SlotNumber."

                if @capture(ir_b, ir_op_(ir_args__))
                    maybe = handle_expr_call(irctx, ir_op, ir_args)
                else
                    maybe = maybe_ir_value(irctx, ir_b)
                end
                @stop_if_null maybe
                v = something(maybe)

                irctx.slots[(ir_a::SlotNumber).id] = v

            elseif @capture(ir_line, return ir_ret_)
                # If a function has no return value, `r===nothing`, where `r` is part of the Expr.
                # But if it explicitly returns `nothing`, an instance of type `Nothing`, `r` here
                # is a GlobalRef or SSAValue pointing to `Base.nothing`.
                ir_ret === nothing && return nothing

                @stop_if_null maybe = maybe_ir_value(irctx, ir_ret)
                # Terminator `return`, ensure the potential constant to return has been appended.
                v = _ensure_node!(irctx.graph, something(maybe))

            elseif @capture(ir_line, ir_op_(ir_args__))
                @stop_if_null maybe = handle_expr_call(irctx, ir_op, ir_args)
                v = something(maybe)
            else
                # May be `:gotoifnot`.
                return nothing
            end # !switch(ir_line.head)

        elseif ir_line isa GlobalRef
            @stop_if_null maybe = maybe_ir_value(irctx, ir_line::GlobalRef)
            v = something(maybe)

        else
            return nothing
        end # !switch(typeof(ir_line))

        irctx.ssa_values[idx] = v

    end # !for (idx, ir_line)

    irctx.graph
end


end # !module