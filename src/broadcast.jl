using Base.Broadcast
using Base.Broadcast: Broadcasted, BroadcastStyle

struct WrapperStyle <: BroadcastStyle end

Broadcast.BroadcastStyle(::Type{<:RealWrapper}) = WrapperStyle()
Broadcast.BroadcastStyle(::Type{<:ArrayWrapper{T, N}}) where {T,N} = WrapperStyle()

# WrapperStyle precedes all other BroadcastStyles, including ArrayConflict (which may appear when
# SparseMatrix and DiagonalMatrix meet).
# Since by default array-related BroadcastStyles are about memory layout in an array, not about any
# additional functionalities (e.g. tracing).
Broadcast.BroadcastStyle(a::WrapperStyle, ::BroadcastStyle) = a

_bc_materialize(bc::Broadcasted) = Broadcast.materialize(bc)
_bc_materialize(x) = x

_bc_unwrap(x::Wrapper) = x.tape[x.nodeidx]
_bc_unwrap(x) = x

_bc_reg_value(x::AbstractNode) = x.result
_bc_reg_value(x) = x

# Copied from `src/broadcast.jl`.
@inline _raw_broadcasted_with_style(::S, f, args...) where S = Broadcasted{S}(f, args)

#TODO(caox) This is not a broadcast API official doc suggests extending.
#=
Here we want to take over the call to `broadcasted` if **at least** one argument is a wrapper.
A convention to achieve this is to combine argument types first (where special ones take priority),
then dispatch using the result type as an argument.

Function `broadcasted` with first parameter a `BroadcastStyle` is how Base.Broadcast does the type
combination. If we get a `WrapperStyle` as winner, we know there's at least one wrapper argument.
=#
function Broadcast.broadcasted(s::WrapperStyle, f, args...; primitives=PRIMITIVES)
    # Realize all lazy broadcasts. Only after this can we get accurate element type.
    args = _bc_materialize.(args) # ::Union{Wrapper, Real/Array/String...}

    # At least one in the materialized `args` is a wrapper. All wrappers must be in the same tape.
    @assert are_args_wrapped(args...)

    g = Inspection.extract_primitive_graph(f, eltype.(args)...; primitives=primitives)
    if g === nothing
        # Use trace-via-scalar by default.
        # Since we are extending `broadcasted`, we have no way to call another implementation of
        # it, but only to call the content of that implementation explicitly.
        return _raw_broadcasted_with_style(s, f, args...)
    end

    tape = args[findfirst(x->x isa Wrapper, args)].tape

    # No need to allocate wrappers during the broadcast.
    args = _bc_unwrap.(args) # Union{AbstractNode, Real/String/...}

    # Mimic registers. Has the same structure as `g` (graph of primitives composition).
    regs = Vector{Any}(undef, length(g.nodes)) 

    # `args` is a tuple, tuple cannot be passed to set a range of array. Converted to vector first.
    regs[1:g.arity] = collect(args)
    for (idx, n) in enumerate(g.nodes)
        if n isa Inspection.Call
            f_args = regs[n.inputs] # Constants and TapeNodes.
            f_arg_idxes = ensure_node_idx!.(Ref(tape), f_args)
            op = n.func

            result = op.(_bc_reg_value.(f_args)...)
            bc_idx = add_bc!(tape, result, op, f_arg_idxes...)

            regs[idx] = tape[bc_idx]

        elseif n isa Inspection.Const
            # Don't add constant to tape here. A constant is only recorded when used as argument.
            regs[idx] = n.val
        end
    end

    # May be a constant like Real/Array, if in `f` its parameters are not used at all.
    ret = regs[end]
    if ret isa AbstractNode
        wrapper(ret.result, tape, ret.id)
    else
        #TODO(caox) Return an array, filled with `ret::Real`. Because we don't invoke standard broadcast process,
        # we need to check shapes of all arguments obey the rule of broadcast.
        error("Returning a constant in function to broadcast is not supported.")
    end
end

"""
    BroadcastNode

Represents a broadcast where the broadcasted function is a primitive.
"""
mutable struct BroadcastNode{T<:RealOrRealArray} <: AbstractNode
    id::NodeIndex  # the index of current node in the Tape.
    operation
    result::T
    parentidx::Vector{NodeIndex}
end

@inline parents(node::BroadcastNode) = node.parentidx

function add_bc!(tape::Tape, result::RealOrRealArray, op, pids::Vararg{NodeIndex})
    result isa Wrapper && error("result must be a plain scalar or an array.")

    nodeidx = length(tape) + 1
    push!(tape, BroadcastNode(nodeidx, op, result, collect(pids)))

    nodeidx
end

#=
`ElemT` is normally `RealWrapper{Int}` or `Any` (if types of inputs involved in the broadcast is
too complex).
It's possible, but not observed yet, for `ElemT` to be `RealWrapper{Real}` (the
type parameter is abstract).
=#
function Base.similar(bc::Broadcasted{WrapperStyle}, ::Type{ElemT}) where {N, ElemT}
    #TODO(caox) can we leverage, say SparseMatrix, here if we carry inner BroadcastStyle in WrapperStyle?
    similar(Array{ElemT}, axes(bc))
end

# Find the first `Wrapper` in the recorded broadcast arguments of `Broadcasted`. And get its tape.
# There may be nested `Broadcasted` being an argument, so we DFS the tree of argument.
function broadcast_get_tape(bc::Broadcasted{WrapperStyle})
    tape = broadcast_get_tape(bc.args)
    @assert tape != nothing
    tape
end
broadcast_get_tape(args::Tuple{}) = nothing
broadcast_get_tape(args::Tuple{Any}) = broadcast_get_tape(args[1])
function broadcast_get_tape(args::Tuple)
    tape = broadcast_get_tape(args[1])
    tape != nothing && return tape # Short-circuit after getting any valid wrapper.
    broadcast_get_tape(args[2:end])
end
broadcast_get_tape(a::Wrapper) = a.tape
broadcast_get_tape(::Any) = nothing

#TODO(caox) Official JPL doc doesn't suggest extending `materialize`. We need to re-impl it.
# However, it's the easiest way to guard both exit points of `copyto!` and non-`copyto!` path of
# `copy` (When JPL runtime cannot decide the result type of the broadcasted operator).
function Broadcast.materialize(bc::Broadcasted{WrapperStyle})
    real_wrappers = copy(Broadcast.instantiate(bc))
    join_wrapper_array(bc, real_wrappers)
end

struct BroadcastJoinNode{T<:Real, N} <: AbstractNode
    id::NodeIndex  # the index of current node in the Tape.
    result::Array{T,N}
    input_indexes::Array{NodeIndex,N}
end

@inline parents(node::BroadcastJoinNode) = node.input_indexes

function Base.show(io::IO, node::BroadcastJoinNode)
    println(io, typeof(node), " {")
    println(io, "\tid : ", node.id)
    println(io, "\tvalue : ", node.result)
    println(io, "\tinput_indexes : ", node.input_indexes)
end

# If not all steps in the broadcast result in the same RealWrapper{T}, we get RealWrapper without
# parameter type.
function join_wrapper_array(bc::Broadcasted{WrapperStyle}, a::AbstractArray{<:RealWrapper})
    result = val.(a)
    inputs = nodeidx.(a)

    t = isempty(a) ? broadcast_get_tape(bc) : a[1].tape

    node = BroadcastJoinNode(length(t)+1, result, inputs)
    push!(t, node)

    ArrayWrapper(result, t, length(t))
end

# If we broadcast on a scalar, `a` will be an `Array{RW, 0}`.
# Don't create a redundent `BroadcastJoinNode` but use that unique Wrapper directly.
join_wrapper_array(bc::Broadcasted{WrapperStyle}, a::AbstractArray{<:RealWrapper, 0}) = a[1]
