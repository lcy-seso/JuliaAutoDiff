accum!(x::TapeNode{<:Real}, Δ::Real) = (x.result = x.result + Δ)
accum!(x::TapeNode{<:RealArray}, Δ::RealOrRealArray) = (x.result .+= Δ)
accum!(x::BroadcastNode, Δ::RealOrRealArray) = (x.result .+= Δ)
accum!(x::BroadcastJoinNode, Δ::RealOrRealArray) = (x.result .+= Δ)

"""
1. A tape node in the adjoint program (gradient node) stores one intermediate
value in gradient computation which is the gradient of an intermediate value
in forward computation.

2. One tape node in the adjoint program has exactly one corresponding
tape node in the primal program (value node).

3. A tape node in the adjoint program always has two parents: the first one is
fixed to be the index of the tape node that stores its adjoint value, the
second one is the index of the corresponding value node in the primal program.
"""
const ADJ_VAL_POS = 1
const DUAL_NODE_POS = 2

"""
    compute_vjps(t::Tape, node::TapeNode, adjoint_val::TapeNode)

Compute the Vector-Jacobian-Products of `node` with given adjoint value.
Constant and Input node do not have Vector-Jacobian-Products, so they should
not be dispatched into this function.
"""
function compute_vjps(tape::Tape, node::TapeNode, adjoint_val::TapeNode)
    # FIXME(ying): Fix this implementation after `vjp` is refined
    proc_node!(node::Constant, val::Any) = return # Constant can have any type.
    proc_node!(node::Union{TapeNode, Input, BroadcastNode,
                           BroadcastJoinNode}, val::RealOrRealArray) =
        push!(vjps, node.id => val)

    # TODO(ying): To-be-discussed: whether it will be better if vjp is a
    # overloadable function so that the implementation more efficient and
    # elegant if there is only one input is differentiable.
    inputs = parents(node)
    args = [value(tape[i]) for i in inputs] # RealOrRealArray or any constant value.
    vals = vjp(node.operation, #=operation produced `node`=#
               value(adjoint_val),  #=adjoint val=#
               value(node) #=output=#,
               args... #=inputs to forward computation=#)
    @assert length(vals) == length(args)

    vjps = Vector{Pair{Int, RealOrRealArray}}()
    for (i, input) in enumerate(inputs) proc_node!(tape[input], vals[i]) end
    return vjps
end

function update_adjoint_val!(tape::Tape, pid::NodeIndex, vjps)
    foreach(vjps) do vjp
        vjp::Pair{Int,<:RealOrRealArray}

        dual = get_dualnode(tape, tape[vjp.first])
        accum!(dual, vjp.second)
        #=TODO(caox)
        Currently, setting ADJ_VAL_POS is not needed. And should be changed to a more
        efficient implementation.
        =#
        #parents(dual)[ADJ_VAL_POS] = get_dualnode(tape, tape[pid])
        tape.ref[vjp.first] -= 1
    end
end

"""
    get_dualnode(tape::Tape, node::AbstractNode)

Each differentiable intermediate value in the primal program has exactly one
tape node to store the corresponding gradients. Given the `node` in the primal
program, this function returns the tape node in the adjoint program that stores
the corresponding gradient.
"""
function get_dualnode(tape::Tape, node::AbstractNode)
    # FIXME(ying) Try to optmize the underlying data structure of Tape, so
    # that to get the dual node in O(1).
    for i = outputid(tape) + 1 : length(tape)
        parents(tape[i])[DUAL_NODE_POS] == node.id && return tape[i]
    end
    node isa Input && return nothing
    error("Fail to find the adjoint node of $(node.id).")
end

function post_process_vjp(tape::Tape, vjps::Vector)
    # vjps is like Vector{Pair{Int64, RealOrRealArray }}}... But hard to specify correctly.
    next_to_visit = Vector{NodeIndex}()
    for v in vjps
        n = tape[v.first]
        if (n isa Union{TapeNode,BroadcastNode,BroadcastJoinNode} && get_ref(tape, n) == 0)
            push!(next_to_visit, v.first)
        end
    end
    next_to_visit
end


"""
    process_a_node!(tape::Tape, node::TapeNode)

Given `tape` and `node` which is a tape node in the primal program, this
function:
1. Retrieve tape node in the adjoint program that stores `node`'s gradient,
denoted as `node'`.
2. Compute Vector-Jacobian-Products(short for vjps) for each differentiable
input of `node`.
3. Update the computed vjps on `tape`.
4. Add `node`'s parents into next round's traverse only if gradients along
every path from output to `node` are accumulated already.
"""
function process_a_node!(tape::Tape, node::TapeNode)
    """
    If there are multiple paths from the output to the current node that
    propagate gradients to the current node, for a tape node, only until
    gradients from all these paths are accumulated at this node(`ref` reduce
    to 1), its children will be enqueued.
    """
    adjoint_val = get_dualnode(tape, node)
    vjps = compute_vjps(tape, node, adjoint_val)
    update_adjoint_val!(tape, node.id, vjps)
    return post_process_vjp(tape, vjps)
end

# In broadcast, the dimensions of the output may not match those (some are 1) of inputs.
# Accumulate VJP among those dimensions and keep result shape the same as the shape of input.
#
# Rule: Shape `vjp` is always bigger than or equal to the shape of `input`.
# This method is for when shape of `input` is an array.
function _reduce_bc_dims(vjp::AbstractArray{<:Real}, input::AbstractArray{<:Real})
    vjp_shape = size(vjp)
    vjp_shape == size(input) && return vjp # Fast detection for common cases.

    max_ndims = ndims(vjp)
    input_ndims = ndims(input)
    # VJP always has bigger dimensions than relevant inputs.
    # Dimensions not existing are with length of 1.
    input_shape = ones(Int, max_ndims)
    input_shape[1:input_ndims] = collect(size(input))

    # Dimensions not existing are with length of 1.
    accum_dims = findall(vjp_shape .!= input_shape)
    accum_vjp = sum(vjp, dims=accum_dims) 

    # Only drop dimensions not existing in `input` (always at the tail).
    drop_dims = filter(dim->dim>input_ndims, accum_dims)
    dropdims(accum_vjp; dims=tuple(drop_dims...)) # kwarg `dims` must be a tuple.
end

# This method is for when input is 0-dimensional, i.e. a scalar.
@inline _reduce_bc_dims(vjp::RealOrRealArray, input::Real) = sum(vjp)

@inline function _compute_bc_vjps(
    vjp_tuples::Array{<:Tuple},
    bc_Xs::AbstractVector{<:Any}, # Any constants
    diff_inputs::AbstractVector{NodeIndex})::Vector

    map(diff_inputs) do diff_dix
        vjp = getindex.(vjp_tuples, Ref(diff_dix))::Array # Always array.
        _reduce_bc_dims(vjp, bc_Xs[diff_dix]::RealOrRealArray) # Assert inputs to differentiate.
    end
end

@inline function _compute_bc_vjps(
    vjp_tuples::Tuple,
    # Not really used when the broadcast result is a scalar, since inputs can only be scalars too.
    bc_Xs::AbstractVector{<:Any}, # Any constants.
    diff_inputs::AbstractVector{NodeIndex})::Vector{<:Real}

    map(diff_inputs) do diff_idx
        vjp_tuples[diff_idx]::Real
    end
end

function process_a_node!(tape::Tape, node::BroadcastNode)
    # The dual node of a `BroadcastNode` is also a `BroadcastNode`.
    # in-gradient propagated to the current node.
    adjoint_node = get_dualnode(tape, node)::BroadcastNode
    input_idxes = parents(node)
    arity = length(input_idxes)

    bc_Δ = value(adjoint_node)
    bc_primal = value(node)
    bc_Xs = value.(tape[input_idxes])::AbstractVector{<:Any} # May be any constants.

    # Constant inputs are not to propogate.
    diff_inputs = filter(idx->!(tape[input_idxes[idx]] isa Constant), 1:arity)

    # `Ref(x)` prevents `x` gets iterated in the broadcast,
    # since `node.operation` may be an iteratable value.
    #
    # Each call to `vjp` return a tuple, whose elements are VJP to passed to children nodes.
    # Thus broadcasting `vjp` results in an Array{Tuple{typeof(vjp1),typeof(vjpX),...}}.
    # Or simply a Tuple{typeof(vjpX)...} if the primal result of this BroadcastNode is a scalar,
    # i.e. the broadcast in primal itself is about scalars, e.g. `3 .* 4`.
    vjp_tuples = vjp.(Ref(node.operation), bc_Δ, bc_primal, bc_Xs...)::Union{Array{<:Tuple},Tuple}

    vjps = _compute_bc_vjps(vjp_tuples, bc_Xs, diff_inputs)

    idxes_vjps = Pair.(input_idxes[diff_inputs], vjps)
    update_adjoint_val!(tape, -1#=pid=#, idxes_vjps)

    # Push children nodes that are differentiable to the waiting list.
    post_process_vjp(tape, idxes_vjps)
end

function process_a_node!(tape::Tape, node::BroadcastJoinNode)
    # The adjoint value for this BroadcastJoinNode has been properly accumulated.
    adjoint_node = get_dualnode(tape, node)

    #TODO(caox) Should add length(join_node) TapeNodes during `scan`.
    # All these new TapeNodes are `getindex`.
    idxes_vjps = Pair.(node.input_indexes, adjoint_node.result)
    update_adjoint_val!(tape, -1#=pid=#, idxes_vjps)

    # Broadcast-specific behavior: All final results of element-wise
    # computation (nodes to join) are only and only once used by the
    # `BroadcastJoinNode`.
    @assert all(i -> (get_ref(tape, tape[i]) == 0), node.input_indexes)
    node.input_indexes
end

"""
    bfs_visit!(tape::Tape, root::Union{TapeNode,BroadcastNode})

Traverse the given tape from the root by breadth-first-search to perform a
reverse accumulation for gradient computations.
"""
function bfs_visit!(tape::Tape, root::Union{TapeNode,BroadcastNode})
    # Initialize the queue that stores tape nodes to be visited.
    to_visit = Queue{NodeIndex}()
    foreach(i -> enqueue!(to_visit, i), process_a_node!(tape, root))

    while(!isempty(to_visit))
        curid = front(to_visit)
        dequeue!(to_visit)

        foreach(i -> enqueue!(to_visit, i), process_a_node!(tape, tape[curid]))
    end
end

_zero_adj(::Real) = 0.0
_zero_adj(v::AbstractArray) = zeros(Float64, size(v))

function allocate_adjoint_node!(tape::Tape,
        primal_node::Union{Input{<:Real}, TapeNode{<:Real}})
    undef_adj_val_idx = 0
    # FIXME(ying) Temporarily use `Base.error` as a placeholder for
    # the operation that produces the ajoint node.
    add_op!(tape, _zero_adj(value(primal_node)) #=Initialize gradient to zero=#,
            Base.error, undef_adj_val_idx, primal_node.id)
end

function allocate_adjoint_node!(tape::Tape,
        primal_node::Union{Input{<:RealArray},
                           TapeNode{<:RealArray}, BroadcastJoinNode})
    undef_adj_val_idx = 0
    val = value(primal_node)
    # FIXME(ying) Temporarily use `Base.error` as a placeholder for
    # the operation that produces the ajoint node.
    add_op!(tape, _zero_adj(val),
            Base.error, undef_adj_val_idx, primal_node.id)
end

function allocate_adjoint_node!(tape::Tape, primal_node::BroadcastNode)
    val = value(primal_node)

    # The dual node of a `BroadcastNode` is also a `BroadcastNode`.
    add_bc!(tape, _zero_adj(val),
            Base.error,
            -1#=adjoint value yet-to-give=#, primal_node.id)
end

@inline scan!(tape::Tape, root::Constant) = return
function scan!(tape::Tape, root::AbstractNode)
    get_ref(tape, root) == 0 && allocate_adjoint_node!(tape, root)
    inc_ref!(tape, root)

    get_ref(tape, root) == 1 && foreach(i->scan!(tape, tape[i]), parents(root))
end

allocate_adjoints!(tape::Tape) = scan!(tape, output(tape))

"""
    gen_adjoint!(primal::Tape)

Given an un-empty tape that will be taken as the primal program, append tape
nodes for gradient computations into the given tape.
"""
function gen_adjoint!(primal::Tape)
    # pre-allocate adjoints node before execute reverse sweep.
    allocate_adjoints!(primal)

    # The magic value 1. is the gradient of output node w.r.t itself
    primal[outputid(primal) + 1].result = 1.
    bfs_visit!(primal, output(primal))
end

function init_tape!(tape::Tape)
    # reset visiting counter for tape nodes in primal program.
    resize!(tape.ref, length(tape))
    fill!(tape.ref, zero(Int))
end

"""
    back(tape::Tape)

Given a non-empty tape `tape`:
1. If last tape node in `tape` is the output node, it is indicated that the
given tape is constructed first time without adjoint tape nodes generated, then
append tape nodes for gradient computations into the tape.
2. [Not Implemented Yet] If `length(tape)` is larger than the index of the
output node, it is indicated that adjoints have already been generated, then
execute the tape without generating adjoints.
"""
function back!(tape::Tape)
    isempty(tape) && return
    @assert(length(tape.outputs) == 1 && value(last(tape)) isa Real,
            "Current implementation only differentiates a scalar function.")

    init_tape!(tape)
    gen_adjoint!(tape)
end
