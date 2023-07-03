"""
    add_constant!(tape::Tape, val::Any)

Appends a `Constant` node to the tape.
Returns the index of that `Constant` node in the tape.
"""
function add_constant!(tape::Tape, val)
    val isa Wrapper && error("val must not be a wrapper.")

    # This is a node to be added into Tape, thus its index in tape is
    # (length of tape) + 1
    const_node_idx = length(tape) + 1
    push!(tape, Constant(const_node_idx, val))
    return const_node_idx
end

function add_op!(tape::Tape, result::RealOrRealArray, f::Function,
                 pids::Vararg{NodeIndex})
    result isa Wrapper && error("result must be a plain scalar or an array.")

    id = length(tape) + 1
    op_node = TapeNode(id, f, result, pids...)
    push!(tape, op_node)

    # Dispatch to some wrapper.
    # The result is not neccessarily of some type as inputs.
    return wrapper(result, tape, id)
end

# If input is already recorded in tape, retrieve its index.
ensure_node_idx!(tape::Tape, x::AbstractNode) = x.id
ensure_node_idx!(tape::Tape, x::Wrapper) = x.nodeidx
# If it's a new constant value, record it into tape and return index of the newly added node.
ensure_node_idx!(tape::Tape, x) = add_constant!(tape, x)

# Specific case for unary function, the only input must be a wrapper.
function trace_step(f::Function, x::Wrapper)
    tape = x.tape
    result = f(val(x))
    return add_op!(tape, result, f, x.nodeidx)
end

# General case for binary functions. Each input is either a wrapped type or a constant value.
function trace_step(f::Function, x, y)
    # Since it is in `trace_step`, there's at least one wrapper in [x,y].
    # And, if both inputs are wrapper, they must be in a same tape.
    @assert are_args_wrapped(x, y)

    tape = x isa Wrapper ? x.tape : y.tape
    result = f(val(x), val(y))
    return add_op!(tape, result, f,
                   ensure_node_idx!(tape, x),
                   ensure_node_idx!(tape, y))
end

function trace_step(f::Function, args...)
    @assert are_args_wrapped(args...)

    tape = args[findfirst(x->x isa Wrapper, args)].tape
    result = f(val.(args)...)
    return add_op!(tape, result, f,
                   (ensure_node_idx!(tape, x) for x in args)...)
end

"""
    trace!(tape::Tape, f::Function, args...)

TODO: keyword arguments are not supported yet.
"""
function trace!(tape::Tape, f::Function, args...)
    wrapped_args = []
    for arg in args
        push!(tape.inputs, [])
        push!(wrapped_args, add_input!(tape, arg))
    end
    @assert length(tape.inputs) == length(args)
    return f(wrapped_args...).val
end
