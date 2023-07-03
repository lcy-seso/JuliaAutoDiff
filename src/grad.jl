"""
    `retrieve_grad_and_unwrapval` does two things:

1. Introspect the type definition of `val`, if a value with wrapper type (a
   differentiable value) is encountered:
    - retrieve gradient from the tape.
    - if no gradients are found, it means such a value is not used in user's
      computation, then set its gradients to zero.
2. Unwrap the wrapped value.
"""

"""
    retrieve_grad_and_unwrapval!(val::AbstractArray, grad::AbstractArray)

This methods handles `val` and `grad` that are container type AbstractArray{T},
and T is user-defined type.
"""
function retrieve_grad_and_unwrapval!(val::AbstractArray, grad::AbstractArray)
    for (i, (v, g)) in enumerate(zip(val, grad))
        if v isa Wrapper
            !(v.tape[v.nodeidx] isa Input) && continue
            grad_node = get_dualnode(v.tape, v.tape[v.nodeidx])

            grad[i] = (grad_node === nothing ? zero(grad[i]) :
                                               grad_node.result)
            # unwrap the wrapper type.
            val[i] = v.val
        else
            retrieve_grad_and_unwrapval!(v, g)
        end
    end
end

"""
    retrieve_grad_and_unwrapval!(val, grad)

This methods handles `val` and `grad` that are user-defined composition types.
"""
function retrieve_grad_and_unwrapval!(val, grad)
    vtype = typeof(grad)
    isstructtype(vtype) && foreach(
        fn -> retrieve_grad_and_unwrapval!(val, grad, fn), fieldnames(vtype))
end

function retrieve_grad_and_unwrapval!(val, grad, fieldname::Symbol)
    g = getfield(grad, fieldname)
    v = getfield(val, fieldname)

    if g isa RealOrRealArray
        @assert v isa Wrapper
        grad_node = get_dualnode(v.tape, v.tape[v.nodeidx])
        grad_node === nothing ? 
            setfield!(grad, fieldname, zero(g)) :
            setfield!(grad, fieldname, grad_node.result)
        # unwrap the wrapper type.
        setfield!(val, fieldname, v.val)
    else
        retrieve_grad_and_unwrapval!(v, g)
    end
end

function collect_grads!(program::Tape, grads, args...)
    # TODO(Ying): keyword arguments are not processed yet.
    for (i, arg) in enumerate(args)
        if arg isa RealOrRealArray
            # primitive differentiable type RealOrRealArray hits this branch
            @assert length(program.inputs[i]) == 1
            nid = program.inputs[i][1]
            grad_node = get_dualnode(program, program[nid])
            grad_node === nothing && continue
            grads[i] = value(grad_node)
        else
            # user-defined type hits this branch.
            retrieve_grad_and_unwrapval!(arg, grads[i])
        end
    end
    grads
end

"""
    gradient(f::Function, args...) -> Tuple{Real, Any}

Compute gradients of `f` w.r.t given `args`.
"""
function gradient(f::Function, args...)
    tape = Tape()

    # FIXME(Ying) Given input differentiable value A, `B = deepcopy(A)`
    # is a TEMPORARY solution to create a value `B` that has exactly the
    # same structure as A to store the returned gradients.
    # `deepcopy` is NOT necessary, and it is sure to CAUSE EFFICIENCY ISSUE.
    grads = [deepcopy(arg) for arg in args]

    val = trace!(tape, f, args...)
    push!(tape.outputs, length(tape))
    back!(tape)

    collect_grads!(tape, grads, args...)
    return val, grads
end
