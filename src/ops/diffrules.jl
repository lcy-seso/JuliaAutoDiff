# Import only the qualified names of the dependencies of DiffRules module

import DiffRules, SpecialFunctions, NaNMath

for (M, f, arity) in DiffRules.diffrules()
    if arity == 1
        df_dx = DiffRules.diffrule(M, f, :x)
        @eval begin
            @vjp $M.$f(Δ, p, x) = ( Δ*$df_dx ,) #TODO utilize `p`.

            $M.$f(x::Wrapper) = trace_step($M.$f, x)
        end
    elseif arity == 2
        df_dx, df_dy = DiffRules.diffrule(M, f, :x, :y)
        @eval begin
            @vjp $M.$f(Δ, p, x, y) = ( Δ*$df_dx , Δ*$df_dy ) #TODO utilize `p`.

            $M.$f(x::Real, y::RealWrapper) = trace_step($M.$f, x, y)
            $M.$f(x::RealWrapper, y::Real) = trace_step($M.$f, x, y)
            $M.$f(x::RealWrapper, y::RealWrapper) = trace_step($M.$f, x, y)
        end
    end
end

# Currently suppose all parameters of the op are to differentiate and are arrays.
"""
    @diff_arrayop(op, delegateparams=nothing)

Mark an unary/binary function to be differentiable, with all parameters being arrays.
That function must be already imported to be extended.

Please note, to provide specially implemented derivatives to this function, users need to call
`@vjp` for this function. This is sometimes unnecessary, for example if this function is a 
composition of other differentiable functions(primitives).

If any parameters are concrete types, or even less-abstract types than `AbstractArray`, users
need to provide extra arguments for this macro, to specify the indexes of other parameters. 
And the types of the parameters in `op` expression must be declared the same as the target method.

# Examples
```jldoctest
@diff_arrayop matmul(x,y)
@vjp matmul(d,p,x,y) = ...

# For less-abstract types.
struct MySparseArray{T,N} <: AbstractArray{T,N}
    ...
end
Base.:(x::AbstractMatrix, y::MySparseArray) = ...
Base.:(x::MySparseArray, y::AbstractMatrix) = ...
@diff_arrayop Base.:*(x::AbstractMatrix, y::MySparseArray) [1]
@diff_arrayop Base.:*(x::MySparseArray, y::AbstractMatrix) [2]
```
"""
macro diff_arrayop(op, delegateparams=nothing)
    @assert op.head == :call "Given operation is not a function call."
    @assert (length(op.args) == 2 || length(op.args) == 3)
        "Only unary or binary array operation is acceptable."

    @capture(shortdef(op), (qname_(args__))) || error("Invalid expression.")

    if length(args) == 1
        p1 = args[1]
        #TODO Better to have subtype check.
        p1 isa Expr && p1.head == :(::) && error("Unary primitive must not have type constraint.") 
        quote
            $qname($(args[1])::ReverseAD.ArrayWrapper) = ReverseAD.trace_step($qname, $(args...))
        end |> esc
    elseif length(args) == 2
        p1, p2 = args
        @capture(p1, p1v_::p1t_ | p1v_)
        @capture(p2, p2v_::p2t_ | p2v_)

        if delegateparams != nothing
            #TODO Add more complete checks for `delegateparams` and `pNt_hook`.
            delegateparams.head == :vect || error("delegateparams must be a list of ints.")
            param = delegateparams.args[1] # There should be only index, since we are dealing with binary function.
            p1t_hook = param == 1 ? :(ReverseAD.ArrayWrapper) : p1t
            p2t_hook = param == 2 ? :(ReverseAD.ArrayWrapper) : p2t
            quote
                # `invoke` can bypass multiple-dispatch and directly call the method with
                # specified parameter types.
                $qname($p1v::$p1t_hook, $p2v::$p2t_hook) = Base.invoke($qname, Tuple{$p1t, $p2t}, $p1v, $p2v)
            end |> esc
        else
            if p1t == nothing && p2t == nothing
                quote
                    $qname($(args[1])::ReverseAD.ArrayWrapper, $(args[2])::ReverseAD.ArrayWrapper) = ReverseAD.trace_step($qname, $(args...))
                    $qname($(args[1])::ReverseAD.ArrayWrapper, $(args[2])::AbstractArray) = ReverseAD.trace_step($qname, $(args...))
                    $qname($(args[1])::AbstractArray, $(args[2])::ReverseAD.ArrayWrapper) = ReverseAD.trace_step($qname, $(args...))
                end |> esc
            elseif p1t != nothing && p2t != nothing
                error("Binary primitive must not have type constraints on both parameters.")
            else
                p1t = p1t == nothing ? :(ReverseAD.ArrayWrapper) : p1t
                p2t = p2t == nothing ? :(ReverseAD.ArrayWrapper) : p2t
                quote
                    $qname($p1v::$p1t, $p2v::$p2t) = ReverseAD.trace_step($qname, $p1v, $p2v)
                end |> esc
            end
        end
    else
        #TODO support n-arity primitives.
        error("Not implemented yet.")
    end
end