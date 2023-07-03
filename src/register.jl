function vjp end

const PRIMITIVES = Set()

"""
    @vjp(def::Expr)

Register differentiable primitive and its derivative function.

# Example
```jldoctest
julia> @vjp pow(Δ, p, x, y) = ( Δ * y*x^(y-1), Δ * p * ln(x) )

julia> @vjp exp = (Δ, p, x)->( Δ*p, )

julia> vjp_pow(Δ, p, x, y) = ( Δ * y*x^(y-1), Δ * p * ln(x) )
vjp_pow (generic function with 1 method)

julia> @vjp pow = vjp_pow
```
"""
macro vjp(def::Expr)
    """
    Calling @capture generates following local variables:
    - qname::Expr:
        The qualified name of a function. Example: `:(Base.:*)`,
        `:(NaNMath.tan)`
    - args::Vector{Expr}:
        The parameters of the function. Example: `[:g, :a, :b]`
    - body::Expr:
        The function body that is an Expression. The function implements the
        vector-Jacobian-product. `:(( g*b, g*a ))` for a primitive `a*b`
    - delegate::Expr:
        The function declaration. It is either a qualified function name or
        a lambda expression `:(NaNMath.pow)` or `:((g,x)->cos(x)*g)`.
    """
    @capture(shortdef(def), (qname_(args__) = body_) | (qname_ = delegate_)) ||
        error("An ill-formed expression.")

    if args != nothing && body != nothing
        # Here comma is required to expand `args` to a parameter list.
        vjp_func = :( ReverseAD.vjp(::typeof($qname), $(args...)) = $body )
    else
        # Simply pass all inputs to delegate using varargs.
        vjp_func = :( ReverseAD.vjp(::typeof($qname), xs...) = $delegate(xs...) )
    end

    #TODO guarantee this function definition is always in the ReverseAD module.
    quote
        $vjp_func
        push!(ReverseAD.PRIMITIVES, $qname)
    end |> esc
end
