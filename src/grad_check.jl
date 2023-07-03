"""
    numerical_grad(f, args...)

Calculate the gradient of a scalar-valued `f` at the given point (`args`) using numerical
differentiation.

The implementation uses _symmetric derivative_ to do the numerical differentiation, the formula of
_symmetric derivative_ for unary functions is defined as:

`` \\lim_{h \\to 0} \\frac{f(x+h)-f(x-h)}{2h} ``

For `args` that is a vector of scalars and arrays, this function will flatten them all into a
single vector of scalars, and treat only one scalar as variable each time (thus an unary
function), other scalars are kept fixed.

Due to the addition/subtraction of the small number `h`, numerical differentiation can only be
applied to functions whose inputs are (or convertiable to) floats and arrays of float.

Example:
```julia
julia> numerical_grad(sin, 3.14159)
1-element Array{Float64,1}:
 -1.0000000001360907

julia> numerical_grad((a,b)->sum(a.^b), [2,3,4], 3)
 2-element Array{Any,1}:
     [12.0, 27.0, 48.0]
  123.93054836934425
```
"""
function numerical_grad(f, args::Vararg{RealOrRealArray}; h=1e-5)
    !(f(args...) isa Real) && error("The result of `f` must be a real scalar.")

    # Convert Ints and Int arrays to Float64 ones. Float64 ones remain intact.
    args = collect(float.(args))
    grad = zero.(args)

    for i in eachindex(args)
        arg = args[i]

        if arg isa AbstractArray
            # Arrays are always accessed by reference
            for idx in eachindex(arg)
                origin = arg[idx]

                arg[idx] = origin - h
                y1 = f(args...)
                arg[idx] = origin + h
                y2 = f(args...)
                arg[idx] = origin # Recover

                grad[i][idx] = (y2 - y1) / (2h)
            end
        else
            # Julia primitive types (e.g. Int, Float64) are Real but accessed by value.
            @assert arg isa Real
            origin = arg

            args[i] = origin - h
            y1 = f(args...)
            args[i] = origin + h
            y2 = f(args...)

            args[i] = origin

            grad[i] = (y2 - y1) / (2h)
        end
    end

    return grad
end

function grad_check(f::Function, args::Vararg{RealOrRealArray}; rtol=1e-3, h=1e-5)::Bool
    num_grad = numerical_grad(Statistics.mean ∘ f, args...; h=h)
    _, ad_grad = gradient(Statistics.mean ∘ f, args...)
    isapprox.(num_grad, ad_grad; rtol=rtol) |> all
end
