# For implmentations of vector-Jacobian-product of scalar-specific primitives

# `Base.^` has a special definition that accepts an `Integer`.
# `Integer` is a subtype of `Real`, we need to manually specify dispatch rule for this definition.
Base.:^(x::RealWrapper, y::Integer) = trace_step(^, x, y)

#@diff_arrayop - Duplicated with that in `array.jl`.
@vjp Base.prod(Δ, p, x::Number) = (Δ,)

Base.one(x::RealWrapper) = one(val(x))
