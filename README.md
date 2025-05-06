# Tape-based Auto-differentiation for Julia

### Calculate gradient
```julia
julia> using ReverseAD

julia> f(x, y) = 4x * y
f (generic function with 1 methods)

julia> val, grads = gradient(f, 2, 3)
(24, [12.0, 8.0])
```

## API
```julia
function gradient(func, inputs...)::Tuple{Real, Vector{Real}}  
```
**Public** in the module `ReverseAD`.

This function is to calculate the gradient of some function at a given point.

Parameters:
1. `func`: a callable whose gradient will be calculated. `func` must return a scalar value. Valid inputs include normal functions, lambda functions and callable objects.
    
    The implementation of `func` **CANNOT** use array-related arithmetic functions.

1. `inputs`: arguments to `func`. Every element must be a `Real`.

Returns:

An instance of `Tuple{Real, Vector{Real}}`.

1. The first element is a scalar. It is the result of evaluating `func` with `inputs`.

1. The second element is the gradient of `func` at the point of `inputs`.
