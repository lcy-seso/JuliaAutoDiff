# For implmentations of vector-Jacobian-product of array-specific primitives

import Statistics
import NNlib

using SparseArrays: sparsevec

#TODO(caox) Currently stdlib `SparseArrays` only supports 2-D matrix.
# So we build a N-D array from 1-D vector, by firstly converting N-D coordinates to linear indexes.
@inline function _sparse_array(shape::Tuple, idx::CartesianIndex, value)
    linear_idx = Base._sub2ind(shape, idx.I...)        
    return _sparse_array(shape, linear_idx, value)
end
function _sparse_array(shape::Tuple, idx::Integer, value)
    length = prod(shape)
    reshape(sparsevec([idx], [value], length), shape)
end

# The second parameter of `getindex` is not differentiable.
@vjp Base.getindex(Δ, p, a, i::Union{Integer,CartesianIndex}) = (_sparse_array(size(a), i, Δ), nothing)
Base.getindex(a::ArrayWrapper, i::Union{Integer,CartesianIndex}) = trace_step(getindex, a, i)

Base.getindex(A::ArrayWrapper, I...) = trace_step(getindex, A, I...)
@vjp Base.getindex(Δ, _, A, I...) = begin
    VJP = zeros(Float64, size(A))
    VJP[I...] = Δ

    (VJP, fill(nothing, length(I))...)
end

Base.setindex!(a::ArrayWrapper, v, i...) =
    error("Not implemented. Require support for N-ary primitive and design for in-place modification.")

@diff_arrayop Base.sum(x)
@vjp Base.sum(Δ, y, x) = fill(Δ, size(x)) |> tuple

#TODO(caox) Although robust, this implementation of VJP function is unfriendly to tape-based AD.
# Simplify the internal logic to exploit broadcast.
@diff_arrayop Base.prod(X)
@vjp Base.prod(Δ, p, X::AbstractArray) = tuple(let 
        VJP = fill(Δ, size(X))
        A = nothing
        for idx in eachindex(X)
            x = X[idx]
            if x != 0
                VJP[idx] *= p / x
            else
                A === nothing && (A = copy(X))
                A[idx] = 1
                VJP[idx] *= prod(A)
                A[idx] = 0
            end
        end
        VJP
    end)

@diff_arrayop Statistics.mean(x)
@vjp Statistics.mean(Δ, y, x) = tuple(isempty(x) ? similar(x) : fill(Δ/length(x), size(x)))

@diff_arrayop Base.:*(x, y)
@diff_arrayop Base.:*(x::AbstractVector, y)
@diff_arrayop Base.:*(x::AbstractMatrix, y)
@diff_arrayop Base.:*(x, y::AbstractVector)
@diff_arrayop Base.:*(x, y::AbstractMatrix)
@vjp Base.:*(Δ, z::AbstractArray, x::AbstractArray, y::AbstractArray) =
    (Δ * transpose(y), transpose(x) * Δ)

@diff_arrayop NNlib.softmax(x)
@vjp NNlib.softmax(Δ, p, x) = ( NNlib.∇softmax(Δ, x) ,)

@diff_arrayop NNlib.sigmoid(x)
@vjp NNlib.sigmoid(Δ, p, _) = ( p*(1-p) ,)