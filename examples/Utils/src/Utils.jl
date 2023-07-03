module Utils

include("optimizers.jl")

using ReverseAD

export OneHotVector, Tree,
    randg, glorot_uniform, glorot_normal, cross_entropy

using Distributions: Gaussian, Uniform
randg(dims...; μ=0., σ=1e-3) = rand(Gaussian(μ, σ), dims...);
glorot_uniform(dims...; T=Float64) = (rand(T, dims...) .- T(0.5)) .* sqrt(T(24.0)/sum(dims))
glorot_normal(dims...; T=Float64) = randn(T, dims...) .* sqrt(T(2.0)/sum(dims))

cross_entropy(p::AbstractArray{<:Real}, q::AbstractArray{<:Real}) =
    -sum( p .* log.(q) )

"""
    OneHotVector
Column vector that only one dimension has value 1, others have value 0.
"""
struct OneHotVector <: AbstractVector{Bool}
    idx::Int
    length::Int
end
Base.size(v::OneHotVector) = (v.length,)
Base.getindex(v::OneHotVector, i::Integer) = i == v.idx
function Base.:*(A::AbstractMatrix, v::OneHotVector)
    size(A)[2] != length(v) && throw(ArgumentError("Dimensions dismatched."))
    A[:, v.idx]
end
@diff_arrayop Base.:*(A::AbstractMatrix, v::OneHotVector) [1]

"""
    Tree{T}
General tree-structure container.
"""
mutable struct Tree{T}
    value::T
    children::Vector{Tree{T}}
end
Tree(x::T, xs::Tree{T}...) where T = Tree{T}(x, collect(xs))
function Base.map(f, x::Tree)
    if isempty(x.children)
        Tree(f(x.value))
    else
        Tree(f(x.value), [map(f, c) for c in x.children])
    end
end

import AbstractTrees
AbstractTrees.children(tree::Tree) = tree.children

end #! module Utils
