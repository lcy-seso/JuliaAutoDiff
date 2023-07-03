<!-- TOC -->

- [Handle Broadcast on Wrapper Types](#handle-broadcast-on-wrapper-types)
    - [Official Documents](#official-documents)
    - [Broadcast Functions on Wrapper Types](#broadcast-functions-on-wrapper-types)
        - [Join Operator](#join-operator)
        - [Back-propagation Behavior of Join](#back-propagation-behavior-of-join)

<!-- /TOC -->

# Handle Broadcast on Wrapper Types

This document describes the design of the adaption to the broadcast mechanism of JPL for the AD
system. 

The prerequisite to read this document is to understand the basic design of AD system in
[`tape.md`](./tape.md).

## Official Documents
The broadcast mechanism of JPL is well documented and implemented. Please refer to...
- <https://docs.julialang.org/en/v1.0/manual/interfaces/#man-interfaces-broadcasting-1>
- <https://github.com/JuliaLang/julia/blob/v1.0.3/base/broadcast.jl>

... for detailed information.

## Broadcast Functions on Wrapper Types

Broadcasting a function on some inputs with the same shape (e.g. `f.(A, B, C)`) is essentially equivalent to:

```julia
R = Array(undef, size(A))
for i in eachindex(A)
    R[i] = f(A[i], B[i], C[i])
end
return R
```

Since `A[i]` is a syntactic sugar over `Base.getindex(A, i)`, we make `Base.getindex` a
_primitive_ in the AD system, extending this function to accept a `ArrayWrapper`:

```julia
Base.getindex(A::ArrayWrapper{T}, i::CartesianIndex)::RealWrapper{T} where T<:Real
```

Calling this implementation of `getindex` will record `i::CartesianIndex` as a `Constant` node in
the corresponding _tape_.

After all steps of normal broadcast process, we will get a `R::Array{RealWrapper{T<:Real},N}`. We
employ an operation called _join_ to convert an array of wrappers (`Array{RealWrapper{T},N}`)
into a wrapper of an array (`ArrayWrapper{T,N}`).

### Join Operator

To _join_ an array of wrappers (`Array{RealWrapper{T<:Real},N}`) will result in an individual node
appending to the tape. The number of inputs of a _join_ operation is much greater than the number
of inputs of a _primitive_, to ease the recognition of _join_ operation and the retrieval of
input nodes in the tape, we add a special node type for _join_ operation:

```julia
mutable struct BroadcastJoinNode{T<:Real, N} <: AbstractNode
    id::NodeIndex  # the index of current node in the Tape.
    result::Array{T,N}
    input_indexes::Array{NodeIndex,N}

    # Currently backward-pass relies on `:ref` field to properly sum up vector-Jacobian product.
    ref::Int
end
```

Key points are:

- The element with index `i` of the `result::Array{T<:Real,N}` in a `BroadcastJoinNode` is the
the `result` of the `RealWrapper{T}` got from the computation `f(A[i], B[i], C[i])`.

- The element with index `i` of the `input_indexes::Array{NodeIndex,N}` in a `BroadcastJoinNode`
is the index of the `TapeNode`, which corresponds to the computation `f(A[i], B[i], C[i])`, in
the tape.

- In a `BroadcastJoinNode`, `result` and `input_indexes` have the same dimension and shapes.

### Back-propagation Behavior of Join

_Join_ operations, i.e. `BroadcastJoinNode` in the tape, require properly accumulated _adjoint value_ in the _backward pass_ of AD.

Given:

- the adjoint value `g::AbstractArray{Float64,N}` of a `BroadcastJoinNode{T<:Real,N}`,
- an array index `I::CartesianIndex` for the `input_indexes` field of that `BroadcastJoinNode`,

the adjoint value of the `(input_indexes[I])`-th tape node is `getindex(g, I)`.

