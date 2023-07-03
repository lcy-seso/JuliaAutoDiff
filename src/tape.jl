const NodeIndex = Int
const EMPTY_INDEX = Vector{NodeIndex}() # Singleton

const RealOrRealArray = Union{Real,AbstractArray{<:Real}}
const RealArray = AbstractArray{<:Real}

abstract type AbstractNode end

"""
    Constant

A constant node stores a non-differentiate intermediate value.
"""
mutable struct Constant{T} <:AbstractNode
    id::NodeIndex  # the index of curent node in the Tape.
    result::T

    Constant(id::NodeIndex, val::T) where T = new{T}(id, val)
end

Base.show(io::IO, node::Constant) =
    println(io, typeof(node), "{\n\tid:  ", node.id,
            "\n\tvalue : ", node.result, "\n}")

"""
    Input

An input node stores a variable to be differentiated. An input node has no
parents and is conceptually produced by a nullary operation.
"""
struct Input{T<:RealOrRealArray} <:AbstractNode
    id::NodeIndex  # the index of curent node in the Tape.
    result::T

    Input(id::NodeIndex, val::T) where {T<:RealOrRealArray} = new{T}(id, val)
end

function Base.show(io::IO, node::Input)
    println(io, typeof(node), " {")
    println(io, "\tid : ", node.id)
    println(io, "\tvalue : ", node.result, "\n}")
end

"""
    TapeNode

A TapeNode stores a differentiate intermediate value and operation produced it.
"""
mutable struct TapeNode{T<:RealOrRealArray} <: AbstractNode
    id::NodeIndex  # the index of curent node in the Tape.
    operation::Union{Function, Nothing}
    result::T

    # Tape nodes in the primal program have at most 2 parents, but
    # nodes in the adjoint program have at most 4 parents(adjoint value, forward's
    # output and all forward's inputs)
    parentidx::Vector{NodeIndex}

    # To avoid circular reference, we cannot guard T is NOT wrapper here.
    TapeNode(id::NodeIndex, x::T) where T<:RealOrRealArray = new{T}(id, nothing, x, EMPTY_INDEX)
    TapeNode(id::NodeIndex, f::Function, x::T, pids::Vararg{NodeIndex}) where T<:RealOrRealArray =
        new{T}(id, f, x, collect(pids))
end

function Base.show(io::IO, node::AbstractNode)
    println(io, typeof(node), "{")
    println(io, "\tid : ", node.id)
    println(io, "\tvalue : ", node.result)
    println(io, "\toperation : ", node.operation)
    println(io, "\tparents : ", join(node.parentidx, " "), "\n}")
end

@inline parents(n::Union{Constant, Input}) = Vector{NodeIndex}()
@inline parents(n::TapeNode) = n.parentidx
@inline value(n::AbstractNode) = n.result  # FIXME Dangerous for primary type.

"""
    Tape

Tape is the internal representation of a differentiable function.
Every time `gradient` is invoked, one Tape will be constructed to store all
intermediate differentiable values.
"""
struct Tape
    nodes::Vector{AbstractNode}
    # parameters to be differentated.
    inputs::Vector{Vector{NodeIndex}}
    outputs::Vector{NodeIndex}

    # This field is defined ONLY for tape nodes in the primal program.
    # It is initialized after tracing to the outdegree of a tapenode, and then
    # is decreased in the reverse sweep.
    ref::Vector{Int}

    Tape() = new(Vector{AbstractNode}(),
                 Vector{Vector{NodeIndex}}(),  #= inputs =#
                 Vector{NodeIndex}(), Vector{Int}())
end

get_ref(tape::Tape, n::AbstractNode) = n.id > outputid(tape) ? 
    error("`ref` is only counted for tape node in primal program.") :
    tape.ref[n.id]

function inc_ref!(tape::Tape, n::AbstractNode)
    n.id > outputid(tape) &&
        error("`ref` is only counted for tape node in primal program.")
    tape.ref[n.id] += 1
end

function Tape(t::Tape)
    # Create a new Tape from an existing non-empty one. The existing tape nodes
    # are shallow copied.
    error("Not Implemented yet.")
end

@forward Tape.nodes (
    Base.size,
    Base.length,
    Base.getindex,
    Base.setindex!,
    Base.push!,
    Base.firstindex,
    Base.lastindex,
    Base.isempty,
    Base.iterate)

@inline function output(tape::Tape)
    length(tape.outputs) == 1 && return tape[first(tape.outputs)]
    isempty(tape.outputs) && error("No output node.")
    error("Not Implemented.")
end
@inline outputid(t::Tape) = (out = output(t) ; out.id)

function Base.show(io::IO, tape::Tape)
    println(io, "\ninputs : ", tape.inputs)
    println(io, "outputs : ", tape.outputs)
    println(io, "visiting counter : ", join(tape.ref, ", "))
    foreach(node -> Base.show(io, node), tape)
end
