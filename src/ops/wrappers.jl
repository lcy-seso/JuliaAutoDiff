struct RealWrapper{T<:Real} <: Real
    val::T
    tape::Tape
    nodeidx::NodeIndex
end

struct ArrayWrapper{T<:Real, N, A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    val::A
    tape::Tape
    nodeidx::NodeIndex
end

# Not-to-trace array-related functions.
for f in (:[
    Base.size,
    Base.axes,
    Base.ndims
]).args
    @eval @inline $f(a::ArrayWrapper, xs...) = $f(a.val, xs...)
end

# Aligned with `Base.getindex`.
Base.eltype(::ArrayWrapper{T}) where T = RealWrapper{T}

const Wrapper = Union{RealWrapper, ArrayWrapper}

wrapper(val::Real, tape::Tape, id::NodeIndex) = RealWrapper(val, tape, id)
wrapper(val::RealArray, tape::Tape, id::NodeIndex) = ArrayWrapper(val, tape, id)

val(x::Wrapper) = x.val
val(x) = x

nodeidx(x::Wrapper) = x.nodeidx

#TODO This is just simple implementation.
Base.show(io::IO, w::Wrapper) = println(io, typeof(w), "\n", val(w))

# For display in REPL. Or by default REPL uses functions of interface
# `AbstractArray` that haven't been implemented now.
Base.show(io::IO, ::MIME"text/plain", w::ArrayWrapper) = show(io, w)

"""
    are_args_wrapped(args...)::Bool

At least one in `args` is a wrapper. All wrappers must be in the same tape.
"""
@inline are_args_wrapped(::Wrapper) = true
@inline are_args_wrapped(x) = false
@inline are_args_wrapped(x::Wrapper, y::Wrapper) = x.tape === y.tape
@inline are_args_wrapped(x::Wrapper, y) = true
@inline are_args_wrapped(x, y::Wrapper) = true
@inline are_args_wrapped(x, y) = false
function are_args_wrapped(args...)
    wrappers = filter(x->(x isa Wrapper), collect(args))
    isempty(wrappers) && return false
    all(wrappers) do w
        w.tape === wrappers[1].tape
    end
end

#==============================================================================

Given a value that has a user-defined type, iteratively traverse the value's
type definition, and wrap every differentiable field to have an
iternally-defined wrapper type.

Rules for differentiable types that are supported by current implementations:

1. Real and RealArray(AbstractArray{Real}) are primitive differentiable types.
2. For a composition type (struct) A, A is differentiable, if at least one field
of A has a type satisfying rule 1,
3. For a composition type (struct) A, A is differentiable, if at least one field
of A has a type satisfying rule 1 and 2,
4. For container type, current implementation ONLY supports Vector{T}. T
should satisfy rule 1, 2, 3.
5. Differentiable fields of a struct or differentialb elements in an Array will
be wrapped into an internal wrapper type which is not exposed to users.
6. Undifferentiable types are unaffected at all. Gradient computation only
wraps differentiable values and tracks operations on them.

TODO(ying) self-referential type is not supported and tested yet.
===============================================================================#
"""
    add_input!(tape::Tape, val::RealOrRealArray)

When input is our primitive differentiable type Real or RealArray, this method
is called. It directly add input to internal tape representation, and wrap
user's original input to have an internal wrappped type.
"""
function add_input!(tape::Tape, val::RealOrRealArray)
    id = length(tape) + 1
    push!(tape, Input(id, val))

    # This is for usage that adding input into tape by directly calling this
    # method instead of calling `trace!`,
    isempty(tape.inputs) && push!(tape.inputs, [])

    push!(tape.inputs[end], id)

    wrapper(val, tape, id)
end

"""
    add_input!(tape::Tape, val::Any)

When input is a user-defined type, this method is called. This method
iteratively introspects the value's type composition until a primitive
differentiable is reached. This proces is similar to traversing a tree by a DFS
order. Once a primitive differentiable type (Real or RealArray is reached), it
will be wrapped into an internally wrapped type.
"""
function add_input!(tape::Tape, val)
    wrapval!(tape, val)
    val
end


"""
    wrapval!(tape::Tape, val)

Values will be dispatched to the most specific method.
Real and RealArray is the primitive differentiable type for AD.
"""
wrapval!(tape::Tape, val::RealOrRealArray) = add_input!(tape, val)

"""
    wrapval!(tape::Tape, val::AbstractArray{T})

This method handles container type Array whose element types is not
primitive differentiable types, but user-defined types.
"""
function wrapval!(tape::Tape, val::AbstractArray)
    for (id, v) in enumerate(val)
        v isa RealOrRealArray && (val[id] = add_input!(tape, v); continue)
        wrapval!(tape, v)
    end
end

"""
    wrapval!(tape::Tape, val::Any)

This method handles user-defined composition type.
"""
function wrapval!(tape::Tape, val)
    vtype = typeof(val)
    isstructtype(vtype) && foreach(fn -> wrapval!(tape, val, fn),
                                   fieldnames(vtype))
end

wrapval!(tape, val::RealArray, fieldname::Symbol) = wrapval!(tape, val)
function wrapval!(tape::Tape, val, fieldname::Symbol)
    fval = getfield(val, fieldname)
    fval isa RealOrRealArray ?
        setfield!(val, fieldname, add_input!(tape, fval)) :
        wrapval!(tape, fval)
end
