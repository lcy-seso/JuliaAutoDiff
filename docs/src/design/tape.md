<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Data Types and Concepts](#data-types-and-concepts)
	- [Background](#background)
	- [Tape Node](#tape-node)
	- [Tape](#tape)
	- [Gradient Computation](#gradient-computation)
		- [Forward Phrase: Construct Tape Representation](#forward-phrase-construct-tape-representation)
			- [Wrapper Types](#wrapper-types)
				- [About Type Parameter T](#about-type-parameter-t)
	- [Reverse Phrase: Append Derivative Computation into Tape](#reverse-phrase-append-derivative-computation-into-tape)
	- [Compute High-order Gradients](#compute-high-order-gradients)
		- [Example of local tracing](#example-of-local-tracing)
		- [Notes on local tracing](#notes-on-local-tracing)

<!-- /TOC -->

# Data Types and Concepts

## Background

Auto differentiation (AD) relies on the decomposition of a program that implements a differentiable function $f$ into elementary operations for which derivatives are known and to which chain rule can be applied. We call these elementary operations _**primitives**_.

To differentiate a complicated program, it is required to register a set of _**primitives**_ and their derivative functions. Specifically:
* A primitive $f$ calculates the output with given inputs.
* The gradient function $\bigtriangledown f$ of $f$ calculates gradients w.r.t each input of $f$.

Tape-based reverse mode AD requires the storage of the intermediate variables as well as the differentiable operations that produced them in a linear data structure known as a [Wengert list](http://www.cs.cmu.edu/~wcohen/10-605/notes/autodiff.pdf), or "Tape".

This document describes concepts which are essential to a tape-based AD. Note that some concepts in this document do not correspond to exact implementations in the code base.

## Tape Node

`AbstractNode` is the supertype of all kinds of tape nodes.
```julia
abstract type AbstractNode end
```

1. A tape node stores one intermediate differentiable value and the primitive producing it. A primitive is either a **unary** or a **binary** function that returns **ONLY** one output.
1. After a tape node is created, the recorded _primitive_ that creates this tape node is immutable.
1. A constant node only has literal value but primitive that produces this value is left empty.
1. A value stored in one tape node `A` can be an operand to another primitive, and evaluating this primitive will produce a new tape node `B`. We call `A` parent of `B`.
1. Apparently, there are data dependencies among tape nodes, forming a DAG (Directed Acyclic Graph, short for DAG below).

## Tape

Tape definition:
```julia
struct Tape
    nodes::Vector{AbstractNode}
    # Nodes for parameters to be differentiate in the original program.
    # They all must be constant nodes.
    inputs::Vector{NodeIndex}
    outputs::Vector{NodeIndex}
end
```

After execution, a program is linearized into an internal tape representation. In essence, a tape is a growable array(`Vector` in JPL), and elements of which are tape nodes.

1. Vector indices are used as unique identifiers of tape nodes in the tape.
1. Every time a tape node is created, indices of its parents are also stored.

## Gradient Computation

There are two phrases for a tape-based AD:

|Phrase|Description|Data types involved|
|---|---|---|
|Forward pass|Construct internal _tape_ representation and calculate the return value.|`Tape, Node, Wrappers`|
|Backward pass|Construct derivative computation and calculate gradients|`Tape, Node, ...`|

### Forward Phrase: Construct Tape Representation

1. The original program is executed and then linearized into a series of elementary primitives execution in the forward phase.
1. JPL's built-in arithmetic functions for the scalar/array are overloaded so that they additionally perform a tracing operation.
1. Executing an overloaded primitive will append a tape node into the tape.

We call all tape nodes created in the forward phrase the _**primal program**_.

A downside of the tracing-based AD is that it unrolls primitive executions through control-flows. After execution, the control-flow constructs in the original program are wiped out. Only one fixed primitive execution trace is recorded. Primitive traces in those unreachable codes in the original program are ignored.

Notes:

1. Once a tape is constructed after the forward phase, users are not allowed to change it.
1. A user is only allowed to change the original program and then re-execute it to construct a new tape representation.
1. Registered primitives are the minimal unrolled unit. A _primitive_ can be implemented using any control-flow construct or recursion if the registrar guaranteed the derivative function is also implemented correctly.

    Here we use `max` as an example. `max(x, y)` and derivative function of `max(x, y)` w.r.t its two inputs are all implemented with an `if` construct:
    ```julia
    julia> DiffRules.diffrule(:Base, :max, :a, :b)
    (:(if a > b one(a)  else zero(a) end),
    :(if a > b zero(b) else one(b)  end))
    ```

#### Wrapper Types

To enable tracing operations, we provides _wrapper types_ for JPL builtin scalar and array type:

```julia
struct RealWrapper{T<:Real>} <: Real end
struct ArrayWrapper{T<:Real, N, A<:AbstractArray{T,N}} <: AbstractArray{T,N}
```

Overloaded methods have to be added for every registered _primitive_. For example for a primitive `*(x, y)`, the following methods should be added:

```julia
*(x, y::RealWrapper)
*(x::RealWrapper, y)
*(x::RealWrapper, y::RealWrapper)
```

Overloaded methods are only added to registered primitives. As a result, to automatically differentiate users' programs, a user has to guarantee his program is able to be decomposed into registered primitive otherwise runtime dispatch error will occur when invoking a function with wrapper types.

For example:

```julia
length(a::Array)         # Dispatch error!
length(a::AbstractArray) # Good
func(a::Float64)         # Dispatch error!
func(a::Real)            # Good
```

##### About Type Parameter T

It is useful to allow a range of sub-`Real` types like `Int32, Float64, Bool...`. If we convert all sub-`Real` to `Float64`, we may fail to support operations like array indexing, since the index is required to be `Int`.

A range of subtypes also allows proper type conversion or type promotion.

## Reverse Phrase: Append Derivative Computation into Tape

The reverse phrase is a BFS traverse of tape nodes in a given primal program. After the reverse phrase, new tape nodes will be appended into an existing tape. We call tape nodes created in reverse phrase the _**adjoint program**_.

## Compute High-order Gradients

$N$-order gradient is obtained by iteratively applying the gradient computation to an already-evaluated $(N-1)$-order gradient function, represented by the internal tape representation returned obtained from the last invocation of the gradient computation interface. Higher order gradients required operations written into the tape in reverse phrase need also to be differentiable. This problem is also known as reverse-over-reverse that is a challenge for any taped-based AD.

Let's take the second-order gradient for example. The primal program and the adjoint program is taken as a new primal program, as a result, derivative functions in originally an adjoint program will be regarded as a primitive now. There are two possibilities: (1) the derivative function is not a simple registered primitive, but complicated combinations of several primitives; (2) the derivative function contains some unregistered primitives. For the latter case, the high-order gradient computation will just fail and throw an "undifferentiable primitive error".

We only consider the first case. We use the derivative of _sigmoid_ as an example, which is implemented as follows:

$$ \frac{\partial{\sigma}}{\partial{x}}(x) = \sigma(x)\left(1-\sigma(x)\right)$$

Without inspecting into AST, it is impossible to get information of such a decomposition through merely analyzing the existing tape. To address this problem, we propose to **locally trace** the derivative again to get a _tape_ representation made up of registered primitives.

The API for local tracing is:
```julia
Vector{AbstractNode} locally_trace(
    derivative::Expr,
    out_gradient::AbstractNode,
    inputs::Vector{Union{Real,AbstractArray,AbstractNode}})
```

### Example of local tracing
For example, given a tape node that represents a computation of `sigmoid` in the forward-pass,
```c
Node #42 {
     op=sigmoid
     input=[#16]
}
```
locally tracing the derivative of `sigmoid` will produce the following new tape nodes:
```c
Node #56 {  // 1-sigmoid(x)
     op=minus
     input=[Const(1), #42] // Reuse #42, which stands for the result of sigmoid
}
Node #57 {  // sigmoid(x)*(1-sigmoid(x))
     op=multiply
     input=[#42, #56] // Reuse #42.
}
Node #58 {  // Δ * sigmoid(x)*(1-sigmoid(x))
     op=multiply
     input=[#Δ/*outer gradient*/, #57]
}
```

### Notes on local tracing

1. Local tracing unrolls control flow in the derivatives.

    Just like the fact that tape-based AD traces the original program by executing it, local
    tracing executes the derivatives of a function, too. If there is any control flow (e.g.
    `if-else`, `while` loop), during the execution of the derivatives, the control flow will be
    unrolled.

    For example, recall how `DiffRules` defines derivatives for `max`:
    ```julia
    julia> DiffRules.diffrule(:Base, :max, :a, :b)
    (:(if a > b one(a)  else zero(a) end),  # Derivative w.r.t. a
     :(if a > b zero(b) else one(b)  end))  # Derivative w.r.t. b
    ```

    If local tracing is applied to one of derivatives of `max`, only one branch of `if` construct
    in the derivative will be reached and recorded, and the other branch is ignored.
