<!-- TOC -->

- [Register primitive and its derivatives](#register-primitive-and-its-derivatives)
    - [Primitive, derivatives and reverse accumulation](#primitive-derivatives-and-reverse-accumulation)
    - [Requirement of primitive](#requirement-of-primitive)
        - [Example of $vjp(f)$](#example-of-vjpf)
    - [API to register $vjp(f)$](#api-to-register-vjpf)

<!-- /TOC -->

# Register primitive and its derivatives

As stated in [Data Types and Concepts](./tape.md), tape-based AD transforms a piece of program
code into a _tape_ by tracing the execution of the program. The resulted tape consists of tape
nodes, and every tape node stores an intermediate result within the traced program and a
_primitive_ producing it. [All tape nodes form a _directed acyclic graph_ (hereafter
DAG)](./tape.md#tape-node).

A _primitive_ is an unary/binary arithmetic function, such as $sin(x)$ or $pow(x,y)$. An
arithmetic function must be properly _registered_ for itself to be recognized as _primitive_.

Tape-based AD employs [_reverse accumulation_][ReverseAccum] to calculate the gradient of the
original program. Reverse accumulation requires that not only the _primitive_, but also 1) the
partial derivatives of the primitive w.r.t. all its inputs and 2) how _chain rule_ is applied to
those derivatives must be registered. This document describes the design of the registration.

## Primitive, derivatives and reverse accumulation
According to [Reverse Accumulation][ReverseAccum], for each tape node $w$, we have:
1. the associated primitive $f$ (assuming $f$ is binary and has inputs $x_1, x_2$),
1. the intermediate result values $\bar x_1, \bar x_2$ stored in its input/parent nodes,
1. the _**adjoint value**_ $\bar{w}$.

Then for each its $i$-th input node, node $w$ contributes a quantity of...

> Formula-$(1)$:
>
> $$ \bar{w} \cdot \frac{\partial f}{\partial x_i} (\bar x_1, \bar x_2) $$

...to the adjoint value of that input node.

Before we apply the process above to node $w$, all contributed quantities from $w$'s children nodes
must be summed up to get the correct adjoint value $\bar w$. When we traverse the DAG (starting
from the node representing the final primal result) in the [reverse
phrase](./tape.md#reverse-phrase-append-derivative-computation-into-tape), we ensure that the adjoint
value of every node is properly accumulated from all its direct children nodes.

The adjoint value $\bar w = 1$ if $w$ is the node that represents the final result of the
original program.

## Requirement of primitive
To well support the process of [reverse accumulation][ReverseAccum], we require that each
primitive $f$ is also provided with an additional function $vjp(f)$. This function must satisfy
requirements below:

1. It accepts $(2 + arity(f))$ parameters, $arity(f)$ is the number of parameters of function
$f$.

    The first parameter of $vjp(f)$ is the adjoint value for the node associated with $f$, the
    second parameter is the result of evaluating $f$ in the forward pass stored in that node,
    other parameters are the same as parameters passed to $f$ when generating that node.

1. It returns an ordered tuple. The $i$-th element of the tuple is the product
of the adjoint value and the result of derivative w.r.t. the $i$-th input of the
primitive.

### Example of $vjp(f)$
An example of such a function for primitive `pow(x, y)` looks like:
```julia
vjp_pow(g, p, x, y) = ( g * y*x^(y-1), g * p *ln(x) )  # p == x^y
```

> Note: $\partial(x^y)/\partial x = y \cdot x^{y-1}$ , $\partial(x^y)/\partial y = x^y \cdot ln(x)$

> **Remark:**
> In Formula-$(1)$ we see that $\bar{w}$ is **multiplied** with the result of the partial derivative.
> This formula only describes the mathematical concept in the process of reverse accumulation.
> When implementing Formula-$(1)$ in code, allowing registrars to provide their implementation abstracting
> the multiplication can lead to more efficient and less redundant code, especially when
> implementing complex array-related functions for $vjp(f)$. For example, the registrar may prefer a
> more efficient function `matmul` to default `:*` for matrix multiplication.
>
> Returning a tuple also speeds up the calculation, since intermediate results can
> be efficiently reused within $vjp(f)$. Accepting an additional parameter for the result of the
> primitive in the forward pass also saves computation for some operations.

## API to register $vjp(f)$
To make it easier for the registrar to register, and also to make it easier for AD developer to
lookup $vjp(f)$ for a primitive $f$, we provide a macro `@vjp`.

For example:
```julia
using ReverseAD
@vjp pow(g, p, x, y) = ( g * y*x^(y-1), g * p * ln(x) )
```

Such a call to `@vjp` will generate an internal function in AD system:
```julia
vjp(::typeof(pow), g, p, x, y) = ( g * y*x^(y-1), g * p * ln(x) )
```

Besides example above, this AD framework accepts other valid forms of registration:
```julia
@vjp exp = (g, p, x)->( g*p ,)  # p == exp(x)
# or
vjp_pow(g,p,x,y) = ...
@vjp pow = vjp_pow
```

[ReverseAccum]: https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation
