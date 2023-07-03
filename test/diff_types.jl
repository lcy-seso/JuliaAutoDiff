using Random:seed!
using Statistics

seed!(1234)

using ReverseAD: Tape, RealWrapper, ArrayWrapper, RealArray

# TODO(ying): enhance the unittest. Current unittest is very weak.
# In current unittest, not many meaningful things are tested!

""" Test Case. """
mutable struct Type1 # This is a differentiable type.
    a::AbstractArray
    b
end
Type1() = Type1(randn(1, 3), rand())

mutable struct Type2
    x::Type1
    y::AbstractArray{Type1, 1}
    z
end
Type2() = Type2(Type1(), [Type1() for _ in 1:3], 5.5)

function func1(val)
    x1 = val[1]  # Type2

    x2 = val[2]  # Non-differentiable type String
    println(x2, ": $(typeof(x2))")

    t1 = sum(x1.x.a) .* val[3].a .+ x1.y[2].b + val[4]
    Statistics.mean(t1) / val[end]
end

# val has a type Type1, but here user's function has to be declared typeless.
func2(val) = Statistics.mean(val.y[1].a .+ val.z)

function test1(val)
    ans1 = func1(val)
    ans2, grads = gradient(func1, val)

    ans1 == ans2
end

function test2(val)
    ans1 = func2(val)
    ans2, grads = gradient(func2, val)

    ans1 == ans2
end

@testset "DiffType" begin
    val = [Type2(), "Non-differentiable type", Type1(), randn(1, 3), rand()]

    @test test1(val)
    @test test2(val[1])
end
