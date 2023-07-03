using Random: seed!
using Statistics

seed!(1234)

@testset "Gradient Checking" begin

using ReverseAD: numerical_grad, grad_check

f1(x) = x^3
grad1 = numerical_grad(f1, 5)
@test size(grad1) == (1,)
@test isapprox(grad1[1], 75)

f2(a::AbstractMatrix, b::AbstractMatrix) = sum(a*b)
grad2 = numerical_grad(f2, [3 4]', [5 6]) # [[(b1+b2), (b1+b2)], [...]]
@test size(grad2) == (2,)
@test size(grad2[1]) == (2,1)
@test size(grad2[2]) == (1,2)
@test isapprox(grad2[1], [11 11]')
@test isapprox(grad2[2], [7 7])

#=
Derivative w.r.t. a_i:   (b+c)a_i^(b+c-1) * (\prod_{j \neq i} a_j^(b+c))
Derivative w.r.t. b & c: (\prod_i a_i)^(b+c) * ln(\prod_i a_i)
=#
f3(a::AbstractVector, b::Real, c::Real) = prod(a .^ (b+c))
grad3 = numerical_grad(f3, [4,5], 2, 3)
@test size(grad3) == (3,)
@test size(grad3[1]) == (2,)
@test grad3[2] isa Real && grad3[3] isa Real
deriv3a(x) = (2+3) * x^(2+3-1)
deriv3bc(a) = a^(2+3) * log(a)
@test isapprox(grad3[1], [deriv3a(4)*5^5, 4^5*deriv3a(5)])
@test isapprox(grad3[2], deriv3bc(4*5))
@test isapprox(grad3[3], deriv3bc(4*5))

using NNlib: sigmoid

# Each element of list `fds` is a pair of some primal function and its derivative.
fds = [
    sin => cos,
    (x->sin(sin(x))) => x->cos(x)*cos(sin(x)),
    exp => exp,
    log => inv,
    sigmoid => x->sigmoid(x)*(1-sigmoid(x))
]
for fd in fds
    for x in [2.3, 3.14, 5, 42]
        gradx = numerical_grad(fd[1], x)
        @test isapprox(gradx[1], fd[2](x))
    end
end

@test grad_check(Statistics.mean, randn(2, 3, 11, 7))
@test grad_check(sum, randn(1, 13, 7))
@test grad_check(*, randn(7, 11), randn(11, 3))

end # !testset "Gradient Checking"
