@testset "Register" begin

# The vjp function should be added into ReverseAD module even if we don't `using ReverseAD` wholly 
# but only import macro `@vjp` into current scope.
using ReverseAD: @vjp

direct_reg(x) = x*x
@vjp direct_reg(g, p, x) = (g*2*x,)
@test ReverseAD.vjp(direct_reg, 3, 99, 5) == (3*2*5,)

lambda_reg(x) = x*x
@vjp lambda_reg = (g,p,x)->(g*2*x,)
@test ReverseAD.vjp(lambda_reg, 3, 99, 5) == (3*2*5,)

delegate_reg(x) = x*x
delegate(g, p, x) = (g*2*x,)
@vjp delegate_reg = delegate
@test ReverseAD.vjp(delegate_reg, 3, 99, 5) == (3*2*5,)

typed_arg_reg(x::Real) = 5
typed_arg_reg(x::AbstractArray) = [2 3;]
@vjp typed_arg_reg(g, p, x::Real) = (78,)
@vjp typed_arg_reg(g, p, x::AbstractArray) = ([36 36;],)
@test ReverseAD.vjp(typed_arg_reg, 99, 99, 2) == (78,)
@test ReverseAD.vjp(typed_arg_reg, 99, 99, [1]) == ([36 36;],)

end