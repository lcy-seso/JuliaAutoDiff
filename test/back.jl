function test_gradient1()
    z(x, y) = 4x * y + 3x * sin(4y)
    dx(x, y) = 4y + 3 * sin(4y)
    dy(x, y) = 4x + 12x * cos(4y)

    x = 5.
    y = 1.5

    val, grads = gradient(z, 5., 1.5)
    val == z(x, y) && grads[1] == dx(x, y) && grads[2] == dy(x, y)
end

function test_gradient2()
    function f(x)
        a = 2x
        b = 3a
        c = 4a
        b * c
    end
    dx(x) = 96x

    x = 3.
    val, grad = gradient(f, x)
    val == f(x) && grad[1] == dx(x)
end

function test_gradient3()
    function f(x)
        a = exp(x)
        b = a^2
        c = a + b
        d = exp(c)
        e = sin(c)
        d + e
    end

    function df(x)
        a = exp(x)
        b = a^2
        c = a + b
        d = exp(c)
        e = a + 2b
        d * e + cos(c) * e
    end

    x = 0.15
    val, grad = gradient(f, x)
    f(x) == val && isapprox(df(x), grad[1], atol=1e-3)
end

function test_grad_pow()
    f(x, y) = 4x^3 * y^2
    dx(x, y) = 12x^2 * y^2
    dy(x, y) = 8x^3 * y
    x, y = 2.5, 3.5
    val, grad = gradient(f, x, y)
    val == f(x, y) && grad == [dx(x,y), dy(x,y)]
end

@testset "Gradient" begin
    @test test_gradient1()
    @test test_gradient2()
    @test test_gradient3()
    @test test_grad_pow()
end

function test_broadcast_inspect()
    X = [2 3]'  # 2x1 matrix
    Y = [4 5 6] # 1x3 matrix

    # x_1*x_2*...*x_{len(Y)} * y_1*y_2*...*y_{len(X)}
    f(X, Y) = prod(X .* Y)
    dx(X, Y) = [f(X,Y)/x*(length(Y)) for x in X]
    dy(X, Y) = [f(X,Y)/y*(length(X)) for y in Y]
    val, grad = gradient(f, X, Y)
    @test val == f(X, Y) 
    @test grad[1] == dx(X, Y) 
    @test grad[2] == dy(X, Y)
end

function test_broadcast_by_scalar()
    X = [2 3]'  # 2x1 matrix
    Y = [4 5 6] # 1x3 matrix

    multiply(x, y) = true ? x * y : missing # Control flow deprecates code inspection.

    # x_1*x_2*...*x_{len(Y)} * y_1*y_2*...*y_{len(X)}
    f(X, Y) = prod(multiply.(X, Y))
    dx(X, Y) = [f(X,Y)/x*(length(Y)) for x in X]
    dy(X, Y) = [f(X,Y)/y*(length(X)) for y in Y]
    val, grad = gradient(f, X, Y)
    @test val == f(X, Y) 
    @test grad[1] == dx(X, Y) 
    @test grad[2] == dy(X, Y)
end

function test_broadcast_join_scalar()
    multiply(x, y) = true ? x * y : missing # Control flow deprecates code inspection.
    v, g = gradient(multiply, 3, 4)
    @test v == 12
    @test g == [4, 3]
end

using ReverseAD: _reduce_bc_dims

@testset "Broadcast" begin
    @test all(_reduce_bc_dims(ones(3,4), ones(3,4)) .== ones(3,4))
    @test all(_reduce_bc_dims(ones(2,42), ones(2)) .== fill(42, (2,)))
    # Only drop dims not existing in input.
    @test all(_reduce_bc_dims(ones(3,3,3,3), ones(1,3)) .== fill(27, (1,3)))
    @test _reduce_bc_dims(ones(2,3), 42) == 6 # Input is 0-dimensional.

    let (v, g) = gradient(x->sum(x .* 1), [2,3,4])
        @test v == 9
        @test g[1] == [1,1,1]
    end
    let (v, g) = gradient((x,y)->sum(x .* y), [2,3,4], 1)
        @test v == 9
        @test g[1] == [1,1,1]
        @test g[2] == 9
    end
    let (v, g) = gradient(x->sum(x .* [1]), [2,3,4])
        @test v == 9
        @test g[1] == [1,1,1]
    end
    let (v, g) = gradient(x->x .* 3, 42)
        @test v == 126
        @test g[1] == 3
    end

    # Test multiple broadcasts in sequence.
    let (v, g) = gradient(x->let a = x .* 2
            b = a .* 3
            sum(b)
        end, [2,3,4])
        @test v == 54 # (2+3+4)*6
        @test g[1] == [6,6,6]
    end

    test_broadcast_inspect()
    test_broadcast_by_scalar()
end # !testset "Broadcast"
