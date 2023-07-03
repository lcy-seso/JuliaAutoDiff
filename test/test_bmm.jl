function test_bmm1()
    A = randn(4, 5, 3)
    B = randn(5, 6, 3)

    broadcast_dim = size(A, 3)
    C = bmm(A, B)
    for i in 1:broadcast_dim
        a = A[:, :, i]
        b = B[:, :, i]
        c = C[:, :, i]
       !(isapprox.(a * b, c) |> all) && return false
    end
    return true
end

function test_bmm2()
    A = randn(7, 3, 2)
    B = randn(7, 5, 2)

    broadcast_dim = size(A, 3)
    C = bmm(A, B, transA=true)
    for i in 1:broadcast_dim
        a = A[:, :, i]
        b = B[:, :, i]
        c = C[:, :, i]
       !(isapprox.(a' * b, c) |> all) && return false
    end
    return true
end

function test_bmm3()
    A = randn(3, 5, 11)
    B = randn(7, 5, 11)

    broadcast_dim = size(A, 3)
    C = bmm(A, B, transB=true)
    for i in 1:broadcast_dim
        a = A[:, :, i]
        b = B[:, :, i]
        c = C[:, :, i]
        !(isapprox.(a * b', c) |> all) && return false
    end
    return true
end

function test_bmm4()
    A = randn(7, 2, 1)
    B = randn(11, 7, 1)

    broadcast_dim = size(A, 3)
    C = bmm(A, B, transA=true, transB=true)
    for i in 1:broadcast_dim
        a = A[:, :, i]
        b = B[:, :, i]
        c = C[:, :, i]
        !(isapprox.(a' * b', c) |> all) && return false
    end
    return true
end

@testset "bmm" begin
    @test test_bmm1()
    @test test_bmm2()
    @test test_bmm3()
    @test test_bmm4()
end
