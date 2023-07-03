using ReverseAD: vjp, Tape, add_input!, _sparse_array, are_args_wrapped

@testset "Wrappers" begin

t1 = Tape()
w1 = add_input!(t1, 42)
t2 = Tape()
w2 = add_input!(t2, [1 2 3])

@test are_args_wrapped(4) == false
@test are_args_wrapped(w1) == true
@test are_args_wrapped(w1, w1) == true
@test are_args_wrapped(w1, 1) == true
@test are_args_wrapped(1, w1) == true
@test are_args_wrapped(w1, w2) == false
@test are_args_wrapped(w1, 1, 2, 3) == true
@test are_args_wrapped(1, 2, 3, w1) == true
@test are_args_wrapped(1, 2) == false
@test are_args_wrapped(1, 2, 3) == false

end # !testset "Wrappers

@testset "Operators" begin

let sa = _sparse_array((2,3,4), CartesianIndex(2,1,3), 9)
    @test size(sa) == (2,3,4)
    @test sa[2,1,3] == 9
    @test all(i->i==CartesianIndex(2,1,3)||sa[i]==0, CartesianIndices(size(sa)))

    b = fill(1.5, (2,3,4))
    c = sa .+ b
    @test c[2,1,3] == 10.5
    @test all(i->i==CartesianIndex(2,1,3)||c[i]==1.5, CartesianIndices(size(c)))
end

@testset "getindex" begin
let t = Tape()
    a = add_input!(t, [2 3 4; 5 6 7])
    b = a[1]
    test_node(t[b.nodeidx], getindex, 2, [1,2])

    c = a[CartesianIndex(2,3)]
    test_node(t[c.nodeidx], getindex, 7, [1,4])

    d = a[2, 1]
    test_const_node(t[6], 2)
    test_const_node(t[7], 1)
    test_node(t[d.nodeidx], getindex, 5, [1,6,7])
end

X = [2 3; 
     4 5]
@test vjp(getindex, 3.14#=Δ=#, 4#=primal=#, X, CartesianIndex(2,1))[1] == [0 0; 3.14 0]
@test vjp(getindex, 3.14#=Δ=#, 5#=primal=#, X, 4)[1] == [0 0; 0 3.14]
@test vjp(getindex, [3.14;42]#=Δ=#, [3;5]#=primal=#, X, :, 2)[1] == [0 3.14; 0 42]

end # !testset "getindex"

@testset "prod" begin
X = [2 3; 4 5]
A = vjp(prod, 3.14#=Δ=#, prod(X), X)[1]
@test size(A) == size(X)
@test A == [60 40; 30 24] .* 3.14
X = [2 0 3]
@test vjp(prod, 3.14#=Δ=#, prod(X), X)[1] == [0 6 0] .* 3.14
X = [2 0 3 0 4]
@test vjp(prod, 3.14#=Δ=#, prod(X), X)[1] == [0 0 0 0 0] .* 3.14
@test vjp(prod, 3.14#=Δ=#, 42, 42) == (3.14,)
end #! prod

end # !testset "Operators"
