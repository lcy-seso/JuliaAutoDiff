const Mat = AbstractArray{Float64, 2}

# Because our AD will wrap user-defined data type into an internal wrapped type,
# so, user's type HAS TO be declared as mutable.
mutable struct Dense
    W::Mat
    b::Mat
end
Dense(in_dim::Int, out_dim::Int) = Dense(randn(in_dim, out_dim), randn(1, out_dim))
(a::Dense)(x) = tanh.(x * a.W .+ a.b)

# NOTE: When a user-defined function accepts an argument that has a
# user-defined type, this function HAS TO BE declared typeless.
# We DO NOT check wheather user-defined functions accepting user-defined types
# are declared typeless or not, but just rely on dispatch error at runtime.
dnn(weights, x) = foldl((input, w) -> w(input), weights; init=x)

sqnorm(x) = sum(x .* x)
mse(x, y) = sqnorm(x .- y)
