include("model.jl")

using ReverseAD
using Utils

using Random: seed!
seed!(1235)

function get_batch(batch_size, in_dim)
    inputs = randn(batch_size, in_dim)
    lbl = randn(batch_size, 1)
    (inputs, lbl)
end

function train_dnn(batch_num=10, opt=SGD())
    batch_size = 5
    in_dim = 16
    shapes = [(in_dim, 32), (32, 28), (28, 1)]

    weights = Vector{Dense}()
    foreach(shape -> push!(weights, Dense(shape...)), shapes)

    for i in 1:batch_num
        loss, grads = gradient(weights) do w
            inputs, lbl = get_batch(batch_size, in_dim)
            mse(dnn(w, inputs), lbl)
        end
        println("batch ", i, "; loss = ", loss)
        update!(opt, weights, grads[1])
    end
end

train_dnn()
