include("model.jl")
include("data.jl")

using ReverseAD
using Utils
using Random: seed!

seed!(42)

function train(dataset, opt, w::Weights)
    for (i, sample) in enumerate(dataset)
        loss(w) = cross_entropy(
            OneHotVector(sample[2], LABEL_NUM),
            run_tree(sample[1], w))

        l, grads = gradient(loss, w)
        @show l
        update!(opt, w, grads[1])
    end
end

LABEL_NUM = 6
VOCAB_NUM = 1000
HIDDEN_DIM = 128

w = Weights(
    glorot_normal(HIDDEN_DIM, VOCAB_NUM),
    glorot_normal(HIDDEN_DIM, HIDDEN_DIM),
    glorot_normal(HIDDEN_DIM, HIDDEN_DIM),
    glorot_normal(LABEL_NUM, HIDDEN_DIM)
)
train(load_tree_data(), SGD(), w)
