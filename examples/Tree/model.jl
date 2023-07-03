using NNlib
using Utils

mutable struct Weights
    embedding::AbstractMatrix
    left_weights::AbstractMatrix
    right_weights::AbstractMatrix
    dense_weights::AbstractMatrix
end

function run_tree(sentence::Tree{<:AbstractVector}, w::Weights)
    function encode_tree(tree::Tree{<:AbstractVector})
        if (isempty(tree.children))
            return w.embedding * tree.value
        else
            l = encode_tree(tree.children[1])
            r = encode_tree(tree.children[2])
            return tanh.(w.left_weights*l .+ w.right_weights*r)
        end
    end

    softmax(w.dense_weights * encode_tree(sentence))
end
