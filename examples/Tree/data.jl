"""
    load_tree_data()

Generate fake tree-like data.
"""
function load_tree_data()
    function rand_layer!(root, depth)
        if depth < 3 || depth < 10 && rand() < 0.5
            for _ in 1:2
                push!(root.children, Tree(OneHotVector(rand(1:VOCAB_NUM), VOCAB_NUM)))
            end
        end
        foreach(root.children) do subtree
            rand_layer!(subtree, depth+1)
        end
    end

    map(1:100) do _
        tree = Tree(OneHotVector(rand(1:VOCAB_NUM), VOCAB_NUM))
        rand_layer!(tree, 1)
        (tree, rand(1:LABEL_NUM))
    end
end
