export SGD, update!

const RealArray = AbstractArray{<:Real}
const RealOrRealArray = Union{Real, AbstractArray{<:Real}}

struct SGD
    learning_rate
end
SGD() = SGD(0.0001)
(opt::SGD)(x, Δ) = x .-= opt.learning_rate .* Δ


update!(opt, w::RealArray, grad::RealArray) = opt(w, grad)

update!(opt, ws::Union{AbstractArray,Tuple}, grads::Union{AbstractArray,Tuple}) =
    map(arg -> update!(opt, arg...), zip(ws, grads))

function update!(opt, w::T, grad) where {T}
    isstructtype(T) && foreach(fieldnames(T)) do fname
        opt(getfield(w, fname), getfield(grad, fname))
    end
end

update!(opt, ::Real, ::Real) = error("Cannot update a Real in-place.")
