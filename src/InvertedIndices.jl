module InvertedIndices

export InvertedIndex, Not

struct InvertedIndex{T}
    skip::T
end
const Not = InvertedIndex

@inline function Base.to_indices(A, inds, I::Tuple{InvertedIndex, Vararg{Any}})
    v = trues(Base.uncolon(inds, (:, Base.tail(I)...)))
    v[I[1].skip] = false
    to_indices(A, inds, (v, Base.tail(I)...))
end

end # module
