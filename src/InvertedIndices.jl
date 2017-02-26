module InvertedIndices

export InvertedIndex, Not

struct InvertedIndex{T}
    skip::T
end
const Not = InvertedIndex

# TODO: Remove once Base trues becomes indices-aware
@inline trues(tup) = Base.trues(map(Base.unsafe_length, tup))

@inline function Base.to_indices(A, inds, I::Tuple{InvertedIndex, Vararg{Any}})
    v = trues(spanned_indices(inds, I))
    v[I[1].skip] = false
    to_indices(A, inds, (v, Base.tail(I)...))
end

@inline spanned_indices(inds, I::Tuple{InvertedIndex,Vararg{Any}}) = (Base.uncolon(inds, (:, Base.tail(I)...)),)

NIdx{N} = Union{CartesianIndex{N}, AbstractArray{CartesianIndex{N}}, AbstractArray{Bool,N}}
@inline spanned_indices(inds, I::Tuple{InvertedIndex{<:NIdx{0}},Vararg{Any}}) = ()
@inline spanned_indices(inds, I::Tuple{InvertedIndex{<:NIdx{1}},Vararg{Any}}) = (Base.uncolon(inds, (:, Base.tail(I)...)),)
@inline function spanned_indices{N}(inds, I::Tuple{InvertedIndex{<:NIdx{N}},Vararg{Any}})
    heads, tails = Base.IteratorsMD.split(inds, Val{N})
    (Base.front(heads)..., Base.uncolon((heads[end], tails...), (:, Base.tail(I)...)))
end

end # module
