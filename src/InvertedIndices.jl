module InvertedIndices

export InvertedIndex, Not

struct InvertedIndex{T}
    skip::T
end
const Not = InvertedIndex

# nindices(::InvertedIndex) = 1
# nindices{N}(::InvertedIndex{<:AbstractArray{Bool,N}}}) = N
# nindices{N}(::InvertedIndex{<:AbstractArray{CartesianIndex{N}}}}) = N

@inline trues(tup) = Base.trues(map(Base.unsafe_length, tup))

@inline function Base.to_indices(A, inds, I::Tuple{InvertedIndex, Vararg{Any}})
    v = trues(spanned_indices(inds, I))
    v[I[1].skip] = false
    to_indices(A, inds, (v, Base.tail(I)...))
end

@inline spanned_indices(inds, I::Tuple{InvertedIndex,Vararg{Any}}) = (Base.uncolon(inds, (:, Base.tail(I)...)),)
@inline spanned_indices(inds, I::Tuple{InvertedIndex{<:AbstractVector},Vararg{Any}}) = (Base.uncolon(inds, (:, Base.tail(I)...)),)
@inline function spanned_indices{N}(inds, I::Tuple{InvertedIndex{<:AbstractArray{Bool,N}},Vararg{Any}})
    heads, tails = Base.IteratorsMD.split(inds, Val{N})
    (Base.front(heads)..., Base.uncolon((heads[end], tails...), (:, Base.tail(I)...)))
end

end # module
