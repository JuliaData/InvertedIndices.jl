module InvertedIndices

export InvertedIndex, Not

using Base: tail

struct InvertedIndex{S}
    skip::S
end
const Not = InvertedIndex
# Support easily inverting multiple indices without a temporary array in Not([...])
InvertedIndex(i₁::Integer, i₂::Integer, iₓ::Integer...) = InvertedIndex(TupleVector((i₁, i₂, iₓ...)))

"""
    InvertedIndex(idx)
    Not(idx)

Construct an inverted index, selecting all indices not in the passed `idx`.

Upon indexing into an array, the `InvertedIndex` behaves like a 1-dimensional
collection of the indices of the array that are not in `idx`. Bounds are
checked to ensure that all indices in `idx` are within the bounds of the array
— even though they are skipped. The `InvertedIndex` behaves like a
1-dimensional collection of its inverted indices. If `idx` spans multiple
dimensions (like a multidimensional logical mask or `CartesianIndex`), then the
inverted index will similarly span multiple dimensions.
"""
InvertedIndex, Not


# A very simple and primitive static array to avoid allocations for Not(1,2,3) while fulfilling the indexing API
struct TupleVector{T<:Tuple} <: AbstractVector{Int}
    data::T
end
Base.size(::TupleVector{<:NTuple{N}}) where {N} = (N,)
@inline function Base.getindex(t::TupleVector, i::Int)
    @boundscheck checkbounds(t, i)
    return @inbounds t.data[i]
end

# Like Base.LogicalIndex, the InvertedIndexIterator is a pseudo-vector that is
# just used as an iterator and does not support getindex.
struct InvertedIndexIterator{T,S,P} <: AbstractVector{T}
    skips::S
    picks::P
end
InvertedIndexIterator(skips, picks) = InvertedIndexIterator{eltype(picks), typeof(skips), typeof(picks)}(skips, picks)
Base.size(III::InvertedIndexIterator) = (length(III.picks) - length(III.skips),)

@inline Base.iterate(I::InvertedIndexIterator) = iterate(I, (iterate(I.skips), iterate(I.picks)))
Base.iterate(I::InvertedIndexIterator, ::Tuple{Any, Nothing}) = nothing
@inline function Base.iterate(I::InvertedIndexIterator, (skipitr, pickitr))
    while should_skip(skipitr, pickitr)
        skipitr = iterate(I.skips, tail(skipitr)...)
        pickitr = iterate(I.picks, tail(pickitr)...)
        pickitr === nothing && return nothing
    end
    return (pickitr[1], (skipitr, iterate(I.picks, tail(pickitr)...)))
end
Base.collect(III::InvertedIndexIterator) = [i for i in III]

should_skip(::Nothing, ::Any) = false
should_skip(s::Tuple, p::Tuple) = _should_skip(s[1], p[1])
_should_skip(s, p) = s == p
_should_skip(s::Integer, p::CartesianIndex{1}) = s == p.I[1]
_should_skip(s::CartesianIndex{1}, p::Integer) = s.I[1] == p

@inline Base.checkbounds(::Type{Bool}, A::AbstractArray, I::InvertedIndexIterator{<:Integer}) =
    checkbounds(Bool, A, I.skips) && eachindex(IndexLinear(), A) == eachindex(IndexLinear(), I.picks)
@inline Base.checkbounds(::Type{Bool}, A::AbstractArray, I::InvertedIndexIterator) = checkbounds(Bool, A, I.skips) && axes(A) == axes(I.picks)
@inline Base.checkindex(::Type{Bool}, indx::AbstractUnitRange, I::InvertedIndexIterator) = checkindex(Bool, indx, I.skips) && (indx,) == axes(I.picks)
@inline Base.checkindex(::Type{Bool}, inds::Tuple, I::InvertedIndexIterator) = checkindex(Bool, indx, I.skips) && inds == axes(I.picks)

@inline Base.ensure_indexable(I::Tuple{InvertedIndexIterator, Vararg{Any}}) = (collect(I[1]), Base.ensure_indexable(tail(I))...)

# This is a little hacky, but we display InvertedIndexIterators like the `Not`s they come from
function Base.show(io::IO, I::InvertedIndexIterator)
    print(io, "Not(")
    show(io, I.skips)
    print(io, ")")
end

# Inverted indices must be sorted and unique to ensure that iterating over
# them and the axes simultaneously will work appropriately. Doing this fully
# generically is a challenge. It's a little annoying to need to take a pass
# through the inverted index before actually doing the indexing.
uniquesort(A::AbstractArray) = uniquesort(vec(A))
uniquesort(A::DenseVector) = issorted(A, lt=(<=)) ? A : (unique! ∘ sort)(A)
uniquesort(A::AbstractVector) = issorted(A, lt=(<=)) ? A : (unique ∘ sort)(A)
uniquesort(r::AbstractRange) = step(r) > 0 ? r : step(r) == 0 ? r[end:end] : reverse(r)
uniquesort(A::Base.LogicalIndex) = A
uniquesort(x) = x

@inline function Base.to_indices(A, inds, I::Tuple{InvertedIndex, Vararg{Any}})
    new_indices = to_indices(A, inds, (I[1].skip, tail(I)...))
    skips = uniquesort(new_indices[1])
    picks = spanned_indices(inds, skips)[1]
    return (InvertedIndexIterator(skips, picks), tail(new_indices)...)
end

struct ZeroDArray{T} <: AbstractArray{T,0}
    x::T
end
Base.size(::ZeroDArray) = ()
Base.getindex(Z::ZeroDArray) = Z.x

# Be careful with CartesianIndex as they splat out to a variable number of new indices and do not iterate
function Base.to_indices(A, inds, I::Tuple{InvertedIndex{<:CartesianIndex}, Vararg{Any}})
    skips = ZeroDArray(I[1].skip)
    picks, tails = spanned_indices(inds, skips)
    return (InvertedIndexIterator(skips, picks), to_indices(A, tails, tail(I))...)
end

# Either return a CartesianRange or an axis vector
@inline spanned_indices(inds, ::Any) = inds[1], tail(inds)
const NIdx{N} = Union{CartesianIndex{N}, AbstractArray{CartesianIndex{N}}, AbstractArray{Bool,N}}
@inline spanned_indices(inds, ::NIdx{0}) = CartesianIndices(()), inds
@inline spanned_indices(inds, ::NIdx{1}) = inds[1], tail(inds)
@inline function spanned_indices(inds, ::NIdx{N}) where N
    heads, tails = Base.IteratorsMD.split(inds, Val(N))
    return CartesianIndices(heads), tails
end

# It's possible more indices were specified than there were axes
@inline spanned_indices(inds::Tuple{}, ::Any) = Base.OneTo(1), ()
@inline spanned_indices(inds::Tuple{}, ::NIdx{0}) = CartesianIndices(()), ()
@inline spanned_indices(inds::Tuple{}, ::NIdx{1}) = Base.OneTo(1), ()
@inline spanned_indices(inds::Tuple{}, ::NIdx{N}) where N = CartesianIndices(ntuple(i->Base.OneTo(1), N)), ()

# This is an interesting need — we need this because otherwise indexing with a
# single multidimensional boolean array ends up comparing a multidimensional cartesian
# index to a linear index. Does this need addressing in Base, too?
@inline Base.to_indices(A, I::Tuple{Not{<:NIdx{1}}}) = to_indices(A, (eachindex(IndexLinear(), A),), I)
@inline Base.to_indices(A, I::Tuple{Not{<:NIdx}}) = to_indices(A, axes(A), I)
# Arrays of Bool are even more confusing as they're sometimes linear and sometimes not
@inline Base.to_indices(A, I::Tuple{Not{<:AbstractArray{Bool, 1}}}) = to_indices(A, (eachindex(IndexLinear(), A),), I)
@inline Base.to_indices(A, I::Tuple{Not{<:Union{Array{Bool}, BitArray}}}) = to_indices(A, (eachindex(A),), I)

end # module
