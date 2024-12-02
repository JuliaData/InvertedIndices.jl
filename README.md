# InvertedIndices

[![Build Status](https://github.com/JuliaData/InvertedIndices.jl/workflows/CI/badge.svg)](https://github.com/JuliaData/InvertedIndices.jl/actions?query=workflow%3ACI+branch%3Amain)
[![Code coverage](https://codecov.io/gh/JuliaData/InvertedIndices.jl/graph/badge.svg?token=D1B9JKlQG5)](https://codecov.io/gh/JuliaData/InvertedIndices.jl)

This very small package just exports one type: the `InvertedIndex`, or `Not`
for short. It can wrap any supported index type and may be used as an index
into any `AbstractArray` subtype, including OffsetArrays.

Upon indexing into an array, the `InvertedIndex` behaves like a 1-dimensional
collection of the indices of the array that are not in the index it wraps. Bounds
are checked to ensure that the excluded index is within the bounds of the array
— even though it is skipped. The `InvertedIndex` behaves like a
1-dimensional collection of its inverted indices. If the excluded index spans multiple
dimensions (like a multidimensional logical mask or `CartesianIndex`), then the
inverted index will similarly span multiple dimensions.

```julia
julia> using InvertedIndices

help?> InvertedIndex
search: InvertedIndex InvertedIndices

  InvertedIndex(idx)
  Not(idx)

  Construct an inverted index, selecting all indices not in the passed idx.

  ...

julia> A = reshape(1:12, 3, 4)
3×4 Base.ReshapedArray{Int64,2,UnitRange{Int64},Tuple{}}:
 1  4  7  10
 2  5  8  11
 3  6  9  12

julia> A[Not(2), Not(2:3)]
2×2 Array{Int64,2}:
 1  10
 3  12

julia> A[Not(iseven.(A))]
6-element Array{Int64,1}:
  1
  3
  5
  7
  9
 11

julia> A[Not(:)]
0-element Array{Int64,1}
```
