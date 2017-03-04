using InvertedIndices
using Base.Test
using OffsetArrays

@testset "0-d" begin
    A = fill(1)
    @test A[Not(A.==1)] == []
    @test A[Not(CartesianIndex())] == []
    @test A[Not(A.==2)] == [1]
    A[Not(A.==2)] = 0
    @test A[] == 0
end

@testset "1-d readonly" for A in (-10:13, reshape(-10:13,2,:), reshape(-10:13,3,2,:))
    @test A[Not(1)] == A[Not(1:1)] == A[Not(A.==-10)] == collect(-9:13)
    @test @views A[Not(1)] == A[Not(1:1)] == A[Not(A.==-10)] == collect(-9:13)
    @test A[Not(end)] == A[Not(end:end)] == A[Not(A.==13)] == collect(-10:12)
    @test @views A[Not(end)] == A[Not(end:end)] == A[Not(A.==13)] == collect(-10:12)
    @test A[Not(iseven.(A))] == A[isodd.(A)] == collect(-9:2:13)
    @test A[Not([])] == A[collect(1:end)] == collect(-10:13)
    @test A[Not(1:end)] == A[Not(:)] == A[[]] == []

    @test_throws BoundsError A[Not(0)]
    @test_throws BoundsError A[Not(end+1)]
    @test_throws BoundsError A[Not(0:end)]
    @test_throws BoundsError A[Not(1:end+1)]
end

@testset "1-d offset readonly" for A in (OffsetArray(-10:13, -3), OffsetArray(reshape(-10:13,2,:), -5, 32), OffsetArray(reshape(-10:13,3,2,:), 4,-2,20))
    f = first(linearindices(A))
    l = last(linearindices(A))
    @test A[Not(f)] == A[Not(f:f)] == A[Not(A.==-10)] == collect(-9:13)
    @test @views A[Not(f)] == A[Not(f:f)] == A[Not(A.==-10)] == collect(-9:13)
    @test A[Not(l)] == A[Not(l:l)] == A[Not(A.==13)] == collect(-10:12)
    @test @views A[Not(l)] == A[Not(l:l)] == A[Not(A.==13)] == collect(-10:12)
    @test A[Not(iseven.(A))] == A[isodd.(A)]
    @test A[Not([])] == A[collect(f:l)] == collect(-10:13)
    @test A[Not(f:l)] == A[Not(:)] == A[[]] == []

    @test_throws BoundsError A[Not(f-1)]
    @test_throws BoundsError A[Not(l+1)]
end

@testset "1-d read/write" for A in (collect(1:4), reshape(collect(1:4),2,2))
    A[Not(2:3)] = [44, 11]
    @test vec(A) == [44, 2, 3, 11]
    A[Not(2:4)] = 0
    @test vec(A) == [0, 2, 3, 11]
    A[Not(1:end)] = 100
    @test vec(A) == [0, 2, 3, 11]
    A[Not(:)] = 100
    @test vec(A) == [0, 2, 3, 11]

    A[Not([])] = 0
    @test all(A.== 0)

    B = copy(A)
    @test_throws BoundsError A[Not(0)] = 200
    @test_throws BoundsError A[Not(end+1)] = 300
    @test A == B
end

@testset "2-d readonly" for A in (reshape(-10:13,3,:), reshape(-10:13,3,4,:))
    @test A[Not(2), :] == (@view A[Not(2), :]) == A[[1,3],:]
    @test A[:, Not(2)] == (@view A[:, Not(2)]) == A[:,[1;3:end]]
    @test A[Not(2), Not(2)] == (@view A[Not(2), Not(2)]) == A[[1;3:end],[1;3:end]]
    R = collect(CartesianRange(size(A)))
    @test A[Not(first(R))] == (@view A[Not(first(R))]) == A[2:end]
    @test A[Not(R[1:2])] == (@view A[Not(R[1:2])]) == A[3:end]
    @test A[Not(iseven.(A))] == (@view A[Not(iseven.(A))]) == A[isodd.(A)] == collect(-9:2:13)
end

@testset "2-d offset readonly" for A in (OffsetArray(reshape(-10:13,3,:), -1, 0), OffsetArray(reshape(-10:13,3,4,:), -10, 20, -30),)
    inds = indices(A)
    f₁, l₁ = first(inds[1]), last(inds[1])
    f₂, l₂ = first(inds[2]), last(inds[2])
    # TODO: Re-enable these tests for 3-d after PLI deprecation
    if ndims(A) == 2
        @test A[Not(f₁+1), f₂:l₂] == (@view A[Not(f₁+1), f₂:l₂]) == A[[f₁,l₁],f₂:l₂]
        @test A[f₁:l₁, Not(f₂+1)] == (@view A[f₁:l₁, Not(f₂+1)]) == A[f₁:l₁,[f₂;f₂+2:l₂]]
        @test A[Not(f₁+1), Not(f₂+1)] == (@view A[Not(f₁+1), Not(f₂+1)]) == A[[f₁,l₁],[f₂;f₂+2:l₂]]
    end
    R = collect(CartesianRange(indices(A)))
    @test A[Not(first(R))] == (@view A[Not(first(R))]) == A[linearindices(A)[2:end]]
    @test A[Not(R[1:2])] == (@view A[Not(R[1:2])]) == A[linearindices(A)[3:end]]
    @test A[Not(iseven.(A))] == (@view A[Not(iseven.(A))]) == A[isodd.(A)] == collect(-9:2:13)
end
