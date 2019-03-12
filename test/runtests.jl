using InvertedIndices
using Test
using OffsetArrays

@testset "0-d" begin
    A = fill(1)
    @test A[Not(fill(A.==1))] == []
    @test A[Not(CartesianIndex())] == []
    @test A[Not(fill(A.==2))] == [1]
    A[Not(fill(A.==2))] = fill(0)
    @test A[] == 0
    A[Not(fill(A.==2))] .= 1
    @test A[] == 1
    @test A[Not([1,1])] == []
    @test A[Not(1,1)] == []
end

@testset "1-d readonly" for A in (-10:13, reshape(-10:13,2,:), reshape(-10:13,3,2,:))
    @test A[Not(1)] == A[Not(1:1)] == A[Not(A.==-10)] == collect(-9:13)
    @test @views A[Not(1)] == A[Not(1:1)] == A[Not(A.==-10)] == collect(-9:13)
    @test A[Not(end)] == A[Not(end:end)] == A[Not(A.==13)] == collect(-10:12)
    @test @views A[Not(end)] == A[Not(end:end)] == A[Not(A.==13)] == collect(-10:12)
    @test A[Not(iseven.(A))] == A[isodd.(A)] == collect(-9:2:13)
    @test A[Not([])] == A[collect(1:end)] == collect(-10:13)
    @test A[Not(1:end)] == A[Not(:)] == A[[]] == []
    @test A[Not([1,1,1,2,2])] == A[3:end]
    @test A[Not([end,end,end,end-1,end-1])] == A[1:end-2]
    @test A[Not(3,2,1)] == A[Not(3,2,3,1,2,1)] == A[4:end]

    @test_throws BoundsError A[Not(0)]
    @test_throws BoundsError A[Not(end+1)]
    @test_throws BoundsError A[Not(0:end)]
    @test_throws BoundsError A[Not(1:end+1)]
end

@testset "1-d offset readonly" for A in (OffsetArray(-10:13, -3), OffsetArray(reshape(-10:13,2,:), -5, 32), OffsetArray(reshape(-10:13,3,2,:), 4,-2,20))
    f = first(LinearIndices(A))
    l = last(LinearIndices(A))
    @test A[Not(f)] == A[Not(f:f)] == A[Not(A.==-10)] == collect(-9:13)
    @test @views A[Not(f)] == A[Not(f:f)] == A[Not(A.==-10)] == collect(-9:13)
    @test A[Not(l)] == A[Not(l:l)] == A[Not(A.==13)] == collect(-10:12)
    @test @views A[Not(l)] == A[Not(l:l)] == A[Not(A.==13)] == collect(-10:12)
    @test A[Not(iseven.(A))] == A[isodd.(A)]
    @test A[Not([])] == A[collect(f:l)] == collect(-10:13)
    @test A[Not(f:l)] == A[Not(:)] == A[[]] == []
    @test A[Not([f,f,f,f+1,f+1])] == A[Not(f,f+1,f,f+1,f)] == A[f+2:l]
    @test A[Not([l,l,l,l-1,l-1])] == A[Not(l-1,l,l,l-1,l-1)] == A[f:l-2]

    @test_throws BoundsError A[Not(f-1)]
    @test_throws BoundsError A[Not(l+1)]
end

@testset "1-d read/write" for A in (collect(1:4), reshape(collect(1:4),2,2))
    A[Not(2:3)] = [44, 11]
    @test vec(A) == [44, 2, 3, 11]
    A[Not(2:4)] .= 0
    @test vec(A) == [0, 2, 3, 11]
    A[Not(1:end)] .= 100
    @test vec(A) == [0, 2, 3, 11]
    A[Not(:)] .= 100
    @test vec(A) == [0, 2, 3, 11]

    A[Not([])] .= 0
    @test all(A.== 0)

    B = copy(A)
    @test_throws BoundsError A[Not(0)] = 200
    @test_throws BoundsError A[Not(end+1)] = 300
    @test A == B
end

@testset "2-d readonly" for A in (reshape(-10:13,3,:), reshape(-10:13,3,:,1))
    @test A[Not(2), :] == (@view A[Not(2), :]) == A[[1,3],:]
    @test A[:, Not(2)] == (@view A[:, Not(2)]) == A[:,[1;3:end]]
    @test A[Not(2), Not(2)] == (@view A[Not(2), Not(2)]) == A[[1;3:end],[1;3:end]]
    R = collect(CartesianIndices(size(A)))
    @test A[Not(first(R))] == (@view A[Not(first(R))]) == A[2:end]
    @test A[Not(R[1:2])] == (@view A[Not(R[1:2])]) == A[3:end]
    @test A[Not(iseven.(A))] == (@view A[Not(iseven.(A))]) == A[isodd.(A)] == collect(-9:2:13)
    @test A[Not([1,1,1,2,2]), Not([2,2,1,1,1])] == A[Not(2,1,1,2), Not(2,2,1,1,2)] == A[3:end, 3:end]
    @test A[Not([end,end-1,end,end-1]), Not([end,end-1,end,end-1])] == A[Not(end,end-1,end,end-1), Not(end,end-1,end,end-1)] == A[1:end-2, 1:end-2]
    @test A[Not([2,2,1,1,1]), Not([end,end-1,end,end-1])] == A[Not(2,2,1,1,2), Not(end,end-1,end,end-1)] == A[3:end, 1:end-2]
end

@testset "2-d offset readonly" for A in (OffsetArray(reshape(-10:13,3,:), -1, 0), OffsetArray(reshape(-10:13,3,4,:), -10, 20, -30),)
    inds = axes(A)
    f₁, l₁ = first(inds[1]), last(inds[1])
    f₂, l₂ = first(inds[2]), last(inds[2])
    # TODO: Re-enable these tests for 3-d after PLI deprecation
    if ndims(A) == 2
        @test A[Not(f₁+1), f₂:l₂] == (@view A[Not(f₁+1), f₂:l₂]) == A[[f₁,l₁],f₂:l₂]
        @test A[f₁:l₁, Not(f₂+1)] == (@view A[f₁:l₁, Not(f₂+1)]) == A[f₁:l₁,[f₂;f₂+2:l₂]]
        @test A[Not(f₁+1), Not(f₂+1)] == (@view A[Not(f₁+1), Not(f₂+1)]) == A[[f₁,l₁],[f₂;f₂+2:l₂]]
    end
    R = collect(CartesianIndices(axes(A)))
    @test A[Not(first(R))] == (@view A[Not(first(R))]) == A[LinearIndices(A)[2:end]]
    @test A[Not(R[1:2])] == (@view A[Not(R[1:2])]) == A[LinearIndices(A)[3:end]]
    @test A[Not(iseven.(A))] == (@view A[Not(iseven.(A))]) == A[isodd.(A)] == collect(-9:2:13)
end

@testset "Utilities" begin
    @test sprint(show, InvertedIndices.InvertedIndexIterator([2,4], 1:5)) == "Not([2, 4])"
    A = [1]
    ex = try A[Not(2,3)] catch ex; ex end
    @test occursin("Not([2, 3])", sprint(Base.showerror, ex))
end
