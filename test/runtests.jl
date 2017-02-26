using InvertedIndices
using Base.Test

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
    @test A[Not(iseven.(A))] == A[isodd.(A)]
    @test A[Not([])] == A[collect(1:end)] == collect(A[:])
end

@testset "1-d read/write" for A in (collect(1:4), reshape(collect(1:4),2,2))
    A[Not(2:3)] = [44, 11]
    @test vec(A) == [44, 2, 3, 11]
    A[Not(2:4)] = 0
    @test vec(A) == [0, 2, 3, 11]
    A[Not([])] = 0
    @test all(A.== 0)
end

@testset "2-d readonly" for A in (reshape(-10:13,3,:), reshape(-10:13,3,4,:))
    @test A[Not(2), :] == (@view A[Not(2), :]) == A[[1,3],:]
    @test A[:, Not(2)] == (@view A[:, Not(2)]) == A[:,[1;3:end]]
    R = collect(CartesianRange(size(A)))
    @test A[Not(first(R))] == (@view A[Not(first(R))]) == A[2:end]
    @test A[Not(R[1:2])] == (@view A[Not(R[1:2])]) == A[3:end]
    @test A[Not(iseven.(A))] == (@view A[Not(iseven.(A))]) == A[isodd.(A)] == collect(-9:2:13)
end
