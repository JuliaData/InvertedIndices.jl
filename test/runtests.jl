using InvertedIndices
using Base.Test

@testset "1-d readonly" for A in (-10:13, reshape(-10:13,2,:), reshape(-10:13,3,2,:))
    @test A[Not(1)] == A[Not(1:1)] == A[Not(A.==-10)] == collect(-9:13)
    @test @views A[Not(1)] == A[Not(1:1)] == A[Not(A.==-10)] == collect(-9:13)
    @test A[Not(end)] == A[Not(end:end)] == A[Not(A.==13)] == collect(-10:12)
    @test @views A[Not(end)] == A[Not(end:end)] == A[Not(A.==13)] == collect(-10:12)
    @test A[Not(iseven.(A))] == A[isodd.(A)]
    @test A[Not([])] == A[collect(1:end)] == collect(A[:])
end

@testset "vector array" begin
    A = collect(1:4)
    A[Not(2:3)] = [44, 11]
    @test A == [44, 2, 3, 11]
    A[Not(2:4)] = 0
    @test A == [0, 2, 3, 11]
    A[Not([])] = 0
    @test all(A.== 0)
end
