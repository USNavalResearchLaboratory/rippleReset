using Test, KernelRidgePoissonRegression, Random, Distributions

const KRR = KernelRidgePoissonRegression

Random.seed!(83639)

N = 10
D = 5

# Generate test data
X = randn(N,D)
β = randn(D)
η = X * β

@testset "Normal data" begin
    μ1 = link(IdentityLink,η)
    Y1 = rand.(Normal.(μ1,1.0))

    KRR.loglikelihood(NormalLikelihood,Y1,μ1)

    @test KRR.gradient(NormalLikelihood,Y1,μ1) ≈ (Y1 .- μ1)
end

@testset "Poisson data" begin
    μ2 = link(LogLink,η)
    Y2 = rand.(Poisson.(μ2))

    KRR.loglikelihood(PoissonLikelihood,Y2,μ2)

    @test KRR.gradient(PoissonLikelihood,Y2,μ2) ≈ Y2 ./ μ2 .- 1
end

@testset "Exponential data" begin
    μ3 = link(LogLink,η)
    Y3 = rand.(Exponential.(μ3))

    KRR.loglikelihood(ExponentialLikelihood,Y3,μ3)

    @test KRR.gradient(ExponentialLikelihood,Y3,μ3) ≈ Y3 ./ abs2.(μ3) .- inv.(μ3)
end

m1 = fit(KRRModel,RBFKernel(1.0),NormalLikelihood,IdentityLink,X,Y1,1.0,1.0,verbose=true)
m2 = fit(KRRModel,RBFKernel(1.0),PoissonLikelihood,LogLink,X,Y2,1.0,1.0,verbose=true,rank=N-1)
m3 = fit(KRRModel,RBFKernel(1.0),ExponentialLikelihood,LogLink,X,Y3,1.0,1.0,verbose=true)



