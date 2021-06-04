using Test, KernelRidgePoissonRegression, Random, Distributions

const KRR = KernelRidgePoissonRegression

Random.seed!(83639)

N = 10
D = 5

# Generate test data
X = randn(N,D)
β = randn(D)
η = X * β

@testset "KRR" begin
    
    @testset "Normal data" begin
        μ1 = link(IdentityLink,η)
        Y1 = rand.(Normal.(μ1,1.0))

        l1 = KRR.loglikelihood(NormalLikelihood,Y1,μ1)

        @test KRR.gradient(NormalLikelihood,Y1,μ1) ≈ (Y1 .- μ1)
        @test KRR.gradient(IdentityLink,η) ≈ ones(N)

        @test KRR.hessian(NormalLikelihood,Y1,μ1) ≈ -ones(N)
        @test KRR.hessian(IdentityLink,η) ≈ zeros(N)

        @test KRR.loglikelihood(KRRModel,NormalLikelihood,IdentityLink,Y1,X,β) ≈ l1

        g1 = KRR.gradient(KRRModel,NormalLikelihood,IdentityLink,Y1,X,β)        
        
        @test g1 ≈ X'*(Y1 .- μ1)

        m1 = fit(KRRModel,RBFKernel(1.0),NormalLikelihood,IdentityLink,X,Y1,1.0,1.0,verbose=true)
    end

    @testset "Poisson data" begin
        μ2 = link(LogLink,η)
        Y2 = rand.(Poisson.(μ2))

        l2 = KRR.loglikelihood(PoissonLikelihood,Y2,μ2)

        @test KRR.gradient(PoissonLikelihood,Y2,μ2) ≈ Y2 ./ μ2 .- 1
        @test KRR.gradient(LogLink,η) ≈ exp.(η)

        @test KRR.hessian(PoissonLikelihood,Y2,μ2) ≈ -Y2 ./ abs2.(μ2)
        @test KRR.hessian(LogLink,η) ≈ exp.(η)

        @test KRR.loglikelihood(KRRModel,PoissonLikelihood,LogLink,Y2,X,β) ≈ l2

        g2 = KRR.gradient(KRRModel,PoissonLikelihood,LogLink,Y2,X,β)

        @test g2 ≈ X'*(Y2 .- μ2)

        m2 = fit(KRRModel,RBFKernel(1.0),PoissonLikelihood,LogLink,X,Y2,1.0,1.0,verbose=true,rank=N-1)
    end

    @testset "Exponential data" begin
        μ3 = link(LogLink,η)
        Y3 = rand.(Exponential.(μ3))

        l3 = KRR.loglikelihood(ExponentialLikelihood,Y3,μ3)

        @test KRR.gradient(ExponentialLikelihood,Y3,μ3) ≈ Y3 ./ abs2.(μ3) .- inv.(μ3)

        @test KRR.hessian(ExponentialLikelihood,Y3,μ3) ≈ inv.(abs2.(μ3)) .- 2 .* Y3 ./ μ3.^3

        @test KRR.loglikelihood(KRRModel,ExponentialLikelihood,LogLink,Y3,X,β) ≈ l3

        g3 = KRR.gradient(KRRModel,ExponentialLikelihood,LogLink,Y3,X,β)

        @test g3 ≈ X'*(Y3./μ3 .- 1)

        m3 = fit(KRRModel,RBFKernel(1.0),ExponentialLikelihood,LogLink,X,Y3,1.0,1.0,verbose=true)
    end
end



