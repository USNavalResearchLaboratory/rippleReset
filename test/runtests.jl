using Test, KernelRidgePoissonRegression, Random, Distributions, LinearAlgebra, StatsBase

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

        h1 = KRR.hessian(KRRModel,NormalLikelihood,IdentityLink,Y1,X,β)

        @test h1 ≈ -X'X

        m1 = fit(KRRModel,NormalLikelihood,IdentityLink,RBFKernel(1.0),X,Y1,1.0,1.0,RankNumber(N-1),verbose=true)
        m1 = fit(KRRModel,NormalLikelihood,IdentityLink,RBFKernel(1.0),X,Y1,1.0,1.0,RankThreshold(0.99999),verbose=true)

        println("REML fitting")
        m1 = fit(KRRModel,NormalLikelihood,IdentityLink,X,Y1,1.0,RankThreshold(0.99999),verbose=true)

        predict(m1)
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

        h2 = KRR.hessian(KRRModel,PoissonLikelihood,LogLink,Y2,X,β)

        @test h2 ≈ -X'Diagonal(μ2)*X

        m2 = fit(KRRModel,PoissonLikelihood,LogLink,RBFKernel(1.0),X,Y2,1.0,1.0,RankNumber(N-1),verbose=true)
        m2 = fit(KRRModel,PoissonLikelihood,LogLink,RBFKernel(1.0),X,Y2,1.0,1.0,RankThreshold(0.99999),verbose=true)


        predict(m2)
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

        h3 = KRR.hessian(KRRModel,ExponentialLikelihood,LogLink,Y3,X,β)

        @test h3 ≈ X'Diagonal(-Y3 ./ μ3)*X

        m3 = fit(KRRModel,ExponentialLikelihood,LogLink,RBFKernel(1.0),X,Y3,1.0,1.0,RankNumber(N-1),verbose=true)
        m3 = fit(KRRModel,ExponentialLikelihood,LogLink,RBFKernel(1.0),X,Y3,1.0,1.0,RankThreshold(0.99999),verbose=true)

        predict(m3)
    end

    @testset "Bernoulli data" begin
        μ4 = link(LogitLink,η)
        Y4 = rand.(Bernoulli.(μ4))

        l4 = KRR.loglikelihood(BernoulliLikelihood,Y4,μ4)

        @test KRR.gradient(BernoulliLikelihood,Y4,μ4) ≈ Y4 ./ μ4 .- (1 .- Y4) ./ (1 .- μ4)

        @test KRR.gradient(LogitLink,η) ≈ exp.(η) ./ abs2.(1 .+ exp.(η))

        @test KRR.hessian(BernoulliLikelihood,Y4,μ4) ≈ -Y4 ./ abs2.(μ4) .- (1 .- Y4) ./ abs2.(1 .- μ4)       
        
        @test KRR.hessian(LogitLink,η) ≈ exp.(η) ./ abs2.(1 .+ exp.(η)) .* (1 .- 2 ./ (1 .+ exp.(-η)))

        @test KRR.loglikelihood(KRRModel,BernoulliLikelihood,LogitLink,Y4,X,β) ≈ l4

        g4 = KRR.gradient(KRRModel,BernoulliLikelihood,LogitLink,Y4,X,β)

        μp4 = μ4 .* (1 .- μ4)
        μpp4= μp4 .* (1 .- 2 * μ4)

        fp4 = Y4 ./ μ4 .- (1 .- Y4) ./ (1 .- μ4)
        fpp4= -Y4 ./ abs2.(μ4) .- (1 .- Y4)./abs2.(1 .- μ4)
        
        @test g4 ≈ X'*(fp4.*μp4)

        h4 = KRR.hessian(KRRModel,BernoulliLikelihood,LogitLink,Y4,X,β)

        @test h4 ≈ X' * Diagonal(abs2.(μp4) .* fpp4 .+ fp4 .* μpp4) * X

        m4 = fit(KRRModel,BernoulliLikelihood,LogitLink,RBFKernel(1.0),X,Y4,1.0,1.0,RankNumber(N-1),verbose=true)

        m4 = fit(KRRModel,BernoulliLikelihood,LogitLink,RBFKernel(1.0),X,Y4,1.0,1.0,RankThreshold(0.99999),verbose=true)

        predict(m4)
    end
end



