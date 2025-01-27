####################
# Generalized kernel ridge regression

export KRRModel, RankNumber, RankThreshold, RankNumberEigs

struct KRRModel{LikelihoodType <: Likelihood, LinkType <: Link}
    k
    U
    Z
    X
    Y
    α
    Δ
    dof
    reml
end

function StatsBase.predict(m::KRRModel{LikelihoodType,LinkType},X=m.X) where {LikelihoodType <: Likelihood, LinkType <: Link}
    K = m.k(X,m.X)
    η = K * m.U*m.α
    link(LinkType,η)
end
StatsBase.fitted(m::KRRModel) = predict(m)

function intensity(m::KRRModel,X=m.X,k=m.k,U=m.U,Z=m.Z,α=m.α,Δ=m.Δ)
    K = k(X,m.X)
    exp.(K*U*α .+ Δ)
end

function StatsBase.loglikelihood(::Type{KRRModel},::Type{T},::Type{U},Y,M,α) where {T <: Likelihood, U <: Link}
    η = M*α
    loglikelihood(T,Y,link(U,η))
end

function gradient(::Type{KRRModel},::Type{T},::Type{U},Y,M,α) where {T<: Likelihood, U <: Link}
    η = M*α
    M'*(gradient(T,Y,link(U,η)) .* gradient(U,η))
end

function hessian(::Type{KRRModel},::Type{T},::Type{U},Y,M,α) where {T <: Likelihood, U <: Link}
    η = M*α
    μ = link(U,η)
    
    W = Diagonal(abs2.(gradient(U,η)) .* hessian(T,Y,μ) .+ gradient(T,Y,μ) .* hessian(U,η))
    M'W*M
end

StatsBase.nobs(m::KRRModel) = length(m.Y)
StatsBase.dof(m::KRRModel) = m.dof
StatsBase.aic(m::KRRModel) = 2 * dof(m) - 2 * loglikelihood(m)
StatsBase.deviance(m::KRRModel) = 2 * (lmax(m) - loglikelihood(m))
lmax(m::KRRModel) = -sum(m.Y)
gcv(m::KRRModel,scale=1.0) = nobs(m) * deviance(m)/(nobs(m) - scale*dof(m))^2
reml(m::KRRModel) = m.reml

abstract type RankChoice
end

"""
Choose rank based on an eigenvalue threshold

`RankThreshold(threshold)(K)` will return
the eigendecomposition that corresponds to the 
highest eigenvalues/vectors such that the sum of the 
selected eigenvalues is at least `threshold` 
times the sum of all the eigenvalues
"""
struct RankThreshold <: RankChoice
    threshold
end

function (rt::RankThreshold)(K)
    λ,U = eigen(Symmetric(K))
    R = findfirst(x->x>rt.threshold,cumsum(reverse(λ))./sum(λ))
    Diagonal(λ[end-R+1:end]),U[:,end-R+1:end]
end


"""
Choose rank explicitly

`RankNumber(R)(K)` will return the R
highest eigenvalues/vectors of K.
"""
struct RankNumber <: RankChoice
    R
end

function (rt::RankNumber)(K)
    N = size(K,1)
    λ,U = eigen(Symmetric(K),N-rt.R+1:N)
    Diagonal(λ),U
end

"""
Choose rank explicitly using iterative solver

`RankNumberEigs(R)(K)` returns the R highest
eigenvalues/vectors of K, computed using the 
iterative solvers in the `eigs` method
from Arpack. This can be more efficient for
large eigenvalue problems.
"""
struct RankNumberEigs <: RankChoice
    R
end

function (rt::RankNumberEigs)(K)
    N = size(K,1)
    λ,U = eigs(Symmetric(K),nev=rt.R)
    Diagonal(λ),U
end

function StatsBase.fit(::Type{KRRModel},::Type{LikelihoodType},::Type{LinkType},
                       kernel,
                       X,Y,
                       γ, # Regularization parameter
                       Δ, # Scale parameter
                       rankchoice::RankChoice;
                       verbose=false,
                       opt_alg = NewtonTrustRegion(),
                       optargs...
                       ) where {LikelihoodType <: Likelihood, LinkType <: Link}

    N = size(X,1)

    # Compute kernel matrix    
    K = kernel(X)

    # Reduced-rank eigenvalue decomposition
    D,U = rankchoice(K)
    R = size(D,1)
    
    # Model matrix
    M = U*D

    # Initialize optimization
    α0 = zeros(R)

    # Define optimization functions
    # Target is the negative loglikelihood plus the penalty/prior term
    f(α) = -loglikelihood(KRRModel,LikelihoodType,LinkType,Y,M,α) + 0.5 * γ * α'D*α
    
    function g!(G,α)        
        G .= -gradient(KRRModel,LikelihoodType,LinkType,Y,M,α) + γ*D*α
        G
    end
    function h!(H,α)
        H .= -hessian(KRRModel,LikelihoodType,LinkType,Y,M,α) + γ*D
    end
    opt = optimize(f,g!,h!,α0,opt_alg,Optim.Options(show_trace=verbose;optargs...))
    Optim.converged(opt) || @warn "Optimization failed to converge, γ=$γ, L = $(kernel.L)"

    α = Optim.minimizer(opt)
    η = M*α
    μ = link(LinkType,η)

    # Negative Hessian of the joint probability
    H = -hessian(KRRModel,LikelihoodType,LinkType,Y,M,α) + γ*D
    # Not sure this DOF calculation is correct
    dof = R # tr(M * (H\ (M'Diagonal(μ))))

    reml = loglikelihood(KRRModel,LikelihoodType,LinkType,Y,M,α) -0.5*length(Y)*log(2π) +
        -0.5*γ*α'D*α + 0.5*logabsdet(γ*D)[1] - 0.5*length(η)*log(2π) +
        -0.5*logabsdet(H)[1] + 0.5*length(η) * log(2π)

    KRRModel{LikelihoodType,LinkType}(kernel,U,D,X,Y,α,Δ,dof,reml)
end

function simulate(m::KRRModel,B=1,X = m.X,k=m.k,U=m.U,Z=m.Z,α=m.α,Δ=m.Δ)
    μ = intensity(m,X,k,U,Z,α,Δ)
    [rand.(Poisson.(μ)) for i in 1:B]
end
