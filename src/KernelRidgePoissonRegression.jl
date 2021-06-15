module KernelRidgePoissonRegression

using Distances, Arpack, StatsBase, Optim, LinearAlgebra, Distributions

export LaggedRegression, KRRModel, simulate, intensity, lmax, gcv, reml

include("kernels.jl")
include("likelihoods.jl")
include("links.jl")

####################
# Model designs

"""
Lagged regression design matrix
"""
struct LaggedRegression
    k
end

function lagmatrix(x,k)
    X = ones(eltype(x),length(x)-k,k+2)
    for i in 1:(k+1)
        X[:,i+1] = x[1 + (i-1):end-k + (i-1)]
    end
    X
end

(m::LaggedRegression)(Y,x) = (lagmatrix(x,m.k),Y[m.k+1:end])

####################
# Kernel Ridge Poisson Regression

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

function StatsBase.fit(::Type{KRRModel},::Type{LikelihoodType},::Type{LinkType},
                       kernel,
                       X,Y,
                       γ, # Regularization parameter
                       Δ; # Scale parameter
                       verbose=false,rank=size(X,1)-1,
                       opt_alg = NewtonTrustRegion(),
                       optargs...
                       ) where {LikelihoodType <: Likelihood, LinkType <: Link}

    N = size(X,1)

    # Compute kernel matrix    
    K = kernel(X)

    # Reduced-rank eigenvalue decomposition
    D,U = eigen(Symmetric(K),N-rank+1:N)
    #D,U = eigs(Symmetric(K),nev=rank)
    D = Diagonal(D)
    
    # Model matrix
    M = U*D

    # Initialize optimization
    α0 = zeros(rank)

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

    H = hessian(KRRModel,LikelihoodType,LinkType,Y,M,α)
    # Not sure this DOF calculation is correct
    dof = rank # tr(M * (H\ (M'Diagonal(μ))))

    reml = -Optim.minimum(opt) + 0.5*logabsdet(γ*D)[1] - 0.5*logabsdet(H)[1] + 0.5*length(η) * log(2π)

    KRRModel{LikelihoodType,LinkType}(kernel,U,D,X,Y,Optim.minimizer(opt),Δ,dof,reml)
end

function simulate(m::KRRModel,B=1,X = m.X,k=m.k,U=m.U,Z=m.Z,α=m.α,Δ=m.Δ)
    μ = intensity(m,X,k,U,Z,α,Δ)
    [rand.(Poisson.(μ)) for i in 1:B]
end

include("reml.jl")
include("old.jl")

end # module

