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

struct KRRModel{K<:Kernel}
    k::K
    U
    Z
    X
    Y
    α
    Δ
    dof
    reml
end

function StatsBase.predict(m::KRRModel,X=m.X)
    K = k(X,m.X)
    αp = m.U*m.Z*m.α[2:end]
    exp.(m.α[1] .+ K*αp .+ m.Δ)
end
StatsBase.fitted(m::KRRModel) = predict(m)

function intensity(m::KRRModel,X=m.X,k=m.k,U=m.U,Z=m.Z,α=m.α,Δ=m.Δ)
    K = k(X,m.X)
    αp = U*Z*α[2:end]
    exp.(α[1] .+ K*αp .+ Δ)
end

function StatsBase.loglikelihood(::Type{KRRModel},::Type{T},::Type{U},Y,M,α) where {T <: Likelihood, U <: Link}
    η = M*α
    loglikelihood(T,Y,link(U,η))
end

function gradient(::Type{KRRModel},::Type{T},::Type{U},Y,M,α) where {T<: Likelihood, U <: Link}
    η = M*α
    M'*(gradient(T,Y,link(U,η)) .* gradient(U,η))
end

function StatsBase.loglikelihood(::Type{KRRModel},Y,M,α,Δ)
    μ = M*α .+ Δ
    dot(Y,μ) - sum(exp,μ)
end

function StatsBase.loglikelihood(m::KRRModel,X = m.X,Y = m.Y,k=m.k,U=m.U,Z=m.Z,α = m.α,Δ = m.Δ)
    K = k(X,m.X)
    αp = U*Z*α[2:end]
    μ = α[1] .+ K*αp .+ Δ
    dot(Y,μ) - sum(exp,μ)
end

StatsBase.nobs(m::KRRModel) = length(m.Y)
StatsBase.dof(m::KRRModel) = m.dof
StatsBase.aic(m::KRRModel) = 2 * dof(m) - 2 * loglikelihood(m)
StatsBase.deviance(m::KRRModel) = 2 * (lmax(m) - loglikelihood(m))
lmax(m::KRRModel) = -sum(m.Y)
gcv(m::KRRModel,scale=1.0) = nobs(m) * deviance(m)/(nobs(m) - scale*dof(m))^2
reml(m::KRRModel) = m.reml

function StatsBase.fit(::Type{KRRModel},k,X,Y,γ,Δ;
                       rank=size(X,1),
                       verbose=false,
                       iterations=1000,
                       opt_alg=NewtonTrustRegion(),
                       allow_f_increases=false)
    N = size(X,1)
    K = k(X)
    #D,U = eigen(Symmetric(K),N-rank+1:N)
    D,U = eigs(Symmetric(K),nev=rank)
    D = Diagonal(D)
    T = ones(N,1)
    A = U'T
    F = qr(A)

    Z = (F.Q * Matrix(I,rank,rank))[:,2:end]
    P = Z'D*Z
    S = [zeros(1,size(P,2)+1); zeros(size(P,1),1) P]
    W = U*D*Z

    M = [T W]

    α0 = zeros(rank)
    
    f(α) = -loglikelihood(KRRModel,Y,M,α,Δ) + 0.5 * γ * α'S*α 
    function g!(G,α)        
        μ = M*α .+ Δ
        G .= M'*(exp.(μ) .- Y) .+ γ*S*α
        G
    end
    function h!(H,α)
        μ = M*α .+ Δ
        H .= M'Diagonal(exp.(μ))*M .+ γ*S
    end
    opt = optimize(f,g!,h!,α0,opt_alg,Optim.Options(show_trace=verbose,iterations=iterations,allow_f_increases=allow_f_increases))
    Optim.converged(opt) || @warn "Optimization failed to converge, γ=$γ, L=$(k.L)"
    α = Optim.minimizer(opt)
    λ = exp.(M*α .+ Δ)
    H = M'Diagonal(λ)*M + γ*S
    dof = tr(M * (H\ (M'Diagonal(λ))))
    reml = -Optim.minimum(opt) + 0.5 * logabsdet(γ*P)[1] - 0.5 * logabsdet(H)[1] + 0.5 * size(T,2) * log(2π)
    KRRModel(k,U,Z,X,Y,Optim.minimizer(opt),Δ,dof,reml)
end

function simulate(m::KRRModel,B=1,X = m.X,k=m.k,U=m.U,Z=m.Z,α=m.α,Δ=m.Δ)
    μ = intensity(m,X,k,U,Z,α,Δ)
    [rand.(Poisson.(μ)) for i in 1:B]
end

include("reml.jl")

end # module
