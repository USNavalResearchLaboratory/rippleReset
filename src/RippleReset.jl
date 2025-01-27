module RippleReset

using Distances, Arpack, StatsBase, Optim, LinearAlgebra, Distributions, CSV, DataFrames

export LaggedRegression, RippleResetModel, simulate, intensity, lmax, gcv, reml,
    bootstrap_simulate, bootstrap_resample

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

function (m::LaggedRegression)(Y,x)
    (lagmatrix(x,m.k),Y[m.k+1:end])
end

function (m::LaggedRegression)(Ys::Vector{Vector{S}},xs::Vector{Vector{T}}) where {S, T}
    X = reduce(vcat,lagmatrix(x,m.k) for x in xs)
    Y = reduce(vcat,Y[m.k+1:end] for Y in Ys)
    X,Y
end

struct RippleResetModel
    design
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

StatsBase.nobs(m::RippleResetModel) = length(m.Y)
StatsBase.dof(m::RippleResetModel) = m.dof
StatsBase.aic(m::RippleResetModel) = 2 * dof(m) - 2 * loglikelihood(m)
StatsBase.deviance(m::RippleResetModel) = 2 * (lmax(m) - loglikelihood(m))
lmax(m::RippleResetModel) = -sum(m.Y)
gcv(m::RippleResetModel,scale=1.0) = nobs(m) * deviance(m)/(nobs(m) - scale*dof(m))^2
reml(m::RippleResetModel) = m.reml

function intensity(m::RippleResetModel,X=m.X::Matrix,k=m.k,U=m.U,Z=m.Z,α=m.α,Δ=m.Δ)
    K = k(X,m.X)
    αp = U*Z*α[2:end]
    exp.(α[1] .+ K*αp .+ Δ)
end

function intensity(m::RippleResetModel,Λ::Vector)
    X,_ = m.design(zeros(Bool,length(Λ)),Λ)
    intensity(m,X)
end

function simulate(m::RippleResetModel,X = m.X::Matrix,B=1,k=m.k,U=m.U,Z=m.Z,α=m.α,Δ=m.Δ)
    μ = intensity(m,X,k,U,Z,α,Δ)
    [[zeros(m.design.k);rand.(Poisson.(μ))] for i in 1:B]
end

function simulate(m::RippleResetModel,Λ::Vector{Float64},B=1)
    X,_ = m.design(zeros(Bool,length(Λ)),Λ)
    simulate(m,X,B)
end

function simulate(m::RippleResetModel,Λ::Vector{Vector{Float64}},B=1)
    Zs = map(Λ) do λ
        simulate(m,λ,B)
    end
    # Invert the inner and outer dimensions of Zs
    [collect(z) for z in zip(Zs...)]
end

function bootstrap_simulate(m::RippleResetModel,Λ::Vector{Float64},B=1)
    X,_ = m.design(zeros(Bool,length(Λ)),Λ)
    simulate(m,X,B)
end

function bootstrap_simulate(m::RippleResetModel,Λ::Vector{Vector{Float64}},B=1)
    Zs = map(Λ) do λ
        bootstrap_simulate(m,λ,B)
    end
    [collect(z) for z in zip(Zs...)]
end

function bootstrap_resample(design::LaggedRegression,Λ,resets,B=1)
    X,Y = design(resets,Λ)
    bs = [sample(1:length(Y),length(Y)) for i in 1:B]
    [(X[b,:],Y[b]) for b in bs]
end

function StatsBase.loglikelihood(::Type{RippleResetModel},Y,M,α,Δ)
    μ = M*α .+ Δ
    dot(Y,μ) - sum(exp,μ)
end

function StatsBase.loglikelihood(m::RippleResetModel,X = m.X::Matrix,Y = m.Y,k=m.k,U=m.U,Z=m.Z,α = m.α,Δ = m.Δ)
    K = k(X,m.X)
    αp = U*Z*α[2:end]
    μ = α[1] .+ K*αp .+ Δ
    dot(Y,μ) - sum(exp,μ)
end

function StatsBase.loglikelihood(m::RippleResetModel,Λ::Vector,resets::Vector)
    X,Y = m.design(resets,Λ)
    loglikelihood(m,X,Y)
end

function StatsBase.fit(::Type{RippleResetModel},design,k,Λ::Vector,resets,γ,Δ;
                       kwargs...)
    X,Y = design(resets,Λ)

    fit(RippleResetModel,design,k,X,Y,γ,Δ;kwargs...)
end
    
function StatsBase.fit(::Type{RippleResetModel},design,k,X::Matrix,Y,γ,Δ;
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
    
    f(α) = -loglikelihood(RippleResetModel,Y,M,α,Δ) + 0.5 * γ * α'S*α 
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
    RippleResetModel(design,k,U,Z,X,Y,Optim.minimizer(opt),Δ,dof,reml)
end

"""
    StatsBase.fit(::Type{RippleResetModel},design::LaggedRegression,Λ,resets,Δ;verbose=false,rank=5)

Fit the ripple reset model with the LaggedRegression design to a ripple reset time series `Λ` and the
corresponding vector of ripple `resets`. `Δ` is the time step in seconds. `verbose` turns on logging
of the optimization routine and `rank` is the size of the reduced-rank approximation of the kernel
ridge regression.

If you want to fit a model to multiple time series, pass in a vector of vectors for `Λ` and `resets`.
"""
function StatsBase.fit(::Type{RippleResetModel},design::LaggedRegression,Λ,resets,Δ;verbose=false,rank=5)
    L0 = 1000.0
    γ0 = 1.0
    
    f(θ) = -reml(fit(RippleResetModel,design,RBFKernel(exp(θ[1])),Λ,resets,exp(θ[2]),log(Δ),rank=rank,allow_f_increases=true,iterations=1000)) - θ[1] + 2*exp(θ[1])/L0 -θ[2] + 2*exp(θ[2])/γ0
    opt = optimize(f,log.([L0/2;γ0/2]),NelderMead(),Optim.Options(show_trace=verbose))
    θn = Optim.minimizer(opt)
    fit(RippleResetModel,design,RBFKernel(exp(θn[1])),Λ,resets,exp(θn[2]),log(Δ),rank=rank,allow_f_increases=true)
end


include("krr.jl")
include("reml.jl")
include("data.jl")


end # module
