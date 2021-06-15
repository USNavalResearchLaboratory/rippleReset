struct OldKRRModel
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

StatsBase.nobs(m::OldKRRModel) = length(m.Y)
StatsBase.dof(m::OldKRRModel) = m.dof
StatsBase.aic(m::OldKRRModel) = 2 * dof(m) - 2 * loglikelihood(m)
StatsBase.deviance(m::OldKRRModel) = 2 * (lmax(m) - loglikelihood(m))
lmax(m::OldKRRModel) = -sum(m.Y)
gcv(m::OldKRRModel,scale=1.0) = nobs(m) * deviance(m)/(nobs(m) - scale*dof(m))^2
reml(m::OldKRRModel) = m.reml

function intensity(m::OldKRRModel,X=m.X,k=m.k,U=m.U,Z=m.Z,α=m.α,Δ=m.Δ)
    K = k(X,m.X)
    αp = U*Z*α[2:end]
    exp.(α[1] .+ K*αp .+ Δ)
end

function simulate(m::OldKRRModel,B=1,X = m.X,k=m.k,U=m.U,Z=m.Z,α=m.α,Δ=m.Δ)
    μ = intensity(m,X,k,U,Z,α,Δ)
    [rand.(Poisson.(μ)) for i in 1:B]
end

function StatsBase.loglikelihood(::Type{OldKRRModel},Y,M,α,Δ)
    μ = M*α .+ Δ
    dot(Y,μ) - sum(exp,μ)
end

function StatsBase.loglikelihood(m::OldKRRModel,X = m.X,Y = m.Y,k=m.k,U=m.U,Z=m.Z,α = m.α,Δ = m.Δ)
    K = k(X,m.X)
    αp = U*Z*α[2:end]
    μ = α[1] .+ K*αp .+ Δ
    dot(Y,μ) - sum(exp,μ)
end

function StatsBase.fit(::Type{OldKRRModel},k,X,Y,γ,Δ;
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
    
    f(α) = -loglikelihood(OldKRRModel,Y,M,α,Δ) + 0.5 * γ * α'S*α 
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
    OldKRRModel(k,U,Z,X,Y,Optim.minimizer(opt),Δ,dof,reml)
end

"""
Fit a KRR model to the data (X,Y) with log timestep Δ

Optimizes a penalized REML over the kernel length scale and regularization parameter
"""
function StatsBase.fit(::Type{OldKRRModel},X,Y,Δ;verbose=false,rank=5)
    L0 = 1000.0
    γ0 = 1.0
    
    f(θ) = -reml(fit(OldKRRModel,RBFKernel(exp(θ[1])),X,Y,exp(θ[2]),Δ,rank=rank,allow_f_increases=true,iterations=1000)) - θ[1] + 2*exp(θ[1])/L0 -θ[2] + 2*exp(θ[2])/γ0
    opt = optimize(f,log.([L0/2;γ0/2]),NelderMead(),Optim.Options(show_trace=verbose))
    θn = Optim.minimizer(opt)
    fit(OldKRRModel,RBFKernel(exp(θn[1])),X,Y,exp(θn[2]),Δ,rank=rank,allow_f_increases=true)
end
