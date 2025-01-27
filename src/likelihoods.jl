export NormalLikelihood, PoissonLikelihood, ExponentialLikelihood, BernoulliLikelihood

abstract type Likelihood
end

struct NormalLikelihood <: Likelihood
end

function StatsBase.loglikelihood(::Type{NormalLikelihood},Y,μ)
    -0.5 * sum(abs2,Y .- μ)    
end

function gradient(::Type{NormalLikelihood},Y,μ)
    Y .- μ
end

function hessian(::Type{NormalLikelihood},Y,μ)
    N = length(μ)
    -ones(eltype(μ),N)
end

struct PoissonLikelihood <: Likelihood
end

function StatsBase.loglikelihood(::Type{PoissonLikelihood},Y,μ)
    dot(Y,log.(μ)) - sum(μ)
end

function gradient(::Type{PoissonLikelihood},Y,μ)
    Y./μ .- 1
end

function hessian(::Type{PoissonLikelihood},Y,μ)
    -Y./ abs2.(μ)
end

struct ExponentialLikelihood <: Likelihood
end

function StatsBase.loglikelihood(::Type{ExponentialLikelihood},Y,μ)
    -sum(Y ./ μ) - sum(log,μ)
end

function gradient(::Type{ExponentialLikelihood},Y,μ)
    Y ./ abs2.(μ) - inv.(μ)
end

function hessian(::Type{ExponentialLikelihood},Y,μ)
    inv.(abs2.(μ)) .- 2Y ./ μ.^3
end

struct BernoulliLikelihood <: Likelihood
end

function StatsBase.loglikelihood(::Type{BernoulliLikelihood},Y,μ)
    sum(Y .* log.(μ) + (1 .- Y) .* log.(1 .- μ))
end

function gradient(::Type{BernoulliLikelihood},Y,μ)
    Y ./ μ .- (1 .- Y) ./ (1 .- μ)
end

function hessian(::Type{BernoulliLikelihood},Y,μ)
    -Y ./ abs2.(μ) .- (1 .- Y) ./ abs2.(1 .- μ)
end
