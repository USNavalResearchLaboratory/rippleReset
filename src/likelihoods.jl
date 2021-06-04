export NormalLikelihood, PoissonLikelihood, ExponentialLikelihood

abstract type Likelihood
end

struct NormalLikelihood <: Likelihood
end

function StatsBase.loglikelihood(::Type{NormalLikelihood},Y,μ)
    -0.5 * sum(abs2,Y .- μ)    
end

struct PoissonLikelihood <: Likelihood
end

function StatsBase.loglikelihood(::Type{PoissonLikelihood},Y,μ)
    dot(Y,log.(μ)) - sum(μ)
end

struct ExponentialLikelihood <: Likelihood
end

function StatsBase.loglikelihood(::Type{ExponentialLikelihood},Y,μ)
    -sum(Y ./ μ) - sum(log,μ)
end


