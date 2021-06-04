export NormalLikelihood, PoissonLikelihood, ExponentialLikelihood

abstract type Likelihood
end

struct NormalLikelihood <: Likelihood
end

function StatsBase.loglikelihood(::Type{NormalLikelihood},Y,μ)
    -0.5 * sum(abs2,Y .- μ)    
end

function gradient(::Type{NormalLikelihood},Y,μ)
    -(Y .- μ)
end

struct PoissonLikelihood <: Likelihood
end

function StatsBase.loglikelihood(::Type{PoissonLikelihood},Y,μ)
    dot(Y,log.(μ)) - sum(μ)
end

function gradient(::Type{PoissonLikelihood},Y,μ)
    Y./μ .- 1
end

struct ExponentialLikelihood <: Likelihood
end

function StatsBase.loglikelihood(::Type{ExponentialLikelihood},Y,μ)
    -sum(Y ./ μ) - sum(log,μ)
end

function gradient(::Type{ExponentialLikelihood},Y,μ)
    Y ./ abs2.(μ) - inv.(μ)
end


