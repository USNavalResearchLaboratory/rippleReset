function StatsBase.fit(::Type{KRRModel},::Type{LikelihoodType},::Type{LinkType},
                       X,Y,
                       Δ,
                       rankchoice::RankChoice;
                       verbose=false,
                       kwargs...) where {LikelihoodType <: Likelihood, LinkType <: Link}

    L0 = 1.0
    γ0 = 1.0
    
    f(θ) = -reml(fit(KRRModel,LikelihoodType,LinkType,
                     RBFKernel(exp(θ[1])),
                     X,Y,
                     exp(θ[2]),
                     Δ,
                     rankchoice;
                     kwargs...))

    opt = optimize(f,log.([L0;γ0]),NelderMead(),Optim.Options(show_trace=verbose))
    θn = Optim.minimizer(opt)
    fit(KRRModel,LikelihoodType,LinkType,RBFKernel(exp(θn[1])),X,Y,exp(θn[2]),Δ,rankchoice;kwargs...)                    
end
