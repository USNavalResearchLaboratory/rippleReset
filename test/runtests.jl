using Test, KernelRidgePoissonRegression, Random, Distributions

const KRR = KernelRidgePoissonRegression

Random.seed!(83639)

N = 10
D = 5

# Generate test data
X = randn(N,D)
β = randn(D)
μ = X * β

## Normal data
Y1 = rand.(Normal.(μ,1.0))

KRR.loglikelihood(NormalLikelihood,Y1,link(IdentityLink,μ))

## Poisson data
Y2 = rand.(Poisson.(exp.(μ)))

KRR.loglikelihood(PoissonLikelihood,Y2,link(LogLink,μ))

## Exponential data
Y3 = rand.(Exponential.(exp.(μ)))

KRR.loglikelihood(ExponentialLikelihood,Y3,link(LogLink,μ))


m1 = fit(KRRModel,RBFKernel(1.0),NormalLikelihood,IdentityLink,X,Y1,1.0,1.0,verbose=true)
m2 = fit(KRRModel,RBFKernel(1.0),PoissonLikelihood,LogLink,X,Y2,1.0,1.0,verbose=true,rank=N-1)
m3 = fit(KRRModel,RBFKernel(1.0),ExponentialLikelihood,LogLink,X,Y3,1.0,1.0,verbose=true)



