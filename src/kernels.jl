export Kernel, RBFKernel, InhomogeneousPolynomialKernel, WindowedRBFKernel

abstract type Kernel
end

struct RBFKernel <: Kernel
    L
end

(k::RBFKernel)(X,Y,L=k.L) = exp.(-0.5 .* pairwise(SqEuclidean(),X,Y,dims=1)./L^2)
(k::RBFKernel)(X) = exp.(-0.5 .* pairwise(SqEuclidean(),X,dims=1) ./ k.L^2)

struct InhomogeneousPolynomialKernel <: Kernel
    c
    d
end

(k::InhomogeneousPolynomialKernel)(X,Y,c=k.c,d=k.d) = (c * X*Y' .+ 1) .^ d

struct WindowedRBFKernel <: Kernel
    C
    L
end

spherical(r,C) = 1 - 1.5 * r/C + 0.5 * (r/C)^3

function (k::WindowedRBFKernel)(X,Y)
    C,L=k.C,k.L
    D = pairwise(Euclidean(),X,Y,dims=1)
    K = exp.(-0.5 .* (D./L).^2)
    W = spherical.(D,C)
    W .* K
end

function (k::WindowedRBFKernel)(X)
    C,L = k.C,k.L
    D = pairwise(Euclidean(),X,dims=1)
    K = exp.(-0.5 .* (D./L).^2)
    W = spherical.(D,C)
    W .* K
end
