The main interface for kernel ridge Poisson regression is

```julia
fit(KRRModel,X,Y,Δ;verbose=false,rank=5)
```

which fits a reduced-rank kernel ridge regression with a Poisson likelihood to the data in matrix `X` and integer vector `Y`. A radial basis function kernel is used by default and the kernel length scale and ridge regularization parameter are chosen by optimizing a penalized restricted maximum likelihood criterion. `Δ` can be used to add an arbitrary scale to the Poisson intensity function. For temporal point process modeling, `Δ` is the log of the time step.
