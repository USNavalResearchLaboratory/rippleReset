# RippleReset.jl

This Julia package supplies code for fitting a kernel ridge Poisson regression to time series, with particular application to modeling ripple resets.

## Installation

#### HPC

If you are working on HPC, RippleReset.jl is registered in the [NSEARegistry](https://gitlab.hpc.mil/William.Kearney.ctr/NSEARegistry). Follow the instructions there to add the NSEARegistry to your Julia registries list, and then open a Julia REPL and enter the package manager using

```julia
julia> ]
```

Next add the RippleReset package

```julia
pkg> add RippleReset
```

This should download the RippleReset repository and its dependencies.

## Usage

First, load the package into Julia:

```julia
julia> using RippleReset
```

### Loading data

To start with, you need forcing data in the form of a time series of the ripple reset number and a time series that identifies when the ripple resets occur. If you have a `forcing.csv` as supplied in the [RippleResetForcing](https://gitlab.hpc.mil/William.Kearney.ctr/rippleresetforcing) repository, you can load the required time series using `read_forcing`

```julia
julia> lambda, resets = read_forcing("forcing.csv")
```

where `lambda` is the time series of ripple reset number and `resets` is the time series identifying the ripple resets.

If you want to train the model with more than one forcing file, you will want to load the data into two vectors of vectors, one for the ripple reset number and one for the resets. For example, in the RippleResetForcing repository layout, each site has a `forcing.csv` file in a dedicated directory, we can map `read_forcing` over the list of sites to load the data


```julia
julia> sites = ["WQS1406",
                "WQS1409",
                "TREX20m",
                "TREX8m",
                "ASIS",
                "GA",
                "LongBay",
                "MVCO02",
                "MVCO05",
                "LEO95"]
julia> data = map(read_forcing,[joinpath(RippleResetForcingDirectory,site,"forcing.csv") for site in sites])
```

The resulting `data` is a vector of tuples, where the first element is the ripple reset number time series for each site and the second element is the time series of resets. We can extract each of these by mapping `first` and `last` over the `data` vector

```julia
julia> lambda = map(first,data)
julia> resets = map(last,data)
```

Now `lambda` has type `Vector{Vector{Float64}}` and `resets` has type `Vector{Vector{Bool}}`. 

If you are training a model with more than one time series, you should *not* load the data for each site and then concatenate the resulting `lambda` vectors and the `resets` vectors, because the lagged regression will then use data from another site as the first time steps of each site, rather than discarding the first few time steps as it should. Instead, follow the instructions above to load vectors of vectors, and the model fitting functions will handle combining the data from multiple sites.

### Fitting the model

Once you have the forcing data, you fit the model using something like the following

```julia
    julia> m = RippleReset.fit(RippleResetModel,LaggedRegression(k),lambda,resets,timestep,rank=5,verbose=false)
```

The first argument to `fit` is th `RippleResetModel` type (not a value with that type, but the type itself, simply `RippleResetModel`), which tells the `StatsBase.jl` backend that we are fitting a ripple reset model.

The second argument is a value of the `LaggedRegression` type, which tells you how many lagged time steps `k` to include in the regression. If you set `k = 9`, for example, the regression will be fit to the present time step and the past 9 time steps for 10 steps in total.

The third and fourth arguments are the time series `lambda` and `resets` that you obtained in the previous step. These can be either individual vectors if you are fitting to a single time series or vectors of vectors if you are fitting to multiple time series.

The fifth argument is the time step of the time series in seconds. If you are fitting a model to multiple time series, they should all have the same time step.

Finally there are two keyword arguments. `rank` determines the size of the reduced-order model that is fit to the data. The default value is `5`. Higher values allow more flexibility in the model, but require more computational time. `verbose` displays logging information during the optimization. This is not usually necessary.

### Using the model

With a fit model, you can compute the intensity function for a new ripple reset number time series

```julia
julia> test_gamma = intensity(m,test_lambda)
```

`test_gamma` is the integrated intensity function output by the model. It will be a vector with `k` fewer time steps than `test_lambda` because the first `k` time steps are undefined for the lagged regression.

You can also simulate time series of resets from the model using

```julia
julia> test_sims = simulate(m,test_lambda,B)
```

where `B` is the desired number of simulations. The result, `test_sims`, is a vector of vectors, with each element giving a simulated reset time series with `k` fewer time steps than the `test_lambda` time series for the same reason as above.

Finally, you can compute the log likelihood of a data set under the model as

```julia
julia> ll = RippleReset.loglikelihood(m,test_lambda,test_resets)
```

where `test_lambda` is the ripple reset number time series as above and `test_resets` is the corresponding time series identifying the ripple reset events (as might be generated by `read_forcing`).

### Bootstrapping

One way to estimate the uncertainty in the model's predictions is with bootstrapping, in which we replace obtaining new data by simulation. The simplest way to bootstrap the predictions of the ripple reset model is by using a model-based bootstrap that draws samples from the fitted Poisson process. If we have a model `m::RippleResetModel`, we can draw `B` samples from it given a ripple reset number time series `test_lambda` using `bootstrap_simulate`

```julia
julia> bootstrap_resets = bootstrap_simulate(m,test_lambda,B)
```

Finally, if you do not trust the model you fitted, you might want to avoid simulating bootstrap samples from it, and instead resample the data directly. Since the kernel ridge regression assumes that each time step is independent, but dependent on the last `k` time steps, we can do this by first forming vectors consisting of the forcing at each time step and the `k` previous time steps, and then resampling these vectors along with the corresponding reset values at each time step. The resulting resampled time series is not realistic, but it does not matter for the regression model that we use to fit ripple reset models. We use `bootstrap_resample` to resample the data, but we do not need a fitted `RippleResetModel`. Instead, we give it a `LaggedRegression` design and both the ripple reset number and the reset event time series:

```julia
julia> resampled_data = bootstrap_resample(LaggedRegression(k),lambda,resets,B)
```

Once you have the bootstrap samples, acquired by either `bootstrap_simulate` or `resampled_data`, you probably want to fit a new RippleResetModel to each of them. In the case of the `bootstrap_simulate` samples, you might do something like

```julia
julia> bootstrap_models = [RippleReset.fit(RippleResetModel,LaggedRegression(k),test_lambda,reset,timestep,rank=5,verbose=false) for reset in bootstrap_resets]
```

while with `bootstrap_resample` samples, you want something like

```julia
julia> bootstrap_models = [RippleReset.fit(RippleResetModel,LaggedRegression(k),lambda,reset,timestep,rank=5,verbose=false) for (lambda,reset) in resampled_data]
```

The difference is because the model-based bootstrap uses the test forcing information  while the resampling bootstrap also has to resample the forcing data.

## Example

The bootstrap comparison tests used to generate the histograms in the Ripple Reset manuscript can be implemented as follows:

```julia
function comparison_test(null_site_forcing,test_site_forcing,B)
    null_data = map(read_forcing,null_site_forcing)
    null_lambda = map(first,null_data)
    null_resets = map(last,null_data)
    
    test_data = map(read_forcing,test_site_forcing)
    test_lambda = map(first,test_data)
    test_resets = map(last,test_data)

    design = LaggedRegression(9)
    # Fit model to null data
    null_model = RippleReset.fit(RippleResetModel,design,null_lambda,null_resets,3600.0,rank=5,verbose=false)
    
    # Fit model to test data
    test_model = RippleReset.fit(RippleResetModel,design,test_lambda,test_resets,3600.0,rank=5,verbose=false)

    # Likelihood difference on the test data.
    ls0 = 2*(loglikelihood(test_model,test_lambda,test_resets) - loglikelihood(null_model,test_lambda,test_resets))

    # Simulate from the null model with the null forcing
    null_bootstraps = bootstrap_simulate(null_model,null_lambda,B)

    # Simulate from the null model with the test forcing
    test_bootstraps = bootstrap_simulate(null_model,test_lambda,B)
    
    # Fit models to the bootstrap samples with the null forcing
    null_bootmodel = map(null_bootstraps) do reset
        RippleReset.fit(RippleResetModel,design,null_lambda,reset,3600.0,rank=5,verbose=false)
    end

    test_bootmodel = map(test_bootstraps) do reset
        RippleReset.fit(RippleResetModel,design,test_lambda,reset,3600.0,rank=5,verbose=false)
    end

    # Log-likelihood difference between the model fitted to the simulated data and the null model
    ls = map(zip(null_bootmodel,test_bootmodel)) do (null_model,test_model)
        2*(RippleReset.loglikelihood(test_model,test_lambda,test_reset) -
           RippleReset.loglikelihood(null_model,test_lambda,test_reset))
    end

    ls0,ls
end
```

The inputs are a list of forcing files for the null model, a list of forcing files for the test model and a number of bootstrap samples. The outputs are the difference in log likelihoods between the test and null models, `ls0` and the bootstrapped values of the difference in log likelihoods `ls`. The histograms in the figures are histograms of `ls` and the red line is the value of `ls0`.
