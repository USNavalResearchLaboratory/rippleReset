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
julia> data = map(read_forcing,sites)
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





