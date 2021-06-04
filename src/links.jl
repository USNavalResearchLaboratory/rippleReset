export IdentityLink, LogLink, LogitLink

abstract type Link
end

struct IdentityLink <: Link
end

# This computes g⁻¹(η)
link(::Type{IdentityLink},η) = η

# This computes g(μ)
inv(::Type{IdentityLink},μ) = μ

gradient(::Type{IdentityLink},η) = ones(eltype(η),length(η))
hessian(::Type{IdentityLink},η)  = zeros(eltype(η),length(η))

struct LogLink <: Link
end

link(::Type{LogLink},η) = exp.(η)
inv(::Type{LogLink},μ)  = log.(μ)

gradient(::Type{LogLink},η) = exp.(η)
hessian(::Type{LogLink},η)  = exp.(η)

struct LogitLink <: Link
end

link(::Type{LogitLink},η) = 1 ./ (1 .+ exp.(η))
inv(::Type{LogitLink},μ) = log.(μ ./ (1-μ))

gradient(::Type{LogitLink},η) = link(LogitLink,η) .* (1 .- link(LogitLink,η))
hessian(::Type{LogitLink},η) = gradient(LogitLink,η) .* (1 .- 2 .* link(LogitLink,η))
