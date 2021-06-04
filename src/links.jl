export IdentityLink, LogLink, LogitLink

abstract type Link
end

struct IdentityLink <: Link
end

# This computes g⁻¹(η)
link(::Type{IdentityLink},η) = η

# This computes g(μ)
inv(::Type{IdentityLink},μ) = μ

struct LogLink <: Link
end

link(::Type{LogLink},η) = exp.(η)
inv(::Type{LogLink},μ)  = log.(μ)

struct LogitLink <: Link
end

link(::Type{LogitLink},η) = 1 ./ (1 .+ exp.(η))
inv(::Type{LogitLink},μ) = log.(μ ./ (1-μ))
