export read_forcing

"""
    read_forcing(forcing_file)

Read a RippleReset forcing file and return the ripple reset number and the
resets.
"""
function read_forcing(forcing_file)
    D = CSV.read(forcing_file,DataFrame)
    Λ = D[!,:Λ]
    Y = D[!,:reset2]
    Λ,Y
end
