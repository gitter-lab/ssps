# dbn_distributions.jl
# 2019-11-12
# David Merrell
#
#



include("dbn_distributions.jl")

@gen (static) function dbn_single_context(X::Vector{Array{Float64,2}}, reference_graph::PSDiGraph, deg_max::Int64)

    V = sort(collect(vertices(reference_graph)))
    lambda = @trace(Gen.gamma(1,1), :lambda)

    G = @trace(graphprior(lambda, reference_graph), :G)

    parent_vecs = get_parent_vecs(G, V)
    Xcomb = combine_X(X)
    Xminus = Xcomb[1]
    Xplus = Xcomb[2]

    @trace(dbnmarginal(parent_vecs, Xminus, Xplus, deg_max), :X)

end
