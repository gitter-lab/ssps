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


######################################
# IMPLEMENTATION OF GRAPH PRIOR
######################################
#
# P(G) \propto exp(-lambda * sum( z_ij, ij not in reference graph ))
#

"""
Sample z_ij, a variable representing the existence of edge i-->j.
"""
@gen function sample_edge(ref_adj_entry::Bool, probs::Float64)

    # If this edge is *in* the reference graph:
    if ref_adj_entry != 0
        z = @trace(Gen.bernoulli(0.5), :z)
    # If this edge is not in the reference graph:
    else
        z = @trace(Gen.bernoulli(probs), :z)
    end

    return z
end

# Two `Map` applications generate an
# entire adjacency matrix of sampled edges.
sample_parents = Gen.Map(sample_edge)
sample_children = Gen.Map(sample_row)

"""
Sample an adjacency matrix from the graph prior:

P(G) propto exp(-lambda * sum( z_ij, ij not in reference graph ))
"""
@gen (static) function graph_edge_prior(reference_adj::Vector{Vector{Bool}}, lambda::Float64)

    V = size(reference_adj)[1]
    enl = exp(-1.0*lambda)
    probs = fill(fill(enl/(1.0 + enl), V), V)
    adj = @trace(sample_children(reference_adj, probs), :adj)

    return adj

end



