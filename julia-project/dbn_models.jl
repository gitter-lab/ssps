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
sample_children = Gen.Map(sample_parents)

"""
Sample an adjacency matrix from the graph prior:

P(G) propto exp(-lambda * sum( z_ij, ij not in reference graph ))
"""
@gen (static) function graph_edge_prior(reference_adj::Vector{Vector{Bool}}, lambda::Float64)

    V = size(reference_adj)[1]
    enl = exp(-1.0*lambda)
    probs = fill(fill(enl/(1.0 + enl), V), V)
    edges = @trace(sample_children(reference_adj, probs), :edges)

    return edges 

end


"""

"""
@gen (static) function generate_Xp(Xminus, ind, parents, regression_deg, phi_ratio)
    return @trace(cpdmarginal(Xminus, ind, 
                              parents, 
			      regression_deg, phi_ratio), :Xp)
end

generate_Xplus = Gen.Map(generate_Xp)

function convert_to_vec(adj)::Vector{Vector{Bool}}
	return [[Bool(u) for u in v] for v in adj] 
end

"""
Model the data-generating process:
P(X,G,lambda | G') = P(X|G) * P(G|lambda, G') * P(lambda)
"""
@gen (static) function dbn_model(reference_adj::Vector{Vector{Bool}}, 
				 Xminus_stacked::Vector{Array{Float64,2}}, 
				 Xplus::Vector{Vector{Float64}},
				 lambda_prior_param::Float64,
				 regression_deg::Float64,
				 phi_ratio::Float64)

    lambda = @trace(Gen.exponential(lambda_prior_param), :lambda)

    adj = @trace(graph_edge_prior(reference_adj, lambda), :adjacency)
    adj = convert_to_vec(adj) 

    V = length(Xplus)
    Xpl = @trace(generate_Xplus(Xminus_stacked, collect(1:V),
                                adj, fill(regression_deg, V), fill(phi_ratio, V)), :Xplus)
    return Xpl

end


