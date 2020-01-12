# dbn_distributions.jl
# 2019-11-12
# David Merrell
#
#

include("dbn_distributions.jl")


######################################
# IMPLEMENTATION OF GRAPH PRIOR
######################################
#
# P(G) \propto exp(-lambda * sum( z_ij, ij not in reference graph ))
#

#"""
#Sample z_ij, a variable representing the existence of edge i-->j.
#"""
#@gen function sample_edge(ref_adj_entry::Bool, prob::Float64)
#
#    # If this edge is *in* the reference graph:
#    if ref_adj_entry != 0
#        z = @trace(Gen.bernoulli(0.5), :z)
#    # If this edge is not in the reference graph:
#    else
#        z = @trace(Gen.bernoulli(prob), :z)
#    end
#
#    return z
#end
#
## Two `Map` applications generate an
## entire adjacency matrix of sampled edges.
## The first Map creates a single parent set.
#sample_parents = Gen.Map(sample_edge)
#sample_children = Gen.Map(sample_parents)
#
#
#"""
#Sample an adjacency matrix from the graph prior:
#
#P(G) propto exp(-lambda * sum( z_ij, ij not in reference graph ))
#"""
#@gen (static) function graph_edge_prior(reference_adj::Vector{Vector{Bool}}, lambda::Float64)
#
#    V = size(reference_adj)[1]
#    enl = exp(-1.0*lambda)
#    probs = fill(fill(enl/(1.0 + enl), V), V)
#    edges = @trace(sample_children(reference_adj, probs), :edges)
#
#    return edges 
#
#end

@gen (static) function sample_parents(V::Int64, ref_parents::Vector{Int64},
                                                lambda::Float64)
    parents = @trace(parentprior(V, ref_parents, lambda), :parents)
    return parents
end
graph_prior = Gen.Map(sample_parents)


"""

"""
@gen (static) function generate_Xp(ind, Xp_prev, Xminus, adj, regression_deg)
    return @trace(cpdmarginal(Xminus, ind, 
                              adj[ind], 
			      regression_deg), :Xp)
end
generate_Xplus = Gen.Unfold(generate_Xp)

#function convert_to_vec(adj)::Vector{Vector{Bool}}
#	return [[Bool(u) for u in v] for v in adj] 
#end

import Base: getindex, length

mutable struct SingletonVec{T}
    item::T
    len::Int64
end

getindex(sa::SingletonVec, idx) = sa.item
length(sa::SingletonVec) = sa.len

"""
Model the data-generating process:
P(X,G,lambda | G') = P(X|G) * P(G|lambda, G') * P(lambda)
"""
@gen (static) function dbn_model(reference_adj::Vector{Vector{Bool}}, 
				 Xminus::Array{Float64,2}, 
				 Xplus::Vector{Vector{Float64}},
				 lambda_max::Float64,
				 regression_deg::Int64)

    lambda = @trace(Gen.uniform(0.0, lambda_max), :lambda)

    V = length(Xplus)
    Vvec = SingletonVec(V,V)
    lambda_vec = SingletonVec(lambda,V)

    parent_sets = @trace(graph_edge_prior(Vvec, reference_adj, lambda_vec), :parent_sets)

    Xpl = @trace(generate_Xplus(V, Vector{Float64}(), Xminus, parent_sets, regression_deg), :Xplus)
    return Xpl

end


