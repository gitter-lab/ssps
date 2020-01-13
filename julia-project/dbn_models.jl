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
@gen (static) function generate_Xp(ind, Xminus, parents, regression_deg)
    return @trace(cpdmarginal(Xminus, ind, 
                              parents, 
			      regression_deg), :Xp)
end
generate_Xplus = Gen.Map(generate_Xp)


# This type helps us use the `Map` combinator efficiently
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
@gen (static) function dbn_model(reference_adj::Vector{Vector{Int64}}, 
				 Xminus::Array{Float64,2}, 
				 Xplus::Vector{Vector{Float64}},
				 lambda_max::Float64,
				 regression_deg::Int64)

    lambda = @trace(Gen.uniform(0.0, lambda_max), :lambda)

    V = length(Xplus)
    Vvec = SingletonVec(V, V)
    lambda_vec = SingletonVec(lambda, V)

    parent_sets = @trace(graph_prior(Vvec, reference_adj, lambda_vec), :parent_sets)

    Xminus_vec = SingletonVec(Xminus, V)
    regdeg_vec = SingletonVec(regression_deg, V)
    Xpl = @trace(generate_Xplus(1:V, Xminus_vec, parent_sets, regdeg_vec), :Xplus)
    return Xpl

end


