# dbn_distributions.jl
# 2019-11-12
# David Merrell
#
#

include("dbn_distributions.jl")


######################################
# IMPLEMENTATION OF GRAPH PRIOR
######################################

"""
parent set prior parameterized by (1) reference graph and (2) lambda
P(G | G', lambda) \\propto exp(-lambda * sum( z_ij, ij not in reference graph ))
"""
@gen (static) function ref_parents_prior(V::Int64, ref_parents::Vector{Int64},
#@gen function ref_parents_prior(V::Int64, ref_parents::Vector{Int64},
                                                lambda::Float64)
    parents = @trace(refparentprior(V, ref_parents, lambda), :parents)
    return parents
end
# A `Map` operation over vertices yields a graph prior distribution.
ref_graph_prior = Gen.Map(ref_parents_prior)


"""
parent set prior parameterized by (1) edge confidences and (2) lambda;
a "smooth" version of the other graph prior.
"""
@gen  function conf_parents_prior(V::Int64, parent_confs::Dict,
                                          lambda::Float64)
    parents = @trace(confparentprior(V, parent_confs, lambda), :parents)
    return parents
end
# A `Map` operation over vertices yields a graph prior distribution.
conf_graph_prior = Gen.Map(conf_parents_prior)


"""
`generate_Xp` models the linear relationship between a variable and its parents
in the time series data (i.e., this is the marginal likelihood function)
"""
@gen (static) function generate_Xp(ind::Int64, Xminus::Matrix{Float64}, 
#@gen function generate_Xp(ind::Int64, Xminus::Matrix{Float64}, 
                                   parents::Vector{Int64}, regression_deg::Int64)
    return @trace(cpdmarginal(Xminus, ind, parents, regression_deg), :Xp)
end
generate_Xplus = Gen.Map(generate_Xp)


# This `SingletonVec` type helps us use the 
# `Map` combinator more efficiently.
import Base: getindex, length
mutable struct SingletonVec{T}
    item::T
    len::Int64
end
getindex(sa::SingletonVec, idx) = sa.item
length(sa::SingletonVec) = sa.len

"""
Uniform prior distribution for the lambda variables
(inverse temperatures). 
"""
@gen (static) function generate_lambda(l_min::Float64, l_max::Float64)
#@gen function generate_lambda(l_min, l_max)
    return @trace(Gen.uniform(l_min, l_max), :lambda)
end
generate_lambda_vec = Gen.Map(generate_lambda)


"""
Model for the generative process, i.e.
P(G, Lambda|X) propto P(X|G) * P(G|Lambda) * P(Lambda)
"""
@gen (static) function vertex_lambda_dbn_model(parent_confs::Vector{Dict}, 
#@gen function vertex_lambda_dbn_model(parent_confs::Vector{Dict}, 
                                               Xminus::Array{Float64,2}, 
                                               lambda_min::Float64,
                                               lambda_max::Float64,
                                               regression_deg::Int64)

    V = size(Xminus, 2)

    # We make SingletonVecs out of these arguments
    # only so they can be shoe-horned into the `Map` operations.
    Vvec = SingletonVec(V, V)
    min_vec = SingletonVec(lambda_min, V)
    max_vec = SingletonVec(lambda_max, V)
    Xminus_vec = SingletonVec(Xminus, V)
    regdeg_vec = SingletonVec(regression_deg, V)

    # This is the actual substance of the probabilistic program
    lambda_vec = @trace(generate_lambda_vec(min_vec, max_vec), :lambda_vec)
    parent_sets = @trace(conf_graph_prior(Vvec, parent_confs, lambda_vec), :parent_sets)
    Xplus = @trace(generate_Xplus(1:V, Xminus_vec, parent_sets, regdeg_vec), :Xplus)
    return Xplus

end

