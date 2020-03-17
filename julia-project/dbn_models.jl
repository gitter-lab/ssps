# dbn_distributions.jl
# 2019-11-12
# David Merrell
#
#

include("dbn_distributions.jl")


######################################
# IMPLEMENTATION OF GRAPH PRIOR
######################################

# graph prior parameterized by (1) reference graph and (2) lambda
# P(G | G', lambda) \propto exp(-lambda * sum( z_ij, ij not in reference graph ))
@gen (static) function ref_parents_prior(V::Int64, ref_parents::Vector{Int64},
                                                lambda::Float64)
    parents = @trace(refparentprior(V, ref_parents, lambda), :parents)
    return parents
end
ref_graph_prior = Gen.Map(ref_parents_prior)


# graph prior parameterized by (1) edge confidences and (2) lambda;
# a "smooth" version of the other graph prior.
@gen (static) function conf_parents_prior(V::Int64, parent_confs::Dict,
                                          lambda::Float64)
    parents = @trace(confparentprior(V, parent_confs, lambda), :parents)
    return parents
end
conf_graph_prior = Gen.Map(conf_parents_prior)

"""

"""
@gen (static) function generate_Xp(ind, Xminus, parents, regression_deg)
    return @trace(cpdmarginal(Xminus, ind, parents, regression_deg), :Xp)
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
@gen (static) function dbn_model(reference_parents::Vector{Vector{Int64}}, 
                                 Xminus::Array{Float64,2}, 
                                 lambda_min::Float64,
                                 lambda_max::Float64,
                                 regression_deg::Int64)

    lambda = @trace(Gen.uniform(lambda_min, lambda_max), :lambda)

    V = shape(Xminus, 2)
    Vvec = SingletonVec(V, V)
    lambda_vec = SingletonVec(lambda, V)

    parent_sets = @trace(ref_graph_prior(Vvec, reference_parents, lambda_vec), :parent_sets)

    Xminus_vec = SingletonVec(Xminus, V)
    regdeg_vec = SingletonVec(regression_deg, V)

    Xpl = @trace(generate_Xplus(1:V, Xminus_vec, parent_sets, regdeg_vec), :Xplus)
    return Xpl

end


"""
A variant of `dbn_model` which allows prior knowledge in the form of 
real-valued 'edge confidences', rather than boolean edge existences.

We've made an entirely new function in order to avoid `if` statements,
which are incompatible with Gen's `static` modeling language.
"""
@gen (static) function conf_dbn_model(parent_confs::Vector{Dict}, 
                                      Xminus::Array{Float64,2}, 
                                      lambda_min::Float64,
                                      lambda_max::Float64,
                                      regression_deg::Int64)

    lambda = @trace(Gen.uniform(lambda_min, lambda_max), :lambda)

    V = size(Xplus,2)
    Vvec = SingletonVec(V, V)
    lambda_vec = SingletonVec(lambda, V)

    parent_sets = @trace(conf_graph_prior(Vvec, parent_confs, lambda_vec), :parent_sets)

    Xminus_vec = SingletonVec(Xminus, V)

    regdeg_vec = SingletonVec(regression_deg, V)

    Xpl = @trace(generate_Xplus(1:V, Xminus_vec, parent_sets, regdeg_vec), :Xplus)
    return Xpl

end


@gen (static) function generate_lambda(l_min, l_max)
    return @trace(Gen.uniform(l_min, l_max), :lambda)
end
generate_lambda_vec = Gen.Map(generate_lambda)


"""
A variant of `conf_dbn_model` which has one lambda variable *per vertex*
(rather than a single lambda variable for the whole graph). 

We've made an entirely new function in order to avoid `if` statements,
which are incompatible with Gen's `static` modeling language.
"""
@gen (static) function vertex_lambda_dbn_model(parent_confs::Vector{Dict}, 
                                               Xminus::Array{Float64,2}, 
                                               lambda_min::Float64,
                                               lambda_max::Float64,
                                               regression_deg::Int64)

    V = shape(Xminus, 2)

    Vvec = SingletonVec(V, V)
    min_vec = SingletonVec(lambda_min, V)
    max_vec = SingletonVec(lambda_max, V)
    Xminus_vec = SingletonVec(Xminus, V)
    regdeg_vec = SingletonVec(regression_deg, V)
    
    lambda_vec = @trace(generate_lambda_vec(min_vec, max_vec), :lambda_vec)
    parent_sets = @trace(conf_graph_prior(Vvec, parent_confs, lambda_vec), :parent_sets)
    Xplus = @trace(generate_Xplus(1:V, Xminus_vec, parent_sets, regdeg_vec), :Xplus)
    return Xplus

end

