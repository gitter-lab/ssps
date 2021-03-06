# dbn_distributions.jl
# David Merrell
# 2019-11
#
# We define some probability distribution objects:
#     * prior distributions for parent sets
#     * a marginal likelihood function for Gaussian DBNs

using LRUCache
using Combinatorics
using LinearAlgebra
import Distributions: Bernoulli
import Gen: Distribution, random, logpdf

###########################################
# PARENT SET PRIOR DISTRIBUTIONS
###########################################

# This distribution assumes prior knowledge in the form of
# "reference edges" -- roughly speaking, it assigns uniform probability
# to all subsets of the reference parent set.
struct RefParentPrior <: Distribution{Vector{Int64}} end
const refparentprior = RefParentPrior()

refparentprior(V::Int64,
               ref_parents::Vector{Int64}, 
               lambda::Float64) = random(refparentprior,
                                         V, ref_parents, lambda)

function random(pp::RefParentPrior, V::Int64, 
                                    ref_parents::Vector{Int64}, 
                                    lambda::Float64)
    enl = exp(-lambda)
    p = enl /(1.0+enl)
    parents = Vector{Int64}()
    for v=1:V
        if v in ref_parents
            if Bool(rand(Bernoulli(0.5)))
                push!(parents, v)
            end
        else
            if Bool(rand(Bernoulli(p)))
                push!(parents, v)
            end
        end
    end

    return parents
end


function logpdf(pp::RefParentPrior, x_parents, V, ref_parents, lambda)

    lp = -log(1.0 + exp(-lambda))
    lhalf = -log(2)
    n_ref = length(ref_parents)
    total = n_ref*lhalf + (V - n_ref)*lp

    n_extra = length(setdiff(x_parents, ref_parents))
    total += n_extra*(-lambda + lp - lhalf)

    return total
end

# This distribution assumes prior knowledge in the form of 
# edge confidences -- it's a "smooth" variant of the 
# RefParentPrior distribution.
struct ConfParentPrior <: Distribution{Vector{Int64}} end
const confparentprior = ConfParentPrior()


confparentprior(idx, V, parent_confs::Dict,
                lambda::Float64) = random(confparentprior, idx, V, parent_confs, lambda)

function random(pp::ConfParentPrior, idx, V, parent_confs::Dict, lambda::Float64)

    enl = exp(-lambda)

    parents = Vector{Int64}()
    for i=1:V
        conf = get(parent_confs, i, 0.0)
        p = enl/(enl + exp(-conf * lambda))
        if Bool(rand(Bernoulli(p)))
            push!(parents, i)
        end
    end

    return parents
end

# We use LRU caching to reduce redundant computation.
TENMB = 10000000
global prior_cache = LRU{Vector{Int64}, Float64}(maxsize=10*TENMB, by=Base.summarysize)


function logpdf(pp::ConfParentPrior, x_parents::Vector{Int64}, 
				     idx::Int64,
				     V::Int64,
                                     parent_confs::Dict, 
                                     lambda::Float64)
    
    lp = get!(prior_cache, pushfirst!(copy(x_parents), idx)) do
    
        enl = exp(-lambda)
        ref_nonzero = keys(parent_confs)
        lp = 0.0

        # If the number of nonzero confidences is large,
        # then we use array computations
        if length(ref_nonzero) > 50

            # All zero-confidence vertices
            lp += -log(1.0 + enl) * (V - length(parent_confs))

            # All nonzero-confidence vertices
            npl = map(x-> -lambda * x, values(parent_confs)) 
            lp += sum(map(x-> x - log(enl + exp(x)), npl))

            # parents
            for parent in x_parents
                # nonzero-confidence parents
                if haskey(parent_confs, parent)
                    lp += -lambda * (1.0 - parent_confs[parent]) 
                # zero-confidence parents
                else 
                    lp += -lambda
                end
            end

        # Otherwise, we operate on the dictionaries
        else
            # parents 
            for parent in x_parents
                conf = get(parent_confs, parent, 0.0)
                lp += -lambda - log(enl + exp(-conf*lambda))
            end
            
            # nonparents AND ref_nonzero 
            for nonparent in setdiff(ref_nonzero, x_parents)
                npl = -parent_confs[nonparent]*lambda
                lp += npl - log(enl + exp(npl))
            end 
            
            # nonparents AND NOT ref_nonzero
            zero_n_lp = - log(1.0 + enl)
            lp += zero_n_lp*(V - length(union(ref_nonzero, x_parents)))
        end

        return lp
    end

    return lp
end


###########################################
# DBN MARGINAL LIKELIHOOD HELPERS
###########################################

# We use LRU caching to reduce redundant computation.
global lml_cache = LRU{Vector{Int64}, Float64}(maxsize=10*TENMB, by=Base.summarysize)

function clear_caches()
    empty!(prior_cache)
    empty!(lml_cache)
end

function compute_B_col(inds::Vector{Int64}, Xminus)
    
    # Multiply the columns together
    col = prod(Xminus[:,inds], dims=2)[:,1]
    
    # The result ought to be standardized.
    col = (col .- mean(col)) ./std(col)
    return col
end


function construct_B(parent_inds::Vector{Int64}, Xminus::Array{Float64,2}, regression_deg::Int64)

    n_parents = length(parent_inds)
    n_cols = sum([binomial(n_parents, k) for k=1:min(regression_deg, n_parents)])
    B = zeros(size(Xminus)[1], n_cols)

    col = 1

    for deg=1:regression_deg
        for comb in Combinatorics.combinations(parent_inds, deg)
            B[:,col] = compute_B_col(comb, Xminus)
            col += 1
        end
    end
    return B
end


"""
Wrap Julia's linear solver so that singular matrices have
small "ridge" terms added to them until they become nonsingular.
""" 
function safe_solve(A, y; step=1e-6, max_steps=5)
    try
        return A\y
    catch e
        if isa(e, LinearAlgebra.SingularException)
            AA = transpose(A) * A + step*I
            Ay = transpose(A) * y
            steps = 1
            while steps <= max_steps
                try
                    solved = AA\Ay
                    return solved
                catch ee
                    if isa(ee, LinearAlgebra.SingularException)
                        AA += step*I
                        steps += 1
                        continue
                    else
                        throw(ee)
                    end
                end
            end
        else
            throw(e)
        end

    end
    throw(LinearAlgebra.SingularException)
end
 

"""

"""
function compute_lml(parent_inds::Vector{Int64},
		     Xminus::Array{Float64,2}, Xp::Vector{Float64},
		     regression_deg::Int64)
    
    B = construct_B(parent_inds, Xminus, regression_deg) 
    m = length(parent_inds)
    Bwidth = size(B,2)
    
    n = size(Xp)[1]

    ip = dot(Xp, Xp - (n/(n+1.0))*B*safe_solve(B, Xp))
    return -0.5*Bwidth*log(1.0 + n) - 0.5*n*log(ip)

end


"""
This log marginal likelihood is used by the "vertex-specific" model
formulation. `Xp` is a single column of forward-time data values. 
"""
function log_marg_lik(ind::Int64, parent_inds::Vector{Int64},
		      Xminus::Array{Float64,2}, Xp::Vector{Float64}, 
		      regression_deg::Int64)

    lml = get!(lml_cache, pushfirst!(copy(parent_inds), ind)) do
        compute_lml(parent_inds, Xminus, Xp, regression_deg)
    end

    return lml 
end

#######################
# END HELPER FUNCTIONS
#######################


##################################################
# Marginal likelihood
##################################################

"""
    CPDMarginal

Implementation of the regression marginal likelihood
described in Hill et al. 2012:

P(X|G) ~ (1.0 + n)^(-Bwidth/2) * ( X+^T X+ - n/(n+1) * X+^T B2invconj X+)^(n/2)

This is expensive to compute. In order to reduce redundant
comptuation, we use multiple levels of caching.
"""
struct CPDMarginal <: Distribution{Vector{Float64}} end
const cpdmarginal = CPDMarginal()

cpdmarginal(Xminus, ind, parents, regression_deg) = random(cpdmarginal,
							   Xminus,
							   ind, parents,
							   regression_deg)

"""
The random function does nothing -- we will always observe these values.
"""
function random(cpdm::CPDMarginal, Xminus::Array{Float64,2},
		ind::Int64, parents::Vector{Int64}, regression_deg::Int64)
    return zeros(size(Xminus)[1])
end


function logpdf(cpdm::CPDMarginal, Xp::Vector{Float64}, Xminus::Array{Float64,2},
		ind::Int64, parents::Vector{Int64}, regression_deg::Int64)
    return log_marg_lik(ind, parents, Xminus, Xp, regression_deg)
end

