using LRUCache
using Combinatorics
using LinearAlgebra
import Distributions: Bernoulli
import Gen: Distribution, random, logpdf
import FunctionalCollections: PersistentVector


###########################################
# PARENT SET PRIOR DISTRIBUTION
###########################################

struct ParentPrior <: Distribution{Vector{Int64}} end
const parentprior = ParentPrior()

parentprior(V::Int64,
            ref_parents::Vector{Vector{Int64}}, 
            lambda::Float64) = random(parentprior,
                                      V, ref_parents, lambda)

function random(pp::ParentPrior, V::Int64, 
                               ref_parents::Vector{Int64}, 
                               lambda::Float64)
    enl = exp(-lambda)
    p = enl /(1.0+enl)
    parents = Vector{Int64}()
    for v=1:V
        if v in ref_parents
            if rand(Bernoulli(0.5))
                push!(parents, v)
            end
        else
            if rand(Bernoulli(p))
                push!(parents, v)
            end
        end
    end

    return parents
end

function logpdf(pp::ParentPrior, x_parents, V, ref_parents, lambda)

    lp = -log(1.0 + exp(-lambda))
    lhalf = -log(2)
    n_ref = length(ref_parents)
    total = n_ref*lhalf + (V - n_ref)*lp

    n_extra = length(setdiff(x_parents, ref_parents))
    total += n_extra*(-lambda + lp - lhalf)

    return total
end


###########################################
# DBN MARGINAL LIKELIHOOD HELPERS
###########################################

# We use LRU caching to reduce redundant computation.
const TENMB = 10000000

global lp_cache = LRU{Vector{Int64}, Float64}(maxsize=10*TENMB, by=Base.summarysize)

function clear_caches()
    empty!(lp_cache)
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


function construct_B2invconj(parent_inds::Vector{Int64}, Xminus::Array{Float64,2}, regression_deg::Int64)
    if length(parent_inds) == 0
        return zeros(size(Xminus)[1], size(Xminus)[1])
    end
    B = construct_B(parent_inds, Xminus, regression_deg)
    B2inv = LinearAlgebra.inv(LinearAlgebra.Symmetric(transpose(B) * B + 0.001*I))
    return B * (B2inv * transpose(B))
end


"""

"""
function compute_lml(parent_inds::Vector{Int64},
		     Xminus::Array{Float64,2}, Xp::Vector{Float64},
		     regression_deg::Int64)
    
    B2invconj = construct_B2invconj(parent_inds, Xminus, regression_deg)
    
    m = length(parent_inds)
    Bwidth = sum([Base.binomial(m, i) for i=1:min(m,regression_deg)])
    
    n = size(Xp)[1]
    return -0.5*Bwidth*log(1.0 + n) - 0.5*n*log( dot(Xp,Xp) - (n/(n+1.0))*dot( Xp, B2invconj * Xp))

end


"""
This log marginal likelihood is used by the "vertex-specific" model
formulation. `Xp` is a single column of forward-time data values. 
"""
function log_marg_lik(ind::Int64, parent_inds::Vector{Int64},
		      Xminus::Array{Float64,2}, Xp::Vector{Float64}, 
		      regression_deg::Int64)

    lml = get!(lp_cache, pushfirst!(copy(parent_inds), ind)) do
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

