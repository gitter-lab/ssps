

using LRUCache
using Combinatorics
using LinearAlgebra
import Gen: Distribution, random, logpdf

"""
    GraphPrior

Implementation of the graph prior described in
Hill et al. 2012:
    P(G) ~ exp(-lambda * |G \\ G'|)
"""
struct GraphPrior <: Distribution{PSDiGraph} end
const graphprior = GraphPrior()

random(gp::GraphPrior, lambda::Float64, reference_graph::PSDiGraph) = reference_graph

function graph_edge_diff(g::PSDiGraph, g_ref::PSDiGraph)
    e1 = edges(g)
    eref = edges(g_ref)
    #e1 = Set([g.edges[i,:] for i=1:size(g.edges)[1]])
    #e_ref = Set([g_ref.edges[i,:] for i=1:size(g_ref.edges)[1]])
    return length(setdiff(e1, eref))
end

logpdf(gp::GraphPrior, graph::PSDiGraph, lambda::Float64, reference_graph::PSDiGraph) = -lambda * graph_edge_diff(graph, reference_graph)

graphprior(lambda::Float64, reference_graph::PSDiGraph) = random(graphprior, lambda, reference_graph);




# We use a few levels of LRU caching to reduce redundant computation.
const TENMB = 10000000

# Top level: store computed log-marginal likelihoods: log P(x_i | parents(x_i))
lp_cache = LRU{Vector{Int64}, Float64}(maxsize=10*TENMB, by=Base.summarysize)

# Middle level: store the value of B inv(B^T B) B^T;
# may be reused when different children have the same parents.
B2invconj_cache = LRU{Vector{Int64},Array{Float64,2}}(maxsize=100*TENMB, by=Base.summarysize)

# Bottom level: store B columns. B columns may be reused by different B matrices
# (so caching may be useful). But the number of possible columns is
# enormous, so an LRU policy is called for.
B_col_cache = LRU{Vector{Int64},Vector{Float64}}(maxsize=100*TENMB, by=Base.summarysize)

function clear_caches()
    empty!(lp_cache)
    empty!(B2invconj_cache)
    empty!(B_col_cache)
end

###########################################
# DBN MARGINAL LIKELIHOOD HELPER FUNCTIONS
###########################################
function compute_B_col(inds::Vector{Int64}, Xminus)
    return prod(Xminus[:,inds], dims=2)[:,1]
end


function construct_B(parent_inds::Vector{Int64}, Xminus::Array{Float64,2}, regression_deg::Int64)

    n_cols = sum([binomial(length(parent_inds), k) for k=1:min(regression_deg,length(parent_inds))])
    #println("N_COLS: ", n_cols)
    B = zeros(size(Xminus)[1], n_cols)

    col = 1
    for deg=1:regression_deg
        for comb in Combinatorics.combinations(parent_inds, deg)
            B[:,col] = get!(B_col_cache, comb) do
                return compute_B_col(comb, Xminus)
            end
            col += 1
        end
    end
    return B
end


function construct_B2invconj(parent_inds::Vector{Int64}, Xminus::Array{Float64,2}, regression_deg::Int64)
    if length(parent_inds) == 0
        return Matrix{Float64}(I, size(Xminus)[1], size(Xminus)[1])
    end
    B = construct_B(parent_inds, Xminus, regression_deg)
    B2inv = LinearAlgebra.inv(LinearAlgebra.Symmetric(transpose(B) * B + 0.001*I))
    return B * (B2inv * transpose(B))
end



"""
This log marginal likelihood is used by the "whole-graph" model
formulation. `Xplus` is the whole array of forward-time data values.
"""
function log_marg_lik(ind::Int64, parent_inds::Vector{Int64},
                      Xminus::Array{Float64,2}, Xplus::Array{Float64,2}, regression_deg::Int64)
    Xp = Xplus[:, ind]
    return log_marg_lik(parent_inds, Xminus, Xp, regression_deg)
end

"""
This log marginal likelihood is used by the "vertex-specific" model
formulation. `Xp` is a single column of forward-time data values. 
"""
function log_marg_lik(parent_inds::Vector{Int64},
		      Xminus::Array{Float64,2}, Xp::Vector{Float64}, 
		      regression_deg::Int64)

    B2invconj = get!(B2invconj_cache, parent_inds) do
        construct_B2invconj(parent_inds, Xminus, regression_deg)
    end
    n = size(Xp)[1]
    m = length(parent_inds)
    Bwidth = sum([Base.binomial(m, i) for i=1:min(m,regression_deg)])
    return -0.5*Bwidth*log(1.0 + n) - 0.5*n*log( dot(Xp,Xp) - (1.0*n/(n+1))*dot( Xp, B2invconj * Xp))
    
end

function get_parent_vecs(graph::PSDiGraph, vertices)
    return [convert(Vector{Int64}, sort([indexin([n], vertices)[1] for n in in_neighbors(graph, v)])) for v in vertices]
end

#######################
# END HELPER FUNCTIONS
#######################


####################################################
# Marginal Likelihood: "Entire-graph" formulation
####################################################
"""
    DBNMarginal

Implementation of the DBN marginal distribution
described in Hill et al. 2012:

P(X|G) ~ (1.0 + n)^(-Bwidth/2) * ( X+^T X+ - n/(n+1) * X+^T B2invconj X+)^(n/2)

This is expensive to compute. In order to reduce redundant
comptuation, we use multiple levels of caching.
"""
struct DBNMarginal <: Distribution{Vector{Array{Float64,2}}} end
const dbnmarginal = DBNMarginal()


"""
DBNMarginal's sampling method does nothing.
In our inference task, the Xs will always be observed.
"""
random(dbnm::DBNMarginal, parents::Vector{Vector{Int64}}, 
       Xminus::Array{Float64,2}, Xplus::Array{Float64,2}, 
       regression_deg::Int64) where T = [zeros(length(parents), length(Xminus))]

dbnmarginal(parents, Xminus, Xplus, regression_deg) = random(dbnmarginal, 
							     parents, 
							     Xminus, 
							     Xplus, 
							     regression_deg)


"""
DBNMarginal's log_pdf, in effect, returns a score for the
network topology. We use a dictionary to cache precomputed terms of the sum.
"""
function logpdf(dbn::DBNMarginal, X::Vector{Array{Float64,2}},
                parents::Vector{Vector{Int64}}, 
		Xminus::Array{Float64,2}, Xplus::Array{Float64,2}, 
		regression_deg::Int64)

    lp = 0.0
    for i=1:length(parents)
        lp += get!(lp_cache, [[i]; parents[i]]) do
           log_marg_lik(i, parents[i], Xminus, Xplus, regression_deg)
        end
    end

    return lp
end


##################################################
# Marginal likelihood: "Vertex-wise" formulation
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
    lp = get!(lp_cache, [[ind]; parents]) do
        log_marg_lik(parents, Xminus, Xp, regression_deg)
    end
    return lp
end

