

using LRUCache
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


function construct_B(parent_inds::Vector{Int64}, Xminus::Array{Float64,2}, deg_max::Int64)

    n_cols = sum([binomial(length(parent_inds), k) for k=1:min(deg_max,length(parent_inds))])
    #println("N_COLS: ", n_cols)
    B = zeros(size(Xminus)[1], n_cols)

    col = 1
    for deg=1:deg_max
        for comb in Combinatorics.combinations(parent_inds, deg)
            B[:,col] = get!(B_col_cache, comb) do
                return compute_B_col(comb, Xminus)
            end
            col += 1
        end
    end
    return B
end


function construct_B2invconj(parent_inds::Vector{Int64}, Xminus::Array{Float64,2}, deg_max::Int64)
    if length(parent_inds) == 0
        return Matrix{Float64}(I, size(Xminus)[1], size(Xminus)[1])
    end
    B = construct_B(parent_inds, Xminus, deg_max)
    B2inv = LinearAlgebra.inv(LinearAlgebra.Symmetric(transpose(B) * B + 0.01*I))
    return B * (B2inv * transpose(B))
end


function combine_X(X::Vector{Array{Float64,2}})

    len = sum([size(x)[1] - 1 for x in X])
    wid = size(X[1])[2]

    Xminus = zeros(len, wid)
    Xplus = zeros(len, wid)

    l_ind = 1
    for x in X
        r_ind = l_ind + size(x)[1] - 2
        Xminus[l_ind:r_ind, :] = x[1:size(x)[1]-1, :]
        Xplus[l_ind:r_ind, :] = x[2:size(x)[1], :]
        l_ind = r_ind + 1
    end

    mus = Statistics.mean(Xminus, dims=1)
    sigmas = statistics.std(Xminus, dims=1)

    Xminus = (Xminus .- mus) ./ sigmas
    Xplus = (Xplus .- mus) ./ sigmas

    return Xminus, Xplus
end


function log_marg_lik(ind::Int64, parent_inds::Vector{Int64},
                      Xminus::Array{Float64,2}, Xplus::Array{Float64,2}, deg_max::Int64)

    B2invconj = get!(B2invconj_cache, parent_inds) do
        construct_B2invconj(parent_inds, Xminus, deg_max)
    end
    Xp = Xplus[:, ind]
    n = size(Xplus)[1]
    m = length(parent_inds)
    Bwidth = sum([Base.binomial(m, i) for i=1:min(m,deg_max)])
    return -0.5*Bwidth*log(1.0 + n) - 0.5*n*log( dot(Xp,Xp) - (1.0*n/(n+1))*dot( Xp, B2invconj * Xp))

end

function get_parent_vecs(graph::PSDiGraph, vertices)
    return [convert(Vector{Int64}, sort([indexin([n], vertices)[1] for n in in_neighbors(graph, v)])) for v in vertices]
end

#######################
# END HELPER FUNCTIONS
#######################

"""
DBNMarginal's sampling method does nothing.
In our inference task, the Xs will always be observed.
"""
random(dbnm::DBNMarginal, parents::Vector{Vector{T}}, Xminus::Array{Float64,2}, Xplus::Array{Float64,2}, deg_max::Int64) where T = [zeros(length(parents), length(Xminus))]

dbnmarginal(parents, Xminus, Xplus, deg_max) = random(dbnmarginal, parents, Xminus, Xplus, deg_max)


"""
DBNMarginal's log_pdf, in effect, returns a score for the
network topology. We use a dictionary to cache precomputed terms of the sum.
"""
function logpdf(dbn::DBNMarginal, X::Vector{Array{Float64,2}},
                    parents::Vector{Vector{Int64}}, Xminus::Array{Float64,2}, Xplus::Array{Float64,2}, deg_max::Int64)

    lp = 0.0
    for i=1:length(parents)
        lp += get!(lp_cache, [[i]; parents[i]]) do
           log_marg_lik(i, parents[i], Xminus, Xplus, deg_max)
        end
    end

    return lp
end
