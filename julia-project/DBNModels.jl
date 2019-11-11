
@gen (static) function full_model(X::Vector{Array{Float64,2}}, reference_graph::DiGraph, deg_max::Int64)
    
    V = sort(collect(reference_graph.vertices))
    lambda = @trace(Gen.gamma(1,1), :lambda)
    
    G = @trace(graphprior(lambda, reference_graph), :G)
    
    parent_vecs = get_parent_vecs(G, V)
    Xcomb = combine_X(X)
    Xminus = Xcomb[1]
    Xplus = Xcomb[2]
    
    @trace(dbnmarginal(parent_vecs, Xminus, Xplus, deg_max), :X)
    
end
