
"""
    DiGraph(edgelist::Array{UInt16, 2}, V::UInt16)

Construct a directed graph with a given edgelist and number of vertices.
"""
mutable struct DiGraph
    edgelist::Array{UInt16, 2}
    V::UInt16
end


"""
    DiGraph(V::Int)

Construct a DiGraph with known number of vertices (V) and empty edgelist.
"""
DiGraph(V::Int) = DiGraph(Array{UInt16, 2}(undef, (0,2)), UInt16(V))


"""
    DiGraph()

Construct a DiGraph with no vertices and no edges.
"""
DiGraph() = DiGraph(0::UInt16)


"""
    add_edge(dg::DiGraph, orig::UInt16, dest::UInt16)

Add a directed edge to the DiGraph: orig -> dest.

We always assume that the existence of a vertex labeled 'x'
implies the existence of all vertices from 1 through x.
For now, we also assume the user doesn't stupidly add multiple
copies of the same edge.
"""
function add_edge(dg::DiGraph, orig::Int, dest::Int)

    dg.edgelist = vcat(dg.edgelist, [UInt16(orig) UInt16(dest)])
    dg.V = maximum([dg.V; orig; dest])

end
