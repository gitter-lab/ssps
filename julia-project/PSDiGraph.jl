# PSDiGraph.jl
# 2019-11-11
# David Merrell
#
# Simple digraph implementation based on parent-sets

module PSDiGraphs

import Base: copy, isapprox

export PSDiGraph, vertices, edges, add_edge!, remove_edge!, 
       add_vertex!, remove_vertex!, out_neighbors, in_neighbors, 
       dfs_traversal, bfs_traversal, is_cyclic, get_ancestors, 
       transpose!, topological_sort 

"""
    PSDiGraph{T}(parents::Dict{T,Set{T}})

A minimal directed graph parametric type.
"""
mutable struct PSDiGraph{T}
    parents::Dict{T,Set{T}}
end

#############################
# CONSTRUCTORS
#############################
PSDiGraph{T}() where T = PSDiGraph(Dict{T,Set{T}}())

function PSDiGraph(edge_array::Array{T,2}) where T 
    dg = PSDiGraph{T}()
    vertices = Set(edge_array)
    for v in vertices
	dg.parents[v] = Set{T}()
    end
    for i in size(edge_array)[1]
	push!(dg.parents[edge_array[i,1]], edge_array[i,2])
    end
    return dg
end

################################
# `Base` EXTENSIONS
################################
copy(dg::PSDiGraph{T}) where T = PSDiGraph{T}(copy(dg.parents))
isapprox(x::PSDiGraph{T}, y::PSDiGraph{T}) where T = (x.parents == y.parents)


################################
# INTERFACE FUNCTIONS
################################
"""
    vertices(dg::PSDiGraph{T})

Return the set of dg's vertices.
"""
function vertices(dg::PSDiGraph{T}) where T
    return Set(keys(dg.parents))
end


"""
    edges(dg::PSDiGraph{T})

Return a set of Pairs representing the graph's edges.
"""
function edges(dg::PSDiGraph{T}) where T
    result = Set{Pair{T,T}}()
    for (k, ps) in dg.parents
        for p in ps
	    push!(result, Pair(p, k))
	end
    end
    return result
end


"""
    add_edge!(dg::PSDiGraph{T}, orig::T, dest::T)

Add edge orig-->dest to the graph.   
"""
function add_edge!(dg::PSDiGraph{T}, orig::T, dest::T) where T
    if dest in keys(dg.parents)
	push!(dg.parents[dest], orig)
    else
	dg.parents[dest] = Set([orig])
    end
    if !(orig in keys(dg.parents))
	dg.parents[orig] = Set{T}()
    end
end


"""
    remove_edge!(dg::PSDiGraph{T}, orig::T, dest::T)

Remove edge orig-->dest from the graph.
"""
function remove_edge!(dg::PSDiGraph{T}, orig::T, dest::T) where T
    if dest in keys(dg.parents)
        delete!(dg.parents[dest], orig)
    end
end


"""
    add_vertex!(dg::PSDiGraph{T}, v::T)

Add vertex v to the graph.
"""
function add_vertex!(dg::PSDiGraph{T}, v::T) where T
    if !(v in keys(dg.parents))
        dg.parents[v] = Set{T}()
    end
end


"""
    remove_vertex!(dg::PSDiGraph{T}, v::T)

Remove vertex v from the graph.
"""
function remove_vertex!(dg::PSDiGraph{T}, v::T) where T
    delete!(dg.parents, v)
    for pset in values(dg.parents)
        delete!(pset, v)
    end
end


"""
    out_neighbors(dg::PSDiGraph{T}, v::T)

Get the set of vertices having an edge from v
(i.e., the children of v)
"""
function out_neighbors(dg::PSDiGraph{T}, v::T) where T
    return Set{T}([ch for (ch, pset) in dg.parents if v in pset])
end


"""
    in_neighbors(dg::PSDiGraph{T}, v::T)

Get the set of vertices having an edge to v
(i.e., the parents of v)
"""
function in_neighbors(dg::PSDiGraph{T}, v::T) where T
    return get!(dg.parents, v, Set{T}())
end


"""
    transpose!(dg::PSDiGraph{T})

Reverse all of the graph's edges
"""
function transpose!(dg::PSDiGraph{T}) where T
    newparents = Dict{T,Set{T}}()
    for k in keys(dg.parents)
	newparents[k] = Set{T}()
    end
    for (k, pset) in dg.parents
        for p in pset
	    newparents[p] = k
        end
    end
    dg.parents = newparents
end


##############################
# GRAPH ALGORITHMS
##############################
include("DPMCollections.jl")
using .DPMCollections

"""
    graph_traversal(dg::PSDiGraph{T}, root::T, ds::AbstractCollection{T})

Generic graph traversal method. 
ds::Stack implies DFS traversal; ds::Queue implies BFS traversal
"""
function graph_traversal(dg::PSDiGraph{T}, root::T, ds::AbstractCollection{T}) where T
    
    visited = Vector{T}()
    push!(ds, root)
    
    while ! isempty(ds)
        
        v = pop!(ds)
        if in(v, visited)
            continue
        end
        push!(visited, v)
        
        for succ in out_neighbors(dg, v)
            push!(ds, succ)
            
        end  
    end
    
    return visited
end


"""
    dfs_traversal(dg::PSDiGraph{T}, root::T)
    
Traverse the directed graph `dg` in a Depth-First fashion, 
starting at vertex `root`.
"""
function dfs_traversal(dg::PSDiGraph{T}, root::T) where T
    ds = Stack{T}()
    return graph_traversal(dg, root, ds)
end


"""
    bfs_traversal(dg::PSDiGraph{T}, root::T)
    
Traverse the directed graph `dg` in a Breadth-First fashion, 
starting at vertex `root`.
"""
function bfs_traversal(dg::PSDiGraph{T}, root::T) where T
    ds = Queue{T}()
    return graph_traversal(dg, root, ds)
end


function _is_cyclic_rooted(dg::PSDiGraph{T}, root::T) where T
    
    visited = Set{T}()
    push!(visited, root)
    
    paths = Stack{Array{T,1}}()
    push!(paths, [root])
   
    
    while ! isempty(paths)
        
        p = pop!(paths)
  
        for s in out_neighbors(dg, last(p))
            
            push!(visited, s)
            
            if in(s, p)
               return true, visited
            end
            
            newpath = vcat(p, [s])
            push!(paths, newpath)
        end
        
    end
    
    return false, visited
end


"""
    is_cyclic(dg::PSDiGraph{T})

Check whether a directed graph contains cycles.
"""
function is_cyclic(dg::PSDiGraph{T}) where T

    dgc = copy(dg)
    
    while ! isempty(vertices(dgc))
        
	cycle_found, visited = _is_cyclic_rooted(dgc, first(vertices(dgc)))
        
        if cycle_found
            return true, visited
        end
        
        for v in visited
            remove_vertex!(dgc, v)
        end
        
    end
    
    return false
    
end


"""
    get_ancestors(dg::PSDiGraph{T}, v::T)

Find all the ancestors of vertex v in the directed graph,
assuming no cycles. (if necessary, this can be checked
by calling is_cyclic(dg).
"""
function get_ancestors(dg::PSDiGraph{T}, v::T) where T 

    dg_copy = copy(dg)
    transpose!(dg_copy)

    return set(dfs_traversal(dg_copy, v))
end


"""
    topological_sort(dg::PSDiGraph{T})

Assuming dg is a dag, this yields a DFS-based topological 
sort of the vertices.
"""
function topological_sort(dg::PSDiGraph{T}) where T

    dgc = copy(dg)
    visited = Array{T,1}()

    while ! isempty(vertices(dgc))
	v = first(vertices(dgc))
        _visit!(dgc, visited, v)
    end

    return visited

end


function _visit!(dg::PSDiGraph{T}, visited::Array{T,1}, v::T) where T
    
    succ = out_neighbors(dg, v)
    if length(succ) > 0
        for s in succ
            _visit!(dg, visited, s)
	end
    end

    if ! in(v, visited)
        pushfirst!(visited, v)
        remove_vertex!(dg, v)
    else
        throw(CycleException)
    end

end


struct CycleException <: Exception end


end

