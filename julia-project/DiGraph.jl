# DiGraph.jl
# 2019-09-26
# David Merrell
#
# Module for simple edgelist-based directed graphs

module DiGraphs

import Base: copy

export DiGraph, add_edge!, remove_edge!, add_vertex!, remove_vertex!, 
       out_neighbors, in_neighbors, dfs_traversal, bfs_traversal,
       is_cyclic, get_ancestors, transpose!, topological_sort 

"""
    DiGraph{T}(vertices::Set{T}, edges::Array{T,2})

A minimal directed graph parametric type.
"""
mutable struct DiGraph{T}
    vertices::Set{T}
    edges::Array{T,2} 
end

DiGraph(edges::Array{T,2}) where T = DiGraph(Set(edges), edges)
DiGraph{T}() where T = DiGraph(Array{T,2}(undef, 0, 2))
copy(dg::DiGraph{T}) where T =  DiGraph(copy(dg.vertices), copy(dg.edges))


function add_edge!(dg::DiGraph{T}, orig::T, dest::T) where T
    dg.edges = vcat(dg.edges, [orig dest])
    push!(dg.vertices, orig)
    push!(dg.vertices, dest)
end

function remove_edge!(dg::DiGraph{T}, orig::T, dest::T) where T
    dg.edges = dg.edges[(dg.edges[:,1] .!= [orig]) .| (dg.edges[:,2] .!= [dest]),:]
end

function add_vertex!(dg::DiGraph{T}, v::T) where T
    push!(dg.vertices, v)
end

function remove_vertex!(dg::DiGraph{T}, v::T) where T
    delete!(dg.vertices, v)
    dg.edges = dg.edges[(dg.edges[:,1] .!= [v]) .& (dg.edges[:,2] .!= [v]), :]
end

function out_neighbors(dg::DiGraph{T}, v::T) where T
    return Set(dg.edges[dg.edges[:,1] .== [v], 2])
end

function in_neighbors(dg::DiGraph{T}, v::T) where T
    return Set(dg.edges[dg.edges[:,2] .== [v], 1])
end

function transpose!(dg::DiGraph{T}) where T
    dg.edges = dg.edges[:,[2,1]]
end

include("Containers.jl")
using .Containers

"""
    graph_traversal(dg::DiGraph{T}, root::T, ds::AbstractContainer{T})

Generic graph traversal method. 
ds::Stack implies DFS traversal; ds::Queue implies BFS traversal
"""
function graph_traversal(dg::DiGraph{T}, root::T, ds::AbstractContainer{T}) where T
    
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
    dfs_traversal(dg::DiGraph{T}, root::T)
    
Traverse the directed graph `dg` in a Depth-First fashion, 
starting at vertex `root`.
"""
function dfs_traversal(dg::DiGraph{T}, root::T) where T
    ds = Stack{T}()
    return graph_traversal(dg, root, ds)
end


"""
    bfs_traversal(dg::DiGraph{T}, root::T)
    
Traverse the directed graph `dg` in a Breadth-First fashion, 
starting at vertex `root`.
"""
function bfs_traversal(dg::DiGraph{T}, root::T) where T
    ds = Queue{T}()
    return graph_traversal(dg, root, ds)
end


function _is_cyclic_rooted(dg::DiGraph{T}, root::T) where T
    
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
    is_cyclic(dg::DiGraph{T})

Check whether a directed graph contains cycles.
"""
function is_cyclic(dg::DiGraph{T}) where T

    dgc = copy(dg)
    
    while ! isempty(dgc.vertices)
        
        cycle_found, visited = _is_cyclic_rooted(dgc, first(dgc.vertices))
        
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
    get_ancestors(dg::DiGraph{T}, v::T)

Find all the ancestors of vertex v in the directed graph,
assuming no cycles. (if necessary, this can be checked
by calling is_cyclic(dg).
"""
function get_ancestors(dg::DiGraph{T}, v::T) where T 

    dg_copy = copy(dg)
    transpose!(dg_copy)

    return set(dfs_traversal(dg_copy, v))
end


"""
    topological_sort(dg::DiGraph{T})

Assuming dg is a dag, this yields a DFS-based topological 
sort of the vertices.
"""
function topological_sort(dg::DiGraph{T}) where T

    dgc = copy(dg)
    visited = Array{T,1}()

    while ! isempty(dgc.vertices)
        v = first(dgc.vertices)
        _visit!(dgc, visited, v)
    end

    return visited

end


function _visit!(dg::DiGraph{T}, visited::Array{T,1}, v::T) where T
    
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
