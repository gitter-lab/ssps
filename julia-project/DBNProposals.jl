

module DBNProposals

using Gen
include("PSDiGraph.jl")
using .PSDiGraphs

export digraph_proposal, digraph_involution


"""
Proposal distribution for exploring the unconstrained space of
directed graphs.

`expected_indegree` guides the exploration -- if a vertex's in-degree
is higher than this, then we are much more likely to remove an edge
from one of its parents.
"""
@gen function digraph_proposal(tr, expected_indegree::Float64)
    
    G = copy(tr[:G])
    ordered_vertices = sort(collect(vertices(G)))
    V = length(ordered_vertices)
    
    u_idx = @trace(Gen.uniform_discrete(1,V), :u_idx)
    u = ordered_vertices[u_idx]
    ordered_inneighbors = sort(collect(in_neighbors(G,u)))
    in_deg = length(ordered_inneighbors)
    
    prob_remove = (1.0*in_deg / V) ^ (-1.0/log2(1.0*expected_indegree/V))
    remove_edge = @trace(Gen.bernoulli(prob_remove), :remove_edge)
    
    if remove_edge
        v_idx = @trace(Gen.uniform_discrete(1, in_deg), :v_idx)
        v = ordered_inneighbors[v_idx]
        remove_edge!(G, v, u)
    else
        
        outneighbors = out_neighbors(G,u)
        ordered_exc_outneighbors = sort(collect(setdiff(outneighbors, ordered_inneighbors)))
        out_exc_deg = length(ordered_exc_outneighbors)
        deg = length(union(outneighbors, ordered_inneighbors))
        
        prob_reverse = 1.0 * out_exc_deg / deg
        reverse_edge = @trace(Gen.bernoulli(prob_reverse), :reverse_edge)
        if reverse_edge
            
            v_idx = @trace(Gen.uniform_discrete(1, out_exc_deg), :v_idx)
            v = ordered_exc_outneighbors[v_idx]
            remove_edge!(G, u, v)
            add_edge!(G, v, u)
        else
            nonparents = sort(collect(setdiff(ordered_vertices, ordered_inneighbors)))
            len = length(nonparents)
            v_idx = @trace(Gen.uniform_discrete(1,len), :v_idx)
            v = nonparents[v_idx]
            add_edge!(G, v, u)
        end
        
    end
    
    return G
    
end;


function digraph_involution(cur_tr, fwd_choices, fwd_ret, prop_args)
    
    # Update the trace 
    new_G = fwd_ret
    old_G = cur_tr[:G]
    update_choices = Gen.choicemap()
    update_choices[:G] = new_G
    new_tr, weight, retdiff, discard = Gen.update(cur_tr, Gen.get_args(cur_tr), (), update_choices)
    
    # figure out what has changed
    fwd_u_idx = fwd_choices[:u_idx]
    sorted_vertices = sort(collect(vertices(old_G)))
    fwd_u = sorted_vertices[fwd_u_idx]
    fwd_v_idx = fwd_choices[:v_idx]
    
    # Deduce the correct backward choices
    bwd_choices = Gen.choicemap()
    if fwd_choices[:remove_edge] # an edge was removed -- we must add it back.
        
        fwd_parents = collect(in_neighbors(old_G, fwd_u))
        fwd_v = sort(fwd_parents)[fwd_v_idx]
        bwd_nonparents = sort(collect(setdiff(sorted_vertices,in_neighbors(new_G, fwd_u))))
        bwd_v_idx = indexin([fwd_v], bwd_nonparents)[1]
        
        bwd_choices[:u_idx] = fwd_u_idx 
        bwd_choices[:remove_edge] = false
        bwd_choices[:reverse_edge] = false
        bwd_choices[:v_idx] = bwd_v_idx
        
    else
        if fwd_choices[:reverse_edge] # an edge was reversed -- reverse it back.
            
            fwd_exc_outneighbors = sort(collect(setdiff(out_neighbors(old_G, fwd_u), in_neighbors(old_G, fwd_u))))
            fwd_v = fwd_exc_outneighbors[fwd_v_idx]
            bwd_u_idx = indexin([fwd_v], sorted_vertices)[1]
            bwd_exc_outneighbors = sort(collect(setdiff(out_neighbors(new_G, fwd_v), in_neighbors(new_G, fwd_v))))
            bwd_v_idx = indexin([fwd_u], bwd_exc_outneighbors)[1]
            
            bwd_choices[:u_idx] = bwd_u_idx
            bwd_choices[:remove_edge] = false
            bwd_choices[:reverse_edge] = true
            bwd_choices[:v_idx] = bwd_v_idx
            
        else # an edge was added -- remove it.
            
            fwd_nonparents = sort(collect(setdiff(sorted_vertices, in_neighbors(old_G, fwd_u))))
            fwd_v = fwd_nonparents[fwd_v_idx]
            bwd_parents = sort(collect(in_neighbors(new_G, fwd_u)))
            bwd_v_idx = indexin([fwd_v], bwd_parents)[1]
            
            bwd_choices[:u_idx] = fwd_u_idx
            bwd_choices[:remove_edge] = true
            bwd_choices[:v_idx] = bwd_v_idx
            
        end
    end
    
    return new_tr, bwd_choices, weight
end;

end
