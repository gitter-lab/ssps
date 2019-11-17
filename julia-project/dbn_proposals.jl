# dbn_proposals.jl
# 2019-11-12
# David Merrell
#
#


"""
    digraph_e_proposal(tr, ordered_vertices)

A proposal distribution for exploring the unconstrained
space of directed graphs (with a fixed set of vertices).
Chooses a pair of vertices (u,v) at random and then 
proposes an update dependent on the edge(s) that exist
between them.

This is more complicated than one might have expected--the
reason is that Gen's metropolis-hastings sampler requires
proposal distributions to have very special properties.
"""
@gen function digraph_e_proposal(tr, ordered_vertices)

    V = length(ordered_vertices)

    # Random choices
    u_idx = @trace(Gen.uniform_discrete(1,V), :u_idx)
    u = ordered_vertices[u_idx]
    v_idx = @trace(Gen.uniform_discrete(1,V), :v_idx)
    v = ordered_vertices[v_idx]

    # Propose an update
    new_G = copy(tr[:G])
    if (u in new_G.parents[v]) && (v in new_G.parents[u])
        del_choice = @trace(Gen.bernoulli(0.5), :del_choice)
	if del_choice
            remove_edge!(new_G, u, v)
	else
            remove_edge!(new_G, v, u)
	end
    elseif !(u in new_G.parents[v]) && !(v in new_G.parents[u])
        add_choice = @trace(Gen.bernoulli(0.5), :add_choice)
        if add_choice
            add_edge!(new_G, u, v)
        else
    	    add_edge!(new_G, v, u)
        end
    else
        action = @trace(Gen.uniform_discrete(1,3), :action)
        if u in new_G.parents[v]
            if action == 1
                remove_edge!(new_G, u, v)
	    elseif action == 2
                remove_edge!(new_G, u, v)
		add_edge!(new_G, v, u)
	    else
                add_edge!(new_G, v, u)
	    end
        else # v in new_G.parents[u]
            if action == 1
                remove_edge!(new_G, v, u) 
            elseif action == 2
                remove_edge!(new_G, v, u)
		add_edge!(new_G, u, v)
            else
                add_edge!(new_G, u, v)
	    end
	end
    end
    return new_G
end


"""
    digraph_e_involution(tr, cur_tr, fwd_choices, fwd_ret, prop_args)

The involution corresponding to `digraph_e_proposal`.
"""
function digraph_e_involution(cur_tr, fwd_choices, fwd_ret, prop_args)

    ordered_vertices = prop_args[1]
    old_G = cur_tr[:G]
    new_G = fwd_ret
    
    update_constraints = Gen.choicemap()
    update_constraints[:G] = new_G
    new_tr, weight, _, _ = Gen.update(cur_tr, get_args(cur_tr), (),
                                      update_constraints)

    fwd_u_idx = fwd_choices[:u_idx]
    fwd_u = ordered_vertices[fwd_u_idx]
    fwd_v_idx = fwd_choices[:v_idx]
    fwd_v = ordered_vertices[fwd_v_idx]

    bwd_choices = Gen.choicemap()
    bwd_choices[:u_idx] = fwd_u_idx
    bwd_choices[:v_idx] = fwd_v_idx

    if has_value(fwd_choices, :del_choice)
        if (fwd_u_idx != fwd_v_idx)
	    bwd_choices[:action] = 3
        else
            bwd_choices[:add_choice] = true
        end
    elseif has_value(fwd_choices, :add_choice) 
	if (fwd_u_idx != fwd_v_idx)
            bwd_choices[:action] = 1
	else
            bwd_choices[:del_choice] = true
	end
    elseif has_value(fwd_choices, :action)
	if fwd_choices[:action] == 1 
            bwd_choices[:add_choice] = true
        elseif fwd_choices[:action] == 2
            bwd_choices[:action] = 2
        elseif fwd_choices[:action] == 3
            bwd_choices[:del_choice] = true
	end
    end

    return new_tr, bwd_choices, weight
end


"""
Proposal distribution for exploring the unconstrained space of
directed graphs.

`expected_indegree` guides the exploration -- if a vertex's in-degree
is higher than this, then we are much more likely to remove an edge
from one of its parents.
"""
@gen function digraph_v_proposal(tr, ordered_vertices, expected_indegree::Float64)

    G = copy(tr[:G])
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

end


function digraph_v_involution(cur_tr, fwd_choices, fwd_ret, prop_args)

    # Update the trace
    new_G = fwd_ret
    old_G = cur_tr[:G]
    update_choices = Gen.choicemap()
    update_choices[:G] = new_G
    new_tr, weight, retdiff, discard = Gen.update(cur_tr, Gen.get_args(cur_tr), (), update_choices)

    # figure out what has changed
    fwd_u_idx = fwd_choices[:u_idx]
    sorted_vertices = prop_args[1]
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
end


"""
Proposal distribution for exploring the unconstrained space of
directed graphs.

This version of the proposal returns a "graph diff", rather than an updated graph.
Should yield better efficiency, as it avoids copying the graph or calling
"logpdf(dbnmarginal(...))" for the whole graph. I.e., reductions from
O(V) to O(1) complexity.

`expected_indegree` guides the exploration -- if a vertex's in-degree
is higher than this, then we are much more likely to remove an edge
from one of its parents.
"""
@gen function digraph_v_diff_proposal(tr, ordered_vertices, expected_indegree::Float64)

    G = tr[:G]
    result = nothing
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
        #remove_edge!(G, v, u)
	result = ("remove", v, u)
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
            #remove_edge!(G, u, v)
            #add_edge!(G, v, u)
	    result = ("reverse", u, v)
        else
            nonparents = sort(collect(setdiff(ordered_vertices, ordered_inneighbors)))
            len = length(nonparents)
            v_idx = @trace(Gen.uniform_discrete(1,len), :v_idx)
            v = nonparents[v_idx]
            #add_edge!(G, v, u)
	    result = ("add", v, u)
        end

    end

    return result 

end


function digraph_v_diff_involution(cur_tr, fwd_choices, fwd_diff, prop_args)

    old_G = cur_tr[:G]
    update_choices = Gen.choicemap()

    bwd_diff = nothing

    # figure out what has changed
    fwd_u_idx = fwd_choices[:u_idx]
    sorted_vertices = prop_args[1]
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

	bwd_diff = ("add", fwd_v, fwd_u) 

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

	    bwd_diff = ("reverse", fwd_v, fwd_u)

        else # an edge was added -- remove it.

            fwd_nonparents = sort(collect(setdiff(sorted_vertices, in_neighbors(old_G, fwd_u))))
            fwd_v = fwd_nonparents[fwd_v_idx]
            bwd_parents = sort(collect(in_neighbors(new_G, fwd_u)))
            bwd_v_idx = indexin([fwd_v], bwd_parents)[1]

            bwd_choices[:u_idx] = fwd_u_idx
            bwd_choices[:remove_edge] = true
            bwd_choices[:v_idx] = bwd_v_idx

	    bwd_diff = ("remove", fwd_v, fwd_u) 
        end
    end

    return bwd_diff, bwd_choices
end


function digraph_v_diff_update!(trace, trace_diff)

    G = trace[:G]
    action, v1, v2 = trace_diff

    if action == "add"
        add_edge!(G, v1, v2)
    elseif action == "reverse"
        remove_edge!(G, v1, v2)
	add_edge!(G, v2, v1)
    elseif action == "remove"
        remove_edge!(G, v1, v2)
    end

    update_choices = Gen.choicemap()
    update_choices[:G] = G

    trace, _, _, _ = update(trace, get_args(trace), (),
			    update_choices)

end


function digraph_v_diff_weight(trace, trace_diff)

end

#######################################
# EDGE INDICATOR MODEL FORMULATION
#######################################
"""
Proposal distribution for updating the parent set of a vertex.

P(remove edge) = (in degree / |V| )^t

We'll typically choose t = 1 / log2(|V| / median_indegree).
"""
@gen function parentvec_proposal(tr, vertex::Int64, V::Int64, t::Float64)

    parent_vec = [tr[:adjacency => :edges => vertex => j => :z] for j=1:V]
    in_degree = sum(parent_vec)
    remove_edge = @trace(Gen.bernoulli( (convert(Float64, in_degree)/V)^t ), :remove_edge)

    if remove_edge
        parent = @trace(Gen.uniform_discrete(1, in_degree), :parent)
	idx = findall(x->x, parent_vec)[parent]
    else
        nonparent = @trace(Gen.uniform_discrete(1, V - in_degree), :nonparent)
	idx = findall(x->!x, parent_vec)[nonparent]
    end

    return idx, parent_vec
end


function parentvec_involution(cur_tr, fwd_choices, fwd_ret, prop_args)
    
    vertex = prop_args[1]
    idx, parent_vec = fwd_ret
    parent_vec[idx] = !parent_vec[idx]
    update_constraints = Gen.choicemap()
    k = :adjacency => :edges => vertex => idx => :z
    update_constraints[k] = !cur_tr[k]

    new_tr, weight, _, _ = Gen.update(cur_tr, Gen.get_args(cur_tr),
                                      (), update_constraints)

    bwd_choices = Gen.choicemap()
    bwd_choices[:remove_edge] = !fwd_choices[:remove_edge]
    if Gen.has_value(fwd_choices, :parent)
        bwd_choices[:nonparent] = indexin([idx], findall(x->!x, parent_vec))[1]
    elseif Gen.has_value(fwd_choices, :nonparent)
        bwd_choices[:parent] = indexin([idx], findall(x->x, parent_vec))[1]
    end

    return new_tr, bwd_choices, weight 
end


@gen function lambda_proposal(tr, radius::Int64)

    @trace(Gen.uniform_discrete(max(tr[:lambda]-radius, 0), 
				tr[:lambda]+radius ), :lambda)

end

