# dbn_proposals.jl
# 2019-11-12
# David Merrell
#
#


function compute_move_probs(indeg::Int64, params::Tuple)
    degmax = params[1]
    t = params[2]
    frac = (indeg/degmax)^t
    comp = 1.0 - frac
    swp = 2.0 * frac * comp
    return [comp; swp; frac] ./ (frac + swp + comp)
end


function ith_nonparent(i, parents::Vector{Int64})
    
    n = length(parents)
    if n == 0
        return i
    end
    
    p = 1
    while p <= n
	if (parents[p] - p) >= i
	    break
        end
	p += 1
    end
    return i + p - 1
end

"""
Return the index of x in the *complement* of a;
that is, the index of x in the array [1,2,...] setminus a.
If x is in a, then we return the index it *would* have
if it were moved into a.
"""
function searchsorted_exclude(a::Vector{Int64}, x::Int64)
    r = searchsorted(a, x)
    if r.stop < r.start
        return x - r.stop
    else
        return x - r.stop + 1
    end
end

"""
Proposal distribution for updating the parent set of a vertex.

This proposal prioritizes "swap" moves: rather than adding or removing
and edge, just move it to a different parent. 

"Swap" proposals preserve in-degree. This is a good thing, since our 
model's posterior distribution varies pretty strongly with in-degree.
"""
@gen function parentvec_smart_swp_proposal(tr, vertex::Int64, 
					   compute_prob_params::Tuple,
					   V::Int64)

    #parents = [i for i=1:V if tr[:adjacency => :edges => vertex => i => :z]]
    parents = tr[:parent_sets => vertex]
    indeg = length(parents)
    p_add_swp_rem = compute_move_probs(indeg, compute_prob_params)
    action = @trace(Gen.categorical(p_add_swp_rem), :action)

    if action == 1
        to_add = @trace(Gen.uniform_discrete(1, V-indeg), :to_add)
        return (parents, action, to_add)
    elseif action == 2
        to_swp_add = @trace(Gen.uniform_discrete(1, V-indeg), :to_swp_add)
        to_swp_rem = @trace(Gen.uniform_discrete(1, indeg), :to_swp_rem)
	return (parents, action, to_swp_add, to_swp_rem)
    else
        to_rem = @trace(Gen.uniform_discrete(1, indeg), :to_rem)
	return (parents, action, to_rem)
    end
end

# TODO: MODIFY INVOLUTION FUNCTION
function parentvec_smart_swp_involution(cur_tr, fwd_choices, fwd_ret, prop_args)

    parents = fwd_ret[1]
    action = fwd_ret[2]
    vertex = prop_args[1]

    update_choices = Gen.choicemap()
    bwd_choices = Gen.choicemap()

    if action == 1
        to_add = fwd_ret[3]
	add_idx = ith_nonparent(to_add, parents) 
	update_choices[:adjacency => :edges => vertex => add_idx => :z] = true
	bwd_choices[:action] = 3
	bwd_choices[:to_rem] = searchsorted(parents, add_idx).start
    
    elseif action == 2
        to_swp_add = fwd_ret[3]
	to_swp_rem = fwd_ret[4]
	add_idx = ith_nonparent(to_swp_add, parents)
	rem_idx = parents[to_swp_rem]
	update_choices[:adjacency => :edges => vertex => add_idx => :z] = true
        update_choices[:adjacency => :edges => vertex => rem_idx => :z] = false
	bwd_choices[:action] = 2
	bwd_rem_idx = searchsorted(parents, add_idx).start
	bwd_add_idx = searchsorted_exclude(parents, rem_idx)
	if rem_idx < add_idx
            bwd_rem_idx -= 1
        else
            bwd_add_idx -= 1
	end
	bwd_choices[:to_swp_add] = bwd_add_idx
	bwd_choices[:to_swp_rem] = bwd_rem_idx

    else
        to_rem = fwd_ret[3]
	rem_idx = parents[to_rem]
	update_choices[:adjacency => :edges => vertex => rem_idx => :z] = false
        bwd_choices[:action] = 1
	bwd_choices[:to_add] = searchsorted_exclude(parents, rem_idx)
    end

    new_tr, weight, _, _ = Gen.update(cur_tr, Gen.get_args(cur_tr),
                                      (), update_choices)

    return new_tr, bwd_choices, weight
end


@gen function lambda_proposal(tr, radius::Int64)

    @trace(Gen.normal(tr[:lambda], radius), :lambda)

end


@gen (static) function edge_proposal(cur_tr, parent_idx::Int64, 
				             child_idx::Int64,
					     prob::Float64)
    @trace(Gen.bernoulli(prob), 
           :adjacency => :edges => child_idx => parent_idx => :z)
end
