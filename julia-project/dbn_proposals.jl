# dbn_proposals.jl
# 2019-11-12
# David Merrell
#
#


function compute_move_probs(indeg::Int64, degmax::Int64, prob_param::Float64)
    frac = (indeg/degmax)^prob_param
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
@gen function smart_proposal(tr, vertex::Int64, 
	                         prob_param::Float64,
                                 V::Int64)

    parents = copy(tr[:parent_sets => vertex => :parents])
    indeg = length(parents)
    action_probs = compute_move_probs(indeg, V, prob_param)
    action = @trace(Gen.categorical(action_probs), :action)

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


function smart_involution(cur_tr, fwd_choices, fwd_ret, prop_args)

    parents = fwd_ret[1]
    action = fwd_ret[2]
    vertex = prop_args[1]

    update_choices = Gen.choicemap()
    bwd_choices = Gen.choicemap()

    if action == 1
        to_add = fwd_ret[3]
	add_parent = ith_nonparent(to_add, parents)
        insert_idx = searchsortedfirst(parents, add_parent)
        insert!(parents, insert_idx, add_parent)
        update_choices[:parent_sets => vertex => :parents] = parents
	bwd_choices[:action] = 3
	bwd_choices[:to_rem] = insert_idx 
    
    elseif action == 2
        to_swp_add = fwd_ret[3]
	to_swp_rem = fwd_ret[4]
	add_parent = ith_nonparent(to_swp_add, parents)
	rem_parent = parents[to_swp_rem]
        
        deleteat!(parents, to_swp_rem)
        insert_idx = searchsortedfirst(parents, add_parent)
        insert!(parents, insert_idx, add_parent) 
        
        update_choices[:parent_sets => vertex => :parents] = parents
	
        bwd_choices[:action] = 2
	bwd_choices[:to_swp_add] = searchsorted_exclude(parents, rem_parent)
	bwd_choices[:to_swp_rem] = insert_idx

    else
        to_rem = fwd_ret[3]
	rem_parent = parents[to_rem]
        deleteat!(parents, to_rem)
	update_choices[:parent_sets => vertex => :parents] = parents
        bwd_choices[:action] = 1
	bwd_choices[:to_add] = searchsorted_exclude(parents, rem_parent)
    end

    new_tr, weight, _, _ = Gen.update(cur_tr, Gen.get_args(cur_tr),
                                      (), update_choices)

    return new_tr, bwd_choices, weight
end


@gen function lambda_proposal(tr, std::Int64)

    @trace(Gen.normal(tr[:lambda], std), :lambda)

end


@gen function lambda_vec_proposal(tr, i::Int64, std::Int64)

    @trace(Gen.normal(tr[:lambda_vec => i => :lambda], std), :lambda_vec => i => :lambda)

end


