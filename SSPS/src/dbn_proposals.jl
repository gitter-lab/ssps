# dbn_proposals.jl
# 2019-11-12
# David Merrell
#
# Some proposal distributions for exploring the space
# of directed graphs.

############################################
# PARENT SET PROPOSAL DISTRIBUTION
############################################
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
Parent set proposal distribution.

This proposal allows "add parent", "remove parent", and "swap parent" moves.

"Swap" moves preserve in-degree. This is a good thing, since our 
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



####################################
# UNIFORM GRAPH PROPOSAL
####################################

"""
In order to propose reverse-edge moves, 
we need to know which of the edges are not paired.
"""
function unpaired_edges(tr, V)
    result = Set()
    for child = 1:V
        for p in tr[:parent_sets => child => :parents]
            if (child, p) in result
                delete!(result, (child,p))
            elseif p != child
                push!(result, (p,child))
            end
        end
    end
    return sort(collect(result))
end


"""
Uniform graph proposal distribution.
"""
@gen function uniform_proposal(tr, V)

    ue = unpaired_edges(tr,V)
    bound = V^2 + length(ue)
    idx = @trace(Gen.uniform_discrete(1, bound), :idx)

    if idx <= V^2
        u_idx = div(idx-1, V) + 1
        v_idx = ((idx-1) % V) + 1 

        if u_idx in tr[:parent_sets => v_idx => :parents]
            return (u_idx, v_idx), bound, ue, "remove"
        else
            return (u_idx, v_idx), bound, ue, "add"
        end
    else
        return (ue[idx - V^2]), bound, ue, "reverse"
    end
end

"""
Involution for the uniform graph proposal
"""
function uniform_involution(cur_tr, fwd_choices, fwd_ret, prop_args)
    
    idx = fwd_ret[1]
    u_idx = idx[1]
    v_idx = idx[2]
    bound = fwd_ret[2]
    ue = fwd_ret[3]
    action = fwd_ret[4]
    V = prop_args[1]
    
    update_choices = Gen.choicemap()
    bwd_choices = Gen.choicemap()

    if action == "add"
        bwd_choices[:idx] = (u_idx-1)*V + v_idx
        
        v_parents = copy(cur_tr[:parent_sets => v_idx => :parents])
        insert_idx = searchsortedfirst(v_parents, u_idx)
        insert!(v_parents, insert_idx, u_idx)
        update_choices[:parent_sets => v_idx => :parents] = v_parents

    elseif action == "remove"
        bwd_choices[:idx] = (u_idx-1)*V + v_idx
        
        v_parents = copy(cur_tr[:parent_sets => v_idx => :parents])
        rem_idx = searchsortedfirst(v_parents, u_idx)
        deleteat!(v_parents, rem_idx)
        update_choices[:parent_sets => v_idx => :parents] = v_parents

    elseif action == "reverse"
        
        ue_idx = searchsortedfirst(ue, (u_idx, v_idx))
        deleteat!(ue, ue_idx)

        bwd_choices[:idx] = V^2 + searchsortedfirst(ue, (v_idx, u_idx)) 

        v_parents = copy(cur_tr[:parent_sets => v_idx => :parents])
        rem_idx = searchsortedfirst(v_parents, u_idx)
        deleteat!(v_parents, rem_idx)
        update_choices[:parent_sets => v_idx => :parents] = v_parents
        
        u_parents = copy(cur_tr[:parent_sets => u_idx => :parents])
        insert_idx = searchsortedfirst(u_parents, v_idx)
        insert!(u_parents, insert_idx, v_idx)
        update_choices[:parent_sets => u_idx => :parents] = u_parents
    end 

    new_tr, weight, _, _ = Gen.update(cur_tr, Gen.get_args(cur_tr),
                                      (), update_choices)
    
    return new_tr, bwd_choices, weight
end


@gen function lambda_proposal(tr, std::Float64)

    @trace(Gen.normal(tr[:lambda], std), :lambda)

end


@gen function lambda_vec_proposal(tr, i::Int64, std::Float64)

    @trace(Gen.normal(tr[:lambda_vec => i => :lambda], std), :lambda_vec => i => :lambda)

end


