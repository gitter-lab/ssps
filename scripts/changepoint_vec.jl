
##################################################################
# Define a convenience class for working with the sparse
# MCMC sample data. 
#
# Samples are stored as vectors of "changepoints":
# (idx,val) pairs where the sequence changes to 
# value `val` at index `idx`, and stays constant otherwise.
##################################################################

mutable struct ChangepointVec
    changepoints::Vector
    len::Int64
    default_v::Number
end

ChangepointVec(changepoints::Vector, len::Int64) = ChangepointVec(changepoints, len, 0)

import Base: map, reduce, getindex, length, sum

length(cpv::ChangepointVec) = cpv.len


function getindex(cpv::ChangepointVec, r::UnitRange)

    # take care of this corner case
    if length(cpv.changepoints) == 0
        return ChangepointVec([], length(r), cpv.default_v)
    end

    # Index the range into the changepoints
    start_searched = searchsorted(cpv.changepoints, r.start; by=x-> x[1])
    start_idx, start_2 = start_searched.start, start_searched.stop
    stop_searched = searchsorted(cpv.changepoints, r.stop; by=x-> x[1])
    stop_1, stop_idx = stop_searched.start, stop_searched.stop

    new_changepoints = cpv.changepoints[start_idx:stop_idx]

    # Adjust the start of the vector
    if start_idx > start_2 && start_2 != 0
        pushfirst!(new_changepoints, (r.start, cpv.changepoints[start_2][2]))
    end

    # Shift their indices so that we start from 1
    new_changepoints = map(x->(x[1] - r.start + 1, x[2]), new_changepoints)

    return ChangepointVec(new_changepoints, length(r), cpv.default_v)
end


"""
We assume t in [1, length(cpv)].
"""
function getindex(cpv::ChangepointVec, t::Int)
    
    if length(cpv.changepoints) == 0
        return cpv.default_v
    end
    
    idx = searchsortedfirst(cpv.changepoints, t; by=x->x[1])
    
    if idx > length(cpv.changepoints) 
        return cpv.changepoints[end][2]
    elseif cpv.changepoints[idx][1] == t
        return cpv.changepoints[idx][2]
    else
        if idx == 1
            return cpv.default_v
        else
            return cpv.changepoints[idx-1][2]
        end
    end
end


function map(f::Function, cpv::ChangepointVec)
    new_changepoints = map(x->(x[1],f(x[2])), cpv.changepoints)
    return ChangepointVec(new_changepoints, cpv.len, cpv.default_v)
end


function sum(cpv::ChangepointVec)
    
    if length(cpv.changepoints) == 0
        return length(cpv) * cpv.default_v
    end
    
    s = zero(cpv.default_v)
    if cpv.changepoints[1][1] > 1
        cur_t_v = (1, cpv.default_v)
    else
        cur_t_v = cpv.changepoints[1]
    end
    
    for cur_changepoint=1:length(cpv.changepoints)-1
        next_t_v = cpv.changepoints[cur_changepoint+1]
        s += cur_t_v[2] * (next_t_v[1] - cur_t_v[1])
        cur_t_v = next_t_v
    end 
    s += cur_t_v[2] * (length(cpv) - cur_t_v[1] + 1)
    
    return s
end

function mult_ind(binary_expr, value)
    if Bool(binary_expr)
        return value 
    else
        return 0
    end
end

function count_nonzero(cpv::ChangepointVec)

    if length(cpv.changepoints) == 0
        return mult_ind(cpv.default_v, length(cpv))
    end
    
    s = zero(cpv.default_v)
    if cpv.changepoints[1][1] > 1
        cur_t_v = (1, cpv.default_v)
    else
        cur_t_v = cpv.changepoints[1]
    end

    for cur_changepoint=1:length(cpv.changepoints)-1
        next_t_v = cpv.changepoints[cur_changepoint+1]
        s += mult_ind(cur_t_v[2], next_t_v[1] - cur_t_v[1])
        cur_t_v = next_t_v
    end

    s += mult_ind(cur_t_v[2], length(cpv) - cur_t_v[1] + 1)

    return s 
end


function mean(cpv::ChangepointVec; is_binary::Bool=false)
    if is_binary
        return count_nonzero(cpv) / length(cpv)
    else
        return sum(cpv) / length(cpv)
    end
end


function binop(f::Function, cpva::ChangepointVec, cpvb::ChangepointVec)

    new_default = f(cpva.default_v, cpvb.default_v)
    new_len = length(cpva)
    ncp_a = length(cpva.changepoints)
    ncp_b = length(cpvb.changepoints)
    new_changepoints = Vector{Tuple}()

    # Handle this corner case
    if new_len == 0
        return ChangepointVec([], new_len, new_default)
    end

    cur_t = 1
    cur_cp_a = 0
    cur_cp_b = 0
    cur_v_a = cpva.default_v
    cur_v_b = cpvb.default_v

    while cur_t < new_len

        if cur_cp_a < ncp_a
            #next_t_a, next_v_a = cpva.changepoints[cur_cp_a+1]
            next_cpa = cpva.changepoints[cur_cp_a+1]
            next_t_a = next_cpa[1]
            next_v_a = next_cpa[2]
        else
            next_t_a = new_len
            next_v_a = cur_v_a
        end
        if cur_cp_b < ncp_b
            next_t_b, next_v_b = cpvb.changepoints[cur_cp_b+1]
        else
            next_t_b = new_len
            next_v_b = cur_v_b
        end

        if next_t_a <= next_t_b
            cur_t = next_t_a
            cur_cp_a += 1
            cur_v_a = next_v_a
        end
        if next_t_b <= next_t_a
            cur_t = next_t_b
            cur_cp_b += 1
            cur_v_b = next_v_b
        end

        push!(new_changepoints, (cur_t, f(cur_v_a, cur_v_b)))
    end

    return ChangepointVec(new_changepoints, new_len, new_default)
end

function to_vec(cpv::ChangepointVec)
    vec = Vector{Float64}(undef, cpv.len)
    if cpv.changepoints[1][1] > 1
        vec[1:cpv.changepoints[1][1]] .= cpv.default_v 
    end
    for (i, cp) in enumerate(cpv.changepoints[1:end-1])
        vec[cp[1]:cpv.changepoints[i+1][1]] .= cp[2]
    end
    last = cpv.changepoints[end]
    vec[last[1]:end] .= last[2]
    return vec
end
