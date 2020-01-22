# postprocess_samples.jl
# 2020-01-17
# David Merrell
# 
# Functions for postprocessing MCMC sample output.
# Specifically for computing convergence statistics
# (PSRF and effective samples).

module SamplePostprocess

using JSON

"""
Read an MCMC samples output file
"""
function load_samples(samples_file_path::String)

    f = open(samples_file_path, "r")
    str = read(f, String)
    close(f)
    d = JSON.parse(str)

    return d["parent_sets"], d["lambda"], d["n"]
end


#####
# Define a convenience class for working with the sparse-formatted
# MCMC sample data. 
#
# Samples are stored as vectors of "changepoints"; 
# (idx,val) pairs where the sequence changes to 
# value `val` at index `idx`, and stays constant otherwise.
##################################################################

mutable struct ChangepointVec{T}
    changepoints::Vector{Tuple{Int64,T}}
    len::Int64
    default_v::T
end

ChangepointVec(changepoints::Vector{Tuple{Int64,T}}, len::Int64) where T = ChangepointVec(changepoints, len, zero(T))

import Base: map, reduce, getindex, length

length(cpv::ChangepointVec) = cpv.len


function getindex(cpv::ChangepointVec, r::UnitRange)
    
    # take care of this corner case
    if length(cpv.changepoints) == 0
        return ChangepointVec([], length(r), cpv.default_v)
    end
    
    # Index the range into the changepoints
    start_idx = searchsortedfirst(cpv.changepoints, r.start; by=x-> x[1])
    stop_idx = searchsortedfirst(cpv.changepoints, r.stop; by=x-> x[1])
    
    # Adjust the stop_idx
    if stop_idx > length(cpv.changepoints) || r.stop != cpv.changepoints[stop_idx][1]
        stop_idx -= 1
    end
    
    new_changepoints = cpv.changepoints[start_idx:stop_idx]
    
    # Adjust the start of the vector
    if r.start != cpv.changepoints[start_idx][1]
        pushfirst!(new_changepoints, (r.start, cpv.changepoints[end][2]))
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
    s = zero(cpv.default_v)
    cur_t = 0
    cur_changepoint = 0
    cur_v = cpv.default_v
    
    if length(cpv.changepoints) == 0
        return length(cpv) * cpv.default_v
    end
    
    while cur_changepoint < length(cpv.changepoints)
        next_t, next_v = cpv.changepoints[cur_changepoint+1]
        s += cur_v * (next_t - cur_t)
        cur_changepoint += 1
        cur_t = next_t
        cur_v = next_v
    end 
    s += cur_v * (length(cpv) - cur_t + 1)
    
    return s
end


function mean(cpv::ChangepointVec)
    return sum(cpv) / length(cpv)
end


#######
# These functions are used for computing PSRF and 
# N_eff (diagnostic scores for MCMC convergence)
#################################################

function var(cpv::ChangepointVec, mean::Float64)
    return sum(map( x->abs2(x-mean), cpv )) / (length(cpv) - 1)
end

function w_stat(variances::Vector{Float64})
    return sum(variances) / length(variances)
end

function b_stat(means::Vector{Float64}, n::Int64)
   big_mean = sum(means)/length(means)
   return n * sum(abs2(means .- big_mean)) / length(means)
end

function varplus(w_stat::Float64, b_stat::Float64, n::Int64)
    return (n-1)*w_stat/n + b_stat/n
end

function psrf_stat(varplus, w_stat)
    return sqrt(varplus / w_stat)
end



function compute_psrf_neff(seqs::Vector{ChangepointVec})
    
end


end # END MODULE
