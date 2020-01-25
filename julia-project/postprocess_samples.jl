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
# Samples are stored as vectors of "changepoints":
# (idx,val) pairs where the sequence changes to 
# value `val` at index `idx`, and stays constant otherwise.
##################################################################

mutable struct ChangepointVec
    changepoints::Vector{Tuple{Int64,Number}}
    len::Int64
    default_v::Number
end

ChangepointVec(changepoints::Vector{Tuple{Int64,T}}, len::Int64) where T = ChangepointVec(changepoints, len, zero(T))

import Base: map, reduce, getindex, length, sum

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
    cur_t = 1
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

mean(cpv::ChangepointVec) = sum(cpv) / length(cpv)


function binop(f::Function, cpva::ChangepointVec, cpvb::ChangepointVec)

    new_default = f(cpva.default_v, cpvb.default_v)
    new_len = length(cpva)
    new_changepoints = []

    # Handle this corner case
    if new_len == 0
        return ChangepointVec(new_changepoints, new_len, new_default)
    end

    cur_t = 1
    cur_cp_a = 0
    cur_cp_b = 0
    cur_v_a = cpva.default_v
    cur_v_b = cpvb.default_v

    while cur_t < new_len

        if cur_cp_a < length(cpva.changepoints)
            next_t_a, next_v_a = cpva.changepoints[cur_cp_a+1]
        else
            next_t_a = new_len
            next_v_a = cur_v_a
        end
        if cur_cp_b < length(cpvb.changepoints)
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


function seq_variogram(seq::ChangepointVec)
    
    n = length(seq)
    result = zeros(n - 1)

    if length(seq.changepoints) == 0
        return result
    end    

    changepoints = copy(seq.changepoints)
    if changepoints[1][1] > 1 
        pushfirst!(changepoints, (1, seq.default_v))
    end 

    # Get the sizes of "blocks" of same-valued entries
    blocksizes = zeros(Int64, length(changepoints))
    for k=1:length(changepoints) - 1 
        blocksizes[k] = changepoints[k+1][1] - changepoints[k][1]
    end 
    blocksizes[end] = n - changepoints[end][1] + 1 

    for (i, cpi) in enumerate(changepoints)
        start_i = cpi[1]
        bsize_i = blocksizes[i] 

        for (j, cpj) in enumerate(changepoints[i+1:end])
            start_j = cpj[1]
            bsize_j = blocksizes[j+i]
                
            diffsq = abs2(cpi[2] - cpj[2])
            if diffsq == 0
                continue
            end 
    
            for ii=start_i:(start_i+bsize_i-1)
                for jj=start_j:(start_j+bsize_j-1)
                    result[jj - ii] += diffsq
                end
            end 
        end
    end 
    
    return result ./ (n .- collect(1:n-1))
end


function correlation_sum(variogram::Vector, varplus::Float64)
    corrs = 1.0 .- variogram ./ (2.0*varplus)
    s = 0.0
    for i=1:length(corrs)-2 
        if corrs[i+1] + corrs[i+2] >= 0.0
            s += corrs[i]
        else
            break
        end
    end
    return s
end

n_eff_stat(corr_sum::Float64, m::Int64, n::Int64) = m*n/(1.0 + 2.0*corr_sum)


#######
# These functions are used for computing PSRF and 
# N_eff (diagnostic scores for MCMC convergence)
#################################################

seq_var(cpv::ChangepointVec, mean::Float64) = sum(map( x->abs2(x-mean), cpv )) / (length(cpv) - 1)

w_stat(variances::Vector{Float64}) = sum(variances) / length(variances)

b_stat(means::Vector{Float64}, n::Int64) = n*sum(abs2.(means .- sum(means)/length(means)))/length(means)

varplus(w_stat::Float64, b_stat::Float64, n::Int64) = (n-1)*w_stat/n + b_stat/n

psrf_stat(varplus, w_stat) = sqrt(varplus / w_stat)


"""
Compute the PSRF and N_eff for a collection of sequences.
In the context of MCMC, we assume the seqs have already been
split in half. 

We assume the seqs all have equal length.
"""
function compute_psrf_neff(seqs::Vector{ChangepointVec})
    seq_means = [mean(seq) for seq in seqs]
    seq_variances = [seq_var(seq, seq_means[i]) for (i, seq) in enumerate(seqs)]
    m = length(seqs)
    n = length(seqs[1])
    B = b_stat(seq_means, n)
    W = w_stat(seq_variances)
    vp = varplus(W, B, n)
    psrf = psrf_stat(vp, W)
    
    seq_vgrams = [seq_variogram(seq) for seq in seqs]
    #println(seq_vgrams)
    #vgram = combine_variograms(seq_vgrams)
    vgram = sum(seq_vgrams) / m
    corr_sum = correlation_sum(vgram, vp)
    n_eff = n_eff_stat(corr_sum, m, n)

    return psrf, n_eff
end


"""
Compute convergence diagnostics for a collection of sequences,
at multiple stop indices
"""
function evaluate_convergence(whole_seqs::Vector{ChangepointVec}, stop_idxs;
                              burnin::Float64=0.5)
    
    # We'll collect a list of (psrf, n_eff values) 
    results = []
    
    for len in stop_idxs
        # Split the whole sequences into half sequences
        burnin_idx = Int(round(burnin*len))
        split_idx = burnin_idx + div(len - burnin_idx, 2)
        half_seqs = []
        for ws in whole_seqs
            push!(half_seqs, ws[burnin_idx+1:split_idx])
            push!(half_seqs, ws[split_idx+1:len])
        end
        
        # compute the convergence diagnostics
        psrf, n_eff = compute_psrf_neff(half_seqs)
        push!(results, (psrf, n_eff))
    end
    
    return results
end


seq_converged(psrf, n_eff, psrf_ub, n_eff_lb) = (psrf < psrf_ub && n_eff >= n_eff_lb)


function get_all_at(dict_vec::Vector, key_vec::Vector)
    results = []
    for d in dict_vec
        for k in key_vec
            d = d[k]
        end
        push!(results, d)
    end
    return results
end


function evaluate_convergence_at(dict_vec::Vector, key_vec::Vector, 
                                 stop_idxs, burnin)
    whole_seqs = get_all_at(dict_vec, key_vec)
    diag_vec = evaluate_convergence(whole_seqs, stop_idxs; burnin=burnin)
    return diag_vec
end


function push_nonconverged!(nonconverged::Vector, dict_vec, key_vec, 
                            stop_idx, burnin, psrf_ub, n_eff_lb)
    
    diag_vec = evaluate_convergence_at(dict_vec, key_vec, 
                                       stop_idx, burnin)
    for (i, diag) in enumerate(diag_vec)
        if !seq_converged(diag[1], diag[2], psrf_ub, n_eff_lb)
            push!(nonconverged[i], diag)
        end
    end
end


function collect_dbn_nonconverged(chain_results::Vector, stop_idxs; 
                                  burnin::Float64=0.5,
                                  psrf_ub::Float64=1.1,
                                  n_eff_lb::Float64=10.0)
    
    nonconverged = fill([], length(seq_lens))
    
    # convergence for lambda?
    push_nonconverged!(nonconverged, chain_results, ["lambdas"],
                       stop_idxs, burnin, psrf_up, n_eff_lb)
    
    # convergence for edges?
    V = length(chain_results[1]["parent_sets"])
    for i=1:V
        for j=1:V
            push_nonconverged!(nonconverged, chain_results, ["parent_sets", j, i],
                       stop_idxs, burnin, psrf_up, n_eff_lb)
        end
    end
    
    return nonconverged
end


end
