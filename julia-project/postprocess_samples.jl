# postprocess_samples.jl
# 2020-01-17
# David Merrell
# 
# Functions for postprocessing MCMC sample output.
# Specifically for computing convergence statistics
# (PSRF and effective samples).

module SamplePostprocess

using JSON
using ArgParse
using DataStructures

#export ChangepointVec, count_nonzero, sum, mean, seq_var, seq_variogram, postprocess_sample_files
export postprocess_sample_files

"""
Read an MCMC samples output file
"""
function load_samples(samples_file_path::String)

    f = open(samples_file_path, "r")
    str = read(f, String)
    close(f)
    d = JSON.parse(str)
    d["parent_sets"] = [DefaultDict([], ps) for ps in d["parent_sets"]]

    return d 
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
            next_t_a, next_v_a = cpva.changepoints[cur_cp_a+1]
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


function seq_variogram(seq::ChangepointVec, t::Int64; is_binary::Bool=false)
    n = length(seq)
    if is_binary
        return mean(binop(!=, seq[1:n-t], seq[1+t:n]); is_binary=is_binary)
    else
        return mean(map(abs2, binop(-, seq[1:n-t], seq[1+t:n])); is_binary=is_binary)
    end
end


function correlation_sum(seqs::Vector{ChangepointVec}, varplus::Float64; is_binary::Bool=false)
    
    s = 0.0
    m = length(seqs)
    prev2_corr = 0.0
    prev1_corr = 0.0
    
    for t=1:length(seqs[1])-2

        variograms = [seq_variogram(seq, t; is_binary=is_binary) for seq in seqs]
        comb_variogram = sum(variograms) / m

        corr = 1.0 - (comb_variogram / (2.0*varplus))

        if t > 2 
            if corr + prev1_corr >= 0.0
                s += prev2_corr
            else
                break
            end
        end
        prev2_corr = prev1_corr
        prev1_corr = corr

    end
    
    return s
end

n_eff_stat(corr_sum::Float64, m::Int64, n::Int64) = m*n/(1.0 + 2.0*corr_sum)


#######
# These functions are used for computing PSRF and 
# N_eff (diagnostic scores for MCMC convergence)
#################################################

function seq_var(cpv::ChangepointVec, mean::Float64; is_binary::Bool=false)
    if is_binary 
        N = length(cpv)
        return mean*(1 - mean)*N/(N - 1)
    else
        return sum(map( x->abs2(x-mean), cpv )) / (length(cpv) - 1)
    end
end

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
function compute_psrf_neff(seqs::Vector{ChangepointVec}; is_binary::Bool=false)
    seq_means = [mean(seq; is_binary=is_binary) for seq in seqs]
    seq_variances = [seq_var(seq, seq_means[i]; is_binary=is_binary) for (i, seq) in enumerate(seqs)]
    m = length(seqs)
    n = length(seqs[1])
    B = b_stat(seq_means, n)
    W = w_stat(seq_variances)
    vp = varplus(W, B, n)
    psrf = psrf_stat(vp, W)
    
    corr_sum = correlation_sum(seqs, vp; is_binary=is_binary)
    n_eff = n_eff_stat(corr_sum, m, n)

    return psrf, n_eff
end


"""
Compute convergence diagnostics for a collection of sequences,
at multiple stop indices
"""
function evaluate_convergence(whole_seqs::Vector{ChangepointVec}, stop_idxs;
                              burnin::Float64=0.5, is_binary::Bool=false)
    
    # We'll collect a list of (psrf, n_eff values) 
    results = []
    
    for len in stop_idxs
        # Split the whole sequences into half sequences
        burnin_idx = Int(round(burnin*len))
        split_idx = burnin_idx + div(len - burnin_idx, 2)
        half_seqs = Vector{ChangepointVec}()
        for ws in whole_seqs
            push!(half_seqs, ws[burnin_idx+1:split_idx])
            push!(half_seqs, ws[split_idx+1:len])
        end
        
        # compute the convergence diagnostics
        psrf, n_eff = compute_psrf_neff(half_seqs; is_binary=is_binary)
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
                                 stop_idxs, burnin; is_binary::Bool=false)
    whole_seqs = get_all_at(dict_vec, key_vec)
    whole_seqs = [ChangepointVec(seq, dict_vec[i]["n"]) for (i, seq) in enumerate(whole_seqs)]
    diag_vec = evaluate_convergence(whole_seqs, stop_idxs; burnin=burnin, is_binary=is_binary)
    return diag_vec
end


function collect_conv_stats(dict_vec, stop_points; burnin::Float64=0.5)

    V = length(dict_vec[1]["parent_sets"])
    
    conv_stats = Dict("lambda"=> [],
                      "parent_sets" => [[] for i=1:V] )

    conv_stats["lambda"] = evaluate_convergence_at(dict_vec, ["lambda"], 
                                                   stop_points, burnin;
                                                   is_binary=false)
    
    for i=1:V
        for j=1:V
            push!(conv_stats["parent_sets"][i],
                  evaluate_convergence_at(dict_vec, 
                                          ["parent_sets", i, string(j)],
                                          stop_points, burnin; is_binary=true)
                 )
        end
    end
    results = Dict()
    results["stop_points"] = stop_points
    results["conv_stats"] = conv_stats
    
    return results
end
#function collect_dbn_nonconverged(chain_results::Vector, stop_idxs; 
#                                  burnin::Float64=0.5,
#                                  psrf_ub::Float64=1.1,
#                                  n_eff_lb::Float64=10.0)
#    
#    nonconverged = [Vector() for l=1:length(stop_idxs)]
# 
#    # convergence for lambda?
#    push_nonconverged!(nonconverged, chain_results, ["lambda"],
#                       stop_idxs, burnin, psrf_ub, n_eff_lb, false)
#    
#    # convergence for edges?
#    V = length(chain_results[1]["parent_sets"])
#    for i=1:V
#        for j=1:V
#            push_nonconverged!(nonconverged, chain_results, ["parent_sets", i, string(j)],
#                               stop_idxs, burnin, psrf_ub, n_eff_lb, true)
#        end
#    end
#    
#    return nonconverged
#end


function get_max_n(dict_vec)
    n_vec = [d["n"] for d in dict_vec]
    return minimum(n_vec)
end


function postprocess_sample_files(sample_filenames, output_file::String, stop_points::Vector{Int},
                                  burnin::Float64)
    
    dict_vec = [load_samples(fname) for fname in sample_filenames]   

    n_max = get_max_n(dict_vec)
    stop_points = [sp for sp in stop_points if sp <= n_max] 

    results = collect_conv_stats(dict_vec, stop_points;
                                 burnin=burnin)
    
    js_str = JSON.json(results)
    f = open(output_file, "w")
    write(f, js_str)
    close(f)
    println("Saved postprocessed MCMC results to ", output_file)
end


function get_args(args::Vector{String})
    
    s = ArgParseSettings()
    @add_arg_table s begin
        "--chain-samples"
            help = "mcmc sample files for one or more chains"
    	    required = true
            nargs = '+'
        "--stop-points"
            help = "points in the sequence where convergence should be analyzed."
            required = false
            nargs = '+'
            default = [-1]
            arg_type = Int
        "--burnin"
            help = "fraction of sequence to discard as burnin."
            required = false
            arg_type = Float64
            default = 0.5
        "--output-file"
            help = "name of output JSON file containing convergence information"
    	    required = true
            arg_type = String
    end
    args = parse_args(args, s)

    arg_vec = [args["chain-samples"], args["output-file"], args["stop-points"],
               args["burnin"]]
    return arg_vec
end


# main function -- for purposes of static compilation
Base.@ccallable function julia_main(args::Vector{String})::Cint
    arg_vec = get_args(ARGS)
    postprocess_sample_files(arg_vec...)
    return 0
end


end


