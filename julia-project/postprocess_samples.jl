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

include("changepoint_vec.jl")
include("nested_data.jl")

export postprocess_sample_files, julia_main

"""
Read an MCMC samples output file
"""
function load_samples(samples_file_path::String)

    f = open(samples_file_path, "r")
    str = read(f, String)
    close(f)
    d = JSON.parse(str)
    
    d["parent_sets"] = [DefaultDict([], ps) for ps in d["parent_sets"]]
    V = length(d["parent_sets"])
    d["parent_sets"] = [[ps[string(p)] for p=1:V] for ps in d["parent_sets"]]

    return d 
end


function seq_variogram(seq::ChangepointVec, t::Int64; is_binary::Bool=false)
    n = length(seq)
    if is_binary
        return mean(binop(!=, seq[1:n-t], seq[1+t:n]); is_binary=is_binary)
    else
        return mean(map(abs2, binop(-, seq[1:n-t], seq[1+t:n])); is_binary=is_binary)
    end
end


function seq_variogram(seq::Vector, t::Int64; is_binary::Bool=false)
    n = length(seq)
    if is_binary
        return sum(seq[1:n-t] .!= seq[1+t:n])/length(seq)
    else
        return sum(abs2.(seq[1:n-t] .- seq[1+t:n]))/length(seq)
    end
end


function correlation_sum(seqs::Vector{ChangepointVec}, varplus::Float64; 
                         is_binary::Bool=false, density_threshold::Float64=0.001)
    
    s = 0.0
    m = length(seqs)
    prev2_corr = 0.0
    prev1_corr = 0.0
  
    # Determine whether the sequences need to be converted to dense vectors. 
    seq_densities = [length(seq.changepoints)/length(seq) for seq in seqs]
    seqs = [(seq_densities[i] < density_threshold ? seq : to_vec(seq) ) for (i, seq) in enumerate(seqs)]

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
    #println("\tDENSITIES: ", seq_densities)
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


function is_binary(cpv)
    for pair in cpv.changepoints
        if !(pair[2] == 0 || pair[2] == 1)
            return false
        end
    end
    return true
end


"""
Compute convergence diagnostics for a collection of sequences,
at multiple stop indices
"""
function evaluate_convergence(whole_seqs::Vector{ChangepointVec}, stop_idxs;
                              burnin::Float64=0.5)
    
    # Determine whether the data is binary
    isbin = is_binary(whole_seqs[1])
    #println("IS BINARY?: ", isbin)

    # We'll collect a list of (psrf, n_eff values) 
    diagnostics = []

    for len in stop_idxs
        # Split the whole sequences into half sequences
        burnin_idx = Int(floor(burnin*len))
        split_idx = burnin_idx + div(len - burnin_idx, 2)
        half_seqs = Vector{ChangepointVec}()
        for ws in whole_seqs
            push!(half_seqs, ws[burnin_idx+1:split_idx])
            push!(half_seqs, ws[split_idx+1:len])
        end
        
        # compute the convergence diagnostics
        psrf, n_eff = compute_psrf_neff(half_seqs; is_binary=isbin)
        push!(diagnostics, (psrf, n_eff))
    end
    
    return diagnostics
end


function evaluate_summary(whole_seqs::Vector{ChangepointVec}, stop_idxs;
                          burnin::Float64=0.5)

    burnin_idxs = [Int(floor(burnin*length(seq))) for seq in whole_seqs]
    sums = [sum(cpv[burnin_idxs[i]:length(cpv)]) for (i, cpv) in enumerate(whole_seqs)]
    ns = [length(cpv) - burnin_idxs[i] for (i, cpv) in enumerate(whole_seqs)]

    return sum(sums) / sum(ns)
end


function is_seq(node)
    if !(typeof(node) <: AbstractVector)
        return false
    end
    for (i, item) in enumerate(node)
        if !(length(item) == 2) || !(typeof(item[1]) <: Number)
            return false
        end
    end
    return true
end

is_cpv = x -> (typeof(x) <: ChangepointVec)

###############################################################

###############################################################
# CORE FUNCTIONS
function collect_stats(dict_vec, stop_points; burnin::Float64=0.5)

    results = Dict()
    results["stop_points"] = stop_points

    # Convert all of the sequences to ChangepointVecs
    dict_vec = [leaf_map(x -> ChangepointVec(x, d["n"]), d, is_seq) for d in dict_vec]
    
    results["summary"] = aggregate_trees(x -> evaluate_summary(x, stop_points; burnin=burnin),
                                         dict_vec, is_cpv)
    results["conv_stats"] = aggregate_trees(x -> evaluate_convergence(x, stop_points; burnin=burnin),
                                            dict_vec, is_cpv)
   
    results["parent_sets"] = results["summary"]["parent_sets"]
    results["edge_conf_key"] = "parent_sets"

    delete!(results["summary"], "parent_sets") 
    return results
end


function get_max_n(dict_vec)
    n_vec = [d["n"] for d in dict_vec]
    return minimum(n_vec)
end


function postprocess_sample_files(sample_filenames, output_file::String, stop_points::Vector{Int},
                                  burnin::Float64)
    dict_vec = [load_samples(fname) for fname in sample_filenames]   

    n_max = get_max_n(dict_vec)
    if length(stop_points) == 0
        stop_points = [n_max]
    else
        stop_points = [sp for sp in stop_points if sp <= n_max] 
    end

    results = collect_stats(dict_vec, stop_points; burnin=burnin)
   
    f = open(output_file, "w")
    JSON.print(f, results)
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
            default = Vector{Int64}()
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
function julia_main()
    arg_vec = get_args(ARGS)
    postprocess_sample_files(arg_vec...)
    return 0
end

end

#using .SamplePostprocess

#julia_main()

