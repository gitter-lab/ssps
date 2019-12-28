# dbn_postprocess.jl
# 2019-12-13
# David Merrell
#
# Functions for postprocessing MCMC results.
# We assume the results reside in a JSON
# file. Samples are segregated into a "first_half"
# and "second_half" for purposes of computing
# R statistics (assessing mixing & stationarity).

module Postprocess

using JSON
using ArgParse

export postprocessor

"""
Extract a vector of parent sets from the given JSON file.
Return a dictionary of vectors of counts (for parent sets;
for lambdas, return the actual samples rather than counts).
"""
function extract_results(json_filename::String)

    f = open(json_filename, "r")
    str = read(f, String)
    close(f)
    d = JSON.parse(str)

    res1 = d["first_half"]
    res2 = d["second_half"]
    res1["parent_sets"] = [Vector{Int64}(row) for row in res1["parent_sets"]]
    res2["parent_sets"] = [Vector{Int64}(row) for row in res2["parent_sets"]]
    
    res1["lambdas"] = [Float64(v) for v in res1["lambdas"]]
    res2["lambdas"] = [Float64(v) for v in res2["lambdas"]]

    n_samples = d["n_samples"]
    res1["n_samples"] = div(n_samples,2)
    res2["n_samples"] = n_samples - res1["n_samples"]

    return [res1; res2]
end

"""
Extract MCMC results from a set of files given by the 
json_filenames parameter. Return a vector of Dictionaries,
each one containing a summary of a MCMC run.
"""
function extract_all_results(json_filenames::Vector{String})
    return vcat([extract_results(jsf) for jsf in json_filenames]...)
end


"""
Performs a reduce operation `f` on a vector of numbers.
"""
function myreduce(f::Function, vec::Vector{T}) where {T<:Number}
    return f(vec)
end


"""
Performs a reduce operation `f` on a vector of nested vectors.
(Reduction applied on the top-level index.)
"""
function myreduce(f::Function, vec::Vector{Vector{T}}) where T
    return [myreduce(f, [item[field] for item in vec]) for (field, _) in enumerate(vec[1])]
end


"""
Perform a binary operation f(vec1, vec2) on arbitrarily nested vectors
(assume identical structure between vec1 and vec2, though)
"""
function mybinop(f::Function, vec1::Vector{T}, vec2::Vector{U}) where {T,U}
    return [mybinop(f, v1i, vec2[i]) for (i,v1i) in enumerate(vec1)]
end

function mybinop(f::Function, vec1::Vector{T}, v2::U) where{T, U<:Number}
    return [mybinop(f, v1, v2) for v1 in vec1]
end

function mybinop(f::Function, v1::T, vec2::Vector{U}) where{T<:Number, U}
    return [mybinop(f, v1, v2) for v2 in vec2]
end

function mybinop(f::Function, v1::T, v2::U) where {T<:Number,U<:Number}
    return f(v1, v2)
end

"""
Map a function on to values stored in arbitrarily nested vectors.
"""
function mymap(f::Function, vec::Vector{T}) where T
    return [mymap(f, v) for v in vec]
end

function mymap(f::Function, v::T) where {T<:Number}
    return f(v)
end

"""
Compute the sample averages over a vector of MCMC results
"""
function sample_mean(sample_vec::Vector)
    return myreduce(v->1.0*sum(v)/length(v), sample_vec) 
end

"""
Compute a sample average from *counts*
"""
function count_mean(count_vec::Vector, n_total::Int64)
    return mymap(x->x/n_total, count_vec)
end


"""
Compute the sample variances over a vector of MCMC results
"""
function sample_variance(sample_vec::Vector)
    mu = sample_mean(sample_vec)
    centered = [mybinop(-, s, mu) for s in sample_vec]
    return myreduce(x->sum(abs2.(x))/(length(centered)-1.0), centered)
end


"""
Compute a sample variance from *counts*
"""
function count_variance(count_vec::Vector, n_total::Int64)
    return mybinop(*, mymap(x->1.0 - x/n_total, count_vec),
		       mymap(x->x/(n_total-1.0), count_vec)
		    )
end


"""
Split a vector of samples into first/second halves
"""
function split_sample(sample_vec::Vector)
    h = div(length(sample_vec), 2)
    return sample_vec[1:h], sample_vec[h+1:end] 
end

"""
Compute the Between-sequence variance (precursor to R statistic)
"""
function B_statistic(sequences::Vector{T}) where T

    seq_means = [sample_mean(s) for s in sequences]
    return sample_variance(seq_means)
end

"""
Compute Between-sequence variance from *counts*
"""
function count_B_statistic(seq_counts::Vector{T}, seq_ns::Vector{Int64}) where T

    seq_means = [count_mean(c, seq_ns[i]) for (i,c) in enumerate(seq_counts)]
    return sample_variance(seq_means)    
end

"""
Compute the Within-sequence variance (precursor to R statistic)
"""
function W_statistic(sequences::Vector{T}) where T
    
    seq_vars = [sample_variance(s) for s in sequences]
    return sample_mean(seq_vars)
end

"""
Compute the Within-sequence variance (precursor to R statistic) using *counts*
"""
function count_W_statistic(seq_counts::Vector{T}, seq_ns::Vector{Int64}) where T

    seq_vars = [count_variance(c, seq_ns[i]) for (i,c) in enumerate(seq_counts)]
    return sample_mean(seq_vars)
end


"""
Compute the R statistic for a collection of sequences.
Mainly used as a helper function in `compute_R_statistic`. 
"""
function _R_stat(sequences::Vector{T}) where T
    
    B = B_statistic(sequences)
    W = W_statistic(sequences)
    n = length(sequences[1])
    return mymap(x-> sqrt(1.0 - 1.0/n + x),
		     mybinop(/, B, W)
		 )
end


"""
Compute the R statistic for a collection of sequences -- from *counts*.
"""
function _count_R_stat(seq_counts::Vector{T}, seq_ns::Vector{Int64}) where T

    B = count_B_statistic(seq_counts, seq_ns)
    W = count_W_statistic(seq_counts, seq_ns)
    n = seq_ns[1]
    return mymap(x-> sqrt(1.0 - 1.0/n + x),
                     mybinop(/, B, W)
                 )
end


"""
Split each sequence in half, and compute their R statistic.
"""
function compute_R_statistic(sequences::Vector{T} where T)

    split_sequences = []
    for seq in sequences
	    a,b = split_sample(seq)
	    push!(split_sequences, a)
	    push!(split_sequences, b)
    end

    return _R_stat(split_sequences)
end


function parse_script_arguments(args::Vector{String})
    
    s = ArgParseSettings()
    @add_arg_table s begin
        "filenames"
            help="A list (variable length) of whitespace-separated file names."
            nargs='+'
        "--skip-R"
            help="Do NOT return the R statistic"
	        action = :store_true
        "--skip-var"
            help="Do NOT return the posterior sample variance"
            action = :store_true
        "--output-file"
            required = true
            arg_type = String
    end

    args = parse_args(args, s)
   
    return args
end


"""
Docstring
"""
function postprocessor(args::Vector{String})

    arg_dict = parse_script_arguments(args)
    filenames = [string(fn) for fn in arg_dict["filenames"]]
    mcmc_results = extract_all_results(filenames)
    postprocessed = Dict()

    ps_counts = [mr["parent_sets"] for mr in mcmc_results]
    ps_ns = [mr["n_samples"] for mr in mcmc_results]
    lambdas = [mr["lambdas"] for mr in mcmc_results]

    summed_ps = myreduce(sum, ps_counts)
    summed_lambdas = sum(vcat(lambdas...))
    summed_ns = sum(ps_ns)

    # compute the means of posterior estimands
    postprocessed["ps_mean"] = count_mean(summed_ps, summed_ns)
    postprocessed["lambda_mean"] = summed_lambdas / summed_ns

    # compute the variances of posterior estimands
    if !arg_dict["skip-var"]

        postprocessed["lambda_var"] = sample_variance(vcat(lambdas...)) 
        postprocessed["ps_var"] = count_variance(myreduce(sum, ps_counts), 
                                                 sum(ps_ns))
    end

    # compute the R statistics for posterior estimands
    if !arg_dict["skip-R"]
        postprocessed["lambda_R"] = _R_stat(lambdas) 
        postprocessed["ps_R"] = _count_R_stat(ps_counts, ps_ns)
    end

    pp_str = JSON.json(postprocessed)
    out_fname = arg_dict["output-file"]
    f = open(out_fname, "w")
    write(f, pp_str)
    close(f)
    println("Saved postprocessed MCMC results to ", out_fname)

end


# The main function, called from the command line.
Base.@ccallable function julia_main(args::Vector{String})::Cint
    postprocessor(args)
    return 0
end


end


