# dbn_postprocess.jl
# 2019-12-13
# David Merrell
#
# Functions for postprocessing MCMC results.
# We assume the results reside in a JSON
# file. Samples are segregated into a "first_half"
# and "second_half" for purposes of computing
# R statistics (assessing mixing & stationarity).

using JSON

"""
Extract a vector of parent sets from the given JSON file.
"""
function extract_samples(json_filename::String)

    f = open(json_filename, "r")
    str = read(f, String)
    close(f)
    d = JSON.parse(str)

    ps_samples_1 = [[[Bool(v) for v in row] for row in mat ] for mat in d["first_half"]["parent_sets"]]
    ps_samples_2 = [[[Bool(v) for v in row] for row in mat ] for mat in d["first_half"]["parent_sets"]]
    lambda_samples_1 = [Float64(v) for v in d["first_half"]["lambdas"]]
    lambda_samples_2 = [Float64(v) for v in d["second_half"]["lambdas"]]

    n_samples = d["n_samples"]

    return ps_samples_1, ps_samples_2, lambda_samples_1, lambda_samples_2, n_samples
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
    return my_binop(*, mymap(x->1.0 - x/n_total, count_vec),
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
function _count_R_stat(seq_counts::Vector{T}, seq_ns::Vector{Float64}) where T

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


function dumb_test()
    ps_samples, lambda_samples = extract_samples("dumb_output.json")
    println(compute_R_statistic([ps_samples]))
end

dumb_test()
