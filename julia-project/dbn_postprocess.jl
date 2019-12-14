# dbn_postprocess.jl
# 2019-12-13
# David Merrell
#
# Functions for postprocessing MCMC results.
# We usually assume the results reside in a JSON
# file.

using JSON

"""
Extract a vector of parent sets from the given JSON file.
"""
function extract_samples(json_filename::String)

    f = open(json_filename, "r")
    str = read(f, String)
    close(f)
    d = JSON.parse(str)

    ps_samples = [[[Bool(v) for v in row] for row in mat ] for mat in d["parent_sets"]]
    
    lambda_samples = [Float64(v) for v in d["lambdas"]]

    return ps_samples, lambda_samples
end


"""

"""
function sample_average(sample_vec::Vector)
    return 1.0*sum(sample_vec) / length(sample_vec)
end


"""
Split a vector of samples into (1) the first half and (2) the second half
"""
function split_sample(sample_vec::Vector)
    h = div(length(sample_vec), 2)
    return sample_vec[1:h], sample_vec[h+1:end] 
end


function dumb_test()
    ps_samples, lambda_samples = extract_samples("dumb_output.json")
end

dumb_test()
