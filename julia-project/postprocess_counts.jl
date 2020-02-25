# postprocess_counts.jl
# 2020-01-17
# David Merrell
#
# Functions for postprocessing MCMC results.
# Specifically for converting counts -> predictions,
# which can then be used in evaluation.

module PostprocessCounts

using JSON
using ArgParse
using DataStructures

export julia_main

"""
Get the summary statistics from a MCMC run.
"""
function extract_results(json_filename::String)

    f = open(json_filename, "r")
    str = read(f, String)
    close(f)
    d = JSON.parse(str)

    return d
end


"""
Return probabilistic parent sets;
i.e., prob_ps[j][i] = prob(edge i->j exists)
"""
function prob_parent_sets(parent_sets_vec::Vector, n_vec::Vector)

    V = length(parent_sets_vec[1])
    prob_ps = [zeros(V) for v=1:V]

    for psets in parent_sets_vec
        for (j, ps) in enumerate(psets)
            ps = DefaultDict(0, ps)
            for i=1:V
                prob_ps[j][i] += ps[string(i)] 
            end
        end
    end

    n = sum(n_vec)
    return map(v -> v./n, prob_ps)
end


function parse_script_arguments(args::Vector{String})
    
    s = ArgParseSettings()
    @add_arg_table s begin
        "input_filenames"
            help="One or more MCMC summary filenames"
            nargs='+'
        "--output-file"
            help="Name of output file containing edge probabilities"
            required = true
            arg_type = String
    end

    args = parse_args(args, s)
    return args
end


"""
Docstring
"""
function postprocessor(args::Vector{String}; conf_key::String="parent_sets")

    arg_dict = parse_script_arguments(args)
    filenames = [string(fn) for fn in arg_dict["input_filenames"]]
    summary_dicts = [extract_results(fname) for fname in filenames]

    results = Dict()
    results["edge_conf_key"] =  conf_key

    psets = [d["parent_sets"] for d in summary_dicts]
    n_vec = [d["n"] for d in summary_dicts]

    prob_ps = prob_parent_sets(psets, n_vec)
    results[conf_key] = prob_ps
    results["n"] = n_vec

    out_fname = arg_dict["output-file"]

    #res_str = JSON.json(results)

    f = open(out_fname, "w")
    # write(f, res_str)
    JSON.print(f, results)
    close(f)

    println("Saved postprocessed MCMC results to ", out_fname)

end


# The main function, called from the command line.
function julia_main()
    postprocessor(ARGS)
    return 0
end

end

using .PostprocessCounts

julia_main()


