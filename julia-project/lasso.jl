# lasso.jl
# 2020-01-02
# David Merrell
#
# A module for performing network inference via adaptive LASSO.
# It selects the LASSO parameter using Bayesian Information Criterion.

module Lasso

include("dbn_preprocess.jl")
using GLMNet, JSON, ArgParse

export julia_main

"""
    compute_bic(lasso_path, n::Int64)

Compute a vector of Bayesian Information Criterion values for a
GLMNet path.
"""
function compute_bic(lasso_path, n::Int64) 
    
    df_vec = sum(lasso_path.betas .!= 0.0, dims=1)[1,:]
    bic_vec = (1.0 .- lasso_path.dev_ratio) .+ df_vec.* log(n)./n 

    return bic_vec
end


"""
    priorlasso_parents(X::Matrix{Float64}, y::Vector{Float64}, prior_parents::Vector{Bool})
Use prior-weighted LASSO (informed by prior knowledge of edge existence)
and a Bayesian Information Criterion to get a set of regression coefficients;
nonzero coefficients imply the existence of an edge in the MAP graph.
Note that we assume normally-distributed variables.
"""
function priorlasso_parents(X::Matrix{Float64}, y::Vector{Float64}, prior_parents::Vector)
    
    n = size(y,1)
    yp = y .* n ./ (n+1)
    ypvar = var(yp)

    penalties = exp.(-1.0.* prior_parents);
    lasso_path = GLMNet.glmnet(X, yp, penalty_factor=penalties)
 
    bic_vec = compute_bic(lasso_path, n)
    minloss_idx = argmin(bic_vec)

    return lasso_path.betas[:,minloss_idx]

end


"""
    priorlasso_edge_recovery(Xminus::Matrix{Float64}, Xplus::Matrix{Float64},
                     reference_parents::Vector{Vector{Bool}})

use adaptive LASSO to estimate the existence of network edges
from timeseries data.
"""
function priorlasso_edge_recovery(Xminus, Xplus, reference_parents)
   
    parents = []
    for i=1:length(Xplus)
        betas = priorlasso_parents(Xminus, Xplus[i], reference_parents[i])
        betas = abs.(betas)
        mx = maximum(betas)
        if mx == 0.0
            mx = 1.0
        end
        push!(parents, betas ./ mx )
    end
    return parents
end


function parentdict2vec(parent_dicts::Vector{Dict{Int64,Float64}})
    V = length(parent_dicts)

    parent_vecs = [[0.0 for i=1:V] for j=1:V]
    for (i, d) in enumerate(parent_dicts)
        for pair in pairs(d)
            parent_vecs[i][first(pair)] = last(pair)
        end
    end

    return parent_vecs
end

function get_args(ARGS::Vector{String})

    s = ArgParseSettings()
    @add_arg_table s begin
        "timeseries_fname"
            help="Path to a TSV file containing time series data"
            required=true
            arg_type=String
        "graph_prior"
            help="Path to a CSV file containing a prior distribution for edges"
            required=true
            arg_type=String
        "outfile"
            help="The path of an output JSON file"
            required=true
            arg_type=String
    end

    args = parse_args(ARGS, s)
    return args
end



function julia_main()

    arg_dict = get_args(ARGS)
    ts_fname = arg_dict["timeseries_fname"]
    ref_dg_fname = arg_dict["graph_prior"]

    # load and preprocess data
    ts_vec, ref_ps = load_formatted_data(ts_fname, ref_dg_fname; boolean_adj=false)
    Xminus, Xplus = combine_X(ts_vec)
    Xminus = standardize_X(Xminus)[1]
    Xplus = standardize_X(Xplus)[1]
    Xminus, Xplus = vectorize_X(Xminus, Xplus)
    
    ref_ps = parentdict2vec(ref_ps)

    # Perform the Prior LASSO predictions
    pred = priorlasso_edge_recovery(Xminus, Xplus, ref_ps)

    result = Dict{String,Any}()
    result["edge_conf_key"] = "abs_coeffs"
    result["abs_coeffs"] = pred

    f = open(arg_dict["outfile"], "w")
    #js_str = JSON.json(result)
    JSON.print(f, result)
    close(f)
end


end # END MODULE

using .Lasso
julia_main()

