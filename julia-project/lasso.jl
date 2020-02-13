# lasso.jl
# 2020-01-02
# David Merrell
#
# A module for performing network inference
# via adaptive LASSO.
# It's intended for static compilation.

module Lasso

include("dbn_preprocess.jl")
using GLMNet, JSON, ArgParse

"""
    compute_bic(ols_path, lasso_path, n::Int64, ols_df::Int64)

Compute a vector of Bayesian Information Criterion values for a
GLMNet path.
"""
function compute_bic(ols_path, lasso_path, n::Int64, ols_df::Int64) 
    
    ols_ssr = (1.0 - ols_path.dev_ratio[1]) # *null deviation
    ssr_vec = (1.0 .- lasso_path.dev_ratio) # *null deviation
    df_vec = sum(lasso_path.betas .!= 0.0, dims=1)[1,:]
    
    bic_vec = ((n - ols_df).*ssr_vec./ols_ssr  .+  log(n).*df_vec) ./n # null deviations cancel out
    
    return bic_vec
end


"""
    adalasso_parents(X::Matrix{Float64}, y::Vector{Float64}, prior_parents::Vector{Bool})
Use Adaptive LASSO (informed by prior knowledge of edge existence)
and a Bayesian Information Criterion to get a set of regression coefficients;
nonzero coefficients imply the existence of an edge in the MAP graph.
Note that we assume normally-distributed variables.
"""
function adalasso_parents(X::Matrix{Float64}, y::Vector{Float64}, prior_parents::Vector{Bool})
    
    n = size(y,1)
    yp = y .* n ./ (n+1)
    ols_result = GLMNet.glmnet(X, yp, lambda=[0.0])

    ada_weights = 1.0 .- prior_parents;
    adaptive_penalties = abs.(ada_weights ./ ols_result.betas)[:,1]
    
    lasso_path = GLMNet.glmnet(X, yp, penalty_factor=adaptive_penalties)
    
    bic_vec = compute_bic(ols_result, lasso_path, n, size(X,2))
    
    minloss_idx = argmin(bic_vec)
    
    return lasso_path.betas[:,minloss_idx]

end


"""
    adalasso_edge_recovery(Xminus::Matrix{Float64}, Xplus::Matrix{Float64},
                     reference_parents::Vector{Vector{Bool}})

use adaptive LASSO to estimate the existence of network edges
from timeseries data.
"""
function adalasso_edge_recovery(Xminus, Xplus, reference_parents)
   
    parents = []
    for i=1:size(Xplus,2)
        betas = adalasso_parents(Xminus, Xplus[i], reference_parents[i])
        betas = abs.(betas)
        mx = maximum(betas)
        if mx == 0.0
            mx = 1.0
        end
        push!(parents, betas ./ mx )
    end
    return parents
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


# Main function -- used for static compilation
Base.@ccallable function julia_main(args::Vector{String})::Cint

    arg_dict = get_args(args)
    ts_fname = arg_dict["timeseries_fname"]
    ref_dg_fname = arg_dict["ref_dag_fname"]

    ts_vec, ref_ps = load_simulated_data(ts_fname, ref_dg_fname)
    Xminus, Xplus = combine_X(ts_vec)

    pred = adalasso_edge_recovery(Xminus, Xplus, ref_ps)

    result = Dict{String,Any}()
    result["edge_conf_key"] = "abs_coeffs"
    result["abs_coeffs"] = pred

    f = open(arg_dict["outfile"], "w")
    js_str = JSON.json(result)
    write(f, js_str)
    close(f)
end


end # END MODULE
