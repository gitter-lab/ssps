# dbn_inference.jl
# 2019-11-11
# David Merrell
#
# Contains functions relevant for inference
# in our network reconstruction setting.



"""
Generic MCMC inference wrapper for DBN model.
Some important arguments:
* update_loop_fn: a function with signature
    (trace, acceptances) = update_loop_fn(trace, lambda_prop_args, ps_prop_args, fixed_lambda, update_lambda)
* update_results!: a function with the signature
    update_results!(results, trace)
* update_results!    
"""
function dbn_mcmc_inference(reference_adj::Vector{Vector{Bool}},
			    X::Vector{Array{Float64,2}},
                            regression_deg::Int64,
			    lambda_max::Float64,
                            n_samples::Int64,
                            burnin::Int64, thinning::Int64,
			    update_loop_fn::Function,
			    update_results!::Function,
			    lambda_prop_args::Tuple, 
			    ps_prop_args::Tuple;
			    fixed_lambda::Float64=1.0,
			    update_lambda::Bool=true,
			    track_acceptance::Bool=false,
			    update_acceptances!::Function=identity)

    # Some data preprocessing
    Xminus, Xplus  = combine_X(X)
    Xminus_stacked, Xplus = vectorize_X(Xminus, Xplus)


    # Condition the model on the data
    observations = Gen.choicemap()
    for (i, Xp) in enumerate(Xplus)
        observations[:Xplus => i => :Xp] = Xp
    end

    # Initialize the lambda value (if we aren't sampling it)
    if !update_lambda
        observations[:lambda] = fixed_lambda
    end

    # Generate an initial trace
    tr, _ = Gen.generate(dbn_model, (reference_adj, 
				     Xminus_stacked, 
                                     Xplus, 
				     lambda_max,
				     regression_deg),
			 observations)
    
    # The results we care about
    results = nothing
    acceptances = nothing
    V = length(Xplus)

    # Burn in the Markov chain
    for i=1:burnin
        tr, acc = update_loop_fn(tr, lambda_prop_args, ps_prop_args, update_lambda) 
        if track_acceptance
            acceptances = update_acceptances!(acceptances, acc, V)
	end
    end
    results = update_results!(results, tr, V)

    # Perform sampling
    for j=1:n_samples-1
	# (with thinning)
        for k=1:thinning
            tr, acc = update_loop_fn(tr, lambda_prop_args, ps_prop_args, update_lambda)
            if track_acceptance
                acceptances = update_acceptances!(acceptances, acc, V)
	    end
        end
	results = update_results!(results, tr, V)
    end

    return results, acceptances 
end


"""
Loop through the model variables and perform
Metropolis-Hastings updates on them.
Use a simple proposal that adds/removes vertices 
from parent sets.
"""
function ps_update_loop(tr, lambda_prop_args::Tuple, ps_prop_args::Tuple, 
			update_lambda::Bool)

    lambda_acc = false
    if update_lambda
        tr, lambda_acc = Gen.mh(tr, lambda_proposal, lambda_prop_args)
    end

    V = ps_prop_args[1]
    t_vec = ps_prop_args[2]
    acc_vec = zeros(V)
    for i=1:V
        tr, acc_vec[i] = Gen.mh(tr, parentvec_proposal, 
				(i, V, t_vec[i]), 
                                parentvec_involution)
    end

    return tr, (lambda_acc, acc_vec)
end


"""
Loop through the model variables and perform
Metropolis-Hastings updates on them.
Use a proposal distribution with add, remove, 
and "parent swap" moves.
"""
function ps_smart_swp_update_loop(tr, lambda_prop_args::Tuple, # (lambda_step,)
				      ps_prop_args::Tuple, # ((V, t), V) 
				      update_lambda::Bool)

    V = ps_prop_args[2]
    lambda_acc = false
    if update_lambda
        tr, lambda_acc = Gen.mh(tr, lambda_proposal, lambda_prop_args)
    end

    acc_vec = zeros(V)
    for i=1:V
        tr, acc_vec[i] = Gen.mh(tr, parentvec_smart_swp_proposal,
				(i, ps_prop_args...),
				parentvec_smart_swp_involution)
				#check_round_trip=true)
    end

    return tr, (lambda_acc, acc_vec)
end


"""

"""
function update_results_z_lambda!(results, tr, V)

    if results == nothing
        results = (zeros(Float64, V, V), Vector{Float64}())
    end

    increment_counts!(results[1], tr)
    push!(results[2], tr[:lambda])

    return results
end

function update_acc_z_lambda!(acceptances, acc, V)

    if acceptances == nothing
        acceptances = ([0.0], zeros(V))
    end

    acceptances[1] .+= acc[1]
    acceptances[2] .+= acc[2]

    return acceptances
end

"""
Helper function for updating edge counts during inference.

The entry
tr[:adjacency => :edges => i => j => :z] 
indicates existence of edge j --> i
(kind of backward from what you might expect.)

So this matrix should be read "row = child; column = parent"
"""
function increment_counts!(edge_counts, tr)
    for i=1:size(edge_counts)[1] # Child
        for j=1:size(edge_counts)[2] # Parent
            edge_counts[i,j] += tr[:adjacency => :edges => i => j => :z]
	end
    end
end




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
    adalasso_edge_recovery(X::Matrix{Float64}, y::Vector{Float64}, prior_parents::Vector{Bool})
Use Adaptive LASSO (informed by prior knowledge of edge existence)
and a Bayesian Information Criterion to get a set of regression coefficients;
nonzero coefficients imply the existence of an edge in the MAP graph.
Note that we assume normally-distributed variables.
"""
function adalasso_edge_recovery(X::Matrix{Float64}, y::Vector{Float64}, prior_parents::Vector{Bool})
    
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
    adalasso_parents(Xminus::Matrix{Float64}, Xplus::Matrix{Float64},
                     reference_parents::Vector{Vector{Bool}})

use adaptive LASSO to estimate the existence of inbound edges.
"""
function adalasso_parents(Xminus, Xplus, reference_parents)
   
    parents = []
    for i=1:size(Xplus,2)
        betas = adalasso_edge_recovery(Xminus, Xplus[i], reference_parents[i])
	push!(parents, betas .!= 0.0)
    end
    return parents
end

