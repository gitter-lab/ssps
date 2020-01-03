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
* update_results: a function with the signature
    update_results(results, trace)
* update_results    
"""
function dbn_mcmc_inference(reference_adj::Vector{Vector{Bool}},
			    X::Vector{Array{Float64,2}},
                            regression_deg::Int64,
			    lambda_max::Float64,
                            n_samples::Int64,
                            burnin::Int64, thinning::Int64,
			    update_loop_fn::Function,
			    update_results::Function,
			    lambda_prop_std::Float64;
			    fixed_lambda::Float64=1.0,
			    update_lambda::Bool=true,
			    track_acceptance::Bool=false,
			    update_acc_fn::Function=identity)

    # Some data preprocessing
    Xminus, Xplus  = combine_X(X)
    Xminus_stacked, Xplus = vectorize_X(Xminus, Xplus)

    # prepare some useful parameters
    V = length(Xplus)
    avg_parents = 1.0*sum([sum(ps) for ps in reference_adj]) / V
    t = 1.0 / log2(V / avg_parents)
    ps_prop_args = ((V, t), V)
    lambda_prop_args = (lambda_prop_std,)
    if regression_deg == -1
        regression_deg = V
    end

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

    # Burn in the Markov chain
    for i=1:burnin
        tr, acc = update_loop_fn(tr, lambda_prop_args, ps_prop_args, update_lambda) 
        if track_acceptance
            acceptances = update_acc_fn(acceptances, acc, V)
	end
    end

    # Perform sampling
    results = update_results(results, tr, V, n_samples)
    for j=1:n_samples-1
	# (with thinning)
        for k=1:thinning
            tr, acc = update_loop_fn(tr, lambda_prop_args, ps_prop_args, update_lambda)
            if track_acceptance
                acceptances = update_acc_fn(acceptances, acc, V)
	    end
        end
	results = update_results(results, tr, V, n_samples)
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
function update_results_z_lambda(results, tr, V, n_samples)

    if results == nothing
        results = (zeros(Float64, V, V), Vector{Float64}())
    end

    increment_counts!(results[1], tr)
    push!(results[2], tr[:lambda])

    return results
end


function update_results_split(results, tr, V, n_samples)

    if results == nothing
        results = Dict("i" => 1,
		       "n_samples" => n_samples,
		       "first_half" => Dict("parent_sets" => zeros(Float64, V, V),
                                    "lambdas" => Vector{Float64}()
					    ),
		       "second_half" => Dict("parent_sets" => zeros(Float64, V, V),
                                     "lambdas" => Vector{Float64}()
					    )
		       )
    end

    if results["i"] <= div(n_samples,2)
	increment_counts!(results["first_half"]["parent_sets"], tr)
	push!(results["first_half"]["lambdas"], tr[:lambda])
    else
	increment_counts!(results["second_half"]["parent_sets"], tr)
	push!(results["second_half"]["lambdas"], tr[:lambda])
    end

    results["i"] += 1

    return results
end


function update_results_store_samples(results, tr, V, n_samples)

    if results == nothing
        results = Dict("parent_sets" => [], 
		       "lambdas" => []
		       )
    end
    ps = [[tr[:adjacency => :edges => c => p => :z] for c=1:V] for p=1:V]
    push!(results["parent_sets"], ps)
    push!(results["lambdas"], tr[:lambda])

    return results
end


function update_acc_z_lambda(acceptances, acc, V)

    if acceptances == nothing
        acceptances = Dict("lambdas" => 0, 
			   "parent_sets" => zeros(V),
			   "n_proposals" => 0
			   )
    end

    acceptances["n_proposals"] += 1
    acceptances["lambdas"] += acc[1]
    acceptances["parent_sets"] .+= acc[2]

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




