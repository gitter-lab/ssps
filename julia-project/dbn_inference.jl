# dbn_inference.jl
# 2019-11-11
# David Merrell
#
# Contains functions relevant for inference
# in our network reconstruction setting.

import DataStructures: DefaultDict

"""
Generic MCMC inference wrapper for DBN model.
Some important arguments:
* update_loop_fn: a function with signature
    (trace, acceptances) = update_loop_fn(trace, lambda_prop_args, ps_prop_args, fixed_lambda, update_lambda)
* update_results: a function with the signature
    update_results(results, trace)
* update_results    
"""
function dbn_mcmc_inference(reference_parents::Vector{Vector{Int64}},
			    X::Vector{Array{Float64,2}},
                            regression_deg::Int64,
			    lambda_max::Float64,
                            n_samples_v::Vector{Int64},
                            burnin_v::Vector{Int64}, 
                            thinning_v::Vector{Int64},
			    update_loop_fn::Function,
			    update_results::Function,
			    lambda_prop_std::Float64;
			    fixed_lambda::Float64=1.0,
			    update_lambda::Bool=true,
			    track_acceptance::Bool=false,
			    update_acc_fn::Function=identity,
                            timeout::Float64=Inf)

    # start the timer
    t_start = time()

    # Some data preprocessing
    Xminus, Xplus  = combine_X(X)
    Xminus, Xplus = vectorize_X(Xminus, Xplus)

    # prepare some useful parameters
    V = length(Xplus)
    ref_parent_counts = [max(length(ps),1) for ps in reference_parents]
    proposal_param_vec = 1.0 ./ log2.(V ./ ref_parent_counts)
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
    tr, _ = Gen.generate(dbn_model, (reference_parents, 
				     Xminus, 
                                     Xplus, 
				     lambda_max,
				     regression_deg),
			 observations)
   
    # The results we care about
    results = nothing
    acceptances = nothing

    # Run the markov chain
    while true
        # update the variables    
        tr, acc = update_loop_fn(tr, lambda_prop_std, proposal_param_vec, update_lambda)
        if track_acceptance
            acceptances = update_acc_fn(acceptances, acc, V)
        end

        # update the results
        t_elapsed = time() - t_start
        results = update_results(results, tr, V, n_samples_v, burnin_v, thinning_v, t_elapsed)
        
        # check for termination
        if results["is_finished"] || (t_elapsed > timeout)
            results["elapsed"] = t_elapsed
            break
        end
    end

    return results, acceptances 
end


"""
Loop through the model variables and perform
Metropolis-Hastings updates on them.
Use a proposal distribution with add, remove, 
and "parent swap" moves.
"""
function ps_smart_swp_update_loop(tr, lambda_prop_std::Float64, 
				      proposal_param_vec::Vector{Float64},
				      update_lambda::Bool)

    V = length(proposal_param_vec)
    lambda_acc = false
    if update_lambda
        tr, lambda_acc = Gen.mh(tr, lambda_proposal, (lambda_prop_std,))
    end

    acc_vec = zeros(V)
    for i=1:V
        tr, acc_vec[i] = Gen.mh(tr, parentvec_smart_swp_proposal,
				(i, proposal_param_vec[i], V),
				parentvec_smart_swp_involution)
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


function update_results_split(results, tr, V, n_samples_v, burnin_v, thinning_v, t_elapsed)

    # Initialize the results 
    # (if they haven't been already)
    if results == nothing
        results = initialize_results_split(V, n_samples_v, burnin_v, thinning_v)
    end

    # for each combination of N, burnin, and thinning, update the results
    for (n,b,t) in keys(results["splits"])
        if (results["steps"] % t == 0) && (results["steps"] >= b) && (results["steps"] < results["splits"][(n,b,t)]["max_steps"])
            if results["splits"][(n,b,t)][1]["n"] < div(n,2)
                increment_counts!(results["splits"][(n,b,t)][1]["parent_sets"], tr)
                push!(results["splits"][(n,b,t)][1]["lambdas"], tr[:lambda])
                results["splits"][(n,b,t)][1]["n"] += 1
            else
                increment_counts!(results["splits"][(n,b,t)][2]["parent_sets"], tr)
                push!(results["splits"][(n,b,t)][2]["lambdas"], tr[:lambda])
                results["splits"][(n,b,t)][2]["n"] += 1
            end
            results["splits"][(n,b,t)]["time"] = t_elapsed
        end

    end
    results["steps"] += 1
    
    if results["steps"] >= results["max_steps"]
        results["is_finished"] = true
    end

    return results
end

function initialize_results_split(V, n_samples_v, burnin_v, thinning_v)
   
   results = Dict()
   results["steps"] = 0
   results["is_finished"] = false
   results["max_steps"] = maximum(n_samples_v)*maximum(thinning_v) + maximum(burnin_v)
   results["splits"] = Dict()

   for n in n_samples_v
       for b in burnin_v
           for t in thinning_v
               results["splits"][(n,b,t)] = Dict(
                                          1 => Dict("parent_sets" => [DefaultDict{Int64,Int64}(0) for i=1:V],
                                                    "lambdas" => Vector{Float64}(),
                                                    "n" => 0
   	                             		    ),
   	                                  2 => Dict("parent_sets" => [DefaultDict{Int64,Int64}(0) for i=1:V],
                                                    "lambdas" => Vector{Float64}(),
                                                    "n" => 0
   	                             		    ),
                                          "max_steps" => n*t + b,
                                          "time" => 0.0
   	                                         )
           end
       end
   end
   return results
end

#function update_results_store_samples(results, tr, V, n_samples)
#
#    if results == nothing
#        results = Dict("parent_sets" => [], 
#		       "lambdas" => []
#		       )
#    end
#    ps = [[tr[:adjacency => :edges => c => p => :z] for c=1:V] for p=1:V]
#    push!(results["parent_sets"], ps)
#    push!(results["lambdas"], tr[:lambda])
#
#    return results
#end
#
#
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
   
    for i=1:length(edge_counts) # for each vertex
        # increment the parents
        ps = tr[:parent_sets => i => :parents]
        for p in ps
            edge_counts[i][p] += 1
	end
    end
end




