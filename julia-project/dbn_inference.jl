# dbn_inference.jl
# 2019-11-11
# David Merrell
#
# Contains functions relevant for inference
# in our network reconstruction setting.


"""
Generic MCMC inference wrapper for DBN model.
Some important arguments:
"""
function dbn_mcmc_inference(reference_parents::Vector,
			    X::Vector{Array{Float64,2}};
                            regression_deg::Int64=3,
                            timeout::Float64=3600.0,
                            n_steps::Int64=-1,
                            store_samples::Bool=false,
                            burnin::Float64=0.5, 
                            thinning::Int64=5,
			    update_loop_fn::Function=smart_update_loop,
			    update_results_fn::Function=update_results_summary,
                            large_indeg::Float64=20.0,
			    lambda_max::Float64=15.0,
			    lambda_prop_std::Float64=0.25,
			    track_acceptance::Bool=false,
			    update_acc_fn::Function=update_acc,
                            bool_prior::Bool=true)

    # Some data preprocessing
    Xminus, Xplus  = combine_X(X)
    Xminus, Xplus = vectorize_X(Xminus, Xplus)
    
    # start the timer
    t_start = time()
    t_elapsed = 0.0

    # prepare some useful parameters
    V = length(Xplus)
    if bool_prior
        ref_parent_counts = [max(length(ps),1) for ps in reference_parents]
    else
        ref_parent_counts = [max(sum(values(ps)), 2.0) for ps in reference_parents]
    end
    proposal_param_vec = 1.0 ./ log2.(V ./ ref_parent_counts)
    lambda_min = log(max(V/large_indeg - 1.0, exp(0.5)))
    
    # Check for some default arguments 
    if regression_deg == -1
        regression_deg = V
    end
    if n_steps == -1
        n_steps = Inf
    end
    if store_samples 
        track_acceptance = false
        burnin = 0.0
    end 

    # Condition the model on the data
    observations = Gen.choicemap()
    for (i, Xp) in enumerate(Xplus)
        observations[:Xplus => i => :Xp] = Xp
    end

    # Determine which generative model we should use
    # (it depends on the kind of prior knowledge available)
    if bool_prior
        gen_model = dbn_model
    else
        gen_model = conf_dbn_model
    end

    # Generate an initial trace
    tr, _ = Gen.generate(gen_model, (reference_parents, 
				     Xminus, 
                                     Xplus,
                                     lambda_min, 
				     lambda_max,
				     regression_deg),
			 observations)
   
    # The results we care about
    results = nothing
    acceptances = nothing

    # Burn-in loop:
    burnin_count = 0
    t_burn = burnin*timeout
    n_burn = burnin*n_steps
    println("Burning in for ", round(t_burn - t_elapsed), " seconds. (Or ", n_burn, " steps).")
    t_print = 0.0
    while t_elapsed < t_burn && burnin_count < n_burn

        if (burnin_count > 0) && (t_elapsed - t_print >= t_burn/10.0)
            println("\t", burnin_count, " steps in ", round(t_elapsed), " seconds." )
            t_print = t_elapsed 
        end 

        tr, acc = update_loop_fn(tr, lambda_prop_std, proposal_param_vec)     
        t_elapsed = time() - t_start
        burnin_count += 1
    end
    t_end_burn = time()
    t_burn = t_end_burn - t_start

    # Sampling loop
    prop_count = 0
    println("Sampling for ", round(timeout - t_burn), " seconds. (Or ", n_steps - n_burn," steps).")
    t_print = 0.0
    while t_elapsed < timeout && prop_count <= n_steps
        
        # thinning loop (if applicable)
        for i=1:thinning
            
            # Print progress
            if (prop_count > 0) && (t_elapsed - t_print >= 20.0)
                println("\t", prop_count," steps in ", round(t_elapsed - t_burn), " seconds.")
                t_print = t_elapsed 
            end

            # update the variables    
            tr, acc = update_loop_fn(tr, lambda_prop_std, proposal_param_vec)
            if track_acceptance
                acceptances = update_acc_fn(acceptances, acc, V)
            end
            prop_count += 1
            
            t_elapsed = time() - t_start
        end

        # update the results
        results = update_results_fn(results, tr, V)
        
    end
 
    # Some last updates to the `results` object
    results["burnin_count"] = burnin_count
    if store_samples
        delete!(results, "prev_parents") 
    end

    return results, acceptances 
end


"""
Loop through the model variables and perform
Metropolis-Hastings updates on them.
It's called ``smart'' because it uses a fancy
proposal distribution for the parent sets.
"""
function smart_update_loop(tr, lambda_prop_std::Float64, 
	  		       proposal_param_vec::Vector{Float64})

    V = length(proposal_param_vec)
    lambda_acc = false
    tr, lambda_acc = Gen.mh(tr, lambda_proposal, (lambda_prop_std,))

    acc_vec = zeros(V)
    for i=1:V
        tr, acc_vec[i] = Gen.mh(tr, smart_proposal,
				(i, proposal_param_vec[i], V),
				smart_involution)
    end

    return tr, (lambda_acc, acc_vec)
end


import DataStructures: DefaultDict


"""
Aggregate summary statistics for the quantities of interest
while the markov chain runs, in an online fashion.
"""
function update_results_summary(results, tr, V)

    if results == nothing
        results = Dict("parent_sets" => [DefaultDict{Int64,Int64}(0) for i=1:V],
                       "lambda_mean" => 0.0,
                       "lambda_var" => 0.0,
                       "n" => 0
                       )
    end

    increment_counts!(results["parent_sets"], tr)
    
    results["lambda_mean"], 
    results["lambda_var"], 
    results["n"] = update_moments(results["lambda_mean"],
                                  results["lambda_var"],
                                  tr[:lambda],results["n"])

    return results
end


function update_moments(cur_mean, cur_var, new_x, cur_N)
    new_N = cur_N + 1
    diff = new_x - cur_mean
    new_mean = cur_mean + diff/new_N
    diff2 = new_x - new_mean
    if cur_N == 0
        new_var = 0.0
    else
        new_var = ((cur_N-1)*cur_var + diff*diff2)/cur_N
    end
    return new_mean, new_var, new_N
end


"""
Store a compact representation of all samples by
tracking *changes* in the model variables.
"""
function update_results_storediff(results, tr, V)

    # Initialize the state
    if results == nothing
        results = Dict("parent_sets" => [DefaultDict{Int64,Vector{Tuple{Int64,Int64}}}([]) for i=1:V],
                       "lambda" => Vector{Tuple{Int64,Float64}}(),
                       "n" => 1,
                       "prev_parents" => [Vector{Int64}() for i=1:V]
                       )
        
        push!(results["lambda"], (1, tr[:lambda]))
        for i=1:V
            for parent in tr[:parent_sets => i => :parents]
                push!(results["parent_sets"][i][parent], (1,1))
            end
            results["prev_parents"][i] = tr[:parent_sets => i => :parents]
        end
    
    else # Update the state
       
        n = results["n"] + 1 
        # Lambda diffs
        if results["lambda"][end][2] != tr[:lambda]
            push!(results["lambda"], (n, tr[:lambda]))
        end 
        # Parent set diffs
        for i=1:V
            prev_ps = results["prev_parents"][i]
            cur_ps = tr[:parent_sets => i => :parents] 
            # Has a new parent been added?
            new_parents = setdiff(cur_ps, prev_ps)
            for new_parent in new_parents
                push!(results["parent_sets"][i][new_parent], (n, 1))
            end
         
            # Has a parent been removed?
            removed_parents = setdiff(prev_ps, cur_ps)
            for rem_parent in removed_parents
                push!(results["parent_sets"][i][rem_parent], (n, 0))
            end

            results["prev_parents"][i] = copy(cur_ps)
        end 
        # Finished. update n. 
        results["n"] = n
 
    end
    
    return results
end


function update_acc(acceptances, acc, V)

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
tr[:parent_sets => i => :parents] 
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


