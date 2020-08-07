# state_updates.jl
# David Merrell
# 
# A collection of functions for updating variables in a trace,
# or storing updates from MCMC.

import Base: lock, ReentrantLock
import Base.Threads: @threads


function old_update_loop(tr, proposal_param_vec::Vector{Float64})

    lambda_prop_std = proposal_param_vec[end]

    V = length(proposal_param_vec) - 1
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


"""
Loop through the model variables and perform
Metropolis-Hastings updates on them.
It uses the parent set proposal distribution.
"""
function vertex_lambda_update_loop(tr, proposal_param_vec::Vector{Float64})
    
    lambda_prop_std = proposal_param_vec[end]
    
    V = length(proposal_param_vec) - 1

    acc_vec = zeros(V)
    lambda_acc = false
    for i=1:V
        tr, lambda_acc = Gen.mh(tr, lambda_vec_proposal, (i, lambda_prop_std))
        tr, acc_vec[i] = Gen.mh(tr, smart_proposal,
				(i, proposal_param_vec[i], V),
				smart_involution)
    end

    return tr, (lambda_acc, acc_vec)
end


"""
Equivalent to `vertex_lambda_update_loop`, except that it uses
multithread parallelism to update the parent sets.
"""
function multithread_update_loop(tr, proposal_param_vec::Vector{Float64})

    lambda_prop_std = proposal_param_vec[end]
    cmap = Gen.choicemap()
    r = ReentrantLock()

    V = length(proposal_param_vec) - 1

    acc_vec = zeros(Bool, V)
    lambda_acc = zeros(Bool, V)
    # Parallelize the loop over vertices
    @threads for i=1:V
        
        # Update the inverse temperature for this vertex
        new_tr, lambda_acc[i] = Gen.mh(tr, lambda_vec_proposal, (i, lambda_prop_std))
        if lambda_acc[i]
            lock(r) do
                cmap[:lambda_vec => i => :lambda] = new_tr[:lambda_vec => i => :lambda]
            end
        end
        
        # Update the parent set for this vertex
        new_tr, acc_vec[i] = Gen.mh(new_tr, smart_proposal,
				(i, proposal_param_vec[i], V),
				smart_involution)
        if acc_vec[i]
            # Update the choice map in a thread-safe fashion
            lock(r) do
                cmap[:parent_sets => i => :parents] = new_tr[:parent_sets => i => :parents]
            end
        end
    end

    tr, _, _, _ = Gen.update(tr, Gen.get_args(tr), (), cmap)
    return tr, (true, acc_vec)
end


function uniform_update_loop(tr, proposal_param_vec::Vector{Float64})
    
    lambda_prop_std = proposal_param_vec[end]
    
    V = length(proposal_param_vec) - 1

    acc_vec = zeros(V)
    lambda_acc = false
    for i=1:V
        tr, lambda_acc = Gen.mh(tr, lambda_vec_proposal, (i, lambda_prop_std))
        tr, acc_vec[i] = Gen.mh(tr, uniform_proposal, (V,), uniform_involution)
    end

    return tr, (lambda_acc, acc_vec)
end


import DataStructures: DefaultDict


"""
Aggregate summary statistics for the quantities of interest
while the markov chain runs, in an online fashion.
"""
function update_results_summary(results, tr, args)

    V = args[1]
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
function update_results_storediff(results, tr, args)

    V = args[1]
    vector_lambda = Bool(args[2])
    # Initialize the state
    if results == nothing
        results = Dict("parent_sets" => [DefaultDict{Int64,Vector{Tuple{Int64,Int64}}}([]) for i=1:V],
                       "n" => 0,
                       "prev_parents" => [Vector{Int64}() for i=1:V]
                       )
        if vector_lambda
            results["lambda"] = [Vector{Tuple{Int64,Float64}}() for i=1:V]
            for i=1:V
                push!(results["lambda"][i], (0, tr[:lambda_vec => i => :lambda]))
            end
        else
            results["lambda"] = Vector{Tuple{Int64,Float64}}()
            push!(results["lambda"], (0, tr[:lambda]))
        end

        for i=1:V
            for parent in tr[:parent_sets => i => :parents]
                push!(results["parent_sets"][i][parent], (0,1))
            end
            results["prev_parents"][i] = tr[:parent_sets => i => :parents]
        end
    
    else # Update the state
       
        n = results["n"] + 1 
        # Lambda diffs
        if vector_lambda
            for i=1:V
                if results["lambda"][i][end][2] != tr[:lambda_vec => i => :lambda]
                    push!(results["lambda"][i], (n, tr[:lambda_vec => i => :lambda]))
                end
            end
        else
            if results["lambda"][end][2] != tr[:lambda]
                push!(results["lambda"], (n, tr[:lambda]))
            end
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


