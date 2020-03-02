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
function dbn_mcmc_inference(gen_model, 
                            model_args,
                            observations,
                            update_loop_fn::Function,
                            update_loop_args::Vector,
                            update_results_fn::Function,
                            update_results_args::Vector,
                            update_acc_fn::Function,
                            update_acc_args::Vector;
                            regression_deg::Int64=3,
                            timeout::Float64=3600.0,
                            n_steps::Int64=-1,
                            store_samples::Bool=false,
                            burnin::Float64=0.5, 
                            thinning::Int64=1,
			    track_acceptance::Bool=false)

    
    # start the timer
    t_start = time()
    t_elapsed = 0.0

    # Generate an initial trace
    tr, _ = Gen.generate(gen_model, model_args,
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

        tr, acc = update_loop_fn(tr, update_loop_args)     
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
            tr, acc = update_loop_fn(tr, update_loop_args)
            if track_acceptance
                acceptances = update_acc_fn(acceptances, acc, update_acc_args)
            end
            prop_count += 1
            
            t_elapsed = time() - t_start
        end

        # update the results
        results = update_results_fn(results, tr, update_results_args)
        
    end
 
    # Some last updates to the `results` object
    results["burnin_count"] = burnin_count
    if store_samples
        delete!(results, "prev_parents") 
    end

    return results, acceptances 
end

