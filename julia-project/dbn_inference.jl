# dbn_inference.jl
# 2019-11-11
# David Merrell
#
# Contains functions relevant for inference
# in our network reconstruction setting.


"""
Generic MCMC inference wrapper.
Some important arguments:
"""
function mcmc_inference(gen_model, 
                        model_args,
                        observations,
                        update_loop_fn::Function,
                        update_loop_args::Vector,
                        update_results_fn::Function,
                        update_results_args::Vector;
                        timeout::Float64=3600.0,
                        n_steps::Int64=-1,
                        thinning::Int64=1)

    
    # start the timer
    t_start = time()
    t_elapsed = 0.0

    # Generate an initial trace
    tr, _ = Gen.generate(gen_model, model_args,
			 observations)
   
    # The results we care about
    results = nothing

    # Sampling loop
    prop_count = 0
    println("Sampling for ", timeout, " seconds. (Or ", n_steps," steps).")
    t_print = 0.0
    while t_elapsed < timeout && prop_count <= n_steps
        
        # thinning loop (if applicable)
        for i=1:thinning
            
            # Print progress
            if (prop_count > 0) && (t_elapsed - t_print >= 20.0)
                println("\t", prop_count," steps in ", round(t_elapsed), " seconds.")
                t_print = t_elapsed 
            end

            # update the variables    
            tr, acc = update_loop_fn(tr, update_loop_args)
            prop_count += 1
            
            t_elapsed = time() - t_start
        end

        # update the results
        results = update_results_fn(results, tr, update_results_args)
        
    end
 
    return results
end

