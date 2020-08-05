# mcmc_inference.jl
# 2019-11-11
# David Merrell
#
# Contains functions relevant for inference
# in our network reconstruction setting.


"""
Generic MCMC inference wrapper.
Arguments:
    * gen_model: a Gen probabilistic program
    * model_args: a tuple of arguments for the "gen_model"
    * update_loop_fn: a function which updates the variables in "gen_model"
    * update_loop_args: a tuple of arguments for "update_loop_fn"
    * update_results_fn: a function which updates the data structure holding MCMC results
    * update_results_args: arguments for "update_results_fn"
    * timeout: maximum execution time (in seconds) (default 1 hour)
    * n_steps: maximum number of iterations (default no limit)
    * thinning: only store results once every "thinning"th iteration
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
                        thinning::Int64=1,
                        chain_id::String="")


    # start the timer
    t_start = time()
    t_elapsed = 0.0

    # Generate an initial trace
    tr, _ = Gen.generate(gen_model, model_args,
			 observations)
  
    # The results we care about
    results = update_results_fn(nothing, tr, update_results_args)

    # Sampling loop
    print_str = string("Sampling for ",string(timeout), " seconds.")
    if n_steps > 0
        print_str = string(print_str, " (Or ", n_steps, " iterations.)")
    else
        n_steps = typemax(Int64)
        print_str = string(print_str, " (No iteration limit.)")
    end
    println(print_str)

    prop_count = 0
    t_print = 0.0
    while t_elapsed < timeout && prop_count <= n_steps
        
        # thinning loop (if applicable)
        for i=1:thinning
            
            # Print progress
            if (prop_count > 0) && (t_elapsed - t_print >= 20.0)
                println("[", chain_id, "]\t", prop_count," iterations;\t", round(t_elapsed), " seconds")
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

