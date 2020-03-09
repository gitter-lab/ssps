
module Catsupp

using Gen
using GLMNet
using ArgParse
using JSON

export parse_script_arguments, perform_inference, make_output_dict, julia_main 

include("dbn_preprocess.jl")
include("dbn_models.jl")
include("dbn_proposals.jl")
include("dbn_inference.jl")
include("state_updates.jl")

Gen.load_generated_functions()

"""
parse the script's arguments using ArgParse.
"""
function parse_script_arguments()
    
    s = ArgParseSettings()
    @add_arg_table s begin
        "timeseries_filename"
            help = "name of timeseries data file"
    	    required = true
        "ref_graph_filename"
            help = "name of reference graph data file"
    	    required = true
        "output_path"
            help = "name of output JSON file"
    	    required = true
        "timeout"
            help = "execution timeout (in seconds). When reached, terminate and output results as they are."
            arg_type = Float64
        "--thinning"
            help = "number of proposals to make for each sample taken"
    	    arg_type = Int64
            default=1
        "--large-indeg"
            help = "an approximate upper bound on the number of parents for any node"
            arg_type = Float64
    	    default = 20.0
        "--lambda-max"
            help = "hyperparameter for lambda prior"
    	    default = 15.0
        "--regression-deg"
	    help = "regression polynomial degree in Gaussian DBN (default: full)"
	    arg_type = Int64
    	    default = -1 
        "--lambda-prop-std"
            help = "standard deviation of lambda's proposal distribution"
	    arg_type = Float64
    	    default = 0.25
        "--n-steps"
            help = "Terminate the markov chain after it runs this many steps."
            arg_type = Int64
            default = -1
        "--continuous-reference"
            help = "Allow continuous-valued weights in [0,1] in the reference graph."
            action = :store_true
        "--vertex-lambda"
            help = "Model an inverse temperature variable *for each vertex*, rather than having one for the entire prior graph."
            action = :store_true
    end

    args = parse_args(s)
    arg_vec = transform_arguments(args)
   
    return arg_vec
end


function transform_arguments(parsed_arg_dict)

    arg_vec = []
    push!(arg_vec, parsed_arg_dict["timeseries_filename"])
    push!(arg_vec, parsed_arg_dict["ref_graph_filename"])
    push!(arg_vec, parsed_arg_dict["output_path"])
    push!(arg_vec, parsed_arg_dict["timeout"])
    push!(arg_vec, parsed_arg_dict["n-steps"])
    push!(arg_vec, parsed_arg_dict["thinning"])
    push!(arg_vec, parsed_arg_dict["large-indeg"])
    push!(arg_vec, parsed_arg_dict["lambda-max"])
    push!(arg_vec, parsed_arg_dict["regression-deg"])
    push!(arg_vec, parsed_arg_dict["lambda-prop-std"])
    push!(arg_vec, parsed_arg_dict["continuous-reference"])
    push!(arg_vec, parsed_arg_dict["vertex-lambda"])

    return arg_vec
end


function perform_inference(timeseries_filename::String,
			   ref_graph_filename::String,
                           output_path::String,
                           timeout::Float64,
                           n_steps::Int64,
                           thinning::Int64,
			   large_indeg::Float64,
			   lambda_max::Float64,
			   regression_deg::Int64,
			   lambda_prop_std::Float64,
                           continuous_reference::Bool,
                           vertex_lambda::Bool)

    clear_caches()

    bool_prior = !continuous_reference

    ts_vec, ref_ps = load_formatted_data(timeseries_filename, 
					 ref_graph_filename;
                                         boolean_adj=bool_prior) 
    
    # Some data preprocessing
    Xminus, Xplus  = combine_X(ts_vec)
    Xplus = standardize_X(Xplus)[1]
    Xminus, Xplus = vectorize_X(Xminus, Xplus)
    V = length(Xplus)
  
    println("Invoking Catsupp on input files:\n\t", 
	    timeseries_filename, "\n\t", ref_graph_filename)

    # Decide the kind of results to store:
    # summary statistics -- or -- a record of all samples
    update_results_fn = update_results_storediff
    update_results_args = [V, vertex_lambda]
    
    # prepare parameters for proposal distributions
    if bool_prior
        ref_parent_counts = [max(length(ps),1) for ps in ref_ps]
    else
        ref_parent_counts = [max(sum(values(ps)), 2.0) for ps in ref_ps]
    end
    update_loop_args = 1.0 ./ log2.(V ./ ref_parent_counts)
    push!(update_loop_args, lambda_prop_std)

    lambda_min = log(max(V/large_indeg - 1.0, exp(0.5)))

    # Choose the right generative model, and prepare its arguments
    update_loop_fn = smart_update_loop
    if bool_prior 
        gen_model = dbn_model
    elseif !vertex_lambda
        gen_model = conf_dbn_model
    else
        gen_model = vertex_lambda_dbn_model
        update_loop_fn = vertex_lambda_update_loop 
    end
    model_args = (ref_ps,
                  Xminus, 
                  Xplus,
                  lambda_min, 
                  lambda_max,
                  regression_deg)

    
    # Check for some default arguments 
    if n_steps == -1
        n_steps = Inf
    end

    # Load observations into a choice map
    observations = Gen.choicemap()
    for (i, Xp) in enumerate(Xplus)
        observations[:Xplus => i => :Xp] = Xp
    end

    t_start = time()
    results = mcmc_inference(gen_model, model_args, observations,
                             update_loop_fn,
                             update_loop_args,
                             update_results_fn,
                             update_results_args;
                             timeout=timeout,
                             n_steps=n_steps,
                             thinning=thinning)
    
    # Some last updates to the `results` object
    results["t_elapsed"] = time() - t_start   
    delete!(results, "prev_parents") 

    println("Saving results to JSON file:")

    f = open(output_path, "w")
    JSON.print(f, results)
    close(f)
    println("\t", output_path) 

end


function julia_main()
    arg_vec = parse_script_arguments()
    perform_inference(arg_vec...)
    return 0
end


# END MODULE
end

using .Catsupp

julia_main()

