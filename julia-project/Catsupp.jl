__precompile__()

module Catsupp

using Gen
using GLMNet
using ArgParse
using JSON

export parse_script_arguments, perform_inference, make_output_dict 

include("dbn_preprocess.jl")
include("dbn_models.jl")
include("dbn_proposals.jl")
include("dbn_inference.jl")

Gen.load_generated_functions()

"""
make a json string containing all of the relevant
outputs of MCMC, structured in a sensible way.
"""
function make_output_dict(results, acc)

    if acc == nothing
        acc = Dict("lambdas" => nothing,
		   "n_proposals" => nothing,
		   "parent_sets" => nothing
		   )
    end

    results["parent_sets_acc"] = acc["parent_sets"]
    results["lambdas_acc"] = acc["lambdas"]

    return results
end


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
        "--burnin"
            help = "fraction of runtime to spend warming up the markov chain"
    	    arg_type = Float64
            default=0.5
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
    	    default = 10.0
        "--regression-deg"
	    help = "regression polynomial degree in Gaussian DBN (default: full)"
	    arg_type = Int64
    	    default = -1 
        "--lambda-prop-std"
            help = "standard deviation of lambda's proposal distribution"
    	    default = 0.25
	"--track-acceptance"
	    help = "flag: track proposal acceptance rates during MCMC"
	    action = :store_true
        "--store-samples"
            help = "Store and return the entire sequence without burnin. Used in convergence analyses."
	    action = :store_true
        "--n-steps"
            help = "Terminate the markov chain after it runs this many steps."
            arg_type = Int64
            default = -1
        "--continuous-reference"
            help = "Allow continuous-valued weights in [0,1] in the reference graph."
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
    push!(arg_vec, parsed_arg_dict["burnin"])
    push!(arg_vec, parsed_arg_dict["thinning"])
    push!(arg_vec, parsed_arg_dict["large-indeg"])
    push!(arg_vec, parsed_arg_dict["lambda-max"])
    push!(arg_vec, parsed_arg_dict["regression-deg"])
    push!(arg_vec, parsed_arg_dict["lambda-prop-std"])
    push!(arg_vec, parsed_arg_dict["track-acceptance"])
    push!(arg_vec, parsed_arg_dict["store-samples"])
    push!(arg_vec, parsed_arg_dict["continuous-reference"])

    return arg_vec
end


function perform_inference(timeseries_filename::String,
			   ref_graph_filename::String,
                           output_path::String,
                           timeout::Float64,
                           n_steps::Int64,
			   burnin::Float64, 
                           thinning::Int64,
			   large_indeg::Float64,
			   lambda_max::Float64,
			   regression_deg::Int64,
			   lambda_prop_std::Float64,
                           track_acceptance::Bool,
                           store_samples::Bool,
                           continuous_reference::Bool)

    clear_caches()

    ts_vec, ref_ps = load_formatted_data(timeseries_filename, 
					 ref_graph_filename;
                                         boolean_adj= !continuous_reference) 
  
    println("Invoking Catsupp on input files:\n\t", 
	    timeseries_filename, "\n\t", ref_graph_filename)

    # Decide the kind of results to store:
    # summary statistics -- or -- a record of all samples
    if store_samples
        update_results_fn = update_results_storediff
    else
        update_results_fn = update_results_summary
    end
    
    results, acc = dbn_mcmc_inference(ref_ps, ts_vec; 
				      regression_deg=regression_deg,
                                      timeout=timeout,
                                      n_steps=n_steps,
                                      store_samples=store_samples, 
                                      burnin=burnin, 
                                      thinning=thinning,
			              update_loop_fn=smart_update_loop,
			              update_results_fn=update_results_fn,
                                      large_indeg=large_indeg,
                                      lambda_max=lambda_max,
				      lambda_prop_std=lambda_prop_std,
			              track_acceptance=track_acceptance,
			              update_acc_fn=update_acc,
                                      bool_prior= !continuous_reference)

 
    println("Saving results to JSON file:")
    out_dict = make_output_dict(results, acc)

    f = open(output_path, "w")
    JSON.print(f, out_dict)
    close(f)
    println("\t", output_path) 

end

# main function -- for purposes of static compilation
Base.@ccallable function julia_main(args::Vector{String})::Cint
    arg_vec = parse_script_arguments()
    perform_inference(arg_vec...)
    return 0
end

# END MODULE
end
