__precompile__()

module Catsupp

using Gen
using GLMNet
using ArgParse
using JSON

export dbn_mcmc_inference, load_simulated_data, update_results_z_lambda!,
       update_results_store_samples!, update_acc_z_lambda!, 
       adalasso_edge_recovery, clear_caches, ps_smart_swp_update_loop

include("dbn_preprocessing.jl")
include("dbn_models.jl")
include("dbn_proposals.jl")
include("dbn_inference.jl")

Gen.load_generated_functions()

"""
make a json string containing all of the relevant
outputs of MCMC, structured in a sensible way.
"""
function make_output_json(results, acc, elapsed)

    if acc == nothing
        acc = Dict("lambdas" => nothing,
		   "n_proposals" => nothing,
		   "parent_sets" => nothing
		   )
    end

    d = Dict("parent_sets"=> results["parent_sets"],
             "lambdas"=> results["lambdas"],
	     "parent_sets_acc" => acc["parent_sets"],
	     "lambdas_acc" => acc["lambdas"],
	     "n_proposals" => acc["n_proposals"],
             "time"=> elapsed
	     )

    return JSON.json(d)
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
        "output_json"
            help = "name of output JSON file"
    	    required = true
        "n_samples"
            help = "number of MCMC samples to take"
    	    arg_type = Int
    	    required = true
        "burnin"
            help = "number of burnin proposals to make"
    	    arg_type = Int
    	    required = true
        "thinning"
            help = "number of thinning proposals to make"
    	    arg_type = Int
    	    required = true
        "--fixed_lambda"
            help = "If we give lambda a fixed value, this is that value."
    	    arg_type = Float64
    	    default = -1.0
        "--lambda_prior"
            help = "kind of prior distribution for lambda variable"
    	    default = "uniform"
	    range_tester = x -> x in ["uniform"; "exponential"]
        "--lambda_param"
            help = "hyperparameter for lambda prior"
    	    default = 10.0
        "--regression_deg"
	    help = "regression polynomial degree in Gaussian DBN (default: full)"
	    arg_type = Int64
    	    default = -1 
        "--lambda_prop_std"
            help = "standard deviation of lambda's proposal distribution"
    	    default = 0.5
        "--parent_prop"
            help = "the kind of proposal distribution to use for parent sets"
    	    default = "smart"
	    range_tester = x -> x in ["smart", "dumb"]
	"--track_acceptance"
	    help = "flag: track proposal acceptance rates during MCMC"
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
    push!(arg_vec, parsed_arg_dict["output_json"])
    push!(arg_vec, parsed_arg_dict["n_samples"])
    push!(arg_vec, parsed_arg_dict["burnin"])
    push!(arg_vec, parsed_arg_dict["thinning"])
    push!(arg_vec, parsed_arg_dict["fixed_lambda"] == -1.0)
    push!(arg_vec, parsed_arg_dict["fixed_lambda"])
    push!(arg_vec, parsed_arg_dict["lambda_prior"])
    push!(arg_vec, parsed_arg_dict["lambda_param"])
    push!(arg_vec, parsed_arg_dict["regression_deg"])
    push!(arg_vec, parsed_arg_dict["lambda_prop_std"])
    push!(arg_vec, parsed_arg_dict["parent_prop"])
    push!(arg_vec, parsed_arg_dict["track_acceptance"])

    return arg_vec
end

function perform_inference(timeseries_filename::String,
			   ref_graph_filename::String,
			   output_json::String,
			   n_samples::Int64,
			   burnin::Int64, thinning::Int64,
			   update_lambda::Bool,
			   fixed_lambda::Float64,
			   lambda_prior::String,
			   lambda_param::Float64,
			   regression_deg::Int64,
			   lambda_prop_std::Float64,
			   parent_proposal::String,
			   track_acceptance::Bool)

    ts_vec, ref_ps = load_simulated_data(timeseries_filename, 
					 ref_graph_filename) 
   
    update_loop_fn = ps_smart_swp_update_loop 

    clear_caches()
    
    println("Invoking Catsupp on input files:\n\t", timeseries_filename, "\n\t", ref_graph_filename)
    start_time = time()
    results, acc = dbn_mcmc_inference(ref_ps, ts_vec, 
				      regression_deg, lambda_param,
                                      n_samples, burnin, thinning,
			              update_loop_fn,
			              update_results_store_samples!,
				      lambda_prop_std;
			              fixed_lambda=fixed_lambda,
			              update_lambda=update_lambda,
			              track_acceptance=track_acceptance,
			              update_acc_fn! =update_acc_z_lambda!)
    end_time = time()
    elapsed = end_time - start_time
    js_string = make_output_json(results, acc, elapsed)
    f = open(output_json, "w")
    write(f, js_string)
    close(f)
    println("Printed output to JSON file:\n\t", output_json) 

end




end

