__precompile__()

module Catsupp

using Gen
using GLMNet
using ArgParse
using JSON

export parse_script_arguments, perform_inference, make_output_json 

include("dbn_preprocess.jl")
include("dbn_models.jl")
include("dbn_proposals.jl")
include("dbn_inference.jl")

Gen.load_generated_functions()

"""
make a json string containing all of the relevant
outputs of MCMC, structured in a sensible way.
"""
function make_output_json(results, tup, acc)

    cur_results = results["splits"][tup]

    if acc == nothing
        acc = Dict("lambdas" => nothing,
		   "n_proposals" => nothing,
		   "parent_sets" => nothing
		   )
    end

    cur_results["parent_sets_acc"] = acc["parent_sets"]
    cur_results["lambdas_acc"] = acc["lambdas"]
    cur_results["n_proposals"] = acc["n_proposals"]

    return JSON.json(cur_results)
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
        "output_directory"
            help = "name of output directory. Output will be written to output_directory/{mcmc_param_list}/output_filename"
    	    required = true
        "output_filename"
            help = "name of output JSON file"
    	    required = true
        "--n-samples"
            help = "number of MCMC samples to take"
    	    arg_type = Int
    	    nargs='+'
        "--burnin"
            help = "number of burnin proposals to make"
    	    arg_type = Int
            nargs='+'
        "--thinning"
            help = "number of thinning proposals to make"
    	    arg_type = Int
    	    nargs='+'
        "--fixed-lambda"
            help = "If we give lambda a fixed value, this is that value."
    	    arg_type = Float64
    	    default = -1.0
        "--lambda-prior"
            help = "kind of prior distribution for lambda variable"
    	    default = "uniform"
	    range_tester = x -> x in ["uniform"; "exponential"]
        "--lambda-param"
            help = "hyperparameter for lambda prior"
    	    default = 10.0
        "--regression-deg"
	    help = "regression polynomial degree in Gaussian DBN (default: full)"
	    arg_type = Int64
    	    default = -1 
        "--lambda-prop-std"
            help = "standard deviation of lambda's proposal distribution"
    	    default = 0.5
        "--parent-prop"
            help = "the kind of proposal distribution to use for parent sets"
    	    default = "smart"
	    range_tester = x -> x in ["smart", "dumb"]
	"--track-acceptance"
	    help = "flag: track proposal acceptance rates during MCMC"
	    action = :store_true
        "--timeout"
            help = "execution timeout (in seconds). When reached, terminate and output results as they are."
            arg_type = Float64
            default = Inf
    end

    args = parse_args(s)
    arg_vec = transform_arguments(args)
   
    return arg_vec
end

function transform_arguments(parsed_arg_dict)

    arg_vec = []
    push!(arg_vec, parsed_arg_dict["timeseries_filename"])
    push!(arg_vec, parsed_arg_dict["ref_graph_filename"])
    push!(arg_vec, parsed_arg_dict["output_directory"])
    push!(arg_vec, parsed_arg_dict["output_filename"])
    push!(arg_vec, parsed_arg_dict["n-samples"])
    push!(arg_vec, parsed_arg_dict["burnin"])
    push!(arg_vec, parsed_arg_dict["thinning"])
    push!(arg_vec, parsed_arg_dict["fixed-lambda"] == -1.0)
    push!(arg_vec, parsed_arg_dict["fixed-lambda"])
    push!(arg_vec, parsed_arg_dict["lambda-prior"])
    push!(arg_vec, parsed_arg_dict["lambda-param"])
    push!(arg_vec, parsed_arg_dict["regression-deg"])
    push!(arg_vec, parsed_arg_dict["lambda-prop-std"])
    push!(arg_vec, parsed_arg_dict["parent-prop"])
    push!(arg_vec, parsed_arg_dict["timeout"])

    return arg_vec
end

function perform_inference(timeseries_filename::String,
			   ref_graph_filename::String,
			   output_dir::String,
                           fname::String,
			   n_samples::Vector{Int64},
			   burnin::Vector{Int64}, thinning::Vector{Int64},
			   lambda_prior::String,
			   lambda_param::Float64,
			   regression_deg::Int64,
			   lambda_prop_std::Float64,
			   parent_proposal::String,
                           timeout::Float64)

    ts_vec, ref_ps = load_simulated_data(timeseries_filename, 
					 ref_graph_filename) 
   

    clear_caches()
    println("Invoking Catsupp on input files:\n\t", 
	    timeseries_filename, "\n\t", ref_graph_filename)

    results, acc = dbn_mcmc_inference(ref_ps, ts_vec, 
				      regression_deg, lambda_param,
                                      n_samples, burnin, thinning,
			              smart_update_loop,
			              update_results_split,
				      lambda_prop_std;
			              track_acceptance=true,
			              update_acc_fn=update_acc_z_lambda,
                                      timeout=timeout)
  
    println("Saving results to JSON files:")
    for (n,b,t) in keys(results["splits"]) 
    
        js_str = make_output_json(results, (n,b,t), acc)
        config_str = join(["mcmc_d=", regression_deg, 
                           "_n=", n, 
                           "_b=", b,
                           "_th=", t])

        mcmc_dir = Base.Filesystem.joinpath(output_dir, config_str) 
        Base.Filesystem.mkpath(mcmc_dir)

        output_path = Base.Filesystem.joinpath(mcmc_dir, fname)        
        f = open(output_path, "w")
        write(f, js_str)
        close(f)
        println("\t", output_path) 
    end

end

# main function -- for purposes of static compilation
Base.@ccallable function julia_main(args::Vector{String})::Cint
    arg_vec = parse_script_arguments()
    perform_inference(arg_vec...)
    return 0
end

# END MODULE
end
