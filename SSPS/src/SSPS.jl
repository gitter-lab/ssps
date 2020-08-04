module SSPS

using Gen
using GLMNet
using ArgParse
using JSON
using Base.Threads

export julia_main 

include("dbn_preprocess.jl")
include("dbn_models.jl")
include("dbn_proposals.jl")
include("mcmc_inference.jl")
include("state_updates.jl")

@load_generated_functions()


"""
Given the results of a previous run, recover the 
state of the markov chain and store it in the given
choicemap object.
"""
function recover_state!(cmap, samples)

    # Recover the lambda values
    for (j, svec) in enumerate(samples["lambda"])
        cmap[:lambda_vec => j => :lambda] = svec[end][2]
    end

    # Recover the parent sets
    for (j, pdict) in enumerate(samples["parent_sets"])
        pset = Int64[]
        for (k,v) in pdict
            if v[end][2] == 1
                append!(pset, parse(Int64, k))
            end
        end
        sort!(pset)
        cmap[:parent_sets => j => :parents] = pset 
    end 

end

"""
parse the script's arguments using ArgParse.
"""
function parse_script_arguments()
    
    s = ArgParseSettings()
    @add_arg_table! s begin
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
            help = "execution timeout (in seconds). When reached, terminate and output results as they are. SSPS attempts to divide "
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
            help = "Terminate a markov chain after it runs this many steps."
            arg_type = Int64
            default = -1
        "--n-chains"
            help = "The number of markov chains to run. Will run up to JULIA_NUM_THREADS in parallel."
            arg_type = Int64
            default = 1 
        "--proposal"
            help = "The graph proposal distribution to use in MCMC ('sparse' or 'uniform')"
            arg_type = String
            default = "sparse"
            range_tester = x -> x in ("sparse", "uniform")
        "--resume-sampling"
            help = "Continue sampling the markov chain stored in the given JSON file"
            required = false
            arg_type = String
            default = ""
    end

    args = parse_args(s)
    arg_vec = transform_arguments(args)
   
    return arg_vec
end


"""
Load arguments into a vector
"""
function transform_arguments(parsed_arg_dict)
    arg_vec = []
    push!(arg_vec, parsed_arg_dict["timeseries_filename"])
    push!(arg_vec, parsed_arg_dict["ref_graph_filename"])
    push!(arg_vec, parsed_arg_dict["output_path"])
    push!(arg_vec, parsed_arg_dict["timeout"])
    push!(arg_vec, parsed_arg_dict["n-steps"])
    push!(arg_vec, parsed_arg_dict["n-chains"])
    push!(arg_vec, parsed_arg_dict["thinning"])
    push!(arg_vec, parsed_arg_dict["large-indeg"])
    push!(arg_vec, parsed_arg_dict["lambda-max"])
    push!(arg_vec, parsed_arg_dict["regression-deg"])
    push!(arg_vec, parsed_arg_dict["lambda-prop-std"])
    push!(arg_vec, parsed_arg_dict["proposal"])
    push!(arg_vec, parsed_arg_dict["resume-sampling"])

    return arg_vec
end


"""
A wrapper around the MCMC inference procedure
"""
function perform_inference(timeseries_filename::String,
                           ref_graph_filename::String,
                           output_path::String,
                           timeout::Float64,
                           n_steps::Int64,
                           n_chains::Int64,
                           thinning::Int64,
                           large_indeg::Float64,
                           lambda_max::Float64,
                           regression_deg::Int64,
                           lambda_prop_std::Float64,
                           proposal::String,
			   resume_sampling::String)

    clear_caches()

    ts_vec, ref_ps = load_formatted_data(timeseries_filename, 
                                         ref_graph_filename)
    
    # Some data preprocessing
    Xminus, Xplus  = combine_X(ts_vec)
    Xplus = standardize_X(Xplus)[1]
    Xminus, Xplus = vectorize_X(Xminus, Xplus)
    V = length(Xplus)
  
    println("Invoking SSPS on input files:\n\t", 
            timeseries_filename, "\n\t", ref_graph_filename)

    # prepare parameters for proposal distributions
    ref_parent_counts = [max(sum(values(ps)), 2.0) for ps in ref_ps]
    update_loop_args = 1.0 ./ log2.(V ./ ref_parent_counts)
    push!(update_loop_args, lambda_prop_std)

    # Choose a lower bound for the lambda variable
    lambda_min = log(max(V/large_indeg - 1.0, exp(0.5)))

    # Choose the right update loop
    update_loop_fn = vertex_lambda_update_loop 
    if proposal == "uniform"
        update_loop_fn = uniform_update_loop
    end

    # Load observations into a choice map
    observations = Gen.choicemap()
    for (i, Xp) in enumerate(Xplus)
        observations[:Xplus => i => :Xp] = Xp
    end

    # If necessary, reload state from a previous session
    if resume_sampling != ""
        println("Resuming Markov Chain stored at ", resume_sampling)
        
        f = open(resume_sampling, "r")
        str = read(f, String)
        close(f)
        samples = JSON.parse(str)
       
        recover_state!(observations, samples)
    end

    # Prepare the Gen model's arguments
    model_args = (ref_ps,
                  Xminus,
                  lambda_min, 
                  lambda_max,
                  regression_deg)

    t_start = time()

    time_per_chain = timeout * min(1, nthreads() / n_chains)

    update_results_args = [V, true]
    
    results = mcmc_inference(vertex_lambda_dbn_model, 
                             model_args, observations,
                             update_loop_fn,
                             update_loop_args,
                             update_results_storediff,
                             update_results_args;
                             timeout=time_per_chain,
                             n_steps=n_steps,
                             thinning=thinning)
    # end

    # Some last updates to the `results` object
    results["t_elapsed"] = time() - t_start   
    delete!(results, "prev_parents") 

    println("Saving results to JSON file:")

    f = open(output_path, "w")
    JSON.print(f, results)
    close(f)
    println("\t", output_path) 

end


function julia_main()::Cint

    arg_vec = parse_script_arguments()
    perform_inference(arg_vec...)
    return 0
end


# END MODULE
end

