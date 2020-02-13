
module PriorPredictor

using JSON
using DataFrames
using CSV
using ArgParse 

function load_args(args::Vector{String})
    
    s = ArgParseSettings()
    @add_arg_table s begin
        "prior_graph"
            help="path to a reference graph file"
            required = true
            arg_type = String
        "output_file"
            help="Name of output file containing edge probabilities"
            required = true
            arg_type = String
    end

    args = parse_args(args, s)
    return args
end


function make_output(argd::Dict)
    infile = argd["prior_graph"]
    outfile = argd["output_file"]

    ref_adj_df = CSV.read(infile, header=false)
    ref_ps = Vector{Vector{Int64}}()
    adj_mat = convert(Matrix{Bool}, ref_adj_df)

    for i=1:size(adj_mat,1)
        push!(ref_ps, adj_mat[:,i])
    end

    results = Dict()
    results["edge_conf_key"] = "parent_sets"
    results["parent_sets"] = ref_ps

    #res_json = JSON.json(results)
    f = open(outfile, "w")
    #write(f, res_json)
    JSON.print(f, results)
    close(f)
end


Base.@ccallable function julia_main(args::Vector{String})::Cint
    argd = load_args(args)
    make_output(argd)
    return 0
end

end
