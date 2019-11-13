# Main.jl
# 2019-11-12
# David Merrell
#

using Gen
include("PSDiGraph.jl")
using .PSDiGraphs

include("dbn_preprocessing.jl")
include("dbn_models.jl")
include("dbn_proposals.jl")
include("dbn_inference.jl")
include("dbn_visualization.jl")

proteins_path = "data/protein_names.csv"
prior_graph_path = "data/prior_graph.csv"
timesteps_path = "data/time.csv"
timeseries_path = "data/mukherjee_data.csv"

ts_vec, protein_vec, reference_dg, timesteps = hill_2012_preprocess(timeseries_path,
                                                                    proteins_path,
                                                                    prior_graph_path,
                                                                    timesteps_path)

println("TS_VEC: ", size(ts_vec))
println("PROTEINS: ", protein_vec)


@gen function dumb_model()
    G = @trace(graphprior(3.0, reference_dg), :G)
    return G
end

function dumb_func1()

    tr, _ = Gen.generate(dumb_model, ())
    accepted = 0
    for i=1:100
        tr, was_accepted = Gen.metropolis_hastings(tr, digraph_e_proposal, (protein_vec,), digraph_e_involution, check_round_trip=true)
        accepted += was_accepted
    end
    println(accepted, "/100 proposals accepted")
end

for _=1:10
    dumb_func1()
end
