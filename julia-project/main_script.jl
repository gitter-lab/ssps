# Main.jl
# 2019-11-12
# David Merrell
#
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

#@gen function dumb_model()
#    #G = @trace(graphprior(3.0, reference_dg), :G)
#    G = @trace(normal(0.0,1.0), :G)
#    return G
#end

#tr, w = Gen.generate(dumb_model, ())
#println(typeof(tr))

acc = 0;

for k=1:100
    #tr, was_accepted = Gen.metropolis_hastings(tr, Gen.select(:G,) ) #digraph_v_proposal, (protein_vec, 2.0), digraph_v_involution)
    #println(i, " ", typeof(tr), " ", typeof(was_accepted))
    was_accepted = true
    acc = acc + was_accepted
end

#tr, was_accepted = Gen.metropolis_hastings(tr, Gen.select(:G) ) #digraph_v_proposal, (protein_vec, 2.0), digraph_v_involution)
#println(1, " ", typeof(tr), " ", was_accepted)
#acc += was_accepted
#tr, was_accepted = Gen.metropolis_hastings(tr, Gen.select(:G,) ) #digraph_v_proposal, (protein_vec, 2.0), digraph_v_involution)
#println(2, " ", typeof(tr), " ", was_accepted)
#acc += was_accepted
#tr, was_accepted = Gen.metropolis_hastings(tr, Gen.select(:G,) ) #digraph_v_proposal, (protein_vec, 2.0), digraph_v_involution)
#println(3, " ", typeof(tr), " ", was_accepted)
#acc += was_accepted
#tr, was_accepted = Gen.metropolis_hastings(tr, Gen.select(:G,) ) #digraph_v_proposal, (protein_vec, 2.0), digraph_v_involution)
#println(4, " ", typeof(tr), " ", was_accepted)
#acc += was_accepted


println(acc, "/100 proposals accepted")
