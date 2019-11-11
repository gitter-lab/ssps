include("DiGraph.jl")
using Gen
using Plots
Plots.pyplot()
using .DiGraphs
using LinearAlgebra
using CSV
using DataFrames
using LRUCache
using Combinatorics
using Statistics


reference_dg = build_reference_graph(collect(1:length(protein_vec)), adj_mat);

ts_vec = timeseries_preprocess!(timeseries_data);



Gen.load_generated_functions()

tr = Gen.simulate(full_model, (ts_vec, reference_dg, 1));



empty!(lp_cache)
empty!(B2invconj_cache)
empty!(B_col_cache)
observations = Gen.choicemap()
observations[:X] = ts_vec
observations[:lambda] = 3.0

res, accepted = inference_program(full_model, (ts_vec, reference_dg, 1), observations,
                        digraph_proposal, (3.0,), 
                        digraph_involution, 
                        100, 1000, 1000);

sum(accepted) / length(accepted)


p = visualize_digraph(reference_dg, protein_vec)
Plots.title!(p, "Reference Graph")



   




#@gen function prior_model(lambda::Float64)
#    @trace(graphprior(lambda, reference_dg), :G)
#    return
#end

empty!(lp_cache)
empty!(B2invconj_cache)
empty!(B_col_cache)

lambda = 3.0
n_steps = 1000

observations = Gen.choicemap()
observations[:X] = ts_vec
observations[:lambda] = 3.0

med_indegs = [2.0]
regression_deg_max = 1
rejection_rates = []
thinnings = [2000]
n_samples = 100

# for med_indeg in med_indegs
#     rr = animate_dbn_exploration(reference_dg, protein_vec, full_model, (ts_vec, reference_dg, regression_deg_max), 
#                                  digraph_proposal, (med_indeg,), 
#                                  digraph_involution, n_steps,
#                                  "Graph Posterior Exploration: __STEPS__ steps (__REJECTIONS__ rejected)",
#                                  "posterior-explore-indeg$(med_indeg)-steps$(n_steps)-lambda$(lambda).gif");
#     push!(rejection_rates, rr)
# end

# for thinning in thinnings
#     animate_dbn_sampling(reference_dg, protein_vec, full_model, (ts_vec, reference_dg, regression_deg_max), 
#                          digraph_proposal, (2.0,), 
#                          digraph_involution, n_samples, thinning,
#                          "DBN Posterior Sampling: __SAMPLES__ samples",
#                          "posterior-sample-samples$(n_samples)-thinning$(thinning)-lambda$(lambda).gif"
#                         );
# end

for thinning in thinnings
    animate_dbn_sampling_density(reference_dg, protein_vec, full_model, (ts_vec, reference_dg, regression_deg_max),
                                 observations,
                                 digraph_proposal, (10.0,), 
                                 digraph_involution, n_samples, thinning, 0.50,
                                 "Posterior Edge Marginals: __SAMPLES__ samples",
                                 "posterior-density-samples$(n_samples)-thinning$(thinning)-lambda$(lambda)-deg$(regression_deg_max).gif");
end

# function simple_sampling(dg, model, model_args, proposal, proposal_args,
#                          involution, n_samples, thinning)
    
#     tr, _ = Gen.generate(model, model_args)
#     for i=1:n_samples
#         for t=1:thinning
#             tr, _ = Gen.metropolis_hastings(tr, proposal, proposal_args, involution)
#         end
#     end        
# end    

# @time begin
#     simple_sampling(reference_dg, dumb_model, (3.0,), digraph_proposal, (2.0,), digraph_involution,
#                        1000, 100)
# end

