
module DBNInference

using Gen

export base_mh_sampling

"""
    base_mh_sampling

A simple metropolis hastings sampling program.
Provide a model, model arguments, proposal, and involution;
a function which extracts quantities of interest from the trace;
the number of samples; and the burnin and thinning rates;
and this will return a vector of results.
"""
function base_mh_sampling(model, model_args::Tuple, 
			  choices, proposal, proposal_args::Tuple, 
			  involution, trace_function, 
			  n_samples::Integer, burnin::Integer, thinning::Integer)
    
    tr, _ = Gen.generate(model, model_args, choices)
    
    results = []
    prop_count = 1
    accepted = zeros(n_samples*thinning + burnin)
    for i=1:burnin
        tr, acc = Gen.mh(tr, proposal, proposal_args, involution, check_round_trip=true)
        accepted[prop_count] = acc
        prop_count += 1
    end
    push!(results, trace_function(tr))
    
    for i=1:n_samples-1
        for t=1:thinning
            tr, acc = Gen.mh(tr, proposal, proposal_args, involution, check_round_trip=true)
            accepted[prop_count] = acc
            prop_count += 1
        end
	push!(results, trace_function(tr))
    end
    
    return results, accepted
end


end
