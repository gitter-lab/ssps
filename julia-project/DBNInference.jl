
function inference_program(model, model_args, choices, proposal, proposal_args, 
                           involution, n_samples, burnin, thinning)
    
    tr, _ = Gen.generate(model, model_args, choices)
    
    results = []
    prop_count = 1
    accepted = zeros(n_samples*thinning + burnin)
    for i=1:burnin
        tr, acc = Gen.mh(tr, proposal, proposal_args, involution, check_round_trip=true)
        accepted[prop_count] = acc
        prop_count += 1
    end
    push!(results, tr[:G])
    
    for i=1:n_samples-1
        for t=1:thinning
            tr, acc = Gen.mh(tr, proposal, proposal_args, involution, check_round_trip=true)
            accepted[prop_count] = acc
            prop_count += 1
        end
        push!(results, tr[:G])
    end
    
    return results, accepted
end
