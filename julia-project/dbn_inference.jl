# dbn_inference.jl
# 2019-11-11
# David Merrell
#
# Contains functions relevant for inference
# in our network reconstruction setting.


"""
    (new_tr, accept) = mh_diff(trace, proposal::GenerativeFunction,
	                           proposal_args::Tuple, involution::Function,
                               diff_update!::Function, diff_weight::Function,
							   check_round_trip=false)

Imitates the four-argument Gen.mh function, but works in terms of *differences* between traces.
The point of this new version is to avoid copying the trace on each proposal.
Copying the trace is prohibitively expensive if it contains large objects (e.g., graphs that scale with the size of the data).
The four-argument version requires the involution to return a `new_trace` distinct from the current `trace`, implying that the trace needs to be copied.
So it isn't satisfactory.

The parameters of type (Generative)Function should have the following signatures:

    fwd_diff = proposal(trace, proposal_args...)

where `fwd_diff` is an object (of whatever suitable type) encoding all of the information necessary to update a trace.

    (bwd_diff, bwd_choices) = involution(trace, fwd_choices, fwd_diff, proposal_args)

`bwd_diff` should be of the same type as `fwd_diff`.
Notice that this involution does not return a weight.

    diff_update!(trace, trace_diff)

`diff_update!` modifies `trace` directly using `trace_diff`; it does *not* copy the trace.
Will generally contain a call to Gen.update.

    weight = diff_weight(trace, trace_diff)

Should return the change in log-probability resulting from `trace_diff`;
i.e., the quantity log(P(trace + trace_diff | inputs) / P(trace | inputs)).
IT IS THE USER'S RESPONSIBILITY TO MAKE SURE `diff_weight` IS IMPLEMENTED CORRECTLY.
"""
function mh_diff(trace, proposal::GenerativeFunction,
	             proposal_args::Tuple, involution::Function,
                 diff_update!::Function, diff_weight::Function;
				 check_round_trip=false)

    (fwd_choices, fwd_score, fwd_diff) = propose(proposal, (trace, proposal_args...,))
    (bwd_diff, bwd_choices) = involution(trace, fwd_choices, fwd_diff, proposal_args)

    weight = diff_weight(trace, fwd_diff)
    diff_update!(trace, fwd_diff)

	(bwd_score, bwd_ret) = assess(proposal, (trace, proposal_args...), bwd_choices)

	if check_round_trip
		# check that the involution works correctly
        (rt_diff, fwd_choices_rt) = involution(trace, bwd_choices, bwd_ret, proposal_args)
        if !isapprox(fwd_choices_rt, fwd_choices)
            println("fwd_choices:")
            println(fwd_choices)
            println("round trip fwd_choices:")
            println(fwd_choices_rt)
            error("Involution round trip check failed")
        end
        if !(fwd_diff == rt_diff)
			println("fwd_diff:")
            println(fwd_diff)
            println("round trip diff:")
            println(rt_diff)
            error("Involution round trip check failed")
        end
		weight_rt = diff_weight(trace, bwd_diff)
        if !isapprox(weight, -weight_rt)
            println("weight: $weight, -weight_rt: $(-weight_rt)")
            error("Involution round trip check failed")
        end
		rt_trace = copy(trace)
		diff_update!(rt_trace, bwd_diff)
		diff_update!(rt_trace, fwd_diff)
		if !isapprox(get_choices(trace), get_choices(rt_trace))
			println("trace choices:")
			println(get_choices(trace))
			println("round trip choices:")
			println(get_choices(rt_trace))
			error("Involution round trip check failed")
		end
    end
    if log(rand()) < weight + bwd_score - fwd_score
        # accept
        (trace, true)
    else
        # reject: undo the update to the trace.
		diff_update!(trace, bwd_diff)
        (trace, false)
    end
end


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
