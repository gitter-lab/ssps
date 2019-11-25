# dbn_inference.jl
# 2019-11-11
# David Merrell
#
# Contains functions relevant for inference
# in our network reconstruction setting.


"""
    (new_tr, accept) = mh_diff(trace, diff_proposal::GenerativeFunction,
                               proposal_args::Tuple, diff_involution::Function,
                               diff_update!::Function, diff_weight::Function; check_round_trip=false)

Imitates the four-argument Gen.mh function, but works in terms of *differences* between traces.
The point of this new version is to avoid copying the trace on each proposal.
Copying the trace is prohibitively expensive if it contains large objects (e.g., graphs that scale with the size of the data).
The four-argument version requires the involution to return a `new_trace` distinct from the current `trace`, implying that the trace needs to be copied.
So it isn't satisfactory.

The parameters of type (Generative)Function should have the following signatures:

    fwd_diff = diff_proposal(trace, proposal_args...)

where `fwd_diff` is an object (of whatever suitable type) encoding all of the information necessary to update a trace.

    (bwd_diff, bwd_choices) = diff_involution(trace, fwd_choices, fwd_diff, proposal_args)

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
function mh_diff(trace, diff_proposal::GenerativeFunction,
                 proposal_args::Tuple, diff_involution::Function,
                 diff_update!::Function, diff_weight::Function; check_round_trip=false)

    (fwd_choices, fwd_score, fwd_diff) = propose(diff_proposal, (trace, proposal_args...,))
    (bwd_diff, bwd_choices) = diff_involution(trace, fwd_choices, fwd_diff, proposal_args)

    weight = diff_weight(trace, fwd_diff)
    diff_update!(trace, fwd_diff)

    (bwd_score, bwd_ret) = assess(diff_proposal, (trace, proposal_args...), bwd_choices)
    
    if check_round_trip
    	# check that the involution works correctly
        (rt_diff, fwd_choices_rt) = diff_involution(trace, bwd_choices, bwd_ret, proposal_args)
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

"""
Loop through the parts of the model and perform
Metropolis-Hastings updates on them.
"""
function dbn_gibbs_loop(tr, lambda_r, V, t; update_lambda=true)

    lambda_acc = false
    if update_lambda
        tr, lambda_acc = Gen.mh(tr, lambda_proposal, (lambda_r,))
    end

    acc_vec = zeros(V)
    for i=1:V
	#println(get_args(tr)[end])
        tr, acc_vec[i] = Gen.mh(tr, parentvec_proposal, 
				(i, V, t[i]), 
                                parentvec_involution)
    end

    return tr, lambda_acc, acc_vec
end


"""
Loop through the edges of the graph.
Use the prior distribution to propose a flip. 
"""
function edge_gibbs_loop(tr)
    V = length(ref_adj)
    edge_accs = zeros(V,V)
    for child=1:V
        for parent=1:V
            tr, edge_accs[child, parent] = Gen.mh(tr, Gen.select(:adjacency => :edges => child => parent => :z))
        end
    end

    return tr, edge_accs
end


"""
Helper function for collecting edge counts during inference.

The entry
tr[:adjacency => :edges => i => j => :z] 
indicates existence of edge j --> i
(kind of backward from what you might expect.)

So this matrix should be read "row = child; column = parent"
"""
function increment_counts!(edge_counts, tr)
    for i=1:size(edge_counts)[1] # Child
        for j=1:size(edge_counts)[2] # Parent
            edge_counts[i,j] += tr[:adjacency => :edges => i => j => :z]
	end
    end
end


"""
Inference program for the DBN pathway reconstruction task. 
"""
function dbn_vertexwise_inference(reference_adj::Vector{Vector{Bool}},
				  X::Vector{Array{Float64,2}},
                                  regression_deg::Int64,
				  phi_ratio::Float64,
				  lambda_prior_param::Float64,
                                  n_samples::Int64,
                                  burnin::Int64, thinning::Int64,
				  lambda_r::Float64, 
				  median_degs::Vector{Float64},
				  update_lambda::Bool)

    # Some preprocessing for the data
    Xminus, Xplus  = combine_X(X)
    Xminus_stacked, Xplus = vectorize_X(Xminus, Xplus)

    # Condition the model on the data
    observations = Gen.choicemap()
    for (i, Xp) in enumerate(Xplus)
        observations[:Xplus => i => :Xp] = Xp
    end

    tr, _ = Gen.generate(dbn_model, (reference_adj, 
				     Xminus_stacked, 
                                     Xplus, 
				     lambda_prior_param,
				     regression_deg,
				     phi_ratio),
			 observations)
    
    # Some useful parameters
    V = length(Xplus) 
    t = 1.0 ./ log2.(V./median_degs)

    # The results we care about
    edge_counts = zeros((V,V))
    lambdas = zeros(n_samples)

    ps_accs = zeros(V)
    lambda_accs = 0

    for i=1:burnin
        tr, la, psa = dbn_gibbs_loop(tr, lambda_r, V, t; update_lambda=update_lambda) 
        lambda_accs += la
        ps_accs .+= psa
    end
    increment_counts!(edge_counts, tr)
    lambdas[1] = tr[:lambda]

    for j=1:n_samples-1
        for k=1:thinning
            tr, la, psa = dbn_gibbs_loop(tr, lambda_r, V, t; update_lambda=update_lambda)
	    lambda_accs += la
	    ps_accs .+= psa
	end
	increment_counts!(edge_counts, tr)
	lambdas[j+1] = tr[:lambda]
    end

    n_props = burnin + (n_samples-1)*thinning
    lambda_accs = lambda_accs / n_props
    ps_accs = ps_accs ./ n_props

    edge_probs = convert(Matrix{Float64}, edge_counts)./n_samples

    return edge_probs, lambdas, lambda_accs, ps_accs

end


"""
Inference program for the DBN pathway reconstruction task.

Uses annealing to address the challenge of low acceptance probabilities.

`phi_ratio_schedule` should be a function with the following signature:
    
    phi_ratio = phi_ratio_schedule(t::Float64)

where `t` \\in [0,1] represents the fraction of thinning steps completed thus far.
A good choice of schedule might be a sigmoid function.

For sake of comparison against Hill et al.'s method, we'll give lambda a fixed
value in this inference task.
"""
function dbn_edgeind_annealing_inference(reference_adj::Vector{Vector{Bool}},
				         X::Vector{Array{Float64,2}},
                                         regression_deg::Int64,
				         phi_ratio_schedule::Function,
				         fixed_lambda::Float64,
                                         n_samples::Int64,
                                         thinning::Int64,
				         median_degs::Vector{Float64})

    # Some preprocessing for the data
    Xminus, Xplus  = combine_X(X)
    Xminus_stacked, Xplus = vectorize_X(Xminus, Xplus)

    # Condition the model on the data
    observations = Gen.choicemap()
    for (i, Xp) in enumerate(Xplus)
        observations[:Xplus => i => :Xp] = Xp
    end
    observations[:lambda] = fixed_lambda
    tr, _ = Gen.generate(dbn_model, (reference_adj, 
				     Xminus_stacked, 
                                     Xplus, 
				     1.0,
				     regression_deg,
				     0.0),
			 observations)
    
    # Some useful parameters
    V = length(Xplus) 
    t = 1.0 ./ log2.(V./median_degs)

    # The results we care about
    edge_counts = zeros((V,V))
    ps_accs = zeros(V)

    for j=1:n_samples-1
        for k=1:thinning
        
	    pr = phi_ratio_schedule(1.0*k/thinning)
	    newargs = tuple(Gen.get_args(tr)[1:end-1]..., pr)
	    argdiffs = tuple([[Gen.NoChange() for i=1:length(newargs)-1]; [Gen.UnknownChange()]]...)
	    tr, _ = Gen.update(tr, newargs, argdiffs, Gen.choicemap())
            tr, la, psa = dbn_gibbs_loop(tr, lambda_r, V, t; update_lambda=false)
	    ps_accs .+= psa
	end
	increment_counts!(edge_counts, tr)
    end

    n_props = n_samples*thinning
    ps_accs = ps_accs ./ n_props

    edge_probs = convert(Matrix{Float64}, edge_counts)./n_samples

    return edge_probs,  ps_accs

end



function dbn_edgeind_gibbs_inference(reference_adj::Vector{Vector{Bool}},
				     X::Vector{Array{Float64,2}},
                                     regression_deg::Int64,
				     phi_ratio::Float64,
				     fixed_lambda::Float64,
                                     n_samples::Int64,
				     burnin::Int64,
                                     thinning::Int64)

    # Some preprocessing for the data
    Xminus, Xplus  = combine_X(X)
    Xminus_stacked, Xplus = vectorize_X(Xminus, Xplus)

    # Condition the model on the data
    observations = Gen.choicemap()
    for (i, Xp) in enumerate(Xplus)
        observations[:Xplus => i => :Xp] = Xp
    end
    observations[:lambda] = fixed_lambda
    tr, _ = Gen.generate(dbn_model, (reference_adj, 
				     Xminus_stacked, 
                                     Xplus, 
				     1.0,
				     regression_deg,
				     phi_ratio),
			 observations)
    
    # Some useful parameters
    V = length(Xplus)

    # The results we care about
    edge_counts = zeros((V,V))
    edge_accs = zeros((V,V))
  
    # burnin
    for i=1:burnin
        tr, accs = edge_gibbs_loop(tr)
        edge_accs .+= accs
    end
    increment_counts!(edge_counts, tr)

    # sampling (with thinning)
    for i=1:n_samples-1
        for t=1:thinning
            tr, accs = edge_gibbs_loop(tr)
            edge_accs .+= accs
	end
        increment_counts!(edge_counts, tr)
    end
   
    # A little bit of postprocessing 
    n_props = burnin + (n_samples-1)*thinning
    edge_accs = edge_accs ./ n_props
    posterior_edge_probs = convert(Matrix{Float64}, edge_counts)./n_samples

    return posterior_edge_probs,  edge_accs

end



"""
    compute_bic(ols_path, lasso_path, n::Int64, ols_df::Int64)

Compute a vector of Bayesian Information Criterion values for a
GLMNet path.
"""
function compute_bic(ols_path, lasso_path, n::Int64, ols_df::Int64) 
    
    ols_ssr = (1.0 - ols_path.dev_ratio[1]) # *null deviation
    ssr_vec = (1.0 .- lasso_path.dev_ratio) # *null deviation
    df_vec = sum(lasso_path.betas .!= 0.0, dims=1)[1,:]
    
    bic_vec = ((n - ols_df).*ssr_vec./ols_ssr  .+  log(n).*df_vec) ./n # null deviations cancel out
    
    return bic_vec
end


"""
    adalasso_edge_recovery(X::Matrix{Float64}, y::Vector{Float64}, prior_parents::Vector{Bool})
Use Adaptive LASSO (informed by prior knowledge of edge existence)
and a Bayesian Information Criterion to get a set of regression coefficients;
nonzero coefficients imply the existence of an edge in the MAP graph.
Note that we assume normally-distributed variables.
"""
function adalasso_edge_recovery(X::Matrix{Float64}, y::Vector{Float64}, prior_parents::Vector{Bool})
    
    n = size(y,1)
    yp = y .* n ./ (n+1)
    ols_result = GLMNet.glmnet(X, yp, lambda=[0.0])

    ada_weights = 1.0 .- prior_parents;
    adaptive_penalties = abs.(ada_weights ./ ols_result.betas)[:,1]
    
    lasso_path = GLMNet.glmnet(X, yp, penalty_factor=adaptive_penalties)
    
    bic_vec = compute_bic(ols_result, lasso_path, n, size(X,2))
    
    minloss_idx = argmin(bic_vec)
    
    return lasso_path.betas[:,minloss_idx]

end


"""
    adalasso_parents(Xminus::Matrix{Float64}, Xplus::Matrix{Float64},
                     reference_parents::Vector{

A Metropolis-Hastings inference strategy which uses adaptive LASSO
to initialize the markov chain at a reasonable place.
"""



