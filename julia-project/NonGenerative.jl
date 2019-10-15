# NonGenerative.jl
# 2019-10-14
# David Merrell
#
# Some machinery for dealing with random variables
# that don't have an explicit generative method.
# For example: we may only have access to a variable's
# unnormalized density. 
#
# We develop a framework for updating the values of values
# of variables in a Metropolis-Hastings fashion, when 
# those variables cannot be explicitly resampled.



module NonGenerative

using Gen

export NonGenProposal, nongen_mh


"""
    NonGenProposal{T}()

An abstract type representing a proposal distribution.

Proposal distributions should be implemented as singleton 
subtypes of NonGenProposal.

The following functions need to be implemented for 
NonGenProposal subtypes:

    (choices, logprob) = random(proposal::NonGenProposal,
                                address::Symbol,
                                current_trace,
                                proposal_args...)

    Returns a choicemap containing the proposed values,
    and the log-probability of that proposal: 
    log P(choices | current_trace)

    logprob = logpdf(proposal::NonGenProposal,
                     address::Symbol,
                     new_trace, cur_trace
		     proposal_args...)
                     
    Returns the log-probability of new_trace, given 
    current_trace:
    log P(new_trace | current_trace)
"""
abstract type NonGenProposal{T} end


function nongen_mh(cur_tr, proposal::NonGenProposal{T}, 
                           proposal_args::Tuple) where T

    proposal_args = (cur_tr, proposal_args...,)

    (prop_choices, prop_lp) = proposal(proposal_args)

    (new_tr, logodds, _, _) = update(cur_tr, get_args(cur_tr), 
                                            (), prop_choices) 
  
    bwd_args = (new_tr, cur_tr, proposal_args...,)
    bwd_lp = logpdf(proposal, bwd_args...)

    acceptance_lp = min(0.0, logodds + bwd_lp - prop_lp)

    if log(rand()) < acceptance_lp
        return new_tr, true
    else
        return cur_tr, false
    end

end



end # module


