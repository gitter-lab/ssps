# NonGenerative.jl
# 2019-10-14
# David Merrell
#
#


module NonGenerative

using Gen

export nongen_mh

function nongen_mh(cur_tr, proposal::Gen.Distribution{T}, 
                           proposal_args::Tuple) where T

    proposal_args = (cur_tr, proposal_args...,)

    (prop_choices, prop_lp) = proposal(proposal_args)

    (new_tr, logodds, _, _) = update(cur_tr, get_args(cur_tr), 
                                            (), prop_choices) 
  
    bwd_args = (new_tr, proposal_args...,)
    bwd_lp = logpdf(proposal, cur_tr, new_tr, proposal_args...)

    acceptance_lp = min(0.0, logodds + bwd_lp - prop_lp)

    if log(rand()) < acceptance_lp
        return new_tr, true
    end
    else
        return cur_tr, false
    end

end



end # module


