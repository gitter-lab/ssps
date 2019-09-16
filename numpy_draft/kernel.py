"""
kernel.py
2019-09-13
David Merrell

quick draft of MCMC transition kernels
using numpy (rather than Tensorflow-probability)
"""

import numpy as np


class TransitionKernel:

    def __init__(self, next_step_fn,
                 log_prob_fn=None,
                 log_prob_change_fn=None,
                 log_acc_correction=(lambda xp, x: 0)):
        """
        A TransitionKernel instance encodes a transition system
        used by the MCMC to generate samples.

        next_step_fn: a function which receives the current state
                      and proposes a next state.

        log_prob_fn: a function which scores the log-probability
                     of a state. It has signature (current_state).

        log_prob_change_fn: a function which computes the *change*
                            in log-probability between two states,
                            given some representation of that change.
                            Its signature is (state_change).

        log_acc_correction: a function which receives two states xp, x
                             and returns log( P(x|xp) / P(xp|x) );
                             a correction factor in the Metropolis-Hastings
                             acceptance probability.

        At least one of {log_prob_fn, log_prob_change_fn} must be set.
        If both are set, log_prob_change_fn will override log_prob_fn.

        The next_step_fn's proposal is considered against a 
        Metropolis-Hastings acceptance probability. 
        """

        assert not (log_prob_fn is None and log_prob_change_fn is None)
        self._next_step_fn = next_step_fn
        self._log_prob_fn = log_prob_fn
        self._log_prob_change_fn = log_prob_change_fn
        self._log_acc_correction = log_acc_correction

    def one_step(self, cur_state):
        """
        Execute one step of Metropolis-Hastings:
        * propose a next state
        * compute an acceptance probability
        * if we accept, then move to the next state
        
        return the state
        """
        proposal = self._next_step_fn(cur_state)

        # Compute the acceptance log probability
        log_acceptance = self._log_acc_correction(proposal, cur_state)
        
        if self._log_prob_change_fn is not None:
            log_acceptance += self._log_prob_change_fn(proposal, cur_state)
        else:
            log_acceptance += self._log_prob_fn(proposal)
            log_acceptance -= self._log_prob_fn(cur_state)

        acceptance = np.exp(min(0.0, log_acceptance))
        r = np.random.uniform(0,1)

        if r <= acceptance:
            return proposal

        else:
            return cur_state



