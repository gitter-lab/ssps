"""
sampling.py
2019-09-13
David Merrell

quick implementation of a MCMC sampler,
using numpy rather than Tensorflow-probability
"""

import numpy as np


def sample_chain(initial_state, transition_kernel, n_samples,
                 burnin=100, spacing=1):
    """
    Perform MCMC sampling, where the markov chain is encoded
    within the transition_kernel.

    initial_state: an initial state for the markov chain.
    
    transition_kernel: a TransitionKernel object.

    n_samples: the number of samples to collect.
               (if this works, they'll be drawn approximately 
               iid from a target distribution)

    burnin: a number of (discarded) iterations used to reach
            the stationary distribution

    spacing: only keep one sample in every 'spacing' steps of the
             markov chain. Larger spacing gives less correlation between
             samples.
    """

    state = initial_state

    # "burn in" to the stationary distribution
    for b in range(burnin):
        state = transition_kernel.one_step(state)

    # Collect samples
    samples = [0]*n_samples
    for i in range(n_samples):
        
        samples[i] = state
        for j in range(spacing):
            state = transition_kernel.one_step(state)

    return samples


def simulated_annealing(initial_state, transition_kernel, 
                        cooling_schedule):
    """
    Performs simulated annealing
    """

    return



