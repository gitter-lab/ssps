"""
kernels.py
2019-09-13
David Merrell

TransitionKernel class implementations
for use with tensorflow-probability's MCMC framework.

These kernels walk through spaces of graphs
(undirected, directed, or partially directed)
in a fashion that preserves desired properties
(e.g. connectedness, acyclic-ness, reachability)

These will typically be implemented without calibration.
We assume that they'll be composed with some wrapper
(e.g., MetropolisHastings) to produce a calibrated kernel.
"""

import tensorflow_probability as tfp
import tensorflow


#################################################
# CLASS DEFINITIONS
#################################################
class EfficientUpdateKernel(tfp.mcmc.UncalibratedRandomWalk):
    """
    Sometimes the log-probability is expensive to evaluate,
    but *changes* in log-probability are comparatively efficient.
    In these cases, we're better off evaluating the change directly
    and then using it to compute an acceptance probability.

    Concrete example: suppose we're sampling from a set of graphs
    and our random walk consists of add_edge and remove_edge operations.
    Furthermore: suppose that the log-probability decomposes into a
    sum over edges. In this case, modifying a single edge
    will only affect one term of the sum.
    A smart 'log_prob_update_fn' would exploit this locality.
    """

    def __init__(self,
                 log_prob_update_fn,
                 new_state_fn,
                 initial_score=0.0,
                 seed=None,
                 name=None):

        super(EfficientUpdateKernel, self).__init__(None,
                                                    new_state_fn=new_state_fn,
                                                    seed=seed,
                                                    name=name)

        self._log_prob_update_fn = log_prob_update_fn
        self._parameters['log_prob_update_fn'] = log_prob_update_fn

    @property
    def is_calibrated(self):
        return True

    def one_step(self, current_state, previous_kernel_results):
        return

    def bootstrap_results(self, init_state):
        return

class DigraphKernel(tfp.mcmc.TransitionKernel):

    def __init__(self,
                 target_log_prob_fn,
                 digraph_constraints,
                 seed=None,
                 name=None):

        new_state_fn = define_digraph_new_state_fn(digraph_constraints)

        super(DigraphKernel, self).__init__(target_log_prob_fn,
                                            new_state_fn=new_state_fn,
                                            seed=seed,
                                            name=name)


##################################################
# HELPER FUNCTIONS
##################################################
def define_graph_new_state_fn(graph_constraints):

    def _new_state_fun(current_state_parts, seed):
        return

    return _new_state_fun


def define_digraph_new_state_fn(digraph_constraints):

    def _new_state_fun(current_state_parts, seed):
        return

    return _new_state_fun

