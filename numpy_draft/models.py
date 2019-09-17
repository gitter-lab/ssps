"""
models.py
2019-09-13
David Merrell

quick rough draft of distributions over graphs,
using numpy rather than Tensorflow-probability
"""

import numpy as np
import numpy_draft.bayes_net as bn


def log_edge_precision_prior(adj_mat, reference_adj=None, beta=1.0):
    """
    A simple distribution over graphs:

    P(G) \propto exp( - beta *| E(G) - E(G') | )

    This distribution is uniform over subgraphs of G',
    and penalizes edges that are *not* in G'.

    :param adj_mat: binary adjacency matrix for graph G
    :param reference_adj: binary adjacency matrix for reference graph G'
    :param beta: "inverse temperature", or "precision" parameter
    :return: (unnormalized) log prob of G
    """

    return -beta * np.logical_and(adj_mat, np.logical_not(reference_adj)).sum()


def log_edge_precision_prior_change(adj_new, adj_old, reference_adj=None, beta=1.0):

    return log_edge_precision_prior(adj_new, reference_adj, beta=beta)\
           - log_edge_precision_prior(adj_old, reference_adj, beta=beta)


def log_bayes_net_prob(bayes_net, X=None):
    """
    Return the log-probability of the data (X) given the bayes_net.

    :param bayes_net:
    :param X:
    :return:
    """

    return bayes_net.score(X)


def log_bayes_net_prob_change(bn_new, bn_old, X=None):
    """
    Return the *change* in log-probability that happens
    when a bayesian network's topology gets updated

    :param adj_new:
    :param adj_old:
    :param X:
    :return:
    """
    change = 0.0
    # get the columns where the adjacency matrix has changed
    diff_vertices = np.unique(np.where(bn_new.adj_mat != bn_old.adj_mat)[1])
    for v in diff_vertices:
        pass

    return change
