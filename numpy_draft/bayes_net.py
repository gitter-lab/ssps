"""
bayes_net.py
2019-09-16
David Merrell

Simple implementations of Bayes Net classes.
"""

import numpy as np
import numpy_draft.graphs as gr


class GaussianBayesNet:
    """
    Simple bayes net implementation that assumes
    all conditional probabilities are of the form

    P(x | x_p1, ..., x_pn) = N( sum(w_pi * x_pi) + w0, sigma^2)

    I.e., each variable is a linear regression of its parents.

    We also assume no missing data.
    """

    def __init__(self, adj_mat, check_dag=False):

        if check_dag:
            assert gr.is_dag(adj_mat)
        self._adj_mat = adj_mat
        self._cpts = [None]*adj_mat.shape[0]
        self._scores = [None]*adj_mat.shape[0]

    @property
    def adj_mat(self):
        return self._adj_mat

    @adj_mat.setter
    def adj_mat(self, new_adj_mat):
        self._adj_mat = new_adj_mat

    @property
    def cpts(self):
        return self._cpts

    def fit(self, X):
        """
        Fit all of the network's conditional probability distributions.
        :param X:
        :return:
        """

        for v in range(self._adj_mat.shape[0]):
            parent_inds = np.where(self._adj_mat[:, v] != 0)
            if parent_inds.shape[0] == 0:
                cpt, score = self._fit_conditional_prob(X[:, v], return_score=True)
            else:
                cpt, score = self._fit_conditional_prob(X[:, v], X[:, parent_inds], return_score=True)
            self._cpts[v] = cpt
            self._scores[v] = score

        return

    def _fit_conditional_prob(self, y, X=None, return_score=True):
        """
        Train one conditional probability distribution
        under a "linear regression" assumption:
        P(y | X) = N( X w + w0, sigma^2)

        :param y:
        :param X:
        :return:
        """

        y_var = np.var(y)
        if X is None:
            # If there are no independent variables, then we
            # simply represent y as a gaussian.
            mu = np.mean(y)
            if return_score:
                return (mu, y_var), -(mu - y)**2.0 / y_var
            else:
                return (mu, y_var)
        else:
            # Otherwise, fit a linear model to the data.
            (w, w0), res, _ = np.linalg.lstsq(X, y)[0]
            return w, w0, y_var

    def score(self, X):
        """
        Use the network to compute the log-probability of
        the given data.

        :param X:
        :return:
        """

        for v, cpt in enumerate(self._cpts):
            parents = np.where(self._adj_mat[:, v] != 0)
            if parents.shape[0] == 0:
                parent_data = None
            else:
                parent_data = X[:, parents]
            self._scores[v] = self._score_conditional_prob(cpt, X[:, v], parent_data)

        return np.sum(self._scores)

    def _score_conditional_prob(self, cpt, y, X=None):
        """
        Evaluate the log-probability of y given X, under the
        network's "conditional gaussian" distributional assumptions.

        :param cpt:
        :param y:
        :param X:
        :return:
        """
        if X is None:
            return - (y - cpt[0])**2.0 / cpt[1]

        else:
            return - (y - np.dot(X, cpt[0]) + cpt[1])**2.0 / cpt[2]


