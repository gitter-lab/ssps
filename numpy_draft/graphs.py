"""
graphs.py
2019-09-13
David Merrell

Methods for working with a simple 
adjacency matrix graph representation.
"""

import numpy as np


def get_ancestors(v_ind, adj_mat):
    """
    Return a set of all ancestors for a given vertex.
    If a cycle is detected, we raise a CycleDetectedException.

    :param node_ind: (int) the vertex of interest.
    :param adj_mat: (NxN array) the adjacency matrix.
                    (interpreted as a directed graph)
    :return: set of ancestor vertices
    """

    return set(dfs(np.transpose(adj_mat), v_ind,
                   directed=True, fail_on_cycle=True))


def get_all_ancestors(adj_mat):
    """
    Return a list containing each vertex's set of ancestors.
    :param adj_mat:
    :return: list of sets
    """

    # not the most efficient implementation, but it works :shrug:
    return [get_ancestors(v_ind, adj_mat) for v_ind in range(adj_mat.shape[0])]


def dfs(adj_mat, vertex=0, directed=False, fail_on_cycle=False):
    """
    Performs DFS traversal of the graph, starting at
    the specified vertex (default 0).

    :param adj_mat:
    :param vertex:
    :param directed:
    :param fail_on_cycle:
    :return: list of vertices in order of traversal
    """
    visited = []
    stack = [vertex]

    while len(stack) > 0:

        v = stack.pop()
        if v not in visited:
            visited.append(v)
        else:
            if fail_on_cycle:
                raise CycleDetectedException
            continue

        if directed:
            succ = _d_successors(v, adj_mat)
        else:
            succ = _u_successors(v, adj_mat)

        stack += succ

    return visited


def _d_successors(vertex, adj_mat):
    """
    successor function for dfs on directed graph
    :param vertex:
    :param adj_mat:
    :return:
    """
    succ = [i for i, e in enumerate(adj_mat[vertex, :]) if e != 0]
    return succ


def _u_successors(vertex, adj_mat):
    """
    successor function for dfs on undirected graph
    :param vertex:
    :param adj_mat:
    :return:
    """
    succ = [i for i, e in enumerate(adj_mat[vertex, :]) if e != 0]
    succ2 = [i for i, e in enumerate(adj_mat[:, vertex]) if e != 0]
    succ = sorted(list(set(succ + succ2)))

    return succ


def is_dag(adj_mat):
    """
    Checks whether the given directed graph is acyclic.

    :param adj_mat: Adjacency matrix
    :return_components: (bool) whether to return a
    :return: (bool) whether the graph is a DAG or not.
    """

    tmp_mat = adj_mat
    while tmp_mat.shape[0] > 1:
        try:
            visited = dfs(tmp_mat, 0,
                          directed=True,
                          fail_on_cycle=True)
        except CycleDetectedException:
            return False

        to_visit = [i for i in range(tmp_mat.shape[0]) if i not in visited]
        tmp_mat = tmp_mat[to_visit, to_visit]

    return True


class CycleDetectedException(BaseException):
    pass
