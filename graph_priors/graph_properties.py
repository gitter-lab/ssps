"""
graph_properties.py
2019-09-11
David Merrell

This contains some useful functions for
checking the properties of graph objects.

We assume the graph is represented as an
adjacency matrix
"""


def is_connected(adj_mat):
    """
    Checks whether the graph is connected.

    *In this function, the graph is interpreted as
    an undirected graph.*

    :param adj_mat:
    :return:
    """

    visited = dfs(adj_mat, vertex=0, directed=False)
    return len(visited) == adj_mat.shape[0]


def all_reachable(adj_mat, root):
    """
    Checks whether all vertices are reachable from
    a given root vertex.

    In this function, the graph is interpreted as
    a *directed* graph.

    :param adj_mat:
    :param root:
    :return:
    """

    visited = dfs(adj_mat, vertex=root, directed=True)
    return len(visited) == adj_mat.shape[0]


def is_tree(adj_mat):
    """
    Checks whether the given graph is a tree.

    This function interprets the graph as an
    *undirected* graph.

    :param adj_mat:
    :return:
    """

    vertex_count = adj_mat.shape[0]
    edge_count = (adj_mat != 0).sum()

    # Do we have N-1 edges?
    if edge_count == vertex_count - 1:
        # is the graph connected?
        return is_connected(adj_mat)

    return False


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


def dfs(adj_mat, vertex=0, directed=False, fail_on_cycle=False):
    """
    Performs DFS traversal of the graph, starting at
    the specified vertex (default 0).

    :param adj_mat:
    :param start_vertex:
    :param directed:
    :param visited:
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

    succ = [i for i, e in enumerate(adj_mat[vertex, :]) if e != 0]
    return succ


def _u_successors(vertex, adj_mat):
    succ = [i for i, e in enumerate(adj_mat[vertex, :]) if e != 0]
    succ2 = [i for i, e in enumerate(adj_mat[:, vertex]) if e != 0]
    succ = sorted(list(set(succ + succ2)))

    return succ


class CycleDetectedException(BaseException):
    pass