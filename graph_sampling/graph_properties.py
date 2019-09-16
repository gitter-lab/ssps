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








