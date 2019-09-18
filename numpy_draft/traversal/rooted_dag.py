"""
rooted_dag.py
2019-09-17
David Merrell

The functions in this module are used for traversing
a space of rooted DAGs.
"""

import numpy as np
import numpy_draft.traversal.digraph_operations as diop
import numpy_draft.graphs as gr


def rooted_dag_neighbors(rooted_dag, ancestors=None):
    """
    Assuming we're given a rooted DAG, return a list of
    the graph operations (edge removal, edge addition, edge reversal)
    that can be performed, which maintain rooted-dag-ness.

    :param rooted_dag: an adjacency matrix
    :param ancestors: a list of sets; the set of ancestors for each vertex.
    :return:
    """

    if ancestors is None:
        ancestors = gr.get_all_ancestors(rooted_dag)

    V = rooted_dag.shape[0]
    additions = [diop.EdgeAddition(i, j) for i in range(V) for j in range(V) if j not in ancestors[i]]

    return


