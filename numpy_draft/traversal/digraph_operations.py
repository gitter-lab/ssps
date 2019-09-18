"""
digraph_operations.py
2019-09-17
David Merrell

Define a simple language of operations on directed graphs:
* edge reversal
* edge removal
* edge addition
"""

import numpy as np

REVERSE_KEY = "_v_"
REMOVE_KEY = "_r_"
ADD_KEY = "_a_"


def EdgeAddition(v_0, v_1):
    """
    create an object that represents an edge addition.
    """
    return ADD_KEY, v_0, v_1


def EdgeRemoval(v_0, v_1):
    """
    create an object that represents an edge removal.
    """
    return REMOVE_KEY, v_0, v_1


def EdgeReversal(v_0, v_1):
    """
    create an object that represents an edge reversal.
    """
    return REVERSE_KEY, v_0, v_1


def apply_operation(op, adj_mat, ancestors=None, check_input=False):
    """
    apply a graph operation (edge addition/removal/reversal)
    to the given adjacency matrix.
    """

    # Add edge
    if op[0] == ADD_KEY:

        if check_input:
            assert adj_mat[op[1], op[2]] == 0 and adj_mat[op[2], op[1]] != 0

        adj_mat[op[1], op[2]] = 1  # set entry to 1
        if ancestors is not None:
            ancestors[op[2]].update(ancestors[op[1]])  # add ancestors!

    # Remove edge
    elif op[0] == REMOVE_KEY:

        if check_input:
            assert adj_mat[op[1], op[2]] != 0

        adj_mat[op[1], op[2]] = 0  # set entry to 0
        if ancestors is not None:
            # Get the ancestors of v2's remaining parents
            parents = np.where(adj_mat[:, op[2]] != 0)
            ancestors[op[2]] = set.union(*[ancestors[i] for i in parents])

    # Reverse edge
    elif op[0] == REVERSE_KEY:

        if check_input:
            # check that there is, in fact, a directed edge here.
            assert adj_mat[op[1], op[2]] != 0 and adj_mat[op[2], op[1]] == 0

        adj_mat[op[2], op[1]] = adj_mat[op[1], op[2]]
        adj_mat[op[1], op[2]] = 0
        if ancestors is not None:

            # equivalent to removing an edge and adding an edge...
            ancestors[op[1]].update(ancestors[op[2]])
            parents = np.where(adj_mat[:, op[2]] != 0)
            ancestors[op[2]] = set.union(*[ancestors[i] for i in parents])

    if ancestors is None:
        return adj_mat
    else:
        return adj_mat, ancestors


