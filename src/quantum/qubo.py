"""QUBO utilities for translating graph problems (e.g., MaxCut) into QUBO form."""

import numpy as np


def maxcut_to_qubo(G):
    """Convert a graph G to a QUBO matrix for MaxCut.

    The standard formulation places variable x_i in {0,1} and maximizes sum_{i<j} w_{ij} (x_i + x_j - 2 x_i x_j)
    which can be simplified to a QUBO matrix Q such that x^T Q x is (up to constant) the negative of cut weight.

    Returns:
        Q : 2D numpy array, shape (n,n)
        node_list : list mapping indices to nodes
    """
    node_list = list(G.nodes())
    n = len(node_list)
    Q = np.zeros((n, n))
    idx = {v: i for i, v in enumerate(node_list)}
    for u, v, data in G.edges(data=True):
        i = idx[u]
        j = idx[v]
        w = data.get("weight", 1.0)
        # For MaxCut we can set Q[i,i] += w and Q[j,j] += w and Q[i,j] -= 2*w
        Q[i, i] += w
        Q[j, j] += w
        Q[i, j] -= 2 * w
        Q[j, i] -= 2 * w

    return Q, node_list


def qubo_energy(x, Q):
    """Compute the QUBO energy for binary vector x and matrix Q."""
    x = np.asarray(x)
    return float(x @ Q @ x)
