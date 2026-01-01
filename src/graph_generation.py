"""Graph generation utilities for synthetic graphs used in experiments.

Provides Erdos-Renyi and Barabasi-Albert generation and simple benchmark loader.
"""

import networkx as nx
import random


def generate_er_graph(n, p, seed=None):
    """Generate an Erdős–Rényi graph with n nodes and edge probability p."""
    return nx.erdos_renyi_graph(n, p, seed=seed)


def generate_ba_graph(n, m, seed=None):
    """Generate a Barabási–Albert graph with n nodes and m edges to attach.

    Parameters
    ----------
    n : int
        Number of nodes.
    m : int
        Number of edges to attach from a new node to existing nodes.
    """
    return nx.barabasi_albert_graph(n, m, seed=seed)


def small_benchmark_graph(name="triangle"):
    """Return a named small benchmark graph for quick experiments."""
    if name == "triangle":
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        return G
    raise ValueError("Unknown benchmark graph")
