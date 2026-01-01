"""Classical coarsening algorithms: heavy-edge matching, random matching, and AMG placeholder.

Each method returns a tuple (G_coarse, mapping) where `mapping` maps coarse_node -> list(original_nodes).
"""

import networkx as nx
import random
from collections import defaultdict


def _contract_pairs(G, pairs):
    """Contract node pairs specified in `pairs` and return contracted graph and mapping."""
    mapping = {}
    contracted = nx.Graph()

    # Map each original node to a representative coarse node
    node_to_coarse = {}
    for idx, group in enumerate(pairs):
        label = f"c{idx}"
        for v in group:
            node_to_coarse[v] = label
        mapping[label] = list(group)

    # Nodes that were not in any pair remain as singleton aggregates
    idx = len(pairs)
    for v in G.nodes():
        if v not in node_to_coarse:
            label = f"c{idx}"
            node_to_coarse[v] = label
            mapping[label] = [v]
            idx += 1

    # Build contracted graph edges with summed weights
    edge_weights = defaultdict(float)
    for u, v, data in G.edges(data=True):
        cu = node_to_coarse[u]
        cv = node_to_coarse[v]
        if cu == cv:
            continue
        w = data.get("weight", 1.0)
        if cu > cv:
            cu, cv = cv, cu
        edge_weights[(cu, cv)] += w

    contracted.add_nodes_from(mapping.keys())
    for (cu, cv), w in edge_weights.items():
        contracted.add_edge(cu, cv, weight=w)

    return contracted, mapping


def random_matching(G, seed=None):
    """Randomly pair neighbors to build aggregates."""
    nodes = list(G.nodes())
    if seed is not None:
        random.seed(seed)
    unmatched = set(nodes)
    pairs = []

    while unmatched:
        v = unmatched.pop()
        neighbors = [u for u in G.neighbors(v) if u in unmatched]
        if neighbors:
            u = random.choice(neighbors)
            unmatched.remove(u)
            pairs.append([v, u])
        else:
            pairs.append([v])

    coarse, mapping = _contract_pairs(G, pairs)
    return coarse, mapping


def heavy_edge_matching(G):
    """Greedily match nodes along the heaviest incident edges."""
    matched = set()
    pairs = []

    # Sort edges by weight descending
    edges = sorted(G.edges(data=True), key=lambda e: e[2].get("weight", 1.0), reverse=True)
    for u, v, data in edges:
        if u in matched or v in matched:
            continue
        matched.add(u)
        matched.add(v)
        pairs.append([u, v])

    # Add unmatched singletons
    for v in G.nodes():
        if v not in matched:
            pairs.append([v])

    coarse, mapping = _contract_pairs(G, pairs)
    return coarse, mapping


def amg_inspired_placeholder(G):
    """Conceptual placeholder for algebraic multigrid inspired coarsening.

    This function currently implements a simple heuristic: pick high-degree seeds and grow aggregates.
    """
    degrees = sorted(G.degree(), key=lambda x: -x[1])
    used = set()
    pairs = []
    for v, d in degrees:
        if v in used:
            continue
        group = [v]
        used.add(v)
        for u in G.neighbors(v):
            if u not in used and len(group) < 3:
                used.add(u)
                group.append(u)
        pairs.append(group)

    coarse, mapping = _contract_pairs(G, pairs)
    return coarse, mapping
