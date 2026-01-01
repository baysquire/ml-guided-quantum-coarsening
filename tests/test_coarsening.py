import networkx as nx
from src.coarsening.classical import random_matching, heavy_edge_matching


def test_random_matching_small():
    G = nx.path_graph(4)
    coarse, mapping = random_matching(G, seed=0)
    # coarse graph should have at most 4 nodes and at least 2
    assert len(coarse) >= 2


def test_heavy_edge_matching():
    G = nx.Graph()
    G.add_edge(0, 1, weight=5)
    G.add_edge(1, 2, weight=1)
    coarse, mapping = heavy_edge_matching(G)
    # Nodes 0 and 1 should be matched first (heaviest)
    assert any(0 in m and 1 in m for m in mapping.values())
