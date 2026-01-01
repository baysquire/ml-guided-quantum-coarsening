"""ML-guided coarsening utilities.

Provides simple feature extraction and a RandomForest-based edge scorer prototype.
"""

import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def extract_edge_features(G):
    """Extract features for each edge in G and return (edges, X) where edges are (u,v) tuples.

    Features:
    - degree(u), degree(v)
    - clustering coefficient u/v
    - edge weight
    - common neighbors
    - Jaccard coefficient
    """
    deg = dict(G.degree())
    clustering = nx.clustering(G)

    edges = []
    X = []
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        cn = len(list(nx.common_neighbors(G, u, v)))
        # Jaccard
        try:
            jc = next(nx.jaccard_coefficient(G, [(u, v)]))[2]
        except StopIteration:
            jc = 0.0

        features = [deg[u], deg[v], clustering.get(u, 0.0), clustering.get(v, 0.0), w, cn, jc]
        edges.append((u, v))
        X.append(features)

    return edges, np.array(X)


def train_edge_scorer(G, labels=None, random_state=0):
    """Train a RandomForest regressor to score edges.

    labels: Optional array of target scores for each edge; if None, generates a synthetic target by
    favoring heavy edges.
    """
    edges, X = extract_edge_features(G)
    if labels is None:
        # Synthetic target: heavier edges are better to contract (toy proxy)
        labels = np.array([G[u][v].get("weight", 1.0) for u, v in edges])

    X_train, X_val, y_train, y_val = train_test_split(X, labels, random_state=random_state)
    model = RandomForestRegressor(n_estimators=50, random_state=random_state)
    model.fit(X_train, y_train)

    return model, edges, X


def predict_edges_to_contract(model, G, top_k=0.2):
    """Score edges with the learned model and return a set of edges to contract.

    top_k : fraction or integer. If float in (0,1), treat as fraction.
    """
    edges, X = extract_edge_features(G)
    scores = model.predict(X)
    n = len(edges)
    if 0 < top_k < 1:
        k = max(1, int(top_k * n))
    else:
        k = int(top_k)

    idx = np.argsort(scores)[-k:]
    chosen = [edges[i] for i in idx]
    return chosen


def contract_edge_list(G, edge_list):
    """Apply contractions for each edge in edge_list in a greedy fashion and return contracted graph and mapping."""
    # We'll reuse the classical._contract_pairs logic by transforming edges into disjoint pairs where possible
    pairs = []
    used = set()
    for u, v in edge_list:
        if u in used or v in used:
            continue
        pairs.append([u, v])
        used.add(u)
        used.add(v)

    # Add singletons
    for v in G.nodes():
        if v not in used:
            pairs.append([v])

    # Inline contraction to avoid circular imports
    from collections import defaultdict
    mapping = {}
    contracted = nx.Graph()

    node_to_coarse = {}
    for idx, group in enumerate(pairs):
        label = f"c{idx}"
        for v in group:
            node_to_coarse[v] = label
        mapping[label] = list(group)

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
