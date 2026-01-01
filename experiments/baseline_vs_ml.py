"""Lightweight experiment comparing random, heavy-edge, and ML-guided coarsening on small graphs.

This script is intentionally small and designed for quick runs on CPUs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import networkx as nx
from src.graph_generation import generate_er_graph
from src.coarsening.classical import random_matching, heavy_edge_matching
from src.coarsening.ml_guided import train_edge_scorer, predict_edges_to_contract, contract_edge_list
from src.quantum.qubo import maxcut_to_qubo, qubo_energy


def exact_maxcut_value(G):
    """Compute exact MaxCut value by brute force (works for n <= ~20)."""
    nodes = list(G.nodes())
    n = len(nodes)
    best = -float("inf")
    best_assign = None
    for i in range(1 << n):
        x = [(i >> j) & 1 for j in range(n)]
        val = 0.0
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1.0)
            iu = nodes.index(u)
            iv = nodes.index(v)
            if x[iu] != x[iv]:
                val += w
        if val > best:
            best = val
            best_assign = x
    return best


def evaluate_coarsening(G, coarse_G, mapping):
    # For demonstration, compute exact MaxCut on small coarse graph and lift
    Q, nodes = maxcut_to_qubo(coarse_G)
    # brute force on coarse
    best = -float("inf")
    n = len(nodes)
    for i in range(1 << n):
        x = [(i >> j) & 1 for j in range(n)]
        # convert to dict mapping coarse label -> value
        sol = {nodes[j]: x[j] for j in range(n)}
        # lift
        fine_sol = {}
        for c, members in mapping.items():
            for v in members:
                fine_sol[v] = sol[c]
        # compute cut on fine graph
        val = 0.0
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1.0)
            if fine_sol[u] != fine_sol[v]:
                val += w
        if val > best:
            best = val
    return best


def run_once(n=12, p=0.4, seed=0):
    G = generate_er_graph(n, p, seed=seed)
    # baseline random matching
    Rcoarse, Rmap = random_matching(G, seed=seed)
    Rval = evaluate_coarsening(G, Rcoarse, Rmap)

    Hcoarse, Hmap = heavy_edge_matching(G)
    Hval = evaluate_coarsening(G, Hcoarse, Hmap)

    # ML-guided: train synthetic scorer and contract top edges
    model, edges, X = train_edge_scorer(G)
    chosen = predict_edges_to_contract(model, G, top_k=0.25)
    Mcoarse, Mmap = contract_edge_list(G, chosen)
    Mval = evaluate_coarsening(G, Mcoarse, Mmap)

    exact = exact_maxcut_value(G)

    return {
        "n": n,
        "exact": exact,
        "random": Rval,
        "heavy_edge": Hval,
        "ml_guided": Mval,
        "coarse_sizes": {"random": len(Rcoarse), "heavy": len(Hcoarse), "ml": len(Mcoarse)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    res = run_once(n=14, p=0.35, seed=args.seed)
    print("Experiment results:")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
