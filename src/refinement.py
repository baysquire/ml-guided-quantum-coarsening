"""Refinement and lifting utilities for multilevel workflows.

- lift_solution maps a coarse solution back to the fine graph using the mapping returned by coarsening.
- local_refinement performs simple local improvements for MaxCut via greedy flips.
"""

import random


def lift_solution(coarse_solution, mapping):
    """Lift a solution defined on coarse nodes to the original nodes.

    coarse_solution: dict mapping coarse_label -> binary assignment (0/1)
    mapping: dict mapping coarse_label -> list(original_nodes)

    Returns: dict mapping original_node -> assignment
    """
    fine_sol = {}
    for cnode, members in mapping.items():
        val = coarse_solution.get(cnode, 0)
        for v in members:
            fine_sol[v] = val
    return fine_sol


def local_refinement(G, solution, max_iter=100):
    """Perform a simple local greedy improvement for MaxCut: flip nodes that reduce cut energy.

    solution: dict mapping node -> {0,1}
    """
    def cut_value(sol):
        val = 0.0
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1.0)
            if sol[u] != sol[v]:
                val -= w  # minimize negative cut
            else:
                val += 0
        return val

    sol = solution.copy()
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        nodes = list(G.nodes())
        random.shuffle(nodes)
        for v in nodes:
            current = sol[v]
            sol[v] = 1 - current
            new_val = cut_value(sol)
            sol[v] = current
            if new_val < cut_value(sol):
                sol[v] = 1 - current
                improved = True
    return sol
