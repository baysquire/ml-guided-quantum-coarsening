"""Lightweight QAOA/VQE interface abstraction (simulated).

This module provides a simple simulated hybrid interface rather than real quantum circuits. It is intended to illustrate how coarsened problems are passed to quantum optimization routines.
"""

import numpy as np
import random


class QAOAInterface:
    """A toy simulator for QAOA-like optimization for demonstration only.

    The `optimize` method receives a cost function (callable mapping binary vectors to objective value)
    and performs a randomized search to emulate the hybrid classical-quantum outer loop.
    """

    def __init__(self, n_qubits, p=1, random_state=0):
        self.n_qubits = n_qubits
        self.p = p
        self.rs = random.Random(random_state)

    def optimize(self, cost_fn, n_restarts=100, noise=0.0):
        """Return best found binary vector and its cost (lower is better).

        This is a classical randomized search used as a stand-in for a quantum optimizer.
        """
        best = None
        best_val = float("inf")
        for _ in range(n_restarts):
            x = np.array([self.rs.choice([0, 1]) for _ in range(self.n_qubits)])
            val = cost_fn(x)
            val = val + self.rs.gauss(0, noise)
            if val < best_val:
                best_val = val
                best = x.copy()
        return best, best_val


def demo_usage():
    """Demonstrate how to use QAOAInterface with a simple quadratic cost."""
    n = 6
    rng = np.random.RandomState(0)
    Q = rng.randn(n, n)
    Q = (Q + Q.T) / 2

    def cost(x):
        return float(x @ Q @ x)

    q = QAOAInterface(n_qubits=n, p=1, random_state=0)
    sol, val = q.optimize(cost, n_restarts=1000)
    return sol, val
