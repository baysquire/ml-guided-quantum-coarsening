# Quantum Motivation

Short notes on why quantum optimization benefits from coarsening.

- QAOA and VQE aim to solve combinatorial optimization problems but are constrained by qubit count and noise.
- Coarsening reduces the instance size to fit within realistic qubit budgets while attempting to preserve solution quality.
- Hybrid workflows allow classical preprocessing (coarsening) and postprocessing (refinement) to complement quantum routines.

See `docs/theory.md` and `docs/multilevel-methods.md` for more details.
