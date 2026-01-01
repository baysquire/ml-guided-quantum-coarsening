# Theoretical Background

This document summarizes the theoretical motivation behind ML-guided graph coarsening for hybrid quantum optimization.

## Why coarsen graphs for quantum optimization
- Near-term quantum devices have limited qubits and coherence times, making large graph instances infeasible for direct quantum optimization.
- Coarsening reduces problem size while aiming to preserve the combinatorial structure relevant for objectives (e.g., MaxCut, QUBO).

## ML for coarsening
- Machine learning can learn from graph statistics and previous instances to predict which contractions preserve solution quality.
- Features include local degree, clustering coefficient, spectral properties, and edge weights.

## Relation to multilevel methods
- Multilevel methods (coarsen → solve → refine) provide a theoretically sound framework used in graph partitioning and multigrid.
- The design here is inspired by Safro-style multilevel optimization, with emphasis on learning-based coarsening decisions.

## Limitations & future work
- Current implementation is a prototype and focuses on clarity and extensibility.
- Future improvements include using Graph Neural Networks, better feature engineering, and integration with quantum simulators/hardware.

## References
- Safro et al., multilevel optimization literature
- Farhi et al., QAOA
- Hybrid quantum–classical optimization surveys
