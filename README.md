# ml-guided-quantum-coarsening

**ML-guided graph coarsening for hybrid classical–quantum optimization (MaxCut / QUBO).**

This repository is a research-oriented prototype for the PhD-level project "ML-Guided Graph Coarsening for Hybrid Quantum Optimization". The goal is to design a hybrid classical–quantum workflow where machine learning guides graph coarsening decisions and the coarsened graphs are used for (simulated) quantum optimization methods such as QAOA/VQE.

**This repository is part of ongoing PhD-level research preparation.**

---

**Repository metadata**

- **Name:** `ml-guided-quantum-coarsening`
- **Short description:** ML-guided graph coarsening for hybrid classical–quantum optimization (MaxCut / QUBO).
- **Topics:** `quantum-optimization`, `graph-coarsening`, `ml`, `qaoa`, `multilevel`


## Motivation

Quantum optimization algorithms like QAOA and VQE are limited by current hardware constraints (number of qubits, noise, circuit depth). Graph coarsening is a principled classical technique to reduce problem size; informing coarsening choices with ML can outperform fixed heuristics and make quantum-assisted methods applicable to larger instances.

## Structure

```
ml-guided-quantum-coarsening/
│
├── README.md
├── docs/
│   ├── theory.md
│   ├── quantum-motivation.md
│   ├── multilevel-methods.md
│
├── src/
│   ├── graph_generation.py
│   ├── coarsening/
│   │   ├── classical.py
│   │   ├── ml_guided.py
│   ├── quantum/
│   │   ├── qubo.py
│   │   ├── qaoa_interface.py
│   ├── refinement.py
│
├── experiments/
│   ├── baseline_vs_ml.py
│
├── requirements.txt
└── LICENSE
```

## Quick start

- Create a Python environment (Python 3.9+ recommended)
- Install dependencies: `pip install -r requirements.txt`
- Run a lightweight experiment: `python experiments/baseline_vs_ml.py --seed 0`

## Research notes
- The code intentionally provides readable, well-commented algorithmic sketches rather than production-ready implementations.
- See `docs/` for theoretical context and references.

## License
MIT
