# VQEKrylovResearch: Hybrid Quantum Algorithms

This project explores the synergy between **Variational Quantum Eigensolvers (VQE)** and **Krylov Subspace Methods** for quantum chemistry simulations. 

The goal is to develop a more accurate and error-resistant method for calculating molecular ground state energies by combining the optimization strengths of VQE with the robust spectral properties of Krylov-based approaches.

## 🧪 Project Structure

* `VQE.py`: Implements a standard VQE simulation using Qiskit Nature and the UCCSD ansatz.
* `Krylov.py`: Implements Krylov subspace expansion methods for energy estimation. This is not working at the moment. :(
* `pyproject.toml` & `uv.lock`: Project configuration managed by [uv](https://astral.sh/uv/).

For details on the Krylov method read the KrylovMethods file.

The repo is found at https://github.com/tudorrosca-ui/VQEKrylovResearch
