
# 1. Imports
import numpy as np
import time

# --- FIX: Updated imports for modern Qiskit Nature structure (v0.6+) ---
# The 'second_quantization' path has been removed from these modules.
# from qiskit_nature.conversions.mappers import JordanWignerMapper
# from qiskit_nature.drivers import Molecule, PySCFDriver
# from qiskit_nature.algorithms import GroundStateEigensolver
# from qiskit_nature.circuit.library import UCC
# # -----------------------------------------------------------------------
#
# from qiskit_algorithms import VQE
# from qiskit_algorithms.optimizers import SLSQP
# # Note: NumPyMinimumEigensolver is removed as requested.
#
# from qiskit.primitives import Estimator
# from qiskit.utils.logging import set_global_logging_level
import pennylane as qml
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import StatevectorEstimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver



# Disable Qiskit logging for cleaner output


def run_vqe(molecule_name, bond_length):
    """
    Sets up and runs VQE on a Hydrogen (H2) molecule at a fixed bond distance.
    This version only runs the VQE quantum algorithm.
    """
    print("--- Starting H2 VQE Simulation (Quantum-Only) ---")
    start_time = time.time()
    DATASET = qml.data.load("qchem", molname=molecule_name, bondlength=bond_length, basis="STO-3G")[0]
    COORDS = DATASET.molecule.coordinates
    SYMBOLS = DATASET.molecule.symbols
    CHARGE = DATASET.molecule.charge
    # --- 2. Define the Molecular Problem ---
    # Define H2 molecule geometry (H-H distance = 0.735 Angstrom, near equilibrium)
    atom_info = ""
    print(SYMBOLS)
    print(COORDS)
    for i in range(len(SYMBOLS)):
        atom_info += SYMBOLS[i] + " " + str(COORDS[i][0]) + " " + str(COORDS[i][1]) + " " + str(COORDS[i][2]) + "; "

    atom_info = atom_info[0:-2]
    print(atom_info)
    # Use PySCFDriver to compute molecular integrals (classical pre-processing)
    # sto-3g is the minimal basis set for H2.
    driver = PySCFDriver(atom=atom_info,
        charge=CHARGE,
        unit=DistanceUnit.BOHR,
        basis="sto3g")
    problem = driver.run()

    print(
        f"Classical problem created: {problem.num_spatial_orbitals} spatial orbitals, {problem.num_particles} electrons.")

    # --- 3. Define the Mapper and Qubit Operator ---
    # Jordan-Wigner is the most common mapping for transforming fermionic to qubit operators.
    mapper = JordanWignerMapper()

    # --- 4. Define the Quantum Solver (VQE Setup) ---

    # Ansatz: Unitary Coupled Cluster with Singles and Doubles excitations (UCCSD)
    # This is highly effective for calculating ground state energy in molecular systems.
    ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
        ),
    )

    # Optimizer: Sequential Least Squares Programming (SLSQP)
    optimizer = SLSQP(maxiter=100)

    # Primitive: Estimator is used to calculate the expectation value of the Hamiltonian
    estimator = StatevectorEstimator()

    # VQE Algorithm
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer
    )
    vqe.initial_point = [0.0] * ansatz.num_parameters


    # Use the GroundStateEigensolver wrapper to combine the problem, mapper, and VQE
    vqe_gse = GroundStateEigensolver(mapper, vqe)

    # --- 5. Run VQE Optimization ---
    print("\nStarting VQE optimization...")
    vqe_result = vqe_gse.solve(problem)
    print(vqe_result)
    vqe_energy = vqe_result.total_energies[0]
    print(f"\n-----FINAL ENERGY: {vqe_energy}-----\n")
    end_time = time.time()
    expected_energy = DATASET.fci_energy

    # --- 6. Return and Print Results ---
    print("\n--- Results ---")
    print(f"VQE Calculated Ground State Energy (Hartree): {vqe_energy:.8f}")
    print(f"Expected Energy: {expected_energy} Ha")
    print(f"Optimization Time: {end_time - start_time:.2f} seconds")

    # Return the calculated VQE energy
    return vqe_energy


if __name__ == "__main__":
    lowest_energy = run_vqe("H2", 0.742)

    # Bond lengths H2: 0.742 H3+: 0.874 CH4: 1.086
    # Note: The energy value (Hartree) typically represents the total energy
    # including nuclear repulsion. For H2 at 0.735A (sto-3g), a good result is
    # around -1.1373 Hartree.
