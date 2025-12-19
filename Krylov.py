

import numpy as np
import scipy.linalg as la
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit import QuantumCircuit, Parameter, CircuitInstruction
from qiskit.circuit.library import PauliEvolutionGate, CSwapGate, PauliGate
from qiskit_algorithms import TimeEvolutionProblem, TrotterQRTE
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.units import DistanceUnit
from qiskit.circuit.measure import Measure
from qiskit import transpile
import pennylane as qml
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator

# --- CONFIGURATION VARIABLES ---
KRYLOV_DIMENSION = 3
TAU = 0.1
TROTT_STEPS = 5


def get_krylov_state_circuit(initial_state_circuit, H_op, k, tau, trotter_steps):
    """
    Constructs and returns the circuit for the k-th Krylov state |psi_k> = U(k*tau) |psi_0>.
    """

    evolution_time = k * tau

    if k == 0: # if the timestep is zero return the initial state
        unitary_circuit = initial_state_circuit.copy()
        unitary_circuit = unitary_circuit.decompose()
    else:
        evolution_problem = TimeEvolutionProblem(
            hamiltonian=H_op,
            time=evolution_time,
            initial_state=initial_state_circuit
        )
        trotter_strategy = TrotterQRTE(num_timesteps=trotter_steps)
        full_circuit = trotter_strategy.evolve(evolution_problem)
        unitary_circuit = full_circuit.evolved_state.decompose()

    return unitary_circuit.to_gate()


def haddy_test(num_qubits, circuit_i, circuit_j, pauli_string, ancilla_phase):
    """
    Constructs the Hadamard test circuit to measure <psi_i | pauli_string | psi_j>.
    """
    N = num_qubits
    total_qubits = N + 1
    qc = QuantumCircuit(total_qubits)

    ancilla = 0
    targets = list(range(1, total_qubits))
    qubit_list = [ancilla] + targets


    # --- 1. Define Controlled Versions of Krylov State Circuits ---
    U = QuantumCircuit(N)
    
    # Apply U_i^dagger on the N target qubits
    U.append(circuit_i.inverse(), range(N)) 

    # Apply the Pauli observable P_p on the N target qubits
    if pauli_string != Pauli('I' * N):
        pauli_gate=PauliGate(pauli_string.to_label())
        U.append(pauli_gate, range(N))

    # Apply U_j on the N target qubits
    U.compose(circuit_j, range(N))
    
    # Convert the N-qubit circuit U to a controlled instruction CU (N+1 qubits)
    CU_circuit = U.control(1)
    CU = CU_circuit.data[0].operation

    # --- 2. Ancilla Initialization (Hadamard and Phase) ---
    qc.h(ancilla)
    if ancilla_phase:
        qc.sdg(ancilla)

    # --- 3. Apply First Controlled Circuit (j circuit) ---


    # --- 4. Observable Application (Hamiltonian measurement) ---

    # --- 5. Apply Last Controlled Gate (i dagger circuit) ---
    qc.append(CU, qubit_list)
    # --- 6. Final Ancilla Hadamard ---
    qc.h(ancilla)

    return qc


def quantum_krylov_diagonalization():
    """
    Performs Quantum Krylov Diagonalization (QKD) for a molecule.
    """
    M = KRYLOV_DIMENSION
    tau = TAU
    trotter_steps = TROTT_STEPS

    # --- 1. Define the Molecule and Hamiltonian ---
    print(f"--- 1. Setting up Molecule for QKD (M={M}, tau={tau}, steps={trotter_steps}) ---")
    molecule_name = "H2"
    bond_length = 0.742  # LiH 1.57
    # get molecule data using pennylane
    DATASET = qml.data.load("qchem", molname=molecule_name, bondlength=bond_length, basis="STO-3G")[0]
    COORDS = DATASET.molecule.coordinates
    SYMBOLS = DATASET.molecule.symbols
    CHARGE = DATASET.molecule.charge

    # get pennylane data into readable form for pySCFDriver
    atom_info = ""
    for i in range(len(SYMBOLS)):
        atom_info += SYMBOLS[i] + " " + str(COORDS[i][0]) + " " + str(COORDS[i][1]) + " " + str(COORDS[i][2]) + "; "

    atom_info = atom_info[0:-2]
    # Use PySCFDriver to compute molecular integrals
    driver = PySCFDriver(atom=atom_info,
                         charge=CHARGE,
                         unit=DistanceUnit.BOHR,
                         basis="sto3g")
    problem = driver.run()

    # --- 2. Map to Qubit Operator and Decompose ---
    mapper = JordanWignerMapper()
    second_quantized_op = problem.hamiltonian.second_q_op()

    H_pauli_op = mapper.map(second_quantized_op)


    nuclear_repulsion_energy = problem.nuclear_repulsion_energy

    pauli_coeffs = H_pauli_op.coeffs
    pauli_terms = H_pauli_op.paulis

    num_particles = (problem.num_alpha, problem.num_beta)
    num_spatial_orbitals = problem.num_spatial_orbitals
    num_qubits = H_pauli_op.num_qubits  # N

    # --- 3. Prepare the Initial State (Hartree-Fock) ---
    print("\n--- 3. Preparing Hartree-Fock State ---")

    initial_state_circuit = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper
    )

    # --- 4. Pre-calculate Krylov Circuits ---
    print(f"\n--- 4. Pre-calculating {M} Krylov Circuits (U(k*tau)|HF>) ---")

    krylov_circuits = []

    for k in range(M):
        circuit_k = get_krylov_state_circuit(initial_state_circuit, H_pauli_op, k, tau, trotter_steps)
        krylov_circuits.append(circuit_k)

    # --- 5. Calculate Overlap Matrix and Projected Hamiltonian ---

    H_K = np.zeros((M, M), dtype=complex)
    S_K = np.zeros((M, M), dtype=complex)
    estimator = Estimator()

    observe_ancilla = Pauli('Z' + 'I' * num_qubits)
    print(f"\n--- 5. Building Scalable Krylov Subspace (Dimension M={M}) ---")

    for i in range(M):
        for j in range(i + 1):
            print(f"Measuring element ({i},{j})...")

            circuit_i = krylov_circuits[i]
            circuit_j = krylov_circuits[j]

            # --- OVERLAP MATRIX S_K[i, j] = <psi_i | I | psi_j> ---

            observable_I = Pauli('I' * num_qubits)

            qc_S_real = haddy_test(num_qubits, circuit_i, circuit_j, observable_I, False)
            qc_S_imag = haddy_test(num_qubits, circuit_i, circuit_j, observable_I, True)

            primitive_result = estimator.run([qc_S_real, qc_S_imag], [observe_ancilla, observe_ancilla]).result()

            S_ij_real = primitive_result.values[0]
            S_ij_imag = primitive_result.values[1]
            S_ij = S_ij_real + 1j * S_ij_imag
            print(f"S element {i},{j}: {S_ij}")

            S_K[i, j] = S_ij
            if i != j:
                S_K[j, i] = np.conj(S_ij)  # Hermiticity

            # --- HAMILTONIAN MATRIX H_K[i, j] = sum_p h_p * <psi_i| P_p |psi_j> ---

            H_ij_sum = 0.0 + 0j

            for h_p, P_p in zip(pauli_coeffs, pauli_terms):
                qc_H_real = haddy_test(num_qubits, circuit_i, circuit_j, P_p, ancilla_phase=False)
                qc_H_imag = haddy_test(num_qubits, circuit_i, circuit_j, P_p, ancilla_phase=True)
                results_H = estimator.run([qc_H_real, qc_H_imag], [observe_ancilla, observe_ancilla]).result()

                P_p_overlap_real = results_H.values[0]
                P_p_overlap_imag = results_H.values[1]

                P_p_overlap = P_p_overlap_real + 1j * P_p_overlap_imag
                H_ij_sum += h_p * P_p_overlap
                print(f"Pauli sum: {P_p_overlap} Pauli: {P_p} weight: {h_p}")
            print(f"Final sum: {H_ij_sum}")
            H_K[i, j] = H_ij_sum
            if i != j:
                H_K[j, i] = np.conj(H_ij_sum)  # Hermiticity

    # --- 6. Diagonalize the Subspace Problem (Classical Final Step) ---
    print("\n--- 6. Solving Generalized Eigenvalue Problem ---")
    print("Effective Krylov Hamiltonian Matrix (H_K, Real Part):")
    print(np.round(H_K.real, 6))
    print("Effective Krylov Hamiltonian Matrix (H_K, Imaginary Part):")
    print(np.round(H_K.imag, 6))
    print("\nEffective Krylov Overlap Matrix (S_K, Real Part):")
    print(np.round(S_K.real, 6))
    print("\nEffective Krylov Overlap Matrix (S_K, Imaginary Part):")
    print(np.round(S_K.imag, 6))

    # The Generalized Eigenvalue Problem is H_K * c = E * S_K * c
    eigenvalues = la.eig(H_K, S_K, right=False)

    ground_electronic_energy = np.min(eigenvalues)

    print(f"\nCalculated Krylov Subspace Eigenvalues: {eigenvalues}")
    print(f"QKD Electronic Ground Energy (min eigenvalue): {ground_electronic_energy:.6f} Hartree")

    # Final result (Total Energy)
    total_ground_energy = ground_electronic_energy + nuclear_repulsion_energy

    print("\n========================================================")
    print(f"TOTAL GROUND ENERGY (H2, M={M}): {total_ground_energy:.6f} Hartree")
    print("========================================================")


if __name__ == "__main__":
    quantum_krylov_diagonalization()
