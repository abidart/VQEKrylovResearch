The Krylov.py file uses the Quantum Krylov Diagonalization method to solve for the ground state of a molecule.

It does this by:

1. Defining basis states

The first basis state is an initial guess state meant to be as close to the lowest energy state as possible.

The following basis states are obtained by applying the time evolution operator to this first basis state
To get multiple different basis states, the time evolution is done for different amounts of time.

The circuits (U_k) to prepare each of these states (|psi_k>) is stored. |psi_k>=U_k|o>

2. Calculating the overlap matrix and projected Hamiltonian in the new basis

The overlap matrix (S) can be calculated as S_ij=<psi_i||psi_j>
The projected Hamiltonian (H') can be calculated as H'_ij=<psi_i|H|psi_j>

Using the cicuits to prepare the Krylov states we can write:
S_ij=<0|U_i^dagger U_j |0>
H'_ij = <0|U_i^dagger H U_j |0>

H is decomposed into a linear combination of pauli strings (P_p) with weighting coefficients (h_p) 
H= \sum h_p(P_p)

Now the projected Hamiltonian can be written as:
H'_ij= \sum h_p (<0|U_i^dagger P_p U_j|0>)


Now S and H' can be calculated using these formulas and the Hadamard Test.

The Hadamard calculates Re(<psi|U|psi>) and Im(<psi|U|psi>) by preparing |psi> on a register along with an ancillary qubit |0>
A hadamard gate is applied on the ancillary qubit, then if the imaginary part is to be measured an Sdagger gate is applied to the ancillary qubit
Then a controlled U gate is applied with the ancillary bit as the control and the register with |psi> as the target
Finally one last Hadamard is applied to the ancillary qubit.

Now the expectation value of the ancillary qubit in the Z basis is the value of either  Re(<psi|U|psi>) or Im(<psi|U|psi>) depending on the version used.

We use the Hadamard test with U = U_i^dagger U_j for the S entries and U = U_i^dagger P_p U_j for H' entries


3. Solve the Generalized Eigenvalue Problem

Solve H'*c = E * S * c for the lowest eigenvector (E) representing the lowest energy and the lowest eigenstate c for this lowest energy.



PROBLEM:

First of all, the calculated energy from this method is incorrect.
Also, when calculating the overlap matrix, the diagonal entries should all be 1 because the overlap of a vector with itself should be 1
However this method calulates it to be 1+1i which cannot be correct.
