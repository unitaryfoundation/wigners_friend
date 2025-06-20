import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, Statevector, partial_trace
from functools import reduce

from utils import (
    oracle_access_to_hyp,
    compute_dt_basis,
    get_statevector_from_projector,
    get_inverse_basis_gate,
    get_basis_from_bloch_vector,
    compute_hyp_prime,
    _get_lab_kth_qubit_rho,
)

Id = np.eye(2, dtype=complex)


def certify_algorithm_1(hyp_circuit, lab_circuit):
    """
    Implements Algorithm 1 using a true dynamic circuit simulation.
    NOTE: This is only feasible for a small number of qubits (n).
    """
    num_qubits = lab_circuit.num_qubits
    if num_qubits > 3:
        raise NotImplementedError("Dynamic circuit simulation is only feasible for n <= 3.")
    if hyp_circuit.num_qubits != num_qubits:
        raise ValueError("Circuits must have the same number of qubits.")

    # 1. Sample k
    k = np.random.randint(1, num_qubits + 1)
    k_idx = k - 1
    non_k_indices = [i for i in range(num_qubits) if i != k_idx]
    # print(f"INFO: Randomly chose k={k}. Building dynamic circuit.")

    # 2. Classical Pre-computation of all required gates
    ref_qc = QuantumCircuit(num_qubits)
    dt_basis_gates = compute_dt_basis(hyp_circuit, ref_qc, oracle_access_to_hyp)
    
    gates_for_outcomes = {}
    num_outcomes = 2**len(non_k_indices)
    for i in range(num_outcomes):
        outcome_bitstring = format(i, f'0{len(non_k_indices)}b')
        
        conditioning_projectors = [Id] * num_qubits
        for j, qubit_idx in enumerate(non_k_indices):
            outcome_bit = outcome_bitstring[j]
            target_state = np.array([1,0]) if outcome_bit == '0' else np.array([0,1])
            adjoint_matrix = Operator(dt_basis_gates[qubit_idx]).adjoint().data
            original_basis_state = adjoint_matrix @ target_state
            conditioning_projectors[qubit_idx] = np.outer(original_basis_state, original_basis_state.conj())
        
        hyp_prime_state = compute_hyp_prime(hyp_circuit, k_idx, conditioning_projectors, oracle_access_to_hyp)
        gates_for_outcomes[outcome_bitstring] = get_inverse_basis_gate(hyp_prime_state)

    # 3. Build the single Dynamic Circuit
    # Create separate classical registers for the two measurement stages
    cr_non_k = ClassicalRegister(len(non_k_indices), name='cr_non_k')
    cr_k = ClassicalRegister(1, name='cr_k')
    
    # CORRECTED: Initialize circuit and add registers separately
    qc = QuantumCircuit(num_qubits)
    qc.add_register(cr_non_k)
    qc.add_register(cr_k)

    qc.compose(lab_circuit, inplace=True)
    qc.barrier()
    
    # First measurement on non-k qubits in DT basis
    for i, qubit_idx in enumerate(non_k_indices):
        qc.append(Operator(dt_basis_gates[qubit_idx]), [qubit_idx])
    qc.measure(non_k_indices, cr_non_k)
    qc.barrier()
    
    # --- Dynamic Block ---
    # Apply the correct final gate to the k-th qubit based on the measurement
    for outcome_bitstring, gate_matrix in gates_for_outcomes.items():
        outcome_int = int(outcome_bitstring, 2)
        # The 'if_test' condition must refer to the entire classical register
        with qc.if_test((cr_non_k, outcome_int)):
            qc.append(Operator(gate_matrix), [k_idx])
            
    qc.barrier()
    qc.measure(k_idx, cr_k)

    # 4. Simulate the circuit and get the final outcome
    simulator = AerSimulator()
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=1, memory=True)
    final_outcome_str = job.result().get_memory()[0]
    
    # CORRECTED: The memory string format is "cr_k cr_non_k".
    # The result for the k-th qubit is the first bit of the string.
    kth_qubit_result = final_outcome_str.split(' ')[0]
    
    return "ACCEPT" if kth_qubit_result == '0' else "REJECT"

# --- Example Usage ---
if __name__ == '__main__':
    n_qubits = 3

    hyp_qc = QuantumCircuit(n_qubits)
    hyp_qc.h(0)
    hyp_qc.cx(0, 1)
    hyp_qc.cx(0, 2)

    lab_qc_correct = hyp_qc.copy()

    lab_qc_wrong = QuantumCircuit(n_qubits)
    lab_qc_wrong.x(0)
    lab_qc_wrong.h(0)
    lab_qc_wrong.cx(0, 1)
    lab_qc_wrong.cx(0, 2)

    n_runs = 100
    print(f"Running {n_runs} certification trials...\n")

    correct_results = [certify_algorithm_1(hyp_qc, lab_qc_correct) for _ in range(n_runs)]
    wrong_results = [certify_algorithm_1(hyp_qc, lab_qc_wrong) for _ in range(n_runs)]

    accept_rate_correct = correct_results.count("ACCEPT") / n_runs
    accept_rate_wrong = wrong_results.count("ACCEPT") / n_runs

    print(f"Testing a Bell state against itself...")
    print(f"  Acceptance Rate: {accept_rate_correct:.2f} (Expected: High)")

    print(f"\nTesting a Bell state against an orthogonal state...")
    print(f"  Acceptance Rate: {accept_rate_wrong:.2f} (Expected: Low)")