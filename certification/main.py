import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, Statevector
from functools import reduce

from utils import (
    oracle_access_to_hyp,
    compute_dt_basis,
    get_inverse_basis_gate,
    compute_hyp_prime,
)

Id = np.eye(2, dtype=complex)


def certify_algorithm_1(hyp_circuit, lab_circuit):
    """
    Performs one round of the certification protocol.
    """
    num_qubits = lab_circuit.num_qubits
    if num_qubits > 10:
        raise NotImplementedError("Dynamic circuit simulation is only feasible for n <= 10.")
    if hyp_circuit.num_qubits != num_qubits:
        raise ValueError("Circuits must have the same number of qubits.")
        
    # Randomly select the qubit 'k' to test
    k = np.random.randint(1, num_qubits + 1)
    k_idx = k - 1
    non_k_indices = [i for i in range(num_qubits) if i != k_idx]

    # Compute the discriminating measurement basis (DT-basis) by comparing the
    # hypothesis circuit to the all-|0> state (represented by an empty circuit).
    ref_circuit = QuantumCircuit(num_qubits, name="ref")
    dt_basis_gates = compute_dt_basis(hyp_circuit, ref_circuit, oracle_access_to_hyp)
    
    # Pre-calculate the conditional measurement gates for the k-th qubit
    # for every possible outcome of measuring the other qubits.
    gates_for_outcomes = {}
    num_outcomes = 2**len(non_k_indices)
    for i in range(num_outcomes):
        outcome_bitstring = format(i, f'0{len(non_k_indices)}b')
        conditioning_projectors = [Id] * num_qubits
        
        # Build the projector for the current measurement outcome
        for j, qubit_idx in enumerate(non_k_indices):
            outcome_bit = outcome_bitstring[j]
            target_state = np.array([1,0]) if outcome_bit == '0' else np.array([0,1])
            adjoint_matrix = Operator(dt_basis_gates[qubit_idx]).adjoint().data
            original_basis_state = adjoint_matrix @ target_state
            conditioning_projectors[qubit_idx] = np.outer(original_basis_state, original_basis_state.conj())
        
        # Calculate the expected state of the k-th qubit given this outcome
        hyp_prime_state = compute_hyp_prime(hyp_circuit, k_idx, conditioning_projectors, oracle_access_to_hyp)
        gates_for_outcomes[outcome_bitstring] = get_inverse_basis_gate(hyp_prime_state)

    # --- Build the dynamic certification circuit ---
    cr_non_k = ClassicalRegister(len(non_k_indices), name='cr_non_k')
    cr_k = ClassicalRegister(1, name='cr_k')
    qc = QuantumCircuit(num_qubits)
    qc.add_register(cr_non_k, cr_k)
    
    # 1. Prepare the lab state
    qc.compose(lab_circuit, inplace=True)
    qc.barrier()
    
    # 2. Measure non-k qubits in the DT-basis
    for i, qubit_idx in enumerate(non_k_indices):
        qc.append(Operator(dt_basis_gates[qubit_idx]), [qubit_idx])
    qc.measure(non_k_indices, cr_non_k)
    qc.barrier()
    
    # 3. Conditionally apply the corrective rotation on qubit k
    for outcome_bitstring, gate_matrix in gates_for_outcomes.items():
        outcome_int = int(outcome_bitstring, 2)
        with qc.if_test((cr_non_k, outcome_int)):
            qc.append(Operator(gate_matrix), [k_idx])
            
    qc.barrier()
    
    # 4. Measure qubit k in the Z-basis. "ACCEPT" if result is '0'.
    qc.measure(k_idx, cr_k)

    # --- Simulate the circuit ---
    simulator = AerSimulator()
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=1, memory=True)
    final_outcome_str = job.result().get_memory()[0]
    
    kth_qubit_result = final_outcome_str.split(' ')[0]
    
    return "ACCEPT" if kth_qubit_result == '0' else "REJECT"

# --- Example Usage ---
if __name__ == '__main__':
    n_qubits = 4
    n_runs = 100

    # --- Setup the circuits ---
    # Hypothesis circuit: Creates a GHZ state
    hyp_qc = QuantumCircuit(n_qubits)
    hyp_qc.h(0)
    for i in range(1, n_qubits):
        hyp_qc.cx(i - 1, i)

    # Lab circuit (Correct): Creates the same GHZ state
    lab_qc_correct = hyp_qc.copy()

    # Lab circuit (Wrong): Creates a different, orthogonal stabilizer state
    lab_qc_wrong = QuantumCircuit(n_qubits)
    lab_qc_wrong.x(0) # Start with |1> instead of |0>
    lab_qc_wrong.h(0)
    for i in range(1, n_qubits):
        lab_qc_wrong.cx(i - 1, i)

    # --- Run Certifications ---
    print("--- Testing Stabilizer Circuit (GHZ State) ---")
    
    # Test 1: Certify against the correct state
    print(f"Running {n_runs} certification trials against itself...")
    correct_results = [certify_algorithm_1(hyp_qc, lab_qc_correct) for _ in range(n_runs)]
    accept_rate_correct = correct_results.count("ACCEPT") / n_runs
    print(f"  Acceptance Rate for correct state: {accept_rate_correct:.2f} (Expected: High)")

    # Test 2: Certify against the wrong state
    print(f"\nRunning {n_runs} certification trials against an orthogonal state...")
    wrong_results = [certify_algorithm_1(hyp_qc, lab_qc_wrong) for _ in range(n_runs)]
    accept_rate_wrong = wrong_results.count("ACCEPT") / n_runs
    print(f"  Acceptance Rate for wrong state: {accept_rate_wrong:.2f} (Expected: Low)")
