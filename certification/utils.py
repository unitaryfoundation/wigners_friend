import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.quantum_info import Operator, Statevector, partial_trace, StabilizerState
from qiskit_aer import AerSimulator
from qiskit.exceptions import QiskitError
from functools import reduce


# --- Helper Functions ---

Id = np.eye(2, dtype=complex)

def is_stabilizer_circuit(circuit: QuantumCircuit) -> bool:
    """
    Checks if a circuit is a stabilizer circuit by attempting to convert it
    to a StabilizerState. This is a robust method provided by Qiskit.
    """
    try:
        StabilizerState(circuit)
        return True
    except QiskitError:
        return False

def _oracle_statevector_fallback(hyp_circuit: QuantumCircuit, projectors: list) -> float:
    """
    The original oracle logic using full statevector simulation.
    This is used for non-stabilizer circuits.
    """
    hyp_statevector = Statevector.from_instruction(hyp_circuit)
    if len(projectors) != hyp_statevector.num_qubits:
        raise ValueError(f"Number of projectors ({len(projectors)}) does not match number of qubits ({hyp_statevector.num_qubits}).")
    full_projector_op = reduce(np.kron, reversed(projectors))
    projected_psi = full_projector_op @ hyp_statevector.data
    return np.linalg.norm(projected_psi)**2

def oracle_access_to_hyp(hyp_circuit: QuantumCircuit, projectors: list) -> float:
    """
    Calculates the probability <ψ|Π|ψ> for the state |ψ> produced by hyp_circuit.
    This function implements a hybrid simulation strategy.
    """
    num_qubits = hyp_circuit.num_qubits
    if len(projectors) != num_qubits:
        raise ValueError(f"Projector count ({len(projectors)}) must match qubit count ({num_qubits}).")

    if is_stabilizer_circuit(hyp_circuit):
        p_0 = np.array([[1,0],[0,0]], dtype=complex)
        p_plus = np.outer(np.array([1,1]), np.array([1,1])) / 2.0
        p_i = np.outer(np.array([1,1j]), np.array([1,-1j])) / 2.0

        prob_circuit = QuantumCircuit(num_qubits, name="prob_calc")
        prob_circuit.compose(hyp_circuit, inplace=True)
        
        measured_qubits = []
        for i, proj_matrix in enumerate(projectors):
            if np.allclose(proj_matrix, Id):
                continue
            
            measured_qubits.append(i)
            if np.allclose(proj_matrix, p_0):
                pass 
            elif np.allclose(proj_matrix, p_plus):
                prob_circuit.h(i)
            elif np.allclose(proj_matrix, p_i):
                prob_circuit.sdg(i)
                prob_circuit.h(i)
            else:
                return _oracle_statevector_fallback(hyp_circuit, projectors)

        if not measured_qubits:
            return 1.0
            
        final_state = StabilizerState(prob_circuit)
        prob_dict = final_state.probabilities_dict()

        target_prob = 0.0
        for outcome_bin, prob in prob_dict.items():
            is_target_outcome = all(outcome_bin[num_qubits - 1 - qubit_idx] == '0' for qubit_idx in measured_qubits)
            if is_target_outcome:
                target_prob += prob
        return target_prob
    else:
        return _oracle_statevector_fallback(hyp_circuit, projectors)

def get_statevector_from_projector(proj: np.ndarray):
    """
    Extracts the statevector |ψ> from a rank-1 projector Π = |ψ><ψ|.
    """
    if not np.isclose(np.trace(proj), 1.0) or not np.isclose(np.linalg.matrix_rank(proj), 1):
        raise ValueError("Input must be a valid rank-1 projector.")
    col_norms = np.linalg.norm(proj, axis=0)
    if np.all(np.isclose(col_norms, 0)):
        raise ValueError("Projector matrix cannot be all zeros.")
    max_col_index = np.argmax(col_norms)
    state_vector = proj[:, max_col_index]
    norm = np.linalg.norm(state_vector)
    if np.isclose(norm, 0):
        raise ValueError("Projector appears to be rank-0.")
    return state_vector / norm

def get_inverse_basis_gate(state_vector: np.ndarray):
    """
    Computes the unitary U_dag where U|0> = state_vector.
    """
    alpha, beta = state_vector[0], state_vector[1]
    u_dag = np.array([[np.conj(alpha), np.conj(beta)], [-beta, alpha]], dtype=complex)
    return u_dag

def get_reduced_bloch_vector(hyp_circuit, qubit_index, oracle):
    """
    Calculates the Bloch vector of a single qubit by querying the oracle.
    """
    num_qubits = hyp_circuit.num_qubits
    p_plus = np.outer(np.array([1, 1]) / np.sqrt(2), np.array([1, 1]) / np.sqrt(2))
    p_i = np.outer(np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2))
    p_0 = np.array([[1,0],[0,0]], dtype=complex)
    
    proj_list_x = [Id] * num_qubits; proj_list_x[qubit_index] = p_plus
    exp_x = 2 * oracle(hyp_circuit, proj_list_x) - 1
    
    proj_list_y = [Id] * num_qubits; proj_list_y[qubit_index] = p_i
    exp_y = 2 * oracle(hyp_circuit, proj_list_y) - 1
    
    proj_list_z = [Id] * num_qubits; proj_list_z[qubit_index] = p_0
    exp_z = 2 * oracle(hyp_circuit, proj_list_z) - 1
    
    return np.array([exp_x, exp_y, exp_z])

def get_basis_from_bloch_vector(b_vec):
    """
    Computes the basis-changing unitary U† from a Bloch vector.
    """
    norm = np.linalg.norm(b_vec)
    if np.isclose(norm, 0): return Id
    b_vec_norm = b_vec / norm
    x, y, z = b_vec_norm
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    state_vec = np.array([alpha, beta])
    return get_inverse_basis_gate(state_vec)

def get_conditional_reduced_bloch_vector(hyp_circuit: QuantumCircuit, qubit_index: int, conditioning_projectors: list, oracle) -> np.ndarray:
    """
    Calculates the Bloch vector of a qubit conditioned on projections on other qubits,
    by querying the oracle.
    """
    prob_condition = oracle(hyp_circuit, conditioning_projectors)

    if np.isclose(prob_condition, 0):
        return np.array([0, 0, 0])

    p_plus = np.outer(np.array([1, 1]) / np.sqrt(2), np.array([1, 1]) / np.sqrt(2))
    p_i = np.outer(np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2))
    p_0 = np.array([[1, 0], [0, 0]], dtype=complex)

    proj_x_list = conditioning_projectors.copy(); proj_x_list[qubit_index] = p_plus
    prob_x = oracle(hyp_circuit, proj_x_list)
    exp_x = (2 * prob_x / prob_condition) - 1 if not np.isclose(prob_condition, 0) else 0

    proj_y_list = conditioning_projectors.copy(); proj_y_list[qubit_index] = p_i
    prob_y = oracle(hyp_circuit, proj_y_list)
    exp_y = (2 * prob_y / prob_condition) - 1 if not np.isclose(prob_condition, 0) else 0

    proj_z_list = conditioning_projectors.copy(); proj_z_list[qubit_index] = p_0
    prob_z = oracle(hyp_circuit, proj_z_list)
    exp_z = (2 * prob_z / prob_condition) - 1 if not np.isclose(prob_condition, 0) else 0

    bloch_vector = np.array([exp_x, exp_y, exp_z])
    
    norm = np.linalg.norm(bloch_vector)
    if norm > 1.0 and not np.isclose(norm, 1.0):
        bloch_vector = bloch_vector / norm

    return bloch_vector

def compute_dt_basis(hyp_circuit_0: QuantumCircuit, hyp_circuit_1: QuantumCircuit, oracle):
    """
    Computes a DT basis by comparing two hypothesis circuits.
    """
    num_qubits = hyp_circuit_0.num_qubits
    if num_qubits != hyp_circuit_1.num_qubits:
        raise ValueError("Input circuits must have the same number of qubits.")

    final_basis_gates = [Id] * num_qubits
    path_projectors = [Id] * num_qubits
    for i in range(num_qubits):
        v0 = get_conditional_reduced_bloch_vector(hyp_circuit_0, i, path_projectors, oracle)
        v1 = get_conditional_reduced_bloch_vector(hyp_circuit_1, i, path_projectors, oracle)

        v_b = np.cross(v0, v1)
        # BUG FIX: Robustly handle the case where v0 and v1 are collinear.
        if np.isclose(np.linalg.norm(v_b), 0):
            # If v0 is not the zero vector, find an orthogonal vector to it.
            if not np.isclose(np.linalg.norm(v0), 0):
                # Pick a standard axis to cross with, ensuring it's not parallel to v0.
                if np.abs(np.dot(v0 / np.linalg.norm(v0), [1, 0, 0])) < 0.99:
                    v_b = np.cross(v0, [1, 0, 0])
                else:
                    v_b = np.cross(v0, [0, 1, 0])
            # If v0 is the zero vector, do the same for v1.
            elif not np.isclose(np.linalg.norm(v1), 0):
                if np.abs(np.dot(v1 / np.linalg.norm(v1), [1, 0, 0])) < 0.99:
                    v_b = np.cross(v1, [1, 0, 0])
                else:
                    v_b = np.cross(v1, [0, 1, 0])
        
        # If a valid cross product still hasn't been found (e.g., both v0 and v1 are zero),
        # default to a standard basis.
        if np.isclose(np.linalg.norm(v_b), 0):
            v_b = np.array([1, 0, 0]) # Default to X-basis measurement

        basis_gate = get_basis_from_bloch_vector(v_b)
        final_basis_gates[i] = basis_gate
        
        zero_vec = np.array([1, 0])
        outcome_state_vec = Operator(basis_gate).adjoint().data @ zero_vec
        path_projectors[i] = np.outer(outcome_state_vec, outcome_state_vec.conj())
        
    return final_basis_gates

def are_gates_phase_equivalent(U, V, rtol=1e-2, atol=1e-2):
    """
    Checks if two unitary matrices U and V are equivalent up to a global phase.
    """
    if U.shape != V.shape: return False
    dim = U.shape[0]
    trace_val = np.abs(np.trace(V.conj().T @ U))
    return np.isclose(trace_val, dim, rtol=rtol, atol=atol)

def compute_hyp_prime(hyp_circuit: QuantumCircuit, k_idx, conditioning_projectors, oracle):
    """
    Computes |hyp'>, the state of qubit k conditioned on measurements on other qubits.
    """
    bloch_vector = get_conditional_reduced_bloch_vector(hyp_circuit, k_idx, conditioning_projectors, oracle)

    # Convert Bloch vector to statevector
    x, y, z = bloch_vector
    theta = np.arccos(np.clip(z, -1.0, 1.0)); phi = np.arctan2(y, x)
    alpha = np.cos(theta / 2); beta = np.exp(1j * phi) * np.sin(theta / 2)
    return np.array([alpha, beta])

def get_reduced_density_matrix(full_statevector, k_idx):
    """
    Simulation shortcut: Gets the reduced density matrix via partial trace.
    """
    q_args = list(range(full_statevector.num_qubits))
    q_args.remove(k_idx)
    return partial_trace(full_statevector, q_args).data

def _get_lab_kth_qubit_rho(lab_circuit, k_idx, conditioning_projectors):
    """
    (Simulation Shortcut) Calculates the ideal post-measurement density matrix.
    """
    full_projector = reduce(np.kron, reversed(conditioning_projectors))
    lab_state_full = Statevector.from_instruction(lab_circuit)
    post_lab_unnormalized = full_projector @ lab_state_full.data
    lab_norm = np.linalg.norm(post_lab_unnormalized)

    if np.isclose(lab_norm, 0):
        return None

    post_lab_full = Statevector(post_lab_unnormalized / lab_norm)
    q_args = list(range(lab_circuit.num_qubits))
    q_args.remove(k_idx)
    return partial_trace(post_lab_full, q_args).data
