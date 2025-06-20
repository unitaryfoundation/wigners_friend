import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Operator, Statevector, partial_trace
from functools import reduce


# --- Helper Functions ---

Id = np.eye(2, dtype=complex)

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
    This U_dag is the gate that rotates the state_vector to |0>.
    The matrix is [[a*, b*], [-b, a]].
    """
    alpha = state_vector[0]
    beta = state_vector[1]
    
    u_dag = np.array([
        [np.conj(alpha), np.conj(beta)],
        [-beta, alpha]
    ], dtype=complex)
    
    return u_dag

def oracle_access_to_hyp(hyp_circuit: QuantumCircuit, projectors: list):
    """
    Calculates <ψ|Π|ψ> using classical linear algebra from a circuit.
    This is deterministic.
    """
    # Convert circuit to statevector to represent the classical knowledge.
    hyp_statevector = Statevector.from_instruction(hyp_circuit)

    num_qubits = hyp_statevector.num_qubits
    if len(projectors) != num_qubits:
        raise ValueError("Projector count must match qubit count.")

    # Build the full projector operator Π by taking the tensor product
    full_projector_op = reduce(np.kron, reversed(projectors))

    # Project the statevector: |ψ'> = Π|ψ>
    projected_psi = full_projector_op @ hyp_statevector.data

    # The probability is the squared norm of the resulting vector
    return np.linalg.norm(projected_psi)**2

def get_reduced_bloch_vector(hyp_circuit, qubit_index, oracle):
    """
    Calculates the Bloch vector of a single qubit by querying the oracle.
    """
    num_qubits = hyp_circuit.num_qubits

    psi_plus = np.array([1, 1]) / np.sqrt(2); p_plus = np.outer(psi_plus, psi_plus.conj())
    psi_i = np.array([1, 1j]) / np.sqrt(2); p_i = np.outer(psi_i, psi_i.conj())
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

def get_conditional_reduced_bloch_vector(statevector, qubit_index, conditioning_projectors):
    """
    Calculates the bloch vector of a qubit conditioned on projections on other qubits.
    """
    full_projector = reduce(np.kron, reversed(conditioning_projectors))
    post_state_unnormalized = full_projector @ statevector.data
    norm = np.linalg.norm(post_state_unnormalized)

    if np.isclose(norm, 0):
        return np.array([0, 0, 0])

    post_statevector = Statevector(post_state_unnormalized / norm)
    
    rho_k = partial_trace(post_statevector, [q for q in range(statevector.num_qubits) if q != qubit_index])
    
    X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.array([[1, 0], [0, -1]])
    
    exp_x = np.trace(rho_k.data @ X).real
    exp_y = np.trace(rho_k.data @ Y).real
    exp_z = np.trace(rho_k.data @ Z).real

    return np.array([exp_x, exp_y, exp_z])

def compute_dt_basis(hyp_circuit_0, hyp_circuit_1, oracle):
    """
    Computes a single, fixed basis corresponding to one path down an
    adaptive Decision Tree. This is a classical calculation using the
    known hypothesis states.
    """
    num_qubits = hyp_circuit_0.num_qubits
    hyp_sv_0 = Statevector.from_instruction(hyp_circuit_0)
    hyp_sv_1 = Statevector.from_instruction(hyp_circuit_1)

    final_basis_gates = [Id] * num_qubits
    path_projectors = [Id] * num_qubits

    for i in range(num_qubits):
        v0 = get_conditional_reduced_bloch_vector(hyp_sv_0, i, path_projectors)
        v1 = get_conditional_reduced_bloch_vector(hyp_sv_1, i, path_projectors)
        
        v_b = np.cross(v0, v1)
        if np.isclose(np.linalg.norm(v_b), 0):
            v_b = np.cross(v0, [0,0,1]) if not np.allclose(np.abs(v0), [0,0,1]) else np.array([1,0,0])
        
        basis_gate = get_basis_from_bloch_vector(v_b)
        final_basis_gates[i] = basis_gate
        
        # Assume the '0' outcome path to update projectors for the next step
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

def compute_hyp_prime(hyp_circuit, k_idx, conditioning_projectors, oracle):
    """
    Computes the state of the k-th qubit |hyp'> conditioned on other
    measurement outcomes, as described in Footnote 6 of the paper.
    This is done by classically reconstructing the state from oracle calls.
    """
    # First, find the total probability of the conditioning event, for normalization
    prob_condition = oracle(hyp_circuit, conditioning_projectors)
    
    if np.isclose(prob_condition, 0):
        # The conditional state is undefined, return a default.
        return np.array([1, 0])

    p_plus = np.outer(np.array([1,1])/np.sqrt(2), np.array([1,1])/np.sqrt(2))
    p_i = np.outer(np.array([1,1j])/np.sqrt(2), np.array([1,-1j])/np.sqrt(2))
    p_0 = np.array([[1,0],[0,0]], dtype=complex)

    # --- Calculate conditional expectation values ---
    proj_z_list = conditioning_projectors.copy(); proj_z_list[k_idx] = p_0
    exp_z = (2 * oracle(hyp_circuit, proj_z_list) / prob_condition) - 1

    proj_x_list = conditioning_projectors.copy(); proj_x_list[k_idx] = p_plus
    exp_x = (2 * oracle(hyp_circuit, proj_x_list) / prob_condition) - 1

    proj_y_list = conditioning_projectors.copy(); proj_y_list[k_idx] = p_i
    exp_y = (2 * oracle(hyp_circuit, proj_y_list) / prob_condition) - 1

    bloch_vector = np.array([exp_x, exp_y, exp_z])
    
    # --- Reconstruct statevector from Bloch vector ---
    norm = np.linalg.norm(bloch_vector)
    if norm > 1.0: bloch_vector = bloch_vector / norm
        
    x, y, z = bloch_vector
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    
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
    (Simulation Shortcut) Calculates the ideal post-measurement density matrix
    of the lab state's k-th qubit by using its statevector. This avoids
    simulating a full tomography protocol.
    """
    full_projector = reduce(np.kron, reversed(conditioning_projectors))
    lab_state_full = Statevector.from_instruction(lab_circuit)
    post_lab_unnormalized = full_projector @ lab_state_full.data
    lab_norm = np.linalg.norm(post_lab_unnormalized)

    if np.isclose(lab_norm, 0):
        # This outcome was impossible for the lab state. Return None.
        return None

    post_lab_full = Statevector(post_lab_unnormalized / lab_norm)
    
    q_args = list(range(lab_circuit.num_qubits))
    q_args.remove(k_idx)
    
    return partial_trace(post_lab_full, q_args).data