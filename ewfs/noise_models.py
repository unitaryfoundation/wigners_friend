"""Custom noise models for simulator-based experiments."""

from qiskit import QuantumCircuit

from qiskit_aer.noise import (
    depolarizing_error,
    NoiseModel,
    pauli_error,
)


def get_n_qubit_gateset(circ: QuantumCircuit, num_qubits: int) -> set[str]:
    """Extracts the set of gates of size num_qubits from a quantum circuit.
    Args:
        circ: Quantum circuit.
        num_qubits: The size of the gates to get.
    Returns:
        Set of string names of <num_qubits>-qubit gates.
    """
    return {
        instr.operation.name for instr in circ.data 
        if instr.operation.num_qubits == num_qubits and instr.operation.name != "measure"
    }


def depolarizing_model(circ: QuantumCircuit, 
                       single_qubit_error_rate: float=0.01, 
                       two_qubit_error_rate:float=0.03, 
                       meas_error_rate : float | None = None) -> NoiseModel:
    """Depolarizing noise model with error rates applied to the single and two-qubit gates of the circuit.
    Args:
        circ: Quantum circuit to apply the model to.
        single_qubit_error_rate: Error rate for a single qubit gate.
        two_qubit_error_rate: Error rate for a two qubit gate.
        meas_error_rate (optional): Bit flip error for a measurement.
    Returns:
        Depolarizing noise model.
    """
    single_qubit_gates = get_n_qubit_gateset(circ, num_qubits=1)
    two_qubit_gates = get_n_qubit_gateset(circ, num_qubits=2)
    
    noise_model = NoiseModel()
    if meas_error_rate:
        error_meas = pauli_error([("X", meas_error_rate), ("I", 1 - meas_error_rate)])
        noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    noise_model.add_all_qubit_quantum_error(depolarizing_error(single_qubit_error_rate, 1), list(single_qubit_gates))
    noise_model.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error_rate, 2), list(two_qubit_gates))

    return noise_model
