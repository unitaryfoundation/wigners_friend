"""Custom noise models for simulator-based experiments."""

from qiskit import QuantumCircuit

from qiskit_aer.noise import (
    depolarizing_error,
    NoiseModel,
    pauli_error,
)


def get_single_qubit_gates(circ: QuantumCircuit) -> set[str]:
    """Extracts single-qubit gates from a quantum circuit.
    Args:
        circ: Quantum circuit.
    Returns:
        List of string names of single-qubit gates.
    """
    gate_set = set()
    for instruction in circ.data:
        op = instruction.operation
        if op.num_qubits == 1:
            if op.name != 'measure':
                gate_set.add(op.name)
    return gate_set


def get_two_qubit_gates(circ: QuantumCircuit) -> set[str]:
    """Extracts two-qubit gates from a quantum circuit.
    Args:
        circ: Quantum circuit.
    Returns:
        Set of string names of two-qubit gates.
    """
    gate_set = set()
    for instruction in circ.data:
        op = instruction.operation
        if op.num_qubits == 2:
            gate_set.add(op.name)
    return gate_set

def get_depolarizing_model(circ: QuantumCircuit, single_qubit_error_rate=0.01, two_qubit_error_rate=0.03) -> NoiseModel:
    """Depolarizing noise model with error rates applied to the single and two-qubit gates of the circuit.
    Args:
        circ: Quantum circuit.
    Returns:
        Depolarizing noise model.
    """
    single_qubit_gates = get_single_qubit_gates(circ)
    two_qubit_gates = get_two_qubit_gates(circ)

    return depolarizing_noise_model(single_qubit_error_rate, two_qubit_error_rate, single_qubit_gates, two_qubit_gates)


def depolarizing_noise_model(single_qubit_error_rate: float=0.01, two_qubit_error_rate: float=0.03, 
                             single_qubit_gates: set | None = None, two_qubit_gates: set | None = None) -> NoiseModel:
    """Defines an depolarizing noise model with one-qubit.

    Args:
        error: One-qubit gate error rate (default 1%).
    Returns:
        Depolarizing noise model.
    """
    single_qubit_gates = ["u1", "u2", "u3"] if None else single_qubit_gates
    two_qubit_gates = ["cx"] if None else two_qubit_gates

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(single_qubit_error_rate, 1), list(single_qubit_gates))
    noise_model.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error_rate, 2), list(two_qubit_gates))
    return noise_model


def bitflip_model(p: float) -> NoiseModel:
    """Bitflip noise model with majority vote approach.

    Args:
        p: Probability to flip.
    Returns:
        Bit-flip noise model.
    """
    # Example error probabilities.
    p_meas = p
    p_gate1 = p

    # QuantumError objects.
    error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])
    error_gate1 = pauli_error([("X", p_gate1), ("I", 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model.
    noise_bit_flip = NoiseModel()
    noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
    noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
    noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

    return noise_bit_flip
