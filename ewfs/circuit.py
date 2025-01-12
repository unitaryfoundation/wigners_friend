"""This module contains utility functions for constructing quantum circuits."""

from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from ewfs.ghz import GHZCircuitBuilder


def cnot_ladder(
    qc: QuantumCircuit,
    observer: int,
    friend_qubit: int,
    friend_size: int,
    reverse: bool = False,
    internal_copy: bool = False,
) -> None:
    """Constructs a CNOT ladder circuit between an observer and their friend subsystem.

    Args:
        qc: The quantum circuit to apply the CNOT ladder to.
        observer: The qubit index of the observer.
        friend_qubit: The starting qubit index of the friend's subsystem.
        friend_size: The number of qubits in the friend's subsystem.
        reverse: Whether the ladder is constructed in reverse order.
        internal_copy: Whether the internal state of the friend's qubits should be copied.

    Raises:
        ValueError: If friend_size is less than 1.
    """
    if friend_size < 1:
        raise ValueError("friend_size must be at least 1.")

    if internal_copy:
        if reverse:
            for i in range(friend_size - 1):
                qc.cx(friend_qubit + friend_size - 2 - i, friend_qubit + friend_size - 1 - i)
            qc.cx(observer, friend_qubit)
        else:
            qc.cx(observer, friend_qubit)
            for i in range(friend_size - 1):
                qc.cx(friend_qubit + i, friend_qubit + i + 1)
    else:
        if reverse:
            for i in range(friend_size):
                qc.cx(observer, friend_qubit + friend_size - 1 - i)
        else:
            for i in range(friend_size):
                qc.cx(observer, friend_qubit + i)


def ibm_fez_ghz_circuit(friend_size: int, num_ghz_qubits: int = 30) -> QuantumCircuit:
    """GHZ circuit for the IBM FEZ backend.

    Note: This circuit is specific to the IBM FEZ backend and coupling map.
    """
    backend = GenericBackendV2(num_qubits=156)
    if num_ghz_qubits == 30:
        qubits_to_remove = [56]
        start_qubit = 67
        flags_physical = [56]

    elif num_ghz_qubits == 54:
        qubits_to_remove = [56, 58, 77, 78, 37, 38, 59, 20, 40, 60, 80, 34, 16, 18, 96, 94, 91]
        start_qubit = 57
        flags_physical = [
            56,
            58,
            77,
            38,
            37,
            59,
        ]

    elif num_ghz_qubits == 70:
        qubits_to_remove = [56, 58, 77, 78, 37, 38, 59, 20, 40, 60, 80, 34, 16, 18, 96, 97, 110]
        start_qubit = 67
        flags_physical = [56, 58, 77, 78, 38, 37, 59, 113]

    elif num_ghz_qubits == 75:
        qubits_to_remove = [56, 58, 77, 78, 37, 38, 59, 20, 40, 60, 80, 34, 16, 18, 96, 97, 110, 98, 39, 32, 16, 98]
        start_qubit = 67
        flags_physical = [56, 58, 77, 78, 38, 37, 59, 113]

    elif num_ghz_qubits == 80:
        qubits_to_remove = [56, 58, 77, 78, 37, 38, 59, 20, 40, 60, 80, 34, 16, 18, 96, 97, 110, 98, 39, 32, 16, 98, 18]
        start_qubit = 57
        flags_physical = [56, 58, 77, 78, 38, 37, 59, 16, 98, 18]

    ghz_builder = GHZCircuitBuilder(
        coupling_map=backend.coupling_map,
        start_qubit=start_qubit,
        num_ghz_qubits=friend_size,
        qubits_to_remove=qubits_to_remove,
        flags_physical=flags_physical,
    )
    build_result = ghz_builder.build()

    return build_result["circuit_with_flags"]
