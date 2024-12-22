from qiskit import QuantumCircuit


def cnot_ladder(qc: QuantumCircuit, observer: int, friend_qubit: int, friend_size: int, reverse: bool = False, internal_copy: bool = False):
    """Constructs a CNOT ladder circuit between an observer and their friend subsystem.

    Args:
        qc (QuantumCircuit): The quantum circuit to apply the CNOT ladder to.
        observer (int): The qubit index of the observer.
        friend_qubit (int): The starting qubit index of the friend's subsystem.
        friend_size (int): The number of qubits in the friend's subsystem.
        reverse (bool): Whether the ladder is constructed in reverse order.
        internal_copy (bool): Whether the internal state of the friend's qubits should be copied.

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
