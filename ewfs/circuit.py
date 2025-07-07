"""This module contains utility functions for constructing quantum circuits."""

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers.fake_provider import GenericBackendV2
from ewfs.ghz import GHZCircuitBuilder


def extract_qiskit_indices_by_prefix(qc: QuantumCircuit, prefix: str, exact_match: bool = False) -> list[int]:
    """Extracts qubit indices of a qiskit circuit based on an exact or prefix match.

    Args:
        qc: The quantum circuit.
        prefix: The register prefix (e.g., "Charlie", "Debbie", "Debbie_flag").
        exact_match: If True, matches only exact register names (no substrings).

    Returns:
        List of qubit indices matching the given register name.
    """
    indices = []
    for idx, qubit in enumerate(qc.qubits):
        reg_name = qubit._register.name

        if exact_match:
            if reg_name == prefix:
                indices.append(idx)
        else:
            # Ensure only "Debbie_*" is matched, but NOT "Debbie_flag_*"
            if reg_name.startswith(prefix) and not reg_name.startswith(f"{prefix}_flag"):
                indices.append(idx)

    return indices


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


def ghz_ewfs_circuit(qc: QuantumCircuit, charlie_size: int, debbie_size: int) -> QuantumCircuit:
    """EWFS with GHZ friend states.

    Constructs a quantum circuit for an extended Wigner's friend scenario (EWFS) with
    GHZ states prepared for Charlie and Debbie, and includes control operations from Alice to Charlie
    and Bob to Debbie.

    This function combines an initial circuit (`qc`) with GHZ circuits for two friends, Charlie and Debbie.
    Control operations (CNOT gates) from Alice to Charlie and Bob to Debbie are applied
    directly after their respective GHZ circuits are composed.

    Args:
        qc (QuantumCircuit): The initial quantum circuit containing operations for Alice and Bob.
        charlie_size (int): The number of qubits in Charlie's subsystem.
        debbie_size (int): The number of qubits in Debbie's subsystem.

    Returns:
        QuantumCircuit: A combined quantum circuit with the initial operations, control gates, and GHZ circuits
                        for Charlie and Debbie.

    Circuit Description:
    - The input circuit (`qc`) is composed first and includes operations for Alice and Bob.
    - A CNOT gate is added from Alice to the first qubit in Charlie's subsystem.
    - A GHZ circuit is composed for Charlie's subsystem.
    - A CNOT gate is added from Bob to the first qubit in Debbie's subsystem.
    - A GHZ circuit is composed for Debbie's subsystem.
    - Barriers are added for visual clarity and to separate operations logically in the circuit diagram.
    """
    # Compose Charlie's GHZ circuit
    charlie_ghz_qc = ibm_fez_ghz_circuit(charlie_size, friend_label="Charlie")
    
    debbie_ghz_qc = ibm_fez_ghz_circuit(debbie_size, friend_label="Debbie") if debbie_size > 0 else QuantumCircuit()
    circuits = [qc, charlie_ghz_qc, debbie_ghz_qc]

    combined_circuit = QuantumCircuit(
        *[qreg for circuit in circuits for qreg in circuit.qregs],
        *[creg for circuit in circuits for creg in circuit.cregs],
    )

    # Compose `qc` first (it includes Alice and Bob)
    combined_circuit.compose(qc, inplace=True)
    combined_circuit.barrier()

    # Add CNOT gates before Charlie's GHZ circuit
    alice_qubit = combined_circuit.qubits[0]  # Assuming Alice is the first qubit in qc
    bob_qubit = combined_circuit.qubits[1]  # Assuming Bob is the second qubit in qc
    charlie_0_qubit = combined_circuit.qubits[len(qc.qubits)]  # First qubit of Charlie (before GHZ circuit)
    combined_circuit.cx(alice_qubit, charlie_0_qubit)  # CNOT from Alice to Charlie_0
    
    combined_circuit.barrier()

    # Offset for Charlie registers
    charlie_qubit_offset = len(qc.qubits)
    combined_circuit.compose(
        charlie_ghz_qc,
        qubits=range(charlie_qubit_offset, charlie_qubit_offset + charlie_ghz_qc.num_qubits),
        inplace=True,
    )
    combined_circuit.barrier()

    if debbie_ghz_qc:
        debbie_0_qubit = combined_circuit.qubits[len(qc.qubits) + len(charlie_ghz_qc.qubits)]  # First qubit of Debbie
        combined_circuit.cx(bob_qubit, debbie_0_qubit)  # CNOT from Bob to Debbie_0
        # Offset for Debbie registers
        debbie_qubit_offset = charlie_qubit_offset + len(charlie_ghz_qc.qubits)
        combined_circuit.compose(
            debbie_ghz_qc,
            qubits=range(debbie_qubit_offset, debbie_qubit_offset + debbie_ghz_qc.num_qubits),
            inplace=True,
        )

    return combined_circuit


def ibm_fez_ghz_circuit(friend_size: int, num_ghz_qubits: int = 54, friend_label: str = "Charlie") -> QuantumCircuit:
    """GHZ circuit for the IBM FEZ backend with labeled qubits.

    Args:
        friend_size: The number of friend qubits to label.
        num_ghz_qubits: Total number of qubits in the GHZ circuit.
        friend_label: Prefix label for the friend qubits (default: 'Charlie').

    Returns:
        QuantumCircuit: A quantum circuit with labeled friend qubits and ancilla qubits.
    """
    if friend_size > num_ghz_qubits:
        raise ValueError(f"{friend_size=} cannot exceed {num_ghz_qubits=}.")

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
    circuit: QuantumCircuit = build_result["circuit_with_flags"]

    # If there is just one element, append a zero to the label.
    friend_register_label = f"{friend_label}_0" if friend_size == 1 else friend_label

    # Create the friend register with the adjusted label.
    friend_register = QuantumRegister(friend_size, name=friend_register_label)

    # Create an ancilla register for the remaining qubits.
    num_ancilla_qubits = circuit.num_qubits - friend_size
    if num_ancilla_qubits < 0:
        raise ValueError("Number of ancilla qubits cannot be negative.")

    friend_ancilla_label = f"{friend_label}_flag_0" if num_ancilla_qubits == 1 else f"{friend_label}_flag"
    ancilla_register = QuantumRegister(num_ancilla_qubits, name=friend_ancilla_label)

    # Create a new circuit with both registers.
    new_circuit = QuantumCircuit(friend_register, ancilla_register)

    # Compose the original circuit into the new circuit.
    if len(new_circuit.qubits) != circuit.num_qubits:
        raise ValueError("Mismatch in the number of qubits between the circuits.")
    new_circuit.compose(circuit, qubits=list(range(circuit.num_qubits)), inplace=True)

    return new_circuit
