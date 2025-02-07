import pytest
from qiskit import QuantumCircuit, QuantumRegister
from ewfs.circuit import extract_qiskit_indices_by_prefix


@pytest.fixture
def sample_circuit():
    """Creates a sample QuantumCircuit for testing."""
    alice = QuantumRegister(1, name="Alice")
    bob = QuantumRegister(1, name="Bob")
    charlie = QuantumRegister(3, name="Charlie")
    charlie_flag = QuantumRegister(6, name="Charlie_flag")
    debbie = QuantumRegister(1, name="Debbie")
    debbie_flag = QuantumRegister(6, name="Debbie_flag")

    qc = QuantumCircuit(alice, bob, charlie, charlie_flag, debbie, debbie_flag)
    return qc


def test_extract_charlie_qubits(sample_circuit):
    """Test extracting Charlie's qubits (excluding Charlie_flag)."""
    indices = extract_qiskit_indices_by_prefix(sample_circuit, "Charlie", exact_match=True)
    assert indices == [2, 3, 4], f"Expected [2, 3, 4], got {indices}"


def test_extract_charlie_flag_qubits(sample_circuit):
    """Test extracting Charlie's flag qubits."""
    indices = extract_qiskit_indices_by_prefix(sample_circuit, "Charlie_flag", exact_match=True)
    assert indices == [5, 6, 7, 8, 9, 10], f"Expected [5, 6, 7, 8, 9, 10], got {indices}"


def test_extract_debbie_qubits(sample_circuit):
    """Test extracting Debbie's qubits (excluding Debbie_flag)."""
    indices = extract_qiskit_indices_by_prefix(sample_circuit, "Debbie", exact_match=True)
    assert indices == [11], f"Expected [11], got {indices}"


def test_extract_debbie_flag_qubits(sample_circuit):
    """Test extracting Debbie's flag qubits."""
    indices = extract_qiskit_indices_by_prefix(sample_circuit, "Debbie_flag", exact_match=True)
    assert indices == [12, 13, 14, 15, 16, 17], f"Expected [12, 13, 14, 15, 16, 17], got {indices}"
