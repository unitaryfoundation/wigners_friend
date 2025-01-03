from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from ewfs.noise_models import get_n_qubit_gateset, depolarizing_model


def test_get_single_qubit_gates():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(0)
    assert get_n_qubit_gateset(qc, num_qubits=1) == {'h', 'x'}


def test_get_two_qubit_gates():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 1)
    qc.cz(0, 1)
    assert get_n_qubit_gateset(qc, num_qubits=2) == {'cx', 'cz'}
    assert get_n_qubit_gateset(qc, num_qubits=3) == set()


def test_get_depolarizing_model():
    n_qubits = 4
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for qubit in range(n_qubits - 1):
        qc.cx(qubit, qubit + 1)
    qc.measure_all()

    # lots of noise
    noise_model = depolarizing_model(circ=qc, single_qubit_error_rate=0.5, two_qubit_error_rate=0.5)
    # Create noisy simulator backend
    sim_noise = AerSimulator(noise_model=noise_model)
    # Run noisy simulation
    result_noise = sim_noise.run(qc).result()
    counts_bit_flip = result_noise.get_counts(0)
    # if no noise then this should be close to 500
    # noise should make it smaller
    assert counts_bit_flip['0000'] <= 300
