import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

# Import the functions to be tested from the main file
from utils import (
    get_statevector_from_projector,
    oracle_access_to_hyp,
    get_reduced_bloch_vector,
    get_basis_from_bloch_vector,
    compute_dt_basis,
    are_gates_phase_equivalent,
    Id
)

from main import (
    certify_algorithm_1,
    compute_hyp_prime,
)

class TestCertificationAlgorithm(unittest.TestCase):

    def test_get_statevector_from_projector(self):
        # Test for |0>
        p0 = np.array([[1, 0], [0, 0]], dtype=complex)
        v0 = get_statevector_from_projector(p0)
        self.assertTrue(np.allclose(v0, [1, 0]))

        # Test for |+>
        p_plus_vec = np.array([1, 1]) / np.sqrt(2)
        p_plus = np.outer(p_plus_vec, p_plus_vec.conj())
        v_plus = get_statevector_from_projector(p_plus)
        self.assertTrue(np.allclose(v_plus, p_plus_vec))

    def test_classical_oracle(self):
        # Bell state |00> + |11>
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        p0 = np.array([[1, 0], [0, 0]], dtype=complex)
        p1 = np.array([[0, 0], [0, 1]], dtype=complex)
        
        # Prob of measuring |00> should be 0.5
        prob = oracle_access_to_hyp(qc, [p0, p0])
        self.assertAlmostEqual(prob, 0.5)

        # Prob of measuring |01> should be 0.0
        prob = oracle_access_to_hyp(qc, [p0, p1])
        self.assertAlmostEqual(prob, 0.0)

    def test_get_reduced_bloch_vector(self):
        # Test with |0+> state
        qc = QuantumCircuit(2)
        qc.h(1) # Apply H to qubit 1
        
        # Test qubit 0, should be in state |0> -> [0, 0, 1]
        v0 = get_reduced_bloch_vector(qc, 0, oracle_access_to_hyp)
        self.assertTrue(np.allclose(v0, [0, 0, 1], atol=1e-9))
        
        # Test qubit 1, should be in state |+> -> [1, 0, 0]
        v1 = get_reduced_bloch_vector(qc, 1, oracle_access_to_hyp)
        self.assertTrue(np.allclose(v1, [1, 0, 0], atol=1e-9))

    def test_compute_hyp_prime(self):
        # Prepare a product state |+0>
        qc = QuantumCircuit(2)
        qc.h(0)
        
        # Condition on measuring qubit 0 as |+>. The state of qubit 1 should be |0>.
        p_plus = np.outer(np.array([1,1])/np.sqrt(2), np.array([1,1])/np.sqrt(2))
        conditioning_projectors = [p_plus, Id]
        
        # We are asking for the state of qubit 1 (k_idx=1)
        hyp_prime = compute_hyp_prime(qc, 1, conditioning_projectors, oracle_access_to_hyp)
        
        # The resulting state should be |0>
        self.assertTrue(np.allclose(hyp_prime, [1, 0]))

    def test_certify_algorithm_statistics(self):
        """
        Tests the statistical behavior of the main certification algorithm.
        This test now uses a less symmetric case to avoid blind spots.
        """
        n_qubits = 2
        
        # Hypothesis: |00> state
        hyp_qc = QuantumCircuit(n_qubits)
        
        # Lab prepares the same state
        lab_qc_correct = hyp_qc.copy()
        
        # Lab prepares an orthogonal state |11>
        lab_qc_wrong = QuantumCircuit(n_qubits)
        lab_qc_wrong.x(0)
        lab_qc_wrong.x(1)

        n_runs = 100
        
        # Test 1: Correct state should be accepted with high probability
        correct_results = [certify_algorithm_1(hyp_qc, lab_qc_correct) for _ in range(n_runs)]
        accept_rate_correct = correct_results.count("ACCEPT") / n_runs
        print(f"\n[Test] Acceptance rate for correct state (|00> vs |00>): {accept_rate_correct:.2f}")
        self.assertGreater(accept_rate_correct, 0.85, "Correct state should have high acceptance rate")

        # Test 2: Orthogonal state should be rejected with high probability
        wrong_results = [certify_algorithm_1(hyp_qc, lab_qc_wrong) for _ in range(n_runs)]
        accept_rate_wrong = wrong_results.count("ACCEPT") / n_runs
        print(f"[Test] Acceptance rate for orthogonal state (|00> vs |11>): {accept_rate_wrong:.2f}")
        self.assertLess(accept_rate_wrong, 0.65, "Orthogonal state should have low acceptance rate")

if __name__ == '__main__':
    unittest.main()

