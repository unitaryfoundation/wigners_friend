import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from qiskit_aer import AerSimulator
from qiskit import transpile
from ewfs.ghz import GHZ, post_select_results
from ewfs.layout import (
    build_tree_by_node_count,
    get_all_nodes,
    find_optimal_k_pairs,
    create_coupling_map_from_selection,
)


class CompressedSensingEstimator:
    def __init__(self, max_freq_candidate: int = 100, lasso_alpha: float = 0.1):
        """
        Initializes the estimator.

        Args:
            max_freq_candidate: Maximum frequency n to scan for.
            lasso_alpha: Regularization strength for Lasso.
        """
        self.max_freq = max_freq_candidate
        self.alpha = lasso_alpha
        self.n_candidates = np.arange(1, self.max_freq + 1)

    def fit(self, phases, signal_values):
        """
        Estimates the parameters (C, n, theta) from the given data.

        Args:
            phases (np.ndarray): Array of measurement phases.
            signal_values (np.ndarray): Array of measured signal values (real-valued parity).

        Returns:
            dict: Estimated parameters {'C': float, 'n': int, 'theta': float}
        """
        phases = np.array(phases)
        signal_values = np.array(signal_values)

        # Step 1: Support Detection (Find n) using Lasso
        # Construct dictionary matrix D
        # Columns are cos(n*phi) and -sin(n*phi) for all candidate n
        D = np.hstack([np.cos(np.outer(phases, self.n_candidates)), -np.sin(np.outer(phases, self.n_candidates))])

        lasso = Lasso(alpha=self.alpha)
        lasso.fit(D, signal_values)

        # Identify the index of the highest magnitude coefficient
        # Combine (a, b) coefficients for each frequency to get magnitude
        coef_len = len(self.n_candidates)
        # lasso.coef_ has size 2 * max_freq
        a_coeffs = lasso.coef_[:coef_len]
        b_coeffs = lasso.coef_[coef_len:]

        mag = np.sqrt(a_coeffs**2 + b_coeffs**2)
        best_candidate_idx = np.argmax(mag)
        recovered_n = self.n_candidates[best_candidate_idx]

        # Step 2: Parameter Refinement (Standard Least Squares)
        # We build a tiny matrix with only the TWO columns for the found n
        D_refine = np.column_stack([np.cos(recovered_n * phases), -np.sin(recovered_n * phases)])

        # Use Standard Linear Regression (no penalty, no bias)
        refiner = LinearRegression(fit_intercept=False)
        refiner.fit(D_refine, signal_values)
        a_final, b_final = refiner.coef_

        # Calculate Final Parameters
        final_C = np.sqrt(a_final**2 + b_final**2)
        final_theta = np.arctan2(b_final, a_final)

        return {"C": final_C, "n": recovered_n, "theta": final_theta}


if __name__ == "__main__":
    print("--- Running Compressed Sensing on Simulator Data ---")

    # Simulator Setup
    backend = AerSimulator()
    total_node_count = 7
    num_pairs = 0  # Clean GHZ

    # 1. Build GHZ State
    root_node = build_tree_by_node_count(total_node_count)
    all_nodes = get_all_nodes(root_node)
    selected_pairs_k, ids_k = find_optimal_k_pairs(root_node, num_pairs)
    coupling_map, nodes, pair_nodes = create_coupling_map_from_selection(root_node, selected_pairs_k)
    layout = nodes
    flags = pair_nodes

    ghz = GHZ(coupling_map=coupling_map, layout=layout, flag_qubits=flags)
    circuit = ghz._initialize_circuit()

    num_f = len(flags)

    # 2. Get Verification Circuits (Parity Oscillation)
    # Generate random phases
    num_phases = 10
    phases = np.random.uniform(0, 2 * np.pi, num_phases)
    print(f"Generated {len(phases)} phases.")

    circuits = ghz.get_verification_circuits(method="parity_oscillation", phases=phases)

    # 3. Run Simulation
    print("Running simulation...")
    res = backend.run(transpile(circuits, backend), shots=100_000).result()

    # 4. Process Results
    parities = []
    # Index 0 is Z-basis, Indices 1..len match phases
    for k in range(1, len(circuits)):
        counts = post_select_results(res.get_counts(k), num_f)
        total = sum(counts.values())
        even = sum(c for b, c in counts.items() if b.count("1") % 2 == 0)
        parity_val = (2 * even - total) / total if total > 0 else 0
        parities.append(parity_val)

    print(f"Captured {len(parities)} parity points.")

    # 5. Run Compressed Sensing Estiamtor
    estimator = CompressedSensingEstimator(max_freq_candidate=100)
    result = estimator.fit(phases, parities)

    print("--- Results ---")
    print(f"Expected n: {total_node_count}")
    print(f"Recovered:  n = {result['n']}")
    print(f"Amplitude:  C = {result['C']:.4f}")
    print(f"Phase:      θ = {result['theta']:.4f}")
