from ewfs.ghz import GHZ, post_select_results
from compressed_sensing_ghz import CompressedSensingEstimator

from ewfs.layout import (
    build_tree_by_node_count,
    get_leaf_nodes,
    get_all_nodes,
    find_optimal_k_pairs,
    create_coupling_map_from_selection,
)

from qiskit_aer import AerSimulator
from qiskit import transpile

import matplotlib.pyplot as plt

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
    ReadoutError,
)

import numpy as np

single_qubit_error_rate = 0.001  # 99.95%
two_qubit_error_rate = 0.01  # 99%
bit_flip_error_rate = 0.01

# Thermal relaxation parameters
t1 = 50e-6  # T1 time in seconds (50 microseconds)
t2 = 70e-6  # T2 time in seconds (must be <= 2*T1)
gate_time_1q = 50e-9  # single-qubit gate time (50 nanoseconds)
gate_time_2q = 300e-9  # two-qubit gate time (300 nanoseconds)

# Readout error parameters
readout_error_rate = 0.01


def backend_with_noise(noise_model: NoiseModel, noise="depolarizing"):
    if noise == "depolarizing":
        noise_model.add_all_qubit_quantum_error(depolarizing_error(single_qubit_error_rate, 1), ["u2", "u3", "x"])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error_rate, 2), ["cx"])

    elif noise == "bit_flip":
        x_error = pauli_error([("X", bit_flip_error_rate), ("I", 1 - bit_flip_error_rate)])
        noise_model.add_all_qubit_quantum_error(x_error, ["u2", "u3", "x"])
        x_error_2q = pauli_error([("XX", bit_flip_error_rate), ("II", 1 - bit_flip_error_rate)])
        noise_model.add_all_qubit_quantum_error(x_error_2q, ["cx"])

    elif noise == "thermal":
        # Single-qubit thermal relaxation
        error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        noise_model.add_all_qubit_quantum_error(error_1q, ["u2", "u3", "x"])

        # Two-qubit thermal relaxation (tensor product for each qubit)
        error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(thermal_relaxation_error(t1, t2, gate_time_2q))
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    elif noise == "thermal_with_readout":
        # Thermal relaxation
        error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        noise_model.add_all_qubit_quantum_error(error_1q, ["u2", "u3", "x"])

        error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(thermal_relaxation_error(t1, t2, gate_time_2q))
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

        # Readout error
        readout_err = ReadoutError(
            [[1 - readout_error_rate, readout_error_rate], [readout_error_rate, 1 - readout_error_rate]]
        )
        noise_model.add_all_qubit_readout_error(readout_err)

    backend = AerSimulator(noise_model=noise_model)
    return backend


# Example usage:
noise_model = NoiseModel()
noise = "depolarizing"  # Options: 'depolarizing', 'bit_flip', 'thermal', 'thermal_with_readout'
backend = backend_with_noise(noise_model, noise=noise)

# Run simulations across different total node counts and selected pair counts
total_node_counts = [9, 10, 11]
num_pairs_to_select = [0, 1, 2]

results = np.zeros((len(total_node_counts), len(num_pairs_to_select)))
pop_results = np.zeros((len(total_node_counts), len(num_pairs_to_select)))
coh_results = np.zeros((len(total_node_counts), len(num_pairs_to_select)))
coverage_ratios = np.zeros((len(total_node_counts), len(num_pairs_to_select)))


# Store dense simulation data for comparison
full_data_sim = {}

# Initialize CS Estimator
cs_estimator = CompressedSensingEstimator(max_freq_candidate=50)

for total_node_count in total_node_counts:
    for num_pairs in num_pairs_to_select:
        print(f"\nRunning for Total Nodes: {total_node_count}, Selected Pairs: {num_pairs}")

        root_node = build_tree_by_node_count(total_node_count)
        all_nodes = get_all_nodes(root_node)

        leaves = get_leaf_nodes(root_node)

        selected_pairs_k, ids_k = find_optimal_k_pairs(root_node, num_pairs)
        ratio_k = len(ids_k) / len(all_nodes) if all_nodes else 0

        # Store coverage ratio
        idx_n = total_node_counts.index(total_node_count)
        idx_p = num_pairs_to_select.index(num_pairs)
        coverage_ratios[idx_n, idx_p] = ratio_k

        coupling_map, nodes, pair_nodes = create_coupling_map_from_selection(root_node, selected_pairs_k)
        layout = nodes

        flags = pair_nodes
        print(f"Coverage Ratio: {ratio_k:.2%}")

        ghz = GHZ(coupling_map=coupling_map, layout=layout, flag_qubits=flags)

        circuit = ghz._initialize_circuit()
        print(f"depth: {circuit.depth()}")

        num_f = len(flags)

        # --- Compressed Sensing Strategy ---
        # 1. Generate Random Phases
        # M ~ 5 * ln(N), min 15
        M_samples = max(15, int(5 * np.log(total_node_count)))
        phases = np.sort(np.random.uniform(0, 2 * np.pi, M_samples))
        print(f"Using {M_samples} random phases for CS.")

        circuits = ghz.get_verification_circuits(method="parity_oscillation", phases=phases.tolist())

        # Save circuit as LaTeX (of the Z-basis circ)
        circuit_filename = f"circuit_nodes_{total_node_count}_pairs_{num_pairs}"
        # save_circuit_latex(circuits[0], circuit_filename, compile_pdf=True) # Optional, skipping for speed

        # Run Measurements
        res = backend.run(transpile(circuits, backend), shots=10_000).result()

        # 2. Compute Population (Z-basis, index 0)
        z_counts_raw = res.get_counts(0)
        z_counts = post_select_results(z_counts_raw, num_f)
        total_z = sum(z_counts.values())

        if total_z > 0:
            # P = P(0...0) + P(1...1)
            p0 = z_counts.get("0" * total_node_count, 0)
            p1 = z_counts.get("1" * total_node_count, 0)
            population = (p0 + p1) / total_z
        else:
            population = 0.0

        print(f"Population: {population:.4f}")

        # 3. Extract Parities
        measured_parities = []
        for k in range(1, len(circuits)):
            counts = post_select_results(res.get_counts(k), num_f)
            total = sum(counts.values())
            even = sum(c for b, c in counts.items() if b.count("1") % 2 == 0)
            parity_val = (2 * even - total) / total if total > 0 else 0
            measured_parities.append(parity_val)

        # 4. CS Estimation
        fit_result = cs_estimator.fit(phases, measured_parities)
        rec_n = fit_result["n"]
        rec_C = fit_result["C"]
        rec_theta = fit_result["theta"]

        print(f"CS Recovery: N={rec_n} (Expected {total_node_count}), C={rec_C:.4f}")

        # 5. Fidelity
        fidelity = (population + rec_C) / 2
        print(f"Estimated Fidelity: {fidelity:.4f}")

        results[idx_n, idx_p] = fidelity
        pop_results[idx_n, idx_p] = population
        coh_results[idx_n, idx_p] = rec_C

        # Plot Fit - DISABLED to save time/files
        # plt.figure(figsize=(10, 6))
        # plt.plot(phases, measured_parities, 'o', label='Measured Parity')
        # ...

print("\nSimulation Complete.")

# Plotting: Component Analysis (Population, Coherence, Fidelity)
print("\nGenerating Component Analysis Plots...")
for i, total_node_count in enumerate(total_node_counts):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Shared X-axis data
    x_vals = num_pairs_to_select

    # 1. Population
    ax = axes[0]
    ax.plot(x_vals, pop_results[i, :], "o-", color="forestgreen", linewidth=2, markersize=8)
    ax.set_title(f"Population (N={total_node_count})")
    ax.set_xlabel("Num Flag Pairs")
    ax.set_ylabel("Population")
    ax.set_xticks(x_vals)
    ax.grid(True, alpha=0.3)
    ax.margins(y=0.1)

    # 2. Coherence
    ax = axes[1]
    ax.plot(x_vals, coh_results[i, :], "o-", color="darkorange", linewidth=2, markersize=8)
    ax.set_title(f"Coherence (N={total_node_count})")
    ax.set_xlabel("Num Flag Pairs")
    ax.set_ylabel("Coherence")
    ax.set_xticks(x_vals)
    ax.grid(True, alpha=0.3)
    ax.margins(y=0.1)

    # 3. Fidelity
    ax = axes[2]
    ax.plot(x_vals, results[i, :], "o-", color="royalblue", linewidth=2, markersize=8)
    ax.set_title(f"Fidelity (N={total_node_count})")
    ax.set_xlabel("Num Flag Pairs")
    ax.set_ylabel("Fidelity")
    ax.set_xticks(x_vals)
    ax.grid(True, alpha=0.3)
    ax.margins(y=0.1)

    # Annotate Fidelity Plot with Coverage
    for j, pairs in enumerate(x_vals):
        cov = coverage_ratios[i, j]
        fid = results[i, j]
        axes[2].annotate(
            f"{cov:.1%}",
            (pairs, fid),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="darkred",
        )

    plt.tight_layout()
    save_path = f"components_vs_pairs_nodes_{total_node_count}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot: {save_path}")
    plt.close()
