from ewfs.ghz import GHZ, FidelityEstimator

from ewfs.layout import (
    build_tree_by_node_count,
    get_leaf_nodes,
    get_all_nodes,
    find_optimal_k_pairs,
    create_coupling_map_from_selection,
)

from qiskit_aer import AerSimulator

import matplotlib.pyplot as plt

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    # pauli_error,
)

import numpy as np

single_qubit_error_rate  = 0.0005 # 99.95%
two_qubit_error_rate = 0.01 # 99.5%
bit_flip_error_rate = 0.01

noise_model = NoiseModel()

noise_model.add_all_qubit_quantum_error(depolarizing_error(single_qubit_error_rate, 1), ['u2', 'u3', 'x'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error_rate, 2), ['cx'])

# Add bit flip errors (X gates) to all qubits after each gate operation
# x_error = pauli_error([('X', bit_flip_error_rate), ('I', 1 - bit_flip_error_rate)])
# # Apply X error to all single-qubit gates
# noise_model.add_all_qubit_quantum_error(x_error, ['u2', 'u3', 'x'])
# # Apply X error to all two-qubit gates
# x_error_2q = pauli_error([('XX', bit_flip_error_rate), ('II', 1 - bit_flip_error_rate)])
# noise_model.add_all_qubit_quantum_error(x_error_2q, ['cx'])


# Run simulations across different total node counts and selected pair counts
total_node_counts = [7, 8, 9, 10, 11]
num_pairs_to_select = [0, 1, 2]

results = np.zeros((len(total_node_counts), len(num_pairs_to_select)))

for total_node_count in total_node_counts:
    for num_pairs in num_pairs_to_select:
        print(f"\nRunning for Total Nodes: {total_node_count}, Selected Pairs: {num_pairs}")

        root_node = build_tree_by_node_count(total_node_count)
        all_nodes = get_all_nodes(root_node)

        leaves = get_leaf_nodes(root_node)

        selected_pairs_k, ids_k = find_optimal_k_pairs(root_node, num_pairs)
        ratio_k = len(ids_k) / len(all_nodes) if all_nodes else 0

        coupling_map, nodes, pair_nodes = create_coupling_map_from_selection(root_node, selected_pairs_k)
        layout = nodes

        flags = pair_nodes
        print(f"Coverage Ratio: {ratio_k:.2%}")

        backend = AerSimulator(noise_model=noise_model)

        ghz = GHZ(
            coupling_map=coupling_map,
            layout=layout,
            flag_qubits=flags)


        fidelityEstimator = FidelityEstimator(ghz=ghz)

        all_counts = fidelityEstimator.run_measurements(backend=backend, shots=100_000)
        f = fidelityEstimator.estimate_fidelity(all_counts)

        results[total_node_counts.index(total_node_count), num_pairs_to_select.index(num_pairs)] = f

for i, total_node_count in enumerate(total_node_counts):
    plt.figure()
    plt.plot(num_pairs_to_select, results[i], marker='o')
    plt.title(f'Fidelity vs Number of Selected Pairs (Total Nodes: {total_node_count})')
    plt.xlabel('Number of Selected Pairs')
    plt.ylabel('Fidelity')
    plt.grid(True)
    plt.savefig(f'fidelity_vs_pairs_nodes_{total_node_count}.png')
    plt.close()