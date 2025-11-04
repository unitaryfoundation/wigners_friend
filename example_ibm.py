from ewfs.ewfs import (
    EWFS,
    decode_results,
    post_select_results,
    semi_brukner_value,
)
from ewfs.layout import (
    build_tree_by_node_count,
    get_leaf_nodes,
    get_all_nodes,
    find_optimal_k_pairs,
    create_coupling_map_from_selection,
)

from qiskit_aer import Aer
from qiskit_ibm_runtime.fake_provider import FakeFez


PEEK = 'peek'
REVERSE_1 = 'reverse_1'
REVERSE_2 = 'reverse_2'

hardware_connectivity = 'fully_connected'

if hardware_connectivity == 'fez':
    layout = [53, 52, 51, 50, 49, 48, 47, 46, 57, 67, 68, 69, 70, 71, 58]
    flags = [58]
    coupling_map = FakeFez().coupling_map
elif hardware_connectivity == 'fully_connected':
    TOTAL_NODE_COUNT = 7
    NUM_PAIRS_TO_SELECT = 2

    root_node = build_tree_by_node_count(TOTAL_NODE_COUNT)
    all_nodes = get_all_nodes(root_node)

    leaves = get_leaf_nodes(root_node)

    selected_pairs_k, ids_k = find_optimal_k_pairs(root_node, NUM_PAIRS_TO_SELECT)
    ratio_k = len(ids_k) / len(all_nodes) if all_nodes else 0

    coupling_map, nodes, pair_nodes = create_coupling_map_from_selection(root_node, selected_pairs_k)
    coupling_map.add_node(0, [1])
    layout = [0]+nodes

    flags = pair_nodes
    print(f"Coverage Ratio: {ratio_k:.2%}")


backend = Aer.get_backend("aer_simulator")
settings = [(PEEK, REVERSE_1), (PEEK, REVERSE_2), (REVERSE_2, REVERSE_1), (REVERSE_2, REVERSE_2)]

semi_brukner_EWFS = [EWFS(
    alice_setting=s1,
    bob_setting=s2,
    strategy='majority_vote',
    coupling_map=coupling_map,
    layout=layout,
    flag_qubits=flags,
    shots=10_000,
    backend=backend) for s1, s2 in settings]


results = {}
for s in semi_brukner_EWFS:
    results[(s.alice_setting, s.bob_setting)] = s.results[(s.alice_setting, s.bob_setting)]

post_selected_results = post_select_results(results, len(flags))

charlie_size = len(layout)-len(flags)-2
decoded_results = decode_results(post_selected_results, charlie_size)

probs = {}
for setting, res in decoded_results.items():
    probs[setting] = {key: value/sum(list(res.values())) for key, value in res.items()}

print(semi_brukner_value(probs=probs))
