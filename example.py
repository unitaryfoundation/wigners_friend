from ewfs import (
    EWFS,
    decode_results,
    post_select_results,
    semi_brukner_value,
)
from qiskit_aer import Aer
from qiskit_ibm_runtime.fake_provider import FakeFez

coupling_map = FakeFez().coupling_map

def remove_elements(source_list, elements_to_remove):
    return [item for item in source_list if item not in elements_to_remove]

from qiskit_aer import Aer
from qiskit import transpile

PEEK = 'peek'
REVERSE_1 = 'reverse_1'
REVERSE_2 = 'reverse_2'

layout = [53, 52, 51, 50, 49, 48, 47, 46, 57, 67, 68, 69, 70, 71, 58]

flags = [58]

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