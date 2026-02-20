"""Example of running EWFS on Quantinuum devices via qNexus platform."""

import os
import uuid
from datetime import datetime, timezone

import qnexus as qnx
from dotenv import load_dotenv
from pytket.extensions.qiskit import qiskit_to_tk
from qnexus.models.language import Language

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

# Load environment variables from .env file
load_dotenv()

# Authenticate with qNexus if credentials are provided in .env
qnexus_email = os.getenv("QNEXUS_EMAIL")
qnexus_password = os.getenv("QNEXUS_PASSWORD")

if qnexus_email and qnexus_password:
    print("Authenticating with qNexus using credentials from .env...")
    try:
        qnx.auth._request_tokens(user=qnexus_email, pwd=qnexus_password)
        print("Authentication successful!")
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("Please run 'uv run qnx login' or check your credentials in .env")
        exit(1)
else:
    print("No credentials in .env. Using existing qNexus authentication.")
    print("If authentication fails, run: uv run qnx login")


PEEK = 'peek'
REVERSE_1 = 'reverse_1'
REVERSE_2 = 'reverse_2'

# Quantinuum devices are fully connected, so we create a simple fully connected topology
TOTAL_NODE_COUNT = 8
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

# Configuration for Quantinuum
# Environment variables are loaded from .env file
# Required: QNEXUS credentials (set via qnexus login or in .env)
# Optional: QUANTINUUM_DEVICE (defaults to H1-1LE emulator)
# Available devices: H1-1LE, H2-1LE, H1-Emulator, H2-Emulator

DEVICE_NAME = os.getenv("QUANTINUUM_DEVICE", "H1-1LE")  # Default to H1-1LE emulator
SHOTS = int(os.getenv("QUANTINUUM_SHOTS", "10000"))
OPTIMIZATION_LEVEL = int(os.getenv("QUANTINUUM_OPT_LEVEL", "1"))
PROJECT_NAME = os.getenv("QUANTINUUM_PROJECT", "wigners-friend")

settings = [(PEEK, REVERSE_1), (PEEK, REVERSE_2), (REVERSE_2, REVERSE_1), (REVERSE_2, REVERSE_2)]

# Create EWFS instances (backend=None since we'll run manually on Quantinuum)
semi_brukner_EWFS = [
    EWFS(
        alice_setting=s1,
        bob_setting=s2,
        strategy='majority_vote',
        coupling_map=coupling_map,
        layout=layout,
        flag_qubits=flags,
        shots=SHOTS,
        backend=None,  # We'll run manually on Quantinuum
    )
    for s1, s2 in settings
]


def run_circuit_on_quantinuum(circuit, device_name, shots, optimization_level, project_name):
    """Run a single circuit on Quantinuum via qNexus."""
    # Get or create project
    project = qnx.projects.get_or_create(name=project_name)

    # Configure backend
    backend_config = qnx.QuantinuumConfig(device_name=device_name)

    # Convert Qiskit circuit to pytket
    pytket_circuit = qiskit_to_tk(circuit)

    # Generate unique name for this run
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    circuit_name = f"ewfs-circuit-{timestamp}-{run_id}"

    # Upload circuit
    circuit_ref = qnx.circuits.upload(
        name=circuit_name,
        circuit=pytket_circuit,
        project=project,
    )

    # Compile circuit
    compile_name = f"ewfs-compile-{timestamp}-{run_id}"
    compile_job = qnx.start_compile_job(
        programs=[circuit_ref],
        name=compile_name,
        optimisation_level=optimization_level,
        backend_config=backend_config,
        project=project,
    )

    # Wait for compilation to complete
    compile_job_id = getattr(compile_job, "id", getattr(compile_job, "job_id", str(compile_job)))
    print(f"Compiling circuit (job {compile_job_id})...")
    qnx.jobs.wait_for(compile_job)

    # Get compiled circuit
    compiled_refs = [item.get_output() for item in qnx.jobs.results(compile_job)]

    # Execute circuit
    execute_name = f"ewfs-execute-{timestamp}-{run_id}"
    execute_job = qnx.start_execute_job(
        programs=compiled_refs,
        name=execute_name,
        n_shots=[shots],
        backend_config=backend_config,
        project=project,
        language=Language.QIR,
    )

    # Wait for execution to complete
    execute_job_id = getattr(execute_job, "id", getattr(execute_job, "job_id", str(execute_job)))
    print(f"Executing circuit on {device_name} (job {execute_job_id})...")
    qnx.jobs.wait_for(execute_job)

    # Get results
    results = qnx.jobs.results(execute_job)
    if not results:
        raise RuntimeError(f"No results available for job {execute_job_id}")

    # Download and format counts
    counts = results[0].download_result().get_counts()

    # Convert pytket counts format to Qiskit string format
    # pytket returns counts as {(q0, q1, q2, ...): count}
    # Qiskit expects {"...q2q1q0": count} (reversed order)
    # So we need to reverse the bitstrings to match Qiskit convention
    normalized_counts = {"".join(map(str, reversed(k))): v for k, v in counts.items()}

    print(f"Execution complete. Retrieved {sum(normalized_counts.values())} shots.")

    return normalized_counts


# Run experiments on Quantinuum
print(f"\nRunning EWFS experiments on {DEVICE_NAME}...")
results = {}

for i, ewfs in enumerate(semi_brukner_EWFS):
    setting_label = f"{ewfs.alice_setting}, {ewfs.bob_setting}"
    print(f"\n[{i+1}/{len(semi_brukner_EWFS)}] Running setting: ({setting_label})")

    # Get the circuit from EWFS instance
    circuit = ewfs.circuit

    # Run on Quantinuum
    counts = run_circuit_on_quantinuum(
        circuit=circuit,
        device_name=DEVICE_NAME,
        shots=SHOTS,
        optimization_level=OPTIMIZATION_LEVEL,
        project_name=PROJECT_NAME,
    )

    results[(ewfs.alice_setting, ewfs.bob_setting)] = counts

# Post-process results (same as IBM example)
print("\nPost-processing results...")
post_selected_results = post_select_results(results, len(flags))

charlie_size = len(layout) - len(flags) - 2
decoded_results = decode_results(post_selected_results, charlie_size)

probs = {}
for setting, res in decoded_results.items():
    probs[setting] = {key: value / sum(list(res.values())) for key, value in res.items()}

print("\n" + "="*60)
print("SEMI-BRUKNER VALUE:")
print("="*60)
print(semi_brukner_value(probs=probs))
print("="*60)
