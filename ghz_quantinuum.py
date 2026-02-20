
import os
import uuid
import json
from datetime import datetime, timezone
import numpy as np
# import matplotlib.pyplot as plt # Plotting removed as we don't have results right away
from dotenv import load_dotenv

import qnexus as qnx
from pytket.extensions.qiskit import qiskit_to_tk
from qnexus.models.language import Language
# from scipy.optimize import curve_fit # Not needed for submission only

from ewfs.ghz import (
    GHZ,
    FidelityEstimator,
    post_select_results,
    generate_trex_twirled_circuit,
    generate_trex_calibration_circuit,
)
from ewfs.layout import (
    build_tree_by_node_count,
    get_all_nodes,
    get_leaf_nodes,
    find_optimal_k_pairs,
    create_coupling_map_from_selection,
)

# Load environment variables
load_dotenv()

# Configuration
DEVICE_NAME = os.getenv("QUANTINUUM_DEVICE", "H2-1")
SYNTAX_CHECKER_DEVICE = "H2-1SC"
SHOTS = int(os.getenv("QUANTINUUM_SHOTS", "1000"))
OPTIMIZATION_LEVEL = int(os.getenv("QUANTINUUM_OPT_LEVEL", "1"))
PROJECT_NAME = os.getenv("QUANTINUUM_PROJECT", "wigners-friend")
METHOD = os.getenv("QUANTINUUM_METHOD", "parity_oscillation") # Options: 'dfe', 'parity_oscillation'
# Default to False for safety (Dry Run)
EXECUTE = True
POPULATION = True

# Quantum Error Mitigation (QEM) Configuration
ENABLE_DD = os.getenv("QUANTINUUM_ENABLE_DD", "false").lower() == "true"
DD_THRESHOLD = float(os.getenv("QUANTINUUM_DD_THRESHOLD", "0.03"))
ENABLE_TREX = os.getenv("QUANTINUUM_ENABLE_TREX", "false").lower() == "true"
TREX_NUM_RANDOMIZATIONS = int(os.getenv("QUANTINUUM_TREX_RANDOMIZATIONS", "32"))
READOUT_P0 = float(os.getenv("QUANTINUUM_READOUT_P0", "0.003"))  # for REM
READOUT_P1 = float(os.getenv("QUANTINUUM_READOUT_P1", "0.003"))  # for REM

# Authenticate with qNexus
qnexus_email = os.getenv("QNEXUS_EMAIL")
qnexus_password = os.getenv("QNEXUS_PASSWORD")

if qnexus_email and qnexus_password:
    print("Authenticating with qNexus using credentials from .env...")
    try:
        qnx.auth._request_tokens(user=qnexus_email, pwd=qnexus_password)
        print("Authentication successful!")
    except Exception as e:
        print(f"Authentication failed: {e}")
        # Continue anyway, as user might be already logged in
else:
    print("No credentials in .env. Using existing qNexus authentication.")

def get_interactions(pytket_circuit):
    """
    Extracts the set of 2-qubit interactions (edges) from a pytket circuit.
    Returns a set of tuples (q1, q2) where q1 < q2 (sorted) to be order-independent.
    """
    interactions = set()
    for cmd in pytket_circuit.get_commands():
        qubits = cmd.qubits
        if len(qubits) == 2:
            # Get indices. Quantinuum mapping usually preserves indices if no_opt is used
            # We assume logical-to-physical mapping preserves the node index logic 
            # or check physical qubit IDs.
            # pytket Qubit("q", 1) -> 1
            idx1 = qubits[0].index[0]
            idx2 = qubits[1].index[0]
            edge = tuple(sorted((idx1, idx2)))
            interactions.add(edge)
    return interactions

def verify_compilation(original_circuits, compiled_refs):
    """
    Verifies that the compiled circuits preserve the connectivity of the original circuits.
    Returns True if all pass, False otherwise.
    """
    print("Verifying compilation connectivity...")
    all_passed = True
    
    for i, (orig, ref) in enumerate(zip(original_circuits, compiled_refs)):
        # Download compiled circuit
        try:
            # Ref is likely a CircuitRef (output of compilation)
            # We need to download it
            # If ref is from get_output(), it is a CircuitRef.
            # We need to hydrate it to get properties or download.
            # qnx.circuits.get(id=ref.id) -> Circuit container -> .download_circuit()
            c_container = qnx.circuits.get(id=ref.id)
            compiled_circ = c_container.download_circuit()
        except Exception as e:
            print(f"  [Error] Failed to download compiled circuit {i}: {e}")
            all_passed = False
            continue

        # Get interactions
        orig_tk = qiskit_to_tk(orig)
        orig_edges = get_interactions(orig_tk)
        comp_edges = get_interactions(compiled_circ)
        
        missing = orig_edges - comp_edges
        extra = comp_edges - orig_edges
        
        if not missing and not extra:
            # print(f"  [Pass] Circuit {i}")
            pass
        else:
            print(f"  [FAIL] Circuit {i} connectivity mismatch!")
            if missing:
                print(f"    Missing: {missing}")
            if extra:
                print(f"    Extra: {extra}")
            all_passed = False
            
    if all_passed:
        print("Verification Successful: All circuits match connectivity.")
    else:
        print("Verification FAILED: Some circuits have connectivity mismatches.")
        
    return all_passed

def submit_batch_on_quantinuum(circuits, device_name, shots, optimization_level, project_name, execute=False, enable_dd=False, dd_threshold=0.03):
    """Submit a batch of circuits to Quantinuum via qNexus without waiting for results.

    Args:
        enable_dd: If True, enable dynamical decoupling via NEXUS compiler options.
        dd_threshold: DD threshold time in seconds (default 0.03).
    """
    if not circuits:
        return None, None

    project = qnx.projects.get_or_create(name=project_name)

    # Build compiler options for QEM
    compiler_options = {}
    if enable_dd:
        compiler_options["apply_DD"] = True
        compiler_options["DD_threshold_times"] = [dd_threshold]
        print(f"  Dynamical Decoupling ENABLED (threshold={dd_threshold}s)")

    backend_config = qnx.QuantinuumConfig(
        device_name=device_name,
        **({"compiler_options": compiler_options} if compiler_options else {}),
    )
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    batch_id = uuid.uuid4().hex[:6]
    
    # Upload circuits
    circuit_refs = []
    print(f"Uploading {len(circuits)} circuits...")
    for i, circuit in enumerate(circuits):
        circuit_name = f"ghz-circuit-{i}-{timestamp}-{batch_id}"
        pytket_circuit = qiskit_to_tk(circuit)
        ref = qnx.circuits.upload(
            name=circuit_name,
            circuit=pytket_circuit,
            project=project,
        )
        circuit_refs.append(ref)

    # Wait for compilation to complete (Compilation is fast usually, keeping wait here is safer for execute)
    # Actually, user wants to "just send jobs". But execute needs compiled refs.
    # So we MUST wait for compilation. Compilation is usually quick compared to QPU Execution.
    compile_name = f"ghz-compile-{timestamp}-{batch_id}"
    print(f"Compiling batch (job name: {compile_name})...")
    compile_job = qnx.start_compile_job(
        programs=circuit_refs,
        name=compile_name,
        optimisation_level=optimization_level,
        backend_config=backend_config,
        project=project,
    )
    
    try:
        qnx.jobs.wait_for(compile_job, timeout=None)
    except Exception as e:
        print(f"Error waiting for compilation: {e}")
        print(f"Current status: {qnx.jobs.status(compile_job).status}")
        raise

    compiled_refs = [item.get_output() for item in qnx.jobs.results(compile_job)]

    # Verify Connectivity
    if not verify_compilation(circuits, compiled_refs):
        print("Aborting execution due to verification failure.")
        return None, str(compile_job.id)

    # Syntax Check on H2-1SC
    print(f"Running syntax check on {SYNTAX_CHECKER_DEVICE}...")
    backend_config_sc = qnx.QuantinuumConfig(device_name=SYNTAX_CHECKER_DEVICE)
    sc_name = f"ghz-syntax-{timestamp}-{batch_id}"
    
    # Syntax check usually needs just 1 shot or handled by backend
    sc_job = qnx.start_execute_job(
        programs=compiled_refs,
        name=sc_name,
        n_shots=[1] * len(compiled_refs), # Minimal shots for syntax check
        backend_config=backend_config_sc,
        project=project,
        language=Language.QIR,
    )
    
    try:
        qnx.jobs.wait_for(sc_job, timeout=None)
    except Exception as e:
        print(f"Error waiting for syntax check: {e}")
        return None, str(compile_job.id)
        
    sc_status = qnx.jobs.status(sc_job).status
    if sc_status != "COMPLETED":
        print(f"Syntax check failed with status: {sc_status}")
        return None, str(compile_job.id)
        
    # Fetch updated job to get the cost message
    sc_job = qnx.jobs.get(sc_job.id)
    print("Syntax check passed!")
    print(f"Syntax Check Message (Cost): {sc_job.last_message}")

    if not execute:
        print("Dry run completed. Skipping execution.")
        return None, str(compile_job.id)

    # Execute circuits (Submit only)
    execute_name = f"ghz-execute-{timestamp}-{batch_id}"
    print(f"Submitting execution batch on {device_name} (job name: {execute_name})...")
    execute_job = qnx.start_execute_job(
        programs=compiled_refs,
        name=execute_name,
        n_shots=[shots] * len(compiled_refs),
        backend_config=backend_config,
        project=project,
        language=Language.QIR,
    )
    
    job_id = str(execute_job.id)
    compile_job_id = str(compile_job.id)
    print(f"Job submitted successfully! Execute Job ID: {job_id}, Compile Job ID: {compile_job_id}")
    return job_id, compile_job_id

# Main Experiment Parameters
total_node_counts = [50]
num_pairs_to_select = [6]
# Random phases for Compressed Sensing
num_phases = 19
phases = np.sort(np.random.uniform(0, 2 * np.pi, num_phases))
print(f"Phases: {phases}")

job_records = []
filename = "submitted_jobs.json"

# Load existing records if file exists to append
if os.path.exists(filename):
    try:
        with open(filename, 'r') as f:
            job_records = json.load(f)
    except Exception:
        pass

print(f"Starting Async GHZ Submission on {DEVICE_NAME}...")
if ENABLE_DD:
    print(f"  Dynamical Decoupling: ON (threshold={DD_THRESHOLD}s)")
if ENABLE_TREX:
    print(f"  TREX: ON ({TREX_NUM_RANDOMIZATIONS} randomizations)")
print(f"  REM readout rates: p0={READOUT_P0}, p1={READOUT_P1}")

for i, total_node_count in enumerate(total_node_counts):
    for j, num_pairs in enumerate(num_pairs_to_select):
        print(f"\nPreparing for Total Nodes: {total_node_count}, Selected Pairs: {num_pairs}")

        # Setup Topology
        root_node = build_tree_by_node_count(total_node_count)
        all_nodes = get_all_nodes(root_node)
        selected_pairs_k, ids_k = find_optimal_k_pairs(root_node, num_pairs)
        ratio_k = len(ids_k) / len(all_nodes) if all_nodes else 0
        
        print(f"Coverage Ratio: {ratio_k:.2%}")

        coupling_map, nodes, pair_nodes = create_coupling_map_from_selection(root_node, selected_pairs_k)
        layout = nodes
        flags = pair_nodes
        num_flags = len(flags)
        
        num_flags = len(flags)
        
        # Initialize GHZ
        ghz = GHZ(coupling_map=coupling_map, layout=layout, flag_qubits=flags)

        # Generate Circuits
        # Scale phase range to [0, 2pi/N] to capture one full oscillation period
        # This prevents undersampling for large N
        all_circuits = ghz.get_verification_circuits(method=METHOD, phases=phases)
        if not POPULATION:
            all_circuits = all_circuits[1:]

        # Generate TREX twirled + calibration circuits if enabled
        trex_randomization_strings = None
        if ENABLE_TREX:
            rng = np.random.RandomState(42)
            data_qubit_indices = [ghz.layout_dict[d] for d in ghz.data_qubits]
            flag_qubit_indices = [ghz.layout_dict[f] for f in ghz.flag_qubits]
            num_total_qubits = len(ghz.layout)

            trex_randomization_strings = [
                rng.randint(0, 2, size=len(ghz.data_qubits))
                for _ in range(TREX_NUM_RANDOMIZATIONS)
            ]

            trex_circuits = []
            for base_circ in all_circuits:
                for rs in trex_randomization_strings:
                    twirled = generate_trex_twirled_circuit(
                        base_circ, data_qubit_indices, rs
                    )
                    trex_circuits.append(twirled)

            # Calibration circuits (one per randomization)
            calib_circuits = [
                generate_trex_calibration_circuit(
                    num_total_qubits, data_qubit_indices, flag_qubit_indices, rs
                )
                for rs in trex_randomization_strings
            ]

            all_circuits = all_circuits + trex_circuits + calib_circuits
            print(f"  TREX ENABLED: {len(trex_circuits)} twirled + {len(calib_circuits)} calibration circuits added")

        # Submit Batch
        job_id, compile_job_id = submit_batch_on_quantinuum(
            circuits=all_circuits,
            device_name=DEVICE_NAME,
            shots=SHOTS,
            optimization_level=OPTIMIZATION_LEVEL,
            project_name=PROJECT_NAME,
            execute=EXECUTE,
            enable_dd=ENABLE_DD,
            dd_threshold=DD_THRESHOLD,
        )

        if job_id:
            record = {
                "job_id": job_id,
                "compile_job_id": compile_job_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "project": PROJECT_NAME,
                "device": DEVICE_NAME,
                "method": METHOD,
                "total_nodes": total_node_count,
                "num_pairs": num_pairs,
                "flags": num_flags,
                "qubits": total_node_count, # Matches what analyze script expects
                "coverage_ratio": ratio_k,
                "include_population": POPULATION,
                "phases": phases.tolist(), # Save the exact phases used
                "enable_dd": ENABLE_DD,
                "dd_threshold": DD_THRESHOLD,
                "enable_trex": ENABLE_TREX,
                "trex_num_randomizations": TREX_NUM_RANDOMIZATIONS if ENABLE_TREX else 0,
                "trex_randomization_strings": [
                    rs.tolist() for rs in trex_randomization_strings
                ] if trex_randomization_strings else None,
                "readout_p0": READOUT_P0,
                "readout_p1": READOUT_P1,
            }
            job_records.append(record)

            # Save incrementally
            with open(filename, 'w') as f:
                json.dump(job_records, f, indent=2)
            print(f"Saved job record to {filename}")

print("\nAll jobs submitted.")
