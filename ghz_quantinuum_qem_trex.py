"""GHZ state verification with Quantum Error Mitigation (QEM) on Quantinuum H2 Emulator.

This script compares GHZ state fidelity estimates under various QEM
strategies on the Quantinuum H2-Emulator (cloud-based noisy emulator with
a realistic noise model calibrated to actual H2 hardware).

QEM modes:
  none      - baseline (no mitigation beyond post-selection)
  dd        - Dynamical Decoupling via Quantinuum compiler (apply_DD)
  rem       - Readout Error Mitigation via inverse confusion matrix
  trex      - Twirled Readout Error eXtinction
  dd+rem    - DD + REM post-processing
  dd+trex   - DD + TREX post-processing

Outputs:
  - Console table of (N, k_flags, QEM_mode) -> fidelity
  - Comparison plots saved as qem_quantinuum_fidelity_vs_pairs_N{n}.png

Requires:
  - qNexus authentication (run `uv run qnx login` or set credentials in .env)
  - .env file with QUANTINUUM_DEVICE=H2-Emulator (see .env.example)
"""

import os
import uuid
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import qnexus as qnx
from pytket.extensions.qiskit import qiskit_to_tk
from qnexus.models.language import Language

from ewfs.ghz import (
    GHZ,
    post_select_results,
    generate_trex_twirled_circuit,
    generate_trex_calibration_circuit,
    xor_counts_with_bitstring,
    compute_trex_corrected_parity,
)
from compressed_sensing_ghz import CompressedSensingEstimator
from ewfs.layout import (
    build_tree_by_node_count,
    get_all_nodes,
    find_optimal_k_pairs,
    create_coupling_map_from_selection,
)

# ---------------------------------------------------------------------------
# Configuration (from environment / defaults)
# ---------------------------------------------------------------------------
load_dotenv()

DEVICE_NAME = os.getenv("QUANTINUUM_DEVICE", "H2-Emulator")
SHOTS = int(os.getenv("QUANTINUUM_SHOTS", "10000"))
OPTIMIZATION_LEVEL = int(os.getenv("QUANTINUUM_OPT_LEVEL", "1"))
PROJECT_NAME = os.getenv("QUANTINUUM_PROJECT", "GHZ-QEM")
DD_THRESHOLD = float(os.getenv("QUANTINUUM_DD_THRESHOLD", "0.03"))
READOUT_P0 = float(os.getenv("QUANTINUUM_READOUT_P0", "0.003"))
READOUT_P1 = float(os.getenv("QUANTINUUM_READOUT_P1", "0.003"))
TREX_NUM_RANDOMIZATIONS = 8

TOTAL_NODE_COUNTS = [9]
NUM_PAIRS_TO_SELECT = [0, 1, 2]
QEM_MODES = ["none", "trex", "dd+trex"]

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# qNexus authentication
# ---------------------------------------------------------------------------
def authenticate():
    """Authenticate with qNexus using .env credentials or existing session."""
    email = os.getenv("QNEXUS_EMAIL")
    password = os.getenv("QNEXUS_PASSWORD")

    if email and password:
        print("Authenticating with qNexus using credentials from .env...")
        try:
            qnx.auth._request_tokens(user=email, pwd=password)
            print("Authentication successful!")
        except Exception as e:
            print(f"Authentication failed: {e}")
            print("Please run 'uv run qnx login' or check your credentials in .env")
            raise
    else:
        print("No credentials in .env. Using existing qNexus authentication.")
        print("If authentication fails, run: uv run qnx login")


# ---------------------------------------------------------------------------
# qNexus circuit execution helper
# ---------------------------------------------------------------------------
def run_circuits_on_quantinuum(
    circuits: list,
    device_name: str,
    shots: int,
    project_name: str,
    opt_level: int,
    enable_dd: bool = False,
    dd_threshold: float = 0.03,
) -> list[dict]:
    """Submit Qiskit circuits to Quantinuum via qNexus, wait, return counts.

    Handles: Qiskit->pytket conversion, upload, compile (with optional DD),
    execute, wait, download, and pytket->Qiskit bitstring conversion.

    Returns list of counts dicts (one per circuit) in Qiskit format:
        {"010...": count}  (MSB-first string keys)
    """
    if not circuits:
        return []

    project = qnx.projects.get_or_create(name=project_name)

    # Build backend config with optional DD
    compiler_options = {}
    if enable_dd:
        compiler_options["apply_DD"] = True
        compiler_options["DD_threshold_times"] = [dd_threshold]

    backend_config = qnx.QuantinuumConfig(
        device_name=device_name,
        **({"compiler_options": compiler_options} if compiler_options else {}),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    batch_id = uuid.uuid4().hex[:6]

    # Upload circuits (Qiskit -> pytket)
    circuit_refs = []
    for i, circuit in enumerate(circuits):
        circuit_name = f"qem-{batch_id}-{i}-{timestamp}"
        pytket_circuit = qiskit_to_tk(circuit)
        ref = qnx.circuits.upload(
            name=circuit_name,
            circuit=pytket_circuit,
            project=project,
        )
        circuit_refs.append(ref)

    # Compile
    compile_name = f"qem-compile-{timestamp}-{batch_id}"
    dd_label = " [DD]" if enable_dd else ""
    print(f"    Compiling {len(circuits)} circuits{dd_label}...", end="", flush=True)
    compile_job = qnx.start_compile_job(
        programs=circuit_refs,
        name=compile_name,
        optimisation_level=opt_level,
        backend_config=backend_config,
        project=project,
    )
    qnx.jobs.wait_for(compile_job, timeout=None)
    compiled_refs = [item.get_output() for item in qnx.jobs.results(compile_job)]
    print(" done.", flush=True)

    # Execute
    execute_name = f"qem-execute-{timestamp}-{batch_id}"
    print(f"    Executing on {device_name}...", end="", flush=True)
    execute_job = qnx.start_execute_job(
        programs=compiled_refs,
        name=execute_name,
        n_shots=[shots] * len(compiled_refs),
        backend_config=backend_config,
        project=project,
        language=Language.QIR,
    )
    qnx.jobs.wait_for(execute_job, timeout=None)
    print(" done.", flush=True)

    # Download results and convert to Qiskit format
    results = qnx.jobs.results(execute_job)
    counts_list = []
    for result_item in results:
        pytket_counts = result_item.download_result().get_counts()
        # pytket: {(q0, q1, ...): count} (LSB-first tuple)
        # Qiskit: {"...q1q0": count} (MSB-first string, i.e. reversed)
        qiskit_counts = {
            "".join(map(str, reversed(k))): v
            for k, v in pytket_counts.items()
        }
        counts_list.append(qiskit_counts)

    return counts_list


# ---------------------------------------------------------------------------
# REM helpers (same as ghz_example_qem.py)
# ---------------------------------------------------------------------------
def _single_qubit_inv_confusion(p0: float, p1: float) -> np.ndarray:
    """Inverse of a single-qubit confusion matrix [[1-p0, p0],[p1, 1-p1]]."""
    cm = np.array([[1 - p0, p0], [p1, 1 - p1]])
    return np.linalg.inv(cm)


def build_inverse_confusion_matrix(num_qubits: int, p0: float, p1: float) -> np.ndarray:
    """Build the full 2^n x 2^n inverse confusion matrix (tensor product)."""
    inv_single = _single_qubit_inv_confusion(p0, p1)
    inv_full = inv_single
    for _ in range(num_qubits - 1):
        inv_full = np.kron(inv_full, inv_single)
    return inv_full


def apply_rem_to_counts(counts: dict, inv_cm: np.ndarray, num_qubits: int) -> dict:
    """Apply REM inverse confusion matrix to measurement counts."""
    total = sum(counts.values())
    dim = 2 ** num_qubits

    prob_vec = np.zeros(dim)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        prob_vec[idx] = count / total

    corrected_vec = inv_cm @ prob_vec
    corrected_vec = np.clip(corrected_vec, 0, None)
    norm = corrected_vec.sum()
    if norm > 0:
        corrected_vec /= norm

    corrected_counts = {}
    for idx in range(dim):
        if corrected_vec[idx] > 1e-12:
            bitstring = format(idx, f"0{num_qubits}b")
            corrected_counts[bitstring] = corrected_vec[idx] * total

    return corrected_counts


# ---------------------------------------------------------------------------
# TREX execution on Quantinuum (single-batch)
# ---------------------------------------------------------------------------
def run_trex_parity(
    ghz: GHZ,
    base_circuits: list,
    phases: np.ndarray,
    num_randomizations: int,
    rng: np.random.RandomState,
    device_name: str,
    shots: int,
    project_name: str,
    opt_level: int,
    enable_dd: bool = False,
    dd_threshold: float = 0.03,
) -> tuple[float, list[float]]:
    """Run TREX-mitigated parity oscillation on Quantinuum and return (population, parities).

    All twirled + calibration circuits are submitted in a SINGLE batch to
    minimise compile/execute round-trips to the cloud emulator.

    Circuit layout in the batch:
        [Z twirled (R)] [phase_1 twirled (R)] ... [phase_M twirled (R)] [calibration (R)]

    where R = num_randomizations.  Calibration circuits are shared across
    all phases (they only depend on the randomisation strings, not the
    base circuit).
    """
    data_qubit_indices = [ghz.layout_dict[d] for d in ghz.data_qubits]
    flag_qubit_indices = [ghz.layout_dict[f] for f in ghz.flag_qubits]
    num_data = len(ghz.data_qubits)
    num_flags = len(ghz.flag_qubits)
    num_total_qubits = len(ghz.layout)
    R = num_randomizations

    rand_strings = [
        rng.randint(0, 2, size=num_data) for _ in range(R)
    ]

    # --- Build ALL circuits in one list ---
    all_circuits = []

    # Z-basis twirled (indices 0 .. R-1)
    z_circuit = base_circuits[0]
    for rs in rand_strings:
        all_circuits.append(generate_trex_twirled_circuit(z_circuit, data_qubit_indices, rs))

    # Parity-phase twirled (indices R .. R + M*R - 1, where M = len(base_circuits)-1)
    num_phases = len(base_circuits) - 1
    for k in range(1, len(base_circuits)):
        for rs in rand_strings:
            all_circuits.append(
                generate_trex_twirled_circuit(base_circuits[k], data_qubit_indices, rs)
            )

    # Calibration circuits (indices R + M*R .. R + M*R + R - 1)
    # These are shared: same for every base circuit
    cal_offset = len(all_circuits)
    for rs in rand_strings:
        all_circuits.append(
            generate_trex_calibration_circuit(
                num_total_qubits, data_qubit_indices, flag_qubit_indices, rs
            )
        )

    total_circuits = len(all_circuits)  # R + M*R + R = R*(M+2)
    print(f"    TREX: {total_circuits} circuits in single batch "
          f"({R} Z-twirled + {num_phases * R} phase-twirled + {R} calibration)")

    # --- Single submission ---
    counts_list = run_circuits_on_quantinuum(
        all_circuits, device_name, shots, project_name, opt_level, enable_dd, dd_threshold
    )

    # --- Extract calibration results (shared) ---
    cal_pop_values = []
    cal_parity_values = []
    for r in range(R):
        cal_counts = counts_list[cal_offset + r]
        cal_xored = xor_counts_with_bitstring(cal_counts, rand_strings[r], num_data, num_flags)
        cal_ps = post_select_results(cal_xored, num_flags)
        cal_total = sum(cal_ps.values())
        if cal_total > 0:
            cal_p0 = cal_ps.get("0" * num_data, 0)
            cal_pop_values.append(cal_p0 / cal_total)
            cal_even = sum(c for b, c in cal_ps.items() if b.count("1") % 2 == 0)
            cal_parity_values.append((2 * cal_even - cal_total) / cal_total)
        else:
            cal_pop_values.append(0.0)
            cal_parity_values.append(0.0)

    # --- Population from Z-basis twirled (indices 0..R-1) ---
    pop_values = []
    for r in range(R):
        raw_counts = counts_list[r]
        xored = xor_counts_with_bitstring(raw_counts, rand_strings[r], num_data, num_flags)
        ps = post_select_results(xored, num_flags)
        total = sum(ps.values())
        if total > 0:
            p0 = ps.get("0" * num_data, 0)
            p1 = ps.get("1" * num_data, 0)
            pop_values.append((p0 + p1) / total)
        else:
            pop_values.append(0.0)

    population = float(np.mean(pop_values))
    cal_factor_pop = float(np.mean(cal_pop_values))
    if cal_factor_pop > 1e-10:
        population = population / cal_factor_pop

    # --- Coherence from parity-phase twirled ---
    measured_parities = []
    for k_idx in range(num_phases):
        offset = R + k_idx * R  # start index for this phase's twirled circuits
        raw_pars = []
        for r in range(R):
            raw_counts = counts_list[offset + r]
            xored = xor_counts_with_bitstring(raw_counts, rand_strings[r], num_data, num_flags)
            ps = post_select_results(xored, num_flags)
            total = sum(ps.values())
            if total > 0:
                even = sum(c for b, c in ps.items() if b.count("1") % 2 == 0)
                raw_pars.append((2 * even - total) / total)
            else:
                raw_pars.append(0.0)

        corrected_parity = compute_trex_corrected_parity(raw_pars, cal_parity_values)
        measured_parities.append(corrected_parity)

    return population, measured_parities


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    total_node_count: int,
    num_pairs: int,
    qem_mode: str,
    cs_estimator: CompressedSensingEstimator,
    phases: np.ndarray,
    rng: np.random.RandomState,
) -> dict:
    """Run the full GHZ verification pipeline with a given QEM mode on Quantinuum.

    Returns dict with keys: fidelity, population, coherence, coverage_ratio, recovered_n.
    """
    # --- Build topology ---
    root_node = build_tree_by_node_count(total_node_count)
    all_nodes = get_all_nodes(root_node)
    selected_pairs_k, ids_k = find_optimal_k_pairs(root_node, num_pairs)
    ratio_k = len(ids_k) / len(all_nodes) if all_nodes else 0

    coupling_map, nodes, pair_nodes = create_coupling_map_from_selection(
        root_node, selected_pairs_k
    )
    layout = nodes
    flags = pair_nodes
    num_flags = len(flags)

    ghz = GHZ(coupling_map=coupling_map, layout=layout, flag_qubits=flags)

    # --- Generate circuits ---
    circuits = ghz.get_verification_circuits(
        method="parity_oscillation", phases=phases.tolist()
    )

    enable_dd = "dd" in qem_mode

    # --- TREX path ---
    if "trex" in qem_mode:
        population, measured_parities = run_trex_parity(
            ghz, circuits, phases, TREX_NUM_RANDOMIZATIONS, rng,
            DEVICE_NAME, SHOTS, PROJECT_NAME, OPTIMIZATION_LEVEL,
            enable_dd=enable_dd, dd_threshold=DD_THRESHOLD,
        )
    else:
        # --- Standard execution on Quantinuum ---
        counts_list = run_circuits_on_quantinuum(
            circuits, DEVICE_NAME, SHOTS, PROJECT_NAME, OPTIMIZATION_LEVEL,
            enable_dd=enable_dd, dd_threshold=DD_THRESHOLD,
        )

        # Population (Z-basis, index 0)
        z_counts_raw = counts_list[0]
        z_counts = post_select_results(z_counts_raw, num_flags)

        # Apply REM if requested
        if "rem" in qem_mode:
            inv_cm = build_inverse_confusion_matrix(total_node_count, READOUT_P0, READOUT_P1)
            z_counts = apply_rem_to_counts(z_counts, inv_cm, total_node_count)

        total_z = sum(z_counts.values())
        if total_z > 0:
            p0 = z_counts.get("0" * total_node_count, 0)
            p1 = z_counts.get("1" * total_node_count, 0)
            population = (p0 + p1) / total_z
        else:
            population = 0.0

        # Parities (oscillation circuits, indices 1..N)
        measured_parities = []
        for k in range(1, len(circuits)):
            counts_raw = counts_list[k]
            counts = post_select_results(counts_raw, num_flags)

            if "rem" in qem_mode:
                inv_cm = build_inverse_confusion_matrix(total_node_count, READOUT_P0, READOUT_P1)
                counts = apply_rem_to_counts(counts, inv_cm, total_node_count)

            total = sum(counts.values())
            if total > 0:
                even = sum(c for b, c in counts.items() if b.count("1") % 2 == 0)
                parity_val = (2 * even - total) / total
            else:
                parity_val = 0.0
            measured_parities.append(parity_val)

    # --- CS estimation ---
    fit_result = cs_estimator.fit(phases, measured_parities)
    coherence = fit_result["C"]

    fidelity = (population + coherence) / 2

    return {
        "fidelity": fidelity,
        "population": population,
        "coherence": coherence,
        "coverage_ratio": ratio_k,
        "recovered_n": fit_result["n"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    authenticate()

    print(f"\nDevice: {DEVICE_NAME}")
    print(f"Shots: {SHOTS}")
    print(f"Optimization level: {OPTIMIZATION_LEVEL}")
    print(f"DD threshold: {DD_THRESHOLD}s")
    print(f"REM error rates: p0={READOUT_P0}, p1={READOUT_P1}")
    print(f"TREX randomizations: {TREX_NUM_RANDOMIZATIONS}")
    print(f"Node counts: {TOTAL_NODE_COUNTS}")
    print(f"Flag pairs: {NUM_PAIRS_TO_SELECT}")
    print(f"QEM modes: {QEM_MODES}")

    np.random.seed(RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)
    cs_estimator = CompressedSensingEstimator(max_freq_candidate=50)

    all_results = {mode: {} for mode in QEM_MODES}

    for total_node_count in TOTAL_NODE_COUNTS:
        M_samples = max(15, int(5 * np.log(total_node_count)))
        phases = np.sort(rng.uniform(0, 2 * np.pi, M_samples))

        for num_pairs in NUM_PAIRS_TO_SELECT:
            print(f"\n{'='*60}")
            print(f"N={total_node_count}, flags={num_pairs}")
            print(f"{'='*60}")

            for mode in QEM_MODES:
                print(f"  QEM mode: {mode:12s} ... ", flush=True)
                try:
                    result = run_pipeline(
                        total_node_count, num_pairs, mode,
                        cs_estimator, phases, rng,
                    )
                    all_results[mode][(total_node_count, num_pairs)] = result
                    print(
                        f"  -> F={result['fidelity']:.4f}  "
                        f"(P={result['population']:.4f}, "
                        f"C={result['coherence']:.4f}, "
                        f"n_rec={result['recovered_n']})"
                    )
                except Exception as e:
                    print(f"  -> FAILED: {e}")
                    all_results[mode][(total_node_count, num_pairs)] = {
                        "fidelity": 0.0,
                        "population": 0.0,
                        "coherence": 0.0,
                        "coverage_ratio": 0.0,
                        "recovered_n": -1,
                    }

    # --- Summary table ---
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'N':>3} {'k':>3} {'Coverage':>9}"
    for mode in QEM_MODES:
        header += f" {mode:>10}"
    print(header)
    print("-" * len(header))

    for n in TOTAL_NODE_COUNTS:
        for k in NUM_PAIRS_TO_SELECT:
            cov = all_results[QEM_MODES[0]].get((n, k), {}).get("coverage_ratio", 0)
            row = f"{n:>3} {k:>3} {cov:>8.1%}"
            for mode in QEM_MODES:
                fid = all_results[mode].get((n, k), {}).get("fidelity", 0)
                row += f" {fid:>10.4f}"
            print(row)

    # --- Plots ---
    print("\nGenerating comparison plots...")
    colors = {
        "none": "gray",
        "dd": "green",
        "rem": "blue",
        "trex": "red",
        "dd+rem": "purple",
        "dd+trex": "orange",
    }

    for n in TOTAL_NODE_COUNTS:
        fig, ax = plt.subplots(figsize=(10, 6))

        for mode in QEM_MODES:
            fids = []
            for k in NUM_PAIRS_TO_SELECT:
                fid = all_results[mode].get((n, k), {}).get("fidelity", 0)
                fids.append(fid)
            linestyle = "o--" if mode == "none" else "o-"
            label = "none (unmitigated)" if mode == "none" else mode
            ax.plot(
                NUM_PAIRS_TO_SELECT,
                fids,
                linestyle,
                color=colors.get(mode, "black"),
                label=label,
                linewidth=2,
                markersize=8,
            )

        ax.set_title(f"Fidelity vs. Flag Pairs with QEM on {DEVICE_NAME} (N={n})")
        ax.set_xlabel("Number of Flag Pairs")
        ax.set_ylabel("Estimated Fidelity")
        ax.set_xticks(NUM_PAIRS_TO_SELECT)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.margins(y=0.1)

        # Annotate coverage ratios using the first available mode
        ref_mode = QEM_MODES[0]
        for k in NUM_PAIRS_TO_SELECT:
            cov = all_results[ref_mode].get((n, k), {}).get("coverage_ratio", 0)
            fid = all_results[ref_mode].get((n, k), {}).get("fidelity", 0)
            ax.annotate(
                f"{cov:.0%}",
                (k, fid),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=8,
                color="gray",
            )

        plt.tight_layout()
        save_path = f"qem_quantinuum_trex_fidelity_vs_pairs_N{n}.png"
        plt.savefig(save_path, dpi=300)
        print(f"  Saved: {save_path}")
        plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
