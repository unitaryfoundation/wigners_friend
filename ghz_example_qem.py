"""GHZ state verification with Quantum Error Mitigation (QEM).

This script compares GHZ state fidelity estimates under various QEM
strategies applied on top of the existing flag-qubit error detection
protocol.  It runs on a noisy Qiskit Aer simulator with both gate
errors *and* readout errors, then applies the following mitigation modes:

  none      – baseline (no mitigation beyond post-selection)
  rem       – Readout Error Mitigation via inverse confusion matrix
  ddd       – Digital Dynamical Decoupling (mitiq)
  trex      – Twirled Readout Error eXtinction
  ddd+rem   – DDD circuits with REM post-processing
  ddd+trex  – DDD circuits with TREX post-processing

Outputs:
  - Console table of (N, k_flags, QEM_mode) -> fidelity
  - Comparison plots saved as qem_fidelity_vs_pairs_N{n}.png
"""

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

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit import transpile

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SINGLE_QUBIT_ERROR_RATE = 0.001
TWO_QUBIT_ERROR_RATE = 0.01
READOUT_ERROR_RATE = 0.01  # p(0->1) = p(1->0) = 1%

TOTAL_NODE_COUNTS = [9, 10, 11]
NUM_PAIRS_TO_SELECT = [0, 1, 2]
QEM_MODES = ["none", "rem", "ddd", "trex", "ddd+rem", "ddd+trex"]

SHOTS = 10_000
TREX_NUM_RANDOMIZATIONS = 32
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

def build_noise_model():
    """Depolarizing gate errors + symmetric readout errors."""
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(SINGLE_QUBIT_ERROR_RATE, 1), ["u2", "u3", "x"]
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(TWO_QUBIT_ERROR_RATE, 2), ["cx"]
    )
    readout_err = ReadoutError(
        [
            [1 - READOUT_ERROR_RATE, READOUT_ERROR_RATE],
            [READOUT_ERROR_RATE, 1 - READOUT_ERROR_RATE],
        ]
    )
    noise_model.add_all_qubit_readout_error(readout_err)
    return noise_model


# ---------------------------------------------------------------------------
# REM helpers
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
    """Apply REM inverse confusion matrix to measurement counts.

    Converts counts -> probability vector, multiplies by inv_cm, clips
    negatives, re-normalises, and converts back to counts dict.
    """
    total = sum(counts.values())
    dim = 2 ** num_qubits

    # Build probability vector (index = integer value of bitstring)
    prob_vec = np.zeros(dim)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        prob_vec[idx] = count / total

    # Apply inverse confusion matrix
    corrected_vec = inv_cm @ prob_vec

    # Clip negatives and re-normalise
    corrected_vec = np.clip(corrected_vec, 0, None)
    norm = corrected_vec.sum()
    if norm > 0:
        corrected_vec /= norm

    # Convert back to counts
    corrected_counts = {}
    for idx in range(dim):
        if corrected_vec[idx] > 1e-12:
            bitstring = format(idx, f"0{num_qubits}b")
            corrected_counts[bitstring] = corrected_vec[idx] * total

    return corrected_counts


# ---------------------------------------------------------------------------
# DDD helpers
# ---------------------------------------------------------------------------

def apply_ddd_to_circuits(circuits: list) -> list:
    """Apply Digital Dynamical Decoupling to a list of Qiskit circuits.

    Uses mitiq's DDD with the XYXY rule to insert DD sequences into idle
    windows.
    """
    try:
        from mitiq import ddd

        modified = []
        for circ in circuits:
            try:
                ddd_circ = ddd.insert_ddd_sequences(circ, rule=ddd.rules.xyxy)
                modified.append(ddd_circ)
            except Exception:
                # If DDD insertion fails for a circuit, use the original
                modified.append(circ)
        return modified
    except ImportError:
        print("Warning: mitiq not installed. Skipping DDD.")
        return circuits


# ---------------------------------------------------------------------------
# TREX helpers
# ---------------------------------------------------------------------------

def run_trex_parity(
    ghz: GHZ,
    base_circuits: list,
    phases: np.ndarray,
    backend,
    num_randomizations: int,
    rng: np.random.RandomState,
) -> tuple[float, float, list[float]]:
    """Run TREX-mitigated parity oscillation and return (population, coherence, parities).

    For each parity circuit (and the Z-basis circuit), this generates
    ``num_randomizations`` twirled copies + calibration circuits, runs them,
    and applies TREX correction.
    """
    data_qubit_indices = [ghz.layout_dict[d] for d in ghz.data_qubits]
    flag_qubit_indices = [ghz.layout_dict[f] for f in ghz.flag_qubits]
    num_data = len(ghz.data_qubits)
    num_flags = len(ghz.flag_qubits)
    num_total_qubits = len(ghz.layout)

    # Generate randomization strings (shared across all circuits)
    rand_strings = [
        rng.randint(0, 2, size=num_data) for _ in range(num_randomizations)
    ]

    # --- Population (Z-basis, circuit index 0) ---
    z_circuit = base_circuits[0]
    z_twirled = [
        generate_trex_twirled_circuit(z_circuit, data_qubit_indices, rs)
        for rs in rand_strings
    ]
    z_calibrations = [
        generate_trex_calibration_circuit(
            num_total_qubits, data_qubit_indices, flag_qubit_indices, rs
        )
        for rs in rand_strings
    ]

    all_z = z_twirled + z_calibrations
    z_res = backend.run(transpile(all_z, backend), shots=SHOTS).result()

    # Compute TREX-corrected population
    pop_values = []
    cal_pop_values = []
    for r in range(num_randomizations):
        # Twirled result
        raw_counts = z_res.get_counts(r)
        xored = xor_counts_with_bitstring(raw_counts, rand_strings[r], num_data, num_flags)
        ps = post_select_results(xored, num_flags)
        total = sum(ps.values())
        if total > 0:
            p0 = ps.get("0" * num_data, 0)
            p1 = ps.get("1" * num_data, 0)
            pop_values.append((p0 + p1) / total)
        else:
            pop_values.append(0.0)

        # Calibration result
        cal_counts = z_res.get_counts(num_randomizations + r)
        cal_xored = xor_counts_with_bitstring(cal_counts, rand_strings[r], num_data, num_flags)
        cal_ps = post_select_results(cal_xored, num_flags)
        cal_total = sum(cal_ps.values())
        if cal_total > 0:
            cal_p0 = cal_ps.get("0" * num_data, 0)
            cal_pop_values.append(cal_p0 / cal_total)
        else:
            cal_pop_values.append(0.0)

    # For population, the calibration factor is the fraction of all-zeros
    # in the calibration circuits (ideally 1.0)
    population = float(np.mean(pop_values))
    cal_factor_pop = float(np.mean(cal_pop_values))
    if cal_factor_pop > 1e-10:
        population = population / cal_factor_pop

    # --- Coherence (parity oscillation circuits, indices 1..N) ---
    measured_parities = []
    for k in range(1, len(base_circuits)):
        phase_circuit = base_circuits[k]
        p_twirled = [
            generate_trex_twirled_circuit(phase_circuit, data_qubit_indices, rs)
            for rs in rand_strings
        ]
        p_calibrations = [
            generate_trex_calibration_circuit(
                num_total_qubits, data_qubit_indices, flag_qubit_indices, rs
            )
            for rs in rand_strings
        ]

        all_p = p_twirled + p_calibrations
        p_res = backend.run(transpile(all_p, backend), shots=SHOTS).result()

        raw_pars = []
        cal_pars = []
        for r in range(num_randomizations):
            # Twirled parity
            raw_counts = p_res.get_counts(r)
            xored = xor_counts_with_bitstring(raw_counts, rand_strings[r], num_data, num_flags)
            ps = post_select_results(xored, num_flags)
            total = sum(ps.values())
            if total > 0:
                even = sum(c for b, c in ps.items() if b.count("1") % 2 == 0)
                raw_pars.append((2 * even - total) / total)
            else:
                raw_pars.append(0.0)

            # Calibration parity (ideal = +1 for all-zeros state)
            cal_counts = p_res.get_counts(num_randomizations + r)
            cal_xored = xor_counts_with_bitstring(cal_counts, rand_strings[r], num_data, num_flags)
            cal_ps = post_select_results(cal_xored, num_flags)
            cal_total = sum(cal_ps.values())
            if cal_total > 0:
                cal_even = sum(c for b, c in cal_ps.items() if b.count("1") % 2 == 0)
                cal_pars.append((2 * cal_even - cal_total) / cal_total)
            else:
                cal_pars.append(0.0)

        corrected_parity = compute_trex_corrected_parity(raw_pars, cal_pars)
        measured_parities.append(corrected_parity)

    return population, measured_parities


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    total_node_count: int,
    num_pairs: int,
    qem_mode: str,
    backend,
    cs_estimator: CompressedSensingEstimator,
    phases: np.ndarray,
    rng: np.random.RandomState,
) -> dict:
    """Run the full GHZ verification pipeline with a given QEM mode.

    Returns dict with keys: fidelity, population, coherence, coverage_ratio.
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

    # --- Apply DDD if requested ---
    use_ddd = "ddd" in qem_mode
    if use_ddd:
        circuits = apply_ddd_to_circuits(circuits)

    # --- TREX path (special: runs its own circuits) ---
    if "trex" in qem_mode:
        population, measured_parities = run_trex_parity(
            ghz, circuits, phases, backend, TREX_NUM_RANDOMIZATIONS, rng
        )
    else:
        # --- Standard execution ---
        res = backend.run(transpile(circuits, backend), shots=SHOTS).result()

        # Population (Z-basis, index 0)
        z_counts_raw = res.get_counts(0)
        z_counts = post_select_results(z_counts_raw, num_flags)

        # Apply REM if requested
        if "rem" in qem_mode and num_flags == 0:
            # When no flags, apply REM directly to data qubits
            inv_cm = build_inverse_confusion_matrix(
                total_node_count, READOUT_ERROR_RATE, READOUT_ERROR_RATE
            )
            z_counts = apply_rem_to_counts(z_counts, inv_cm, total_node_count)
        elif "rem" in qem_mode and num_flags > 0:
            # With flags, apply REM to the post-selected data-qubit counts
            inv_cm = build_inverse_confusion_matrix(
                total_node_count, READOUT_ERROR_RATE, READOUT_ERROR_RATE
            )
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
            counts_raw = res.get_counts(k)
            counts = post_select_results(counts_raw, num_flags)

            if "rem" in qem_mode:
                inv_cm = build_inverse_confusion_matrix(
                    total_node_count, READOUT_ERROR_RATE, READOUT_ERROR_RATE
                )
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
    np.random.seed(RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)

    noise_model = build_noise_model()
    backend = AerSimulator(noise_model=noise_model)
    cs_estimator = CompressedSensingEstimator(max_freq_candidate=50)

    # Store all results: results[mode][(N, k)] = dict
    all_results = {mode: {} for mode in QEM_MODES}

    for total_node_count in TOTAL_NODE_COUNTS:
        # Generate phases (shared across all modes for fair comparison)
        M_samples = max(15, int(5 * np.log(total_node_count)))
        phases = np.sort(rng.uniform(0, 2 * np.pi, M_samples))

        for num_pairs in NUM_PAIRS_TO_SELECT:
            print(f"\n{'='*60}")
            print(f"N={total_node_count}, flags={num_pairs}")
            print(f"{'='*60}")

            for mode in QEM_MODES:
                print(f"  QEM mode: {mode:12s} ... ", end="", flush=True)
                try:
                    result = run_pipeline(
                        total_node_count, num_pairs, mode, backend,
                        cs_estimator, phases, rng,
                    )
                    all_results[mode][(total_node_count, num_pairs)] = result
                    print(
                        f"F={result['fidelity']:.4f}  "
                        f"(P={result['population']:.4f}, "
                        f"C={result['coherence']:.4f}, "
                        f"n_rec={result['recovered_n']})"
                    )
                except Exception as e:
                    print(f"FAILED: {e}")
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
        "rem": "blue",
        "ddd": "green",
        "trex": "red",
        "ddd+rem": "purple",
        "ddd+trex": "orange",
    }

    for n in TOTAL_NODE_COUNTS:
        fig, ax = plt.subplots(figsize=(10, 6))

        for mode in QEM_MODES:
            fids = []
            for k in NUM_PAIRS_TO_SELECT:
                fid = all_results[mode].get((n, k), {}).get("fidelity", 0)
                fids.append(fid)
            ax.plot(
                NUM_PAIRS_TO_SELECT,
                fids,
                "o-",
                color=colors.get(mode, "black"),
                label=mode,
                linewidth=2,
                markersize=8,
            )

        ax.set_title(f"Fidelity vs. Flag Pairs with QEM (N={n})")
        ax.set_xlabel("Number of Flag Pairs")
        ax.set_ylabel("Estimated Fidelity")
        ax.set_xticks(NUM_PAIRS_TO_SELECT)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.margins(y=0.1)

        # Annotate coverage on the baseline
        for k in NUM_PAIRS_TO_SELECT:
            cov = all_results["none"].get((n, k), {}).get("coverage_ratio", 0)
            fid = all_results["none"].get((n, k), {}).get("fidelity", 0)
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
        save_path = f"qem_fidelity_vs_pairs_N{n}.png"
        plt.savefig(save_path, dpi=300)
        print(f"  Saved: {save_path}")
        plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
