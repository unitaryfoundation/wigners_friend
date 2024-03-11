import argparse

from typing import Optional

import numpy as np
import os
import pickle

from enum import Enum
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

import qiskit
import qiskit_aer
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
from qiskit.compiler import transpile
from qiskit.providers import Backend
from qiskit.providers import fake_provider
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService

DATA_PATH = os.path.join("..", "data", "majority_vote")


#######################################################################################################################
# NOISE MODELS
#######################################################################################################################
def depolarizing_noise_model(error: float = 0.01) -> NoiseModel:
    """Defines an depolarizing noise model with one-qubit.

    Args:
        error: One-qubit gate error rate (default 1%).
    Returns:
        Depolarizing noise model.
    """
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error, 1), ["u1", "u2", "u3"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error, 2), "cx")
    return noise_model


def bitflip_model(p: float) -> NoiseModel:
    """Bitflip noise model with majority vote approach.

    Args:
        p: Probability to flip.
    Returns:
        Bit-flip noise model.
    """
    # Example error probabilities
    p_meas = p
    p_gate1 = p

    # QuantumError objects
    error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])
    error_gate1 = pauli_error([("X", p_gate1), ("I", 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model
    noise_bit_flip = NoiseModel()
    noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
    noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
    noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

    return noise_bit_flip


#######################################################################################################################
# CONSANTS
#######################################################################################################################
# Settings for extended Wigner's friend scenario.
class Setting(Enum):
    PEEK = 1
    REVERSE_1 = 2
    REVERSE_2 = 3

# Observers for scenario are Alice and Bob.
class Observer(Enum):
    ALICE = 0
    BOB = 1

# Experiment settings (peek, reverse_1, and reverse_2).
PEEK = Setting.PEEK.value
REVERSE_1 = Setting.REVERSE_1.value
REVERSE_2 = Setting.REVERSE_2.value
SETTINGS = [PEEK, REVERSE_1, REVERSE_2]

# "Super"-observers (Alice and Bob).
ALICE = Observer.ALICE.value 
BOB = Observer.BOB.value
OBSERVERS = [ALICE, BOB]

# Angles and beta term used for Alice and Bob measurement operators from arXiv:1907.05607.
# Note that despite the fact that degrees are used, we need to convert this to radians.
ANGLES = {PEEK: np.deg2rad(168), REVERSE_1: np.deg2rad(0), REVERSE_2: np.deg2rad(118)}
BETA = np.deg2rad(175)


#######################################################################################################################
# EXPECTATION VALUES
#######################################################################################################################
def decode_results(results: dict, charlie_size: int, debbie_size: int = 1) -> dict[str, float]:
    """Take majority vote of measurement bit-strings."""
    decoded_results = {}

    # For each setting, there is a dictionary of measurement results.
    for setting in results:
        if setting == (PEEK, REVERSE_1) or setting == (PEEK, REVERSE_2):
            # Debbie's size is 1 because no PEEK setting
            debbie_size = 1

            setting_results = {}
            # Decode the keys for each measurement result of the setting.
            for k, v in results[setting].items():
                alice_friend, bob_friend = k[:charlie_size], k[-debbie_size:]

                alice_zero_count, bob_zero_count = alice_friend.count("0"), bob_friend.count("0")

                alice_decoding = "0" if alice_zero_count >= charlie_size // 2 + 1 else "1"
                bob_decoding = "0" if bob_zero_count >= 1 else "1"

                if alice_decoding + bob_decoding in setting_results.keys():
                    setting_results[alice_decoding + bob_decoding] += v
                else:
                    setting_results[alice_decoding + bob_decoding] = v
            decoded_results[setting] = setting_results
        else:
            decoded_results[setting] = results[setting]

    return decoded_results

def double_expect(settings: tuple[int, int], results: dict) -> float:
    """Expectation value of product of two operators."""
    probs = results[settings]
    # <AB> = P(00) - P(01) - P(10) + P(11)
    return probs.get("00", 0) - probs.get("01", 0) - probs.get("10", 0) + probs.get("11", 0)


#######################################################################################################################
# CONSIDER ALL EXPERIMENTAL SETTINGS
#######################################################################################################################
def generate_all_experiments(
    backend: Backend,
    noise_model: Optional[NoiseModel],
    shots: int,
    angles: dict,
    beta: float,
    charlie_size: int,
    debbie_size: int,
    optimize: bool = True
) -> dict[tuple[int, int], list[float]]:
    """Generate probabilitites for all combinations of experimental settings."""
    all_experiment_combos = [[PEEK, REVERSE_1], [PEEK, REVERSE_2], [REVERSE_2, REVERSE_1], [REVERSE_2, REVERSE_2]]

    results = {}
    circuits = {}
    for alice, bob in all_experiment_combos:
        circuits[(alice, bob)] = ewfs(alice, bob, angles, beta, charlie_size, debbie_size)

    if optimize:
        transpiled_circuits = transpile(
            list(circuits.values()),
            backend=backend,
            optimization_level=0,
            initial_layout=None,
        )
        circuits = {key: circuit for key, circuit in zip(circuits.keys(), transpiled_circuits)}

    job = qiskit.execute(
        experiments=list(circuits.values()),
        backend=backend,
        noise_model=noise_model,
        basis_gates=noise_model.basis_gates if noise_model is not None else None,
        shots=shots,
    )
    counts = job.result().get_counts()

    # Convert counts to probabilities.
    for key, count in zip(circuits.keys(), counts):
        probabilities = {k[::-1]: v / shots for k, v in count.items()}
        results[key] = probabilities
    return results

#######################################################################################################################
# STATE PREPARATION
#######################################################################################################################
def prepare_bipartite_system(qc: QuantumCircuit):
    """Generates the state: 1/sqrt(2) * (|01> - |10>)"""
    qc.x(ALICE)
    qc.x(BOB)
    qc.h(ALICE)
    qc.cx(ALICE, BOB)


#######################################################################################################################
# CNOT LADDER CIRCUIT
#######################################################################################################################
def cnot_ladder(qc: QuantumCircuit, observer: int, friend_qubit: int, friend_size: int, reverse: bool, internal_copy: bool):
    """CNOT ladder circuit (GHZ without Hadamard)."""
    if internal_copy:
        if reverse:
            for i in range(friend_size-1):
                qc.cx(friend_qubit + friend_size-2-i, friend_qubit+friend_size-1-i)
            qc.cx(observer, friend_qubit)
        else:
            qc.cx(observer, friend_qubit)
            for i in range(friend_size-1):
                qc.cx(friend_qubit+i, friend_qubit + i + 1)
    else:
        if reverse:
            for i in range(friend_size):
                qc.cx(observer, friend_qubit+friend_size-1-i)
        else:
            for i in range(friend_size):
                qc.cx(observer, friend_qubit + i)


#######################################################################################################################
# CIRCUIT FOR EXTENDED WIGNER'S FRIEND SCENARIO
#######################################################################################################################
def ewfs_rotation(qc: QuantumCircuit, qubit: int, angle: float):
    qc.rz(-angle, qubit)
    qc.h(qubit)

def apply_setting(qc: QuantumCircuit,
                  observer: int,
                  setting: int,
                  angle: float,
                  observer_creg: list[int] | int,
                  friend_qubits: list[int],
                  friend_size: int):
    """Apply either the PEEK or REVERSE_1/REVERSE_2 settings."""
    if setting is PEEK:
        # Ask friend for the outcome.
        qc.measure(friend_qubits, observer_creg)

    elif setting in [REVERSE_1, REVERSE_2]:
        cnot_ladder(qc, observer, friend_qubits[0], friend_size, reverse=True, internal_copy=True)

        # For either REVERSE_1 or REVERSE_2, apply the appropriate angle rotations.
        # Note that in this case, the rotation should occur on the observer's qubit.
        if observer is ALICE:
            qc.h(ALICE)
            qc.rz(ANGLES[1], ALICE)

        if observer is BOB:
            qc.h(BOB)
            qc.rz((BETA - ANGLES[1]), BOB)
        ewfs_rotation(qc, observer, angle)
        qc.measure(observer, observer_creg)

def ewfs(alice_setting: int,
        bob_setting: int,
        angles: list[float],
        beta: float,
        charlie_size: int,
        debbie_size: int = 1) -> QuantumCircuit:
    """Generate the circuit for extended Wigner's friend scenario."""
    # Define quantum registers
    alice_size, bob_size = 1, 1
    sys_size = alice_size + bob_size
    meas_size = 2

    alice, bob, charlie, debbie = [
        QuantumRegister(size, name=name) 
        for size, name in zip([alice_size, bob_size, charlie_size, debbie_size], 
                              ["Alice", "Bob", "Charlie", "Debbie"])
    ]
    if (alice_setting == PEEK and bob_setting != PEEK):
        measurement = ClassicalRegister(charlie_size + 1, name="Measurement")
        alice_creg = list(range(charlie_size))
        bob_creg = charlie_size
    else:
        measurement = ClassicalRegister(meas_size, name="Measurement")
        alice_creg = 0
        bob_creg = 1

    # Create the Quantum Circuit with the defined registers
    qc = QuantumCircuit(alice, bob, charlie, debbie, measurement)

    charlie_qubits = list(range(sys_size, (sys_size + charlie_size)))
    debbie_qubits = list(range(sys_size + charlie_size, sys_size + (charlie_size + debbie_size)))

    # Prepare the bipartite quantum system
    prepare_bipartite_system(qc)

    # Rotations for measurement.
    ewfs_rotation(qc, ALICE, angles[1])
    ewfs_rotation(qc, BOB, beta - angles[1])

    # Apply the CNOT ladder for Alice-Charlie and Bob-Debbie
    cnot_ladder(qc, ALICE, charlie_qubits[0], charlie_size, reverse=False, internal_copy=True)
    cnot_ladder(qc, BOB, debbie_qubits[0], debbie_size, reverse=False, internal_copy=True)

    # Apply the settings for Alice/Charlie and Bob/Debbie
    apply_setting(qc, ALICE, alice_setting, angles[alice_setting], alice_creg, charlie_qubits, charlie_size)
    apply_setting(qc, BOB, bob_setting, (beta - angles[bob_setting]), bob_creg, debbie_qubits, debbie_size)

    return qc


#######################################################################################################################
# FILE I/O
#######################################################################################################################
def save_data(
    results: dict,
    backend: Backend,
    friend_sizes: list[int],
    num_trials: int,
    shots: int,
    backend_name: Optional[str] = None,
    data_path: str = DATA_PATH,
):
    """Writes data to a file name format of `<MACHINE_NAME>_qubits_<NUM_QUBITS>_trial_<NUM_TRIALS>_shots_<NUM_SHOTS>`."""
    if backend_name is None:
        backend_name = backend.name

    for friend_size in friend_sizes:
        qubits = friend_size

        # If not output file name is given, use this format.
        output_file_name = f"{backend_name}_qubits_{qubits}_trial_{num_trials}_shots_{shots}.pickle"
        output_path = os.path.join(data_path, output_file_name)

        print(f"Writing data to: {output_path}")
        with open(output_path, "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_experiments(machine_name: str, friend_sizes: list[int], num_trials: int, shots: int, data_path: str = DATA_PATH) -> dict:
    """Load experiments from multiple files."""
    all_results = {
        fs: {inequality: [] for inequality in ["semi_brukner"]}
        for fs in friend_sizes
    }

    for friend_size in friend_sizes:
        for trial in range(1, num_trials+1):
            with open(os.path.join(data_path, f"{machine_name}_qubits_{friend_size}_trial_{trial}_shots_{shots}.pickle"), "rb") as file:
                results = pickle.load(file)
            violations = compute_inequalities(decode_results(results, charlie_size=friend_size, debbie_size=1))
            for key in violations:
                all_results[friend_size][key].append(violations[key])
    return all_results


#######################################################################################################################
# INEQUALITIES
#######################################################################################################################
def compute_inequalities(results, verbose=False) -> dict[str, float]:
    """Compute the semi-Brukner inequalities."""
    A1B2 = double_expect((PEEK, REVERSE_1), results)
    A1B3 = double_expect((PEEK, REVERSE_2), results)

    A3B2 = double_expect((REVERSE_2, REVERSE_1), results)
    A3B3 = double_expect((REVERSE_2, REVERSE_2), results)

    # Eq. (18) from [1].
    semi_brukner = -A1B2 + A1B3 - A3B2 - A3B3 - 2

    if verbose:
        print(f"{semi_brukner=} -- is violated: {semi_brukner > 0}")

    return {"semi_brukner": semi_brukner}


#######################################################################################################################
# EXPERIMENT
#######################################################################################################################
def run_experiment(
    backend: Backend,
    backend_name: str,
    noise_model: Optional[NoiseModel],
    friend_sizes: list[int],
    shots: int,
    num_trials: int = 1,
    verbose: bool = False,
    save: bool = False,
    optimize: bool = False,
) -> dict:
    """Run the main experiment for a specified backend."""
    all_results = {
        fs: {inequality: [] for inequality in ["semi_brukner"]}
        for fs in friend_sizes
    }

    # Create directory to save results.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_dir_name = f"{backend_name}_{timestamp}"
    new_dir_path = os.path.join(DATA_PATH, new_dir_name)

    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    for friend_size in friend_sizes:
        print(f"{friend_size=}")
        for trial in range(num_trials):
            results = generate_all_experiments(
                backend=backend,
                noise_model=noise_model,
                shots=shots,
                angles=ANGLES,
                beta=BETA,
                charlie_size=friend_size,
                debbie_size=1,
                optimize=optimize,
            )
            violations = compute_inequalities(decode_results(results, charlie_size=friend_size, debbie_size=1), verbose=verbose)
            for key in violations:
                all_results[friend_size][key].append(violations[key])

            # Save data after each trial of each friend_size
            if save:
                save_data(results=results,
                          backend=backend,
                          friend_sizes=[friend_size],
                          num_trials=trial + 1,
                          shots=shots,
                          data_path=new_dir_path,
                          backend_name=backend_name)
    return all_results


def plot_results(
    ax,
    results: dict,
    friend_sizes: list[int],
    plot_title: str,
    plot_error_bars: bool = False,
    color: str = "tab:blue",
    label: Optional[str] = None,
    show_legend: bool = False,
    marker: str = "o",
    marker_size: float = 10,
    line_width: float = 2.5,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    # Compute averages and standard deviations
    avg_results = {}
    std_results = {}
    for fs in results:
        avg_results[fs] = {}
        std_results[fs] = {}
        for key in results[fs]:
            avg_results[fs][key] = np.mean(results[fs][key])
            if plot_error_bars:
                std_results[fs][key] = np.std(results[fs][key])

    for _, key in enumerate(["semi_brukner"]):
        means = [np.mean(results[fs][key]) for fs in friend_sizes]
        errors = [np.std(results[fs][key]) for fs in friend_sizes] if plot_error_bars else None
        ax.plot(
            friend_sizes,
            means,
            label=label,
            marker=marker,
            markersize=marker_size,
            linestyle="-",
            linewidth=line_width,
            color=color,
        )
        if plot_error_bars:
            ax.errorbar(friend_sizes, means, yerr=errors, fmt="none", color=color, capsize=5, elinewidth=line_width)

    ax.axhline(0.380364, color="tab:green", linestyle="dashed", label="_nolegend_", linewidth=line_width)
    ax.axhline(0, color="tab:red", linestyle="dotted", label="_nolegend_", linewidth=line_width)

    ax.set_xticks(friend_sizes)
    ax.set_title(plot_title)
    ax.grid(True)

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)

    if show_legend:
        ax.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended Wigner's friend scenario (EWFS).")
    parser.add_argument("-backend", help="IBM hardware backend.", required=True, type=str)
    parser.add_argument("-shots", help="Number of shots", type=int, default=10_000)
    parser.add_argument("-trials", help="Number of trials", type=int, default=10)
    parser.add_argument("-friend_max", help="Max friend size", type=int, default=11)
    parser.add_argument("-optimize", help="Turn off optimization", type=bool, default=True)
    parser.add_argument("-verbose", help="Verbose output", type=bool, default=True)
    parser.add_argument("-save", help="Save data", type=bool, default=True)

    args = parser.parse_args()

    IBM_PROVIDER_TOKEN="f93ccee55d555d956b5bd12641edb2b79f29c28235f064dccb3cfba3109d144271ce5a31fa05b26430abe31be8e87cf50f542cdf84abe3473c2f58fa7a1f56c5"
    QiskitRuntimeService.save_account(channel="ibm_quantum", token=IBM_PROVIDER_TOKEN, set_as_default=True, overwrite=True)
    service = QiskitRuntimeService()

    backend = service.backend(args.backend)
    friend_sizes = list(range(1, args.friend_max))

    results = run_experiment(
        backend=backend,
        backend_name=args.backend,
        noise_model=None,
        friend_sizes=friend_sizes,
        num_trials=args.trials,
        shots=args.shots,
        verbose=args.verbose,
        optimize=args.optimize,
        save=args.save,
    )
