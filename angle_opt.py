from typing import Optional

from enum import Enum
from datetime import datetime

import random
import numpy as np
import os

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
from qiskit.providers import Backend
from qiskit_aer import AerSimulator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from tqdm import tqdm

DATA_PATH = os.path.join("..", "data")

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


def prepare_bipartite_system(qc: QuantumCircuit):
    """Generates the state: 1/sqrt(2) * (|01> - |10>)"""
    qc.x(ALICE)
    qc.x(BOB)
    qc.h(ALICE)
    qc.cx(ALICE, BOB)


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

def cnot_ladder_random(qc: QuantumCircuit, observer: int, friend_qubit: int, friend_size: int):
    """CNOT ladder circuit (GHZ without Hadamard) for random strategy."""
    for i in range(friend_size):
        qc.cx(observer, friend_qubit + i)


def ewfs_rotation(qc: QuantumCircuit, qubit: int, angle: float):
    qc.rz(-angle, qubit)
    qc.h(qubit)

def apply_setting(qc: QuantumCircuit,
                  strategy: str,
                  observer: int,
                  setting: int,
                  angle: float, 
                  observer_creg: list[int] | int,
                  friend_qubits: list[int],
                  friend_size: int):
    """Apply either the PEEK or REVERSE_1/REVERSE_2 settings."""
    if setting is PEEK:
        if strategy == "majority_vote":
            # Ask friend for the outcome.
            qc.measure(friend_qubits, observer_creg)
        elif strategy == "random":
            random_offset = random.randint(0, friend_size - 1)
            qc.measure(friend_qubits[0] + random_offset, observer)

    elif setting in [REVERSE_1, REVERSE_2]:
        if strategy == "majority_vote":
            cnot_ladder(qc, observer, friend_qubits[0], friend_size, reverse=True, internal_copy=True)
        elif strategy == "random":
            cnot_ladder_random(qc, observer, friend_qubits[0], friend_size)

        # For either REVERSE_1 or REVERSE_2, apply the appropriate angle rotations.
        # Note that in this case, the rotation should occur on the observer's qubit.
        if observer is ALICE:
            qc.h(ALICE)
            qc.rz(ANGLES[1], ALICE)

        if observer is BOB:
            qc.h(BOB)
            qc.rz((BETA - ANGLES[1]), BOB)
        ewfs_rotation(qc, observer, angle)

        if strategy == "majority_vote":
            qc.measure(observer, observer_creg)
        elif strategy == "random":
            qc.measure(observer, observer)


def ewfs(alice_setting: int,
        bob_setting: int,
        strategy: str,
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
                              ["Alice's qubit", "Bob's qubit", "Charlie", "Debbie"])
    ]

    if strategy == "majority_vote":
        if (alice_setting == PEEK and bob_setting != PEEK):
            measurement = ClassicalRegister(charlie_size + 1, name="Measurement")
            alice_creg = list(range(charlie_size))
            bob_creg = charlie_size
        else:
            measurement = ClassicalRegister(meas_size, name="Measurement")
            alice_creg = 0
            bob_creg = 1
    elif strategy == "random":
        measurement = ClassicalRegister(sys_size, name="Measurement")
        alice_creg, bob_creg = 0, 0
    else:
        raise ValueError(f"Strategy: {strategy} is not defined.")

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
    if strategy == "majority_vote":
        cnot_ladder(qc, ALICE, charlie_qubits[0], charlie_size, reverse=False, internal_copy=True)
        cnot_ladder(qc, BOB, debbie_qubits[0], debbie_size, reverse=False, internal_copy=True)
    elif strategy == "random":
        cnot_ladder_random(qc, ALICE, charlie_qubits[0], charlie_size)
        cnot_ladder_random(qc, BOB, debbie_qubits[0], debbie_size)

    # Apply the settings for Alice/Charlie and Bob/Debbie
    apply_setting(qc, strategy, ALICE, alice_setting, angles[alice_setting], alice_creg, charlie_qubits, charlie_size)
    apply_setting(qc, strategy, BOB, bob_setting, (beta - angles[bob_setting]), bob_creg, debbie_qubits, debbie_size)

    return qc

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

def compute_violations(results: dict, charlie_size: int, debbie_size: int, strategy: str, verbose: bool = False) -> dict[str, float]:
    """Compute violation values based on strategy."""
    if strategy == "random":
        return compute_inequalities(results=results, verbose=verbose)
    elif strategy == "majority_vote":
        return compute_inequalities(decode_results(results=results, charlie_size=charlie_size, debbie_size=debbie_size), verbose=verbose)
    raise ValueError(f"Strategy: {strategy} not defined.")

def generate_all_experiments(
    backend: Backend,
    backend_name: Optional[str],
    noise_model: Optional[NoiseModel],
    shots: int,
    strategy: str,
    angles: dict,
    beta: float,
    charlie_size: int,
    debbie_size: int,
    optimize: bool = True
) -> dict[tuple[Observer, Observer], list[float]]:
    """Generate probabilities for all combinations of experimental settings."""
    all_experiment_combos = [[PEEK, REVERSE_1], [PEEK, REVERSE_2], [REVERSE_2, REVERSE_1], [REVERSE_2, REVERSE_2]]

    results = {}
    circuits = {}
    for alice, bob in all_experiment_combos:
        circuits[(alice, bob)] = ewfs(alice, bob, strategy, angles, beta, charlie_size, debbie_size)
    # Define pass manager for optimizing over single-qubit gates.
    pm = PassManager()
    pm.append(Optimize1qGatesDecomposition(["u1", "u2", "u3"]))

    # If optimize is True, we:
    # 1. Arrange the layout in a linear-like way depending on the architecture.
    # 2. Optimize single-qubit gate decompositions.
    if optimize:
        initial_layout = calculate_optimal_qubit_layout(backend_name, charlie_size)
        transpiled_circuits = transpile(
            list(circuits.values()),
            backend=backend,
            optimization_level=0,
            initial_layout=initial_layout,
        )
        transpiled_circuits = pm.run(transpiled_circuits)
    # Otherwise, we simply transpile the circuit without any optimizations.
    else:
        transpiled_circuits = transpile(
            list(circuits.values()),
            optimization_level=0,
            backend=backend
        )
    circuits = {key: circuit for key, circuit in zip(circuits.keys(), transpiled_circuits)}

    # Assuming the backend is AerSimulator if a noise model is provided
#    if noise_model:
#        backend = AerSimulator(noise_model=noise_model)

    result = backend.run(list(circuits.values()),
                         #noise_model=noise_model,
                         #basis_gates=noise_model.basis_gates if noise_model is not None else None,
                         shots=shots).result()
    # Convert counts to probabilities
    for key, count in zip(circuits.keys(), result.get_counts()):
        probabilities = {k[::-1]: v / shots for k, v in count.items()}
        results[key] = probabilities

    return results


def run_experiment(
    backend: Backend,
    noise_model: Optional[NoiseModel],
    friend_sizes: list[int],
    shots: int,
    strategy: str,
    backend_name: Optional[str] = None,
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
    if backend_name is None:
        backend_name = backend.name

    # Create timestamped directory to save results.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_dir_name = f"{strategy}_{backend_name}_{timestamp}"
    new_dir_path = os.path.join(DATA_PATH, new_dir_name)

    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    for friend_size in friend_sizes:
        #print(f"{friend_size=}")
        for trial in range(num_trials):
            results = generate_all_experiments(
                backend=backend,
                backend_name=backend_name,
                noise_model=noise_model,
                shots=shots,
                angles=ANGLES,
                beta=BETA,
                charlie_size=friend_size,
                debbie_size=1,
                optimize=optimize,
                strategy=strategy,
            )
            violations = compute_violations(results=results, charlie_size=friend_size, debbie_size=1, strategy=strategy, verbose=verbose)
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



opt_violation = -np.inf
opt_angles = {'angle_peek': 185, 'angle_rev1': 23, 'angle_rev2': 112, 'beta': 0}

# Define the range for each parameter, using full range for beta, and fixed values for others as an example
angle_peek_range = range(0, 360, 10)
angle_rev1_range = range(0, 360, 10)
angle_rev2_range = range(0, 360, 10)
beta_range = range(0, 360, 10)

backend = AerSimulator()
noise_model = None
friend_sizes = range(1, 2)
num_trials = 1
shots = 10_000

# Grid search over all angles
for angle_peek in tqdm(angle_peek_range, desc="Peek Angles"):
    for angle_rev1 in tqdm(angle_rev1_range, desc="Rev1 Angles"):
        for angle_rev2 in tqdm(angle_rev2_range, desc="Rev2 Angles"):
            for beta in tqdm(beta_range, desc="Beta Angles"):
                ANGLES = {PEEK: np.deg2rad(angle_peek), REVERSE_1: np.deg2rad(angle_rev1), REVERSE_2: np.deg2rad(angle_rev2)}
                BETA = np.deg2rad(beta)

                results = run_experiment(
                    backend=backend,
                    backend_name="fake_mumbai",
                    noise_model=noise_model,
                    friend_sizes=friend_sizes,
                    num_trials=num_trials,
                    shots=shots,
                    verbose=False,
                    strategy="majority_vote",
                )
                violation = results[1]["semi_brukner"][0]
                if opt_violation < violation:
                    opt_violation = violation
                    opt_angles.update({'angle_peek': angle_peek, 'angle_rev1': angle_rev1, 'angle_rev2': angle_rev2, 'beta': beta})
                    print(f"New best violation: {violation} at Angles: {opt_angles}")

# Print final optimal results
print(f"Optimal angles and violation: {opt_angles}, Violation: {opt_violation}")
