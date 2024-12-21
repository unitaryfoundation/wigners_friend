"""Extended Wigner's friend scenario (EWFS)" functionality."""
from enum import Enum
import numpy as np
import os
import random
from collections import defaultdict
from datetime import datetime

import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
#from file_io import save_data
from ewfs.circuit import (
    prepare_bipartite_system,
    cnot_ladder,
    cnot_ladder_random,
    ewfs_rotation,
)


DATA_PATH = os.path.join("../ewfs_release", "data")


# Observers for scenario are Alice and Bob.
class Observer(Enum):
    ALICE = 0
    BOB = 1


# "Super"-observers (Alice and Bob).
ALICE = Observer.ALICE.value
BOB = Observer.BOB.value


def compute_inequalities(results, verbose=False) -> dict[str, float]:
    """Compute the semi-Brukner inequalities."""
    A1B2 = double_expect(("peek", "reverse_1"), results)
    A1B3 = double_expect(("peek", "reverse_2"), results)

    A3B2 = double_expect(("reverse_2", "reverse_1"), results)
    A3B3 = double_expect(("reverse_2", "reverse_2"), results)

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


def decode_results(results: dict, charlie_size: int, debbie_size: int = 1) -> dict[str, float]:
    """Take majority vote of measurement bit-strings."""
    decoded_results = {}

    # For each setting, there is a dictionary of measurement results.
    for setting in results:
        if setting == ("peek", "reverse_1") or setting == ("peek", "reverse_2"):
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


class EWFS:
    def __init__(
        self,
        alice_setting: str,
        bob_setting: str,
        strategy: str,
        charlie_size: int,
        debbie_size: int,
        angles: list[float] | None = None,
        beta: float | None = None,
    ) -> None:
        self.alice_setting = alice_setting
        self.bob_setting = bob_setting

        if strategy not in ["majority_vote", "random"]:
            raise ValueError(f"Strategy: {strategy} is not defined.")
        self.strategy = strategy

        self.charlie_size = charlie_size
        self.debbie_size = debbie_size

        self.angles = angles if angles else {
            "peek": np.deg2rad(40),
            "reverse_1": np.deg2rad(230),
            "reverse_2": np.deg2rad(310),
        }
        self.beta = beta if beta else np.deg2rad(220)

    def circuit(self) -> QuantumCircuit:
        """Generate the circuit for extended Wigner's friend scenario."""
        # Define quantum registers
        alice_size, bob_size = 1, 1
        sys_size = alice_size + bob_size
        meas_size = 2

        alice, bob, charlie, debbie = [
            QuantumRegister(size, name=name) 
            for size, name in zip([alice_size, bob_size, self.charlie_size, self.debbie_size], 
                                ["Alice's qubit", "Bob's qubit", "Charlie", "Debbie"])
        ]

        if strategy == "majority_vote":
            if self.alice_setting == "peek" and self.bob_setting != "peek":
                measurement = ClassicalRegister(self.charlie_size + 1, name="measurement")
                alice_creg = list(range(self.charlie_size))
                bob_creg = self.charlie_size
            else:
                measurement = ClassicalRegister(meas_size, name="measurement")
                alice_creg = 0
                bob_creg = 1
        elif strategy == "random":
            measurement = ClassicalRegister(sys_size, name="measurement")
            alice_creg, bob_creg = 0, 0

        # Create the Quantum Circuit with the defined registers
        qc = QuantumCircuit(alice, bob, charlie, debbie, measurement)

        charlie_qubits = list(range(sys_size, (sys_size + self.charlie_size)))
        debbie_qubits = list(range(sys_size + self.charlie_size, sys_size + (self.charlie_size + self.debbie_size)))

        # Prepare the bipartite quantum system
        prepare_bipartite_system(qc, ALICE, BOB)

        # Rotations for measurement.
        ewfs_rotation(qc, ALICE, self.angles["peek"])
        ewfs_rotation(qc, BOB, self.beta - self.angles["peek"])

        # Apply the CNOT ladder for Alice-Charlie and Bob-Debbie
        if strategy == "majority_vote":
            cnot_ladder(qc, ALICE, charlie_qubits[0], self.charlie_size, reverse=False, internal_copy=True)
            cnot_ladder(qc, BOB, debbie_qubits[0], self.debbie_size, reverse=False, internal_copy=True)
        elif strategy == "random":
            cnot_ladder_random(qc, ALICE, charlie_qubits[0], self.charlie_size)
            cnot_ladder_random(qc, BOB, debbie_qubits[0], self.debbie_size)

        # Apply the settings for Alice/Charlie and Bob/Debbie
        self.apply_setting(
            qc=qc, 
            observer=ALICE,
            setting=self.alice_setting,
            angle=self.angles[self.alice_setting],
            observer_creg=alice_creg,
            friend_qubits=charlie_qubits,
            friend_size=self.charlie_size
        )
        self.apply_setting(
            qc=qc, 
            observer=BOB,
            setting=self.bob_setting,
            angle=(self.beta - self.angles[self.bob_setting]),
            observer_creg=bob_creg,
            friend_qubits=debbie_qubits, 
            friend_size=self.debbie_size
        )

        return qc

    def apply_setting(
            self, 
            qc: QuantumCircuit,
            observer: int,
            setting: int,
            angle: float,
            observer_creg: list[int] | int,
            friend_qubits: list[int],
            friend_size: int
        ):
        """Apply either the PEEK or REVERSE_1/REVERSE_2 settings."""
        if setting == "peek":
            if self.strategy == "majority_vote":
                # Ask friend for the outcome.
                qc.measure(friend_qubits, observer_creg)
            elif self.strategy == "random":
                random_offset = random.randint(0, friend_size - 1)
                qc.measure(friend_qubits[0] + random_offset, observer)

        elif setting in ["reverse_1", "reverse_2"]:
            qc.barrier(observer, friend_qubits)
            if self.strategy == "majority_vote":
                cnot_ladder(qc, observer, friend_qubits[0], friend_size, reverse=True, internal_copy=True)
            elif self.strategy == "random":
                cnot_ladder_random(qc, observer, friend_qubits[0], friend_size)

            # For either REVERSE_1 or REVERSE_2, apply the appropriate angle rotations.
            # Note that in this case, the rotation should occur on the observer's qubit.
            if observer is ALICE:
                qc.h(ALICE)
                qc.rz(self.angles["peek"], ALICE)

            if observer is BOB:
                qc.h(BOB)
                qc.rz((self.beta - self.angles["peek"]), BOB)
            ewfs_rotation(qc, observer, angle)

            if self.strategy == "majority_vote":
                qc.measure(observer, observer_creg)
            elif self.strategy == "random":
                qc.measure(observer, observer)



if __name__ == "__main__":
    # Experimental settings:
    shots = 10_000
    num_trials = 2
    friend_sizes = range(1, 3)
    strategy = "majority_vote"
    save = False

    # Set backend and noise model:
    backend_name = "aer_simulator"
    backend = qiskit_aer.Aer.get_backend(backend_name)
    noise_model = None

    # Create timestamped directory to save results:
    DATA_PATH = os.path.join("..", "data")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_dir_name = f"{strategy}_{backend_name}_{timestamp}"
    new_dir_path = os.path.join(DATA_PATH, new_dir_name)

    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    all_experiment_combos = [
        ["peek", "reverse_1"],
        ["peek", "reverse_2"],
        ["reverse_2", "reverse_1"],
        ["reverse_2", "reverse_2"],
    ]

    # The tasks dictionary has a key that corresponds to the qubit
    # system size with an associated value of the task for that size.
    tasks = defaultdict(dict)

    # Construct all circuits to be run over all qubit sizes.
    for charlie_size in friend_sizes:
        for trial in range(1, num_trials + 1):
            circuits = {}
        
            # Create circuits for each EWFS setting.
            # Note, we remove barriers in the Qiskit circuit.
            for alice_setting, bob_setting in all_experiment_combos:
                circuit = EWFS(
                    alice_setting=alice_setting,
                    bob_setting=bob_setting,
                    strategy=strategy,
                    charlie_size=charlie_size,
                    debbie_size=1).circuit()
                circuits[(alice_setting, bob_setting)] = circuit
        
            # Use backend to transpile circuits.
            transpiled_circuits = {
                k:  qiskit.transpile(circuit, backend, optimization_level=0)
                for k, circuit in circuits.items()
            }
        
            # Run task.
            print(f"Trial {trial} out of {num_trials} for task of {charlie_size=}")
            tasks[trial][charlie_size] = backend.run(
                list(transpiled_circuits.values()),
                shots=shots,
                verbatim=True,
            )
            print(f"Task with task ID: {tasks[trial][charlie_size].job_id()}\n")

    results = {}
    post_processed_results = {
        fs: {inequality: [] for inequality in ["semi_brukner"]}
        for fs in friend_sizes
    }
    for trial in tasks:
        for charlie_size, task in tasks[trial].items():
            print(f"Processing trial {trial} for task with task ID: {task.job_id()}")

            result = task.result()
            for key, count in zip(transpiled_circuits.keys(), result.get_counts()):
                probabilities = {k[::-1]: v / shots for k, v in count.items()}
                results[key] = probabilities
            print(f"Results: {results}")
        
            # Compute violations from result counts.
            violations = compute_violations(
                results=results,
                charlie_size=charlie_size,
                debbie_size=1,
                strategy=strategy,
                verbose=True
            )
            print(f"Violations: {violations}\n")
        
            for key in violations:
                post_processed_results[charlie_size][key].append(violations[key])
        
            # # Save data after each trial of each charlie_size.
            # if save:
            #     save_data(results=results,
            #             friend_size=charlie_size,
            #             trial=trial,
            #             shots=shots,
            #             data_path=new_dir_path,
            #             backend_name=backend_name)