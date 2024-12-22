"""Extended Wigner's friend scenario (EWFS)" functionality."""
from enum import Enum
import numpy as np
import random

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from ewfs.circuit import (
    cnot_ladder,
    cnot_ladder_random,
    ewfs_rotation,
)


# Observers for scenario are Alice and Bob.
class Observer(Enum):
    ALICE = 0
    BOB = 1

# "Super"-observers (Alice and Bob).
ALICE = Observer.ALICE.value
BOB = Observer.BOB.value

# (Optimized) angles and beta term used for Alice and Bob measurement operators. Adapted from arXiv:1907.05607. Note
# that despite the fact that degrees are used, we need to convert this to radians.
DEFAULT_ANGLES = {
    "peek": np.deg2rad(40),
    "reverse_1": np.deg2rad(230),
    "reverse_2": np.deg2rad(310),
}
DEFAULT_BETA = np.deg2rad(220)


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

        # Settings for Alice and Bob.
        self.alice_setting = alice_setting
        self.bob_setting = bob_setting

        # Strategy for the friend.
        self.strategy = strategy

        self.charlie_size = charlie_size
        self.debbie_size = debbie_size

        self.angles = angles or DEFAULT_ANGLES
        self.beta = beta or DEFAULT_BETA

        self.alice_creg = None
        self.bob_creg = None

        self.charlie_qubits = None
        self.debbie_qubits = None

        self._validate()

    def circuit(self) -> QuantumCircuit:
        """Generate the circuit for extended Wigner's friend scenario."""
        # Define quantum registers
        alice_size, bob_size = 1, 1
        alice, bob, charlie, debbie, measurement = self._initialize_measurement_registers(alice_size, bob_size)

        # Create the Quantum Circuit with the defined registers
        qc = QuantumCircuit(alice, bob, charlie, debbie, measurement)

        self._prepare_bipartite_system(qc)
        self._prepare_rotations(qc)
        self._apply_cnot_ladders(qc)

        # Apply the setting for Alice/Charlie.
        self._apply_setting(
            qc=qc, 
            observer=ALICE,
            setting=self.alice_setting,
            angle=self.angles[self.alice_setting],
            observer_creg=self.alice_creg,
            friend_qubits=self.charlie_qubits,
            friend_size=self.charlie_size
        )
        # Apply the setting for Bob/Debbie.
        self._apply_setting(
            qc=qc, 
            observer=BOB,
            setting=self.bob_setting,
            angle=(self.beta - self.angles[self.bob_setting]),
            observer_creg=self.bob_creg,
            friend_qubits=self.debbie_qubits, 
            friend_size=self.debbie_size
        )

        return qc

    def _validate(self) -> None:
        """Validate the settings for the EWFS scenario."""

        # Validate the settings.
        if self.alice_setting not in ["peek", "reverse_1", "reverse_2"]:
            raise ValueError(f"Alice's setting: {self.alice_setting} is not defined.")
        if self.bob_setting not in ["peek", "reverse_1", "reverse_2"]:
            raise ValueError(f"Bob's setting: {self.bob_setting} is not defined.")

        # Validate the strategy.
        if self.strategy not in ["majority_vote", "random"]:
            raise ValueError(f"Strategy: {self.strategy} is not defined.")

        # Validate the sizes.
        if self.charlie_size < 1:
            raise ValueError(f"Charlie's size: {self.charlie_size} is invalid.")
        if self.debbie_size < 1:
            raise ValueError(f"Debbie's size: {self.debbie_size} is invalid.")  

    def _initialize_measurement_registers(self, alice_size: int, bob_size: int) -> None:
        """Initialize the classical measurement registers based on the strategy."""
        sys_size = alice_size + bob_size
        meas_size = 2

        self.charlie_qubits = list(range(sys_size, (sys_size + self.charlie_size)))
        self.debbie_qubits = list(range(sys_size + self.charlie_size, sys_size + (self.charlie_size + self.debbie_size)))

        if self.strategy == "majority_vote":
            if self.alice_setting == "peek" and self.bob_setting != "peek":
                measurement = ClassicalRegister(self.charlie_size + 1, name="measurement")
                self.alice_creg = list(range(self.charlie_size))
                self.bob_creg = self.charlie_size
            else:
                measurement = ClassicalRegister(meas_size, name="measurement")
                self.alice_creg, self.bob_creg = 0, 1

        elif self.strategy == "random":
            measurement = ClassicalRegister(sys_size, name="measurement")
            self.alice_creg, self.bob_creg = 0, 0

        alice, bob, charlie, debbie = [
            QuantumRegister(size, name=name) 
            for size, name in zip([alice_size, bob_size, self.charlie_size, self.debbie_size], 
                                ["Alice's qubit", "Bob's qubit", "Charlie", "Debbie"])
        ]
        return alice, bob, charlie, debbie, measurement

    def _prepare_bipartite_system(self, qc: QuantumCircuit) -> None:
        """Generates the state: 1/sqrt(2) * (|01> - |10>)"""
        qc.x(ALICE)
        qc.x(BOB)
        qc.h(ALICE)
        qc.cx(ALICE, BOB)

    def _prepare_rotations(self, qc: QuantumCircuit) -> None:
        """Apply rotations for Alice and Bob based on their settings."""
        self._ewfs_rotation(qc, ALICE, self.angles["peek"])
        self._ewfs_rotation(qc, BOB, self.beta - self.angles["peek"])

    def _apply_cnot_ladders(self, qc: QuantumCircuit) -> None:
        """Apply the CNOT ladders based on the strategy."""
        if self.strategy == "majority_vote":
            cnot_ladder(qc, ALICE, self.charlie_qubits[0], self.charlie_size, reverse=False, internal_copy=True)
            cnot_ladder(qc, BOB, self.debbie_qubits[0], self.debbie_size, reverse=False, internal_copy=True)
        elif self.strategy == "random":
            cnot_ladder(qc, ALICE, self.charlie_qubits[0], self.charlie_size)
            cnot_ladder(qc, BOB, self.debbie_qubits[0], self.debbie_size)

    def _apply_observer_rotation(self, qc: QuantumCircuit, observer: int, angle: float) -> None:
        """Apply the observer rotation based on the setting."""
        # For either REVERSE_1 or REVERSE_2, apply the appropriate angle rotations.
        # Note that in this case, the rotation should occur on the observer's qubit.
        if observer is ALICE:
            qc.h(ALICE)
            qc.rz(self.angles["peek"], ALICE)

        if observer is BOB:
            qc.h(BOB)
            qc.rz((self.beta - self.angles["peek"]), BOB)
        self._ewfs_rotation(qc, observer, angle)

    def _apply_setting(
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
                self._apply_observer_rotation(qc, observer, angle)
                qc.measure(observer, observer_creg)

            elif self.strategy == "random":
                cnot_ladder(qc, observer, friend_qubits[0], friend_size)
                self._apply_observer_rotation(qc, observer, angle)
                qc.measure(observer, observer)

    def _ewfs_rotation(self, qc: QuantumCircuit, qubit: int, angle: float) -> None:
        """
        Apply an EWFS-specific rotation to a qubit.

        Args:
            qc (QuantumCircuit): The quantum circuit to apply the rotation to.
            qubit (int): The index of the qubit to rotate.
            angle (float): The angle of rotation in radians.
        """
        qc.rz(-angle, qubit)
        qc.h(qubit)



def double_expect(counts: dict[str, float]) -> float:
    """Expectation value of product of two operators."""
    # <AB> = P(00) - P(01) - P(10) + P(11)
    return counts.get("00", 0) - counts.get("01", 0) - counts.get("10", 0) + counts.get("11", 0)

def compute_inequalities(results, verbose=False) -> dict[str, float]:
    """Compute the semi-Brukner inequalities."""
    A1B2 = double_expect(results[("peek", "reverse_1")])
    A1B3 = double_expect(results[("peek", "reverse_2")])
    
    A3B2 = double_expect(results[("reverse_2", "reverse_1")])
    A3B3 = double_expect(results[("reverse_2", "reverse_2")])
    
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

