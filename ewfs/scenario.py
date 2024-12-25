"""Extended Wigner's friend scenario (EWFS)" functionality."""
import numpy as np
import random

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from ewfs.circuit import cnot_ladder


# "Super"-observers (Alice and Bob).
ALICE, BOB = 0, 1

# Experiment settings (peek, reverse_1, and reverse_2).
SETTINGS = ["peek", "reverse_1", "reverse_2"]
# Supported strategies for the super-observers
STRATEGIES = ["majority_vote", "random"]

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
        alice_size: int = 1,
        bob_size: int = 1,
        angles: dict[str, float] | None = None,
        beta: float | None = None,
    ) -> None:
        """Initialize the extended Wigner's friend scenario."""
        # Settings for Alice and Bob.
        self.alice_setting = alice_setting
        self.bob_setting = bob_setting

        # Strategy for the friend.
        self.strategy = strategy

        # Sizes of the qubit systems.
        self.alice_size = alice_size
        self.bob_size = bob_size
        self.charlie_size = charlie_size
        self.debbie_size = debbie_size

        self.sys_size = self.alice_size + self.bob_size
        self.meas_size = 2

        # Angles for Alice and Bob measurements.
        self.angles = angles or DEFAULT_ANGLES
        self.beta = beta or DEFAULT_BETA

        self.charlie_qubits = list(range(self.sys_size, (self.sys_size + self.charlie_size)))
        self.debbie_qubits = list(range(self.sys_size + self.charlie_size, self.sys_size + (self.charlie_size + self.debbie_size)))

        self._validate()

    def circuit(self) -> QuantumCircuit:
        """Generate the circuit for extended Wigner's friend scenario."""
        # Define quantum registers
        alice, bob, charlie, debbie, measurement, alice_creg, bob_creg = self._initialize_measurement_registers()

        # Create the Quantum Circuit with the defined registers
        qc = QuantumCircuit(alice, bob, charlie, debbie, measurement)

        # Prepare the bipartite system for the observers (Alice and Bob).
        self._prepare_bipartite_system(qc)

        # Perform the rotations for Alice and Bob based on their settings.
        self._ewfs_rotation(qc, ALICE, self.angles["peek"])
        self._ewfs_rotation(qc, BOB, self.beta - self.angles["peek"])

        # Apply the CNOT ladders based on the strategy.
        self._apply_cnot_ladders(qc)

        # Apply the setting for Alice/Charlie.
        self._apply_setting(
            qc=qc, 
            observer=ALICE,
            setting=self.alice_setting,
            angle=self.angles[self.alice_setting],
            observer_creg=alice_creg,
            friend_qubits=self.charlie_qubits,
            friend_size=self.charlie_size
        )
        # Apply the setting for Bob/Debbie.
        self._apply_setting(
            qc=qc, 
            observer=BOB,
            setting=self.bob_setting,
            angle=(self.beta - self.angles[self.bob_setting]),
            observer_creg=bob_creg,
            friend_qubits=self.debbie_qubits, 
            friend_size=self.debbie_size
        )

        return qc

    def _validate(self) -> None:
        """Validate the settings for the EWFS scenario."""

        # Validate the settings.
        if self.alice_setting not in SETTINGS or self.bob_setting not in SETTINGS:
            raise ValueError(f"Super-observer setting is not defined. Supported settings: {SETTINGS}")

        # Validate the strategy.
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Strategy is not defined. Supported strategies: {STRATEGIES}")

        # Validate the friend qubit register sizes.
        if self.charlie_size < 1 or self.debbie_size < 1:
            raise ValueError("Friend size must be at least one qubit.")

    def _initialize_measurement_registers(self) -> None:
        """Initialize the classical measurement registers based on the strategy."""
        if self.strategy == "majority_vote":
            if self.alice_setting == "peek" and self.bob_setting != "peek":
                measurement = ClassicalRegister(self.charlie_size + 1, name="measurement")
                alice_creg = list(range(self.charlie_size))
                bob_creg = self.charlie_size
            else:
                measurement = ClassicalRegister(self.meas_size, name="measurement")
                alice_creg, bob_creg = 0, 1

        elif self.strategy == "random":
            measurement = ClassicalRegister(self.sys_size, name="measurement")
            alice_creg, bob_creg = 0, 0

        alice, bob, charlie, debbie = [
            QuantumRegister(size, name=name) 
            for size, name in zip(
                [self.alice_size, self.bob_size, self.charlie_size, self.debbie_size], 
                ["Alice's qubit", "Bob's qubit", "Charlie", "Debbie"]
            )
        ]
        return alice, bob, charlie, debbie, measurement, alice_creg, bob_creg

    def _prepare_bipartite_system(self, qc: QuantumCircuit) -> None:
        """Generates the state: 1/sqrt(2) * (|01> - |10>)"""
        qc.x(ALICE)
        qc.x(BOB)
        qc.h(ALICE)
        qc.cx(ALICE, BOB)

    def _apply_cnot_ladders(self, qc: QuantumCircuit) -> None:
        """Apply the CNOT ladders based on the strategy."""
        if self.strategy == "majority_vote":
            cnot_ladder(qc, ALICE, self.charlie_qubits[0], self.charlie_size, reverse=False, internal_copy=True)
            cnot_ladder(qc, BOB, self.debbie_qubits[0], self.debbie_size, reverse=False, internal_copy=True)
        elif self.strategy == "random":
            cnot_ladder(qc, ALICE, self.charlie_qubits[0], self.charlie_size)
            cnot_ladder(qc, BOB, self.debbie_qubits[0], self.debbie_size)

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
            self._apply_peek(qc, observer, observer_creg, friend_qubits, friend_size)
        elif setting in ["reverse_1", "reverse_2"]:
            self._apply_reverse(qc, observer, observer_creg, friend_qubits, friend_size, angle)

    def _apply_peek(self, qc: QuantumCircuit, observer: int, observer_creg: int, friend_qubits: list[int], friend_size: int) -> None:
        if self.strategy == "majority_vote":
            # Ask friend for the outcome.
            qc.measure(friend_qubits, observer_creg)
        elif self.strategy == "random":
            random_offset = random.randint(0, friend_size - 1)
            qc.measure(friend_qubits[0] + random_offset, observer)

    def _apply_reverse(self, qc: QuantumCircuit, observer: int, observer_creg: int, friend_qubits: list[int], friend_size: int, angle: float) -> None:
        qc.barrier(observer, friend_qubits)

        # Apply the CNOT ladder based on the strategy.
        if self.strategy == "majority_vote":
            cnot_ladder(qc, observer, friend_qubits[0], friend_size, reverse=True, internal_copy=True)
        elif self.strategy == "random":
            cnot_ladder(qc, observer, friend_qubits[0], friend_size)

        # Apply the rotation based on the observer.
        if observer is ALICE:
            self._ewfs_rotation(qc, observer, self.angles["peek"], invert=False)
        if observer is BOB:
            self._ewfs_rotation(qc, observer, self.beta - self.angles["peek"], invert=False)

        # Apply a rotation.
        self._ewfs_rotation(qc, observer, angle)

        # Apply the measurement based on the strategy.
        if self.strategy == "majority_vote":
            qc.measure(observer, observer_creg)
        elif self.strategy == "random":
            qc.measure(observer, observer)

    def _ewfs_rotation(self, qc: QuantumCircuit, observer: int, angle: float, invert: bool = True) -> None:
        """Apply an EWFS-specific rotation to a qubit."""
        if invert:
            qc.rz(-angle, observer)
            qc.h(observer)
        else:
            qc.h(observer)
            qc.rz(angle, observer)
