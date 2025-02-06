"""Extended Wigner's friend scenario (EWFS)" functionality."""

import random

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from ewfs.circuit import cnot_ladder, ghz_ewfs_circuit, extract_qiskit_indices_by_prefix
from ewfs.observer import ALICE, BOB, DEFAULT_ANGLES, DEFAULT_BETA
from ewfs.friend_state import CNOT_LADDER, GHZ
from ewfs.strategy import MAJORITY_VOTE, RANDOM
from ewfs.setting import PEEK, REVERSE_1, REVERSE_2


class EWFS:
    def __init__(
        self,
        alice_setting: str,
        bob_setting: str,
        strategy: str,
        friend_state: str,
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

        # State for the friend.
        self.friend_state = friend_state

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

        self.charlie_qubits: list[int] = []
        self.debbie_qubits: list[int] = []

    def circuit(self) -> QuantumCircuit:
        """Generate the circuit for extended Wigner's friend scenario."""
        # Create the Quantum Circuit with the defined registers
        qc = self._initialize_circuit()

        # Prepare the bipartite system for the observers (Alice and Bob).
        self._prepare_bipartite_system(qc)

        # Perform the rotations for Alice and Bob based on their settings.
        self._ewfs_rotation(qc, ALICE, self.angles[PEEK])
        self._ewfs_rotation(qc, BOB, self.beta - self.angles[PEEK])

        # Apply state for the friends (Charlie and Debbie).
        qc = self._apply_friend_state(qc)

        alice_creg, bob_creg = self._get_classical_registers()

        # Apply the setting for Alice/Charlie.
        self._apply_setting(
            qc=qc,
            observer=ALICE,
            setting=self.alice_setting,
            angle=self.angles[self.alice_setting],
            observer_creg=alice_creg,
            friend_qubits=self.charlie_qubits,
            friend_size=self.charlie_size,
        )
        # Apply the setting for Bob/Debbie.
        self._apply_setting(
            qc=qc,
            observer=BOB,
            setting=self.bob_setting,
            angle=(self.beta - self.angles[self.bob_setting]),
            observer_creg=bob_creg,
            friend_qubits=self.debbie_qubits,
            friend_size=self.debbie_size,
        )

        return qc

    def _initialize_circuit(self) -> QuantumCircuit:
        """Initialize the classical measurement registers based on the strategy."""
        if self.friend_state is CNOT_LADDER:
            if self.strategy is MAJORITY_VOTE:
                if self.alice_setting is PEEK and self.bob_setting is not PEEK:
                    measurement = ClassicalRegister(self.charlie_size + 1, name="measurement")
                else:
                    measurement = ClassicalRegister(self.meas_size, name="measurement")

            elif self.strategy is RANDOM:
                measurement = ClassicalRegister(self.sys_size, name="measurement")

            alice, bob, charlie, debbie = [
                QuantumRegister(size, name=name)
                for size, name in zip(
                    [self.alice_size, self.bob_size, self.charlie_size, self.debbie_size],
                    ["Alice", "Bob", "Charlie", "Debbie"],
                )
            ]

            return QuantumCircuit(alice, bob, charlie, debbie, measurement)

        elif self.friend_state is GHZ:
            # For the GHZ case, we just initialize the circuit for Alice and Bob now and deal with the friends later.
            measurement = ClassicalRegister(self.meas_size, name="measurement")
            alice, bob = [
                QuantumRegister(size, name=name)
                for size, name in zip([self.alice_size, self.bob_size], ["Alice", "Bob"])
            ]
            return QuantumCircuit(alice, bob, measurement)

    def _get_classical_registers(self) -> tuple:
        """Define the classical registers for the observers."""
        if self.friend_state is CNOT_LADDER:
            if self.strategy is MAJORITY_VOTE:
                if self.alice_setting is PEEK and self.bob_setting is not PEEK:
                    alice_creg = list(range(self.charlie_size))
                    bob_creg = [self.charlie_size]
                else:
                    alice_creg, bob_creg = [0], [1]
            elif self.strategy is RANDOM:
                alice_creg, bob_creg = [0], [0]
        elif self.friend_state is GHZ:
            # TODO: Fix
            alice_creg, bob_creg = [0], [1]

        return alice_creg, bob_creg

    def _prepare_bipartite_system(self, qc: QuantumCircuit) -> None:
        """Generates the state: 1/sqrt(2) * (|01> - |10>)"""
        qc.x(ALICE)
        qc.x(BOB)
        qc.h(ALICE)
        qc.cx(ALICE, BOB)

    def _apply_friend_state(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Apply the state for the friends Charlie and Debbie."""
        if self.friend_state is CNOT_LADDER:
            self.charlie_qubits = extract_qiskit_indices_by_prefix(qc, "Charlie")
            self.debbie_qubits = extract_qiskit_indices_by_prefix(qc, "Debbie")
            if self.strategy is MAJORITY_VOTE:
                cnot_ladder(qc, ALICE, self.charlie_qubits[0], self.charlie_size, reverse=False, internal_copy=True)
                cnot_ladder(qc, BOB, self.debbie_qubits[0], self.debbie_size, reverse=False, internal_copy=True)
            elif self.strategy is RANDOM:
                cnot_ladder(qc, ALICE, self.charlie_qubits[0], self.charlie_size)
                cnot_ladder(qc, BOB, self.debbie_qubits[0], self.debbie_size)
            return qc
        elif self.friend_state is GHZ:
            qc = ghz_ewfs_circuit(qc, self.charlie_size, self.debbie_size)
            self.charlie_qubits = extract_qiskit_indices_by_prefix(qc, "Charlie")
            self.debbie_qubits = extract_qiskit_indices_by_prefix(qc, "Debbie")
            return qc

    def _apply_setting(
        self,
        qc: QuantumCircuit,
        observer: int,
        setting: str,
        angle: float,
        observer_creg: list[int],
        friend_qubits: list[int],
        friend_size: int,
    ):
        """Apply either the PEEK or REVERSE_1/REVERSE_2 settings."""
        if setting is PEEK:
            self._apply_peek(qc, observer, observer_creg, friend_qubits, friend_size)
        elif setting in [REVERSE_1, REVERSE_2]:
            self._apply_reverse(qc, observer, observer_creg, friend_qubits, friend_size, angle)

    def _apply_peek(
        self, qc: QuantumCircuit, observer: int, observer_creg: list[int], friend_qubits: list[int], friend_size: int
    ) -> None:
        if self.strategy is MAJORITY_VOTE:
            # Ask friend for the outcome.
            qc.measure(friend_qubits, observer_creg)
        elif self.strategy is RANDOM:
            random_offset = random.randint(0, friend_size - 1)
            qc.measure(friend_qubits[0] + random_offset, observer)

    def _apply_reverse(
        self,
        qc: QuantumCircuit,
        observer: int,
        observer_creg: list[int],
        friend_qubits: list[int],
        friend_size: int,
        angle: float,
    ) -> None:
        qc.barrier(observer, friend_qubits)

        # Apply the appropriate friend state.
        if self.friend_state is CNOT_LADDER:
            if self.strategy is MAJORITY_VOTE:
                cnot_ladder(qc, observer, friend_qubits[0], friend_size, reverse=True, internal_copy=True)
            elif self.strategy is RANDOM:
                cnot_ladder(qc, observer, friend_qubits[0], friend_size)
        elif self.friend_state is GHZ:
            # Reverse GHZ state preparation
            for i in range(friend_size - 1, 0, -1):
                qc.cx(friend_qubits[i - 1], friend_qubits[i])  # Reverse CNOT gates
            qc.h(friend_qubits[0])  # Reverse Hadamard gate on the first qubit

        # Apply the rotation based on the observer.
        if observer is ALICE:
            self._ewfs_rotation(qc, observer, self.angles[PEEK], invert=False)
        if observer is BOB:
            self._ewfs_rotation(qc, observer, self.beta - self.angles[PEEK], invert=False)

        # Apply a rotation.
        self._ewfs_rotation(qc, observer, angle)

        # Apply the measurement based on the strategy.
        if self.strategy is MAJORITY_VOTE:
            qc.measure(observer, observer_creg)
        elif self.strategy is RANDOM:
            qc.measure(observer, observer)

    def _ewfs_rotation(self, qc: QuantumCircuit, observer: int, angle: float, invert: bool = True) -> None:
        """Apply an EWFS-specific rotation to a qubit."""
        if invert:
            qc.rz(-angle, observer)
            qc.h(observer)
        else:
            qc.h(observer)
            qc.rz(angle, observer)


if __name__ == "__main__":
    ewfs = EWFS(
        alice_setting=PEEK,
        bob_setting=REVERSE_1,
        strategy=MAJORITY_VOTE,
        friend_state=GHZ,
        charlie_size=3,
        debbie_size=1,
    )
    qc = ewfs.circuit()
    print(qc)
