import numpy as np
from collections import deque
from dataclasses import dataclass

from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend
from qiskit.transpiler import CouplingMap

from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_ibm_runtime import SamplerV2


@dataclass
class Friend:
    size: int
    qubits: list[int]
    label: str


class EWFS:
    def __init__(
        self,
        alice_setting: str,
        bob_setting: str,
        strategy: str,
        coupling_map: CouplingMap,
        layout: list,
        flag_qubits: list,
        shots: int,
        backend: Backend,
    ) -> None:
        """Initialize the extended Wigner's friend scenario."""
        # Settings for Alice and Bob.
        self.alice_setting = alice_setting
        self.bob_setting = bob_setting
        self.strategy = strategy
        self.shots = shots
        self.backend = backend

        self.coupling_map = coupling_map
        self.layout = layout
        self.flag_qubits = flag_qubits
        self.data_qubits = [item for item in layout if item not in flag_qubits]
        self.layout_dict = dict(zip(layout, range(len(layout))))

        self.ghz_ops = self._get_directed_tree_edges()
        self.flag_ops: list[tuple[int, int]] = []

        # Hardcoded locations on the circuit for Alice and Bob.
        self.alice = self.layout_dict[self.layout[1]]
        self.bob = self.layout_dict[self.layout[0]]

        # The first two qubits should always be the system qubits.
        self.charlie_qubits = [self.layout_dict[self.data_qubits[i]] for i in range(2, len(self.data_qubits))]
        self.charlie_size = len(self.charlie_qubits)

        self.meas_size = 2

        # (Optimized) angles and beta term used for Alice and Bob measurement
        # operators. Adapted from arXiv:1907.05607. Note that despite the fact
        # that degrees are used, we need to convert this to radians.
        self.angles = {
            "peek": np.deg2rad(40),
            "reverse_1": np.deg2rad(230),
            "reverse_2": np.deg2rad(310),
        }
        self.beta = np.deg2rad(220)

    @property
    def branch_factor(self) -> float:
        """Branch factor is defined in arXiv:2106.16044v1 as the number of friends minus one."""
        return self.charlie_size - 1

    @property
    def circuit(self) -> QuantumCircuit:
        """Generate the circuit for extended Wigner's friend scenario."""
        # Initialize system qubits and entangle with Charlie
        qc = self._initialize_circuit()

        alice_creg, bob_creg, flag_creg = self._get_classical_registers()
        charlie = Friend(size=self.charlie_size, qubits=self.charlie_qubits, label="Charlie")

        # Apply the setting for Alice/Charlie.
        self._apply_setting(
            qc=qc,
            observer=self.alice,
            setting=self.alice_setting,
            angle=self.angles[self.alice_setting],
            observer_creg=alice_creg,
            friend=charlie,
        )
        # Apply the setting for Bob.
        self._apply_setting(
            qc=qc,
            observer=self.bob,
            setting=self.bob_setting,
            angle=(self.beta - self.angles[self.bob_setting]),
            observer_creg=bob_creg,
            friend=None,
        )

        # Finally, measure the flag qubits.
        flag_qubits = [self.layout_dict[flag] for flag in self.flag_qubits]
        qc.measure(flag_qubits, flag_creg)

        return qc

    @property
    def probability_distribution(self) -> dict:
        transpiled_circuit = transpile(self.circuit, self.backend, optimization_level=3)
        sampler = SamplerV2(self.backend)

        res = sampler.run([transpiled_circuit], shots=self.shots).result()[0]

        results = {}
        counts = res.data.c.get_counts()
        probabilities = {key: value / self.shots for key, value in counts.items()}

        results[(self.alice_setting, self.bob_setting)] = probabilities
        return self._post_select_results(results, len(self.flag_qubits))

    def _get_directed_tree_edges(self) -> list:
        """Determines the set of directed edges for a tree by performing a BFS.

        This function simulates "pouring water" into a start_node and tracking
        its flow to neighboring nodes within a specified tree, creating a
        directed graph.

        Returns:
            A set of tuples, where each tuple (u, v) represents a directed edge
            from node u to node v.
        """
        coupling_map = self.coupling_map
        tree_nodes = self.data_qubits[1:]
        start_node = tree_nodes[0]

        tree_nodes_set = set(tree_nodes)
        queue = deque([start_node])
        visited = {start_node}
        directed_edges = []

        while queue:
            # Get the next node to process from the front of the queue
            current_node = queue.popleft()

            # Check all its neighbors in the original graph
            for neighbor in coupling_map.neighbors(current_node):
                # The neighbor is valid if it's in our target tree AND we
                # haven't visited it yet.
                if neighbor in tree_nodes_set and neighbor not in visited:
                    # Mark it as visited so we don't process it again.
                    visited.add(neighbor)
                    # Add it to the queue to process its own neighbors later.
                    queue.append(neighbor)
                    # Add the directed edge from our current node to this new neighbor.
                    directed_edges.append((current_node, neighbor))
        return directed_edges

    def _ghz(self, qc: QuantumCircuit, invert: bool = False) -> None:
        print("ghz_ops: ", self.ghz_ops)
        if not invert:
            # Forward application: generate and store GHZ ops
            self.ghz_ops = self._get_directed_tree_edges()
            print("ghz_ops: ", self.ghz_ops)
            for op in self.ghz_ops:
                qc.cx(self.layout_dict[op[0]], self.layout_dict[op[1]])
        else:
            # Inverse application: replay stored ops in reverse order
            for op in reversed(self.ghz_ops):
                qc.cx(self.layout_dict[op[0]], self.layout_dict[op[1]])

    def _flag_operations(self, qc: QuantumCircuit, invert: bool = False) -> None:
        if not invert:
            # Forward application: generate and store flag_ops
            self.flag_ops = []
            for flag in self.flag_qubits:
                neighbors = self.coupling_map.neighbors(flag)
                for qubit in neighbors:
                    qc.cx(self.layout_dict[qubit], self.layout_dict[flag])
                    self.flag_ops.append((self.layout_dict[qubit], self.layout_dict[flag]))
            print("flag_ops: ", self.flag_ops)
        else:
            # Inverse application: replay stored ops in reverse order.
            for op in reversed(self.flag_ops):
                qc.cx(op[0], op[1])

    def _initialize_circuit(self) -> QuantumCircuit:
        """Initialize the classical measurement registers based on the strategy."""
        if self.alice_setting == "peek":
            meas_size = self.charlie_size + 1
        else:
            meas_size = 2
        meas_size += len(self.flag_qubits)

        qc = QuantumCircuit(len(self.layout), meas_size)

        # First create entanglement between the system qubits.
        self._prepare_bipartite_system(qc, self.alice, self.bob)

        # Perform the rotations for Alice and Bob based on their settings.
        self._ewfs_rotation(qc, self.alice, self.angles["peek"])
        self._ewfs_rotation(qc, self.bob, self.beta - self.angles["peek"])

        # then create the ghz state on the data qubits ({layout}\{flags} u {bob}).
        self._ghz(qc)

        # now implement the check operations
        self._flag_operations(qc)

        return qc

    def _get_classical_registers(self) -> tuple:
        """Define the classical registers for the observers and the flag register."""
        if self.alice_setting == "peek" and self.bob_setting != "peek":
            alice_creg = list(range(self.charlie_size))
            bob_creg = [self.charlie_size]
        else:
            alice_creg, bob_creg = [0], [1]

        flag_creg = list(range(bob_creg[0] + 1, bob_creg[0] + 1 + len(self.flag_qubits)))

        return alice_creg, bob_creg, flag_creg

    def _prepare_bipartite_system(self, qc: QuantumCircuit, alice: int, bob: int) -> None:
        """Generates the state: 1/sqrt(2) * (|01> - |10>)"""
        qc.x(alice)
        qc.x(bob)
        qc.h(alice)
        qc.cx(alice, bob)

    def _apply_setting(
        self,
        qc: QuantumCircuit,
        observer: int,
        setting: str,
        angle: float,
        observer_creg: list[int],
        friend: Friend | None,
    ):
        """Apply either the PEEK or REVERSE_1/REVERSE_2 settings."""
        if setting == "peek" and friend is not None:
            self._apply_peek(qc, observer_creg, friend)
        elif setting in ["reverse_1", "reverse_2"]:
            self._apply_reverse(qc, observer, observer_creg, friend, angle)

    def _apply_peek(self, qc: QuantumCircuit, observer_creg: list[int], friend: Friend) -> None:
        # Ask friend for the outcome.
        qc.measure(friend.qubits, observer_creg)

    def _apply_reverse(
        self,
        qc: QuantumCircuit,
        observer: int,
        observer_creg: list[int],
        friend: Friend | None,
        angle: float,
    ) -> None:
        if friend is not None:
            qc.barrier(observer, friend.qubits)
            # do the inverse of ghz() function
            self._flag_operations(qc, invert=True)
            self._ghz(qc, invert=True)

        # Apply the rotation based on the observer.
        if observer is self.alice:
            self._ewfs_rotation(qc, observer, self.angles["peek"], invert=False)
        if observer is self.bob:
            self._ewfs_rotation(qc, observer, self.beta - self.angles["peek"], invert=False)

        self._ewfs_rotation(qc, observer, angle)
        qc.measure(observer, observer_creg)

    def _ewfs_rotation(self, qc: QuantumCircuit, observer: int, angle: float, invert: bool = True) -> None:
        """Apply an EWFS-specific rotation to a qubit."""
        if invert:
            qc.rz(-angle, observer)
            qc.h(observer)
        else:
            qc.h(observer)
            qc.rz(angle, observer)

    def _post_select_results(self, results: dict, flag_size: int = 0) -> dict:
        """Check the measurement outcomes of flag qubits and post-select results."""
        post_selected_results = {}

        for setting in results:
            post_selected_results_setting = {}

            for bitstring, count in results[setting].items():
                processed_bitstring = bitstring.replace(" ", "")

                flags_bitstring = processed_bitstring[:flag_size]
                friends_bitstring = processed_bitstring[flag_size:]

                if flags_bitstring == "0" * flag_size:
                    post_selected_results_setting[friends_bitstring] = count

            post_selected_results[setting] = post_selected_results_setting
        return post_selected_results

    def _decode_results(self, results: dict) -> dict:
        """Take majority vote of measurement bit-strings."""
        decoded_results = {}

        # For each setting, there is a dictionary of measurement results.
        for setting in results:
            if setting[0] == "peek":
                # Debbie's size is 1 because no PEEK setting
                bob_size = 1

                setting_results: dict = {}
                # Decode the keys for each measurement result of the setting.
                for k, v in results[setting].items():
                    alice_result, bob_result = k[-self.charlie_size :], k[:bob_size]

                    alice_zero_count, bob_zero_count = alice_result.count("0"), bob_result.count("0")

                    alice_decoding = "0" if alice_zero_count >= self.charlie_size // 2 + 1 else "1"
                    bob_decoding = "0" if bob_zero_count >= 1 else "1"

                    if alice_decoding + bob_decoding in setting_results.keys():
                        setting_results[alice_decoding + bob_decoding] += v
                    else:
                        setting_results[alice_decoding + bob_decoding] = v
                decoded_results[setting] = setting_results
            else:
                decoded_results[setting] = results[setting]

        return decoded_results


if __name__ == "__main__":
    # layout = [53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 56, 63, 64, 65, 66, 67, 57, 68, 69, 70, 71, 58]

    # flags = [56, 58]

    layout = [53, 52, 51, 50, 49, 48, 47, 46, 57, 67, 68, 69, 70, 71, 58]
    flags = [58]
    coupling_map = FakeFez().coupling_map
    backend = Aer.get_backend("aer_simulator")

    settings = [("peek", "reverse_1"), ("peek", "reverse_2"), ("reverse_2", "reverse_1"), ("reverse_2", "reverse_2")]
    semi_brukner = [
        EWFS(
            alice_setting=s1,
            bob_setting=s2,
            backend=backend,
            shots=10_000,
            coupling_map=coupling_map,
            layout=layout,
            strategy="majority_vote",
            flag_qubits=flags,
        )
        for s1, s2 in settings
    ]

    for x in semi_brukner:
        print(x.probability_distribution)
