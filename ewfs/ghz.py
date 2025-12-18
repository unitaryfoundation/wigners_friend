import numpy as np
from collections import deque
from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime import SamplerV2


@dataclass
class Friend:
    size: int
    qubits: list[int]
    label: str


class GHZ:
    def __init__(
        self,
        coupling_map: CouplingMap,
        layout: list,
        flag_qubits: list,
    ) -> None:
        """Initialize the GHZ class."""

        self.coupling_map = coupling_map
        self.layout = layout
        self.flag_qubits = flag_qubits
        self.data_qubits = [item for item in layout if item not in flag_qubits]
        self.layout_dict = dict(zip(layout, range(len(layout))))

        self.ghz_ops = self._get_directed_tree_edges()
        self.flag_ops: list[tuple[int, int]] = []

        self.meas_size = 2

    @property
    def circuit(self) -> QuantumCircuit:
        """Generate the circuit for the GHZ state."""
        # Initialize system qubits and entangle with Charlie
        qc = self._initialize_circuit()

        data_creg, flag_creg = self._get_classical_registers()

        # Measure the data qubits
        data_qubits = [self.layout_dict[data] for data in self.data_qubits]
        qc.measure(data_qubits, data_creg)

        # Finally, measure the flag qubits.
        flag_qubits = [self.layout_dict[flag] for flag in self.flag_qubits]
        qc.measure(flag_qubits, flag_creg)

        return qc

    def _get_directed_tree_edges(self) -> list:
        """Determines the set of directed edges for a tree by performing a BFS.

        Returns:
            A set of tuples, where each tuple (u, v) represents a directed edge
            from node u to node v.
        """
        coupling_map = self.coupling_map
        tree_nodes = self.data_qubits
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
        if not invert:
            # Forward application: generate and store GHZ ops
            self.ghz_ops = self._get_directed_tree_edges()
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
                    if qubit in self.layout:
                        qc.cx(self.layout_dict[qubit], self.layout_dict[flag])
                        self.flag_ops.append((self.layout_dict[qubit], self.layout_dict[flag]))
        else:
            # Inverse application: replay stored ops in reverse order.
            for op in reversed(self.flag_ops):
                qc.cx(op[0], op[1])

    def _initialize_circuit(self) -> QuantumCircuit:
        """Initialize the classical measurement registers based on the strategy."""

        meas_size = len(self.data_qubits) + len(self.flag_qubits)

        qc = QuantumCircuit(len(self.layout), meas_size)

        # First create entanglement between the system qubits.
        qc.h(self.layout_dict[self.data_qubits[0]])

        # then create the ghz state on the data qubits ({layout}\{flags}).
        self._ghz(qc)

        # Add barrier between state preparation and flag operations
        qc.barrier()

        # now implement the check operations
        self._flag_operations(qc)

        return qc

    def _get_classical_registers(self) -> tuple:
        """Define the classical registers."""
        data_creg = list(range(len(self.data_qubits)))

        flag_creg = list(range(data_creg[-1] + 1, data_creg[-1] + 1 + len(self.flag_qubits)))

        return data_creg, flag_creg

    
def post_select_results(results, flag_size) -> dict:
    """Check the measurement outcomes of flag qubits and post-select results."""

    post_selected_results = {}

    for bitstring, count in results.items():
        processed_bitstring = bitstring.replace(" ", "")

        flags_bitstring = processed_bitstring[:flag_size]
        data_bitstring = processed_bitstring[flag_size:]

        if flags_bitstring == "0" * flag_size:
            post_selected_results[data_bitstring] = post_selected_results.get(data_bitstring, 0) + count

    return post_selected_results


class FidelityEstimator:
    def __init__(
            self,
            ghz: GHZ,
            method: str = 'dfe'): # TODO: support parity oscillations method
        """
        Initializes the Fidelity Estimator.
        
        The GHZ state fidelity is F = (1/2^n) * sum( <S_i> ).
        For GHZ, this simplifies to checking the Z-parity and the X-coherence.
        """
        self.ghz = ghz
        self.method = method
        self.n = len(self.ghz.data_qubits)
        # Generators for GHZ: {Z_i Z_{i+1}} for i=1..n-1  and  {X_1 X_2 ... X_n}
        # To get the full sum, we only measure in the Z basis and X basis separately.

    @property
    def generate_measurement_circuits(self) -> list[QuantumCircuit]:
        """
        Creates the circuits needed to estimate fidelity.
        1. Z-basis circuit: Measures all Z-type stabilizers (parity).
        2. X-basis circuit: Measures the XX...X stabilizer (coherence).
        """
        circuits = []

        # --- 1. Z-Basis Measurement (Detects Bit-flips) ---
        # This allows us to calculate <Z1Z2>, <Z2Z3>, etc., and all their products.
        z_qc = self.ghz.circuit.copy()
        z_qc.name = "meas_z"
        # The base GHZ class already measures in Z basis at the end of the circuit
        circuits.append(z_qc)

        # --- 2. X-Basis Measurement (Detects Phase-flips) ---
        # To measure in X, we apply Hadamard to all data qubits before measurement.
        x_qc = self.ghz._initialize_circuit() # Get circuit before final Z measurements
        
        # Apply H-gates to data qubits to rotate to X-basis
        for qubit in self.ghz.data_qubits:
            x_qc.h(self.ghz.layout_dict[qubit])
        
        # Define registers and measure
        data_creg, flag_creg = self.ghz._get_classical_registers()
        data_indices = [self.ghz.layout_dict[d] for d in self.ghz.data_qubits]
        x_qc.measure(data_indices, data_creg)
        
        # Measure flags (in Z-basis) for the post-selection logic
        flag_indices = [self.ghz.layout_dict[f] for f in self.ghz.flag_qubits]
        x_qc.measure(flag_indices, flag_creg)
        
        x_qc.name = "meas_x"
        circuits.append(x_qc)

        return circuits
    
    def run_measurements(self, backend: Backend, shots: int) -> list[dict]:
        """Runs the Z and X circuits on the provided backend."""
        transpiled = transpile(self.generate_measurement_circuits, backend, optimization_level=3)
        sampler = SamplerV2(backend)
        
        job_res = sampler.run(transpiled, shots=shots).result()
        
        # Return a list of count dictionaries, one for Z and one for X
        # Apply post-selection to each result
        all_counts = []
        for i in range(len(transpiled)):
            raw_counts = job_res[i].data.c.get_counts()
            post_counts = post_select_results(raw_counts, len(self.ghz.flag_qubits))
            all_counts.append(post_counts)
            
        return all_counts

    def estimate_fidelity(self, results: list[dict]) -> float:
        """
        Calculates fidelity using the results from Z and X basis measurements.
        
        F = 0.5 * (Population + Coherence)
        Population: Probability of being in |00...0> or |11...1>
        Coherence: Expectation value of <X1 X2 ... Xn>
        """
        z_counts = results[0]
        x_counts = results[1]
        
        total_z = sum(z_counts.values())
        total_x = sum(x_counts.values())

        # 1. Calculate Population
        # For a GHZ state, we want only 00...0 and 11...1
        target_0 = "0" * self.n
        target_1 = "1" * self.n
        population = (z_counts.get(target_0, 0) + z_counts.get(target_1, 0)) / total_z

        # 2. Calculate Coherence
        # <X...X> = (Number of even-parity bitstrings - Number of odd-parity bitstrings) / Total
        even_parity_count = 0
        for bitstring, count in x_counts.items():
            if bitstring.count('1') % 2 == 0:
                even_parity_count += count
        
        coherence = (2 * even_parity_count - total_x) / total_x

        fidelity = (population + coherence) / 2
        
        return fidelity
