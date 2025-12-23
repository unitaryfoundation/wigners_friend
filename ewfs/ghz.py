
from collections import deque

from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime import SamplerV2

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class GHZ:
    def __init__(
        self,
        coupling_map: CouplingMap,
        layout: list,
        flag_qubits: list,
        xx_check: bool = False, # Toggle for XX check
        root_flag_qubit: int = None # Specific ancilla for the XX check
    ) -> None:
        """Initialize the GHZ class."""
        self.xx_check = xx_check
        self.root_flag_qubit = root_flag_qubit

        self.coupling_map = coupling_map
        self.layout = layout
        self.flag_qubits = flag_qubits
        self.data_qubits = [item for item in layout if item not in flag_qubits]
        if root_flag_qubit is not None and root_flag_qubit in self.data_qubits:
            self.data_qubits.remove(root_flag_qubit)
        self.layout_dict = dict(zip(layout, range(len(layout))))

        self.ghz_ops = self._get_directed_tree_edges()
        self.flag_ops: list[tuple[int, int]] = []

        self.meas_size = 2

    @property
    def circuit(self) -> QuantumCircuit:
        """Generate the circuit for the GHZ state."""
        # Initialize system qubits and flag qubits and apply GHZ + flag operations
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
        """
        Apply standard ZZ checks at the end of the circuit.
        """
        if not invert:
            # We reset flag_ops only if we aren't performing an Early XX check
            # or we manage the list carefully to avoid clearing the XX op.
            
            # --- Standard ZZ Checks (Detects Bit-flips) ---
            for flag in self.flag_qubits:
                # Skip the root flag if it was already used for the Early XX check
                if self.xx_check and flag == self.root_flag_qubit: 
                    continue
                    
                neighbors = self.coupling_map.neighbors(flag)
                for qubit in neighbors:
                    if qubit in self.layout and qubit != self.root_flag_qubit:
                        # Apply ZZ check: CNOT from data to flag
                        qc.cx(self.layout_dict[qubit], self.layout_dict[flag])
                        self.flag_ops.append((self.layout_dict[qubit], self.layout_dict[flag]))
        else:
            # Inverse application for uncomputing (if needed for resets)
            for op in reversed(self.flag_ops):
                qc.cx(op[0], op[1])

    def _initialize_circuit(self) -> QuantumCircuit:
        """Initialize the circuit with an Early XX Flag Check."""
        meas_size = len(self.data_qubits) + len(self.flag_qubits)
        qc = QuantumCircuit(len(self.layout), meas_size)

        # 1. Start the root
        root = self.data_qubits[0]
        qc.h(self.layout_dict[root])

        # 2. Check if we are doing an Early XX Flag
        if self.xx_check and self.root_flag_qubit is not None and len(self.ghz_ops) > 0:
            # Create the initial Bell pair (Seed of the GHZ state)
            ctrl, target = self.ghz_ops[0]
            qc.cx(self.layout_dict[ctrl], self.layout_dict[target])
            
            # XX check on the Bell pair while it is still a subsystem stabilizer
            f_idx = self.layout_dict[self.root_flag_qubit]
            d0_idx = self.layout_dict[ctrl]
            d1_idx = self.layout_dict[target]
            
            qc.h(f_idx)
            qc.cx(f_idx, d0_idx)
            qc.cx(f_idx, d1_idx)
            qc.h(f_idx) # Flag is now unentangled and contains the phase-error status
            
            # 3. Complete the rest of the BFS GHZ tree
            for op in self.ghz_ops[1:]:
                qc.cx(self.layout_dict[op[0]], self.layout_dict[op[1]])
        else:
            # Standard GHZ prep if XX check is off
            self._ghz(qc)

        qc.barrier()
        # Apply standard ZZ flags (for bit-flips) at the end as usual
        self._flag_operations(qc) 

        return qc

    def _get_classical_registers(self) -> tuple:
        """Define the classical registers."""
        data_creg = list(range(len(self.data_qubits)))

        flag_creg = list(range(data_creg[-1] + 1, data_creg[-1] + 1 + len(self.flag_qubits)))

        return data_creg, flag_creg


def post_select_results(results, num_flag_qubits) -> dict:
    """
    Check the measurement outcomes of all flag qubits and post-select results.
    
    Args:
        results (dict): The raw counts dictionary from the sampler.
        num_data_qubits (int): Number of qubits used for the GHZ state.
        num_flag_qubits (int): Total number of flags (including the XX flag).
    """
    post_selected_results = {}

    for bitstring, count in results.items():
        # Remove any whitespace from the bitstring
        processed_bitstring = bitstring.replace(" ", "")

        # Qiskit bitstrings are usually [flags][data] in the string representation
        # because higher index registers (flags) appear on the left.
        flags_part = processed_bitstring[:num_flag_qubits]
        data_part = processed_bitstring[num_flag_qubits:]

        # Post-selection: Keep only if ALL flags measured '0'
        if flags_part == "0" * num_flag_qubits:
            post_selected_results[data_part] = post_selected_results.get(data_part, 0) + count

    return post_selected_results

class FidelityEstimator:
    def __init__(self, ghz: GHZ, method: str = 'dfe'):
        """
        Initializes the Fidelity Estimator.
        Methods: 'dfe' (Fast) or 'parity_oscillation' (Rigorous).
        'dfe' - Direct Fidelity Estimation using Z and X basis measurements 
                (this implementation works only with incoherent noise).
        'parity_oscillation' - Measures parity oscillations by varying phase
                (works under any noise model).
        """
        self.ghz = ghz
        self.method = method
        self.n = len(self.ghz.data_qubits)

    def run_measurements(self, backend: Backend, shots: int, num_points=20):
        """
        Executes the required circuits based on the selected method.
        """
        num_f = len(self.ghz.flag_qubits)
        
        if self.method == 'dfe':
            z_qc = self.ghz.circuit
            x_qc = self.ghz._initialize_circuit()
            for d in self.ghz.data_qubits: 
                x_qc.h(self.ghz.layout_dict[d])
            
            # Map measurements for X-basis
            data_idxs = [self.ghz.layout_dict[d] for d in self.ghz.data_qubits]
            flag_idxs = [self.ghz.layout_dict[f] for f in self.ghz.flag_qubits]
            x_qc.measure(data_idxs, range(len(data_idxs)))
            x_qc.measure(flag_idxs, range(len(data_idxs), len(self.ghz.layout)))

            res = SamplerV2(backend).run(transpile([z_qc, x_qc], backend), shots=shots).result()
            return [post_select_results(res[i].data.c.get_counts(), num_f) for i in range(2)]

        elif self.method == 'parity_oscillation':
            phases = np.linspace(0, 2 * np.pi, num_points)
            z_qc = self.ghz.circuit
            osc_circs = [self._parity_oscillation_circuit(p) for p in phases]
            
            res = SamplerV2(backend).run(transpile([z_qc] + osc_circs, backend), shots=shots).result()
            
            z_counts = post_select_results(res[0].data.c.get_counts(), num_f)
            parities = []
            for i in range(1, num_points + 1):
                counts = post_select_results(res[i].data.c.get_counts(), num_f)
                total = sum(counts.values())
                even = sum(c for b, c in counts.items() if b.count('1') % 2 == 0)
                parities.append((2 * even - total) / total if total > 0 else 0)
                
            return z_counts, phases, parities

    def _osc_func(self, phi, amplitude, offset):
        """The theoretical oscillation function for an n-qubit GHZ state."""
        return amplitude * np.cos(self.n * phi + offset)

    def estimate_fidelity(self, results) -> float:
        """Calculates fidelity based on the chosen method."""
        if self.method == 'dfe':
            z_counts, x_counts = results[0], results[1]
            p = (z_counts.get("0"*self.n, 0) + z_counts.get("1"*self.n, 0)) / sum(z_counts.values())
            even = sum(c for b, c in x_counts.items() if b.count('1') % 2 == 0)
            c = (2 * even - sum(x_counts.values())) / sum(x_counts.values())
            return (p + c) / 2

        elif self.method == 'parity_oscillation':
            z_counts, phases, parities = results
            p = (z_counts.get("0"*self.n, 0) + z_counts.get("1"*self.n, 0)) / sum(z_counts.values())
            popt, _ = curve_fit(self._osc_func, phases, parities, p0=[0.5, 0])
            c = abs(popt[0])
            return (p + c) / 2

    def plot_oscillation(self, phases, parities, title="GHZ Parity Oscillation"):
        """
        Plots the experimental parity points and the fitted curve.
        """
        popt, _ = curve_fit(self._osc_func, phases, parities, p0=[0.5, 0])
        
        plt.figure(figsize=(10, 6))
        plt.scatter(phases, parities, color='black', label='Experimental Parity')
        
        # Smooth curve for the fit
        fine_phases = np.linspace(0, 2 * np.pi, 200)
        plt.plot(fine_phases, self._osc_func(fine_phases, *popt), 
                 color='red', linestyle='--', label=f'Fit (n={self.n})')
        
        plt.xlabel(r"Phase $\phi$ (rad)") 
        plt.ylabel(r"Global Parity $\langle P \rangle$") 
        plt.title(f"{title}\nAmplitude = {abs(popt[0]):.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def _parity_oscillation_circuit(self, phi):
        """Helper to create oscillation circuits."""
        qc = self.ghz._initialize_circuit()
        for q in self.ghz.data_qubits:
            qc.rz(phi, self.ghz.layout_dict[q])
            qc.h(self.ghz.layout_dict[q])
        
        data_idxs = [self.ghz.layout_dict[d] for d in self.ghz.data_qubits]
        flag_idxs = [self.ghz.layout_dict[f] for f in self.ghz.flag_qubits]
        qc.measure(data_idxs, range(len(data_idxs)))
        qc.measure(flag_idxs, range(len(data_idxs), len(self.ghz.layout)))
        return qc
