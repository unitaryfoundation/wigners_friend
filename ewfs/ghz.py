from collections import deque

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class GHZ:
    def __init__(
        self,
        coupling_map: CouplingMap,
        layout: list,
        flag_qubits: list,
        xx_check: bool = False,  # Toggle for XX check
        root_flag_qubit: int = None,  # Specific ancilla for the XX check
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
            qc.h(f_idx)  # Flag is now unentangled and contains the phase-error status

            # 3. Complete the rest of the BFS GHZ tree
            for op in self.ghz_ops[1:]:
                qc.cx(self.layout_dict[op[0]], self.layout_dict[op[1]])
        else:
            # Standard GHZ prep if XX check is off
            self._ghz(qc)

        qc.barrier()
        # Apply standard ZZ flags (for bit-flips) at the end as usual
        self._flag_operations(qc)

        qc = transpile(qc, optimization_level=1)

        return qc

    def _get_classical_registers(self) -> tuple:
        """Define the classical registers."""
        data_creg = list(range(len(self.data_qubits)))

        flag_creg = list(range(data_creg[-1] + 1, data_creg[-1] + 1 + len(self.flag_qubits)))

        return data_creg, flag_creg

    def _parity_oscillation_circuit(self, phi):
        """Helper to create oscillation circuits."""
        qc = self._initialize_circuit()
        for q in self.data_qubits:
            qc.rz(phi, self.layout_dict[q])
            qc.h(self.layout_dict[q])

        data_idxs = [self.layout_dict[d] for d in self.data_qubits]
        flag_idxs = [self.layout_dict[f] for f in self.flag_qubits]
        qc.measure(data_idxs, range(len(data_idxs)))
        qc.measure(flag_idxs, range(len(data_idxs), len(self.layout)))
        return qc

    def get_verification_circuits(self, method: str, phases: np.ndarray = None) -> list[QuantumCircuit]:
        """
        Generate verification circuits based on the chosen method.

        Args:
            method (str): 'dfe' or 'parity_oscillation'
            phases (np.ndarray): Array of phases for parity oscillation.

        Returns:
            list[QuantumCircuit]: List of circuits to execute.
        """
        if method == "dfe":
            # 1. Z-basis
            z_qc = self.circuit

            # 2. X-basis
            x_qc = self._initialize_circuit()
            for d in self.data_qubits:
                x_qc.h(self.layout_dict[d])

            # Maps
            data_idxs = [self.layout_dict[d] for d in self.data_qubits]
            flag_idxs = [self.layout_dict[f] for f in self.flag_qubits]

            x_qc.measure(data_idxs, range(len(data_idxs)))
            x_qc.measure(flag_idxs, range(len(data_idxs), len(self.layout)))

            return [z_qc, x_qc]

        elif method == "parity_oscillation":
            if phases is None:
                raise ValueError("phases required for parity_oscillation")

            z_qc = self.circuit
            osc_circs = [self._parity_oscillation_circuit(p) for p in phases]
            return [z_qc] + osc_circs

        else:
            raise ValueError(f"Unknown verification method: {method}")


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
    def __init__(self, n_data_qubits: int, method: str = "dfe"):
        """
        Initializes the Fidelity Estimator.
        Methods: 'dfe' (Fast) or 'parity_oscillation' (Rigorous).
        """
        self.n = n_data_qubits
        self.method = method

    def _osc_func(self, phi, amplitude, offset):
        """The theoretical oscillation function for an n-qubit GHZ state."""
        return amplitude * np.cos(self.n * phi + offset)

    def estimate_coherence_dft(self, phases, parities):
        """
        Estimate coherence using Discrete Fourier Transform.
        Ideally requires phases to cover [0, 2pi/N] uniformly (for one period).
        Returns the amplitude of the N-th frequency component (normalized).
        """
        # Simple estimate: Compute |sum(parity * exp(-i * N * phi))| * 2 / NumPoints
        # This is essentially the DFT component at frequency N.
        # Amplitude C is 2 * |Fourier Coeff|.

        complex_sum = np.sum(np.array(parities) * np.exp(-1j * self.n * np.array(phases)))
        coherence_dft = 2 * np.abs(complex_sum) / len(parities)

        # Note: This assumes points are distributed such that orthogonality holds roughly.
        # If we scan [0, 2pi/N] with M points, orthogonality holds for DC vs fundamental.
        return coherence_dft

    def estimate_fidelity(self, results) -> tuple[float, float]:
        """
        Calculates population and coherence based on the chosen method.

        Returns:
            tuple: (population, coherence, p_err, c_err, details_dict)
        """
        details = {}
        if self.method == "dfe":
            # Unpack results: can be (z_counts, x_counts) or (z_counts, x_counts, z_errs, x_errs)
            z_counts, x_counts = results[0], results[1]

            # Population
            total_z = sum(z_counts.values())
            p = (z_counts.get("0" * self.n, 0) + z_counts.get("1" * self.n, 0)) / total_z
            p_err = np.sqrt(p * (1 - p) / total_z)

            # Coherence
            total_x = sum(x_counts.values())
            even_x = sum(c for b, c in x_counts.items() if b.count("1") % 2 == 0)
            c = (2 * even_x - total_x) / total_x

            # For DFE, c is simply the expectation value of parity <P_x>
            # sigma_c = sqrt((1 - c^2) / N_x)
            c_err = np.sqrt((1 - c**2) / total_x)

            return p, c, p_err, c_err, details

        elif self.method == "parity_oscillation":
            # Unpack results: can be (z_counts, phases, parities) or (z_counts, phases, parities, errors)
            if len(results) == 4:
                z_counts, phases, parities, errors_c = results
            else:
                z_counts, phases, parities = results
                errors_c = None

            # Population
            total_z = sum(z_counts.values())
            p = (z_counts.get("0" * self.n, 0) + z_counts.get("1" * self.n, 0)) / total_z
            p_err = np.sqrt(p * (1 - p) / total_z)

            # Weighted fit if errors are provided
            sigma = errors_c if errors_c is not None else None
            absolute_sigma = True if errors_c is not None else False

            try:
                popt, pcov = curve_fit(
                    self._osc_func, phases, parities, p0=[0.5, 0], sigma=sigma, absolute_sigma=absolute_sigma
                )
                c = abs(popt[0])
                offset = popt[1]
                # c_err is the standard deviation of the amplitude parameter
                c_err = np.sqrt(pcov[0, 0])

                details["offset"] = offset
                details["fit_popt"] = popt
            except Exception:
                c = 0.0
                c_err = 0.0
                details["calc_error"] = "Fit failed"

            return p, c, p_err, c_err, details

    def plot_oscillation(self, phases, parities, title="GHZ Parity Oscillation", save_path=None, errors=None):
        """
        Plots the experimental parity points and the fitted curve.
        """
        sigma = errors if errors is not None else None
        absolute_sigma = True if errors is not None else False

        popt, _ = curve_fit(self._osc_func, phases, parities, p0=[0.5, 0], sigma=sigma, absolute_sigma=absolute_sigma)

        plt.figure(figsize=(10, 6))

        if errors is not None:
            plt.errorbar(phases, parities, yerr=errors, fmt="o", color="black", label="Experimental Parity", capsize=5)
        else:
            plt.scatter(phases, parities, color="black", label="Experimental Parity")

        # Smooth curve for the fit
        fine_phases = np.linspace(0, 2 * np.pi, 200)
        plt.plot(
            fine_phases, self._osc_func(fine_phases, *popt), color="red", linestyle="--", label=f"Fit (n={self.n})"
        )

        plt.xlabel(r"Phase $\phi$ (rad)")
        plt.ylabel(r"Global Parity $\langle P \rangle$")
        plt.title(f"{title}\nAmplitude = {abs(popt[0]):.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
