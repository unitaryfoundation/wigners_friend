from typing import Any
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.visualization import plot_coupling_map


def remove_qubits_from_coupling_map(
    coupling_map: list[tuple[int, int]], qubits_to_remove: list[int]
) -> list[tuple[int, int]]:
    """
    Remove qubits from the coupling map, so they won't be included in the BFS of generating the GHZ state

    Parameters
    ----------
    coupling_map : list[tuple[int, int]]
        The backend coupling map.
    qubits_to_remove : list[int]
        A list of qubits to remove.

    Returns
    -------
    list[tuple[int, int]]
        The coupling map with the specified qubits removed.
    """
    return [x for x in coupling_map if not any(y in x for y in qubits_to_remove)]


def remove_duplicate_coupling(coupling_map: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Take as input a possibly directed and bidirectional coupling map and return a map with at most 1 link between any two qubits.

    Parameters
    ----------
    coupling_map : list[tuple[int]]
        The backend coupling map.

    Returns
    -------
    list[tuple[int]]
        The undirected coupling map.
    """
    new_coupling_map = []
    for qubit_0, qubit_1 in coupling_map:
        if ((qubit_0, qubit_1) in new_coupling_map) or ((qubit_1, qubit_0) in new_coupling_map):
            continue
        else:
            new_coupling_map.append((qubit_0, qubit_1))
    return new_coupling_map


def get_physical_to_virtual_map(physical_qubits: list[int], num_qubits: int) -> dict[int, int]:
    """
    Given a list of physical qubits, return a dictionary mapping them to virtual qubits.

    Parameters
    ----------
    physical_qubits : list[int]
        The list of physical qubits.
    num_qubits : int
        The number of qubits in the circuit.

    Returns
    -------
    dict[int, int]
        The physical to virtual mapping.
    """
    connections = {}
    for i, node in enumerate(physical_qubits[:num_qubits]):
        connections[node] = i
    return connections


def get_layout(physical_to_virtual_map: dict[int, int]) -> list[int]:
    """
    Get the layout, represented by physical indices.

    Parameters
    ----------
    physical_to_virtual_map : dict[int, int]
        The physical to virtual map.

    Returns
    -------
    list[int]
        The layout.
    """
    # reverse the dictionary to get the virtual to physical map
    virtual_to_physical_map = dict()
    for key, val in physical_to_virtual_map.items():
        virtual_to_physical_map[val] = key
    return [virtual_to_physical_map[i] for i in range(len(physical_to_virtual_map))]


def construct_ghz(
    coupling_map: list[tuple[int, int]],
    num_qubits: int,
    qubits_to_remove: list[int],
    start_qubit: int,
    measure_all: bool = False,
) -> tuple[QuantumCircuit, list[int], dict[int, int]]:
    """
    Spread the entanglement in a BFS manner based on the coupling map of the backend.

    Parameters
    ----------
    coupling_map: list[tuple[int, int]]
        The coupling map of the backend.
    num_qubits : int
        The number of qubits in the GHZ state.
    qubits_to_remove : list[int]
        The qubits to remove from the coupling map.
    start_qubit : int
        The qubit to start the entanglement spread from.
    measure_all : bool, optional
        Whether all qubits should be measured, by default False.

    Returns
    -------
    tuple[QuantumCircuit, list[int], dict[int, int]]
        The GHZ state circuit, the layout, and the physical to virtual mapping.
    """
    coupling = sorted(coupling_map)
    if qubits_to_remove:
        coupling = remove_qubits_from_coupling_map(coupling, qubits_to_remove)
    coupling = nx.from_edgelist(coupling)

    _, visited_edges, _, visited_nodes_list = expand_graph(coupling, start_qubit, max_nodes_per_step=1)
    physical_to_virtual_map = get_physical_to_virtual_map(visited_nodes_list, num_qubits)
    ghz = QuantumCircuit(num_qubits)
    # UF: Commented this out as this Hadamard would be redudnant in the GHZ state.
    # ghz.h(physical_to_virtual_map[start_qubit])
    for edge in visited_edges[0 : num_qubits - 1]:
        ghz.cx(physical_to_virtual_map[edge[0]], physical_to_virtual_map[edge[1]])
    if measure_all:
        ghz.measure_all()
    return ghz, get_layout(physical_to_virtual_map), physical_to_virtual_map


def expand_graph(
    graph: nx.Graph, start_node: int, max_nodes_per_step: int = 1
) -> tuple[list[nx.Graph], list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    """
    Expand a graph in a BFS manner.

    Parameters
    ----------
    graph : nx.Graph
        The graph to expand.
    start_node : int
        The node to start the expansion from.
    max_nodes_per_step : int, optional
        The maximum nodes to expand by each step, by default 1.

    Returns
    -------
    tuple[list[nx.Graph], list[tuple[int, int]], list[list[tuple[int, int]]], list[int]]
        A list of expanded graphs (one for each step)
        A list of visited edges
        A list of visited edges at each step
        A list of visited nodes
    """
    visited_nodes = set([start_node])
    visited_nodes_list = [start_node]
    visited_edges: Any = []
    visited_edges_step: Any = []
    expanded_graphs = []

    while len(visited_nodes) < len(graph.nodes):
        single_step(
            graph,
            visited_nodes,
            visited_nodes_list,
            visited_edges,
            visited_edges_step,
            max_nodes_per_step,
        )
        expanded_graph = graph.edge_subgraph(visited_edges)
        expanded_graphs.append(expanded_graph)

    return expanded_graphs, visited_edges, visited_edges_step, visited_nodes_list


def single_step(
    graph: nx.Graph,
    visited_nodes: set[int],
    visited_nodes_list: list[int],
    visited_edges: list[tuple[int, int]],
    visited_edges_step: list[list[tuple[int, int]]],
    max_nodes_per_step: int = 1,
) -> None:
    """
    Perform a single step of the BFS expansion.
    The inputs are updated in-place.

    Parameters
    ----------
    graph : nx.Graph
        The graph to expand.
    visited_nodes : set[int]
        The set of visited nodes.
    visited_nodes_list : list[int]
        The list of visited nodes.
    visited_edges : list[tuple[int, int]]
        The list of visited edges.
    visited_edges_step : list[list[tuple[int, int]]]
        The list of visited edges at each step.
    max_nodes_per_step : int, optional
        The maximum nodes to update in step, by default 1.
    """
    new_nodes = set()
    new_edges = set()

    for node in list(visited_nodes):
        # Convert to list to avoid modifying the set during iteration
        neighbors = list(graph.neighbors(node))
        unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in visited_nodes]
        selected_neighbors = unvisited_neighbors[:max_nodes_per_step]
        for selected_neighbor in selected_neighbors:
            if selected_neighbor not in visited_nodes:
                new_nodes.add(selected_neighbor)
                new_edges.add((node, selected_neighbor))
                visited_nodes.add(selected_neighbor)
                visited_nodes_list.append(selected_neighbor)
                visited_edges.append((node, selected_neighbor))
    visited_edges_step.append(list(new_edges))


class GHZCircuitBuilder:
    """
    Generates GHZ states starting from the ::start_qubit:: and spread the entanglement vis BFS to cover ::num_ghz_qubits::
    """

    def __init__(
        self,
        coupling_map: list[tuple[int, int]],
        start_qubit: int,
        num_ghz_qubits: int,
        qubits_to_remove: list[int],
        flags_physical: list[int],
    ):
        self.coupling_map = coupling_map
        self.start_qubit = start_qubit
        self.num_ghz_qubits = num_ghz_qubits
        self.qubits_to_remove = qubits_to_remove
        self.coupling = sorted(remove_duplicate_coupling(coupling_map))
        self.flags_physical = flags_physical
        self.num_flag_qubits = len(flags_physical)
        self.flags_physical2virtual = {x: self.num_ghz_qubits + flags_physical.index(x) for x in flags_physical}
        self.ghz_virtual_indices = list(range(self.num_ghz_qubits))
        self.flags_virtual_indices = list(self.flags_physical2virtual.values())

    def build(self) -> dict[str, list[int] | QuantumCircuit]:
        """
        Build the GHZ preparation circuit, first without parity checks, and then with the parity checks added.

        Returns
        -------
        dict[str, list[int] | QuantumCircuit]
            A dictionary containing the circuit and layouts for the preparation circuit without and with flags.
        """

        # build the circuit with no flags
        circuit_without_flags, layout_without_flags, physical_to_virtual_map = construct_ghz(
            coupling_map=self.coupling_map,
            num_qubits=self.num_ghz_qubits,
            qubits_to_remove=self.qubits_to_remove,
            start_qubit=self.start_qubit,
        )
        circuit_without_flags.barrier()

        # check that no flags are in the GHZ layout
        assert all([x not in layout_without_flags for x in self.flags_physical])

        # get the physical and virtual indices of the parity check CNOT qubits (control and target)
        flag_cxs_physical = []
        flag_cxs_virtual = []
        for a, b in self.coupling:
            if ((a in self.flags_physical) or (b in self.flags_physical)) and (
                (a in layout_without_flags) or (b in layout_without_flags)
            ):
                ctrl = a if a in layout_without_flags else b
                targ = a if a in self.flags_physical else b
                flag_cxs_physical.append([ctrl, targ])
                flag_cxs_virtual.append([physical_to_virtual_map[ctrl], self.flags_physical2virtual[targ]])
                assert targ in self.flags_physical

        # build the circuit with parity checks
        qregs = QuantumRegister(circuit_without_flags.num_qubits + len(self.flags_physical), "q")
        circuit_with_flags = QuantumCircuit(qregs)
        circuit_with_flags.compose(circuit_without_flags, qubits=qregs[: self.num_ghz_qubits], inplace=True)

        # add the parity checks
        for control_qubit, target_qubit in flag_cxs_virtual:
            circuit_with_flags.cx(control_qubit, target_qubit)
        circuit_with_flags.barrier()

        layout_with_flags = layout_without_flags + self.flags_physical

        return {
            "circuit_without_flags": circuit_without_flags,
            "layout_without_flags": layout_without_flags,
            "circuit_with_flags": circuit_with_flags,
            "layout_with_flags": layout_with_flags,
        }


def plot_ghz_fez(
    coupling_map,
    data_qubits: list[int],
    flag_qubits: list[int],
    start_qubit: int | None = None,
    swap_flag_qubits: list[int] | None = None,
    extra=None,
    fig_size=(7, 7),
):
    coupling_map = sorted(coupling_map)

    rows = list(range(15))[::-1]
    index2coord_fez = {
        **{i: [rows[0], i] for i in range(16)},
        **{i: [rows[1], 3 + 4 * (i - 16)] for i in range(16, 20)},
        **{i: [rows[2], i - 20] for i in range(20, 36)},
        **{i: [rows[3], 1 + 4 * (i - 36)] for i in range(36, 40)},
        **{i: [rows[4], i - 40] for i in range(40, 56)},
        **{i: [rows[5], 3 + 4 * (i - 56)] for i in range(56, 60)},
        **{i: [rows[6], i - 60] for i in range(60, 76)},
        **{i: [rows[7], 1 + 4 * (i - 76)] for i in range(76, 80)},
        **{i: [rows[8], i - 80] for i in range(80, 96)},
        **{i: [rows[9], 3 + 4 * (i - 96)] for i in range(96, 100)},
        **{i: [rows[10], i - 100] for i in range(100, 116)},
        **{i: [rows[11], 1 + 4 * (i - 116)] for i in range(116, 120)},
        **{i: [rows[12], i - 120] for i in range(120, 136)},
        **{i: [rows[13], 3 + 4 * (i - 136)] for i in range(136, 140)},
        **{i: [rows[14], i - 140] for i in range(140, 156)},
    }
    qubit_coordinates = [index2coord_fez[i] for i in range(156)]
    return plot_coupling_map(
        156,
        qubit_coordinates,
        list(coupling_map),
        figsize=fig_size,
        plot_directed=False,
        line_width=7,
        qubit_size=61,
        font_size=25,
        line_color=["#A9A8AF"] * len(coupling_map),
        qubit_color=get_qubit_colors(
            data_qubits,
            flag_qubits,
            start_qubit,
            swap_flag_qubits,
            extra,
            total_num_qubits=156,
        ),
    )


def get_qubit_colors(
    data_qubits,
    flags,
    start=None,
    swap_flag_qubits=None,
    extra=None,
    total_num_qubits=127,
):
    qubit_colors = ["#A9A8AF"] * total_num_qubits
    if swap_flag_qubits is not None:
        for q in swap_flag_qubits:
            qubit_colors[q] = "#bd90ff"
    for q in flags:
        qubit_colors[q] = "#E04542"
    for q in data_qubits:
        qubit_colors[q] = "#2aa47b"
    if start is not None:
        qubit_colors[start] = "#4B7AD9"
    if extra is not None:
        if isinstance(extra, int):
            extra = [extra]
        for q in extra:
            qubit_colors[q] = "darkgreen"
    return qubit_colors
